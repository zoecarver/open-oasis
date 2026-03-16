"""Benchmark: Fused post-attention pipeline in TT-Lang vs ttnn separate ops.

The post-attention pipeline per sub-block:
  1. LN(gated_res) + adaLN modulate  (currently ttnn ops, ~0.5ms)
  2. FC1 matmul + bias + GELU        (currently ttnn.linear, ~0.1ms)
  3. FC2 matmul + bias               (currently ttnn.matmul + add, ~0.3ms)
  4. Gated residual                   (currently TT-Lang, ~0.1ms)

The win from fusion: eliminate DRAM round-trips between steps 1-4.
Currently each step reads from and writes to DRAM.

Key insight: steps 1, 3 (GELU), and 4 are elementwise/reduce ops that
stream per-row. The matmuls (FC1, FC2) need K-accumulation across tiles.
We can fuse the elementwise parts around the matmuls.

This kernel fuses: LN + modulate + gated_residual (the non-matmul parts).
FC1/FC2 matmuls stay as ttnn.matmul (72-core parallel vs TT-Lang ~10 cores).

Compared to current approach (7 separate ops -> 3 ops + 2 matmuls):
  Before: ttnn.layer_norm -> slice -> add -> multiply -> add -> matmul -> ...
  After:  fused_ln_mod(x) -> matmul(FC1+GELU) -> matmul(FC2) -> fused_gate_res(x)

Actually, the biggest opportunity is fusing steps 1+4 (LN+mod and gated_res)
since these bracket the matmuls and are the same streaming pattern.
"""
import torch
import ttnn
import ttl
import time

TILE = 32
D_MODEL = 1024
DIM_TILES = D_MODEL // TILE  # 32
D_MLP = 4096
MLP_TILES = D_MLP // TILE  # 128
SEQ = 320

device = ttnn.open_device(device_id=0)

def to_tt(t):
    return ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                           device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)

def to_l1(t):
    return ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                           device=device, memory_config=ttnn.L1_MEMORY_CONFIG)


# Fused: gated_residual + LN + adaLN modulate (single-pass, (1, DIM_TILES) DFBs)
# This replaces: multiply(o_proj, gate) -> add(residual, tmp) -> layer_norm ->
#                add(scale, 1) -> multiply(normed, scale+1) -> add(shift)
# 6 ttnn ops -> 1 TT-Lang kernel, 1 DRAM read pass
@ttl.kernel(grid="auto")
def fused_gated_res_ln_adaln(residual, o_proj, gate, scaler, mean_scale,
                              shift, scale_param, gated_res_out, modulated_out):
    grid_cols, _ = ttl.grid_size(dims=2)
    seq_tiles = residual.shape[0] // TILE
    tiles_per_core = -(-seq_tiles // grid_cols)

    res_dfb = ttl.make_dataflow_buffer_like(residual, shape=(1, DIM_TILES), buffer_factor=2)
    oproj_dfb = ttl.make_dataflow_buffer_like(o_proj, shape=(1, DIM_TILES), buffer_factor=2)
    gate_dfb = ttl.make_dataflow_buffer_like(gate, shape=(1, DIM_TILES), buffer_factor=2)
    sh_dfb = ttl.make_dataflow_buffer_like(shift, shape=(1, DIM_TILES), buffer_factor=2)
    scl_dfb = ttl.make_dataflow_buffer_like(scale_param, shape=(1, DIM_TILES), buffer_factor=2)
    sc_dfb = ttl.make_dataflow_buffer_like(scaler, shape=(1, 1), buffer_factor=1)
    ms_dfb = ttl.make_dataflow_buffer_like(mean_scale, shape=(1, 1), buffer_factor=1)
    # Intermediates
    gr_dfb = ttl.make_dataflow_buffer_like(residual, shape=(1, DIM_TILES), buffer_factor=2)
    red_dfb = ttl.make_dataflow_buffer_like(scaler, shape=(1, 1), buffer_factor=2)
    mean_dfb = ttl.make_dataflow_buffer_like(scaler, shape=(1, 1), buffer_factor=2)
    bcast_dfb = ttl.make_dataflow_buffer_like(residual, shape=(1, DIM_TILES), buffer_factor=2)
    sq_dfb = ttl.make_dataflow_buffer_like(residual, shape=(1, DIM_TILES), buffer_factor=2)
    istd_dfb = ttl.make_dataflow_buffer_like(scaler, shape=(1, 1), buffer_factor=2)
    istd_bcast_dfb = ttl.make_dataflow_buffer_like(residual, shape=(1, DIM_TILES), buffer_factor=2)
    gro_dfb = ttl.make_dataflow_buffer_like(gated_res_out, shape=(1, DIM_TILES), buffer_factor=2)
    out_dfb = ttl.make_dataflow_buffer_like(modulated_out, shape=(1, DIM_TILES), buffer_factor=2)

    @ttl.compute()
    def compute():
        core_x, _ = ttl.core(dims=2)
        with sc_dfb.wait() as sc, ms_dfb.wait() as ms:
            for local_t in range(tiles_per_core):
                tile_idx = core_x * tiles_per_core + local_t
                if tile_idx < seq_tiles:
                    # Compute gated_res = residual + o_proj * gate
                    with res_dfb.wait() as rv, oproj_dfb.wait() as ov, gate_dfb.wait() as gv, gr_dfb.reserve() as gr:
                        gr.store(rv + ov * gv)
                    # Write gated_res to output AND start LN
                    with gr_dfb.wait() as grv:
                        with gro_dfb.reserve() as gro:
                            gro.store(grv)
                        # Mean
                        with red_dfb.reserve() as r:
                            r.store(ttl.math.reduce_sum(grv, sc, dims=[1]))
                    with red_dfb.wait() as sum_val, mean_dfb.reserve() as mean_scaled:
                        mean_scaled.store(sum_val * ms)
                    with mean_dfb.wait() as mean_val, bcast_dfb.reserve() as bc:
                        bc.store(ttl.math.broadcast(mean_val, dims=[1]))
                    # Variance (recompute gated_res from the broadcast mean)
                    # Note: grv is out of scope, but gro wrote it. We need to re-read.
                    # Actually we can keep grv in scope by nesting...
                    # Let me restructure to keep grv alive:
                    pass  # restructure below

    @ttl.datamovement()
    def dm_read():
        core_x, _ = ttl.core(dims=2)
        with sc_dfb.reserve() as blk:
            tx = ttl.copy(scaler[0, 0], blk); tx.wait()
        with ms_dfb.reserve() as blk:
            tx = ttl.copy(mean_scale[0, 0], blk); tx.wait()
        for local_t in range(tiles_per_core):
            tile_idx = core_x * tiles_per_core + local_t
            if tile_idx < seq_tiles:
                with res_dfb.reserve() as blk:
                    tx = ttl.copy(residual[tile_idx, 0:DIM_TILES], blk); tx.wait()
                with oproj_dfb.reserve() as blk:
                    tx = ttl.copy(o_proj[tile_idx, 0:DIM_TILES], blk); tx.wait()
                with gate_dfb.reserve() as blk:
                    tx = ttl.copy(gate[tile_idx, 0:DIM_TILES], blk); tx.wait()
                with sh_dfb.reserve() as blk:
                    tx = ttl.copy(shift[tile_idx, 0:DIM_TILES], blk); tx.wait()
                with scl_dfb.reserve() as blk:
                    tx = ttl.copy(scale_param[tile_idx, 0:DIM_TILES], blk); tx.wait()

    @ttl.datamovement()
    def dm_write():
        core_x, _ = ttl.core(dims=2)
        for local_t in range(tiles_per_core):
            tile_idx = core_x * tiles_per_core + local_t
            if tile_idx < seq_tiles:
                with gro_dfb.wait() as blk:
                    tx = ttl.copy(blk, gated_res_out[tile_idx, 0:DIM_TILES]); tx.wait()
                with out_dfb.wait() as blk:
                    tx = ttl.copy(blk, modulated_out[tile_idx, 0:DIM_TILES]); tx.wait()

# Hmm, the compute thread above has a structural issue. Let me restructure
# to keep gated_res in scope through the whole LN computation.
# The trick: nest everything inside the gr_dfb.wait() scope.
# But that means gr_dfb holds one slot for the entire LN computation, which
# works with buffer_factor=2 since we only process one row at a time.

# Let me rewrite the kernel properly:

@ttl.kernel(grid="auto")
def fused_gated_res_ln_adaln_v2(residual, o_proj, gate, scaler, mean_scale,
                                 shift, scale_param, gated_res_out, modulated_out):
    grid_cols, _ = ttl.grid_size(dims=2)
    seq_tiles = residual.shape[0] // TILE
    tiles_per_core = -(-seq_tiles // grid_cols)

    res_dfb = ttl.make_dataflow_buffer_like(residual, shape=(1, DIM_TILES), buffer_factor=2)
    oproj_dfb = ttl.make_dataflow_buffer_like(o_proj, shape=(1, DIM_TILES), buffer_factor=2)
    gate_dfb = ttl.make_dataflow_buffer_like(gate, shape=(1, DIM_TILES), buffer_factor=2)
    sh_dfb = ttl.make_dataflow_buffer_like(shift, shape=(1, DIM_TILES), buffer_factor=2)
    scl_dfb = ttl.make_dataflow_buffer_like(scale_param, shape=(1, DIM_TILES), buffer_factor=2)
    sc_dfb = ttl.make_dataflow_buffer_like(scaler, shape=(1, 1), buffer_factor=1)
    ms_dfb = ttl.make_dataflow_buffer_like(mean_scale, shape=(1, 1), buffer_factor=1)
    gr_dfb = ttl.make_dataflow_buffer_like(residual, shape=(1, DIM_TILES), buffer_factor=2)
    red_dfb = ttl.make_dataflow_buffer_like(scaler, shape=(1, 1), buffer_factor=2)
    mean_dfb = ttl.make_dataflow_buffer_like(scaler, shape=(1, 1), buffer_factor=2)
    bcast_dfb = ttl.make_dataflow_buffer_like(residual, shape=(1, DIM_TILES), buffer_factor=2)
    sq_dfb = ttl.make_dataflow_buffer_like(residual, shape=(1, DIM_TILES), buffer_factor=2)
    istd_dfb = ttl.make_dataflow_buffer_like(scaler, shape=(1, 1), buffer_factor=2)
    istd_bcast_dfb = ttl.make_dataflow_buffer_like(residual, shape=(1, DIM_TILES), buffer_factor=2)
    gro_dfb = ttl.make_dataflow_buffer_like(gated_res_out, shape=(1, DIM_TILES), buffer_factor=2)
    out_dfb = ttl.make_dataflow_buffer_like(modulated_out, shape=(1, DIM_TILES), buffer_factor=2)

    @ttl.compute()
    def compute():
        core_x, _ = ttl.core(dims=2)
        with sc_dfb.wait() as sc, ms_dfb.wait() as ms:
            for local_t in range(tiles_per_core):
                tile_idx = core_x * tiles_per_core + local_t
                if tile_idx < seq_tiles:
                    # Gated residual
                    with res_dfb.wait() as rv, oproj_dfb.wait() as ov, gate_dfb.wait() as gv, gr_dfb.reserve() as gr:
                        gr.store(rv + ov * gv)
                    # Keep gated_res in scope for the whole LN
                    with gr_dfb.wait() as grv:
                        # Write to output
                        with gro_dfb.reserve() as gro:
                            gro.store(grv)
                        # Mean
                        with red_dfb.reserve() as r:
                            r.store(ttl.math.reduce_sum(grv, sc, dims=[1]))
                        with red_dfb.wait() as sum_val, mean_dfb.reserve() as ms_out:
                            ms_out.store(sum_val * ms)
                        with mean_dfb.wait() as mean_val, bcast_dfb.reserve() as bc:
                            bc.store(ttl.math.broadcast(mean_val, dims=[1]))
                        # Variance
                        with bcast_dfb.wait() as mean_bc, sq_dfb.reserve() as sq:
                            sq.store((grv - mean_bc) * (grv - mean_bc))
                        with sq_dfb.wait() as sqv, red_dfb.reserve() as var_r:
                            var_r.store(ttl.math.reduce_sum(sqv, sc, dims=[1]))
                        with red_dfb.wait() as var_sum, istd_dfb.reserve() as istd_out:
                            istd_out.store(ttl.math.rsqrt(var_sum * ms + ttl.math.fill(var_sum, 1e-6)))
                        with istd_dfb.wait() as istd, istd_bcast_dfb.reserve() as istd_bc:
                            istd_bc.store(ttl.math.broadcast(istd, dims=[1]))
                        # Normalize + modulate
                        with bcast_dfb.wait() as mean_bc2, istd_bcast_dfb.wait() as istd_bcv:
                            with sh_dfb.wait() as shv, scl_dfb.wait() as sclv, out_dfb.reserve() as o:
                                normed = (grv - mean_bc2) * istd_bcv
                                o.store(normed * (sclv + ttl.math.fill(sclv, 1.0)) + shv)

    @ttl.datamovement()
    def dm_read():
        core_x, _ = ttl.core(dims=2)
        with sc_dfb.reserve() as blk:
            tx = ttl.copy(scaler[0, 0], blk); tx.wait()
        with ms_dfb.reserve() as blk:
            tx = ttl.copy(mean_scale[0, 0], blk); tx.wait()
        for local_t in range(tiles_per_core):
            tile_idx = core_x * tiles_per_core + local_t
            if tile_idx < seq_tiles:
                with res_dfb.reserve() as blk:
                    tx = ttl.copy(residual[tile_idx, 0:DIM_TILES], blk); tx.wait()
                with oproj_dfb.reserve() as blk:
                    tx = ttl.copy(o_proj[tile_idx, 0:DIM_TILES], blk); tx.wait()
                with gate_dfb.reserve() as blk:
                    tx = ttl.copy(gate[tile_idx, 0:DIM_TILES], blk); tx.wait()
                with sh_dfb.reserve() as blk:
                    tx = ttl.copy(shift[tile_idx, 0:DIM_TILES], blk); tx.wait()
                with scl_dfb.reserve() as blk:
                    tx = ttl.copy(scale_param[tile_idx, 0:DIM_TILES], blk); tx.wait()

    @ttl.datamovement()
    def dm_write():
        core_x, _ = ttl.core(dims=2)
        for local_t in range(tiles_per_core):
            tile_idx = core_x * tiles_per_core + local_t
            if tile_idx < seq_tiles:
                with gro_dfb.wait() as blk:
                    tx = ttl.copy(blk, gated_res_out[tile_idx, 0:DIM_TILES]); tx.wait()
                with out_dfb.wait() as blk:
                    tx = ttl.copy(blk, modulated_out[tile_idx, 0:DIM_TILES]); tx.wait()


# === Setup ===
residual = to_tt(torch.randn(SEQ, D_MODEL, dtype=torch.bfloat16))
o_proj = to_tt(torch.randn(SEQ, D_MODEL, dtype=torch.bfloat16))
gate = to_tt(torch.randn(SEQ, D_MODEL, dtype=torch.bfloat16))
shift = to_tt(torch.randn(SEQ, D_MODEL, dtype=torch.bfloat16))
scale = to_tt(torch.randn(SEQ, D_MODEL, dtype=torch.bfloat16))
scaler = to_l1(torch.ones(1, 32, dtype=torch.bfloat16))
mean_scale = to_l1(torch.full((1, 32), 1.0/D_MODEL, dtype=torch.bfloat16))
gated_res_out = to_tt(torch.zeros(SEQ, D_MODEL, dtype=torch.bfloat16))
modulated_out = to_tt(torch.zeros(SEQ, D_MODEL, dtype=torch.bfloat16))
tmp = to_tt(torch.zeros(SEQ, D_MODEL, dtype=torch.bfloat16))

# === Correctness ===
print("=== Correctness: fused gated_res + LN + adaLN ===")
fused_gated_res_ln_adaln_v2(residual, o_proj, gate, scaler, mean_scale,
                             shift, scale, gated_res_out, modulated_out)
ttnn.synchronize_device(device)

# Reference: ttnn ops
ttnn.multiply(o_proj, gate, output_tensor=tmp)
ttnn.add(residual, tmp, output_tensor=tmp)
gr_ref = ttnn.to_torch(tmp)
normed_ref = ttnn.layer_norm(tmp)
normed_t = ttnn.to_torch(normed_ref)
scale_t = ttnn.to_torch(scale)
shift_t = ttnn.to_torch(shift)
expected_mod = normed_t.float() * (scale_t.float() + 1.0) + shift_t.float()

ttl_gr = ttnn.to_torch(gated_res_out)
ttl_mod = ttnn.to_torch(modulated_out)

gr_diff = (ttl_gr.float() - gr_ref.float()).abs().max().item()
mod_diff = (ttl_mod.float() - expected_mod).abs().max().item()
print("Gated residual diff: %.4f" % gr_diff)
print("Modulated diff: %.4f" % mod_diff)
print("PASS" if gr_diff < 1.0 and mod_diff < 2.0 else "FAIL")

# === Benchmark ===
N_ITERS = 100

# Warmup TT-Lang
for _ in range(5):
    fused_gated_res_ln_adaln_v2(residual, o_proj, gate, scaler, mean_scale,
                                 shift, scale, gated_res_out, modulated_out)
ttnn.synchronize_device(device)

print("\n=== Benchmark: TT-Lang fused gated_res+LN+adaLN (%d iters) ===" % N_ITERS)
t0 = time.time()
for _ in range(N_ITERS):
    fused_gated_res_ln_adaln_v2(residual, o_proj, gate, scaler, mean_scale,
                                 shift, scale, gated_res_out, modulated_out)
ttnn.synchronize_device(device)
t_ttl = time.time() - t0
print("TT-Lang fused: %.1fms / %d iters (%.3fms each)" % (t_ttl * 1000, N_ITERS, t_ttl * 1000 / N_ITERS))

# Warmup ttnn ops
for _ in range(5):
    ttnn.multiply(o_proj, gate, output_tensor=tmp)
    ttnn.add(residual, tmp, output_tensor=gated_res_out)
    normed = ttnn.layer_norm(gated_res_out)
    ttnn.add(scale, 1.0, output_tensor=tmp)
    ttnn.multiply(normed, tmp, output_tensor=modulated_out)
    ttnn.add(modulated_out, shift, output_tensor=modulated_out)
ttnn.synchronize_device(device)

print("\n=== Benchmark: ttnn separate ops (%d iters) ===" % N_ITERS)
t0 = time.time()
for _ in range(N_ITERS):
    ttnn.multiply(o_proj, gate, output_tensor=tmp)
    ttnn.add(residual, tmp, output_tensor=gated_res_out)
    normed = ttnn.layer_norm(gated_res_out)
    ttnn.add(scale, 1.0, output_tensor=tmp)
    ttnn.multiply(normed, tmp, output_tensor=modulated_out)
    ttnn.add(modulated_out, shift, output_tensor=modulated_out)
ttnn.synchronize_device(device)
t_ttnn = time.time() - t0
print("ttnn ops: %.1fms / %d iters (%.3fms each)" % (t_ttnn * 1000, N_ITERS, t_ttnn * 1000 / N_ITERS))

print("\n=== Summary ===")
print("TT-Lang fused: %.3fms" % (t_ttl * 1000 / N_ITERS))
print("ttnn separate: %.3fms" % (t_ttnn * 1000 / N_ITERS))
ratio = t_ttnn / t_ttl if t_ttl > 0 else 0
if ratio > 1:
    print("TT-Lang is %.1fx FASTER" % ratio)
else:
    print("ttnn is %.1fx faster" % (1/ratio if ratio > 0 else 0))

ttnn.close_device(device)
