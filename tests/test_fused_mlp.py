"""
Test mega-fused MLP kernel: LN + adaLN + FC1 + GELU + FC2 + bias + gate + residual
All in one kernel, streaming one sequence row at a time.

Strategy per row:
1. LN pass 1: mean (stream x tiles)
2. LN pass 2: variance (stream x tiles)
3. LN pass 3 + modulate: normalize + adaLN -> write modulated to DRAM row
4. For each FC2 output col:
   Hold modulated row in scope (re-read from DRAM once per out_col).
   For each K chunk: FC1(mod @ fc1_w) + GELU -> FC2 accumulate
   Then: (acc + fc2_b) * gate + residual -> output

This eliminates the GELU DRAM round-trip at the cost of re-reading modulated
and FC1 weights per FC2 output col.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import ttnn
import ttl
import time

TILE = 32
D_MODEL = 1024
D_MLP = 4096
D_TILES = D_MODEL // TILE  # 32
D_MLP_TILES = D_MLP // TILE  # 128
FC2_K_CHUNK = D_TILES  # 32 tiles per K chunk for FC2
FC2_K_ITERS = D_MLP_TILES // FC2_K_CHUNK  # 4


def make_fused_mlp_kernel(dim_tiles, fc2_k_chunk, fc2_k_iters):
    """Mega-fused MLP: out = x + gate * (fc2(gelu(fc1(layernorm(x)*scale+shift) + b1)) + b2)

    For each seq row: LN (3 passes), then for each FC2 output col, hold modulated
    row in scope and recompute FC1+GELU per K chunk, accumulating FC2.

    ~21 DFBs (under 32 limit).
    """
    @ttl.kernel(grid="auto")
    def fused_mlp(x, scaler, mean_scale, shift, scale, gate,
                  fc1_w, fc1_b, fc2_w, fc2_b, out):
        grid_cols, _ = ttl.grid_size(dims=2)
        seq_tiles = x.shape[0] // TILE
        n_out_tiles = fc2_w.shape[1] // TILE
        tiles_per_core = -(-seq_tiles // grid_cols)

        # Input/output DFBs
        x_dfb = ttl.make_dataflow_buffer_like(x, shape=(1, 1), buffer_factor=2)
        sc_dfb = ttl.make_dataflow_buffer_like(scaler, shape=(1, 1), buffer_factor=1)
        ms_dfb = ttl.make_dataflow_buffer_like(mean_scale, shape=(1, 1), buffer_factor=1)
        sh_dfb = ttl.make_dataflow_buffer_like(shift, shape=(1, 1), buffer_factor=2)
        scl_dfb = ttl.make_dataflow_buffer_like(scale, shape=(1, 1), buffer_factor=2)
        g_dfb = ttl.make_dataflow_buffer_like(gate, shape=(1, 1), buffer_factor=2)
        r_dfb = ttl.make_dataflow_buffer_like(x, shape=(1, 1), buffer_factor=2)
        out_dfb = ttl.make_dataflow_buffer_like(out, shape=(1, 1), buffer_factor=2)

        # LN intermediates
        red_dfb = ttl.make_dataflow_buffer_like(scaler, shape=(1, 1), buffer_factor=2)
        acc_dfb = ttl.make_dataflow_buffer_like(scaler, shape=(1, 1), buffer_factor=2)
        bcast_dfb = ttl.make_dataflow_buffer_like(x, shape=(1, 1), buffer_factor=2)
        sq_dfb = ttl.make_dataflow_buffer_like(x, shape=(1, 1), buffer_factor=2)
        mean_dfb = ttl.make_dataflow_buffer_like(x, shape=(1, 1), buffer_factor=2)
        istd_dfb = ttl.make_dataflow_buffer_like(x, shape=(1, 1), buffer_factor=2)

        # FC1+GELU -> FC2 DFBs
        # mod_dfb holds modulated row (1, dim_tiles) - reused across FC2 cols
        mod_dfb = ttl.make_dataflow_buffer_like(x, shape=(1, dim_tiles), buffer_factor=2)
        fc1w_dfb = ttl.make_dataflow_buffer_like(fc1_w, shape=(dim_tiles, fc2_k_chunk), buffer_factor=2)
        fc1b_dfb = ttl.make_dataflow_buffer_like(fc1_b, shape=(1, fc2_k_chunk), buffer_factor=2)
        gelu_dfb = ttl.make_dataflow_buffer_like(fc1_b, shape=(1, fc2_k_chunk), buffer_factor=2)
        fc2w_dfb = ttl.make_dataflow_buffer_like(fc2_w, shape=(fc2_k_chunk, 1), buffer_factor=2)
        fc2b_dfb = ttl.make_dataflow_buffer_like(fc2_b, shape=(1, 1), buffer_factor=2)
        mm_dfb = ttl.make_dataflow_buffer_like(fc2_b, shape=(1, 1), buffer_factor=2)
        fc2acc_dfb = ttl.make_dataflow_buffer_like(fc2_b, shape=(1, 1), buffer_factor=2)

        @ttl.compute()
        def compute():
            core_x, _ = ttl.core(dims=2)
            with sc_dfb.wait() as sc, ms_dfb.wait() as ms:
                for local_t in range(tiles_per_core):
                    tile_idx = core_x * tiles_per_core + local_t
                    if tile_idx < seq_tiles:
                        # === LN Pass 1: mean ===
                        with x_dfb.wait() as x0:
                            with red_dfb.reserve() as r:
                                r.store(ttl.math.reduce_sum(x0, sc, dims=[1]))
                        with red_dfb.wait() as rv, acc_dfb.reserve() as a:
                            a.store(rv)
                        for j in range(dim_tiles - 1):
                            with x_dfb.wait() as xj:
                                with red_dfb.reserve() as r:
                                    r.store(ttl.math.reduce_sum(xj, sc, dims=[1]))
                            with red_dfb.wait() as rv, acc_dfb.wait() as av, acc_dfb.reserve() as a:
                                a.store(av + rv)
                        with acc_dfb.wait() as sum_x, bcast_dfb.reserve() as bc:
                            bc.store(ttl.math.broadcast(sum_x, dims=[1]))
                        with bcast_dfb.wait() as sum_bc, mean_dfb.reserve() as m:
                            m.store(sum_bc * ms)

                        # === LN Pass 2: variance ===
                        with mean_dfb.wait() as mean_val:
                            with x_dfb.wait() as x0:
                                diff = x0 - mean_val
                                with sq_dfb.reserve() as sq:
                                    sq.store(diff * diff)
                            with sq_dfb.wait() as sqv, red_dfb.reserve() as r:
                                r.store(ttl.math.reduce_sum(sqv, sc, dims=[1]))
                            with red_dfb.wait() as rv, acc_dfb.reserve() as a:
                                a.store(rv)
                            for j in range(dim_tiles - 1):
                                with x_dfb.wait() as xj:
                                    diff = xj - mean_val
                                    with sq_dfb.reserve() as sq:
                                        sq.store(diff * diff)
                                with sq_dfb.wait() as sqv, red_dfb.reserve() as r:
                                    r.store(ttl.math.reduce_sum(sqv, sc, dims=[1]))
                                with red_dfb.wait() as rv, acc_dfb.wait() as av, acc_dfb.reserve() as a:
                                    a.store(av + rv)
                            with acc_dfb.wait() as sum_sq, bcast_dfb.reserve() as bc:
                                bc.store(ttl.math.broadcast(sum_sq, dims=[1]))
                            with bcast_dfb.wait() as var_bc, istd_dfb.reserve() as istd:
                                istd.store(ttl.math.rsqrt(var_bc * ms + ttl.math.fill(var_bc, 1e-6)))

                            # === LN Pass 3 + adaLN: normalize+modulate -> mod_dfb (1, dim_tiles) ===
                            with istd_dfb.wait() as inv_std:
                                for j in range(dim_tiles):
                                    with x_dfb.wait() as xj, sh_dfb.wait() as shv, scl_dfb.wait() as sclv:
                                        with mod_dfb.reserve() as mod:
                                            normed = (xj - mean_val) * inv_std
                                            mod.store(normed * (sclv + ttl.math.fill(sclv, 1.0)) + shv)

                                # === FC1 + GELU + FC2 for all output cols ===
                                # Hold modulated row in scope across all out_cols
                                with mod_dfb.wait() as modv:
                                    for out_col in range(n_out_tiles):
                                        # FC2 K-accumulation: for each K chunk,
                                        # compute FC1+GELU then FC2 partial
                                        with fc1w_dfb.wait() as fw, fc1b_dfb.wait() as fb:
                                            with gelu_dfb.reserve() as gl:
                                                h = modv @ fw + fb
                                                x3 = h * h * h
                                                inner = ttl.math.fill(h, 0.7978845608) * (h + ttl.math.fill(h, 0.044715) * x3)
                                                gl.store(ttl.math.fill(h, 0.5) * h * (ttl.math.fill(h, 1.0) + ttl.math.tanh(inner)))
                                        with gelu_dfb.wait() as glv, fc2w_dfb.wait() as f2w, fc2acc_dfb.reserve() as a:
                                            a.store(glv @ f2w)

                                        for ki in range(fc2_k_iters - 1):
                                            with fc1w_dfb.wait() as fw, fc1b_dfb.wait() as fb:
                                                with gelu_dfb.reserve() as gl:
                                                    h = modv @ fw + fb
                                                    x3 = h * h * h
                                                    inner = ttl.math.fill(h, 0.7978845608) * (h + ttl.math.fill(h, 0.044715) * x3)
                                                    gl.store(ttl.math.fill(h, 0.5) * h * (ttl.math.fill(h, 1.0) + ttl.math.tanh(inner)))
                                            with gelu_dfb.wait() as glv, fc2w_dfb.wait() as f2w, mm_dfb.reserve() as m:
                                                m.store(glv @ f2w)
                                            with mm_dfb.wait() as mv, fc2acc_dfb.wait() as av, fc2acc_dfb.reserve() as a:
                                                a.store(av + mv)

                                        # Final: (acc + fc2_b) * gate + residual
                                        with fc2acc_dfb.wait() as final, fc2b_dfb.wait() as b2, g_dfb.wait() as gv, r_dfb.wait() as rv, out_dfb.reserve() as o:
                                            o.store(rv + (final + b2) * gv)

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
                    # LN Pass 1: x tiles for mean
                    for j in range(dim_tiles):
                        with x_dfb.reserve() as blk:
                            tx = ttl.copy(x[tile_idx, j], blk); tx.wait()
                    # LN Pass 2: x tiles for variance
                    for j in range(dim_tiles):
                        with x_dfb.reserve() as blk:
                            tx = ttl.copy(x[tile_idx, j], blk); tx.wait()
                    # LN Pass 3 + modulate: x + shift + scale
                    for j in range(dim_tiles):
                        with x_dfb.reserve() as blk:
                            tx = ttl.copy(x[tile_idx, j], blk); tx.wait()
                        with sh_dfb.reserve() as blk:
                            tx = ttl.copy(shift[tile_idx, j], blk); tx.wait()
                        with scl_dfb.reserve() as blk:
                            tx = ttl.copy(scale[tile_idx, j], blk); tx.wait()
                    # mod_dfb is produced by compute, no dm_read needed for it
                    # FC1+GELU+FC2 for each output col
                    for out_col in range(n_out_tiles):
                        for ki in range(fc2_k_iters):
                            k_start = ki * fc2_k_chunk
                            with fc1w_dfb.reserve() as blk:
                                tx = ttl.copy(fc1_w[0:dim_tiles, k_start:k_start + fc2_k_chunk], blk); tx.wait()
                            with fc1b_dfb.reserve() as blk:
                                tx = ttl.copy(fc1_b[tile_idx, k_start:k_start + fc2_k_chunk], blk); tx.wait()
                            with fc2w_dfb.reserve() as blk:
                                tx = ttl.copy(fc2_w[k_start:k_start + fc2_k_chunk, out_col], blk); tx.wait()
                        with fc2b_dfb.reserve() as blk:
                            tx = ttl.copy(fc2_b[tile_idx, out_col], blk); tx.wait()
                        with g_dfb.reserve() as blk:
                            tx = ttl.copy(gate[tile_idx, out_col], blk); tx.wait()
                        with r_dfb.reserve() as blk:
                            tx = ttl.copy(x[tile_idx, out_col], blk); tx.wait()

        @ttl.datamovement()
        def dm_write():
            core_x, _ = ttl.core(dims=2)
            for local_t in range(tiles_per_core):
                tile_idx = core_x * tiles_per_core + local_t
                if tile_idx < seq_tiles:
                    for out_col in range(n_out_tiles):
                        with out_dfb.wait() as blk:
                            tx = ttl.copy(blk, out[tile_idx, out_col]); tx.wait()

    return fused_mlp


def to_tt(t, device):
    return ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                           device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)

def to_tt_l1(t, device):
    return ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                           device=device, memory_config=ttnn.L1_MEMORY_CONFIG)

def expand_bias(bias_1d, seq_pad):
    return bias_1d.unsqueeze(0).expand(seq_pad, -1).contiguous().to(torch.bfloat16)


if __name__ == "__main__":
    device = ttnn.open_device(device_id=0)
    torch.manual_seed(42)

    TEST_SEQ = TILE  # 1 tile row for fast iteration
    print("Testing fused MLP kernel (SEQ=%d, D=%d, MLP=%d)" % (TEST_SEQ, D_MODEL, D_MLP))

    # Create test inputs with small scale for numerical stability
    x = torch.randn(TEST_SEQ, D_MODEL, dtype=torch.float32) * 0.5
    shift = torch.randn(TEST_SEQ, D_MODEL, dtype=torch.float32) * 0.1
    scale = torch.randn(TEST_SEQ, D_MODEL, dtype=torch.float32) * 0.1
    gate_vals = torch.randn(TEST_SEQ, D_MODEL, dtype=torch.float32) * 0.1 + 0.5
    fc1_w = torch.randn(D_MODEL, D_MLP, dtype=torch.float32) * 0.02
    fc1_b_1d = torch.randn(D_MLP, dtype=torch.float32) * 0.01
    fc2_w = torch.randn(D_MLP, D_MODEL, dtype=torch.float32) * 0.02
    fc2_b_1d = torch.randn(D_MODEL, dtype=torch.float32) * 0.01

    # PyTorch reference (fp32)
    ln = nn.LayerNorm(D_MODEL, eps=1e-6, elementwise_affine=False)
    with torch.no_grad():
        normed = ln(x)
    modulated = normed * (1 + scale) + shift
    fc1_out = modulated @ fc1_w + fc1_b_1d
    gelu_out = 0.5 * fc1_out * (1 + torch.tanh(0.7978845608 * (fc1_out + 0.044715 * fc1_out ** 3)))
    fc2_out = gelu_out @ fc2_w + fc2_b_1d
    ref_out = x + gate_vals * fc2_out
    print("Reference range: [%.4f, %.4f], mean=%.4f" % (ref_out.min().item(), ref_out.max().item(), ref_out.mean().item()))

    # Device tensors
    x_tt = to_tt(x.to(torch.bfloat16), device)
    shift_tt = to_tt(shift.to(torch.bfloat16), device)
    scale_tt = to_tt(scale.to(torch.bfloat16), device)
    gate_tt = to_tt(gate_vals.to(torch.bfloat16), device)
    fc1_w_tt = to_tt(fc1_w.to(torch.bfloat16), device)
    fc1_b_tt = to_tt(expand_bias(fc1_b_1d.to(torch.bfloat16), TEST_SEQ), device)
    fc2_w_tt = to_tt(fc2_w.to(torch.bfloat16), device)
    fc2_b_tt = to_tt(expand_bias(fc2_b_1d.to(torch.bfloat16), TEST_SEQ), device)
    out_tt = to_tt(torch.zeros(TEST_SEQ, D_MODEL, dtype=torch.bfloat16), device)
    scaler = to_tt_l1(torch.ones(TILE, TILE, dtype=torch.bfloat16), device)
    mean_scale = to_tt_l1(torch.full((TILE, TILE), 1.0 / D_MODEL, dtype=torch.bfloat16), device)

    print("Building fused MLP kernel...")
    fused_mlp = make_fused_mlp_kernel(D_TILES, FC2_K_CHUNK, FC2_K_ITERS)

    print("Running on HW...")
    t0 = time.time()
    fused_mlp(x_tt, scaler, mean_scale, shift_tt, scale_tt, gate_tt,
              fc1_w_tt, fc1_b_tt, fc2_w_tt, fc2_b_tt, out_tt)
    elapsed = time.time() - t0
    print("Kernel time: %.3fs" % elapsed)

    result = ttnn.to_torch(out_tt).float()
    diff = (ref_out - result).abs()
    max_err = diff.max().item()
    mean_err = diff.mean().item()
    ref_range = ref_out.max().item() - ref_out.min().item()
    print("Max error: %.4f, Mean error: %.4f, Relative: %.4f" % (max_err, mean_err, max_err / (ref_range + 1e-8)))
    print("Result range: [%.4f, %.4f]" % (result.min().item(), result.max().item()))

    if max_err < 10.0:
        print("PASS")
    else:
        print("FAIL")

    ttnn.close_device(device)
