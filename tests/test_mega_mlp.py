"""Test mega kernel B: O proj + residual + LN + modulate + FC1 + GELU + FC2 + residual.

Per-row kernel. Uses DRAM scratch tensors between phases:
  z_scratch: O proj + residual result (needed for final residual)
  gelu_scratch: FC1 + GELU output (D_MLP wide)

Phase A: O proj matmul (compiler K-accum) + bias + gate + residual → z_scratch
Phase B: LN (3 passes over z_scratch) + modulate → write modulated tiles
Phase C: FC1 matmul (K-accum) + bias + GELU → gelu_scratch
Phase D: FC2 matmul (K-accum) + bias + gate + z residual → out
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import ttnn
import ttl

TILE = 32
D_MODEL = 1024
D_MLP = 4096
N_PATCH_PAD = 160
D_TILES = D_MODEL // TILE    # 32
D_MLP_TILES = D_MLP // TILE  # 128
SEQ_TILES = N_PATCH_PAD // TILE  # 5


def make_mega_post_attn_kernel(dim_tiles, mlp_dim_tiles):
    @ttl.kernel(grid="auto")
    def mega_post_attn(attn_out, x_residual, gate_msa,
                       out_w, out_b,
                       shift_mlp, scale_mlp, gate_mlp,
                       fc1_w, fc1_b, fc2_w, fc2_b,
                       scaler, mean_scale,
                       z_scratch, gelu_scratch, out):
        grid_cols, _ = ttl.grid_size(dims=2)
        seq_tiles = attn_out.shape[0] // TILE
        tiles_per_core = -(-seq_tiles // grid_cols)

        # O proj: attn row held in scope, weight streamed per col
        attn_dfb = ttl.make_dataflow_buffer_like(attn_out, shape=(1, dim_tiles), buffer_factor=2)
        wcol_dfb = ttl.make_dataflow_buffer_like(out_w, shape=(dim_tiles, 1), buffer_factor=2)

        # General-purpose (1,1) tile DFBs reused across phases
        x_dfb = ttl.make_dataflow_buffer_like(attn_out, shape=(1, 1), buffer_factor=2)
        w_dfb = ttl.make_dataflow_buffer_like(out_w, shape=(1, 1), buffer_factor=2)
        p1_dfb = ttl.make_dataflow_buffer_like(attn_out, shape=(1, 1), buffer_factor=2)
        p2_dfb = ttl.make_dataflow_buffer_like(attn_out, shape=(1, 1), buffer_factor=2)
        p3_dfb = ttl.make_dataflow_buffer_like(attn_out, shape=(1, 1), buffer_factor=2)

        # Constants
        scaler_dfb = ttl.make_dataflow_buffer_like(scaler, shape=(1, 1), buffer_factor=1)
        ms_dfb = ttl.make_dataflow_buffer_like(mean_scale, shape=(1, 1), buffer_factor=1)

        # Intermediates
        mm_dfb = ttl.make_dataflow_buffer_like(out, shape=(1, 1), buffer_factor=2)
        acc_dfb = ttl.make_dataflow_buffer_like(out, shape=(1, 1), buffer_factor=2)
        red_dfb = ttl.make_dataflow_buffer_like(scaler, shape=(1, 1), buffer_factor=2)
        bcast_dfb = ttl.make_dataflow_buffer_like(out, shape=(1, 1), buffer_factor=2)
        sq_dfb = ttl.make_dataflow_buffer_like(out, shape=(1, 1), buffer_factor=2)
        mean_dfb = ttl.make_dataflow_buffer_like(out, shape=(1, 1), buffer_factor=2)
        istd_dfb = ttl.make_dataflow_buffer_like(out, shape=(1, 1), buffer_factor=2)
        out_dfb = ttl.make_dataflow_buffer_like(out, shape=(1, 1), buffer_factor=2)
        # 18 DFBs total

        @ttl.compute()
        def compute():
            core_x, _ = ttl.core(dims=2)
            with scaler_dfb.wait() as sclr, ms_dfb.wait() as ms:
                for local_t in range(tiles_per_core):
                    tile_idx = core_x * tiles_per_core + local_t
                    if tile_idx < seq_tiles:

                        # ======== Phase A: O proj + bias + gate + residual ========
                        # Hold attn row in scope, stream weight columns
                        with attn_dfb.wait() as a_row:
                            for col in range(dim_tiles):
                                with wcol_dfb.wait() as w_col, mm_dfb.reserve() as mm:
                                    mm.store(a_row @ w_col)
                                # bias + gate + residual → out_dfb → dm_write → z_scratch
                                with mm_dfb.wait() as oproj, p1_dfb.wait() as bv, p2_dfb.wait() as gv, p3_dfb.wait() as rv:
                                    with out_dfb.reserve() as o:
                                        o.store(rv + (oproj + bv) * gv)

                        # ======== Phase B: LayerNorm + modulate ========
                        # LN pass 1: mean (read z tiles from z_scratch via dm_read → x_dfb)
                        with x_dfb.wait() as z0:
                            with red_dfb.reserve() as r:
                                r.store(ttl.math.reduce_sum(z0, sclr, dims=[1]))
                        with red_dfb.wait() as rv, acc_dfb.reserve() as a:
                            a.store(rv)
                        for j in range(dim_tiles - 1):
                            with x_dfb.wait() as zj:
                                with red_dfb.reserve() as r:
                                    r.store(ttl.math.reduce_sum(zj, sclr, dims=[1]))
                            with red_dfb.wait() as rv, acc_dfb.wait() as prev, acc_dfb.reserve() as a:
                                a.store(prev + rv)
                        with acc_dfb.wait() as sum_x, bcast_dfb.reserve() as bc:
                            bc.store(ttl.math.broadcast(sum_x, dims=[1]))
                        with bcast_dfb.wait() as sum_bc, mean_dfb.reserve() as mn:
                            mn.store(sum_bc * ms)

                        # LN pass 2: variance
                        with mean_dfb.wait() as mean_val:
                            with x_dfb.wait() as z0:
                                with sq_dfb.reserve() as sq:
                                    sq.store((z0 - mean_val) * (z0 - mean_val))
                            with sq_dfb.wait() as sqv, red_dfb.reserve() as r:
                                r.store(ttl.math.reduce_sum(sqv, sclr, dims=[1]))
                            with red_dfb.wait() as rv, acc_dfb.reserve() as a:
                                a.store(rv)
                            for j in range(dim_tiles - 1):
                                with x_dfb.wait() as zj:
                                    with sq_dfb.reserve() as sq:
                                        sq.store((zj - mean_val) * (zj - mean_val))
                                with sq_dfb.wait() as sqv, red_dfb.reserve() as r:
                                    r.store(ttl.math.reduce_sum(sqv, sclr, dims=[1]))
                                with red_dfb.wait() as rv, acc_dfb.wait() as prev, acc_dfb.reserve() as a:
                                    a.store(prev + rv)
                            with acc_dfb.wait() as sum_sq, bcast_dfb.reserve() as bc:
                                bc.store(ttl.math.broadcast(sum_sq, dims=[1]))
                            with bcast_dfb.wait() as var_bc, istd_dfb.reserve() as istd:
                                istd.store(ttl.math.rsqrt(var_bc * ms + ttl.math.fill(var_bc, 1e-6)))

                            # LN pass 3 + modulate → out_dfb → dm_write
                            with istd_dfb.wait() as inv_std:
                                for j in range(dim_tiles):
                                    with x_dfb.wait() as zj, p1_dfb.wait() as shv, p2_dfb.wait() as sclv:
                                        normed = (zj - mean_val) * inv_std
                                        with out_dfb.reserve() as o:
                                            o.store(normed * (sclv + ttl.math.fill(sclv, 1.0)) + shv)

                        # ======== Phase C: FC1 + bias + GELU ========
                        for fc1_col in range(mlp_dim_tiles):
                            # K-accumulation over dim_tiles
                            with x_dfb.wait() as m0, w_dfb.wait() as fw0, acc_dfb.reserve() as a:
                                a.store(m0 @ fw0)
                            for k in range(dim_tiles - 1):
                                with x_dfb.wait() as mk, w_dfb.wait() as fwk, mm_dfb.reserve() as mm:
                                    mm.store(mk @ fwk)
                                with mm_dfb.wait() as mmv, acc_dfb.wait() as prev, acc_dfb.reserve() as a:
                                    a.store(prev + mmv)
                            with acc_dfb.wait() as fc1r, p1_dfb.wait() as fb:
                                h = fc1r + fb
                                x3 = h * h * h
                                inner = ttl.math.fill(h, 0.7978845608) * (h + ttl.math.fill(h, 0.044715) * x3)
                                with out_dfb.reserve() as o:
                                    o.store(ttl.math.fill(h, 0.5) * h * (ttl.math.fill(h, 1.0) + ttl.math.tanh(inner)))

                        # ======== Phase D: FC2 + bias + gate + residual ========
                        for col in range(dim_tiles):
                            # K-accumulation over mlp_dim_tiles
                            with x_dfb.wait() as g0, w_dfb.wait() as fw0, acc_dfb.reserve() as a:
                                a.store(g0 @ fw0)
                            for k in range(mlp_dim_tiles - 1):
                                with x_dfb.wait() as gk, w_dfb.wait() as fwk, mm_dfb.reserve() as mm:
                                    mm.store(gk @ fwk)
                                with mm_dfb.wait() as mmv, acc_dfb.wait() as prev, acc_dfb.reserve() as a:
                                    a.store(prev + mmv)
                            # + bias + gate_mlp + z residual
                            with acc_dfb.wait() as fc2r, p1_dfb.wait() as fb, p2_dfb.wait() as gv, p3_dfb.wait() as zv:
                                with out_dfb.reserve() as o:
                                    o.store(zv + (fc2r + fb) * gv)

        @ttl.datamovement()
        def dm_read():
            core_x, _ = ttl.core(dims=2)
            with scaler_dfb.reserve() as blk:
                tx = ttl.copy(scaler[0, 0], blk); tx.wait()
            with ms_dfb.reserve() as blk:
                tx = ttl.copy(mean_scale[0, 0], blk); tx.wait()

            for local_t in range(tiles_per_core):
                tile_idx = core_x * tiles_per_core + local_t
                if tile_idx < seq_tiles:

                    # Phase A: O proj
                    with attn_dfb.reserve() as blk:
                        tx = ttl.copy(attn_out[tile_idx, 0:dim_tiles], blk); tx.wait()
                    for col in range(dim_tiles):
                        with wcol_dfb.reserve() as blk:
                            tx = ttl.copy(out_w[0:dim_tiles, col], blk); tx.wait()
                        with p1_dfb.reserve() as blk:
                            tx = ttl.copy(out_b[tile_idx, col], blk); tx.wait()
                        with p2_dfb.reserve() as blk:
                            tx = ttl.copy(gate_msa[tile_idx, col], blk); tx.wait()
                        with p3_dfb.reserve() as blk:
                            tx = ttl.copy(x_residual[tile_idx, col], blk); tx.wait()

                    # Phase B: LN (3 passes reading z from z_scratch)
                    for j in range(dim_tiles):
                        with x_dfb.reserve() as blk:
                            tx = ttl.copy(z_scratch[tile_idx, j], blk); tx.wait()
                    for j in range(dim_tiles):
                        with x_dfb.reserve() as blk:
                            tx = ttl.copy(z_scratch[tile_idx, j], blk); tx.wait()
                    for j in range(dim_tiles):
                        with x_dfb.reserve() as blk:
                            tx = ttl.copy(z_scratch[tile_idx, j], blk); tx.wait()
                        with p1_dfb.reserve() as blk:
                            tx = ttl.copy(shift_mlp[tile_idx, j], blk); tx.wait()
                        with p2_dfb.reserve() as blk:
                            tx = ttl.copy(scale_mlp[tile_idx, j], blk); tx.wait()

                    # Phase C: FC1 (read modulated from out tensor, written by Phase B dm_write)
                    for fc1_col in range(mlp_dim_tiles):
                        for k in range(dim_tiles):
                            with x_dfb.reserve() as blk:
                                tx = ttl.copy(out[tile_idx, k], blk); tx.wait()
                            with w_dfb.reserve() as blk:
                                tx = ttl.copy(fc1_w[k, fc1_col], blk); tx.wait()
                        with p1_dfb.reserve() as blk:
                            tx = ttl.copy(fc1_b[tile_idx, fc1_col], blk); tx.wait()

                    # Phase D: FC2 (read gelu from gelu_scratch)
                    for col in range(dim_tiles):
                        for k in range(mlp_dim_tiles):
                            with x_dfb.reserve() as blk:
                                tx = ttl.copy(gelu_scratch[tile_idx, k], blk); tx.wait()
                            with w_dfb.reserve() as blk:
                                tx = ttl.copy(fc2_w[k, col], blk); tx.wait()
                        with p1_dfb.reserve() as blk:
                            tx = ttl.copy(fc2_b[tile_idx, col], blk); tx.wait()
                        with p2_dfb.reserve() as blk:
                            tx = ttl.copy(gate_mlp[tile_idx, col], blk); tx.wait()
                        with p3_dfb.reserve() as blk:
                            tx = ttl.copy(z_scratch[tile_idx, col], blk); tx.wait()

        @ttl.datamovement()
        def dm_write():
            core_x, _ = ttl.core(dims=2)
            for local_t in range(tiles_per_core):
                tile_idx = core_x * tiles_per_core + local_t
                if tile_idx < seq_tiles:
                    # Phase A: O proj result → z_scratch
                    for col in range(dim_tiles):
                        with out_dfb.wait() as blk:
                            tx = ttl.copy(blk, z_scratch[tile_idx, col]); tx.wait()
                    # Phase B: modulated → z_scratch (overwrite z - z is re-read in Phase D from z_scratch before overwrite)
                    # WAIT: Phase D reads z_scratch[tile_idx, col] for residual.
                    # If Phase B overwrites z_scratch, Phase D reads modulated instead of z.
                    # We need to write modulated to z_scratch AND keep z for Phase D.
                    # Solution: Phase D reads z BEFORE Phase B overwrites.
                    # But phases are sequential per row: A, B, C, D. Phase D is after B.
                    # So z_scratch IS overwritten by B before D reads it.
                    #
                    # FIX: Write Phase B modulated to a different address.
                    # Use out tensor as scratch for modulated (it gets overwritten in Phase D anyway).
                    for col in range(dim_tiles):
                        with out_dfb.wait() as blk:
                            tx = ttl.copy(blk, out[tile_idx, col]); tx.wait()
                    # Phase C: GELU → gelu_scratch
                    for fc1_col in range(mlp_dim_tiles):
                        with out_dfb.wait() as blk:
                            tx = ttl.copy(blk, gelu_scratch[tile_idx, fc1_col]); tx.wait()
                    # Phase D: final → out (overwrites modulated from Phase B, which is fine)
                    for col in range(dim_tiles):
                        with out_dfb.wait() as blk:
                            tx = ttl.copy(blk, out[tile_idx, col]); tx.wait()

    return mega_post_attn


def to_tt(t, device):
    return ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                           device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)

def to_tt_l1(t, device):
    return ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                           device=device, memory_config=ttnn.L1_MEMORY_CONFIG)


if __name__ == "__main__":
    device = ttnn.open_device(device_id=0)
    torch.manual_seed(42)

    SEQ = N_PATCH_PAD
    print("Testing mega post-attention kernel B")
    print("  SEQ=%d, D_MODEL=%d, D_MLP=%d" % (SEQ, D_MODEL, D_MLP))

    # Inputs
    attn_out_t = torch.randn(SEQ, D_MODEL, dtype=torch.float32) * 0.1
    x_res_t = torch.randn(SEQ, D_MODEL, dtype=torch.float32) * 0.1
    gate_msa_t = torch.ones(SEQ, D_MODEL, dtype=torch.float32) * 0.5

    out_w_t = torch.randn(D_MODEL, D_MODEL, dtype=torch.float32) * (2.0 / D_MODEL) ** 0.5
    out_b_1d = torch.randn(D_MODEL, dtype=torch.float32) * 0.01
    out_b_t = out_b_1d.unsqueeze(0).expand(SEQ, -1).contiguous()

    shift_mlp_t = torch.randn(SEQ, D_MODEL, dtype=torch.float32) * 0.01
    scale_mlp_t = torch.randn(SEQ, D_MODEL, dtype=torch.float32) * 0.01
    gate_mlp_t = torch.ones(SEQ, D_MODEL, dtype=torch.float32) * 0.5

    fc1_w_t = torch.randn(D_MODEL, D_MLP, dtype=torch.float32) * (2.0 / D_MODEL) ** 0.5
    fc1_b_1d = torch.randn(D_MLP, dtype=torch.float32) * 0.01
    fc1_b_t = fc1_b_1d.unsqueeze(0).expand(SEQ, -1).contiguous()
    fc2_w_t = torch.randn(D_MLP, D_MODEL, dtype=torch.float32) * (2.0 / D_MLP) ** 0.5
    fc2_b_1d = torch.randn(D_MODEL, dtype=torch.float32) * 0.01
    fc2_b_t = fc2_b_1d.unsqueeze(0).expand(SEQ, -1).contiguous()

    # PyTorch reference
    z = x_res_t.float() + (attn_out_t.float() @ out_w_t.float() + out_b_1d.float()) * gate_msa_t.float()
    ln = nn.LayerNorm(D_MODEL, eps=1e-6)
    ln.weight.data.fill_(1.0)
    ln.bias.data.fill_(0.0)
    with torch.no_grad():
        normed = ln(z)
    modulated = normed * (1 + scale_mlp_t.float()) + shift_mlp_t.float()
    fc1_out = modulated @ fc1_w_t.float() + fc1_b_1d.float()
    # GELU approx
    x3 = fc1_out ** 3
    inner = 0.7978845608 * (fc1_out + 0.044715 * x3)
    gelu_out = 0.5 * fc1_out * (1.0 + torch.tanh(inner))
    fc2_out = gelu_out @ fc2_w_t.float() + fc2_b_1d.float()
    ref = z + fc2_out * gate_mlp_t.float()
    print("  Ref range: [%.4f, %.4f]" % (ref.min().item(), ref.max().item()))

    # Device tensors
    bf = lambda t: t.to(torch.bfloat16)
    attn_tt = to_tt(bf(attn_out_t), device)
    xr_tt = to_tt(bf(x_res_t), device)
    gm_tt = to_tt(bf(gate_msa_t), device)
    ow_tt = to_tt(bf(out_w_t), device)
    ob_tt = to_tt(bf(out_b_t), device)
    sh_tt = to_tt(bf(shift_mlp_t), device)
    scl_tt = to_tt(bf(scale_mlp_t), device)
    gmlp_tt = to_tt(bf(gate_mlp_t), device)
    f1w_tt = to_tt(bf(fc1_w_t), device)
    f1b_tt = to_tt(bf(fc1_b_t), device)
    f2w_tt = to_tt(bf(fc2_w_t), device)
    f2b_tt = to_tt(bf(fc2_b_t), device)

    z_scratch = to_tt(torch.zeros(SEQ, D_MODEL, dtype=torch.bfloat16), device)
    gelu_scratch = to_tt(torch.zeros(SEQ, D_MLP, dtype=torch.bfloat16), device)
    out_tt = to_tt(torch.zeros(SEQ, D_MODEL, dtype=torch.bfloat16), device)
    scaler = to_tt_l1(torch.ones(TILE, TILE, dtype=torch.bfloat16), device)
    mean_scale = to_tt_l1(torch.full((TILE, TILE), 1.0 / D_MODEL, dtype=torch.bfloat16), device)

    print("  Building kernel...")
    mega_k = make_mega_post_attn_kernel(D_TILES, D_MLP_TILES)

    print("  Running...")
    mega_k(attn_tt, xr_tt, gm_tt,
           ow_tt, ob_tt,
           sh_tt, scl_tt, gmlp_tt,
           f1w_tt, f1b_tt, f2w_tt, f2b_tt,
           scaler, mean_scale,
           z_scratch, gelu_scratch, out_tt)

    result = ttnn.to_torch(out_tt).float()
    diff = (ref[:SEQ] - result[:SEQ]).abs()
    print("  Max err: %.4f, Mean: %.4f" % (diff.max().item(), diff.mean().item()))
    print("  Result range: [%.4f, %.4f]" % (result.min().item(), result.max().item()))
    print("  Result zeros?", (result == 0).all().item())
    print("  PASS" if diff.max().item() < 3.0 else "  FAIL")

    ttnn.close_device(device)
