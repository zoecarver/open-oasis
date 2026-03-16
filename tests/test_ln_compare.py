"""Compare pipe vs non-pipe fused LN kernels on identical inputs."""
import torch
import ttnn
import sys
sys.path.insert(0, "/tmp")
from pipe_fused_ln import make_pipe_fused_gated_res_ln_adaln

TILE = 32
D_MODEL = 1024
N_PATCH_PAD = 160
D_TILES = D_MODEL // TILE
SEQ_TILES = N_PATCH_PAD // TILE
D_CORES = 8
SEQ = N_PATCH_PAD  # single frame


def to_tt(t, device):
    return ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                           device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)

def to_tt_l1(t, device):
    return ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                           device=device, memory_config=ttnn.L1_MEMORY_CONFIG)


# Inline the non-pipe kernel here to avoid import issues
import ttl

def make_fused_gated_res_ln_adaln_kernel(dim_tiles):
    @ttl.kernel(grid="auto")
    def fused_kernel(residual, x, gate, scaler, mean_scale, adaln_packed, gated_res_out, out):
        grid_cols, _ = ttl.grid_size(dims=2)
        seq_tiles = residual.shape[0] // TILE
        tiles_per_core = -(-seq_tiles // grid_cols)
        res_dfb = ttl.make_dataflow_buffer_like(residual, shape=(1, 1), buffer_factor=2)
        x_dfb = ttl.make_dataflow_buffer_like(x, shape=(1, 1), buffer_factor=2)
        g_dfb = ttl.make_dataflow_buffer_like(gate, shape=(1, 1), buffer_factor=2)
        gr_dfb = ttl.make_dataflow_buffer_like(residual, shape=(1, 1), buffer_factor=2)
        sc_dfb = ttl.make_dataflow_buffer_like(scaler, shape=(1, 1), buffer_factor=1)
        ms_dfb = ttl.make_dataflow_buffer_like(mean_scale, shape=(1, 1), buffer_factor=1)
        red_dfb = ttl.make_dataflow_buffer_like(scaler, shape=(1, 1), buffer_factor=2)
        acc_dfb = ttl.make_dataflow_buffer_like(scaler, shape=(1, 1), buffer_factor=2)
        bcast_dfb = ttl.make_dataflow_buffer_like(residual, shape=(1, 1), buffer_factor=2)
        sq_dfb = ttl.make_dataflow_buffer_like(residual, shape=(1, 1), buffer_factor=2)
        mean_dfb = ttl.make_dataflow_buffer_like(residual, shape=(1, 1), buffer_factor=2)
        istd_dfb = ttl.make_dataflow_buffer_like(residual, shape=(1, 1), buffer_factor=2)
        sh_dfb = ttl.make_dataflow_buffer_like(adaln_packed, shape=(1, 1), buffer_factor=2)
        scl_dfb = ttl.make_dataflow_buffer_like(adaln_packed, shape=(1, 1), buffer_factor=2)
        gro_dfb = ttl.make_dataflow_buffer_like(gated_res_out, shape=(1, 1), buffer_factor=2)
        out_dfb = ttl.make_dataflow_buffer_like(out, shape=(1, 1), buffer_factor=2)

        @ttl.compute()
        def compute():
            core_x, _ = ttl.core(dims=2)
            with sc_dfb.wait() as sc, ms_dfb.wait() as ms:
                for local_t in range(tiles_per_core):
                    tile_idx = core_x * tiles_per_core + local_t
                    if tile_idx < seq_tiles:
                        with res_dfb.wait() as rv, x_dfb.wait() as xv, g_dfb.wait() as gv:
                            with gr_dfb.reserve() as gr:
                                gr.store(rv + xv * gv)
                        with gr_dfb.wait() as grv:
                            with gro_dfb.reserve() as gro:
                                gro.store(grv)
                            with red_dfb.reserve() as r:
                                r.store(ttl.math.reduce_sum(grv, sc, dims=[1]))
                        with red_dfb.wait() as rv, acc_dfb.reserve() as acc:
                            acc.store(rv)
                        for j in range(dim_tiles - 1):
                            with res_dfb.wait() as rv, x_dfb.wait() as xv, g_dfb.wait() as gv:
                                with gr_dfb.reserve() as gr:
                                    gr.store(rv + xv * gv)
                            with gr_dfb.wait() as grv:
                                with gro_dfb.reserve() as gro:
                                    gro.store(grv)
                                with red_dfb.reserve() as r:
                                    r.store(ttl.math.reduce_sum(grv, sc, dims=[1]))
                            with red_dfb.wait() as rv, acc_dfb.wait() as av, acc_dfb.reserve() as acc:
                                acc.store(av + rv)
                        with acc_dfb.wait() as sum_x, bcast_dfb.reserve() as bc:
                            bc.store(ttl.math.broadcast(sum_x, dims=[1]))
                        with bcast_dfb.wait() as sum_x_bc, mean_dfb.reserve() as mean_out:
                            mean_out.store(sum_x_bc * ms)
                        with mean_dfb.wait() as mean_val:
                            with res_dfb.wait() as rv, x_dfb.wait() as xv, g_dfb.wait() as gv:
                                with gr_dfb.reserve() as gr:
                                    gr.store(rv + xv * gv)
                            with gr_dfb.wait() as grv:
                                with sq_dfb.reserve() as sq:
                                    sq.store((grv - mean_val) * (grv - mean_val))
                            with sq_dfb.wait() as sqv, red_dfb.reserve() as r:
                                r.store(ttl.math.reduce_sum(sqv, sc, dims=[1]))
                            with red_dfb.wait() as rv, acc_dfb.reserve() as acc:
                                acc.store(rv)
                            for j in range(dim_tiles - 1):
                                with res_dfb.wait() as rv, x_dfb.wait() as xv, g_dfb.wait() as gv:
                                    with gr_dfb.reserve() as gr:
                                        gr.store(rv + xv * gv)
                                with gr_dfb.wait() as grv:
                                    with sq_dfb.reserve() as sq:
                                        sq.store((grv - mean_val) * (grv - mean_val))
                                with sq_dfb.wait() as sqv, red_dfb.reserve() as r:
                                    r.store(ttl.math.reduce_sum(sqv, sc, dims=[1]))
                                with red_dfb.wait() as rv, acc_dfb.wait() as av, acc_dfb.reserve() as acc:
                                    acc.store(av + rv)
                            with acc_dfb.wait() as sum_sq, bcast_dfb.reserve() as bc:
                                bc.store(ttl.math.broadcast(sum_sq, dims=[1]))
                            with bcast_dfb.wait() as var_bc, istd_dfb.reserve() as istd:
                                istd.store(ttl.math.rsqrt(var_bc * ms + ttl.math.fill(var_bc, 1e-6)))
                            with istd_dfb.wait() as inv_std:
                                for j in range(dim_tiles):
                                    with res_dfb.wait() as rv, x_dfb.wait() as xv, g_dfb.wait() as gv:
                                        with gr_dfb.reserve() as gr:
                                            gr.store(rv + xv * gv)
                                    with gr_dfb.wait() as grv, sh_dfb.wait() as shv, scl_dfb.wait() as sclv, out_dfb.reserve() as o:
                                        normed = (grv - mean_val) * inv_std
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
                    for j in range(dim_tiles):
                        with res_dfb.reserve() as blk:
                            tx = ttl.copy(residual[tile_idx, j], blk); tx.wait()
                        with x_dfb.reserve() as blk:
                            tx = ttl.copy(x[tile_idx, j], blk); tx.wait()
                        with g_dfb.reserve() as blk:
                            tx = ttl.copy(gate[tile_idx, j], blk); tx.wait()
                    for j in range(dim_tiles):
                        with res_dfb.reserve() as blk:
                            tx = ttl.copy(residual[tile_idx, j], blk); tx.wait()
                        with x_dfb.reserve() as blk:
                            tx = ttl.copy(x[tile_idx, j], blk); tx.wait()
                        with g_dfb.reserve() as blk:
                            tx = ttl.copy(gate[tile_idx, j], blk); tx.wait()
                    for j in range(dim_tiles):
                        with res_dfb.reserve() as blk:
                            tx = ttl.copy(residual[tile_idx, j], blk); tx.wait()
                        with x_dfb.reserve() as blk:
                            tx = ttl.copy(x[tile_idx, j], blk); tx.wait()
                        with g_dfb.reserve() as blk:
                            tx = ttl.copy(gate[tile_idx, j], blk); tx.wait()
                        with sh_dfb.reserve() as blk:
                            tx = ttl.copy(adaln_packed[tile_idx, 3 * dim_tiles + j], blk); tx.wait()
                        with scl_dfb.reserve() as blk:
                            tx = ttl.copy(adaln_packed[tile_idx, 4 * dim_tiles + j], blk); tx.wait()

        @ttl.datamovement()
        def dm_write():
            core_x, _ = ttl.core(dims=2)
            for local_t in range(tiles_per_core):
                tile_idx = core_x * tiles_per_core + local_t
                if tile_idx < seq_tiles:
                    for j in range(dim_tiles):
                        with gro_dfb.wait() as blk:
                            tx = ttl.copy(blk, gated_res_out[tile_idx, j]); tx.wait()
                    for j in range(dim_tiles):
                        with out_dfb.wait() as blk:
                            tx = ttl.copy(blk, out[tile_idx, j]); tx.wait()
    return fused_kernel


if __name__ == "__main__":
    default_size = ttnn.device.get_max_worker_l1_unreserved_size()
    device = ttnn.open_device(device_id=0, worker_l1_size=default_size - 90112)
    torch.manual_seed(42)

    print("=== Pipe vs Non-Pipe LN Comparison ===")

    residual = torch.randn(N_PATCH_PAD, D_MODEL, dtype=torch.bfloat16) * 0.3
    x_in = torch.randn(N_PATCH_PAD, D_MODEL, dtype=torch.bfloat16) * 0.3
    gate = torch.randn(N_PATCH_PAD, D_MODEL, dtype=torch.bfloat16) * 0.1
    adaln_packed = torch.randn(N_PATCH_PAD, 6 * D_MODEL, dtype=torch.bfloat16) * 0.1
    scaler_t = torch.ones(TILE, TILE, dtype=torch.bfloat16)
    mean_scale_t = torch.full((TILE, TILE), 1.0 / D_MODEL, dtype=torch.bfloat16)

    # Run non-pipe kernel
    res_tt = to_tt(residual, device)
    x_tt = to_tt(x_in, device)
    gate_tt = to_tt(gate, device)
    adaln_tt = to_tt(adaln_packed, device)
    gro_np = to_tt(torch.zeros(N_PATCH_PAD, D_MODEL, dtype=torch.bfloat16), device)
    out_np = to_tt(torch.zeros(N_PATCH_PAD, D_MODEL, dtype=torch.bfloat16), device)
    scaler = to_tt_l1(scaler_t, device)
    ms = to_tt_l1(mean_scale_t, device)

    nonpipe = make_fused_gated_res_ln_adaln_kernel(D_TILES)
    print("Running non-pipe...")
    nonpipe(res_tt, x_tt, gate_tt, scaler, ms, adaln_tt, gro_np, out_np)
    np_gro = ttnn.to_torch(gro_np).float()
    np_out = ttnn.to_torch(out_np).float()

    # Run pipe kernel (fresh tensors, same data)
    res_tt2 = to_tt(residual, device)
    x_tt2 = to_tt(x_in, device)
    gate_tt2 = to_tt(gate, device)
    adaln_tt2 = to_tt(adaln_packed, device)
    gro_p = to_tt(torch.zeros(N_PATCH_PAD, D_MODEL, dtype=torch.bfloat16), device)
    out_p = to_tt(torch.zeros(N_PATCH_PAD, D_MODEL, dtype=torch.bfloat16), device)
    scaler2 = to_tt_l1(scaler_t, device)
    ms2 = to_tt_l1(mean_scale_t, device)

    pipe_k = make_pipe_fused_gated_res_ln_adaln(D_TILES, D_CORES, SEQ_TILES)
    print("Running pipe...")
    pipe_k(res_tt2, x_tt2, gate_tt2, scaler2, ms2, adaln_tt2, gro_p, out_p)
    p_gro = ttnn.to_torch(gro_p).float()
    p_out = ttnn.to_torch(out_p).float()

    # Compare
    gro_diff = (np_gro - p_gro).abs()
    out_diff = (np_out - p_out).abs()
    print("\n=== gated_res (pipe vs non-pipe) ===")
    print("max_err: %.6f, mean: %.6f" % (gro_diff.max().item(), gro_diff.mean().item()))

    print("\n=== modulated output (pipe vs non-pipe) ===")
    print("max_err: %.6f, mean: %.6f" % (out_diff.max().item(), out_diff.mean().item()))

    # Per-row analysis
    row_max = out_diff.max(dim=1).values
    print("\nPer-row max error:")
    for i in range(N_PATCH_PAD // TILE):
        chunk = row_max[i*TILE:(i+1)*TILE]
        print("  rows %d-%d: max=%.6f mean=%.6f" % (
            i*TILE, (i+1)*TILE-1, chunk.max().item(), chunk.mean().item()))

    # Check if any rows are wildly off
    bad_rows = (row_max > 0.5).nonzero(as_tuple=True)[0]
    if len(bad_rows) > 0:
        print("\nBAD ROWS (err > 0.5):", bad_rows.tolist()[:10])
        r = bad_rows[0].item()
        print("  non-pipe[%d, 0:5]: %s" % (r, np_out[r, 0:5].tolist()))
        print("  pipe[%d, 0:5]:     %s" % (r, p_out[r, 0:5].tolist()))
    else:
        print("\nNo bad rows (all < 0.5)")

    print("\nPASS" if out_diff.max().item() < 1.0 else "FAIL")
    ttnn.close_device(device)
