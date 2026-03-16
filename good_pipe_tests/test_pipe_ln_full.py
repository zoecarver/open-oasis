"""Pipe-based parallel LayerNorm for Oasis D=1024 SEQ=160.

Grid: (D_CORES, SEQ_TILES) where D_CORES evenly divides D_TILES.
X dimension: D-cores for reduction (gather+scatter within each row).
Y dimension: independent sequence rows (1 per Y-core).

Grid must satisfy: D_TILES % D_CORES == 0 and tiles_per_core >= 1.
For D=1024: (8,5)=40 cores with 4 tiles/core is the sweet spot.
"""
import torch
import ttnn
import ttl

TILE = 32
D_MODEL = 1024
N_PATCH_PAD = 160
D_TILES = D_MODEL // TILE  # 32
SEQ_TILES = N_PATCH_PAD // TILE  # 5
D_CORES = 8
TILES_PER_CORE = D_TILES // D_CORES  # 4

assert D_TILES % D_CORES == 0, "D_TILES must be divisible by D_CORES"
assert TILES_PER_CORE >= 1, "Need at least 1 tile per core"


@ttl.kernel(grid=(D_CORES, SEQ_TILES))
def pipe_ln(x, scaler, mean_scale, out):
    grid_x, grid_y = ttl.grid_size(dims=2)
    tiles_per_core = D_TILES // grid_x

    # Per-row gather+scatter: X-cores reduce D, Y-rows are independent
    gather_mean = ttl.PipeNet([
        ttl.Pipe(src=(cx, cy), dst=(0, cy))
        for cx in range(1, grid_x) for cy in range(grid_y)])
    scatter_mean = ttl.PipeNet([
        ttl.Pipe(src=(0, cy), dst=(slice(1, grid_x), cy))
        for cy in range(grid_y)])
    gather_var = ttl.PipeNet([
        ttl.Pipe(src=(cx, cy), dst=(0, cy))
        for cx in range(1, grid_x) for cy in range(grid_y)])
    scatter_var = ttl.PipeNet([
        ttl.Pipe(src=(0, cy), dst=(slice(1, grid_x), cy))
        for cy in range(grid_y)])

    x_dfb = ttl.make_dataflow_buffer_like(x, shape=(1, 1), buffer_factor=2)
    sc_dfb = ttl.make_dataflow_buffer_like(scaler, shape=(1, 1), buffer_factor=2)
    ms_dfb = ttl.make_dataflow_buffer_like(mean_scale, shape=(1, 1), buffer_factor=2)
    red_dfb = ttl.make_dataflow_buffer_like(scaler, shape=(1, 1), buffer_factor=2)
    acc_dfb = ttl.make_dataflow_buffer_like(scaler, shape=(1, 1), buffer_factor=2)
    send_dfb = ttl.make_dataflow_buffer_like(scaler, shape=(1, 1), buffer_factor=2)
    recv_dfb = ttl.make_dataflow_buffer_like(scaler, shape=(1, 1), buffer_factor=grid_x)
    mean_out_dfb = ttl.make_dataflow_buffer_like(x, shape=(1, 1), buffer_factor=2)
    mean_in_dfb = ttl.make_dataflow_buffer_like(x, shape=(1, 1), buffer_factor=2)
    send2_dfb = ttl.make_dataflow_buffer_like(scaler, shape=(1, 1), buffer_factor=2)
    recv2_dfb = ttl.make_dataflow_buffer_like(scaler, shape=(1, 1), buffer_factor=grid_x)
    istd_out_dfb = ttl.make_dataflow_buffer_like(x, shape=(1, 1), buffer_factor=2)
    istd_in_dfb = ttl.make_dataflow_buffer_like(x, shape=(1, 1), buffer_factor=2)
    sq_dfb = ttl.make_dataflow_buffer_like(x, shape=(1, 1), buffer_factor=2)
    bcast_dfb = ttl.make_dataflow_buffer_like(x, shape=(1, 1), buffer_factor=2)
    mean_dfb = ttl.make_dataflow_buffer_like(x, shape=(1, 1), buffer_factor=2)
    istd_dfb = ttl.make_dataflow_buffer_like(x, shape=(1, 1), buffer_factor=2)
    out_dfb = ttl.make_dataflow_buffer_like(out, shape=(1, 1), buffer_factor=2)

    @ttl.compute()
    def compute():
        cx, cy = ttl.core(dims=2)
        if cy < SEQ_TILES:
            with sc_dfb.wait() as sc, ms_dfb.wait() as ms:
                # === Pass 1: local reduce of tiles_per_core tiles ===
                with x_dfb.wait() as x0, red_dfb.reserve() as r:
                    r.store(ttl.math.reduce_sum(x0, sc, dims=[1]))
                with red_dfb.wait() as rv, acc_dfb.reserve() as a:
                    a.store(rv)
                for j in range(tiles_per_core - 1):
                    tile_idx = cx * tiles_per_core + j + 1
                    if tile_idx < D_TILES:
                        with x_dfb.wait() as xj, red_dfb.reserve() as r:
                            r.store(ttl.math.reduce_sum(xj, sc, dims=[1]))
                        with red_dfb.wait() as rv, acc_dfb.wait() as prev, acc_dfb.reserve() as a:
                            a.store(prev + rv)

                if cx == 0:
                    # Accumulate gathered partials
                    with acc_dfb.wait() as local_sum:
                        with recv_dfb.wait() as p0:
                            with acc_dfb.reserve() as a:
                                a.store(local_sum + p0)
                    for g in range(grid_x - 2):
                        with acc_dfb.wait() as prev, recv_dfb.wait() as pg:
                            with acc_dfb.reserve() as a:
                                a.store(prev + pg)
                    with acc_dfb.wait() as total, bcast_dfb.reserve() as bc:
                        bc.store(ttl.math.broadcast(total, dims=[1]))
                    with bcast_dfb.wait() as bc_val, mean_dfb.reserve() as m:
                        m.store(bc_val * ms)
                    with mean_dfb.wait() as mv:
                        with mean_out_dfb.reserve() as mo:
                            mo.store(mv)
                        with mean_dfb.reserve() as m2:
                            m2.store(mv)
                else:
                    with acc_dfb.wait() as rv, send_dfb.reserve() as s:
                        s.store(rv)
                    with mean_in_dfb.wait() as mv, mean_dfb.reserve() as m:
                        m.store(mv)

                # === Pass 2: variance ===
                with mean_dfb.wait() as mean_val:
                    with x_dfb.wait() as x0:
                        diff = x0 - mean_val
                        with sq_dfb.reserve() as sq:
                            sq.store(diff * diff)
                    with sq_dfb.wait() as sqv, red_dfb.reserve() as r:
                        r.store(ttl.math.reduce_sum(sqv, sc, dims=[1]))
                    with red_dfb.wait() as rv, acc_dfb.reserve() as a:
                        a.store(rv)
                    for j in range(tiles_per_core - 1):
                        tile_idx = cx * tiles_per_core + j + 1
                        if tile_idx < D_TILES:
                            with x_dfb.wait() as xj:
                                dj = xj - mean_val
                                with sq_dfb.reserve() as sq:
                                    sq.store(dj * dj)
                            with sq_dfb.wait() as sqv, red_dfb.reserve() as r:
                                r.store(ttl.math.reduce_sum(sqv, sc, dims=[1]))
                            with red_dfb.wait() as rv, acc_dfb.wait() as prev, acc_dfb.reserve() as a:
                                a.store(prev + rv)

                    if cx == 0:
                        with acc_dfb.wait() as local_sq:
                            with recv2_dfb.wait() as p0:
                                with acc_dfb.reserve() as a:
                                    a.store(local_sq + p0)
                        for g in range(grid_x - 2):
                            with acc_dfb.wait() as prev, recv2_dfb.wait() as pg:
                                with acc_dfb.reserve() as a:
                                    a.store(prev + pg)
                        with acc_dfb.wait() as total, bcast_dfb.reserve() as bc:
                            bc.store(ttl.math.broadcast(total, dims=[1]))
                        with bcast_dfb.wait() as var_bc, istd_dfb.reserve() as istd:
                            istd.store(ttl.math.rsqrt(var_bc * ms + ttl.math.fill(var_bc, 1e-6)))
                        with istd_dfb.wait() as iv:
                            with istd_out_dfb.reserve() as io:
                                io.store(iv)
                            with istd_dfb.reserve() as i2:
                                i2.store(iv)
                    else:
                        with acc_dfb.wait() as rv, send2_dfb.reserve() as s:
                            s.store(rv)
                        with istd_in_dfb.wait() as iv, istd_dfb.reserve() as istd:
                            istd.store(iv)

                    # === Pass 3: normalize ===
                    with istd_dfb.wait() as inv_std:
                        for j in range(tiles_per_core):
                            tile_idx = cx * tiles_per_core + j
                            if tile_idx < D_TILES:
                                with x_dfb.wait() as xj, out_dfb.reserve() as o:
                                    o.store((xj - mean_val) * inv_std)

    @ttl.datamovement()
    def dm_read():
        cx, cy = ttl.core(dims=2)
        col_start = cx * tiles_per_core

        with sc_dfb.reserve() as blk:
            tx = ttl.copy(scaler[0, 0], blk); tx.wait()
        with ms_dfb.reserve() as blk:
            tx = ttl.copy(mean_scale[0, 0], blk); tx.wait()

        if cy < SEQ_TILES:
            # Pass 1: load tiles
            for j in range(tiles_per_core):
                if col_start + j < D_TILES:
                    with x_dfb.reserve() as blk:
                        tx = ttl.copy(x[cy, col_start + j], blk); tx.wait()

            def send_p1(pipe):
                with send_dfb.wait() as blk:
                    xf = ttl.copy(blk, pipe); xf.wait()
            gather_mean.if_src(send_p1)

            def recv_p1(pipe):
                with recv_dfb.reserve() as blk:
                    xf = ttl.copy(pipe, blk); xf.wait()
            gather_mean.if_dst(recv_p1)

            def send_mean(pipe):
                with mean_out_dfb.wait() as blk:
                    xf = ttl.copy(blk, pipe); xf.wait()
            scatter_mean.if_src(send_mean)

            def recv_mean(pipe):
                with mean_in_dfb.reserve() as blk:
                    xf = ttl.copy(pipe, blk); xf.wait()
            scatter_mean.if_dst(recv_mean)

            # Pass 2: reload tiles
            for j in range(tiles_per_core):
                if col_start + j < D_TILES:
                    with x_dfb.reserve() as blk:
                        tx = ttl.copy(x[cy, col_start + j], blk); tx.wait()

            def send_p2(pipe):
                with send2_dfb.wait() as blk:
                    xf = ttl.copy(blk, pipe); xf.wait()
            gather_var.if_src(send_p2)

            def recv_p2(pipe):
                with recv2_dfb.reserve() as blk:
                    xf = ttl.copy(pipe, blk); xf.wait()
            gather_var.if_dst(recv_p2)

            def send_istd(pipe):
                with istd_out_dfb.wait() as blk:
                    xf = ttl.copy(blk, pipe); xf.wait()
            scatter_var.if_src(send_istd)

            def recv_istd(pipe):
                with istd_in_dfb.reserve() as blk:
                    xf = ttl.copy(pipe, blk); xf.wait()
            scatter_var.if_dst(recv_istd)

            # Pass 3: reload tiles
            for j in range(tiles_per_core):
                if col_start + j < D_TILES:
                    with x_dfb.reserve() as blk:
                        tx = ttl.copy(x[cy, col_start + j], blk); tx.wait()

    @ttl.datamovement()
    def dm_write():
        cx, cy = ttl.core(dims=2)
        col_start = cx * tiles_per_core
        if cy < SEQ_TILES:
            for j in range(tiles_per_core):
                if col_start + j < D_TILES:
                    with out_dfb.wait() as blk:
                        tx = ttl.copy(blk, out[cy, col_start + j]); tx.wait()


def to_tt(t, device):
    return ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                           device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)

def to_tt_l1(t, device):
    return ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                           device=device, memory_config=ttnn.L1_MEMORY_CONFIG)


if __name__ == "__main__":
    default_size = ttnn.device.get_max_worker_l1_unreserved_size()
    # (8,5) grid needs ~78KB config buffer, default is ~70KB
    device = ttnn.open_device(device_id=0, worker_l1_size=default_size - 81920)
    torch.manual_seed(42)

    print("=== Pipe LayerNorm (grid=auto) D=%d, SEQ=%d ===" % (D_MODEL, N_PATCH_PAD))

    x_torch = torch.randn(N_PATCH_PAD, D_MODEL, dtype=torch.float32) * 0.5

    x_f = x_torch.float()
    mean = x_f.mean(dim=-1, keepdim=True)
    var = x_f.var(dim=-1, keepdim=True, unbiased=False)
    ref = ((x_f - mean) / (var + 1e-6).sqrt()).to(torch.float32)

    x_tt = to_tt(x_torch.to(torch.bfloat16), device)
    out_tt = to_tt(torch.zeros(N_PATCH_PAD, D_MODEL, dtype=torch.bfloat16), device)
    scaler = to_tt_l1(torch.ones(TILE, TILE, dtype=torch.bfloat16), device)
    mean_scale = to_tt_l1(torch.full((TILE, TILE), 1.0 / D_MODEL, dtype=torch.bfloat16), device)

    print("Running...")
    pipe_ln(x_tt, scaler, mean_scale, out_tt)

    result = ttnn.to_torch(out_tt).float()
    diff = (ref - result).abs()
    print("Max error: %.4f, Mean: %.4f" % (diff.max().item(), diff.mean().item()))
    print("Result[0, 0:5]:", result[0, 0:5].tolist())
    print("Ref[0, 0:5]:", ref[0, 0:5].tolist())
    print("Result[64, 0:5]:", result[64, 0:5].tolist())
    print("Ref[64, 0:5]:", ref[64, 0:5].tolist())
    print("PASS" if diff.max().item() < 1.0 else "FAIL")

    ttnn.close_device(device)
