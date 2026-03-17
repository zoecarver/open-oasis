"""Pipe-based parallel LayerNorm using gather + scatter pattern.

Pass 1: each core reduces locally, gather partials to core 0, compute mean, scatter back.
Pass 2: each core computes (x-mean)^2, gather to core 0, compute inv_std, scatter back.
Pass 3: each core normalizes its tile.

Each pipe used exactly once. Separate PipeNets for each gather/scatter operation.
"""
import torch
import ttnn
import ttl

TILE = 32
D_MODEL = 128
N_PATCH_PAD = 32  # 1 row
D_TILES = D_MODEL // TILE  # 4
CORES = D_TILES


@ttl.kernel(grid=(CORES, 1))
def pipe_layernorm_1row(x, scaler, mean_scale, out):
    # Gather: cores 1-3 send to core 0 (sequential per source)
    gather_mean = ttl.PipeNet([
        ttl.Pipe(src=(cx, 0), dst=(0, 0)) for cx in range(1, CORES)])
    scatter_mean = ttl.PipeNet([
        ttl.Pipe(src=(0, 0), dst=(slice(1, CORES), 0))])
    gather_var = ttl.PipeNet([
        ttl.Pipe(src=(cx, 0), dst=(0, 0)) for cx in range(1, CORES)])
    scatter_var = ttl.PipeNet([
        ttl.Pipe(src=(0, 0), dst=(slice(1, CORES), 0))])

    x_dfb = ttl.make_dataflow_buffer_like(x, shape=(1, 1), buffer_factor=2)
    sc_dfb = ttl.make_dataflow_buffer_like(scaler, shape=(1, 1), buffer_factor=2)
    ms_dfb = ttl.make_dataflow_buffer_like(mean_scale, shape=(1, 1), buffer_factor=2)
    red_dfb = ttl.make_dataflow_buffer_like(scaler, shape=(1, 1), buffer_factor=2)
    # Pipe I/O DFBs
    send1_dfb = ttl.make_dataflow_buffer_like(scaler, shape=(1, 1), buffer_factor=2)
    # 4 slots: 3 for simultaneous waits in compute + 1 headroom
    recv1_dfb = ttl.make_dataflow_buffer_like(scaler, shape=(1, 1), buffer_factor=4)
    mean_out_dfb = ttl.make_dataflow_buffer_like(x, shape=(1, 1), buffer_factor=2)
    mean_in_dfb = ttl.make_dataflow_buffer_like(x, shape=(1, 1), buffer_factor=2)
    send2_dfb = ttl.make_dataflow_buffer_like(scaler, shape=(1, 1), buffer_factor=2)
    recv2_dfb = ttl.make_dataflow_buffer_like(scaler, shape=(1, 1), buffer_factor=4)
    istd_out_dfb = ttl.make_dataflow_buffer_like(x, shape=(1, 1), buffer_factor=2)
    istd_in_dfb = ttl.make_dataflow_buffer_like(x, shape=(1, 1), buffer_factor=2)
    # Compute intermediates
    sq_dfb = ttl.make_dataflow_buffer_like(x, shape=(1, 1), buffer_factor=2)
    bcast_dfb = ttl.make_dataflow_buffer_like(x, shape=(1, 1), buffer_factor=2)
    mean_dfb = ttl.make_dataflow_buffer_like(x, shape=(1, 1), buffer_factor=2)
    istd_dfb = ttl.make_dataflow_buffer_like(x, shape=(1, 1), buffer_factor=2)
    out_dfb = ttl.make_dataflow_buffer_like(out, shape=(1, 1), buffer_factor=2)

    @ttl.compute()
    def compute():
        cx, _ = ttl.core(dims=2)
        with sc_dfb.wait() as sc, ms_dfb.wait() as ms:
            # === Pass 1: local reduce ===
            with x_dfb.wait() as x0, red_dfb.reserve() as r:
                r.store(ttl.math.reduce_sum(x0, sc, dims=[1]))

            if cx == 0:
                # Core 0: accumulate own partial + received partials
                with red_dfb.wait() as local_sum:
                    with recv1_dfb.wait() as p1, recv1_dfb.wait() as p2, recv1_dfb.wait() as p3:
                        with bcast_dfb.reserve() as total:
                            total.store(local_sum + p1 + p2 + p3)
                with bcast_dfb.wait() as tv:
                    with mean_dfb.reserve() as bc:
                        bc.store(ttl.math.broadcast(tv, dims=[1]))
                with mean_dfb.wait() as bc_val, mean_dfb.reserve() as m:
                    m.store(bc_val * ms)
                # Send mean to scatter and keep a local copy
                with mean_dfb.wait() as mv:
                    with mean_out_dfb.reserve() as mo:
                        mo.store(mv)
                    with mean_dfb.reserve() as m2:
                        m2.store(mv)
            else:
                # Non-zero cores: send local partial, receive mean
                with red_dfb.wait() as rv, send1_dfb.reserve() as s:
                    s.store(rv)
                with mean_in_dfb.wait() as mv, mean_dfb.reserve() as m:
                    m.store(mv)

            # === Pass 2: (x - mean)^2, reduce, gather, scatter inv_std ===
            with mean_dfb.wait() as mean_val:
                with x_dfb.wait() as x0:
                    diff = x0 - mean_val
                    with sq_dfb.reserve() as sq:
                        sq.store(diff * diff)
                with sq_dfb.wait() as sqv, red_dfb.reserve() as r:
                    r.store(ttl.math.reduce_sum(sqv, sc, dims=[1]))

                if cx == 0:
                    with red_dfb.wait() as local_sq:
                        with recv2_dfb.wait() as p1, recv2_dfb.wait() as p2, recv2_dfb.wait() as p3:
                            with bcast_dfb.reserve() as total:
                                total.store(local_sq + p1 + p2 + p3)
                    with bcast_dfb.wait() as tv:
                        with istd_dfb.reserve() as bc:
                            bc.store(ttl.math.broadcast(tv, dims=[1]))
                    with istd_dfb.wait() as var_bc, istd_dfb.reserve() as istd:
                        istd.store(ttl.math.rsqrt(var_bc * ms + ttl.math.fill(var_bc, 1e-6)))
                    # Send inv_std to scatter and keep a local copy
                    with istd_dfb.wait() as iv:
                        with istd_out_dfb.reserve() as io:
                            io.store(iv)
                        with istd_dfb.reserve() as i2:
                            i2.store(iv)
                else:
                    with red_dfb.wait() as rv, send2_dfb.reserve() as s:
                        s.store(rv)
                    with istd_in_dfb.wait() as iv, istd_dfb.reserve() as istd:
                        istd.store(iv)

                # === Pass 3: normalize ===
                with istd_dfb.wait() as inv_std:
                    with x_dfb.wait() as x0, out_dfb.reserve() as o:
                        o.store((x0 - mean_val) * inv_std)

    @ttl.datamovement()
    def dm_read():
        cx, _ = ttl.core(dims=2)
        with sc_dfb.reserve() as blk:
            tx = ttl.copy(scaler[0, 0], blk); tx.wait()
        with ms_dfb.reserve() as blk:
            tx = ttl.copy(mean_scale[0, 0], blk); tx.wait()

        # === Pass 1: load tile ===
        with x_dfb.reserve() as blk:
            tx = ttl.copy(x[0, cx], blk); tx.wait()

        # Gather mean partials: DFB ops inside callbacks so they only
        # execute on the cores that are actually src/dst of the pipe.
        def send_partial1(pipe):
            with send1_dfb.wait() as blk:
                xf = ttl.copy(blk, pipe); xf.wait()
        gather_mean.if_src(send_partial1)

        def recv_partial1(pipe):
            with recv1_dfb.reserve() as blk:
                xf = ttl.copy(pipe, blk); xf.wait()
        gather_mean.if_dst(recv_partial1)

        # Scatter mean from core 0
        def send_mean(pipe):
            with mean_out_dfb.wait() as blk:
                xf = ttl.copy(blk, pipe); xf.wait()
        scatter_mean.if_src(send_mean)

        def recv_mean(pipe):
            with mean_in_dfb.reserve() as blk:
                xf = ttl.copy(pipe, blk); xf.wait()
        scatter_mean.if_dst(recv_mean)

        # === Pass 2: reload tile ===
        with x_dfb.reserve() as blk:
            tx = ttl.copy(x[0, cx], blk); tx.wait()

        # Gather variance partials
        def send_partial2(pipe):
            with send2_dfb.wait() as blk:
                xf = ttl.copy(blk, pipe); xf.wait()
        gather_var.if_src(send_partial2)

        def recv_partial2(pipe):
            with recv2_dfb.reserve() as blk:
                xf = ttl.copy(pipe, blk); xf.wait()
        gather_var.if_dst(recv_partial2)

        # Scatter inv_std from core 0
        def send_istd(pipe):
            with istd_out_dfb.wait() as blk:
                xf = ttl.copy(blk, pipe); xf.wait()
        scatter_var.if_src(send_istd)

        def recv_istd(pipe):
            with istd_in_dfb.reserve() as blk:
                xf = ttl.copy(pipe, blk); xf.wait()
        scatter_var.if_dst(recv_istd)

        # === Pass 3: reload tile ===
        with x_dfb.reserve() as blk:
            tx = ttl.copy(x[0, cx], blk); tx.wait()

    @ttl.datamovement()
    def dm_write():
        cx, _ = ttl.core(dims=2)
        with out_dfb.wait() as blk:
            tx = ttl.copy(blk, out[0, cx]); tx.wait()


def to_tt(t, device):
    return ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                           device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)

def to_tt_l1(t, device):
    return ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                           device=device, memory_config=ttnn.L1_MEMORY_CONFIG)


if __name__ == "__main__":
    device = ttnn.open_device(device_id=0)
    torch.manual_seed(42)

    print("=== Pipe LayerNorm (gather+scatter) ===")
    print("D_MODEL=%d (%d tiles), SEQ=%d, %d cores" % (D_MODEL, D_TILES, N_PATCH_PAD, CORES))

    x_torch = torch.randn(N_PATCH_PAD, D_MODEL, dtype=torch.float32) * 0.5

    x_f = x_torch.float()
    mean = x_f.mean(dim=-1, keepdim=True)
    var = x_f.var(dim=-1, keepdim=True, unbiased=False)
    ref = ((x_f - mean) / (var + 1e-6).sqrt()).to(torch.float32)
    print("Reference range: [%.4f, %.4f]" % (ref.min().item(), ref.max().item()))

    x_tt = to_tt(x_torch.to(torch.bfloat16), device)
    out_tt = to_tt(torch.zeros(N_PATCH_PAD, D_MODEL, dtype=torch.bfloat16), device)
    scaler = to_tt_l1(torch.ones(TILE, TILE, dtype=torch.bfloat16), device)
    mean_scale = to_tt_l1(torch.full((TILE, TILE), 1.0 / D_MODEL, dtype=torch.bfloat16), device)

    print("Running...")
    pipe_layernorm_1row(x_tt, scaler, mean_scale, out_tt)

    result = ttnn.to_torch(out_tt).float()
    diff = (ref - result).abs()
    print("Max error: %.4f, Mean: %.4f" % (diff.max().item(), diff.mean().item()))
    print("Result[0, 0:5]:", result[0, 0:5].tolist())
    print("Ref[0, 0:5]:", ref[0, 0:5].tolist())
    print("PASS" if diff.max().item() < 1.0 else "FAIL")

    ttnn.close_device(device)
