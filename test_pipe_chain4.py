"""4-core forward chain sum reduction using PipeNet (engram pattern).

Both pipe send and recv in dm_read, matching the proven engram pattern.
dm_read waits on compute output to send via pipe.
"""
import torch
import ttnn
import ttl
import os

TILE = 32

@ttl.kernel(grid=(4, 1))
def pipe_chain4(a, scaler, out):
    """4-core forward chain partial sum. Pipe ops in dm_read via PipeNet."""
    pipes = [ttl.Pipe((x, 0), ((x + 1), 0)) for x in range(3)]
    net = ttl.PipeNet(pipes)

    a_dfb = ttl.make_dataflow_buffer_like(a, shape=(1, 1), buffer_factor=2)
    sc_dfb = ttl.make_dataflow_buffer_like(scaler, shape=(1, 1), buffer_factor=2)
    red_dfb = ttl.make_dataflow_buffer_like(scaler, shape=(1, 1), buffer_factor=2)
    fwd_in_dfb = ttl.make_dataflow_buffer_like(scaler, shape=(1, 1), buffer_factor=2)
    fwd_out_dfb = ttl.make_dataflow_buffer_like(scaler, shape=(1, 1), buffer_factor=2)
    out_dfb = ttl.make_dataflow_buffer_like(out, shape=(1, 1), buffer_factor=2)

    @ttl.compute()
    def compute():
        cx, _ = ttl.core(dims=2)
        with sc_dfb.wait() as sc:
            with a_dfb.wait() as tile, red_dfb.reserve() as r:
                r.store(ttl.math.reduce_sum(tile, sc, dims=[1]))

            if cx == 0:
                # Leftmost: forward local sum
                with red_dfb.wait() as rv, fwd_out_dfb.reserve() as fo:
                    fo.store(rv)
            elif cx < 3:
                # Middle: add incoming partial, forward
                with red_dfb.wait() as rv, fwd_in_dfb.wait() as pv, fwd_out_dfb.reserve() as fo:
                    fo.store(rv + pv)
            else:
                # Rightmost: add incoming partial -> output
                with red_dfb.wait() as rv, fwd_in_dfb.wait() as pv, out_dfb.reserve() as o:
                    o.store(rv + pv)

    @ttl.datamovement()
    def dm_read():
        cx, _ = ttl.core(dims=2)
        # Load constants and input tile
        with sc_dfb.reserve() as blk:
            tx = ttl.copy(scaler[0, 0], blk); tx.wait()
        with a_dfb.reserve() as blk:
            tx = ttl.copy(a[0, cx], blk); tx.wait()

        # Cores 1-3: receive partial sum from previous core
        if cx > 0:
            with fwd_in_dfb.reserve() as blk:
                def recv(pipe):
                    xf = ttl.copy(pipe, blk); xf.wait()
                net.if_dst(recv)

        # Cores 0-2: wait for compute to produce partial, send to next core
        if cx < 3:
            with fwd_out_dfb.wait() as blk:
                def send(pipe):
                    xf = ttl.copy(blk, pipe); xf.wait()
                net.if_src(send)

    @ttl.datamovement()
    def dm_write():
        cx, _ = ttl.core(dims=2)
        # Core 3: write final output
        if cx == 3:
            with out_dfb.wait() as blk:
                tx = ttl.copy(blk, out[0, 0]); tx.wait()


if __name__ == "__main__":
    device = ttnn.open_device(device_id=0)
    torch.manual_seed(42)

    a_torch = torch.ones(TILE, 4 * TILE, dtype=torch.bfloat16)
    out_torch = torch.zeros(TILE, TILE, dtype=torch.bfloat16)
    scaler_torch = torch.ones(TILE, TILE, dtype=torch.bfloat16)

    a_tt = ttnn.from_torch(a_torch, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                           device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    out_tt = ttnn.from_torch(out_torch, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                             device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    scaler_tt = ttnn.from_torch(scaler_torch, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                                device=device, memory_config=ttnn.L1_MEMORY_CONFIG)

    print("=== 4-Core Forward Chain Sum ===")
    print("Input: all 1s, (32, 128) = 4 tiles, reduce each col -> sum all 4")
    print("Expected: 128.0 per row")
    pipe_chain4(a_tt, scaler_tt, out_tt)

    result = ttnn.to_torch(out_tt).float()
    print("Result[0,0]:", result[0, 0].item())
    print("PASS" if abs(result[0, 0].item() - 128.0) < 2.0 else "FAIL")

    ttnn.close_device(device)
