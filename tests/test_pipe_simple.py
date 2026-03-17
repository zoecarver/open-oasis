"""Step 1: Simplest pipe test - 2 cores, forward partial sum.

Core 0: reduces its tile -> sends partial sum via pipe to core 1.
Core 1: reduces its tile + receives partial -> has total sum.

No broadcast, no variance, no normalize. Just verify pipe send/recv works.
"""
import torch
import ttnn
import ttl

TILE = 32

@ttl.kernel(grid=(2, 1))
def pipe_sum_2core(a, scaler, out):
    """2-core partial sum via pipe. Core 0 sends partial to core 1."""
    fwd = ttl.Pipe(src=(0, 0), dst=(1, 0))

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
            # Each core reduces its tile
            with a_dfb.wait() as tile, red_dfb.reserve() as r:
                r.store(ttl.math.reduce_sum(tile, sc, dims=[1]))

            if cx == 0:
                # Core 0: forward local sum
                with red_dfb.wait() as rv, fwd_out_dfb.reserve() as fo:
                    fo.store(rv)
                # Core 0 has no output
            else:
                # Core 1: add incoming partial + local
                with red_dfb.wait() as rv, fwd_in_dfb.wait() as pv, out_dfb.reserve() as o:
                    o.store(rv + pv)

    @ttl.datamovement()
    def dm_read():
        cx, _ = ttl.core(dims=2)
        with sc_dfb.reserve() as blk:
            tx = ttl.copy(scaler[0, 0], blk); tx.wait()
        with a_dfb.reserve() as blk:
            tx = ttl.copy(a[0, cx], blk); tx.wait()
        # Core 1 receives from pipe
        if cx == 1:
            with fwd_in_dfb.reserve() as blk:
                tx = ttl.copy(fwd, blk); tx.wait()

    @ttl.datamovement()
    def dm_write():
        cx, _ = ttl.core(dims=2)
        # Core 0 sends via pipe
        if cx == 0:
            with fwd_out_dfb.wait() as blk:
                tx = ttl.copy(blk, fwd); tx.wait()
        # Core 1 writes result
        if cx == 1:
            with out_dfb.wait() as blk:
                tx = ttl.copy(blk, out[0, 0]); tx.wait()


if __name__ == "__main__":
    import os
    os.environ["TT_METAL_SLOW_DISPATCH_MODE"] = "1"
    device = ttnn.open_device(device_id=0)
    torch.manual_seed(42)

    # 1 row, 2 tiles wide = (32, 64)
    a_torch = torch.ones(TILE, 2 * TILE, dtype=torch.bfloat16)
    out_torch = torch.zeros(TILE, TILE, dtype=torch.bfloat16)
    scaler_torch = torch.ones(TILE, TILE, dtype=torch.bfloat16)

    a_tt = ttnn.from_torch(a_torch, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                           device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    out_tt = ttnn.from_torch(out_torch, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                             device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    scaler_tt = ttnn.from_torch(scaler_torch, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                                device=device, memory_config=ttnn.L1_MEMORY_CONFIG)

    print("=== Pipe Sum 2-Core Test ===")
    print("Input: all 1s, (32, 64) = 2 tiles")
    print("Expected: reduce_sum each tile cols -> (32,1), sum both = 64 per row")
    pipe_sum_2core(a_tt, scaler_tt, out_tt)

    result = ttnn.to_torch(out_tt).float()
    print("Result[0,0]:", result[0, 0].item())
    print("Expected: 64.0")
    print("PASS" if abs(result[0, 0].item() - 64.0) < 1.0 else "FAIL")

    ttnn.close_device(device)
