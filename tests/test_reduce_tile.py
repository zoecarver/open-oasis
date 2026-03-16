"""Test 3: reduce_sum on (1,1) tile with dims=[1].
Verify what reduce_sum actually produces and that accumulating
partial sums across tiles gives the correct row-wise sum.
"""
import torch
import ttnn
import ttl

TILE = 32
D_MODEL = 1024
D_TILES = D_MODEL // TILE


def to_tt(t, device):
    return ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                           device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)

def to_tt_l1(t, device):
    return ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                           device=device, memory_config=ttnn.L1_MEMORY_CONFIG)


def make_reduce_acc_kernel(n_tiles):
    """Reduce n_tiles of (1,1) along dim=1, accumulate into single (1,1) output."""
    @ttl.kernel(grid=(1, 1))
    def reduce_acc(inp, scaler, out):
        inp_dfb = ttl.make_dataflow_buffer_like(inp, shape=(1, 1), buffer_factor=2)
        sc_dfb = ttl.make_dataflow_buffer_like(scaler, shape=(1, 1), buffer_factor=1)
        red_dfb = ttl.make_dataflow_buffer_like(scaler, shape=(1, 1), buffer_factor=2)
        acc_dfb = ttl.make_dataflow_buffer_like(scaler, shape=(1, 1), buffer_factor=2)
        out_dfb = ttl.make_dataflow_buffer_like(out, shape=(1, 1), buffer_factor=2)

        @ttl.compute()
        def compute():
            with sc_dfb.wait() as sc:
                with inp_dfb.wait() as iv:
                    with red_dfb.reserve() as r:
                        r.store(ttl.math.reduce_sum(iv, sc, dims=[1]))
                with red_dfb.wait() as rv, acc_dfb.reserve() as a:
                    a.store(rv)
                for j in range(n_tiles - 1):
                    with inp_dfb.wait() as iv:
                        with red_dfb.reserve() as r:
                            r.store(ttl.math.reduce_sum(iv, sc, dims=[1]))
                    with red_dfb.wait() as rv, acc_dfb.wait() as prev, acc_dfb.reserve() as a:
                        a.store(prev + rv)
                with acc_dfb.wait() as total, out_dfb.reserve() as o:
                    o.store(total)

        @ttl.datamovement()
        def dm_read():
            with sc_dfb.reserve() as blk:
                tx = ttl.copy(scaler[0, 0], blk); tx.wait()
            for j in range(n_tiles):
                with inp_dfb.reserve() as blk:
                    tx = ttl.copy(inp[0, j], blk); tx.wait()

        @ttl.datamovement()
        def dm_write():
            with out_dfb.wait() as blk:
                tx = ttl.copy(blk, out[0, 0]); tx.wait()

    return reduce_acc


if __name__ == "__main__":
    device = ttnn.open_device(device_id=0)
    torch.manual_seed(42)

    print("=== Test 3: reduce_sum(dims=[1]) tile accumulation ===")

    # Single row of D_MODEL values
    data = torch.randn(TILE, D_MODEL, dtype=torch.bfloat16) * 0.3

    # Reference: sum across dim=1 for each of 32 rows
    ref_sum = data.float().sum(dim=1)

    inp_tt = to_tt(data, device)
    scaler = to_tt_l1(torch.ones(TILE, TILE, dtype=torch.bfloat16), device)
    out_tt = to_tt(torch.zeros(TILE, TILE, dtype=torch.bfloat16), device)

    # Test with all 32 tiles
    kernel = make_reduce_acc_kernel(D_TILES)
    print("Running full reduce (%d tiles)..." % D_TILES)
    kernel(inp_tt, scaler, out_tt)

    result = ttnn.to_torch(out_tt).float()
    # After reduce_sum(dims=[1]), result should have row sums in column 0
    print("Result col 0 [0:5]:", result[0:5, 0].tolist())
    print("Ref sum [0:5]:     ", ref_sum[0:5].tolist())
    err = (result[:, 0] - ref_sum).abs()
    print("Max error: %.6f" % err.max().item())

    # Now test with 4 tiles (like one pipe core) and check partial sum
    partial_ref = data[:, :128].float().sum(dim=1)
    out_tt2 = to_tt(torch.zeros(TILE, TILE, dtype=torch.bfloat16), device)
    kernel4 = make_reduce_acc_kernel(4)
    print("\nRunning partial reduce (4 tiles)...")
    kernel4(inp_tt, scaler, out_tt2)
    result4 = ttnn.to_torch(out_tt2).float()
    print("Partial col 0 [0:5]:", result4[0:5, 0].tolist())
    print("Ref partial [0:5]:  ", partial_ref[0:5].tolist())
    err4 = (result4[:, 0] - partial_ref).abs()
    print("Max error: %.6f" % err4.max().item())

    print("\nPASS" if err.max().item() < 1.0 and err4.max().item() < 1.0 else "FAIL")
    ttnn.close_device(device)
