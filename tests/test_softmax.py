"""Test row-wise softmax on a (5, 5) tile matrix."""
import torch
import torch.nn.functional as F
import ttnn
import ttl

TILE = 32
SEQ_TILES = 5
N_PATCH_PAD = SEQ_TILES * TILE  # 160


@ttl.kernel(grid=(1, 1))
def softmax_kernel(x, scaler, out):
    x_dfb = ttl.make_dataflow_buffer_like(x, shape=(SEQ_TILES, SEQ_TILES), buffer_factor=1)
    sc_dfb = ttl.make_dataflow_buffer_like(scaler, shape=(1, 1), buffer_factor=1)
    a_dfb = ttl.make_dataflow_buffer_like(x, shape=(SEQ_TILES, SEQ_TILES), buffer_factor=2)
    row_dfb = ttl.make_dataflow_buffer_like(scaler, shape=(SEQ_TILES, 1), buffer_factor=2)
    row_bc_dfb = ttl.make_dataflow_buffer_like(x, shape=(SEQ_TILES, SEQ_TILES), buffer_factor=2)
    out_dfb = ttl.make_dataflow_buffer_like(out, shape=(SEQ_TILES, SEQ_TILES), buffer_factor=2)

    @ttl.compute()
    def compute():
        with x_dfb.wait() as xv, sc_dfb.wait() as sc:
            # Row max
            with row_dfb.reserve() as mx:
                mx.store(ttl.math.reduce_max(xv, sc, dims=[1]))
            with row_dfb.wait() as mxv, row_bc_dfb.reserve() as mxb:
                mxb.store(ttl.math.broadcast(mxv, dims=[1]))
            # exp(x - max)
            with row_bc_dfb.wait() as mxbv:
                with a_dfb.reserve() as ex:
                    ex.store(ttl.math.exp(xv - mxbv))
            # Row sum
            with a_dfb.wait() as exv:
                with row_dfb.reserve() as sm:
                    sm.store(ttl.math.reduce_sum(exv, sc, dims=[1]))
                with row_dfb.wait() as smv, row_bc_dfb.reserve() as smb:
                    smb.store(ttl.math.broadcast(smv, dims=[1]))
                # softmax = exp / sum
                with row_bc_dfb.wait() as smbv, out_dfb.reserve() as o:
                    o.store(exv * ttl.math.recip(smbv))

    @ttl.datamovement()
    def dm_read():
        with x_dfb.reserve() as blk:
            tx = ttl.copy(x[0:SEQ_TILES, 0:SEQ_TILES], blk); tx.wait()
        with sc_dfb.reserve() as blk:
            tx = ttl.copy(scaler[0, 0], blk); tx.wait()

    @ttl.datamovement()
    def dm_write():
        with out_dfb.wait() as blk:
            tx = ttl.copy(blk, out[0:SEQ_TILES, 0:SEQ_TILES]); tx.wait()


def to_tt(t, device):
    return ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                           device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)

def to_tt_l1(t, device):
    return ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                           device=device, memory_config=ttnn.L1_MEMORY_CONFIG)


if __name__ == "__main__":
    device = ttnn.open_device(device_id=0)
    torch.manual_seed(42)

    x = torch.randn(N_PATCH_PAD, N_PATCH_PAD, dtype=torch.float32) * 2.0
    ref = F.softmax(x, dim=-1)
    print("Ref range: [%.4f, %.4f], row sums: %.4f" % (
        ref.min().item(), ref.max().item(), ref.sum(dim=-1).mean().item()))

    x_tt = to_tt(x.to(torch.bfloat16), device)
    out_tt = to_tt(torch.zeros(N_PATCH_PAD, N_PATCH_PAD, dtype=torch.bfloat16), device)
    scaler = to_tt_l1(torch.ones(TILE, TILE, dtype=torch.bfloat16), device)

    softmax_kernel(x_tt, scaler, out_tt)

    result = ttnn.to_torch(out_tt).float()
    diff = (ref - result).abs()
    print("Max err: %.4f, Mean: %.4f" % (diff.max().item(), diff.mean().item()))
    print("Result range: [%.4f, %.4f], row sums: %.4f" % (
        result.min().item(), result.max().item(), result.sum(dim=-1).mean().item()))
    print("PASS" if diff.max().item() < 1.0 else "FAIL")

    ttnn.close_device(device)
