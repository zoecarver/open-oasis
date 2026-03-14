"""Minimal SDPA test: just Q @ K^T to isolate issues."""
import torch
import torch.nn.functional as F
import ttnn
import ttl
import time

TILE = 32
N_PATCH_PAD = 160
D_HEAD = 64
N_HEADS = 1  # Test with 1 head first
SEQ_TILES = N_PATCH_PAD // TILE  # 5
HEAD_TILES = D_HEAD // TILE  # 2
SCALE = 1.0 / (D_HEAD ** 0.5)


@ttl.kernel(grid=(1, 1))
def qk_matmul_test(Q, K, scaler, out):
    """Just compute softmax(Q @ K^T * scale) @ V = out for one head."""
    q_dfb = ttl.make_dataflow_buffer_like(Q, shape=(SEQ_TILES, HEAD_TILES), buffer_factor=1)
    k_dfb = ttl.make_dataflow_buffer_like(K, shape=(SEQ_TILES, HEAD_TILES), buffer_factor=1)
    sc_dfb = ttl.make_dataflow_buffer_like(scaler, shape=(1, 1), buffer_factor=1)

    kt_dfb = ttl.make_dataflow_buffer_like(K, shape=(HEAD_TILES, SEQ_TILES), buffer_factor=2)
    qk_dfb = ttl.make_dataflow_buffer_like(out, shape=(SEQ_TILES, SEQ_TILES), buffer_factor=2)
    out_dfb = ttl.make_dataflow_buffer_like(out, shape=(SEQ_TILES, SEQ_TILES), buffer_factor=2)

    @ttl.compute()
    def compute():
        with q_dfb.wait() as qv, k_dfb.wait() as kv, sc_dfb.wait() as sc:
            with kt_dfb.reserve() as kt:
                kt.store(ttl.transpose(kv))
            with kt_dfb.wait() as ktv, qk_dfb.reserve() as qk:
                qk.store(qv @ ktv)
            with qk_dfb.wait() as qkv, out_dfb.reserve() as o:
                o.store(qkv * ttl.math.fill(qkv, 0.125))

    @ttl.datamovement()
    def dm_read():
        with q_dfb.reserve() as blk:
            tx = ttl.copy(Q[0:SEQ_TILES, 0:HEAD_TILES], blk); tx.wait()
        with k_dfb.reserve() as blk:
            tx = ttl.copy(K[0:SEQ_TILES, 0:HEAD_TILES], blk); tx.wait()
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

    print("Test 1: Q @ K^T * scale (single head)")
    q = torch.randn(N_PATCH_PAD, D_HEAD, dtype=torch.float32) * 0.3
    k = torch.randn(N_PATCH_PAD, D_HEAD, dtype=torch.float32) * 0.3

    ref_qk = (q.float() @ k.float().T) * SCALE
    print("  Ref QK range: [%.4f, %.4f]" % (ref_qk.min().item(), ref_qk.max().item()))

    q_tt = to_tt(q.to(torch.bfloat16), device)
    k_tt = to_tt(k.to(torch.bfloat16), device)
    out_tt = to_tt(torch.zeros(N_PATCH_PAD, N_PATCH_PAD, dtype=torch.bfloat16), device)
    scaler = to_tt_l1(torch.ones(TILE, TILE, dtype=torch.bfloat16), device)

    qk_matmul_test(q_tt, k_tt, scaler, out_tt)

    result = ttnn.to_torch(out_tt).float()
    diff = (ref_qk - result).abs()
    print("  Max err: %.4f, Mean: %.4f" % (diff.max().item(), diff.mean().item()))
    print("  Result range: [%.4f, %.4f]" % (result.min().item(), result.max().item()))
    print("  PASS" if diff.max().item() < 5.0 else "  FAIL")

    ttnn.close_device(device)
