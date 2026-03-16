"""Simple test: just matmul + bias, n_repeat=1. No expand."""
import torch
import ttnn
import sys
sys.path.insert(0, "/tmp")
from adaln_matmul_expand import make_adaln_matmul_expand_kernel

TILE = 32
D_MODEL = 1024
ADALN_DIM = 6 * D_MODEL
K_TILES = D_MODEL // TILE


def to_tt(t, device):
    return ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                           device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)

def to_tt_l1(t, device):
    return ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                           device=device, memory_config=ttnn.L1_MEMORY_CONFIG)


if __name__ == "__main__":
    device = ttnn.open_device(device_id=0)
    torch.manual_seed(42)

    N_REPEAT = 5
    print("=== adaLN matmul+bias+expand (n_repeat=%d) ===" % N_REPEAT)

    silu_cond = torch.randn(TILE, D_MODEL, dtype=torch.bfloat16) * 0.1
    adaln_w = torch.randn(D_MODEL, ADALN_DIM, dtype=torch.bfloat16) * 0.02
    adaln_b = torch.randn(TILE, ADALN_DIM, dtype=torch.bfloat16) * 0.01

    ref_1row = (silu_cond.float() @ adaln_w.float() + adaln_b.float())
    ref = ref_1row.repeat(N_REPEAT, 1)

    silu_tt = to_tt_l1(silu_cond, device)
    w_tt = to_tt(adaln_w, device)
    b_tt = to_tt(adaln_b, device)
    out_tt = to_tt(torch.zeros(N_REPEAT * TILE, ADALN_DIM, dtype=torch.bfloat16), device)

    kernel = make_adaln_matmul_expand_kernel(K_TILES, N_REPEAT)

    print("Running...")
    kernel(silu_tt, w_tt, b_tt, out_tt)

    result = ttnn.to_torch(out_tt).float()
    diff = (ref - result).abs()
    print("Max error: %.4f, Mean: %.4f" % (diff.max().item(), diff.mean().item()))
    print("Result[0, 0:5]:", result[0, 0:5].tolist())
    print("Ref[0, 0:5]:", ref[0, 0:5].tolist())
    expand_diff = (result[0, :] - result[32, :]).abs().max().item()
    print("Expand consistency (row 0 vs row 32): %.6f" % expand_diff)
    print("PASS" if diff.max().item() < 5.0 else "FAIL")

    import time
    N_ITER = 50
    ttnn.synchronize_device(device)
    t0 = time.time()
    for _ in range(N_ITER):
        kernel(silu_tt, w_tt, b_tt, out_tt)
    ttnn.synchronize_device(device)
    elapsed = (time.time() - t0) / N_ITER * 1000
    print("Kernel time: %.2f ms" % elapsed)

    ttnn.close_device(device)
