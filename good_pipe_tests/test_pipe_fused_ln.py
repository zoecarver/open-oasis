"""Test pipe-fused gated_res + LN + adaLN kernel."""
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


def to_tt(t, device):
    return ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                           device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)

def to_tt_l1(t, device):
    return ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                           device=device, memory_config=ttnn.L1_MEMORY_CONFIG)


if __name__ == "__main__":
    default_size = ttnn.device.get_max_worker_l1_unreserved_size()
    # Fused kernel is ~85KB, need config buffer > 85KB
    device = ttnn.open_device(device_id=0, worker_l1_size=default_size - 90112)
    torch.manual_seed(42)

    print("=== Pipe Fused gated_res + LN + adaLN ===")
    print("D=%d, SEQ=%d, grid=(%d,%d) = %d cores" % (
        D_MODEL, N_PATCH_PAD, D_CORES, SEQ_TILES, D_CORES * SEQ_TILES))

    residual = torch.randn(N_PATCH_PAD, D_MODEL, dtype=torch.bfloat16) * 0.3
    x_in = torch.randn(N_PATCH_PAD, D_MODEL, dtype=torch.bfloat16) * 0.3
    gate = torch.randn(N_PATCH_PAD, D_MODEL, dtype=torch.bfloat16) * 0.1
    # adaln_packed: 6 * D_MODEL columns. shift=[3D:4D], scale=[4D:5D]
    adaln_packed = torch.randn(N_PATCH_PAD, 6 * D_MODEL, dtype=torch.bfloat16) * 0.1

    # Reference
    gated_res_ref = (residual.float() + x_in.float() * gate.float())
    mean = gated_res_ref.mean(dim=-1, keepdim=True)
    var = gated_res_ref.var(dim=-1, keepdim=True, unbiased=False)
    normed = (gated_res_ref - mean) / (var + 1e-6).sqrt()
    shift = adaln_packed[:, 3*D_MODEL:4*D_MODEL].float()
    scale = adaln_packed[:, 4*D_MODEL:5*D_MODEL].float()
    modulated_ref = normed * (scale + 1.0) + shift

    # Device tensors
    res_tt = to_tt(residual, device)
    x_tt = to_tt(x_in, device)
    gate_tt = to_tt(gate, device)
    adaln_tt = to_tt(adaln_packed, device)
    gro_tt = to_tt(torch.zeros(N_PATCH_PAD, D_MODEL, dtype=torch.bfloat16), device)
    out_tt = to_tt(torch.zeros(N_PATCH_PAD, D_MODEL, dtype=torch.bfloat16), device)
    scaler = to_tt_l1(torch.ones(TILE, TILE, dtype=torch.bfloat16), device)
    mean_scale = to_tt_l1(torch.full((TILE, TILE), 1.0 / D_MODEL, dtype=torch.bfloat16), device)

    kernel = make_pipe_fused_gated_res_ln_adaln(D_TILES, D_CORES, SEQ_TILES)

    print("Running...")
    kernel(res_tt, x_tt, gate_tt, scaler, mean_scale, adaln_tt, gro_tt, out_tt)

    # Check gated_res
    gro_result = ttnn.to_torch(gro_tt).float()
    gro_diff = (gated_res_ref - gro_result).abs()
    print("gated_res max_err: %.4f" % gro_diff.max().item())

    # Check modulated output
    result = ttnn.to_torch(out_tt).float()
    diff = (modulated_ref - result).abs()
    print("modulated max_err: %.4f, mean: %.4f" % (diff.max().item(), diff.mean().item()))
    print("Result[0, 0:5]:", result[0, 0:5].tolist())
    print("Ref[0, 0:5]:", modulated_ref[0, 0:5].tolist())
    print("PASS" if diff.max().item() < 1.0 else "FAIL")

    ttnn.close_device(device)
