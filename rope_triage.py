"""
RoPE triage: compare CPU VAE rotary_freqs to our precomputed tables.
Check if the tables match, and if the RoPE application produces the same result.
"""
import sys, types
import torch
import torch.nn as nn

# Stub timm
def _to_2tuple(x):
    return tuple(x) if isinstance(x, (tuple, list)) else (x, x)
class _Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None,
                 act_layer=nn.GELU, drop=0.):
        super().__init__()
        hidden_features = hidden_features or in_features
        out_features = out_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
    def forward(self, x):
        return self.drop(self.fc2(self.drop(self.act(self.fc1(x)))))
def _make_stub(name, parent=None):
    m = types.ModuleType(name)
    sys.modules[name] = m
    if parent is not None:
        setattr(parent, name.split('.')[-1], m)
    return m
_timm = _make_stub('timm')
_timm_models = _make_stub('timm.models', _timm)
_timm_models_vit = _make_stub('timm.models.vision_transformer', _timm_models)
_timm_layers = _make_stub('timm.layers', _timm)
_timm_layers_helpers = _make_stub('timm.layers.helpers', _timm_layers)
_timm_models_vit.Mlp = _Mlp
_timm_layers_helpers.to_2tuple = _to_2tuple

import os
from safetensors import safe_open
from einops import rearrange
from rotary_embedding_torch import RotaryEmbedding, apply_rotary_emb

VAE_SEQ_H = 18
VAE_SEQ_W = 32
VAE_SEQ_LEN = VAE_SEQ_H * VAE_SEQ_W  # 576
VAE_DEC_DIM = 1024
VAE_DEC_HEADS = 16
VAE_D_HEAD = VAE_DEC_DIM // VAE_DEC_HEADS  # 64
VAE_ROPE_DIM = VAE_D_HEAD // 2  # 32

def find_vae_weights():
    blob_dir = "/root/.cache/huggingface/hub/models--Etched--oasis-500m/blobs/"
    for f in sorted(os.listdir(blob_dir)):
        path = blob_dir + f
        with safe_open(path, framework="pt") as st:
            if any(k.startswith("encoder.") for k in st.keys()):
                return path
    raise FileNotFoundError("VAE weights not found")

def load_vae_cpu():
    sys.path.insert(0, "/tmp")
    from vae import VAE_models
    from safetensors.torch import load_model
    vae = VAE_models["vit-l-20-shallow-encoder"]()
    load_model(vae, find_vae_weights())
    return vae.eval()

# ============================================================
# Our precomputed RoPE tables (from oasis_inference.py)
# ============================================================
def our_precompute_vae_rope():
    n_freqs = VAE_D_HEAD // 4 // 2  # 8
    pixel_freqs = torch.linspace(1.0, (VAE_SEQ_H * VAE_SEQ_W) / 2, n_freqs)

    h_pos = torch.arange(VAE_SEQ_H, dtype=torch.float32)
    w_pos = torch.arange(VAE_SEQ_W, dtype=torch.float32)

    h_freqs = torch.einsum("i, f -> i f", h_pos, pixel_freqs)
    h_freqs = h_freqs.repeat_interleave(2, dim=-1)  # (18, 16)
    w_freqs = torch.einsum("i, f -> i f", w_pos, pixel_freqs)
    w_freqs = w_freqs.repeat_interleave(2, dim=-1)  # (32, 16)

    h_broad = h_freqs[:, None, :].expand(VAE_SEQ_H, VAE_SEQ_W, -1)
    w_broad = w_freqs[None, :, :].expand(VAE_SEQ_H, VAE_SEQ_W, -1)
    freqs_2d = torch.cat([h_broad, w_broad], dim=-1)  # (18, 32, 32)
    freqs_flat = freqs_2d.reshape(VAE_SEQ_LEN, VAE_ROPE_DIM)  # (576, 32)

    cos_per_head = torch.ones(VAE_SEQ_LEN, VAE_D_HEAD)
    sin_per_head = torch.zeros(VAE_SEQ_LEN, VAE_D_HEAD)
    cos_per_head[:, :VAE_ROPE_DIM] = freqs_flat.cos()
    sin_per_head[:, :VAE_ROPE_DIM] = freqs_flat.sin()

    sign = torch.ones(VAE_D_HEAD)
    sign[:VAE_ROPE_DIM:2] = -1
    sin_perm_per_head = sin_per_head * sign.unsqueeze(0)

    return cos_per_head, sin_perm_per_head, freqs_flat

if __name__ == "__main__":
    print("Loading VAE...")
    vae = load_vae_cpu()

    # Get CPU rotary_freqs from actual model
    cpu_freqs = vae.decoder[0].attn.rotary_freqs  # from RotaryEmbedding
    print("\n=== CPU rotary_freqs ===")
    print("  shape:", cpu_freqs.shape)
    print("  dtype:", cpu_freqs.dtype)
    print("  range: [%.6f, %.6f]" % (cpu_freqs.min().item(), cpu_freqs.max().item()))

    # Reconstruct what RotaryEmbedding produces
    print("\n=== Reconstructing RotaryEmbedding ===")
    rot_emb = RotaryEmbedding(
        dim=VAE_D_HEAD // 4,  # 16
        freqs_for="pixel",
        max_freq=VAE_SEQ_H * VAE_SEQ_W,  # 576
    )
    reconstructed_freqs = rot_emb.get_axial_freqs(VAE_SEQ_H, VAE_SEQ_W)
    print("  reconstructed shape:", reconstructed_freqs.shape)
    print("  matches cpu_freqs:", torch.allclose(cpu_freqs, reconstructed_freqs, atol=1e-6))

    # Now compare to our precomputed tables
    our_cos, our_sin_perm, our_freqs_flat = our_precompute_vae_rope()
    print("\n=== Our precomputed tables ===")
    print("  our_freqs_flat shape:", our_freqs_flat.shape)  # (576, 32)
    print("  our_cos shape:", our_cos.shape)  # (576, 64)

    # The CPU freqs are (H, W, d_head) or similar - let me inspect
    print("\n=== CPU freqs structure ===")
    print("  cpu_freqs shape:", cpu_freqs.shape)
    # Flatten to (576, rot_dim)
    if cpu_freqs.dim() == 3:
        cpu_freqs_flat = cpu_freqs.reshape(VAE_SEQ_LEN, -1)
    elif cpu_freqs.dim() == 2:
        cpu_freqs_flat = cpu_freqs
    else:
        cpu_freqs_flat = cpu_freqs.reshape(VAE_SEQ_LEN, -1)
    print("  cpu_freqs_flat shape:", cpu_freqs_flat.shape)
    print("  cpu_freqs_flat[:3, :8]:")
    print(cpu_freqs_flat[:3, :8])
    print("  our_freqs_flat[:3, :8]:")
    print(our_freqs_flat[:3, :8])

    # Compare cos/sin from CPU freqs vs our tables
    cpu_cos = cpu_freqs_flat.cos()
    cpu_sin = cpu_freqs_flat.sin()
    print("\n=== cos comparison (first rot_dim=%d dims) ===" % cpu_freqs_flat.shape[1])
    rot_dim = cpu_freqs_flat.shape[1]
    our_cos_trimmed = our_cos[:, :rot_dim]
    diff_cos = (our_cos_trimmed - cpu_cos).abs()
    print("  max_diff:", diff_cos.max().item())
    print("  mean_diff:", diff_cos.mean().item())

    # Check sin with the swap/sign trick
    # CPU does: rotate_half(q) * sin
    #   rotate_half pairs (x0,x1) -> (-x1, x0)
    # Our trick: q_swap * sin_perm where q_swap swaps adjacent columns
    #   and sin_perm has alternating signs on rotated dims
    # So: our sin_perm = sin * [-1, +1, -1, +1, ...]
    # While CPU sin is just sin (sign is in rotate_half)
    #
    # The math:
    #   CPU: q_rot = q * cos + rotate_half(q) * sin
    #     = q0*cos0 + (-q1)*sin0, q1*cos1 + q0*sin1, ...
    #   Ours: q_rot = q * cos + q_swap * sin_perm
    #     = q0*cos0 + q1*sin_perm0, q1*cos1 + q0*sin_perm1, ...
    #   For these to match:
    #     sin_perm0 = -sin0 (because CPU has -q1*sin0, ours has q1*sin_perm0)
    #     sin_perm1 = +sin1 (because CPU has +q0*sin1, ours has q0*sin_perm1)
    our_sin_trimmed = our_sin_perm[:, :rot_dim]
    # Expected: sin_perm[even] = -sin, sin_perm[odd] = +sin
    expected_sin_perm = cpu_sin.clone()
    expected_sin_perm[:, 0::2] = -cpu_sin[:, 0::2]
    # expected_sin_perm[:, 1::2] = +cpu_sin[:, 1::2]  # already correct

    diff_sin = (our_sin_trimmed - expected_sin_perm).abs()
    print("\n=== sin_perm comparison ===")
    print("  max_diff:", diff_sin.max().item())
    print("  mean_diff:", diff_sin.mean().item())

    if diff_cos.max().item() > 0.01 or diff_sin.max().item() > 0.01:
        print("\n  !!! MISMATCH DETECTED !!!")
        # Find where they differ most
        print("\n  cos diffs per dim (first 8 dims, first 5 rows):")
        for d in range(min(8, rot_dim)):
            print("    dim %d: max=%.6f  our=%.6f cpu=%.6f (row0)" % (
                d, diff_cos[:, d].max().item(), our_cos_trimmed[0, d].item(), cpu_cos[0, d].item()))
        print("\n  sin_perm diffs per dim (first 8 dims, first 5 rows):")
        for d in range(min(8, rot_dim)):
            print("    dim %d: max=%.6f  our=%.6f expected=%.6f (row0)" % (
                d, diff_sin[:, d].max().item(), our_sin_trimmed[0, d].item(), expected_sin_perm[0, d].item()))

        # Check the raw frequency values
        print("\n  === Raw frequency comparison ===")
        print("  cpu_freqs_flat[0, :8]:", cpu_freqs_flat[0, :8].tolist())
        print("  our_freqs_flat[0, :8]:", our_freqs_flat[0, :8].tolist())
        print("  cpu_freqs_flat[1, :8]:", cpu_freqs_flat[1, :8].tolist())
        print("  our_freqs_flat[1, :8]:", our_freqs_flat[1, :8].tolist())
    else:
        print("\n  Tables match!")

    # Now test the actual RoPE application on a known input
    print("\n=== RoPE application test ===")
    torch.manual_seed(42)
    q_test = torch.randn(VAE_SEQ_LEN, VAE_DEC_DIM)  # (576, 1024)

    # CPU path: reshape to (h, H, W, d), apply_rotary_emb, reshape back
    q_cpu = rearrange(q_test, "(H W) (h d) -> h H W d",
                      H=VAE_SEQ_H, W=VAE_SEQ_W, h=VAE_DEC_HEADS)
    q_cpu_roped = apply_rotary_emb(cpu_freqs, q_cpu)
    q_cpu_flat = rearrange(q_cpu_roped, "h H W d -> (H W) (h d)")

    # Our path: q * cos + q_swap * sin_perm
    q_swap = q_test.clone()
    q_swap_view = q_swap.reshape(VAE_SEQ_LEN, VAE_DEC_HEADS, VAE_D_HEAD)
    for h in range(VAE_DEC_HEADS):
        head = q_swap_view[:, h, :]
        temp = head.clone()
        head[:, 0::2] = temp[:, 1::2]
        head[:, 1::2] = temp[:, 0::2]
    q_swap = q_swap_view.reshape(VAE_SEQ_LEN, VAE_DEC_DIM)

    cos_full = our_cos.repeat(1, VAE_DEC_HEADS)  # (576, 1024)
    sin_full = our_sin_perm.repeat(1, VAE_DEC_HEADS)  # (576, 1024)
    q_our_roped = q_test * cos_full + q_swap * sin_full

    diff = (q_our_roped - q_cpu_flat).abs()
    print("  max_err:", diff.max().item())
    print("  mean_err:", diff.mean().item())
    if diff.max().item() > 0.01:
        print("  !!! RoPE APPLICATION MISMATCH !!!")
        # Find the worst position
        worst_idx = diff.argmax()
        worst_row = worst_idx // VAE_DEC_DIM
        worst_col = worst_idx % VAE_DEC_DIM
        worst_head = worst_col // VAE_D_HEAD
        worst_dim = worst_col % VAE_D_HEAD
        print("  Worst at row=%d head=%d dim=%d: our=%.6f cpu=%.6f" % (
            worst_row, worst_head, worst_dim,
            q_our_roped[worst_row, worst_col].item(),
            q_cpu_flat[worst_row, worst_col].item()))
    else:
        print("  RoPE application matches!")

    print("\nDone!")
