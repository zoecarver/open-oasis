"""Compare our RoPE implementation against the reference open-oasis code."""
import torch
import math
from einops import rearrange, repeat
from safetensors import safe_open
import os

# ============================================================
# Reference implementation (from rotary_embedding_torch.py)
# ============================================================

def ref_rotate_half(x):
    x = rearrange(x, "... (d r) -> ... d r", r=2)
    x1, x2 = x.unbind(dim=-1)
    x = torch.stack((-x2, x1), dim=-1)
    return rearrange(x, "... d r -> ... (d r)")

def ref_apply_rotary_emb(freqs, t, start_index=0, scale=1.0, seq_dim=-2):
    dtype = t.dtype
    if t.ndim == 3:
        seq_len = t.shape[seq_dim]
        freqs = freqs[-seq_len:]
    rot_dim = freqs.shape[-1]
    end_index = start_index + rot_dim
    t_left = t[..., :start_index]
    t_middle = t[..., start_index:end_index]
    t_right = t[..., end_index:]
    t_transformed = (t_middle * freqs.cos() * scale) + (ref_rotate_half(t_middle) * freqs.sin() * scale)
    out = torch.cat((t_left, t_transformed, t_right), dim=-1)
    return out.type(dtype)

def ref_get_axial_freqs(freqs_param, *dims):
    from torch import broadcast_tensors
    all_freqs = []
    for ind, dim in enumerate(dims):
        pos = torch.linspace(-1, 1, steps=dim)
        seq_freqs = torch.einsum("..., f -> ... f", pos.float(), freqs_param.float())
        seq_freqs = repeat(seq_freqs, "... n -> ... (n r)", r=2)
        all_axis = [None] * len(dims)
        all_axis[ind] = slice(None)
        new_axis_slice = (Ellipsis, *all_axis, slice(None))
        all_freqs.append(seq_freqs[new_axis_slice])
    all_freqs = broadcast_tensors(*all_freqs)
    return torch.cat(all_freqs, dim=-1)

def ref_rotate_queries_or_keys(t, freqs_param):
    seq_dim = -2
    seq_len = t.shape[seq_dim]
    seq = torch.arange(seq_len, dtype=t.dtype)
    seq_freqs = torch.einsum("..., f -> ... f", seq.float(), freqs_param.float())
    seq_freqs = repeat(seq_freqs, "... n -> ... (n r)", r=2)
    return ref_apply_rotary_emb(seq_freqs, t, scale=1.0, seq_dim=seq_dim)

# ============================================================
# Our implementation (fixed)
# ============================================================

FRAME_H = 9
FRAME_W = 16
N_PATCHES = FRAME_H * FRAME_W  # 144
N_HEADS = 16
D_HEAD = 64

def our_rotate_half(x):
    x = rearrange(x, "... (d r) -> ... d r", r=2)
    x1, x2 = x.unbind(dim=-1)
    x = torch.stack((-x2, x1), dim=-1)
    return rearrange(x, "... d r -> ... (d r)")

def our_apply_rotary_emb(freqs, t):
    rot_dim = freqs.shape[-1]
    t_rot = t[..., :rot_dim]
    t_pass = t[..., rot_dim:]
    t_rot = (t_rot * freqs.cos()) + (our_rotate_half(t_rot) * freqs.sin())
    return torch.cat((t_rot, t_pass), dim=-1)

def our_precompute_spatial_rope_freqs(freqs_param):
    h_pos = torch.linspace(-1, 1, steps=FRAME_H)
    w_pos = torch.linspace(-1, 1, steps=FRAME_W)
    # Interleave-repeat: [f0, f0, f1, f1, ...] matching reference
    h_freqs = torch.einsum("i, f -> i f", h_pos, freqs_param)
    h_freqs = repeat(h_freqs, "... n -> ... (n r)", r=2)  # (9, 32)
    w_freqs = torch.einsum("i, f -> i f", w_pos, freqs_param)
    w_freqs = repeat(w_freqs, "... n -> ... (n r)", r=2)  # (16, 32)
    h_broad = h_freqs[:, None, :].expand(FRAME_H, FRAME_W, -1)
    w_broad = w_freqs[None, :, :].expand(FRAME_H, FRAME_W, -1)
    axial_freqs = torch.cat([h_broad, w_broad], dim=-1)  # (9, 16, 64)
    return axial_freqs.reshape(N_PATCHES, -1)  # (144, 64)

# ============================================================
# Test
# ============================================================

if __name__ == "__main__":
    blob_dir = "/root/.cache/huggingface/hub/models--Etched--oasis-500m/blobs/"
    files = sorted(os.listdir(blob_dir))
    dit_path = blob_dir + files[1]

    with safe_open(dit_path, framework="pt") as st:
        s_freqs = st.get_tensor("blocks.0.s_attn.rotary_emb.freqs").float()
        t_freqs = st.get_tensor("blocks.0.t_attn.rotary_emb.freqs").float()

    print("Spatial freqs:", s_freqs.shape, s_freqs[:5])
    print("Temporal freqs:", t_freqs.shape, t_freqs[:5])

    # ---- Test 1: Axial freq computation ----
    print("\n=== Test 1: Axial freq computation ===")
    ref_axial = ref_get_axial_freqs(s_freqs, FRAME_H, FRAME_W)
    ref_axial_flat = ref_axial.reshape(N_PATCHES, -1)
    our_axial = our_precompute_spatial_rope_freqs(s_freqs)

    print("Reference shape:", ref_axial_flat.shape)
    print("Our shape:", our_axial.shape)
    max_diff = (ref_axial_flat - our_axial).abs().max().item()
    print("Max diff:", max_diff)
    print("Ref[0,:8]:", ref_axial_flat[0, :8].tolist())
    print("Our[0,:8]:", our_axial[0, :8].tolist())

    if max_diff < 1e-5:
        print("PASS: Axial freqs match")
    else:
        print("FAIL: Axial freqs differ by", max_diff)

    # ---- Test 2: Full spatial RoPE ----
    print("\n=== Test 2: Full spatial RoPE ===")
    torch.manual_seed(42)
    q_5d = torch.randn(1, N_HEADS, FRAME_H, FRAME_W, D_HEAD)

    # Reference path
    freqs_ref = ref_get_axial_freqs(s_freqs, FRAME_H, FRAME_W)
    q_ref_rotated = ref_apply_rotary_emb(freqs_ref, q_5d)
    q_ref_flat = rearrange(q_ref_rotated, "b h H W d -> b h (H W) d")

    # Our path: start from flattened (N_PATCHES, D_MODEL)
    q_flat = q_5d[0].permute(1, 2, 0, 3).reshape(N_PATCHES, N_HEADS * D_HEAD)  # (144, 1024)
    q_mh = q_flat.view(N_PATCHES, N_HEADS, D_HEAD)
    our_freqs = our_precompute_spatial_rope_freqs(s_freqs).unsqueeze(1)  # (144, 1, 64)
    q_our_rotated = our_apply_rotary_emb(our_freqs, q_mh)
    q_our_final = q_our_rotated.permute(1, 0, 2).unsqueeze(0)  # (1, 16, 144, 64)

    max_diff = (q_ref_flat - q_our_final).abs().max().item()
    mean_diff = (q_ref_flat - q_our_final).abs().mean().item()
    print("Max diff:", max_diff)
    print("Mean diff:", mean_diff)
    if max_diff < 1e-5:
        print("PASS: Spatial RoPE matches reference")
    else:
        print("FAIL: Spatial RoPE differs")
        for pos in [0, 72, 143]:
            print("  pos=%d ref[:6]=%s" % (pos, q_ref_flat[0, 0, pos, :6].tolist()))
            print("  pos=%d our[:6]=%s" % (pos, q_our_final[0, 0, pos, :6].tolist()))

    # ---- Test 3: Temporal RoPE at T=1 ----
    print("\n=== Test 3: Temporal RoPE (T=1) ===")
    q_temp = torch.randn(N_PATCHES, N_HEADS, 1, D_HEAD)
    q_temp_ref = ref_rotate_queries_or_keys(q_temp, t_freqs)

    # Our path
    q_temp_flat = q_temp.squeeze(2)  # (144, 16, 64)
    pos0_freqs = torch.einsum("..., f -> ... f", torch.zeros(1), t_freqs)
    pos0_freqs = repeat(pos0_freqs, "... n -> ... (n r)", r=2).unsqueeze(1)  # (1, 1, 64)
    q_temp_our = our_apply_rotary_emb(pos0_freqs, q_temp_flat)

    q_temp_ref_flat = q_temp_ref.squeeze(2)
    max_diff_t = (q_temp_ref_flat - q_temp_our).abs().max().item()
    print("Max diff:", max_diff_t)
    if max_diff_t < 1e-5:
        print("PASS: Temporal RoPE matches reference")
    else:
        print("FAIL: Temporal RoPE differs")

    print("\nAll tests done.")
