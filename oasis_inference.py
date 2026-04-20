"""
Oasis-500M inference on Tenstorrent hardware using TT-Lang kernels.

DiT runs on TT device. VAE encode/decode runs on CPU (PyTorch).
Single-frame generation for initial testing (T=1).

Usage: run via run-test.sh --hw oasis_inference.py
"""
import sys
import types
import os
import math
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import ttnn
import ttl
from safetensors import safe_open
from safetensors.torch import load_model
from einops import rearrange
from einops import repeat as einops_repeat
from PIL import Image
import subprocess

# Stub out timm entirely -- vae.py/dit.py only need Mlp and to_2tuple.
# Trying to stub torchvision (timm's transitive dep) is endless whack-a-mole.

def _to_2tuple(x):
    if isinstance(x, (tuple, list)):
        return tuple(x)
    return (x, x)

class _Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None,
                 act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

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

# TT-Lang kernel imports (src/ directory when running from repo, flat /tmp when via run-test.sh)
_script_dir = os.path.dirname(os.path.abspath(__file__))
_src_dir = os.path.join(_script_dir, "src")
if os.path.isdir(_src_dir):
    sys.path.insert(0, _src_dir)

from linear import make_linear_kernel
from silu import silu_kernel
from gated_residual import gated_residual_kernel
from adaln_modulate import adaln_modulate_kernel
from layernorm import make_layernorm_kernel
from vae_rope import make_vae_rope_kernel
from adaln_matmul_expand import make_adaln_matmul_expand_kernel
from rope_layout_kernel import make_rope_layout_kernel, make_rope_temporal_kernel

# ============================================================
# Constants
# ============================================================

TILE = 32
D_MODEL = 1024
N_HEADS = 16
D_HEAD = 64  # D_MODEL // N_HEADS
D_MLP = 4096
N_BLOCKS = 16
PATCH_SIZE = 2
IN_CHANNELS = 16
INPUT_H = 18
INPUT_W = 32
FRAME_H = INPUT_H // PATCH_SIZE  # 9
FRAME_W = INPUT_W // PATCH_SIZE  # 16
N_PATCHES = FRAME_H * FRAME_W  # 144
N_PATCH_PAD = ((N_PATCHES + TILE - 1) // TILE) * TILE  # 160
FREQ_DIM = 256
EXT_COND_DIM = 25
OUT_DIM = PATCH_SIZE * PATCH_SIZE * IN_CHANNELS  # 64
SCALING_FACTOR = 0.07843137255

# RoPE dimensions (from model weights)
# Spatial: 16 freqs * 2 (interleave) * 2 (H+W axes) = 64
SPATIAL_ROPE_DIM = 64
# Temporal: 32 freqs * 2 (interleave) = 64
TEMPORAL_ROPE_DIM = 64

D_TILES = D_MODEL // TILE  # 32

# High-fidelity compute config for ttnn ops. Default ttnn matmul/SDPA uses LoFi and
# bf16 DST accumulation, which loses precision across large K (1024+) and long seq.
COMPUTE_HIFI = ttnn.WormholeComputeKernelConfig(
    math_fidelity=ttnn.MathFidelity.HiFi4,
    math_approx_mode=False,
    fp32_dest_acc_en=True,
    packer_l1_acc=True,
)

# Multi-chip tensor parallelism: set to 1 for single-chip, 2+ for TP
N_CHIPS = 1

# VAE decoder constants
VAE_DEC_DEPTH = 12
VAE_DEC_DIM = 1024
VAE_DEC_HEADS = 16
VAE_LATENT_DIM = 16
VAE_PATCH_SIZE_VAE = 20
VAE_PATCH_DIM = 3 * VAE_PATCH_SIZE_VAE ** 2  # 1200
VAE_SEQ_H = 18   # 360 / 20
VAE_SEQ_W = 32   # 640 / 20
VAE_SEQ_LEN = VAE_SEQ_H * VAE_SEQ_W  # 576
VAE_SEQ_TILES = VAE_SEQ_LEN // TILE  # 18
VAE_D_HEAD = VAE_DEC_DIM // VAE_DEC_HEADS  # 64
VAE_D_MLP = VAE_DEC_DIM * 4  # 4096
VAE_ROPE_DIM = VAE_D_HEAD // 2  # 32 (first 32 dims of each head rotated)

# Global mutable state (set during init)
_MESH_DEVICE = None              # mesh device reference for readback (set in main)
SPATIAL_ROPE_FREQS = None        # (144, 64) float, set during weight loading
TEMPORAL_ROPE_FREQS = None       # (max_t, 64) float, set during weight loading

# Profiling (set to True / a block name to enable per-phase timing)
PROFILE_BLOCKS = False
PROFILE_BLOCK_DEVICE = None      # e.g. "blocks.1.s" for per-phase profiling (breaks trace)

# ============================================================
# Kernel instantiation
# ============================================================

N_HEADS_TP = N_HEADS // N_CHIPS

linear_k32 = make_linear_kernel(D_TILES)
layernorm_d1024 = make_layernorm_kernel(D_TILES)

# In-progress kernels (defined but superseded by ttnn in hot path, kept for future optimization)
# See src/in_progress.py for definitions
# TODO: pipe-based LN produces incorrect values on repeated calls (many-to-one pipe caching bug)
# TODO: revisit fused_ln_adaln_d1024 (0.3ms vs 0.15ms ttnn, 2x slower)
# TODO: revisit fused_gated_res_ln_adaln_d1024 (0.8ms vs 0.15ms ttnn, 5.5x slower)
# TODO: mega_qkv_rope_sdpa fuses QKV+RoPE+SDPA but only uses 32 cores (ttnn.matmul uses 72)
# TODO: mega_post_attn_kernel fuses O proj+LN+FC1+GELU+FC2+residual but only uses 10 cores

# Fused adaLN params: matmul + bias + expand in one kernel (replaces ~10 ttnn ops)
adaln_matmul_expand_kernel = make_adaln_matmul_expand_kernel(D_TILES, N_PATCH_PAD // TILE)
# Fused RoPE + layout transform: eliminates 5 slices + 2 RoPE + 9 reshape/permute
rope_layout_spatial = make_rope_layout_kernel(N_PATCH_PAD // TILE, D_HEAD // TILE, N_HEADS_TP)
# Fused temporal RoPE: eliminates 5 slices + 2 RoPE kernels
rope_temporal = make_rope_temporal_kernel(D_HEAD // TILE, N_HEADS_TP)
# VAE decoder RoPE + layout (2D pixel-based, seq=576 tokens)
vae_rope_layout = make_rope_layout_kernel(VAE_SEQ_TILES, VAE_D_HEAD // TILE, VAE_DEC_HEADS)
# Fused VAE RoPE: reads qkv_full, writes Q/K/V in heads-first SDPA layout
# Uses (1, 2) tile DFBs (~160KB/core) vs vae_rope_layout's (18, 2) (~1.4MB/core OOM)
vae_rope_fused = make_vae_rope_kernel(VAE_DEC_HEADS, VAE_D_HEAD // TILE)


# ============================================================
# Host helpers
# ============================================================

def _mesh_kwargs(device):
    """Return mesh_mapper kwarg if device is a MeshDevice."""
    if isinstance(device, ttnn.MeshDevice):
        return {"mesh_mapper": ttnn.ReplicateTensorToMesh(device)}
    return {}

def to_tt(t, device):
    return ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                           device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG,
                           **_mesh_kwargs(device))

def to_tt_l1(t, device):
    return ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                           device=device, memory_config=ttnn.L1_MEMORY_CONFIG,
                           **_mesh_kwargs(device))

def shard_tt(t, device, dim):
    """Load tensor sharded across mesh devices along given dimension."""
    return ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                           device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG,
                           mesh_mapper=ttnn.ShardTensorToMesh(device, dim=dim))

def shard_tt_l1(t, device, dim):
    """Load tensor sharded across mesh devices along given dimension (L1)."""
    return ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                           device=device, memory_config=ttnn.L1_MEMORY_CONFIG,
                           mesh_mapper=ttnn.ShardTensorToMesh(device, dim=dim))

def zeros_tt(shape, device):
    return to_tt(torch.zeros(shape, dtype=torch.bfloat16), device)

def zeros_l1(shape, device):
    return to_tt_l1(torch.zeros(shape, dtype=torch.bfloat16), device)

def zeros_l1_f32(shape, device):
    return ttnn.from_torch(torch.zeros(shape, dtype=torch.float32),
                           dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT,
                           device=device, memory_config=ttnn.L1_MEMORY_CONFIG,
                           **_mesh_kwargs(device))

def readback_torch(t):
    """Read tensor from device/mesh to torch. For mesh, reads chip 0."""
    if _MESH_DEVICE is not None:
        return ttnn.to_torch(t, mesh_composer=ttnn.ConcatMeshToTensor(_MESH_DEVICE, dim=0))[:t.shape[0]]
    return ttnn.to_torch(t)

def expand_bias(bias_1d, seq_pad):
    return bias_1d.unsqueeze(0).expand(seq_pad, -1).contiguous().to(torch.bfloat16)

def interleave_qkv_for_tp(qkv_full_w, n_chips, d_model, d_head):
    """Rearrange QKV weight for per-head sharding across chips.
    Input: (d_model, 5*d_model) layout [Q|K|V|Q_swap|K_swap] each d_model cols.
    Output: (d_model, 5*d_model) rearranged so ShardTensorToMesh(dim=1)
    gives each chip n_heads/n_chips heads with [Q|K|V|Q_swap|K_swap] layout."""
    if n_chips == 1:
        return qkv_full_w
    heads_per_chip = d_model // d_head // n_chips
    cols_per_chip = heads_per_chip * d_head  # per section
    sections = torch.split(qkv_full_w, d_model, dim=1)  # 5 sections
    chip_parts = []
    for chip in range(n_chips):
        sc = chip * cols_per_chip
        ec = sc + cols_per_chip
        for section in sections:
            chip_parts.append(section[:, sc:ec])
    return torch.cat(chip_parts, dim=1)

def timestep_embedding(t, dim, max_period=10000):
    half = dim // 2
    freqs = torch.exp(-math.log(max_period) * torch.arange(0, half, dtype=torch.float32) / half)
    args = t[:, None].float() * freqs[None]
    return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)

def sigmoid_beta_schedule(timesteps, start=-3, end=3, tau=1, clamp_min=1e-5):
    steps = timesteps + 1
    t = torch.linspace(0, timesteps, steps, dtype=torch.float64) / timesteps
    v_start = torch.tensor(start / tau).sigmoid()
    v_end = torch.tensor(end / tau).sigmoid()
    alphas_cumprod = (-((t * (end - start) + start) / tau).sigmoid() + v_end) / (v_end - v_start)
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)

# ============================================================
# RoPE (Rotary Positional Embeddings) - applied on host
# ============================================================

def rotate_half(x):
    """Rotary embedding helper: rearrange pairs and negate."""
    x = rearrange(x, "... (d r) -> ... d r", r=2)
    x1, x2 = x.unbind(dim=-1)
    x = torch.stack((-x2, x1), dim=-1)
    return rearrange(x, "... d r -> ... (d r)")

def apply_rotary_emb(freqs, t):
    """Apply rotary embeddings to tensor t using precomputed freqs.
    freqs: (..., rot_dim)  t: (..., d_head)
    Only the first rot_dim dims of t are rotated; rest pass through."""
    rot_dim = freqs.shape[-1]
    t_rot = t[..., :rot_dim]
    t_pass = t[..., rot_dim:]
    t_rot = (t_rot * freqs.cos()) + (rotate_half(t_rot) * freqs.sin())
    return torch.cat((t_rot, t_pass), dim=-1)

def precompute_spatial_rope_freqs(freqs_param):
    """Precompute spatial axial RoPE freqs for H_GRID x W_GRID grid.
    freqs_param: learned freqs tensor shape (half_dim,) from weights.
    Returns: (N_PATCHES, rot_dim) float tensor.
    rot_dim = 2 axes * 2 * half_dim = 4 * half_dim (e.g. 4*16=64 for Oasis)."""

    h_pos = torch.linspace(-1, 1, steps=FRAME_H)
    w_pos = torch.linspace(-1, 1, steps=FRAME_W)

    # Per-axis: outer product then interleave-repeat (f0,f0,f1,f1,...)
    h_freqs = torch.einsum("i, f -> i f", h_pos, freqs_param)
    h_freqs = einops_repeat(h_freqs, "... n -> ... (n r)", r=2)  # (9, 32)
    w_freqs = torch.einsum("i, f -> i f", w_pos, freqs_param)
    w_freqs = einops_repeat(w_freqs, "... n -> ... (n r)", r=2)  # (16, 32)

    # Broadcast to (H, W, 32) each, then cat along last dim -> (H, W, 64)
    h_broad = h_freqs[:, None, :].expand(FRAME_H, FRAME_W, -1)
    w_broad = w_freqs[None, :, :].expand(FRAME_H, FRAME_W, -1)
    axial_freqs = torch.cat([h_broad, w_broad], dim=-1)

    return axial_freqs.reshape(N_PATCHES, -1)  # (144, 64)

def precompute_temporal_rope_freqs(freqs_param, max_t=32):
    """Precompute temporal RoPE freqs.
    freqs_param: learned freqs tensor shape (half_dim,) from weights.
    Returns: (max_t, rot_dim) float tensor. rot_dim = 2 * half_dim."""

    positions = torch.arange(max_t, dtype=torch.float32)
    freqs = torch.einsum("i, f -> i f", positions, freqs_param)
    return einops_repeat(freqs, "... n -> ... (n r)", r=2)  # (max_t, 64)

def swap_adjacent_columns(w):
    """Swap adjacent column pairs: col 2k <-> col 2k+1.
    Used to bake rotate_half permutation into weight matrices for device RoPE."""
    w_swap = w.clone()
    w_swap[:, 0::2] = w[:, 1::2]
    w_swap[:, 1::2] = w[:, 0::2]
    return w_swap

def swap_adjacent_elements(b):
    """1D version of swap_adjacent_columns for bias vectors."""
    b_swap = b.clone()
    b_swap[0::2] = b[1::2]
    b_swap[1::2] = b[0::2]
    return b_swap

def build_rope_device_tables(freqs_per_position, n_positions, n_patch_pad, n_heads, n_frames, tt_device):
    """Build cos/sin_perm device tables from per-position freq table.
    freqs_per_position: (n_positions, freq_dim) float tensor.
    Returns cos_tt, sin_perm_tt each (n_patch_pad * n_frames, n_heads * freq_dim) on device.
    """
    freq_dim = freqs_per_position.shape[-1]
    d_model = n_heads * freq_dim
    SEQ = n_patch_pad * n_frames

    cos_vals = freqs_per_position.cos()  # (n_positions, freq_dim)
    sin_vals = freqs_per_position.sin()
    sign = torch.ones(freq_dim)
    sign[0::2] = -1
    sin_perm_vals = sin_vals * sign.unsqueeze(0)

    cos_expanded = cos_vals.repeat(1, n_heads)  # (n_positions, d_model)
    sin_expanded = sin_perm_vals.repeat(1, n_heads)

    cos_full = torch.zeros(SEQ, d_model, dtype=torch.bfloat16)
    sin_full = torch.zeros(SEQ, d_model, dtype=torch.bfloat16)
    for t in range(n_frames):
        start = t * n_patch_pad
        src_start = t * n_patch_pad
        n = min(n_positions - src_start, n_patch_pad)
        cos_full[start:start + n] = cos_expanded[src_start:src_start + n].to(torch.bfloat16)
        sin_full[start:start + n] = sin_expanded[src_start:src_start + n].to(torch.bfloat16)

    # L1 for RoPE tables: read 16x per step by each spatial/temporal sub-block
    return to_tt_l1(cos_full, tt_device), to_tt_l1(sin_full, tt_device)

def build_spatial_rope_device_tables(spatial_freqs, n_frames, tt_device):
    """Build device-side cos/sin_perm tables for spatial RoPE.
    spatial_freqs: (N_PATCHES, 64) float.
    Returns cos_tt, sin_perm_tt each (SEQ, D_MODEL) on device.
    sin_perm bakes in the rotate_half sign: sin_perm[2k] = -sin, sin_perm[2k+1] = +sin.
    """
    SEQ = N_PATCH_PAD * n_frames
    cos_per_pos = spatial_freqs.cos()  # (N_PATCHES, 64)
    sin_per_pos = spatial_freqs.sin()  # (N_PATCHES, 64)

    # Bake rotate_half sign into sin: [-, +, -, +, ...] pattern
    sign = torch.ones(SPATIAL_ROPE_DIM)
    sign[0::2] = -1
    sin_perm_per_pos = sin_per_pos * sign.unsqueeze(0)  # (N_PATCHES, 64)

    # Repeat across TP heads: (N_PATCHES, 64) -> (N_PATCHES, N_HEADS_TP * D_HEAD)
    d_model_tp = N_HEADS_TP * D_HEAD
    cos_expanded = cos_per_pos.repeat(1, N_HEADS_TP)
    sin_expanded = sin_perm_per_pos.repeat(1, N_HEADS_TP)

    # Pad to N_PATCH_PAD rows, repeat for T frames
    cos_full = torch.zeros(SEQ, d_model_tp, dtype=torch.bfloat16)
    sin_full = torch.zeros(SEQ, d_model_tp, dtype=torch.bfloat16)
    for t in range(n_frames):
        start = t * N_PATCH_PAD
        cos_full[start:start + N_PATCHES] = cos_expanded.to(torch.bfloat16)
        sin_full[start:start + N_PATCHES] = sin_expanded.to(torch.bfloat16)

    # L1 for RoPE tables: read 16x per step by each spatial/temporal sub-block
    return to_tt_l1(cos_full, tt_device), to_tt_l1(sin_full, tt_device)

def precompute_vae_rope(vae, tt_device):
    """Extract 2D pixel-based RoPE tables from CPU VAE model.
    Uses the actual rotary_freqs buffer from RotaryEmbedding to match CPU exactly.
    Returns cos_tt, sin_perm_tt each (VAE_SEQ_LEN, VAE_DEC_DIM) on device."""
    # Extract the actual rotary_freqs from the model (same for all decoder blocks)
    cpu_freqs = vae.decoder[0].attn.rotary_freqs  # (H, W, rot_dim) = (18, 32, 32)
    freqs_flat = cpu_freqs.reshape(VAE_SEQ_LEN, -1)  # (576, 32)
    rot_dim = freqs_flat.shape[1]  # 32

    # Build per-head tables: first rot_dim dims rotated, rest identity
    cos_per_head = torch.ones(VAE_SEQ_LEN, VAE_D_HEAD)
    sin_per_head = torch.zeros(VAE_SEQ_LEN, VAE_D_HEAD)
    cos_per_head[:, :rot_dim] = freqs_flat.cos()
    sin_per_head[:, :rot_dim] = freqs_flat.sin()

    # Bake rotate_half sign into sin: [-, +, -, +, ...] for rotated dims
    sign = torch.ones(VAE_D_HEAD)
    sign[:rot_dim:2] = -1
    sin_perm_per_head = sin_per_head * sign.unsqueeze(0)

    # Repeat across all 16 heads: (576, 64) -> (576, 1024)
    cos_full = cos_per_head.repeat(1, VAE_DEC_HEADS).to(torch.bfloat16)
    sin_full = sin_perm_per_head.repeat(1, VAE_DEC_HEADS).to(torch.bfloat16)

    return to_tt(cos_full, tt_device), to_tt(sin_full, tt_device)

# ============================================================
# Weight loading
# ============================================================

def find_dit_weights():
    blob_dir = "/root/.cache/huggingface/hub/models--Etched--oasis-500m/blobs/"
    for f in sorted(os.listdir(blob_dir)):
        path = blob_dir + f
        with safe_open(path, framework="pt") as st:
            keys = list(st.keys())
            if any(k.startswith("blocks.") for k in keys):
                return path
    raise FileNotFoundError("DiT weights not found")

def find_vae_weights():
    blob_dir = "/root/.cache/huggingface/hub/models--Etched--oasis-500m/blobs/"
    for f in sorted(os.listdir(blob_dir)):
        path = blob_dir + f
        with safe_open(path, framework="pt") as st:
            keys = list(st.keys())
            if any(k.startswith("encoder.") for k in keys):
                return path
    raise FileNotFoundError("VAE weights not found")

def preload_dit_weights(tt_device, n_frames=2):
    t0 = time.time()
    path = find_dit_weights()
    print("Loading DiT weights from:", path)
    dev = {}

    with safe_open(path, framework="pt") as st:
        # Timestep embedder + external cond: computed on HOST in fp32
        dev["t_emb_w0_host"] = st.get_tensor("t_embedder.mlp.0.weight").T.contiguous().float()  # (256, 1024)
        dev["t_emb_b0_host"] = st.get_tensor("t_embedder.mlp.0.bias").float()  # (1024,)
        dev["t_emb_w2_host"] = st.get_tensor("t_embedder.mlp.2.weight").T.contiguous().float()  # (1024, 1024)
        dev["t_emb_b2_host"] = st.get_tensor("t_embedder.mlp.2.bias").float()  # (1024,)

        dev["ext_cond_w_host"] = st.get_tensor("external_cond.weight").T.contiguous().float()  # (25, 1024)
        dev["ext_cond_b_host"] = st.get_tensor("external_cond.bias").float()  # (1024,)

        # Patch embedder: Conv2d(16, 1024, 2, 2) - keep as host tensor for now
        dev["x_emb_conv_w"] = st.get_tensor("x_embedder.proj.weight")  # (1024, 16, 2, 2)
        dev["x_emb_conv_b"] = st.get_tensor("x_embedder.proj.bias")  # (1024,)

        # Final layer adaLN: on device with ttnn.matmul for precision
        dev["final_adaln_w"] = to_tt(st.get_tensor("final_layer.adaLN_modulation.1.weight").T.contiguous().to(torch.bfloat16), tt_device)
        final_adaln_b = st.get_tensor("final_layer.adaLN_modulation.1.bias").to(torch.bfloat16)
        dev["final_adaln_b"] = to_tt(final_adaln_b.unsqueeze(0).expand(TILE, -1).contiguous(), tt_device)
        dev["final_linear_w"] = to_tt(st.get_tensor("final_layer.linear.weight").T.contiguous().to(torch.bfloat16), tt_device)
        dev["final_linear_b"] = to_tt(expand_bias(st.get_tensor("final_layer.linear.bias").to(torch.bfloat16), N_PATCH_PAD * n_frames), tt_device)

        # Per-block weights
        for i in range(N_BLOCKS):
            for prefix in ["s", "t"]:
                p = "blocks.%d.%s" % (i, prefix)

                # adaLN modulation: on device with ttnn.matmul for precision
                dev["%s.adaln_w" % p] = to_tt(st.get_tensor("%s_adaLN_modulation.1.weight" % p).T.contiguous().to(torch.bfloat16), tt_device)
                adaln_b_raw = st.get_tensor("%s_adaLN_modulation.1.bias" % p).to(torch.bfloat16)
                dev["%s.adaln_b" % p] = to_tt(adaln_b_raw.unsqueeze(0).expand(TILE, -1).contiguous(), tt_device)

                # Combined QKV + QK_swap: (1024, 5120) via single ttnn.matmul
                # Layout: [Q | K | V | Q_swap | K_swap] each 1024 cols
                qkv_w = st.get_tensor("%s_attn.to_qkv.weight" % p).T.contiguous().to(torch.bfloat16)
                q_w = qkv_w[:, :D_MODEL]
                k_w = qkv_w[:, D_MODEL:2*D_MODEL]
                v_w = qkv_w[:, 2*D_MODEL:]
                qkv_full_w = torch.cat([q_w, k_w, v_w,
                                        swap_adjacent_columns(q_w),
                                        swap_adjacent_columns(k_w)], dim=1)
                # TP: interleave heads so ShardTensorToMesh(dim=1) gives each chip its heads
                if N_CHIPS > 1:
                    qkv_full_w = interleave_qkv_for_tp(qkv_full_w, N_CHIPS, D_MODEL, D_HEAD)
                    dev["%s.qkv_full_w" % p] = shard_tt(qkv_full_w, tt_device, dim=1)
                else:
                    dev["%s.qkv_full_w" % p] = to_tt(qkv_full_w, tt_device)
                # Separate QK_swap weights still needed for temporal path
                qk_swap_w = torch.cat([swap_adjacent_columns(q_w),
                                       swap_adjacent_columns(k_w)], dim=1)
                dev["%s.qk_swap_w" % p] = to_tt(qk_swap_w, tt_device)
                dev["%s.qkv_w" % p] = to_tt(qkv_w, tt_device)

                # Output projection: row-parallel (shard input dim)
                SEQ = N_PATCH_PAD * n_frames
                out_w = st.get_tensor("%s_attn.to_out.weight" % p).T.contiguous().to(torch.bfloat16)
                if N_CHIPS > 1:
                    dev["%s.out_w" % p] = shard_tt(out_w, tt_device, dim=0)
                else:
                    dev["%s.out_w" % p] = to_tt(out_w, tt_device)
                out_b = st.get_tensor("%s_attn.to_out.bias" % p).to(torch.bfloat16)
                dev["%s.out_b" % p] = to_tt(expand_bias(out_b, SEQ), tt_device)

                # MLP: fc1 column-parallel (shard output dim), fc2 row-parallel (shard input dim)
                fc1_w = st.get_tensor("%s_mlp.fc1.weight" % p).T.contiguous().to(torch.bfloat16)
                fc1_b = st.get_tensor("%s_mlp.fc1.bias" % p).to(torch.bfloat16)
                fc2_w = st.get_tensor("%s_mlp.fc2.weight" % p).T.contiguous().to(torch.bfloat16)
                fc2_b = st.get_tensor("%s_mlp.fc2.bias" % p).to(torch.bfloat16)
                if N_CHIPS > 1:
                    dev["%s.fc1_w" % p] = shard_tt(fc1_w, tt_device, dim=1)
                    dev["%s.fc1_b" % p] = shard_tt(expand_bias(fc1_b, SEQ), tt_device, dim=1)
                    dev["%s.fc2_w" % p] = shard_tt(fc2_w, tt_device, dim=0)
                else:
                    dev["%s.fc1_w" % p] = to_tt(fc1_w, tt_device)
                    dev["%s.fc1_b" % p] = to_tt(expand_bias(fc1_b, SEQ), tt_device)
                    dev["%s.fc2_w" % p] = to_tt(fc2_w, tt_device)
                dev["%s.fc2_b" % p] = to_tt(expand_bias(fc2_b, SEQ), tt_device)
                # 1D bias for ttnn.linear
                dev["%s.fc2_b_1d" % p] = to_tt(fc2_b.unsqueeze(0).contiguous(), tt_device)

        SEQ = N_PATCH_PAD * n_frames
        dev["ln_w_ones"] = to_tt(torch.ones(SEQ, D_MODEL, dtype=torch.bfloat16), tt_device)
        dev["ln_b_zeros"] = to_tt(torch.zeros(SEQ, D_MODEL, dtype=torch.bfloat16), tt_device)

        # RoPE: load learned freqs from block 0 (shared across all blocks)
        global SPATIAL_ROPE_FREQS, TEMPORAL_ROPE_FREQS
        s_freqs = st.get_tensor("blocks.0.s_attn.rotary_emb.freqs").float()  # (16,)
        t_freqs = st.get_tensor("blocks.0.t_attn.rotary_emb.freqs").float()  # (32,)
        SPATIAL_ROPE_FREQS = precompute_spatial_rope_freqs(s_freqs)  # (144, 64)
        TEMPORAL_ROPE_FREQS = precompute_temporal_rope_freqs(t_freqs)  # (max_t, 64)
        print("Precomputed RoPE freqs: spatial %s, temporal %s" % (
            tuple(SPATIAL_ROPE_FREQS.shape), tuple(TEMPORAL_ROPE_FREQS.shape)))

        # Device-side spatial RoPE tables (sized for N_HEADS_TP heads per chip)
        dev["spatial_cos"], dev["spatial_sin_perm"] = \
            build_spatial_rope_device_tables(SPATIAL_ROPE_FREQS, n_frames, tt_device)

        # Device-side temporal RoPE tables: per-frame freqs broadcast to all patches
        # For T frames: frame t uses temporal_freqs[t], broadcast across N_PATCH_PAD rows
        temporal_per_frame = TEMPORAL_ROPE_FREQS[:n_frames]  # (T, 64)
        # Expand each frame's freqs to fill N_PATCH_PAD rows
        temporal_expanded = temporal_per_frame.unsqueeze(1).expand(
            n_frames, N_PATCH_PAD, TEMPORAL_ROPE_DIM).reshape(
            n_frames * N_PATCH_PAD, TEMPORAL_ROPE_DIM)  # (SEQ, 64)
        dev["temporal_cos"], dev["temporal_sin_perm"] = \
            build_rope_device_tables(temporal_expanded, n_frames * N_PATCH_PAD,
                                     N_PATCH_PAD, N_HEADS_TP, n_frames, tt_device)

    elapsed = time.time() - t0
    print("Preloaded %d tensors in %.1fs" % (len(dev), elapsed))
    return dev

def preload_vae_decoder_weights(vae, tt_device):
    """Extract VAE decoder weights from CPU model and load onto device."""
    t0 = time.time()
    vd = {}
    sd = vae.state_dict()

    # post_quant_conv: Linear(16, 1024)
    pqc_w = sd["post_quant_conv.weight"].T.contiguous().to(torch.bfloat16)  # (16, 1024)
    # Pad input dim to 32 for tile alignment
    pqc_w_padded = torch.zeros(32, VAE_DEC_DIM, dtype=torch.bfloat16)
    pqc_w_padded[:VAE_LATENT_DIM] = pqc_w
    vd["pqc_w"] = to_tt(pqc_w_padded, tt_device)
    vd["pqc_b"] = to_tt(expand_bias(sd["post_quant_conv.bias"].to(torch.bfloat16), VAE_SEQ_LEN), tt_device)

    for i in range(VAE_DEC_DEPTH):
        p = "decoder.%d" % i

        # LN weight/bias expanded to (VAE_SEQ_LEN, D) for layernorm kernel
        for ln_idx in ["1", "2"]:
            w = sd["%s.norm%s.weight" % (p, ln_idx)].to(torch.bfloat16)
            b = sd["%s.norm%s.bias" % (p, ln_idx)].to(torch.bfloat16)
            vd["%s.norm%s_w" % (p, ln_idx)] = to_tt(expand_bias(w, VAE_SEQ_LEN), tt_device)
            vd["%s.norm%s_b" % (p, ln_idx)] = to_tt(expand_bias(b, VAE_SEQ_LEN), tt_device)

        # Combined QKV+swap: [Q, K, V, Q_swap, K_swap] = (1024, 5120)
        qkv_w = sd["%s.attn.qkv.weight" % p].T.contiguous().to(torch.bfloat16)
        q_w = qkv_w[:, :VAE_DEC_DIM]
        k_w = qkv_w[:, VAE_DEC_DIM:2*VAE_DEC_DIM]
        v_w = qkv_w[:, 2*VAE_DEC_DIM:]
        qkv_full_w = torch.cat([q_w, k_w, v_w,
                                swap_adjacent_columns(q_w),
                                swap_adjacent_columns(k_w)], dim=1)
        vd["%s.qkv_full_w" % p] = to_tt(qkv_full_w, tt_device)

        # QKV bias with swap for RoPE trick
        qkv_b = sd["%s.attn.qkv.bias" % p].to(torch.bfloat16)
        b_q = qkv_b[:VAE_DEC_DIM]
        b_k = qkv_b[VAE_DEC_DIM:2*VAE_DEC_DIM]
        b_v = qkv_b[2*VAE_DEC_DIM:]
        full_b = torch.cat([b_q, b_k, b_v,
                           swap_adjacent_elements(b_q),
                           swap_adjacent_elements(b_k)])
        vd["%s.qkv_full_b" % p] = to_tt(expand_bias(full_b, VAE_SEQ_LEN), tt_device)

        # Output projection
        vd["%s.proj_w" % p] = to_tt(
            sd["%s.attn.proj.weight" % p].T.contiguous().to(torch.bfloat16), tt_device)
        vd["%s.proj_b" % p] = to_tt(
            expand_bias(sd["%s.attn.proj.bias" % p].to(torch.bfloat16), VAE_SEQ_LEN), tt_device)

        # MLP
        vd["%s.fc1_w" % p] = to_tt(
            sd["%s.mlp.fc1.weight" % p].T.contiguous().to(torch.bfloat16), tt_device)
        vd["%s.fc1_b" % p] = to_tt(
            expand_bias(sd["%s.mlp.fc1.bias" % p].to(torch.bfloat16), VAE_SEQ_LEN), tt_device)
        vd["%s.fc2_w" % p] = to_tt(
            sd["%s.mlp.fc2.weight" % p].T.contiguous().to(torch.bfloat16), tt_device)
        vd["%s.fc2_b" % p] = to_tt(
            expand_bias(sd["%s.mlp.fc2.bias" % p].to(torch.bfloat16), VAE_SEQ_LEN), tt_device)

    # Final LN
    vd["dec_norm_w"] = to_tt(
        expand_bias(sd["dec_norm.weight"].to(torch.bfloat16), VAE_SEQ_LEN), tt_device)
    vd["dec_norm_b"] = to_tt(
        expand_bias(sd["dec_norm.bias"].to(torch.bfloat16), VAE_SEQ_LEN), tt_device)

    # Predictor: Linear(1024, 1200) - pad to 1216 for tile alignment
    VAE_PATCH_DIM_PAD = ((VAE_PATCH_DIM + TILE - 1) // TILE) * TILE  # 1216
    pred_w_raw = sd["predictor.weight"].T.contiguous().to(torch.bfloat16)  # (1024, 1200)
    pred_w_padded = torch.zeros(VAE_DEC_DIM, VAE_PATCH_DIM_PAD, dtype=torch.bfloat16)
    pred_w_padded[:, :VAE_PATCH_DIM] = pred_w_raw
    vd["pred_w"] = to_tt(pred_w_padded, tt_device)
    pred_b_raw = sd["predictor.bias"].to(torch.bfloat16)  # (1200,)
    pred_b_padded = torch.zeros(VAE_PATCH_DIM_PAD, dtype=torch.bfloat16)
    pred_b_padded[:VAE_PATCH_DIM] = pred_b_raw
    vd["pred_b"] = to_tt(expand_bias(pred_b_padded, VAE_SEQ_LEN), tt_device)

    # VAE RoPE tables
    vd["vae_cos"], vd["vae_sin_perm"] = precompute_vae_rope(vae, tt_device)

    elapsed = time.time() - t0
    print("Preloaded %d VAE decoder tensors in %.1fs" % (len(vd), elapsed))
    return vd

def prealloc_vae_scratch(tt_device):
    """Pre-allocate VAE decoder scratch buffers. All DRAM to avoid L1 pressure
    (DiT already uses ~25MB of L1). Can promote hot tensors to L1 later."""
    t0 = time.time()
    vs = {}
    vs["vae_input"] = zeros_tt((VAE_SEQ_LEN, 32), tt_device)  # padded latent for trace
    vs["z_a"] = zeros_tt((VAE_SEQ_LEN, VAE_DEC_DIM), tt_device)
    vs["z_b"] = zeros_tt((VAE_SEQ_LEN, VAE_DEC_DIM), tt_device)
    vs["normed"] = zeros_tt((VAE_SEQ_LEN, VAE_DEC_DIM), tt_device)
    vs["qkv_full"] = zeros_tt((VAE_SEQ_LEN, 5 * VAE_DEC_DIM), tt_device)
    vs["o_proj"] = zeros_tt((VAE_SEQ_LEN, VAE_DEC_DIM), tt_device)
    vs["fc1"] = zeros_tt((VAE_SEQ_LEN, VAE_D_MLP), tt_device)
    vs["gelu"] = zeros_tt((VAE_SEQ_LEN, VAE_D_MLP), tt_device)
    vs["fc2"] = zeros_tt((VAE_SEQ_LEN, VAE_DEC_DIM), tt_device)
    VAE_PATCH_DIM_PAD = ((VAE_PATCH_DIM + TILE - 1) // TILE) * TILE  # 1216
    vs["pred_out"] = zeros_tt((VAE_SEQ_LEN, VAE_PATCH_DIM_PAD), tt_device)
    # SDPA buffers: heads-first layout for fused RoPE kernel output
    SDPA_ROWS = VAE_DEC_HEADS * VAE_SEQ_LEN  # 16 * 576 = 9216
    vs["q_sdpa"] = zeros_tt((SDPA_ROWS, VAE_D_HEAD), tt_device)
    vs["k_sdpa"] = zeros_tt((SDPA_ROWS, VAE_D_HEAD), tt_device)
    vs["v_sdpa"] = zeros_tt((SDPA_ROWS, VAE_D_HEAD), tt_device)
    elapsed = time.time() - t0
    print("Pre-allocated %d VAE scratch tensors in %.1fs" % (len(vs), elapsed))
    return vs

# ============================================================
# Scratch buffers
# ============================================================

def prealloc_scratch(tt_device, n_frames=2):
    """Pre-allocate scratch buffers. n_frames=2 for prompt+generated frame."""
    t0 = time.time()
    s = {}
    SEQ = N_PATCH_PAD * n_frames  # 320 for T=2
    D_MODEL_TP = D_MODEL // N_CHIPS  # per-chip model dim for sharded tensors
    D_MLP_TP = D_MLP // N_CHIPS
    # L1 intermediates: eliminates DRAM round-trips between operations.
    # Residual-carrying tensors (z_a, z_scratch, o_proj, fc2) are f32 to preserve
    # cond-uncond signal across 16 blocks (bf16 accumulation destroys it).
    s["z_a"] = zeros_l1_f32((SEQ, D_MODEL), tt_device)
    s["z_b"] = zeros_l1((SEQ, D_MODEL), tt_device)
    s["normed"] = zeros_l1((SEQ, D_MODEL), tt_device)
    s["modulated"] = zeros_l1((SEQ, D_MODEL), tt_device)
    s["qkv"] = zeros_l1((SEQ, 3 * D_MODEL_TP), tt_device)
    s["qkv_full"] = zeros_l1((SEQ, 5 * D_MODEL_TP), tt_device)
    s["o_proj"] = zeros_l1_f32((SEQ, D_MODEL), tt_device)
    s["fc1"] = zeros_l1((SEQ, D_MLP_TP), tt_device)
    s["gelu"] = zeros_l1((SEQ, D_MLP_TP), tt_device)
    s["fc2"] = zeros_l1_f32((SEQ, D_MODEL), tt_device)
    # Conditioning scratch (per-frame, so TILE * n_frames)
    s["t_emb_a"] = zeros_tt((TILE, D_MODEL), tt_device)
    s["t_emb_b"] = zeros_tt((TILE, D_MODEL), tt_device)
    s["cond"] = zeros_tt((TILE, D_MODEL), tt_device)
    # adaLN scratch (replicated, not sharded)
    s["adaln_out"] = to_tt_l1(torch.zeros(TILE, 6 * D_MODEL, dtype=torch.bfloat16), tt_device)
    s["silu_cond"] = zeros_tt((TILE, D_MODEL), tt_device)
    # Packed adaln: (SEQ, 6*D_MODEL) reused across blocks (replicated)
    s["adaln_packed"] = zeros_l1((SEQ, 6 * D_MODEL), tt_device)
    # Per-frame adaln expanded: (N_PATCH_PAD, 6*D_MODEL) for building packed tensor
    for f in range(n_frames):
        s["adaln_frame_%d" % f] = zeros_l1((N_PATCH_PAD, 6 * D_MODEL), tt_device)
    # RoPE scratch: sized for N_HEADS_TP heads per chip
    s["q_roped"] = zeros_l1((SEQ, D_MODEL_TP), tt_device)
    s["k_roped"] = zeros_l1((SEQ, D_MODEL_TP), tt_device)
    # SDPA-format scratch for fused RoPE+layout kernel (spatial)
    # Stored as 2D: (BATCH_S * N_PATCH_PAD, D_HEAD), reshaped to 4D for SDPA
    BATCH_S = n_frames * N_HEADS_TP
    s["q_sdpa"] = to_tt_l1(torch.zeros(BATCH_S * N_PATCH_PAD, D_HEAD, dtype=torch.bfloat16), tt_device)
    s["k_sdpa"] = to_tt_l1(torch.zeros(BATCH_S * N_PATCH_PAD, D_HEAD, dtype=torch.bfloat16), tt_device)
    s["v_sdpa"] = to_tt_l1(torch.zeros(BATCH_S * N_PATCH_PAD, D_HEAD, dtype=torch.bfloat16), tt_device)
    # Temporal mega kernel scratch (Q/K roped + V) - sized for TP heads
    s["t_q_scratch"] = zeros_l1((SEQ, D_MODEL_TP), tt_device)
    s["t_k_scratch"] = zeros_l1((SEQ, D_MODEL_TP), tt_device)
    s["t_v_scratch"] = zeros_l1((SEQ, D_MODEL_TP), tt_device)
    # SDPA scratch: (SEQ, D_MODEL) for TT-Lang spatial SDPA output
    s["sdpa_out"] = zeros_l1((SEQ, D_MODEL), tt_device)
    # Mega kernel B scratch (f32: residual-carrying, see comment above)
    s["z_scratch"] = zeros_l1_f32((SEQ, D_MODEL), tt_device)
    s["gelu_scratch"] = zeros_l1((SEQ, D_MLP), tt_device)
    # Final layer
    s["final_adaln"] = zeros_l1((TILE, 2 * D_MODEL), tt_device)
    s["final_out"] = zeros_l1((SEQ, OUT_DIM), tt_device)
    # Pre-allocated SiLU output buffers per frame (L1 for fast access, read 32x/step)
    for f in range(n_frames):
        s["silu_out_%d" % f] = to_tt_l1(torch.zeros(TILE, D_MODEL, dtype=torch.bfloat16), tt_device)
    s["n_frames"] = n_frames
    # DDIM arithmetic scratch (N_PATCH_PAD, OUT_DIM)
    s["ddim_x_start"] = zeros_l1((N_PATCH_PAD, OUT_DIM), tt_device)
    s["ddim_x_noise"] = zeros_l1((N_PATCH_PAD, OUT_DIM), tt_device)
    s["ddim_tmp"] = zeros_l1((N_PATCH_PAD, OUT_DIM), tt_device)
    elapsed = time.time() - t0
    print("Pre-allocated %d scratch tensors (T=%d) in %.1fs" % (len(s) - 1, n_frames, elapsed))
    return s

# ============================================================
# DiT forward pass
# ============================================================

def patch_embed_host(x_latent, conv_w, conv_b):
    """x_latent: (B, C, H, W) = (1, 16, 18, 32) -> (N_PATCHES, D_MODEL)"""
    x = F.conv2d(x_latent.float(), conv_w.float(), conv_b.float(), stride=PATCH_SIZE)
    # x: (1, 1024, 9, 16)
    x = rearrange(x, "b d h w -> b h w d")
    # x: (1, 9, 16, 1024) -> flatten spatial
    x = x.reshape(1, N_PATCHES, D_MODEL)
    return x.squeeze(0).to(torch.bfloat16)  # (144, 1024)

def patchify_to_output_space(x_img):
    """Inverse of unpatchify: (1, C, H, W) -> (N_PATCHES, OUT_DIM).
    Matches the pixel ordering that unpatchify_host expects."""
    x = x_img.float().reshape(1, IN_CHANNELS, FRAME_H, PATCH_SIZE, FRAME_W, PATCH_SIZE)
    x = torch.einsum("nchpwq->nhwpqc", x)
    return x.reshape(N_PATCHES, OUT_DIM)

def unpatchify_host(x, patch_size, out_channels, h, w):
    """x: (N_PATCHES, patch_size^2 * out_channels) -> (1, out_channels, H, W)"""
    x = x.float().reshape(1, h, w, patch_size, patch_size, out_channels)
    x = torch.einsum("nhwpqc->nchpwq", x)
    return x.reshape(1, out_channels, h * patch_size, w * patch_size)

def build_per_frame_adaln(silu_cond_list, prefix, dev, scr, tt_device):
    """Compute adaLN params on DEVICE using ttnn.matmul.
    silu_cond_list: list of T pre-computed SiLU(cond) device tensors, each (TILE, D_MODEL).
    Returns: packed (SEQ, 6*D_MODEL) device tensor.
    """
    T = len(silu_cond_list)
    N_REPEAT = N_PATCH_PAD // TILE  # 5

    per_frame_expanded = []
    for silu_cond in silu_cond_list:
        ttnn.linear(silu_cond, dev["%s.adaln_w" % prefix],
                    bias=dev["%s.adaln_b" % prefix],
                    optional_output_tensor=scr["adaln_out"],
                    compute_kernel_config=COMPUTE_HIFI)
        expanded = ttnn.concat([scr["adaln_out"]] * N_REPEAT, dim=0)
        per_frame_expanded.append(expanded)

    if T == 1:
        return per_frame_expanded[0]
    else:
        return ttnn.concat(per_frame_expanded, dim=0)

def run_sub_block(prefix, x_tt, silu_cond_list, dev, scr, tt_device, scaler, mean_scale, attn_type="spatial"):
    """Run one spatial or temporal sub-block.
    prefix: e.g. "blocks.0.s" or "blocks.0.t"
    silu_cond_list: list of T pre-computed SiLU(cond) device tensors, each (TILE, D_MODEL)
    attn_type: "spatial" or "temporal"
    x_tt: (N_PATCH_PAD * T, D_MODEL) device tensor
    """
    T = len(silu_cond_list)
    SEQ = N_PATCH_PAD * T

    if PROFILE_BLOCKS:
        import time as _t
        _p = lambda label: None
        class _Timer:
            def __init__(self):
                self.t = _t.time()
                self.times = []
            def mark(self, label):
                now = _t.time()
                self.times.append((label, (now - self.t) * 1000))
                self.t = now
            def report(self, prefix, attn_type):
                parts = " | ".join("%s:%.1fms" % (l, t) for l, t in self.times)
                total = sum(t for _, t in self.times)
                print("    [%s %s] %.1fms: %s" % (prefix, attn_type, total, parts))
        _timer = _Timer()
    else:
        class _DummyTimer:
            def mark(self, label): pass
            def report(self, prefix, attn_type): pass
        _timer = _DummyTimer()

    _do_dev_profile = (prefix == PROFILE_BLOCK_DEVICE)
    if _do_dev_profile:
        import time as _dt
        ttnn.synchronize_device(tt_device)
        _t0 = _dt.time()

    # Compute per-frame adaLN params (SiLU already applied to cond) - packed (SEQ, 6*D_MODEL)
    adaln_packed = build_per_frame_adaln(silu_cond_list, prefix, dev, scr, tt_device)
    _timer.mark("adaln")
    if _do_dev_profile:
        ttnn.synchronize_device(tt_device)
        _t1 = _dt.time(); print("      %s adaln: %.1fms" % (prefix, (_t1 - _t0) * 1000)); _t0 = _t1

    # LayerNorm + adaLN modulate via ttnn ops
    # TODO: revisit TT-Lang fused_ln_adaln_d1024 kernel (0.3ms vs 0.15ms ttnn, 2x slower)
    # Was: fused_ln_adaln_d1024(x_tt, scaler, mean_scale, adaln_packed, scr["modulated"])
    normed_a = ttnn.layer_norm(x_tt)
    shift_a = ttnn.slice(adaln_packed, [0, 0], [SEQ, D_MODEL])
    scale_a = ttnn.slice(adaln_packed, [0, D_MODEL], [SEQ, 2 * D_MODEL])
    ttnn.add(scale_a, 1.0, output_tensor=scr["normed"])
    ttnn.multiply(normed_a, scr["normed"], output_tensor=scr["modulated"])
    ttnn.add(scr["modulated"], shift_a, output_tensor=scr["modulated"])
    _timer.mark("norm+mod")
    if _do_dev_profile:
        ttnn.synchronize_device(tt_device)
        _t1 = _dt.time(); print("      %s norm+mod: %.1fms" % (prefix, (_t1 - _t0) * 1000)); _t0 = _t1

    # Hybrid: ttnn.matmul for QKV (72-core parallel), TT-Lang for RoPE+SDPA
    # NOTE: mega_qkv_rope_sdpa fuses QKV+RoPE+SDPA into one kernel but only uses
    # 32 cores for K-accumulation matmul. ttnn.matmul uses all 72 cores (30x faster).
    # To restore fused version: mega_qkv_rope_sdpa(scr["modulated"], qkv_w, cos, sin, scaler, out)
    ttnn.matmul(scr["modulated"], dev["%s.qkv_full_w" % prefix],
                optional_output_tensor=scr["qkv_full"],
                compute_kernel_config=COMPUTE_HIFI)
    if _do_dev_profile:
        ttnn.synchronize_device(tt_device)
        _t1 = _dt.time(); print("      %s qkv_matmul: %.1fms" % (prefix, (_t1 - _t0) * 1000)); _t0 = _t1

    D_MODEL_TP = D_MODEL // N_CHIPS

    if attn_type == "spatial":
        cos_key, sin_key = "spatial_cos", "spatial_sin_perm"
        # Fused RoPE + layout: reads qkv_full, writes Q,K,V in SDPA format directly
        rope_layout_spatial(scr["qkv_full"], dev[cos_key], dev[sin_key],
                           scr["q_sdpa"], scr["k_sdpa"], scr["v_sdpa"])
        BATCH_S = T * N_HEADS_TP
        # Already in (BATCH_S * N_PATCH_PAD, D_HEAD) layout, just reshape for SDPA
        q_s = ttnn.reshape(scr["q_sdpa"], [BATCH_S, 1, N_PATCH_PAD, D_HEAD])
        k_s = ttnn.reshape(scr["k_sdpa"], [BATCH_S, 1, N_PATCH_PAD, D_HEAD])
        v_s = ttnn.reshape(scr["v_sdpa"], [BATCH_S, 1, N_PATCH_PAD, D_HEAD])
        attn_out = ttnn.transformer.scaled_dot_product_attention(
            q_s, k_s, v_s, is_causal=False, compute_kernel_config=COMPUTE_HIFI)
        attn_out = ttnn.reshape(attn_out, [T, N_HEADS_TP, N_PATCH_PAD, D_HEAD])
        attn_out = ttnn.permute(attn_out, [0, 2, 1, 3])
        attn_2d = ttnn.reshape(attn_out, [SEQ, D_MODEL_TP])
        _timer.mark("qkv+sdpa")
    else:
        # Fused temporal RoPE: reads qkv_full, writes q/k/v in (SEQ, D_MODEL_TP)
        cos_key, sin_key = "temporal_cos", "temporal_sin_perm"
        rope_temporal(scr["qkv_full"], dev[cos_key], dev[sin_key],
                      scr["q_roped"], scr["k_roped"], scr["t_v_scratch"])
        # Temporal: each patch attends across frames, batch over (patch, head)
        BATCH_T = N_PATCH_PAD * N_HEADS_TP
        q_t = ttnn.reshape(scr["q_roped"], [T, BATCH_T, D_HEAD])
        q_t = ttnn.permute(q_t, [1, 0, 2])
        q_t = ttnn.reshape(q_t, [BATCH_T, 1, T, D_HEAD])
        k_t = ttnn.reshape(scr["k_roped"], [T, BATCH_T, D_HEAD])
        k_t = ttnn.permute(k_t, [1, 0, 2])
        k_t = ttnn.reshape(k_t, [BATCH_T, 1, T, D_HEAD])
        v_t = ttnn.reshape(scr["t_v_scratch"], [T, BATCH_T, D_HEAD])
        v_t = ttnn.permute(v_t, [1, 0, 2])
        v_t = ttnn.reshape(v_t, [BATCH_T, 1, T, D_HEAD])
        attn_out = ttnn.transformer.scaled_dot_product_attention(
            q_t, k_t, v_t, is_causal=True, compute_kernel_config=COMPUTE_HIFI)
        attn_out = ttnn.reshape(attn_out, [BATCH_T, T, D_HEAD])
        attn_out = ttnn.permute(attn_out, [1, 0, 2])
        attn_2d = ttnn.reshape(attn_out, [SEQ, D_MODEL_TP])
        _timer.mark("sdpa")
    if _do_dev_profile:
        ttnn.synchronize_device(tt_device)
        _t1 = _dt.time(); print("      %s rope+sdpa: %.1fms" % (prefix, (_t1 - _t0) * 1000)); _t0 = _t1

    # Separate ops: ttnn.matmul for large matmuls, TT-Lang for fused elementwise
    # NOTE: mega_post_attn_kernel fuses O proj+LN+FC1+GELU+FC2+residual but only uses
    # 10 cores (per-row). ttnn.matmul on 72 cores is 7x faster for the matmuls.
    # To restore fused version: mega_post_attn_kernel(attn_2d, x_tt, adaln_packed,
    #   out_w, out_b, fc1_w, fc1_b, fc2_w, fc2_b, scaler, mean_scale,
    #   z_scratch, gelu_scratch, z_a)
    # Phase A: O proj (row-parallel) + all_reduce + bias
    # TODO: fuse matmul+bias via ttnn.linear when N_CHIPS==1 (no all_reduce between them)
    o_proj = ttnn.matmul(attn_2d, dev["%s.out_w" % prefix],
                         compute_kernel_config=COMPUTE_HIFI)
    if N_CHIPS > 1:
        o_proj = ttnn.all_reduce(o_proj)
    ttnn.add(o_proj, dev["%s.out_b" % prefix], output_tensor=scr["o_proj"])
    gate_msa = ttnn.slice(adaln_packed, [0, 2 * D_MODEL], [SEQ, 3 * D_MODEL])

    # Phase B: gated_residual + LN + adaLN modulate via ttnn ops
    # TODO: revisit TT-Lang fused_gated_res_ln_adaln_d1024 kernel (0.8ms vs 0.15ms ttnn, 5.5x slower)
    # Was: fused_gated_res_ln_adaln_d1024(x_tt, scr["o_proj"], gate_msa, scaler, mean_scale,
    #          adaln_packed, scr["z_scratch"], scr["modulated"])
    ttnn.multiply(scr["o_proj"], gate_msa, output_tensor=scr["normed"])
    ttnn.add(x_tt, scr["normed"], output_tensor=scr["z_scratch"])
    normed = ttnn.layer_norm(scr["z_scratch"])
    shift_b = ttnn.slice(adaln_packed, [0, 3 * D_MODEL], [SEQ, 4 * D_MODEL])
    scale_b = ttnn.slice(adaln_packed, [0, 4 * D_MODEL], [SEQ, 5 * D_MODEL])
    ttnn.add(scale_b, 1.0, output_tensor=scr["normed"])
    ttnn.multiply(normed, scr["normed"], output_tensor=scr["modulated"])
    ttnn.add(scr["modulated"], shift_b, output_tensor=scr["modulated"])
    if _do_dev_profile:
        ttnn.synchronize_device(tt_device)
        _t1 = _dt.time(); print("      %s ln+mod: %.1fms" % (prefix, (_t1 - _t0) * 1000)); _t0 = _t1

    # Phase C: Fused FC1 + bias + GELU (single ttnn.linear call)
    ttnn.linear(scr["modulated"], dev["%s.fc1_w" % prefix], bias=dev["%s.fc1_b" % prefix],
                activation="gelu", optional_output_tensor=scr["gelu"],
                compute_kernel_config=COMPUTE_HIFI)
    if _do_dev_profile:
        ttnn.synchronize_device(tt_device)
        _t1 = _dt.time(); print("      %s fc1+gelu: %.1fms" % (prefix, (_t1 - _t0) * 1000)); _t0 = _t1

    # Phase D: FC2 (row-parallel) + all_reduce + bias + gated residual
    # TODO: fuse matmul+bias via ttnn.linear when N_CHIPS==1 (no all_reduce between them)
    fc2_out = ttnn.matmul(scr["gelu"], dev["%s.fc2_w" % prefix],
                          compute_kernel_config=COMPUTE_HIFI)
    if N_CHIPS > 1:
        fc2_out = ttnn.all_reduce(fc2_out)
    ttnn.add(fc2_out, dev["%s.fc2_b" % prefix], output_tensor=scr["fc2"])
    gate_mlp = ttnn.slice(adaln_packed, [0, 5 * D_MODEL], [SEQ, 6 * D_MODEL])
    # TODO: revisit TT-Lang gated_residual_kernel once it handles mixed f32/bf16 inputs.
    # Currently using ttnn ops because the kernel rejects f32 residual + bf16 gate.
    # Was: gated_residual_kernel(scr["z_scratch"], scr["fc2"], gate_mlp, scr["z_a"])
    ttnn.multiply(scr["fc2"], gate_mlp, output_tensor=scr["z_a"])
    ttnn.add(scr["z_scratch"], scr["z_a"], output_tensor=scr["z_a"])
    _timer.mark("post_attn")
    if _do_dev_profile:
        ttnn.synchronize_device(tt_device)
        _t1 = _dt.time(); print("      %s fc2+res: %.1fms" % (prefix, (_t1 - _t0) * 1000)); _t0 = _t1

    _timer.report(prefix, attn_type)
    return scr["z_a"]

def compute_cond_for_frame(t_scalar, action_vec, dev, scr, tt_device):
    """Compute conditioning vector on HOST in fp32 for precision.
    t_scalar: integer timestep
    action_vec: (25,) action tensor or None
    Returns: (TILE, D_MODEL) device tensor
    """
    t_freq = timestep_embedding(torch.tensor([t_scalar]), FREQ_DIM)  # (1, 256)

    # t_embedder MLP on host in fp32: Linear(256,1024) -> SiLU -> Linear(1024,1024)
    h = (t_freq[0].float() @ dev["t_emb_w0_host"]) + dev["t_emb_b0_host"]
    h = h * torch.sigmoid(h)  # SiLU
    cond = (h @ dev["t_emb_w2_host"]) + dev["t_emb_b2_host"]

    # External conditioning on host
    if action_vec is not None:
        ext = (action_vec.float() @ dev["ext_cond_w_host"]) + dev["ext_cond_b_host"]
        cond = cond + ext

    # Package as (TILE, D_MODEL) device tensor - fill ALL rows for device-side broadcast
    cond_bf16 = cond.to(torch.bfloat16)
    cond_pad = cond_bf16.unsqueeze(0).expand(TILE, -1).contiguous()
    return to_tt(cond_pad, tt_device)

def precompute_gen_cond(action_vec, ddim_steps, noise_range, dev, scr, tt_device):
    """Pre-compute conditioning for all DDIM noise levels with given action."""
    cond_per_step = {}
    for noise_idx in reversed(range(1, ddim_steps + 1)):
        noise_level = int(noise_range[noise_idx].item())
        cond_per_step[noise_idx] = compute_cond_for_frame(
            noise_level, action_vec, dev, scr, tt_device)
    return cond_per_step

def dit_forward_device(z_cur, cond_list, dev, scr, tt_device, scaler, mean_scale, profile_step=False):
    """DiT forward pass, all on device. Returns final_out as device tensor.
    z_cur: (SEQ, D_MODEL) device tensor
    cond_list: list of T device tensors for conditioning
    Returns: device tensor (SEQ, OUT_DIM)
    """
    T = len(cond_list)
    SEQ = N_PATCH_PAD * T

    if profile_step:
        import time as _t
        _pt = _t.time
        ttnn.synchronize_device(tt_device)

    # Pre-compute SiLU(cond) once per step, reuse across all 32 sub-blocks + final layer
    # Uses pre-allocated buffers from scratch (no allocation during trace)
    silu_cond_list = []
    for t_idx in range(T):
        silu_out = scr["silu_out_%d" % t_idx]
        silu_kernel(cond_list[t_idx], silu_out)
        silu_cond_list.append(silu_out)

    if profile_step:
        ttnn.synchronize_device(tt_device)
        print("  [PROFILE] silu: %.1fms" % ((_pt() - _pt.__self__) if False else 0))

    z_cur = ttnn.typecast(z_cur, ttnn.float32)

    # 16 blocks: each has spatial + temporal sub-block
    if profile_step:
        t_blocks = _pt()
    for block_idx in range(N_BLOCKS):
        z_cur = run_sub_block(
            "blocks.%d.s" % block_idx, z_cur, silu_cond_list,
            dev, scr, tt_device, scaler, mean_scale, attn_type="spatial"
        )
        if profile_step:
            ttnn.synchronize_device(tt_device)
            t1 = _pt()
            print("      block %d.s: %.1fms" % (block_idx, (t1 - t_blocks) * 1000))
            t_blocks = t1
        z_cur = run_sub_block(
            "blocks.%d.t" % block_idx, z_cur, silu_cond_list,
            dev, scr, tt_device, scaler, mean_scale, attn_type="temporal"
        )
        if profile_step:
            ttnn.synchronize_device(tt_device)
            t1 = _pt()
            print("      block %d.t: %.1fms" % (block_idx, (t1 - t_blocks) * 1000))
            t_blocks = t1

    if profile_step:
        print("      === final layer ===")
        t_blocks = _pt()
    # Final layer: per-frame adaLN (SiLU already computed)
    N_REPEAT = N_PATCH_PAD // TILE  # 5
    per_frame_final = []
    for t_idx in range(T):
        final_raw = ttnn.linear(silu_cond_list[t_idx], dev["final_adaln_w"],
                                bias=dev["final_adaln_b"],
                                compute_kernel_config=COMPUTE_HIFI)
        expanded = ttnn.concat([final_raw] * N_REPEAT, dim=0)
        per_frame_final.append(expanded)
    if T == 1:
        full_final = per_frame_final[0]
    else:
        full_final = ttnn.concat(per_frame_final, dim=0)

    shift_tt = ttnn.slice(full_final, [0, 0], [SEQ, D_MODEL])
    scale_tt = ttnn.slice(full_final, [0, D_MODEL], [SEQ, 2 * D_MODEL])

    # TT-Lang final layer kernels are bf16-only; typecast the f32 residual stream down.
    z_cur = ttnn.typecast(z_cur, ttnn.bfloat16)
    layernorm_d1024(z_cur, dev["ln_w_ones"], dev["ln_b_zeros"], scaler, mean_scale, scr["normed"])
    adaln_modulate_kernel(scr["normed"], shift_tt, scale_tt, scr["modulated"])
    # TODO: fuse linear+bias via ttnn.linear (currently uses TT-Lang linear_k32 kernel)
    linear_k32(scr["modulated"], dev["final_linear_w"], scr["final_out"])
    ttnn.add(scr["final_out"], dev["final_linear_b"], output_tensor=scr["final_out"])
    result = scr["final_out"]
    if profile_step:
        ttnn.synchronize_device(tt_device)
        print("      final_layer: %.1fms" % ((_pt() - t_blocks) * 1000))
    return result

# ============================================================
# VAE
# ============================================================

def load_vae_cpu():
    """Load VAE model on CPU (needed for encoding prompt image)."""
    import sys
    sys.path.insert(0, "/tmp")
    from vae import VAE_models

    vae = VAE_models["vit-l-20-shallow-encoder"]()
    path = find_vae_weights()
    print("Loading VAE weights from:", path)
    load_model(vae, path)
    vae = vae.eval()
    return vae


def vae_decode_forward(vae_dev, vae_scr, scaler, mean_scale):
    """VAE forward pass on device (trace-compatible, no allocations).
    Input: vae_scr["vae_input"] already populated with padded latent.
    Output: vae_scr["pred_out"] contains predictions."""
    # post_quant_conv: (576, 32) @ (32, 1024) + bias -> (576, 1024) in z_a
    ttnn.linear(vae_scr["vae_input"], vae_dev["pqc_w"], bias=vae_dev["pqc_b"],
                optional_output_tensor=vae_scr["z_a"],
                compute_kernel_config=COMPUTE_HIFI)

    for i in range(VAE_DEC_DEPTH):
        p = "decoder.%d" % i

        # === Attention path: x + attn(LN(x)) ===
        layernorm_d1024(vae_scr["z_a"], vae_dev["%s.norm1_w" % p], vae_dev["%s.norm1_b" % p],
                        scaler, mean_scale, vae_scr["normed"])

        ttnn.linear(vae_scr["normed"], vae_dev["%s.qkv_full_w" % p],
                    bias=vae_dev["%s.qkv_full_b" % p],
                    optional_output_tensor=vae_scr["qkv_full"],
                    compute_kernel_config=COMPUTE_HIFI)

        # Fused RoPE: reads qkv_full, writes Q/K/V in heads-first SDPA layout
        vae_rope_fused(vae_scr["qkv_full"], vae_dev["vae_cos"], vae_dev["vae_sin_perm"],
                       vae_scr["q_sdpa"], vae_scr["k_sdpa"], vae_scr["v_sdpa"])

        # Single reshape to SDPA format (data already in heads-first order)
        q_s = ttnn.reshape(vae_scr["q_sdpa"], [VAE_DEC_HEADS, 1, VAE_SEQ_LEN, VAE_D_HEAD])
        k_s = ttnn.reshape(vae_scr["k_sdpa"], [VAE_DEC_HEADS, 1, VAE_SEQ_LEN, VAE_D_HEAD])
        v_s = ttnn.reshape(vae_scr["v_sdpa"], [VAE_DEC_HEADS, 1, VAE_SEQ_LEN, VAE_D_HEAD])
        attn_out = ttnn.transformer.scaled_dot_product_attention(
            q_s, k_s, v_s, is_causal=False, compute_kernel_config=COMPUTE_HIFI)

        # Reshape back to (576, 1024): (heads, 1, seq, dim) -> (1, seq, heads, dim) -> (seq, dim*heads)
        attn_out = ttnn.permute(attn_out, [1, 2, 0, 3])  # (1, 576, 16, 64)
        attn_2d = ttnn.reshape(attn_out, [VAE_SEQ_LEN, VAE_DEC_DIM])

        # Out proj + bias
        ttnn.linear(attn_2d, vae_dev["%s.proj_w" % p], bias=vae_dev["%s.proj_b" % p],
                    optional_output_tensor=vae_scr["o_proj"],
                    compute_kernel_config=COMPUTE_HIFI)

        # Residual: z_b = z_a + o_proj
        ttnn.add(vae_scr["z_a"], vae_scr["o_proj"], output_tensor=vae_scr["z_b"])

        # === MLP path: x + MLP(LN(x)) ===
        layernorm_d1024(vae_scr["z_b"], vae_dev["%s.norm2_w" % p], vae_dev["%s.norm2_b" % p],
                        scaler, mean_scale, vae_scr["normed"])

        # Fused FC1 + bias + GELU
        ttnn.linear(vae_scr["normed"], vae_dev["%s.fc1_w" % p],
                    bias=vae_dev["%s.fc1_b" % p], activation="gelu",
                    optional_output_tensor=vae_scr["gelu"],
                    compute_kernel_config=COMPUTE_HIFI)

        # FC2 + bias
        ttnn.linear(vae_scr["gelu"], vae_dev["%s.fc2_w" % p],
                    bias=vae_dev["%s.fc2_b" % p],
                    optional_output_tensor=vae_scr["fc2"],
                    compute_kernel_config=COMPUTE_HIFI)

        # Residual: z_a = z_b + fc2
        ttnn.add(vae_scr["z_b"], vae_scr["fc2"], output_tensor=vae_scr["z_a"])

    # Final LN
    layernorm_d1024(vae_scr["z_a"], vae_dev["dec_norm_w"], vae_dev["dec_norm_b"],
                    scaler, mean_scale, vae_scr["normed"])

    # Predictor: (576, 1024) @ (1024, 1200) + bias
    ttnn.linear(vae_scr["normed"], vae_dev["pred_w"], bias=vae_dev["pred_b"],
                optional_output_tensor=vae_scr["pred_out"],
                compute_kernel_config=COMPUTE_HIFI)
    return vae_scr["pred_out"]

def vae_decode_device(z_flat, vae_dev, vae_scr, tt_device, scaler, mean_scale):
    """Decode latent on TT device. 12 transformer blocks with 2D RoPE.
    z_flat: (576, 16) torch bf16 - flattened latent (already divided by scaling factor).
    Returns: (576, 1200) torch float - patch predictions for unpatchify on host.
    """
    # Pad latent from 16 to 32 cols and copy into pre-allocated buffer
    z_padded = torch.zeros(VAE_SEQ_LEN, 32, dtype=torch.bfloat16)
    z_padded[:, :VAE_LATENT_DIM] = z_flat.to(torch.bfloat16)
    z_host = ttnn.from_torch(z_padded, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    ttnn.copy_host_to_device_tensor(z_host, vae_scr["vae_input"])

    vae_decode_forward(vae_dev, vae_scr, scaler, mean_scale)

    result = readback_torch(vae_scr["pred_out"]).float()
    return result[:, :VAE_PATCH_DIM]  # strip tile padding (1216 -> 1200)

def vae_unpatchify(patches):
    """Convert VAE patch predictions to image.
    patches: (576, 1200) float tensor.
    Returns: (1, 3, 360, 640) float tensor in [-1, 1].
    """
    x = patches.reshape(1, VAE_SEQ_H, VAE_SEQ_W, VAE_PATCH_DIM)
    x = x.permute(0, 3, 1, 2)  # (1, 1200, 18, 32)
    x = x.reshape(1, 3, VAE_PATCH_SIZE_VAE, VAE_PATCH_SIZE_VAE, VAE_SEQ_H, VAE_SEQ_W)
    x = x.permute(0, 1, 4, 2, 5, 3)  # (1, 3, 18, 20, 32, 20)
    return x.reshape(1, 3, 360, 640)

# ============================================================
# On-device unpatchify bridge: DiT output (160, 64) -> VAE input (576, 32)
# ============================================================

def build_bridge_matrices(tt_device):
    """Precompute the three matrices for on-device unpatchify.

    The DiT output (N_PATCH_PAD=160, OUT_DIM=64) contains patches in
    (h, w, p, q, c) layout where h=9, w=16, p=q=2, c=16.
    The VAE input needs (576, 32) with data in cols 0-15 and zeros in 16-31,
    scaled by 1/SCALING_FACTOR.

    This is done as: R @ input -> tmp; tmp *= mask; tmp @ C -> output
    """
    # Row permutation: (576, 160) maps each output row to the correct input row
    R = torch.zeros(VAE_SEQ_LEN, N_PATCH_PAD, dtype=torch.bfloat16)
    for r in range(VAE_SEQ_LEN):
        hp = r // 32  # h*2+p index, in [0, 18)
        wq = r % 32   # w*2+q index, in [0, 32)
        h = hp // PATCH_SIZE
        w = wq // PATCH_SIZE
        in_row = h * FRAME_W + w
        if in_row < N_PATCHES:
            R[r, in_row] = 1.0

    # Column mask: (576, 64) selects the right 16 channels per row, with scaling
    inv_scale = 1.0 / SCALING_FACTOR
    mask = torch.zeros(VAE_SEQ_LEN, OUT_DIM, dtype=torch.bfloat16)
    for r in range(VAE_SEQ_LEN):
        hp = r // 32
        wq = r % 32
        p = hp % PATCH_SIZE
        q = wq % PATCH_SIZE
        col_start = p * 32 + q * IN_CHANNELS
        mask[r, col_start:col_start + IN_CHANNELS] = inv_scale

    # Column collapse: (64, 32) maps 4 col ranges into cols 0-15, zeros in 16-31
    C = torch.zeros(OUT_DIM, TILE, dtype=torch.bfloat16)
    for k in range(OUT_DIM):
        c = k % IN_CHANNELS
        C[k, c] = 1.0

    bridge = {
        "R": to_tt(R, tt_device),
        "mask": to_tt(mask, tt_device),
        "C": to_tt(C, tt_device),
    }
    return bridge


def bridge_unpatchify(dit_output, bridge, bridge_tmp, vae_input):
    """On-device unpatchify: (160, 64) -> (576, 32) via 3 ops.

    dit_output: (N_PATCH_PAD, OUT_DIM) device tensor
    bridge: dict with R, mask, C device tensors
    bridge_tmp: (VAE_SEQ_LEN, OUT_DIM) scratch device tensor
    vae_input: (VAE_SEQ_LEN, 32) output device tensor
    """
    ttnn.matmul(bridge["R"], dit_output, optional_output_tensor=bridge_tmp,
                compute_kernel_config=COMPUTE_HIFI)
    ttnn.multiply(bridge_tmp, bridge["mask"], output_tensor=bridge_tmp)
    ttnn.matmul(bridge_tmp, bridge["C"], optional_output_tensor=vae_input,
                compute_kernel_config=COMPUTE_HIFI)


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    # Multi-chip: enable fabric for inter-chip collectives
    if N_CHIPS > 1:
        ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
        tt_device = ttnn.open_mesh_device(ttnn.MeshShape(1, N_CHIPS),
                                           trace_region_size=100000000)
        _MESH_DEVICE = tt_device
    else:
        tt_device = ttnn.open_device(device_id=0, trace_region_size=100000000)
    torch.manual_seed(42)

    print("=" * 60)
    print("Oasis-500M DiT Inference on Tenstorrent")
    print("=" * 60)

    # L1 constants
    scaler = to_tt_l1(torch.ones(TILE, TILE, dtype=torch.bfloat16), tt_device)
    mean_scale = to_tt_l1(torch.full((TILE, TILE), 1.0 / D_MODEL, dtype=torch.bfloat16), tt_device)

    N_FRAMES = 2  # sliding window: T-1 context + 1 generated

    # Load VAE (CPU for encoding, device for decoding)
    print("\nLoading VAE...")
    vae = load_vae_cpu()
    print("Loading VAE decoder weights onto device...")
    vae_dev = preload_vae_decoder_weights(vae, tt_device)
    vae_scr = prealloc_vae_scratch(tt_device)

    # Encode prompt image
    prompt_path = "/tmp/sample_image_0.png"
    print("Encoding prompt image:", prompt_path)
    prompt_img = Image.open(prompt_path).convert("RGB").resize((640, 360))
    prompt_tensor = torch.tensor(list(prompt_img.getdata()), dtype=torch.float32)
    prompt_tensor = prompt_tensor.reshape(1, 360, 640, 3).permute(0, 3, 1, 2) / 255.0  # (1, 3, 360, 640)
    with torch.no_grad():
        prompt_latent = vae.encode(prompt_tensor * 2 - 1).mean * SCALING_FACTOR
    # prompt_latent: (1, H*W, C) -> reshape to (1, 1, C, H, W)
    prompt_latent = rearrange(prompt_latent, "b (h w) c -> b 1 c h w",
                              h=INPUT_H, w=INPUT_W)
    print("Prompt latent shape:", prompt_latent.shape,
          "range: [%.2f, %.2f]" % (prompt_latent.min().item(), prompt_latent.max().item()))

    # Load weights and scratch (sized for T=2)
    dev = preload_dit_weights(tt_device, n_frames=N_FRAMES)
    scr = prealloc_scratch(tt_device, n_frames=N_FRAMES)

    # Diffusion schedule
    max_noise_level = 1000
    ddim_steps = 4
    noise_range = torch.linspace(-1, max_noise_level - 1, ddim_steps + 1)
    noise_abs_max = 20
    stabilization_level = 15

    betas = sigmoid_beta_schedule(max_noise_level).float()
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)  # (1000,)

    # Load real sample actions for better quality
    actions_path = "/tmp/sample_actions_0.one_hot_actions.pt"
    if os.path.exists(actions_path):
        sample_actions = torch.load(actions_path, weights_only=True)  # (T, 25)
        # Prepend zeros for the prompt frame (reference does this)
        sample_actions = torch.cat([torch.zeros_like(sample_actions[:1]), sample_actions], dim=0)
        print("Loaded %d sample actions from %s" % (sample_actions.shape[0], actions_path))
    else:
        print("WARNING: sample actions not found at %s, using zero actions" % actions_path)
        sample_actions = torch.zeros(960, EXT_COND_DIM)
    # Frame 0 = prompt (always zero action), frame 1+ = real actions
    prompt_action = torch.zeros(EXT_COND_DIM)

    # === Pre-compute device-resident data for on-device DDIM loop ===

    # 1. Prompt frame: patch embed once, keep on device permanently
    prompt_z_pad = torch.zeros(N_PATCH_PAD, D_MODEL, dtype=torch.bfloat16)
    prompt_z_pad[:N_PATCHES] = patch_embed_host(
        prompt_latent[0, 0].unsqueeze(0), dev["x_emb_conv_w"], dev["x_emb_conv_b"])
    prompt_z_dev = to_tt(prompt_z_pad, tt_device)

    # 2. Prompt conditioning (constant across all DDIM steps)
    prompt_cond_dev = compute_cond_for_frame(
        stabilization_level - 1, prompt_action, dev, scr, tt_device)

    # 3. Round-trip weight: output_space -> input_space (patch_embed composed with unpatchify)
    # Conv weight (D_MODEL, C, ps, ps) reordered to match patchify pixel order (p, q, c)
    conv_w = dev["x_emb_conv_w"].float()  # (1024, 16, 2, 2)
    W_rt = conv_w.permute(0, 2, 3, 1).reshape(D_MODEL, OUT_DIM).T.contiguous()  # (64, 1024)
    W_rt_dev = to_tt(W_rt.to(torch.bfloat16), tt_device)
    # Bias: expand to (N_PATCH_PAD, D_MODEL) for broadcast add
    b_rt = dev["x_emb_conv_b"].float().to(torch.bfloat16)  # (D_MODEL,)
    b_rt_pad = b_rt.unsqueeze(0).expand(N_PATCH_PAD, -1).contiguous()
    b_rt_dev = to_tt(b_rt_pad, tt_device)

    # 4. Initial chunk: random noise in output space on device
    chunk_img = torch.randn(1, IN_CHANNELS, INPUT_H, INPUT_W)
    chunk_img = torch.clamp(chunk_img, -noise_abs_max, noise_abs_max)
    chunk_patches = patchify_to_output_space(chunk_img)  # (144, 64)
    chunk_pad = torch.zeros(N_PATCH_PAD, OUT_DIM, dtype=torch.bfloat16)
    chunk_pad[:N_PATCHES] = chunk_patches.to(torch.bfloat16)
    chunk_dev = to_tt(chunk_pad, tt_device)

    # Default: zero action for warmup and single-frame test
    print("Pre-computing conditioning for %d steps..." % ddim_steps)
    t_precond = time.time()
    gen_cond_per_step = precompute_gen_cond(prompt_action, ddim_steps, noise_range, dev, scr, tt_device)
    print("Pre-computed conditioning in %.1fs" % (time.time() - t_precond))

    # Pre-compute DDIM scalar coefficients as host tensors for trace-compatible updates
    CHUNK_SHAPE = (N_PATCH_PAD, OUT_DIM)
    ddim_coeffs_host = {}
    for noise_idx in reversed(range(1, ddim_steps + 1)):
        noise_level = int(noise_range[noise_idx].item())
        t_next_noise = int(noise_range[noise_idx - 1].item())
        at = float(alphas_cumprod[noise_level].item())
        an = float(alphas_cumprod[max(t_next_noise, 0)].item())
        if noise_idx == 1:
            an = 1.0
        ddim_coeffs_host[noise_idx] = {
            "at_sqrt": torch.full(CHUNK_SHAPE, at ** 0.5, dtype=torch.bfloat16),
            "1mat_sqrt": torch.full(CHUNK_SHAPE, (1 - at) ** 0.5, dtype=torch.bfloat16),
            "inv_at_sqrt": torch.full(CHUNK_SHAPE, (1.0 / at) ** 0.5, dtype=torch.bfloat16),
            "inv_sigma": torch.full(CHUNK_SHAPE, 1.0 / ((1.0 / at - 1) ** 0.5), dtype=torch.bfloat16),
            "an_sqrt": torch.full(CHUNK_SHAPE, an ** 0.5, dtype=torch.bfloat16),
            "1man_sqrt": torch.full(CHUNK_SHAPE, (1 - an) ** 0.5, dtype=torch.bfloat16),
        }
    # Convert to tilized host tensors (ready for copy_host_to_device_tensor)
    for noise_idx in ddim_coeffs_host:
        for k in ddim_coeffs_host[noise_idx]:
            ddim_coeffs_host[noise_idx][k] = ttnn.from_torch(
                ddim_coeffs_host[noise_idx][k], dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)

    COEFF_KEYS = ["at_sqrt", "1mat_sqrt", "inv_at_sqrt", "inv_sigma", "an_sqrt", "1man_sqrt"]

    # Per-step device coefficient tensors (4 sets, one per DDIM step in the trace)
    ddim_coeff_per_step = []
    for _ in range(ddim_steps):
        step_coeffs = {}
        for k in COEFF_KEYS:
            step_coeffs[k] = to_tt(torch.zeros(*CHUNK_SHAPE, dtype=torch.bfloat16), tt_device)
        ddim_coeff_per_step.append(step_coeffs)

    # Shared context conditioning (same across all DDIM steps within a frame)
    cond_context = []
    for f in range(N_FRAMES - 1):
        cond_context.append(to_tt(torch.zeros(TILE, D_MODEL, dtype=torch.bfloat16), tt_device))

    # Per-step gen-frame conditioning (different per DDIM step)
    gen_cond_step_dev = []
    for _ in range(ddim_steps):
        gen_cond_step_dev.append(to_tt(torch.zeros(TILE, D_MODEL, dtype=torch.bfloat16), tt_device))

    # Build per-step cond_traced lists: shared context + step-specific gen cond
    cond_traced_per_step = []
    for step_idx in range(ddim_steps):
        cond_traced_per_step.append(cond_context + [gen_cond_step_dev[step_idx]])

    # Context frames: T-1 frames of patch-embedded latents (updated between frames)
    context_z_dev = to_tt(torch.zeros((N_FRAMES - 1) * N_PATCH_PAD, D_MODEL, dtype=torch.bfloat16), tt_device)

    # Traced chunk buffer: the trace operates on this tensor in-place
    trace_chunk = to_tt(torch.zeros(*CHUNK_SHAPE, dtype=torch.bfloat16), tt_device)

    # Bridge matrices for on-device unpatchify
    print("Building bridge matrices...")
    bridge = build_bridge_matrices(tt_device)
    bridge_tmp = zeros_tt((VAE_SEQ_LEN, OUT_DIM), tt_device)

    def ddim_step_fn(chunk, step_coeffs, cond_list):
        """One DDIM step with explicit per-step coefficients and conditioning."""
        gen_z = ttnn.linear(chunk, W_rt_dev, bias=b_rt_dev,
                            compute_kernel_config=COMPUTE_HIFI)
        z_cur = ttnn.concat([context_z_dev, gen_z], dim=0)

        final_out = dit_forward_device(z_cur, cond_list, dev, scr, tt_device, scaler, mean_scale)

        gen_start = (N_FRAMES - 1) * N_PATCH_PAD
        v_dev = ttnn.slice(final_out, [gen_start, 0], [gen_start + N_PATCH_PAD, OUT_DIM])
        # x_start = at_sqrt * chunk - (1-at)_sqrt * v
        ttnn.multiply(chunk, step_coeffs["at_sqrt"], output_tensor=scr["ddim_tmp"])
        ttnn.multiply(v_dev, step_coeffs["1mat_sqrt"], output_tensor=scr["ddim_x_start"])
        ttnn.subtract(scr["ddim_tmp"], scr["ddim_x_start"], output_tensor=scr["ddim_x_start"])
        # x_noise = (inv_at_sqrt * chunk - x_start) / sigma
        ttnn.multiply(chunk, step_coeffs["inv_at_sqrt"], output_tensor=scr["ddim_tmp"])
        ttnn.subtract(scr["ddim_tmp"], scr["ddim_x_start"], output_tensor=scr["ddim_x_noise"])
        ttnn.multiply(scr["ddim_x_noise"], step_coeffs["inv_sigma"], output_tensor=scr["ddim_x_noise"])
        # chunk = an_sqrt * x_start + (1-an)_sqrt * x_noise
        ttnn.multiply(scr["ddim_x_start"], step_coeffs["an_sqrt"], output_tensor=scr["ddim_tmp"])
        ttnn.multiply(scr["ddim_x_noise"], step_coeffs["1man_sqrt"], output_tensor=scr["ddim_x_noise"])
        ttnn.add(scr["ddim_tmp"], scr["ddim_x_noise"], output_tensor=chunk)
        return chunk

    # === Compile: run full pipeline once to warm up (no trace yet) ===
    print("Compiling full pipeline (4 DDIM steps + bridge + VAE)...")
    t_compile = time.time()
    # Fill device tensors with valid data for compilation
    prompt_cond_host_til = ttnn.from_torch(
        readback_torch(prompt_cond_dev),
        dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    context_host = ttnn.from_torch(
        torch.cat([readback_torch(prompt_z_dev)] * (N_FRAMES - 1), dim=0),
        dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    ttnn.copy_host_to_device_tensor(context_host, context_z_dev)
    for f in range(N_FRAMES - 1):
        ttnn.copy_host_to_device_tensor(prompt_cond_host_til, cond_context[f])
    for step_idx in range(ddim_steps):
        noise_idx = ddim_steps - step_idx  # reversed: step 0 = highest noise
        for k in COEFF_KEYS:
            ttnn.copy_host_to_device_tensor(ddim_coeffs_host[noise_idx][k],
                                            ddim_coeff_per_step[step_idx][k])
        gen_cond_til = ttnn.from_torch(
            readback_torch(gen_cond_per_step[noise_idx]),
            dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
        ttnn.copy_host_to_device_tensor(gen_cond_til, gen_cond_step_dev[step_idx])
    # Run the full pipeline once (compilation pass)
    for step_idx in range(ddim_steps):
        ddim_step_fn(trace_chunk, ddim_coeff_per_step[step_idx],
                     cond_traced_per_step[step_idx])
    bridge_unpatchify(trace_chunk, bridge, bridge_tmp, vae_scr["vae_input"])
    vae_decode_forward(vae_dev, vae_scr, scaler, mean_scale)
    ttnn.synchronize_device(tt_device)
    print("Compile done in %.1fs" % (time.time() - t_compile))

    # === Decode prompt frame (untraced, before trace capture) ===
    prompt_lat = prompt_latent[0, 0]  # (C, H, W)
    z_flat = rearrange(prompt_lat.float(), "c h w -> (h w) c") / SCALING_FACTOR
    z_padded = torch.zeros(VAE_SEQ_LEN, 32, dtype=torch.bfloat16)
    z_padded[:, :VAE_LATENT_DIM] = z_flat.to(torch.bfloat16)
    z_host = ttnn.from_torch(z_padded, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    ttnn.copy_host_to_device_tensor(z_host, vae_scr["vae_input"])
    vae_decode_forward(vae_dev, vae_scr, scaler, mean_scale)
    ttnn.synchronize_device(tt_device)
    prompt_vae_result = readback_torch(vae_scr["pred_out"]).float()
    prompt_patches = prompt_vae_result[:, :VAE_PATCH_DIM]
    prompt_image = vae_unpatchify(prompt_patches)
    prompt_decoded = torch.clamp((prompt_image + 1) / 2, 0, 1).squeeze(0).permute(1, 2, 0)
    print("Prompt frame decoded via VAE")

    # === Capture single trace: 4 DDIM steps + bridge + VAE decode ===
    print("Capturing single trace...")
    t_trace = time.time()
    trace_id = ttnn.begin_trace_capture(tt_device, cq_id=0)
    for step_idx in range(ddim_steps):
        ddim_step_fn(trace_chunk, ddim_coeff_per_step[step_idx],
                     cond_traced_per_step[step_idx])
    bridge_unpatchify(trace_chunk, bridge, bridge_tmp, vae_scr["vae_input"])
    vae_decode_forward(vae_dev, vae_scr, scaler, mean_scale)
    ttnn.end_trace_capture(tt_device, trace_id, cq_id=0)
    ttnn.synchronize_device(tt_device)
    print("Single trace captured in %.1fs" % (time.time() - t_trace))

    # Host-only conditioning helper (no device allocation, safe with active traces)
    def compute_cond_host(t_scalar, action_vec):
        t_freq = timestep_embedding(torch.tensor([t_scalar]), FREQ_DIM)
        h = (t_freq[0].float() @ dev["t_emb_w0_host"]) + dev["t_emb_b0_host"]
        h = h * torch.sigmoid(h)
        cond = (h @ dev["t_emb_w2_host"]) + dev["t_emb_b2_host"]
        if action_vec is not None:
            ext = (action_vec.float() @ dev["ext_cond_w_host"]) + dev["ext_cond_b_host"]
            cond = cond + ext
        cond_pad = cond.to(torch.bfloat16).unsqueeze(0).expand(TILE, -1).contiguous()
        return ttnn.from_torch(cond_pad, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)

    def precompute_gen_cond_host(action_vec):
        """Returns dict {noise_idx: host_tilized_tensor} for all DDIM steps."""
        cond_per_step = {}
        for noise_idx in reversed(range(1, ddim_steps + 1)):
            noise_level = int(noise_range[noise_idx].item())
            cond_per_step[noise_idx] = compute_cond_host(noise_level, action_vec)
        return cond_per_step

    # Prepare host-side prompt conditioning (reusable across frames)
    prompt_cond_host = ttnn.from_torch(
        readback_torch(prompt_cond_dev),
        dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)

    def run_full_frame(chunk_in_host, label, context_cond_hosts=None, gen_cond_map=None):
        """Execute single trace: copy all inputs, run 4 DDIM + bridge + VAE, return."""
        if gen_cond_map is None:
            gen_cond_map = precompute_gen_cond_host(prompt_action)
        if context_cond_hosts is None:
            context_cond_hosts = [prompt_cond_host] * (N_FRAMES - 1)
        # Copy noise chunk
        ttnn.copy_host_to_device_tensor(chunk_in_host, trace_chunk)
        # Copy context conditioning
        for f in range(N_FRAMES - 1):
            ttnn.copy_host_to_device_tensor(context_cond_hosts[f], cond_context[f])
        # Copy per-step coefficients and gen conditioning
        for step_idx in range(ddim_steps):
            noise_idx = ddim_steps - step_idx  # reversed: step 0 = highest noise
            for k in COEFF_KEYS:
                ttnn.copy_host_to_device_tensor(ddim_coeffs_host[noise_idx][k],
                                                ddim_coeff_per_step[step_idx][k])
            ttnn.copy_host_to_device_tensor(gen_cond_map[noise_idx],
                                            gen_cond_step_dev[step_idx])
        # Single trace execution: 4 DDIM steps + bridge + VAE
        t_exec = time.time()
        ttnn.execute_trace(tt_device, trace_id, cq_id=0, blocking=False)
        ttnn.synchronize_device(tt_device)
        exec_time = time.time() - t_exec
        print("%s: %.3fs (%.1f FPS)" % (label, exec_time, 1.0 / exec_time))
        return vae_scr["pred_out"]

    chunk_host_tilized = ttnn.from_torch(chunk_pad, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    print("\n=== WARMUP (single trace: 4xDDIM + bridge + VAE) ===")
    _ = run_full_frame(chunk_host_tilized, "Warmup")
    print("\n=== TIMED SINGLE FRAME ===")
    _ = run_full_frame(chunk_host_tilized, "Timed")

    # === Generate video ===
    N_VIDEO_FRAMES = 30
    print("\n=== GENERATING %d-FRAME VIDEO (single trace) ===" % N_VIDEO_FRAMES)
    t_video_start = time.time()

    all_decoded_frames = [prompt_decoded]

    # Sliding window
    prompt_z_torch = torch.zeros(N_PATCH_PAD, D_MODEL, dtype=torch.bfloat16)
    prompt_z_torch[:N_PATCHES] = patch_embed_host(
        prompt_latent[0, 0].unsqueeze(0), dev["x_emb_conv_w"], dev["x_emb_conv_b"])
    context_window_z = [prompt_z_torch.clone() for _ in range(N_FRAMES - 1)]
    context_cond_window = [prompt_cond_host] * (N_FRAMES - 1)

    for frame_idx in range(1, N_VIDEO_FRAMES):
        t_frame = time.time()

        # Fresh noise
        chunk_img = torch.randn(1, IN_CHANNELS, INPUT_H, INPUT_W)
        chunk_img = torch.clamp(chunk_img, -noise_abs_max, noise_abs_max)
        chunk_patches = patchify_to_output_space(chunk_img)
        chunk_pad_f = torch.zeros(N_PATCH_PAD, OUT_DIM, dtype=torch.bfloat16)
        chunk_pad_f[:N_PATCHES] = chunk_patches.to(torch.bfloat16)
        chunk_host_f = ttnn.from_torch(chunk_pad_f, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)

        # Update context (host-only, no device alloc)
        context_cat = torch.cat(context_window_z, dim=0)
        context_host = ttnn.from_torch(context_cat, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
        ttnn.copy_host_to_device_tensor(context_host, context_z_dev)

        # Conditioning (host-only, no device alloc)
        action_idx = min(frame_idx, sample_actions.shape[0] - 1)
        frame_cond_host = precompute_gen_cond_host(sample_actions[action_idx])

        # Run single trace: 4 DDIM + bridge + VAE
        pred_out = run_full_frame(chunk_host_f, "Frame %d" % frame_idx,
                                  context_cond_hosts=context_cond_window,
                                  gen_cond_map=frame_cond_host)

        # Readback VAE output directly (no host bridge needed!)
        result = readback_torch(pred_out).float()
        patches = result[:, :VAE_PATCH_DIM]
        image = vae_unpatchify(patches)
        decoded = torch.clamp((image + 1) / 2, 0, 1)
        all_decoded_frames.append(decoded.squeeze(0).permute(1, 2, 0))

        # Update context window: need DiT output for patch embedding
        # The DiT output is in trace_chunk after the 4 DDIM steps (before bridge overwrites it)
        # But bridge reads trace_chunk, doesn't modify it. So trace_chunk still has the
        # denoised DiT output after bridge_unpatchify.
        chunk_result = readback_torch(trace_chunk)[:N_PATCHES].float()
        gen_latent = unpatchify_host(chunk_result, PATCH_SIZE, IN_CHANNELS, FRAME_H, FRAME_W)
        new_z = torch.zeros(N_PATCH_PAD, D_MODEL, dtype=torch.bfloat16)
        new_z[:N_PATCHES] = patch_embed_host(
            gen_latent, dev["x_emb_conv_w"], dev["x_emb_conv_b"])
        context_window_z.pop(0)
        context_window_z.append(new_z)

        new_cond_entry = compute_cond_host(stabilization_level - 1, sample_actions[action_idx])
        context_cond_window.pop(0)
        context_cond_window.append(new_cond_entry)

        elapsed = time.time() - t_frame
        print("  Frame %d: %.2fs" % (frame_idx, elapsed))

    total_video = time.time() - t_video_start
    print("\nDiT+VAE generation: %.1fs for %d frames (%.2f FPS)" % (
        total_video, N_VIDEO_FRAMES - 1, (N_VIDEO_FRAMES - 1) / total_video))

    # Save as individual PNGs
    video_dir = "/tmp/oasis_video_%dstep_T%d" % (ddim_steps, N_FRAMES)
    os.makedirs(video_dir, exist_ok=True)
    for i, frame_hwc in enumerate(all_decoded_frames):
        frame_rgb = (frame_hwc * 255).byte().numpy()
        Image.fromarray(frame_rgb).save("%s/frame_%04d.png" % (video_dir, i))
    print("Saved %d frames to %s/" % (len(all_decoded_frames), video_dir))

    # Save first generated frame
    first_gen = (all_decoded_frames[1] * 255).byte().numpy()
    Image.fromarray(first_gen).save("/tmp/oasis_frame.png")

    # Stitch frames into mp4 with ffmpeg
    mp4_path = "%s/output.mp4" % video_dir
    ffmpeg_cmd = [
        "ffmpeg", "-y",
        "-framerate", "20",
        "-i", "%s/frame_%%04d.png" % video_dir,
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-crf", "18",
        mp4_path
    ]
    result = subprocess.run(ffmpeg_cmd, capture_output=True, text=True)
    if result.returncode == 0:
        print("Saved video to %s" % mp4_path)
    else:
        print("ffmpeg failed: %s" % result.stderr[-200:])

    print("\nDone!")
    if isinstance(tt_device, ttnn.MeshDevice):
        ttnn.close_mesh_device(tt_device)
    else:
        ttnn.close_device(tt_device)
