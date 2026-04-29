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
from sdpa import make_sdpa_kernel
from sdpa_causal import make_sdpa_causal_kernel

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

# SDPA program config: exp_approx_mode=False disables the approximate softmax
# (default is True). Larger chunk sizes reduce bf16 round-trips in the flash
# attention inner loop. tt-metal SDPA hardcodes bf16 intermediate CBs (see
# tt-metal issues #41684/#41686), so these knobs are our only accuracy lever.
def _sdpa_cfg(q_chunk, k_chunk):
    return ttnn.SDPAProgramConfig(
        compute_with_storage_grid_size=ttnn.CoreCoord(8, 8),
        q_chunk_size=q_chunk,
        k_chunk_size=k_chunk,
        exp_approx_mode=False,
    )

# Multi-chip tensor parallelism: set to 1 for single-chip, 2+ for TP
N_CHIPS = 4

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
# TT-Lang spatial SDPA: bf16 compute with fp32 DST accumulation for correct
# multi-tile reduce behavior. Matches torch PCC ~= 1.0 vs ttnn SDPA's ~0.9998.
sdpa_spatial = make_sdpa_kernel(N_PATCH_PAD // TILE, N_PATCH_PAD // TILE,
                                D_HEAD // TILE, 1.0 / math.sqrt(D_HEAD))
# TT-Lang temporal SDPA: block-diagonal batch packing. T_PADDED=TILE rows hold
# T_PADDED/T pairs of (p,h) groups stacked vertically, with the bias mask
# masking cross-pair attention to -1e4 (effectively zero softmax weight).
# This collapses 640 wasted-padding-rows attention calls into 40 fully-packed
# calls. Bias is precomputed once (same for every super-batch).
T_PADDED = TILE
N_FRAMES_TP = 2  # Sliding window: 1 context + 1 generated frame
PACK_GROUPS_PER_BATCH = T_PADDED // N_FRAMES_TP  # 16 (p,h) pairs per super-batch
TEMPORAL_BATCH_PACKED = (N_PATCH_PAD * N_HEADS_TP) // PACK_GROUPS_PER_BATCH  # 40
sdpa_temporal_causal = make_sdpa_causal_kernel(
    sq_tiles=T_PADDED // TILE, skv_tiles=T_PADDED // TILE,
    head_tiles=D_HEAD // TILE, scale_val=1.0 / math.sqrt(D_HEAD),
    total_heads=TEMPORAL_BATCH_PACKED)


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

def to_tt_l1_f32(t, device):
    return ttnn.from_torch(t, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT,
                           device=device, memory_config=ttnn.L1_MEMORY_CONFIG,
                           **_mesh_kwargs(device))

def to_tt_f32(t, device):
    return ttnn.from_torch(t, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT,
                           device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG,
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

def zeros_tt_f32(shape, device):
    return ttnn.from_torch(torch.zeros(shape, dtype=torch.float32),
                           dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT,
                           device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG,
                           **_mesh_kwargs(device))

def readback_torch(t):
    """Read tensor from device/mesh to torch. For mesh, reads chip 0."""
    if _MESH_DEVICE is not None:
        return ttnn.to_torch(t, mesh_composer=ttnn.ConcatMeshToTensor(_MESH_DEVICE, dim=0))[:t.shape[0]]
    return ttnn.to_torch(t)

# PCC dump: captured on first block-0-spatial call, written to /tmp/device_dump_block0.pt.
_PCC_DUMP = {}

def _pcc_stash(name, t):
    if _PCC_DUMP.get("done"):
        return
    _PCC_DUMP[name] = readback_torch(t).float()

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

def build_rope_device_tables(freqs_per_position, n_positions, n_patch_pad, n_heads, n_frames, tt_device, dtype=torch.bfloat16):
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

    cos_full = torch.zeros(SEQ, d_model, dtype=dtype)
    sin_full = torch.zeros(SEQ, d_model, dtype=dtype)
    for t in range(n_frames):
        start = t * n_patch_pad
        src_start = t * n_patch_pad
        n = min(n_positions - src_start, n_patch_pad)
        cos_full[start:start + n] = cos_expanded[src_start:src_start + n].to(dtype)
        sin_full[start:start + n] = sin_expanded[src_start:src_start + n].to(dtype)

    # L1 for RoPE tables: read 16x per step by each spatial/temporal sub-block
    if dtype is torch.float32:
        return to_tt_l1_f32(cos_full, tt_device), to_tt_l1_f32(sin_full, tt_device)
    return to_tt_l1(cos_full, tt_device), to_tt_l1(sin_full, tt_device)

def build_spatial_rope_device_tables(spatial_freqs, n_frames, tt_device, dtype=torch.bfloat16):
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
    cos_full = torch.zeros(SEQ, d_model_tp, dtype=dtype)
    sin_full = torch.zeros(SEQ, d_model_tp, dtype=dtype)
    for t in range(n_frames):
        start = t * N_PATCH_PAD
        cos_full[start:start + N_PATCHES] = cos_expanded.to(dtype)
        sin_full[start:start + N_PATCHES] = sin_expanded.to(dtype)

    # L1 for RoPE tables: read 16x per step by each spatial/temporal sub-block
    if dtype == torch.float32:
        return to_tt_l1_f32(cos_full, tt_device), to_tt_l1_f32(sin_full, tt_device)
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

def _find_weights_by_prefix(prefix):
    # Blob dir contains safetensors + small metadata files (README, .gitattributes).
    # Skip anything that isn't a valid safetensors file.
    blob_dir = "/root/.cache/huggingface/hub/models--Etched--oasis-500m/blobs/"
    for f in sorted(os.listdir(blob_dir)):
        path = blob_dir + f
        try:
            with safe_open(path, framework="pt") as st:
                if any(k.startswith(prefix) for k in st.keys()):
                    return path
        except Exception:
            continue
    raise FileNotFoundError("weights with prefix %r not found in %s" % (prefix, blob_dir))

def find_dit_weights():
    return _find_weights_by_prefix("blocks.")

def find_vae_weights():
    return _find_weights_by_prefix("encoder.")

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

        # Final layer weights (f32) — keep the v predictor end-to-end f32 to match
        # the f32 residual stream feeding it.
        dev["final_adaln_w_f32"] = to_tt_f32(st.get_tensor("final_layer.adaLN_modulation.1.weight").T.contiguous().float(), tt_device)
        final_adaln_b_f32 = st.get_tensor("final_layer.adaLN_modulation.1.bias").float()
        dev["final_adaln_b_f32"] = to_tt_f32(final_adaln_b_f32.unsqueeze(0).expand(TILE, -1).contiguous(), tt_device)
        dev["final_linear_w_f32"] = to_tt_f32(st.get_tensor("final_layer.linear.weight").T.contiguous().float(), tt_device)
        dev["final_linear_b_f32"] = to_tt_f32(expand_bias(st.get_tensor("final_layer.linear.bias").float(), N_PATCH_PAD * n_frames), tt_device)

        # Per-block weights
        for i in range(N_BLOCKS):
            for prefix in ["s", "t"]:
                p = "blocks.%d.%s" % (i, prefix)

                # adaLN modulation (f32) — shift/scale/gate slices stay f32 for
                # downstream modulate/gate ops.
                adaln_w_f32 = st.get_tensor("%s_adaLN_modulation.1.weight" % p).T.contiguous().float()
                adaln_b_f32 = st.get_tensor("%s_adaLN_modulation.1.bias" % p).float()
                dev["%s.adaln_w_f32" % p] = to_tt_f32(adaln_w_f32, tt_device)
                dev["%s.adaln_b_f32" % p] = to_tt_f32(adaln_b_f32.unsqueeze(0).expand(TILE, -1).contiguous(), tt_device)

                # Combined QKV + QK_swap (f32): (1024, 5120) via single ttnn.matmul.
                # Layout: [Q | K | V | Q_swap | K_swap] each 1024 cols. Q_swap/K_swap
                # are read by the rope kernels along with Q/K.
                qkv_full_w_f32 = st.get_tensor("%s_attn.to_qkv.weight" % p).T.contiguous().float()
                q32 = qkv_full_w_f32[:, :D_MODEL]
                k32 = qkv_full_w_f32[:, D_MODEL:2*D_MODEL]
                v32 = qkv_full_w_f32[:, 2*D_MODEL:]
                qkv_full_w_f32 = torch.cat([q32, k32, v32,
                                            swap_adjacent_columns(q32),
                                            swap_adjacent_columns(k32)], dim=1)
                # TP: interleave heads so ShardTensorToMesh(dim=1) gives each chip its heads
                if N_CHIPS > 1:
                    qkv_full_w_f32 = interleave_qkv_for_tp(qkv_full_w_f32, N_CHIPS, D_MODEL, D_HEAD)
                    dev["%s.qkv_full_w_f32" % p] = ttnn.from_torch(
                        qkv_full_w_f32, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT,
                        device=tt_device, memory_config=ttnn.DRAM_MEMORY_CONFIG,
                        mesh_mapper=ttnn.ShardTensorToMesh(tt_device, dim=1))
                else:
                    dev["%s.qkv_full_w_f32" % p] = to_tt_f32(qkv_full_w_f32, tt_device)

                # Output projection (f32): row-parallel (shard input dim)
                SEQ = N_PATCH_PAD * n_frames
                out_w_f32 = st.get_tensor("%s_attn.to_out.weight" % p).T.contiguous().float()
                out_b_f32 = st.get_tensor("%s_attn.to_out.bias" % p).float()
                if N_CHIPS > 1:
                    dev["%s.out_w_f32" % p] = ttnn.from_torch(
                        out_w_f32, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT,
                        device=tt_device, memory_config=ttnn.DRAM_MEMORY_CONFIG,
                        mesh_mapper=ttnn.ShardTensorToMesh(tt_device, dim=0))
                else:
                    dev["%s.out_w_f32" % p] = to_tt_f32(out_w_f32, tt_device)
                dev["%s.out_b_f32" % p] = to_tt_f32(expand_bias(out_b_f32, SEQ), tt_device)

                # MLP (f32): fc1 column-parallel (shard output dim),
                # fc2 row-parallel (shard input dim)
                fc1_w_f32 = st.get_tensor("%s_mlp.fc1.weight" % p).T.contiguous().float()
                fc1_b_f32 = st.get_tensor("%s_mlp.fc1.bias" % p).float()
                fc2_w_f32 = st.get_tensor("%s_mlp.fc2.weight" % p).T.contiguous().float()
                fc2_b_f32 = st.get_tensor("%s_mlp.fc2.bias" % p).float()
                if N_CHIPS > 1:
                    dev["%s.fc1_w_f32" % p] = ttnn.from_torch(
                        fc1_w_f32, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT,
                        device=tt_device, memory_config=ttnn.DRAM_MEMORY_CONFIG,
                        mesh_mapper=ttnn.ShardTensorToMesh(tt_device, dim=1))
                    dev["%s.fc1_b_f32" % p] = ttnn.from_torch(
                        expand_bias(fc1_b_f32, SEQ), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT,
                        device=tt_device, memory_config=ttnn.DRAM_MEMORY_CONFIG,
                        mesh_mapper=ttnn.ShardTensorToMesh(tt_device, dim=1))
                    dev["%s.fc2_w_f32" % p] = ttnn.from_torch(
                        fc2_w_f32, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT,
                        device=tt_device, memory_config=ttnn.DRAM_MEMORY_CONFIG,
                        mesh_mapper=ttnn.ShardTensorToMesh(tt_device, dim=0))
                else:
                    dev["%s.fc1_w_f32" % p] = to_tt_f32(fc1_w_f32, tt_device)
                    dev["%s.fc1_b_f32" % p] = to_tt_f32(expand_bias(fc1_b_f32, SEQ), tt_device)
                    dev["%s.fc2_w_f32" % p] = to_tt_f32(fc2_w_f32, tt_device)
                dev["%s.fc2_b_f32" % p] = to_tt_f32(expand_bias(fc2_b_f32, SEQ), tt_device)

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
        # f32 copies for end-to-end f32 spatial rope+sdpa accuracy path
        dev["spatial_cos_f32"], dev["spatial_sin_perm_f32"] = \
            build_spatial_rope_device_tables(SPATIAL_ROPE_FREQS, n_frames, tt_device,
                                             dtype=torch.float32)

        # Device-side temporal RoPE tables: per-frame freqs broadcast to all patches
        # For T frames: frame t uses temporal_freqs[t], broadcast across N_PATCH_PAD rows
        temporal_per_frame = TEMPORAL_ROPE_FREQS[:n_frames]  # (T, 64)
        # Expand each frame's freqs to fill N_PATCH_PAD rows
        temporal_expanded = temporal_per_frame.unsqueeze(1).expand(
            n_frames, N_PATCH_PAD, TEMPORAL_ROPE_DIM).reshape(
            n_frames * N_PATCH_PAD, TEMPORAL_ROPE_DIM)  # (SEQ, 64)
        dev["temporal_cos"], dev["temporal_sin_perm"] = \
            build_rope_device_tables(temporal_expanded, n_frames * N_PATCH_PAD,
                                     N_PATCH_PAD, N_HEADS_TP, n_frames, tt_device,
                                     dtype=torch.float32)

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
    # f32 spatial+temporal path: modulated cast to f32 feeds the f32 qkv matmul
    # whose output (qkv_full_f32) is consumed by the f32 rope+sdpa pipeline
    # (both spatial and temporal go through TT-Lang sdpa kernels).
    s["modulated_f32"] = zeros_tt_f32((SEQ, D_MODEL), tt_device)
    s["qkv_full_f32"] = zeros_tt_f32((SEQ, 5 * D_MODEL_TP), tt_device)
    # f32 (1+scale) scratch for spatial modulate; avoids a bf16 round-trip
    # before the multiply.
    s["normed_f32"] = zeros_l1_f32((SEQ, D_MODEL), tt_device)
    s["o_proj"] = zeros_l1_f32((SEQ, D_MODEL), tt_device)
    s["fc2"] = zeros_l1_f32((SEQ, D_MODEL), tt_device)
    # f32 MLP scratches for the f32 path.
    s["modulated_b_f32"] = zeros_tt_f32((SEQ, D_MODEL), tt_device)
    s["gelu_f32"] = zeros_l1_f32((SEQ, D_MLP_TP), tt_device)
    # f32 adaln scratch: linear writes here so the spatial slices
    # (shift/scale/gate) stay f32 through modulate/gate ops.
    s["adaln_out_f32"] = ttnn.from_torch(
        torch.zeros(TILE, 6 * D_MODEL, dtype=torch.float32),
        dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT,
        device=tt_device, memory_config=ttnn.L1_MEMORY_CONFIG,
        **_mesh_kwargs(tt_device))
    # Temporal SDPA: full f32 IO + f32 acc via TT-Lang sdpa_causal kernel.
    # rope_temporal writes f32 q/k/v here (kernel auto-handles f32 via DFBs).
    # L1: ~327KB each (SEQ=320 * D_MODEL_TP=256 * 4B), interleaved across cores.
    s["t_q_scratch"] = zeros_l1_f32((SEQ, D_MODEL_TP), tt_device)
    s["t_k_scratch"] = zeros_l1_f32((SEQ, D_MODEL_TP), tt_device)
    s["t_v_scratch"] = zeros_l1_f32((SEQ, D_MODEL_TP), tt_device)
    # Output for packed temporal SDPA: (TEMPORAL_BATCH_PACKED * T_PADDED, D_HEAD).
    # 16x smaller than the old per-(p,h) layout because 16 (p,h) groups share
    # one super-batch via block-diagonal mask packing. ~320KB in L1.
    s["sdpa_temp_out_f32"] = zeros_l1_f32((TEMPORAL_BATCH_PACKED * T_PADDED, D_HEAD), tt_device)
    s["sdpa_temp_attn_scratch_f32"] = ttnn.from_torch(
        torch.zeros(T_PADDED, T_PADDED, dtype=torch.float32),
        dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT,
        device=tt_device, memory_config=ttnn.L1_MEMORY_CONFIG,
        **_mesh_kwargs(tt_device))
    s["sdpa_temp_scaler_f32"] = ttnn.from_torch(
        torch.ones(TILE, TILE, dtype=torch.float32),
        dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT,
        device=tt_device, memory_config=ttnn.L1_MEMORY_CONFIG,
        **_mesh_kwargs(tt_device))
    # Block-diagonal causal bias: T_PADDED=32 rows = 16 (p,h) pairs of T=2.
    # Within pair (rows 2g, 2g+1 vs keys 2g, 2g+1): causal (k>q masked).
    # Across pairs: -1e4 to suppress cross-pair attention via softmax.
    # Same mask for every super-batch; loaded once at startup.
    _temp_bias = torch.zeros(T_PADDED, T_PADDED, dtype=torch.float32)
    for _qi in range(T_PADDED):
        for _kj in range(T_PADDED):
            if (_qi // n_frames) != (_kj // n_frames):
                _temp_bias[_qi, _kj] = -1e4
            elif (_kj % n_frames) > (_qi % n_frames):
                _temp_bias[_qi, _kj] = -1e4
    s["sdpa_temp_bias_f32"] = ttnn.from_torch(
        _temp_bias, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT,
        device=tt_device, memory_config=ttnn.L1_MEMORY_CONFIG,
        **_mesh_kwargs(tt_device))
    # f32 spatial SDPA: rope writes f32 Q/K/V directly; sdpa_spatial reads them.
    # L1: ~320KB each (8 * 160 * 64 * 4B). Saves a DRAM round-trip per spatial sub-block.
    BATCH_S = n_frames * N_HEADS_TP
    s["q_sdpa_f32"] = zeros_l1_f32((BATCH_S * N_PATCH_PAD, D_HEAD), tt_device)
    s["k_sdpa_f32"] = zeros_l1_f32((BATCH_S * N_PATCH_PAD, D_HEAD), tt_device)
    s["v_sdpa_f32"] = zeros_l1_f32((BATCH_S * N_PATCH_PAD, D_HEAD), tt_device)
    s["sdpa_heads_out_f32"] = zeros_l1_f32((BATCH_S * N_PATCH_PAD, D_HEAD), tt_device)
    s["sdpa_attn_scratch_f32"] = ttnn.from_torch(
        torch.zeros(N_PATCH_PAD, N_PATCH_PAD, dtype=torch.float32),
        dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT,
        device=tt_device, memory_config=ttnn.L1_MEMORY_CONFIG,
        **_mesh_kwargs(tt_device))
    s["sdpa_scaler_f32"] = ttnn.from_torch(
        torch.ones(TILE, TILE, dtype=torch.float32),
        dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT,
        device=tt_device, memory_config=ttnn.L1_MEMORY_CONFIG,
        **_mesh_kwargs(tt_device))
    # Residual-carrying tensor for the post-attn add.
    s["z_scratch"] = zeros_l1_f32((SEQ, D_MODEL), tt_device)
    # f32 final layer scratch for the f32 v predictor.
    s["final_adaln_f32"] = ttnn.from_torch(
        torch.zeros(TILE, 2 * D_MODEL, dtype=torch.float32),
        dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT,
        device=tt_device, memory_config=ttnn.L1_MEMORY_CONFIG,
        **_mesh_kwargs(tt_device))
    s["final_out_f32"] = zeros_l1_f32((SEQ, OUT_DIM), tt_device)
    # Pre-allocated SiLU output buffers per frame (L1 for fast access, read 32x/step).
    # f32 so silu runs end-to-end f32 and feeds f32 adaln without a typecast.
    for f in range(n_frames):
        s["silu_out_%d" % f] = to_tt_l1_f32(torch.zeros(TILE, D_MODEL, dtype=torch.float32), tt_device)
    s["n_frames"] = n_frames
    # DDIM arithmetic scratch (N_PATCH_PAD, OUT_DIM)
    # DDIM arithmetic in f32: v_dev (final layer output) is f32, and the
    # cascading chunk feeds the next frame's context via patch_embed. Keeping
    # the noise/x_start composition in f32 avoids re-quantizing v at every step.
    s["ddim_x_start"] = zeros_l1_f32((N_PATCH_PAD, OUT_DIM), tt_device)
    s["ddim_x_noise"] = zeros_l1_f32((N_PATCH_PAD, OUT_DIM), tt_device)
    s["ddim_tmp"] = zeros_l1_f32((N_PATCH_PAD, OUT_DIM), tt_device)
    # bf16 chunk for the bridge_unpatchify path (bridge weights stay bf16).
    s["bridge_chunk_bf16"] = zeros_l1((N_PATCH_PAD, OUT_DIM), tt_device)
    elapsed = time.time() - t0
    print("Pre-allocated %d scratch tensors (T=%d) in %.1fs" % (len(s) - 1, n_frames, elapsed))
    return s

# ============================================================
# DiT forward pass
# ============================================================

def patch_embed_host(x_latent, conv_w, conv_b):
    """x_latent: (B, C, H, W) = (1, 16, 18, 32) -> (N_PATCHES, D_MODEL).
    Returns f32 so the cascading latent stays high-precision into z_cur."""
    x = F.conv2d(x_latent.float(), conv_w.float(), conv_b.float(), stride=PATCH_SIZE)
    x = rearrange(x, "b d h w -> b h w d")
    x = x.reshape(1, N_PATCHES, D_MODEL)
    return x.squeeze(0)  # (144, 1024) f32

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

def build_per_frame_adaln(silu_cond_list, prefix, dev, scr, tt_device, dtype_suffix=""):
    """Compute adaLN params on DEVICE using ttnn.matmul.
    silu_cond_list: list of T pre-computed SiLU(cond) device tensors, each (TILE, D_MODEL).
    dtype_suffix: "" for bf16, "_f32" to use the f32 weights/scratch (spatial path).
    Returns: packed (SEQ, 6*D_MODEL) device tensor.
    """
    T = len(silu_cond_list)
    N_REPEAT = N_PATCH_PAD // TILE  # 5
    out_scr = scr["adaln_out%s" % dtype_suffix]

    per_frame_expanded = []
    for silu_cond in silu_cond_list:
        ttnn.linear(silu_cond, dev["%s.adaln_w%s" % (prefix, dtype_suffix)],
                    bias=dev["%s.adaln_b%s" % (prefix, dtype_suffix)],
                    optional_output_tensor=out_scr,
                    compute_kernel_config=COMPUTE_HIFI)
        expanded = ttnn.concat([out_scr] * N_REPEAT, dim=0)
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

    _pcc_active = (prefix == "blocks.0.s" and not _PCC_DUMP.get("done"))
    if _pcc_active:
        for t_idx, sc_t in enumerate(silu_cond_list):
            _pcc_stash("silu_cond_%d" % t_idx, sc_t)

    # Compute per-frame adaLN params (SiLU already applied to cond) - packed (SEQ, 6*D_MODEL)
    # Both spatial and temporal use f32 weights/scratch so the shift/scale/gate
    # slices stay f32 through modulate/gate ops.
    adaln_packed = build_per_frame_adaln(silu_cond_list, prefix, dev, scr, tt_device,
                                         dtype_suffix="_f32")
    if _pcc_active:
        _pcc_stash("adaln_packed", adaln_packed)
    _timer.mark("adaln")
    if _do_dev_profile:
        ttnn.synchronize_device(tt_device)
        _t1 = _dt.time(); print("      %s adaln: %.1fms" % (prefix, (_t1 - _t0) * 1000)); _t0 = _t1

    # LayerNorm + adaLN modulate via ttnn ops
    # TODO: revisit TT-Lang fused_ln_adaln_d1024 kernel (0.3ms vs 0.15ms ttnn, 2x slower)
    # Was: fused_ln_adaln_d1024(x_tt, scaler, mean_scale, adaln_packed, scr["modulated"])
    normed_a = ttnn.layer_norm(x_tt, compute_kernel_config=COMPUTE_HIFI)
    shift_a = ttnn.slice(adaln_packed, [0, 0], [SEQ, D_MODEL])
    scale_a = ttnn.slice(adaln_packed, [0, D_MODEL], [SEQ, 2 * D_MODEL])
    # Unified f32 modulate: keep (1+scale)*normed + shift in f32 so the qkv
    # matmul reads fresh f32 values rather than bf16-rounded modulated.
    ttnn.add(scale_a, 1.0, output_tensor=scr["normed_f32"])
    ttnn.multiply(normed_a, scr["normed_f32"], output_tensor=scr["modulated_f32"])
    ttnn.add(scr["modulated_f32"], shift_a, output_tensor=scr["modulated_f32"])
    if _pcc_active:
        _pcc_stash("modulated_a", scr["modulated_f32"])
    _timer.mark("norm+mod")
    if _do_dev_profile:
        ttnn.synchronize_device(tt_device)
        _t1 = _dt.time(); print("      %s norm+mod: %.1fms" % (prefix, (_t1 - _t0) * 1000)); _t0 = _t1

    # Hybrid: ttnn.matmul for QKV (72-core parallel), TT-Lang for RoPE+SDPA.
    # Both paths run the qkv matmul in f32 end-to-end so the bf16 quantization
    # at the qkv boundary doesn't cap downstream PCC.
    ttnn.matmul(scr["modulated_f32"], dev["%s.qkv_full_w_f32" % prefix],
                optional_output_tensor=scr["qkv_full_f32"],
                compute_kernel_config=COMPUTE_HIFI)
    if _pcc_active:
        _pcc_stash("qkv_full", scr["qkv_full_f32"])
    if _do_dev_profile:
        ttnn.synchronize_device(tt_device)
        _t1 = _dt.time(); print("      %s qkv_matmul: %.1fms" % (prefix, (_t1 - _t0) * 1000)); _t0 = _t1

    D_MODEL_TP = D_MODEL // N_CHIPS

    if attn_type == "spatial":
        # End-to-end f32 spatial rope+sdpa: qkv matmul above already produced
        # f32 qkv_full_f32, so rope reads f32 directly and writes f32 Q/K/V to
        # the SDPA scratches. The original bf16->f32 typecast is gone.
        rope_layout_spatial(scr["qkv_full_f32"],
                           dev["spatial_cos_f32"], dev["spatial_sin_perm_f32"],
                           scr["q_sdpa_f32"], scr["k_sdpa_f32"], scr["v_sdpa_f32"])
        if _pcc_active:
            _pcc_stash("q_sdpa", scr["q_sdpa_f32"])
            _pcc_stash("k_sdpa", scr["k_sdpa_f32"])
            _pcc_stash("v_sdpa", scr["v_sdpa_f32"])
        BATCH_S = T * N_HEADS_TP
        sdpa_spatial(scr["q_sdpa_f32"], scr["k_sdpa_f32"], scr["v_sdpa_f32"],
                     scr["sdpa_scaler_f32"], scr["sdpa_attn_scratch_f32"],
                     scr["sdpa_heads_out_f32"])
        # Keep attn_2d in f32: the bf16 typecast at this boundary used to cap
        # downstream PCC since o_proj reads attn output. We feed f32 directly
        # into the f32 o_proj matmul below.
        attn_out = ttnn.reshape(scr["sdpa_heads_out_f32"], [T, N_HEADS_TP, N_PATCH_PAD, D_HEAD])
        attn_out = ttnn.permute(attn_out, [0, 2, 1, 3])
        attn_2d = ttnn.reshape(attn_out, [SEQ, D_MODEL_TP])
        if _pcc_active:
            _pcc_stash("attn_2d", attn_2d)
        _timer.mark("qkv+sdpa")
    else:
        # Temporal: full f32 IO + f32 acc via TT-Lang sdpa_causal kernel.
        # rope_temporal kernel auto-handles f32 (DFBs inherit from input).
        rope_temporal(scr["qkv_full_f32"],
                      dev["temporal_cos"], dev["temporal_sin_perm"],
                      scr["t_q_scratch"], scr["t_k_scratch"], scr["t_v_scratch"])
        if _do_dev_profile:
            ttnn.synchronize_device(tt_device)
            _t1 = _dt.time(); print("      %s rope_temporal: %.1fms" % (prefix, (_t1 - _t0) * 1000)); _t0 = _t1
        # Block-diagonal packed layout: 16 (p,h) groups × T=2 frames stack into
        # one super-batch of T_PADDED=32 rows. No pad needed (T*16 == T_PADDED).
        # Bias is precomputed block-diagonal causal mask.
        TOTAL_FLAT = N_PATCH_PAD * N_HEADS_TP * T  # 1280

        def _to_sdpa_layout(scratch_2d):
            x = ttnn.reshape(scratch_2d, [T, N_PATCH_PAD, N_HEADS_TP, D_HEAD])
            x = ttnn.permute(x, [1, 2, 0, 3])
            return ttnn.reshape(x, [TOTAL_FLAT, D_HEAD])

        q_sdpa = _to_sdpa_layout(scr["t_q_scratch"])
        k_sdpa = _to_sdpa_layout(scr["t_k_scratch"])
        v_sdpa = _to_sdpa_layout(scr["t_v_scratch"])
        if _do_dev_profile:
            ttnn.synchronize_device(tt_device)
            _t1 = _dt.time(); print("      %s sdpa_layout_fwd: %.1fms" % (prefix, (_t1 - _t0) * 1000)); _t0 = _t1
        sdpa_temporal_causal(q_sdpa, k_sdpa, v_sdpa,
                             scr["sdpa_temp_scaler_f32"],
                             scr["sdpa_temp_bias_f32"],
                             scr["sdpa_temp_attn_scratch_f32"],
                             scr["sdpa_temp_out_f32"])
        if _do_dev_profile:
            ttnn.synchronize_device(tt_device)
            _t1 = _dt.time(); print("      %s sdpa_kernel: %.1fms" % (prefix, (_t1 - _t0) * 1000)); _t0 = _t1
        # Reverse layout: (TEMPORAL_BATCH_PACKED * T_PADDED, D_HEAD) -> (T*P, H*D_HEAD)
        # 1280 rows = (P, H, T, D) directly, then permute to (T, P, H, D).
        out = ttnn.reshape(scr["sdpa_temp_out_f32"], [N_PATCH_PAD, N_HEADS_TP, T, D_HEAD])
        out = ttnn.permute(out, [2, 0, 1, 3])
        attn_2d = ttnn.reshape(out, [SEQ, D_MODEL_TP])
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
    # Phase A: O proj (row-parallel) + all_reduce + bias.
    # f32 weights so the f32 attn_2d feeds an f32 matmul end-to-end.
    o_proj = ttnn.matmul(attn_2d, dev["%s.out_w_f32" % prefix],
                         compute_kernel_config=COMPUTE_HIFI)
    if N_CHIPS > 1:
        o_proj = ttnn.all_reduce(o_proj, num_links=2)
    ttnn.add(o_proj, dev["%s.out_b_f32" % prefix], output_tensor=scr["o_proj"])
    if _pcc_active:
        _pcc_stash("o_proj", scr["o_proj"])
    gate_msa = ttnn.slice(adaln_packed, [0, 2 * D_MODEL], [SEQ, 3 * D_MODEL])

    # Phase B: gated_residual + LN + adaLN modulate via ttnn ops
    # TODO: revisit TT-Lang fused_gated_res_ln_adaln_d1024 kernel (0.8ms vs 0.15ms ttnn, 5.5x slower)
    # f32 gated residual: o_proj is f32, x_tt is f32, z_scratch is f32. Reuse
    # modulated_f32 as the gate temp (free at this point); we cannot use z_a
    # here because x_tt aliases the previous block's z_a.
    ttnn.multiply(scr["o_proj"], gate_msa, output_tensor=scr["modulated_f32"])
    ttnn.add(x_tt, scr["modulated_f32"], output_tensor=scr["z_scratch"])
    if _pcc_active:
        _pcc_stash("z_scratch_after_attn", scr["z_scratch"])
    normed = ttnn.layer_norm(scr["z_scratch"], compute_kernel_config=COMPUTE_HIFI)
    shift_b = ttnn.slice(adaln_packed, [0, 3 * D_MODEL], [SEQ, 4 * D_MODEL])
    scale_b = ttnn.slice(adaln_packed, [0, 4 * D_MODEL], [SEQ, 5 * D_MODEL])
    ttnn.add(scale_b, 1.0, output_tensor=scr["normed_f32"])
    ttnn.multiply(normed, scr["normed_f32"], output_tensor=scr["modulated_b_f32"])
    ttnn.add(scr["modulated_b_f32"], shift_b, output_tensor=scr["modulated_b_f32"])
    if _pcc_active:
        _pcc_stash("modulated_b", scr["modulated_b_f32"])
    if _do_dev_profile:
        ttnn.synchronize_device(tt_device)
        _t1 = _dt.time(); print("      %s ln+mod: %.1fms" % (prefix, (_t1 - _t0) * 1000)); _t0 = _t1

    # Phase C: Fused FC1 + bias + GELU (single ttnn.linear call).
    ttnn.linear(scr["modulated_b_f32"], dev["%s.fc1_w_f32" % prefix],
                bias=dev["%s.fc1_b_f32" % prefix],
                activation="gelu", optional_output_tensor=scr["gelu_f32"],
                compute_kernel_config=COMPUTE_HIFI)
    if _pcc_active:
        _pcc_stash("gelu", scr["gelu_f32"])
    if _do_dev_profile:
        ttnn.synchronize_device(tt_device)
        _t1 = _dt.time(); print("      %s fc1+gelu: %.1fms" % (prefix, (_t1 - _t0) * 1000)); _t0 = _t1

    # Phase D: FC2 (row-parallel) + all_reduce + bias + gated residual
    # TODO: fuse matmul+bias via ttnn.linear when N_CHIPS==1 (no all_reduce between them)
    fc2_out = ttnn.matmul(scr["gelu_f32"], dev["%s.fc2_w_f32" % prefix],
                          compute_kernel_config=COMPUTE_HIFI)
    if N_CHIPS > 1:
        fc2_out = ttnn.all_reduce(fc2_out, num_links=2)
    ttnn.add(fc2_out, dev["%s.fc2_b_f32" % prefix], output_tensor=scr["fc2"])
    gate_mlp = ttnn.slice(adaln_packed, [0, 5 * D_MODEL], [SEQ, 6 * D_MODEL])
    # Tried fusing the next 3 ops into bias_gated_residual_kernel (out = z + (fc2+b)*g);
    # correct (PCC=1.0) but ~38ms/frame slower on these shapes (DRAM-resident inputs,
    # naive 4-input read pattern beat by ttnn's optimized multi-core dispatch).
    # Revisit once Q/K/V/residual stream lives in L1-sharded memory.
    ttnn.multiply(scr["fc2"], gate_mlp, output_tensor=scr["z_a"])
    ttnn.add(scr["z_scratch"], scr["z_a"], output_tensor=scr["z_a"])
    if _pcc_active:
        _pcc_stash("fc2", scr["fc2"])
        _pcc_stash("z_a_final", scr["z_a"])
        _PCC_DUMP["done"] = True
        torch.save(_PCC_DUMP, "/tmp/device_dump_block0.pt")
        print("[PCC] dumped %d tensors to /tmp/device_dump_block0.pt" %
              sum(1 for k in _PCC_DUMP if k != "done"))
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

    # Package as (TILE, D_MODEL) device tensor - fill ALL rows for device-side broadcast.
    # f32 so silu_kernel runs in f32 end-to-end (input -> sigmoid -> multiply).
    cond_pad = cond.float().unsqueeze(0).expand(TILE, -1).contiguous()
    return to_tt_f32(cond_pad, tt_device)

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

    if not _PCC_DUMP.get("done"):
        _pcc_stash("z_cur_input", z_cur)
        for t_idx, c in enumerate(cond_list):
            _pcc_stash("cond_pre_silu_%d" % t_idx, c)

    if profile_step:
        import time as _t
        _pt = _t.time
        ttnn.synchronize_device(tt_device)

    # Pre-compute SiLU(cond) once per step, reuse across all 32 sub-blocks + final layer.
    # cond_list is f32 device tensor; silu_out is f32; no intermediate typecast.
    silu_cond_list = []
    for t_idx in range(T):
        silu_out = scr["silu_out_%d" % t_idx]
        silu_kernel(cond_list[t_idx], silu_out)
        silu_cond_list.append(silu_out)

    if profile_step:
        ttnn.synchronize_device(tt_device)
        print("  [PROFILE] silu: %.1fms" % ((_pt() - _pt.__self__) if False else 0))

    # z_cur is already f32 (gen_z and context_z_dev are both f32 now).

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
    # Final layer: per-frame adaLN in f32 (silu_cond_list is now f32).
    N_REPEAT = N_PATCH_PAD // TILE  # 5
    per_frame_final = []
    for t_idx in range(T):
        final_raw = ttnn.linear(silu_cond_list[t_idx], dev["final_adaln_w_f32"],
                                bias=dev["final_adaln_b_f32"],
                                optional_output_tensor=scr["final_adaln_f32"],
                                compute_kernel_config=COMPUTE_HIFI)
        expanded = ttnn.concat([final_raw] * N_REPEAT, dim=0)
        per_frame_final.append(expanded)
    if T == 1:
        full_final = per_frame_final[0]
    else:
        full_final = ttnn.concat(per_frame_final, dim=0)

    shift_tt = ttnn.slice(full_final, [0, 0], [SEQ, D_MODEL])
    scale_tt = ttnn.slice(full_final, [0, D_MODEL], [SEQ, 2 * D_MODEL])

    # f32 LN + modulate + linear + bias. z_cur stays f32 end-to-end.
    normed_f = ttnn.layer_norm(z_cur, compute_kernel_config=COMPUTE_HIFI)
    ttnn.add(scale_tt, 1.0, output_tensor=scr["normed_f32"])
    ttnn.multiply(normed_f, scr["normed_f32"], output_tensor=scr["modulated_f32"])
    ttnn.add(scr["modulated_f32"], shift_tt, output_tensor=scr["modulated_f32"])
    final_out = ttnn.matmul(scr["modulated_f32"], dev["final_linear_w_f32"],
                            compute_kernel_config=COMPUTE_HIFI)
    ttnn.add(final_out, dev["final_linear_b_f32"], output_tensor=scr["final_out_f32"])
    result = scr["final_out_f32"]
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
            q_s, k_s, v_s, is_causal=False, compute_kernel_config=COMPUTE_HIFI,
            program_config=_sdpa_cfg(TILE, TILE))

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
        ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D_RING, ttnn.FabricReliabilityMode.RELAXED_INIT)
        tt_device = ttnn.open_mesh_device(ttnn.MeshShape(1, N_CHIPS),
                                           trace_region_size=500000000)
        _MESH_DEVICE = tt_device
    else:
        tt_device = ttnn.open_device(device_id=0, trace_region_size=500000000)
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
    ddim_steps = 12
    noise_range = torch.linspace(-1, max_noise_level - 1, ddim_steps + 1)
    noise_abs_max = 20
    stabilization_level = 15

    betas = sigmoid_beta_schedule(max_noise_level).float()
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)  # (1000,)

    # Load recorded sample actions, mask to walk/camera only, time-stretch 2x
    # with halved camera deltas so the same motion plays out over 2x as many
    # frames. Action layout: 11 = forward (walk), 15 = cameraX, 16 = cameraY.
    N_SYNTH_FRAMES = 150
    actions_path = "/tmp/sample_actions_0.one_hot_actions.pt"
    raw_actions = torch.load(actions_path, weights_only=True).float()  # (T, EXT_COND_DIM)
    keep_mask = torch.zeros(EXT_COND_DIM)
    keep_mask[11] = 1.0  # forward / walk
    keep_mask[15] = 1.0  # cameraX (pitch)
    keep_mask[16] = 1.0  # cameraY (yaw)
    raw_actions = raw_actions * keep_mask
    stretched = raw_actions.repeat_interleave(2, dim=0)
    stretched[:, 15] *= 0.5
    stretched[:, 16] *= 0.5
    # Splice (not overwrite) a walk block into the stretched stream so the
    # surrounding camera motion is shifted later, not clobbered.
    INSERT_AT = 60   # output frame index where walk starts
    WALK_LEN = 20
    walk_block = torch.zeros(WALK_LEN, EXT_COND_DIM)
    walk_block[:, 11] = 1.0
    ins = INSERT_AT - 1  # convert from output-frame to post-prompt-array index
    seq = torch.cat([stretched[:ins], walk_block, stretched[ins:]], dim=0)
    sample_actions = torch.zeros(N_SYNTH_FRAMES, EXT_COND_DIM)
    n_use = min(seq.shape[0], N_SYNTH_FRAMES - 1)
    sample_actions[1:1 + n_use] = seq[:n_use]
    print("Loaded %s: %d->%d stretched, +%d walk@%d, -> %d frames used" % (
        actions_path, raw_actions.shape[0], stretched.shape[0],
        WALK_LEN, INSERT_AT, n_use))
    prompt_action = torch.zeros(EXT_COND_DIM)

    # === Pre-compute device-resident data for on-device DDIM loop ===

    # 1. Prompt frame: patch embed once, keep on device permanently (f32 so
    # the cascading latent stays high-precision into context_z_dev / z_cur).
    prompt_z_pad = torch.zeros(N_PATCH_PAD, D_MODEL, dtype=torch.float32)
    prompt_z_pad[:N_PATCHES] = patch_embed_host(
        prompt_latent[0, 0].unsqueeze(0), dev["x_emb_conv_w"], dev["x_emb_conv_b"])
    prompt_z_dev = to_tt_f32(prompt_z_pad, tt_device)

    # 2. Prompt conditioning (constant across all DDIM steps)
    prompt_cond_dev = compute_cond_for_frame(
        stabilization_level - 1, prompt_action, dev, scr, tt_device)

    # 3. Round-trip weight: output_space -> input_space (patch_embed composed with unpatchify)
    # Conv weight (D_MODEL, C, ps, ps) reordered to match patchify pixel order (p, q, c).
    # f32 to keep the gen_z latent fed into z_cur at full precision (was bf16 → f32
    # cast inside dit_forward_device, which threw away ~3 bits per frame in the
    # cascading context window).
    conv_w = dev["x_emb_conv_w"].float()  # (1024, 16, 2, 2)
    W_rt = conv_w.permute(0, 2, 3, 1).reshape(D_MODEL, OUT_DIM).T.contiguous()  # (64, 1024)
    W_rt_dev = to_tt_f32(W_rt, tt_device)
    b_rt = dev["x_emb_conv_b"].float()  # (D_MODEL,)
    b_rt_pad = b_rt.unsqueeze(0).expand(N_PATCH_PAD, -1).contiguous()
    b_rt_dev = to_tt_f32(b_rt_pad, tt_device)

    # 4. Initial chunk: random noise in output space on device (f32 to match
    # the f32 trace_chunk).
    chunk_img = torch.randn(1, IN_CHANNELS, INPUT_H, INPUT_W)
    chunk_img = torch.clamp(chunk_img, -noise_abs_max, noise_abs_max)
    chunk_patches = patchify_to_output_space(chunk_img)  # (144, 64)
    chunk_pad = torch.zeros(N_PATCH_PAD, OUT_DIM, dtype=torch.float32)
    chunk_pad[:N_PATCHES] = chunk_patches.float()
    chunk_dev = to_tt_f32(chunk_pad, tt_device)

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
            "at_sqrt": torch.full(CHUNK_SHAPE, at ** 0.5, dtype=torch.float32),
            "1mat_sqrt": torch.full(CHUNK_SHAPE, (1 - at) ** 0.5, dtype=torch.float32),
            "inv_at_sqrt": torch.full(CHUNK_SHAPE, (1.0 / at) ** 0.5, dtype=torch.float32),
            "inv_sigma": torch.full(CHUNK_SHAPE, 1.0 / ((1.0 / at - 1) ** 0.5), dtype=torch.float32),
            "an_sqrt": torch.full(CHUNK_SHAPE, an ** 0.5, dtype=torch.float32),
            "1man_sqrt": torch.full(CHUNK_SHAPE, (1 - an) ** 0.5, dtype=torch.float32),
        }
    # Convert to tilized host tensors (ready for copy_host_to_device_tensor)
    for noise_idx in ddim_coeffs_host:
        for k in ddim_coeffs_host[noise_idx]:
            ddim_coeffs_host[noise_idx][k] = ttnn.from_torch(
                ddim_coeffs_host[noise_idx][k], dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT)

    COEFF_KEYS = ["at_sqrt", "1mat_sqrt", "inv_at_sqrt", "inv_sigma", "an_sqrt", "1man_sqrt"]

    # Per-step device coefficient tensors (4 sets, one per DDIM step in the trace)
    ddim_coeff_per_step = []
    for _ in range(ddim_steps):
        step_coeffs = {}
        for k in COEFF_KEYS:
            step_coeffs[k] = to_tt_f32(torch.zeros(*CHUNK_SHAPE, dtype=torch.float32), tt_device)
        ddim_coeff_per_step.append(step_coeffs)

    # Shared context conditioning (same across all DDIM steps within a frame).
    # f32 so silu runs f32 and feeds f32 adaln without an intermediate typecast.
    cond_context = []
    for f in range(N_FRAMES - 1):
        cond_context.append(to_tt_f32(torch.zeros(TILE, D_MODEL, dtype=torch.float32), tt_device))

    # Per-step gen-frame conditioning (different per DDIM step)
    gen_cond_step_dev = []
    for _ in range(ddim_steps):
        gen_cond_step_dev.append(to_tt_f32(torch.zeros(TILE, D_MODEL, dtype=torch.float32), tt_device))

    # Build per-step cond_traced lists: shared context + step-specific gen cond
    cond_traced_per_step = []
    for step_idx in range(ddim_steps):
        cond_traced_per_step.append(cond_context + [gen_cond_step_dev[step_idx]])

    # Context frames: T-1 frames of patch-embedded latents (updated between frames).
    # f32 to preserve the cascading latent across video frames; the host-side
    # patch_embed already runs in f32, so we just keep that precision through
    # to z_cur instead of round-tripping via bf16.
    context_z_dev = to_tt_f32(torch.zeros((N_FRAMES - 1) * N_PATCH_PAD, D_MODEL, dtype=torch.float32), tt_device)

    # Traced chunk buffer: f32 so the noise/v composition stays high precision
    # across DDIM steps and across video frames (the chunk readback feeds the
    # next frame's context via patch_embed_host).
    trace_chunk = to_tt_f32(torch.zeros(*CHUNK_SHAPE, dtype=torch.float32), tt_device)

    # Bridge matrices for on-device unpatchify
    print("Building bridge matrices...")
    bridge = build_bridge_matrices(tt_device)
    bridge_tmp = zeros_tt((VAE_SEQ_LEN, OUT_DIM), tt_device)

    def ddim_step_fn(chunk, step_coeffs, cond_list, profile=False):
        """One DDIM step with explicit per-step coefficients and conditioning."""
        gen_z = ttnn.linear(chunk, W_rt_dev, bias=b_rt_dev,
                            compute_kernel_config=COMPUTE_HIFI)
        z_cur = ttnn.concat([context_z_dev, gen_z], dim=0)

        final_out = dit_forward_device(z_cur, cond_list, dev, scr, tt_device, scaler, mean_scale,
                                       profile_step=profile)

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
        readback_torch(prompt_cond_dev).float(),
        dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT)
    context_host = ttnn.from_torch(
        torch.cat([readback_torch(prompt_z_dev)] * (N_FRAMES - 1), dim=0).float(),
        dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT)
    ttnn.copy_host_to_device_tensor(context_host, context_z_dev)
    for f in range(N_FRAMES - 1):
        ttnn.copy_host_to_device_tensor(prompt_cond_host_til, cond_context[f])
    for step_idx in range(ddim_steps):
        noise_idx = ddim_steps - step_idx  # reversed: step 0 = highest noise
        for k in COEFF_KEYS:
            ttnn.copy_host_to_device_tensor(ddim_coeffs_host[noise_idx][k],
                                            ddim_coeff_per_step[step_idx][k])
        gen_cond_til = ttnn.from_torch(
            readback_torch(gen_cond_per_step[noise_idx]).float(),
            dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT)
        ttnn.copy_host_to_device_tensor(gen_cond_til, gen_cond_step_dev[step_idx])
    # Run the full pipeline once (compilation pass)
    for step_idx in range(ddim_steps):
        ddim_step_fn(trace_chunk, ddim_coeff_per_step[step_idx],
                     cond_traced_per_step[step_idx])
    ttnn.typecast(trace_chunk, ttnn.bfloat16, output_tensor=scr["bridge_chunk_bf16"])
    bridge_unpatchify(scr["bridge_chunk_bf16"], bridge, bridge_tmp, vae_scr["vae_input"])
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
    PROFILE_NO_TRACE = False  # disable trace + measure each component with sync
    if PROFILE_NO_TRACE:
        print("PROFILE MODE: trace disabled, will measure per-component")
        trace_id = None
    else:
        print("Capturing single trace...")
        t_trace = time.time()
        trace_id = ttnn.begin_trace_capture(tt_device, cq_id=0)
        for step_idx in range(ddim_steps):
            ddim_step_fn(trace_chunk, ddim_coeff_per_step[step_idx],
                         cond_traced_per_step[step_idx])
        ttnn.typecast(trace_chunk, ttnn.bfloat16, output_tensor=scr["bridge_chunk_bf16"])
        bridge_unpatchify(scr["bridge_chunk_bf16"], bridge, bridge_tmp, vae_scr["vae_input"])
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
        cond_pad = cond.float().unsqueeze(0).expand(TILE, -1).contiguous()
        return ttnn.from_torch(cond_pad, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT)

    def precompute_gen_cond_host(action_vec):
        """Returns dict {noise_idx: host_tilized_tensor} for all DDIM steps."""
        cond_per_step = {}
        for noise_idx in reversed(range(1, ddim_steps + 1)):
            noise_level = int(noise_range[noise_idx].item())
            cond_per_step[noise_idx] = compute_cond_host(noise_level, action_vec)
        return cond_per_step

    # Prepare host-side prompt conditioning (reusable across frames)
    prompt_cond_host = ttnn.from_torch(
        readback_torch(prompt_cond_dev).float(),
        dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT)

    def run_full_frame(chunk_in_host, label, context_cond_hosts=None, gen_cond_map=None,
                       phase_times=None):
        """Execute single trace: copy all inputs, run 4 DDIM + bridge + VAE, return."""
        if gen_cond_map is None:
            gen_cond_map = precompute_gen_cond_host(prompt_action)
        if context_cond_hosts is None:
            context_cond_hosts = [prompt_cond_host] * (N_FRAMES - 1)
        t_h2d = time.perf_counter()
        ttnn.copy_host_to_device_tensor(chunk_in_host, trace_chunk)
        for f in range(N_FRAMES - 1):
            ttnn.copy_host_to_device_tensor(context_cond_hosts[f], cond_context[f])
        for step_idx in range(ddim_steps):
            noise_idx = ddim_steps - step_idx  # reversed: step 0 = highest noise
            for k in COEFF_KEYS:
                ttnn.copy_host_to_device_tensor(ddim_coeffs_host[noise_idx][k],
                                                ddim_coeff_per_step[step_idx][k])
            ttnn.copy_host_to_device_tensor(gen_cond_map[noise_idx],
                                            gen_cond_step_dev[step_idx])
        h2d_time = time.perf_counter() - t_h2d
        t_exec = time.perf_counter()
        if PROFILE_NO_TRACE:
            do_breakdown = not run_full_frame._profiled_once[0]
            run_full_frame._profiled_once[0] = True
            step_times = []
            for step_idx in range(ddim_steps):
                ttnn.synchronize_device(tt_device)
                t0 = time.perf_counter()
                ddim_step_fn(trace_chunk, ddim_coeff_per_step[step_idx],
                             cond_traced_per_step[step_idx],
                             profile=(do_breakdown and step_idx == 0))
                ttnn.synchronize_device(tt_device)
                step_times.append(time.perf_counter() - t0)
            t0 = time.perf_counter()
            ttnn.typecast(trace_chunk, ttnn.bfloat16, output_tensor=scr["bridge_chunk_bf16"])
            bridge_unpatchify(scr["bridge_chunk_bf16"], bridge, bridge_tmp, vae_scr["vae_input"])
            ttnn.synchronize_device(tt_device)
            bridge_time = time.perf_counter() - t0
            t0 = time.perf_counter()
            vae_decode_forward(vae_dev, vae_scr, scaler, mean_scale)
            ttnn.synchronize_device(tt_device)
            vae_time = time.perf_counter() - t0
            if phase_times is not None:
                phase_times["ddim_steps"] += sum(step_times)
                phase_times["bridge"] += bridge_time
                phase_times["vae_decode"] += vae_time
            print("  ddim_steps=[%s] bridge=%.0fms vae=%.0fms" % (
                ", ".join("%.0f" % (s * 1000) for s in step_times),
                bridge_time * 1000, vae_time * 1000))
        else:
            ttnn.execute_trace(tt_device, trace_id, cq_id=0, blocking=False)
            ttnn.synchronize_device(tt_device)
        exec_time = time.perf_counter() - t_exec
        if phase_times is not None:
            phase_times["h2d_copies"] += h2d_time
            phase_times["trace_exec"] += exec_time
        print("%s: exec=%.3fs h2d=%.3fs (%.1f FPS)" % (
            label, exec_time, h2d_time, 1.0 / (exec_time + h2d_time)))
        return vae_scr["pred_out"]

    run_full_frame._profiled_once = [False]
    chunk_host_tilized = ttnn.from_torch(chunk_pad, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT)
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
    prompt_z_torch = torch.zeros(N_PATCH_PAD, D_MODEL, dtype=torch.float32)
    prompt_z_torch[:N_PATCHES] = patch_embed_host(
        prompt_latent[0, 0].unsqueeze(0), dev["x_emb_conv_w"], dev["x_emb_conv_b"])
    context_window_z = [prompt_z_torch.clone() for _ in range(N_FRAMES - 1)]
    context_cond_window = [prompt_cond_host] * (N_FRAMES - 1)

    phase_times = {"noise_prep": 0.0, "context_h2d": 0.0, "gen_cond_host": 0.0,
                   "h2d_copies": 0.0, "trace_exec": 0.0,
                   "vae_readback": 0.0, "ctx_update": 0.0,
                   "ddim_steps": 0.0, "bridge": 0.0, "vae_decode": 0.0}

    for frame_idx in range(1, N_VIDEO_FRAMES):
        t_frame = time.perf_counter()

        t = time.perf_counter()
        chunk_img = torch.randn(1, IN_CHANNELS, INPUT_H, INPUT_W)
        chunk_img = torch.clamp(chunk_img, -noise_abs_max, noise_abs_max)
        chunk_patches = patchify_to_output_space(chunk_img)
        chunk_pad_f = torch.zeros(N_PATCH_PAD, OUT_DIM, dtype=torch.float32)
        chunk_pad_f[:N_PATCHES] = chunk_patches.float()
        chunk_host_f = ttnn.from_torch(chunk_pad_f, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT)
        phase_times["noise_prep"] += time.perf_counter() - t

        t = time.perf_counter()
        context_cat = torch.cat(context_window_z, dim=0).float()
        context_host = ttnn.from_torch(context_cat, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT)
        ttnn.copy_host_to_device_tensor(context_host, context_z_dev)
        phase_times["context_h2d"] += time.perf_counter() - t

        t = time.perf_counter()
        action_idx = min(frame_idx, sample_actions.shape[0] - 1)
        frame_cond_host = precompute_gen_cond_host(sample_actions[action_idx])
        phase_times["gen_cond_host"] += time.perf_counter() - t

        pred_out = run_full_frame(chunk_host_f, "Frame %d" % frame_idx,
                                  context_cond_hosts=context_cond_window,
                                  gen_cond_map=frame_cond_host,
                                  phase_times=phase_times)

        t = time.perf_counter()
        result = readback_torch(pred_out).float()
        patches = result[:, :VAE_PATCH_DIM]
        image = vae_unpatchify(patches)
        decoded = torch.clamp((image + 1) / 2, 0, 1)
        all_decoded_frames.append(decoded.squeeze(0).permute(1, 2, 0))
        phase_times["vae_readback"] += time.perf_counter() - t

        t = time.perf_counter()
        chunk_result = readback_torch(trace_chunk)[:N_PATCHES].float()
        gen_latent = unpatchify_host(chunk_result, PATCH_SIZE, IN_CHANNELS, FRAME_H, FRAME_W)
        new_z = torch.zeros(N_PATCH_PAD, D_MODEL, dtype=torch.float32)
        new_z[:N_PATCHES] = patch_embed_host(
            gen_latent, dev["x_emb_conv_w"], dev["x_emb_conv_b"])
        context_window_z.pop(0)
        context_window_z.append(new_z)
        new_cond_entry = compute_cond_host(stabilization_level - 1, sample_actions[action_idx])
        context_cond_window.pop(0)
        context_cond_window.append(new_cond_entry)
        phase_times["ctx_update"] += time.perf_counter() - t

        elapsed = time.perf_counter() - t_frame
        print("  Frame %d: %.2fs" % (frame_idx, elapsed))

    total_video = time.time() - t_video_start
    n_frames = N_VIDEO_FRAMES - 1
    print("\nDiT+VAE generation: %.1fs for %d frames (%.2f FPS)" % (
        total_video, n_frames, n_frames / total_video))
    print("\n=== PER-PHASE AVG (ms/frame) ===")
    total_attributed = sum(phase_times.values())
    for k, v in phase_times.items():
        avg_ms = (v / n_frames) * 1000
        pct = (v / total_attributed) * 100 if total_attributed > 0 else 0
        print("  %-15s %7.2f ms  (%5.1f%%)" % (k, avg_ms, pct))
    print("  %-15s %7.2f ms" % ("TOTAL_attrib", (total_attributed / n_frames) * 1000))
    print("  %-15s %7.2f ms" % ("WALL_per_frame", (total_video / n_frames) * 1000))

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
