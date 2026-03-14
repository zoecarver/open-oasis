"""
Triage: Compare TT pipeline vs PyTorch reference for each stage of DiT forward.
Runs a single forward pass with T=1 (simpler) and compares intermediates.
"""
import sys, types

# Stub timm before any imports that might trigger it
def _make_stub(name, parent=None):
    m = types.ModuleType(name)
    sys.modules[name] = m
    if parent is not None:
        setattr(parent, name.split('.')[-1], m)
    return m

import torch
import torch.nn as nn

_timm = _make_stub('timm')
_timm_models = _make_stub('timm.models', _timm)
_timm_models_vit = _make_stub('timm.models.vision_transformer', _timm_models)
_timm_layers = _make_stub('timm.layers', _timm)
_timm_layers_helpers = _make_stub('timm.layers.helpers', _timm_layers)

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
        x = self.drop(self.act(self.fc1(x)))
        x = self.drop(self.fc2(x))
        return x

def _to_2tuple(x):
    return (x, x) if not isinstance(x, (tuple, list)) else tuple(x)

_timm_models_vit.Mlp = _Mlp
_timm_layers_helpers.to_2tuple = _to_2tuple

import torch.nn.functional as F
import ttnn
import ttl
import math
import time
import os
from safetensors import safe_open
from safetensors.torch import load_model
from einops import rearrange, repeat

# ============================================================
# Constants (must match oasis_inference.py)
# ============================================================
TILE = 32
D_MODEL = 1024
N_HEADS = 16
D_HEAD = 64
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
EXT_COND_PAD = 32
OUT_DIM = PATCH_SIZE * PATCH_SIZE * IN_CHANNELS  # 64
D_TILES = D_MODEL // TILE

# ============================================================
# Load reference PyTorch model
# ============================================================
sys.path.insert(0, "/tmp")
from dit import DiT_models

def load_ref_model():
    model = DiT_models["DiT-S/2"]()
    blob_dir = "/root/.cache/huggingface/hub/models--Etched--oasis-500m/blobs/"
    files = sorted(os.listdir(blob_dir))
    dit_path = blob_dir + files[1]
    load_model(model, dit_path)
    model = model.eval()
    return model

# ============================================================
# TT helpers (copied from oasis_inference.py)
# ============================================================
ELEM_GRAN = 8

def make_linear_kernel(k_chunk):
    @ttl.kernel(grid="auto")
    def linear_kernel(x, w, out):
        grid_cols, _ = ttl.grid_size(dims=2)
        m_tiles = x.shape[0] // TILE
        n_tiles = w.shape[1] // TILE
        total_out = m_tiles * n_tiles
        tiles_per_core = -(-total_out // grid_cols)
        x_dfb = ttl.make_dataflow_buffer_like(x, shape=(1, k_chunk), buffer_factor=2)
        w_dfb = ttl.make_dataflow_buffer_like(w, shape=(k_chunk, ELEM_GRAN), buffer_factor=2)
        out_dfb = ttl.make_dataflow_buffer_like(out, shape=(1, ELEM_GRAN), buffer_factor=2)
        @ttl.compute()
        def compute():
            core_x, _ = ttl.core(dims=2)
            for local_idx in range(tiles_per_core):
                idx = core_x * tiles_per_core + local_idx
                if idx < total_out:
                    n_blocks = n_tiles // ELEM_GRAN
                    for nb in range(n_blocks):
                        with x_dfb.wait() as xb, w_dfb.wait() as wb, out_dfb.reserve() as ob:
                            ob.store(xb @ wb)
        @ttl.datamovement()
        def dm_read():
            core_x, _ = ttl.core(dims=2)
            for local_idx in range(tiles_per_core):
                idx = core_x * tiles_per_core + local_idx
                if idx < total_out:
                    m = idx // n_tiles
                    n_blocks = n_tiles // ELEM_GRAN
                    for nb in range(n_blocks):
                        with x_dfb.reserve() as blk:
                            tx = ttl.copy(x[m:m+1, 0:k_chunk], blk); tx.wait()
                        with w_dfb.reserve() as blk:
                            nc = nb * ELEM_GRAN
                            tx = ttl.copy(w[0:k_chunk, nc:nc+ELEM_GRAN], blk); tx.wait()
        @ttl.datamovement()
        def dm_write():
            core_x, _ = ttl.core(dims=2)
            for local_idx in range(tiles_per_core):
                idx = core_x * tiles_per_core + local_idx
                if idx < total_out:
                    m = idx // n_tiles
                    n = idx % n_tiles
                    n_blocks = n_tiles // ELEM_GRAN
                    for nb in range(n_blocks):
                        nc = nb * ELEM_GRAN
                        with out_dfb.wait() as blk:
                            tx = ttl.copy(blk, out[m:m+1, nc:nc+ELEM_GRAN]); tx.wait()
    return linear_kernel

linear_k32 = make_linear_kernel(D_TILES)

def to_tt(t, device):
    return ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                           device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)

def to_tt_l1(t, device):
    return ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                           device=device, memory_config=ttnn.L1_MEMORY_CONFIG)

def zeros_tt(shape, device):
    return to_tt(torch.zeros(shape, dtype=torch.bfloat16), device)

def expand_bias(bias_1d, seq_pad):
    return bias_1d.unsqueeze(0).expand(seq_pad, -1).contiguous().to(torch.bfloat16)

def compare(name, ref, tt_val, atol=1.0):
    """Compare reference (float) vs TT (bf16) tensors."""
    if isinstance(tt_val, torch.Tensor):
        tt_f = tt_val.float()
    else:
        tt_f = ttnn.to_torch(tt_val).float()
    ref_f = ref.float()

    # Trim to matching shape
    min_shape = [min(r, t) for r, t in zip(ref_f.shape, tt_f.shape)]
    for d in range(len(min_shape)):
        ref_f = ref_f.narrow(d, 0, min_shape[d])
        tt_f = tt_f.narrow(d, 0, min_shape[d])

    diff = (ref_f - tt_f).abs()
    max_err = diff.max().item()
    mean_err = diff.mean().item()
    ref_range = ref_f.max().item() - ref_f.min().item()
    rel_err = max_err / (ref_range + 1e-8)
    status = "OK" if max_err < atol else "HIGH"
    print("  %-35s max_err=%.4f  mean_err=%.4f  rel=%.4f  ref_range=[%.2f,%.2f]  [%s]" % (
        name, max_err, mean_err, rel_err,
        ref_f.min().item(), ref_f.max().item(), status))
    return max_err

# ============================================================
# Main comparison
# ============================================================
if __name__ == "__main__":
    torch.manual_seed(42)
    tt_device = ttnn.open_device(device_id=0)

    print("Loading reference PyTorch model...")
    ref_model = load_ref_model()

    print("Loading TT weights...")
    blob_dir = "/root/.cache/huggingface/hub/models--Etched--oasis-500m/blobs/"
    files = sorted(os.listdir(blob_dir))
    dit_path = blob_dir + files[1]

    # Create test input (T=1 for simplicity)
    x_latent = torch.randn(1, 1, IN_CHANNELS, INPUT_H, INPUT_W) * 0.5
    timesteps = torch.tensor([[500]], dtype=torch.long)
    actions = torch.zeros(1, 1, EXT_COND_DIM)

    print("\n" + "=" * 70)
    print("STAGE 1: Patch Embedding")
    print("=" * 70)

    # Reference
    x_ref = rearrange(x_latent, "b t c h w -> (b t) c h w")
    x_ref_emb = ref_model.x_embedder(x_ref)  # (1, H/2, W/2, D)
    x_ref_5d = rearrange(x_ref_emb, "(b t) h w d -> b t h w d", t=1)
    x_ref_2d = x_ref_5d[0, 0].reshape(N_PATCHES, D_MODEL)  # (144, 1024)

    # TT: patch embed on host using same weights
    conv_w = ref_model.x_embedder.proj.weight
    conv_b = ref_model.x_embedder.proj.bias
    x_tt_host = F.conv2d(x_ref.float(), conv_w.float(), conv_b.float(), stride=PATCH_SIZE)
    x_tt_host = rearrange(x_tt_host, "b d h w -> b h w d").reshape(1, N_PATCHES, D_MODEL)[0]

    compare("patch_embed", x_ref_2d, x_tt_host, atol=0.01)

    print("\n" + "=" * 70)
    print("STAGE 2: Timestep + External Conditioning")
    print("=" * 70)

    # Reference
    t_flat = rearrange(timesteps, "b t -> (b t)")
    c_ref = ref_model.t_embedder(t_flat)  # (1, 1024)
    c_ref = rearrange(c_ref, "(b t) d -> b t d", t=1)
    c_ref += ref_model.external_cond(actions)
    c_ref_1d = c_ref[0, 0]  # (1024,)

    # TT: replicate timestep embedding
    from oasis_inference import timestep_embedding, make_linear_bias_kernel, make_linear_kernel
    from oasis_inference import silu_kernel, make_layernorm_kernel, adaln_modulate_kernel
    from oasis_inference import gelu_approx_kernel, gated_residual_kernel, make_linear_accum_kernel

    linear_bias_k8 = make_linear_bias_kernel(FREQ_DIM // TILE, n_chunk=4)
    linear_bias_k32 = make_linear_bias_kernel(D_TILES, n_chunk=4)
    linear_k1 = make_linear_kernel(1)
    linear_accum_k32_4 = make_linear_accum_kernel(D_TILES, 4)
    layernorm_d1024 = make_layernorm_kernel(D_TILES)

    # Load TT weights for conditioning
    with safe_open(dit_path, framework="pt") as st:
        t_emb_w0 = to_tt(st.get_tensor("t_embedder.mlp.0.weight").T.contiguous(), tt_device)
        t_emb_b0 = to_tt(expand_bias(st.get_tensor("t_embedder.mlp.0.bias"), TILE), tt_device)
        t_emb_w2 = to_tt(st.get_tensor("t_embedder.mlp.2.weight").T.contiguous(), tt_device)
        t_emb_b2 = to_tt(expand_bias(st.get_tensor("t_embedder.mlp.2.bias"), TILE), tt_device)
        ext_w_raw = st.get_tensor("external_cond.weight").T.contiguous()
        ext_w_pad = torch.zeros(EXT_COND_PAD, D_MODEL, dtype=torch.float32)
        ext_w_pad[:EXT_COND_DIM] = ext_w_raw
        ext_cond_w = to_tt(ext_w_pad.to(torch.bfloat16), tt_device)
        ext_cond_b = to_tt(expand_bias(st.get_tensor("external_cond.bias"), TILE), tt_device)

    t_freq = timestep_embedding(t_flat, FREQ_DIM)
    t_freq_pad = torch.zeros(TILE, FREQ_DIM, dtype=torch.bfloat16)
    t_freq_pad[0] = t_freq[0].to(torch.bfloat16)
    t_freq_tt = to_tt(t_freq_pad, tt_device)

    t_emb_a = zeros_tt((TILE, D_MODEL), tt_device)
    t_emb_b = zeros_tt((TILE, D_MODEL), tt_device)
    cond_tt = zeros_tt((TILE, D_MODEL), tt_device)

    linear_bias_k8(t_freq_tt, t_emb_w0, t_emb_b0, t_emb_a)
    silu_kernel(t_emb_a, t_emb_b)
    linear_bias_k32(t_emb_b, t_emb_w2, t_emb_b2, cond_tt)

    # External cond
    act_pad = torch.zeros(TILE, EXT_COND_PAD, dtype=torch.bfloat16)
    act_pad[0, :EXT_COND_DIM] = actions[0, 0].to(torch.bfloat16)
    act_tt = to_tt(act_pad, tt_device)
    ext_out = zeros_tt((TILE, D_MODEL), tt_device)
    linear_k1(act_tt, ext_cond_w, ext_out)
    ext_out = ttnn.add(ext_out, ext_cond_b)
    cond_tt = ttnn.add(cond_tt, ext_out)

    cond_tt_host = ttnn.to_torch(cond_tt)[0]  # row 0 = the conditioning
    compare("conditioning", c_ref_1d, cond_tt_host, atol=2.0)

    print("\n" + "=" * 70)
    print("STAGE 3: Block 0 Spatial Sub-block (step by step)")
    print("=" * 70)

    # Reference: run block 0 spatial manually
    block = ref_model.blocks[0]
    x_5d = x_ref_5d.clone()  # (1, 1, 9, 16, 1024)

    # adaLN modulation params
    s_adaln_out_ref = block.s_adaLN_modulation(F.silu(c_ref))  # (1, 1, 6144)
    s_shift_msa, s_scale_msa, s_gate_msa, s_shift_mlp, s_scale_mlp, s_gate_mlp = \
        s_adaln_out_ref.chunk(6, dim=-1)  # each (1, 1, 1024)

    # Step 1: LayerNorm
    x_normed_ref = block.s_norm1(x_5d)  # (1, 1, 9, 16, 1024)

    # Step 2: adaLN modulate
    from dit import modulate
    x_mod_ref = modulate(x_normed_ref, s_shift_msa, s_scale_msa)

    # Step 3: Spatial attention (QKV + RoPE + SDPA + out proj)
    x_attn_ref = block.s_attn(x_mod_ref)  # includes QKV, RoPE, SDPA, out proj

    # Compare with TT pipeline
    # Load block 0 spatial weights
    scaler = to_tt_l1(torch.ones(TILE, TILE, dtype=torch.bfloat16), tt_device)
    mean_scale = to_tt_l1(torch.full((TILE, TILE), 1.0 / D_MODEL, dtype=torch.bfloat16), tt_device)

    with safe_open(dit_path, framework="pt") as st:
        # adaLN weights
        adaln_w = to_tt(st.get_tensor("blocks.0.s_adaLN_modulation.1.weight").T.contiguous().to(torch.bfloat16), tt_device)
        adaln_b_raw = st.get_tensor("blocks.0.s_adaLN_modulation.1.bias").to(torch.bfloat16)
        adaln_b = to_tt(expand_bias(adaln_b_raw, TILE), tt_device)
        # QKV
        qkv_w = to_tt(st.get_tensor("blocks.0.s_attn.to_qkv.weight").T.contiguous().to(torch.bfloat16), tt_device)
        # Out proj
        out_w = to_tt(st.get_tensor("blocks.0.s_attn.to_out.weight").T.contiguous().to(torch.bfloat16), tt_device)
        out_b_raw = st.get_tensor("blocks.0.s_attn.to_out.bias").to(torch.bfloat16)
        out_b = to_tt(expand_bias(out_b_raw, N_PATCH_PAD), tt_device)
        # MLP
        fc1_w = to_tt(st.get_tensor("blocks.0.s_mlp.fc1.weight").T.contiguous().to(torch.bfloat16), tt_device)
        fc1_b_raw = st.get_tensor("blocks.0.s_mlp.fc1.bias").to(torch.bfloat16)
        fc1_b = to_tt(expand_bias(fc1_b_raw, N_PATCH_PAD), tt_device)
        fc2_w = to_tt(st.get_tensor("blocks.0.s_mlp.fc2.weight").T.contiguous().to(torch.bfloat16), tt_device)
        fc2_b_raw = st.get_tensor("blocks.0.s_mlp.fc2.bias").to(torch.bfloat16)
        fc2_b = to_tt(expand_bias(fc2_b_raw, N_PATCH_PAD), tt_device)
        # RoPE freqs
        s_freqs = st.get_tensor("blocks.0.s_attn.rotary_emb.freqs").float()

    ln_w = to_tt(torch.ones(N_PATCH_PAD, D_MODEL, dtype=torch.bfloat16), tt_device)
    ln_b = to_tt(torch.zeros(N_PATCH_PAD, D_MODEL, dtype=torch.bfloat16), tt_device)

    # Input on TT
    x_pad = torch.zeros(N_PATCH_PAD, D_MODEL, dtype=torch.bfloat16)
    x_pad[:N_PATCHES] = x_ref_2d.to(torch.bfloat16)
    x_tt = to_tt(x_pad, tt_device)

    # Step 1: adaLN params
    adaln_scratch = zeros_tt((TILE, 6 * D_MODEL), tt_device)
    t_emb_scratch = zeros_tt((TILE, D_MODEL), tt_device)
    silu_kernel(cond_tt, t_emb_scratch)
    linear_bias_k32(t_emb_scratch, adaln_w, adaln_b, adaln_scratch)

    adaln_host = ttnn.to_torch(adaln_scratch)[0, :6 * D_MODEL].float()
    tt_chunks = adaln_host.reshape(6, D_MODEL)
    ref_chunks = s_adaln_out_ref[0, 0].reshape(6, D_MODEL)
    compare("adaln_params[0] (shift_msa)", ref_chunks[0], tt_chunks[0], atol=2.0)
    compare("adaln_params[2] (gate_msa)", ref_chunks[2], tt_chunks[2], atol=2.0)

    # Step 2: LayerNorm
    normed_tt = zeros_tt((N_PATCH_PAD, D_MODEL), tt_device)
    layernorm_d1024(x_tt, ln_w, ln_b, scaler, mean_scale, normed_tt)

    x_normed_ref_2d = x_normed_ref[0, 0].reshape(N_PATCHES, D_MODEL)
    normed_tt_host = ttnn.to_torch(normed_tt)[:N_PATCHES]
    compare("layernorm", x_normed_ref_2d, normed_tt_host, atol=1.0)

    # Step 3: adaLN modulate
    shift_msa_tt = to_tt(expand_bias(tt_chunks[0].to(torch.bfloat16), N_PATCH_PAD), tt_device)
    scale_msa_tt = to_tt(expand_bias(tt_chunks[1].to(torch.bfloat16), N_PATCH_PAD), tt_device)
    mod_tt = zeros_tt((N_PATCH_PAD, D_MODEL), tt_device)
    adaln_modulate_kernel(normed_tt, shift_msa_tt, scale_msa_tt, mod_tt)

    x_mod_ref_2d = x_mod_ref[0, 0].reshape(N_PATCHES, D_MODEL)
    mod_tt_host = ttnn.to_torch(mod_tt)[:N_PATCHES]
    compare("adaln_modulate", x_mod_ref_2d, mod_tt_host, atol=2.0)

    # Step 4: QKV projection
    qkv_tt = zeros_tt((N_PATCH_PAD, 3 * D_MODEL), tt_device)
    linear_k32(mod_tt, qkv_w, qkv_tt)

    # Reference QKV
    qkv_ref = block.s_attn.to_qkv(x_mod_ref.reshape(1, N_PATCHES, D_MODEL))  # (1, 144, 3072)
    qkv_ref_2d = qkv_ref[0]
    qkv_tt_host = ttnn.to_torch(qkv_tt)[:N_PATCHES]
    compare("qkv_projection", qkv_ref_2d, qkv_tt_host, atol=5.0)

    # Step 5: RoPE
    from oasis_inference import precompute_spatial_rope_freqs, apply_rotary_emb, rotate_half
    SPATIAL_ROPE_FREQS = precompute_spatial_rope_freqs(s_freqs)

    q_ref, k_ref, v_ref = qkv_ref_2d.float().chunk(3, dim=-1)

    # Reference RoPE (using the model's rotary embedding)
    q_ref_5d = rearrange(q_ref.unsqueeze(0), "b (H W) (h d) -> b h H W d", H=FRAME_H, W=FRAME_W, h=N_HEADS)
    k_ref_5d = rearrange(k_ref.unsqueeze(0), "b (H W) (h d) -> b h H W d", H=FRAME_H, W=FRAME_W, h=N_HEADS)
    freqs_ref = block.s_attn.rotary_emb.get_axial_freqs(FRAME_H, FRAME_W)
    from rotary_embedding_torch import apply_rotary_emb as ref_apply_rotary_emb
    q_ref_rope = ref_apply_rotary_emb(freqs_ref, q_ref_5d)
    k_ref_rope = ref_apply_rotary_emb(freqs_ref, k_ref_5d)

    # Our RoPE
    qkv_tt_f = qkv_tt_host.float()
    q_tt, k_tt_h, v_tt_h = qkv_tt_f[:N_PATCHES].chunk(3, dim=-1)
    q_mh = q_tt.view(N_PATCHES, N_HEADS, D_HEAD)
    k_mh = k_tt_h.view(N_PATCHES, N_HEADS, D_HEAD)
    rope_freqs = SPATIAL_ROPE_FREQS.unsqueeze(1)
    q_our_rope = apply_rotary_emb(rope_freqs, q_mh)
    k_our_rope = apply_rotary_emb(rope_freqs, k_mh)

    # Compare RoPE'd Q
    q_ref_flat = rearrange(q_ref_rope, "b h H W d -> b (H W) h d")[0]
    q_our_flat = q_our_rope  # (144, 16, 64)
    compare("rope_q", q_ref_flat, q_our_flat, atol=5.0)

    # Step 6: SDPA
    q_ref_sdpa = rearrange(q_ref_rope, "b h H W d -> b h (H W) d")
    k_ref_sdpa = rearrange(k_ref_rope, "b h H W d -> b h (H W) d")
    v_ref_5d = rearrange(v_ref.unsqueeze(0), "b (H W) (h d) -> b h (H W) d", H=FRAME_H, W=FRAME_W, h=N_HEADS)
    attn_ref = F.scaled_dot_product_attention(q_ref_sdpa.float(), k_ref_sdpa.float(), v_ref_5d.float(), is_causal=False)
    attn_ref_2d = rearrange(attn_ref, "b h s d -> b s (h d)")[0]  # (144, 1024)

    # Our SDPA via ttnn
    q_for_sdpa = q_our_rope.permute(1, 0, 2).unsqueeze(0).to(torch.bfloat16)
    k_for_sdpa = k_our_rope.permute(1, 0, 2).unsqueeze(0).to(torch.bfloat16)
    v_for_sdpa = v_tt_h[:N_PATCHES].view(N_PATCHES, N_HEADS, D_HEAD).permute(1, 0, 2).unsqueeze(0).to(torch.bfloat16)
    q_for_sdpa = F.pad(q_for_sdpa, [0, 0, 0, N_PATCH_PAD - N_PATCHES])
    k_for_sdpa = F.pad(k_for_sdpa, [0, 0, 0, N_PATCH_PAD - N_PATCHES])
    v_for_sdpa = F.pad(v_for_sdpa, [0, 0, 0, N_PATCH_PAD - N_PATCHES])

    q_sdpa_tt = to_tt(q_for_sdpa, tt_device)
    k_sdpa_tt = to_tt(k_for_sdpa, tt_device)
    v_sdpa_tt = to_tt(v_for_sdpa, tt_device)
    attn_tt = ttnn.transformer.scaled_dot_product_attention(q_sdpa_tt, k_sdpa_tt, v_sdpa_tt, is_causal=False)
    attn_perm = ttnn.permute(attn_tt, [0, 2, 1, 3])
    attn_2d_tt = ttnn.reshape(attn_perm, [N_PATCH_PAD, D_MODEL])
    attn_tt_host = ttnn.to_torch(attn_2d_tt)[:N_PATCHES]

    compare("sdpa_output", attn_ref_2d, attn_tt_host, atol=5.0)

    # Step 7: O projection + bias
    o_proj_tt = zeros_tt((N_PATCH_PAD, D_MODEL), tt_device)
    linear_k32(attn_2d_tt, out_w, o_proj_tt)
    o_proj_tt = ttnn.add(o_proj_tt, out_b)

    o_ref = block.s_attn.to_out(attn_ref_2d.unsqueeze(0))  # (1, 144, 1024)
    o_ref_2d = o_ref[0]
    compare("o_proj+bias", o_ref_2d, ttnn.to_torch(o_proj_tt)[:N_PATCHES], atol=5.0)

    # Step 8: Gated residual
    gate_msa_tt = to_tt(expand_bias(tt_chunks[2].to(torch.bfloat16), N_PATCH_PAD), tt_device)
    z_b_tt = zeros_tt((N_PATCH_PAD, D_MODEL), tt_device)
    gated_residual_kernel(x_tt, o_proj_tt, gate_msa_tt, z_b_tt)

    # Reference gated residual: x + gate * attn_out
    from dit import gate as ref_gate
    z_ref = x_5d + ref_gate(x_attn_ref.reshape(1, 1, FRAME_H, FRAME_W, D_MODEL), s_gate_msa)
    z_ref_2d = z_ref[0, 0].reshape(N_PATCHES, D_MODEL)
    compare("spatial_attn_residual", z_ref_2d, ttnn.to_torch(z_b_tt)[:N_PATCHES], atol=5.0)

    # Step 9: MLP path (granular)
    x_normed2_ref = block.s_norm2(z_ref)
    x_mod2_ref = modulate(x_normed2_ref, s_shift_mlp, s_scale_mlp)

    # Reference MLP step by step
    fc1_ref = block.s_mlp.fc1(x_mod2_ref)
    gelu_ref = block.s_mlp.act(fc1_ref)
    fc2_ref = block.s_mlp.fc2(gelu_ref)
    mlp_ref = block.s_mlp.drop(fc2_ref)  # dropout is no-op in eval
    z_final_ref = z_ref + ref_gate(mlp_ref, s_gate_mlp)
    z_final_ref_2d = z_final_ref[0, 0].reshape(N_PATCHES, D_MODEL)

    # TT MLP - step by step with checks
    normed2_tt = zeros_tt((N_PATCH_PAD, D_MODEL), tt_device)
    mod2_tt = zeros_tt((N_PATCH_PAD, D_MODEL), tt_device)
    fc1_tt = zeros_tt((N_PATCH_PAD, D_MLP), tt_device)
    gelu_tt = zeros_tt((N_PATCH_PAD, D_MLP), tt_device)
    fc2_tt = zeros_tt((N_PATCH_PAD, D_MODEL), tt_device)
    z_a_tt = zeros_tt((N_PATCH_PAD, D_MODEL), tt_device)

    shift_mlp_tt = to_tt(expand_bias(tt_chunks[3].to(torch.bfloat16), N_PATCH_PAD), tt_device)
    scale_mlp_tt = to_tt(expand_bias(tt_chunks[4].to(torch.bfloat16), N_PATCH_PAD), tt_device)
    gate_mlp_tt = to_tt(expand_bias(tt_chunks[5].to(torch.bfloat16), N_PATCH_PAD), tt_device)

    layernorm_d1024(z_b_tt, ln_w, ln_b, scaler, mean_scale, normed2_tt)
    compare("mlp_layernorm", x_normed2_ref[0,0].reshape(N_PATCHES, D_MODEL),
            ttnn.to_torch(normed2_tt)[:N_PATCHES], atol=5.0)

    adaln_modulate_kernel(normed2_tt, shift_mlp_tt, scale_mlp_tt, mod2_tt)
    compare("mlp_adaln_mod", x_mod2_ref[0,0].reshape(N_PATCHES, D_MODEL),
            ttnn.to_torch(mod2_tt)[:N_PATCHES], atol=5.0)

    linear_bias_k32(mod2_tt, fc1_w, fc1_b, fc1_tt)
    compare("mlp_fc1", fc1_ref[0,0].reshape(N_PATCHES, D_MLP),
            ttnn.to_torch(fc1_tt)[:N_PATCHES], atol=10.0)

    gelu_approx_kernel(fc1_tt, gelu_tt)
    compare("mlp_gelu", gelu_ref[0,0].reshape(N_PATCHES, D_MLP),
            ttnn.to_torch(gelu_tt)[:N_PATCHES], atol=10.0)

    linear_accum_k32_4(gelu_tt, fc2_w, fc2_tt)
    fc2_tt = ttnn.add(fc2_tt, fc2_b)
    compare("mlp_fc2", fc2_ref[0,0].reshape(N_PATCHES, D_MODEL),
            ttnn.to_torch(fc2_tt)[:N_PATCHES], atol=10.0)

    gated_residual_kernel(z_b_tt, fc2_tt, gate_mlp_tt, z_a_tt)
    compare("block0_spatial_full", z_final_ref_2d, ttnn.to_torch(z_a_tt)[:N_PATCHES], atol=5.0)

    # Check for NaN/inf
    z_a_host = ttnn.to_torch(z_a_tt)[:N_PATCHES]
    print("  block0 output: nan=%d inf=%d range=[%.2f, %.2f]" % (
        z_a_host.isnan().sum().item(), z_a_host.isinf().sum().item(),
        z_a_host[~z_a_host.isnan() & ~z_a_host.isinf()].min().item() if z_a_host.isfinite().any() else 0,
        z_a_host[~z_a_host.isnan() & ~z_a_host.isinf()].max().item() if z_a_host.isfinite().any() else 0))

    print("\n" + "=" * 70)
    print("STAGE 4: Full forward pass comparison (all 16 blocks)")
    print("=" * 70)

    # Reference full forward
    with torch.no_grad():
        v_ref_full = ref_model(x_latent, timesteps, actions)  # (1, 1, 16, 18, 32)
    print("  Reference output range: [%.4f, %.4f]" % (v_ref_full.min().item(), v_ref_full.max().item()))

    # TT full forward (run from oasis_inference module)
    from oasis_inference import preload_dit_weights, prealloc_scratch, dit_forward

    dev = preload_dit_weights(tt_device, n_frames=1)
    scr = prealloc_scratch(tt_device, n_frames=1)

    v_tt_full = dit_forward(x_latent, timesteps, actions, dev, scr, tt_device, scaler, mean_scale)
    print("  TT output range: [%.4f, %.4f]" % (v_tt_full.min().item(), v_tt_full.max().item()))
    compare("full_forward_velocity", v_ref_full, v_tt_full, atol=10.0)

    print("\nDone!")
    ttnn.close_device(tt_device)
