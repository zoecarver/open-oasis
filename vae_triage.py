"""
VAE quality triage: test targeted fp32 fixes one at a time.
Compares CPU (fp32) vs device (bf16) at each stage.
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

import torch.nn.functional as F
import ttnn
import ttl
import os
from safetensors import safe_open
from einops import rearrange

TILE = 32
SCALING_FACTOR = 0.07843137255
VAE_DEC_DEPTH = 12
VAE_DEC_DIM = 1024
VAE_DEC_HEADS = 16
VAE_LATENT_DIM = 16
VAE_PATCH_SIZE_VAE = 20
VAE_PATCH_DIM = 3 * VAE_PATCH_SIZE_VAE ** 2
VAE_SEQ_H = 18
VAE_SEQ_W = 32
VAE_SEQ_LEN = VAE_SEQ_H * VAE_SEQ_W
VAE_D_HEAD = VAE_DEC_DIM // VAE_DEC_HEADS
VAE_D_MLP = VAE_DEC_DIM * 4
VAE_ROPE_DIM = VAE_D_HEAD // 2
D_MODEL = 1024
D_TILES = D_MODEL // TILE
ELEM_GRAN = 8

def to_tt(t, device, dtype=ttnn.bfloat16):
    return ttnn.from_torch(t, dtype=dtype, layout=ttnn.TILE_LAYOUT,
                           device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)

def to_tt_l1(t, device):
    return ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                           device=device, memory_config=ttnn.L1_MEMORY_CONFIG)

def zeros_tt(shape, device, dtype=ttnn.bfloat16):
    return to_tt(torch.zeros(shape, dtype=torch.bfloat16 if dtype == ttnn.bfloat16 else torch.float32), device, dtype)

def readback(t):
    return ttnn.to_torch(t)

def expand_bias(bias_1d, seq_len):
    return bias_1d.unsqueeze(0).expand(seq_len, -1).contiguous()

def swap_adjacent_columns(w):
    w_swap = w.clone()
    w_swap[:, 0::2] = w[:, 1::2]
    w_swap[:, 1::2] = w[:, 0::2]
    return w_swap

def swap_adjacent_elements(b):
    b_swap = b.clone()
    b_swap[0::2] = b[1::2]
    b_swap[1::2] = b[0::2]
    return b_swap

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

def precompute_vae_rope(tt_device):
    n_freqs = VAE_D_HEAD // 4 // 2
    pixel_freqs = torch.linspace(1.0, (VAE_SEQ_H * VAE_SEQ_W) / 2, n_freqs)
    h_freqs = torch.einsum("i, f -> i f", torch.arange(VAE_SEQ_H, dtype=torch.float32), pixel_freqs)
    h_freqs = h_freqs.repeat_interleave(2, dim=-1)
    w_freqs = torch.einsum("i, f -> i f", torch.arange(VAE_SEQ_W, dtype=torch.float32), pixel_freqs)
    w_freqs = w_freqs.repeat_interleave(2, dim=-1)
    h_broad = h_freqs[:, None, :].expand(VAE_SEQ_H, VAE_SEQ_W, -1)
    w_broad = w_freqs[None, :, :].expand(VAE_SEQ_H, VAE_SEQ_W, -1)
    freqs_flat = torch.cat([h_broad, w_broad], dim=-1).reshape(VAE_SEQ_LEN, VAE_ROPE_DIM)
    cos_ph = torch.ones(VAE_SEQ_LEN, VAE_D_HEAD)
    sin_ph = torch.zeros(VAE_SEQ_LEN, VAE_D_HEAD)
    cos_ph[:, :VAE_ROPE_DIM] = freqs_flat.cos()
    sin_ph[:, :VAE_ROPE_DIM] = freqs_flat.sin()
    sign = torch.ones(VAE_D_HEAD)
    sign[:VAE_ROPE_DIM:2] = -1
    sin_perm = sin_ph * sign.unsqueeze(0)
    cos_full = cos_ph.repeat(1, VAE_DEC_HEADS).to(torch.bfloat16)
    sin_full = sin_perm.repeat(1, VAE_DEC_HEADS).to(torch.bfloat16)
    return to_tt(cos_full, tt_device), to_tt(sin_full, tt_device)

# Import kernels
from oasis_inference import make_layernorm_kernel, fused_rope_kernel, make_vae_rope_kernel

layernorm_d1024 = make_layernorm_kernel(D_TILES)
vae_rope_fused = make_vae_rope_kernel(VAE_DEC_HEADS, VAE_D_HEAD // TILE)

def cpu_decode(z_flat, vae):
    """CPU fp32 reference decode, returns final output."""
    x = vae.post_quant_conv(z_flat.float())
    for blk in vae.decoder:
        normed = blk.norm1(x)
        q, k, v = blk.attn.qkv(normed).chunk(3, dim=-1)
        q_r = rearrange(q, "(H W) (h d) -> h H W d", H=VAE_SEQ_H, W=VAE_SEQ_W, h=VAE_DEC_HEADS)
        k_r = rearrange(k, "(H W) (h d) -> h H W d", H=VAE_SEQ_H, W=VAE_SEQ_W, h=VAE_DEC_HEADS)
        v_r = rearrange(v, "(H W) (h d) -> h H W d", H=VAE_SEQ_H, W=VAE_SEQ_W, h=VAE_DEC_HEADS)
        from rotary_embedding_torch import apply_rotary_emb
        q_r = apply_rotary_emb(blk.attn.rotary_freqs, q_r)
        k_r = apply_rotary_emb(blk.attn.rotary_freqs, k_r)
        q_f = rearrange(q_r, "h H W d -> h (H W) d")
        k_f = rearrange(k_r, "h H W d -> h (H W) d")
        v_f = rearrange(v_r, "h H W d -> h (H W) d")
        attn = F.scaled_dot_product_attention(q_f.unsqueeze(0), k_f.unsqueeze(0), v_f.unsqueeze(0))
        attn = rearrange(attn.squeeze(0), "h N d -> N (h d)")
        x = x + blk.attn.proj(attn)
        normed2 = blk.norm2(x)
        x = x + blk.mlp(normed2)
    x = vae.dec_norm(x)
    return vae.predictor(x)

def dev_decode(z_flat, vae, tt_device, scaler, mean_scale, hifi4_config, pqc_f32=False):
    """Device decode with configurable precision."""
    sd = vae.state_dict()

    # Load weights
    vd = {}
    pqc_w = sd["post_quant_conv.weight"].T.contiguous()
    pqc_w_padded = torch.zeros(32, VAE_DEC_DIM, dtype=torch.float32 if pqc_f32 else torch.bfloat16)
    pqc_w_padded[:VAE_LATENT_DIM] = pqc_w.to(pqc_w_padded.dtype)
    vd["pqc_w"] = to_tt(pqc_w_padded, tt_device, dtype=ttnn.float32 if pqc_f32 else ttnn.bfloat16)
    vd["pqc_b"] = to_tt(expand_bias(sd["post_quant_conv.bias"].to(
        torch.float32 if pqc_f32 else torch.bfloat16), VAE_SEQ_LEN), tt_device,
        dtype=ttnn.float32 if pqc_f32 else ttnn.bfloat16)

    for i in range(VAE_DEC_DEPTH):
        p = "decoder.%d" % i
        for ln_idx in ["1", "2"]:
            w = sd["%s.norm%s.weight" % (p, ln_idx)].to(torch.bfloat16)
            b = sd["%s.norm%s.bias" % (p, ln_idx)].to(torch.bfloat16)
            vd["%s.norm%s_w" % (p, ln_idx)] = to_tt(expand_bias(w, VAE_SEQ_LEN), tt_device)
            vd["%s.norm%s_b" % (p, ln_idx)] = to_tt(expand_bias(b, VAE_SEQ_LEN), tt_device)
        qkv_w = sd["%s.attn.qkv.weight" % p].T.contiguous().to(torch.bfloat16)
        q_w, k_w, v_w = qkv_w[:, :VAE_DEC_DIM], qkv_w[:, VAE_DEC_DIM:2*VAE_DEC_DIM], qkv_w[:, 2*VAE_DEC_DIM:]
        qkv_full_w = torch.cat([q_w, k_w, v_w, swap_adjacent_columns(q_w), swap_adjacent_columns(k_w)], dim=1)
        vd["%s.qkv_full_w" % p] = to_tt(qkv_full_w, tt_device)
        qkv_b = sd["%s.attn.qkv.bias" % p].to(torch.bfloat16)
        b_q, b_k, b_v = qkv_b[:VAE_DEC_DIM], qkv_b[VAE_DEC_DIM:2*VAE_DEC_DIM], qkv_b[2*VAE_DEC_DIM:]
        full_b = torch.cat([b_q, b_k, b_v, swap_adjacent_elements(b_q), swap_adjacent_elements(b_k)])
        vd["%s.qkv_full_b" % p] = to_tt(expand_bias(full_b, VAE_SEQ_LEN), tt_device)
        vd["%s.proj_w" % p] = to_tt(sd["%s.attn.proj.weight" % p].T.contiguous().to(torch.bfloat16), tt_device)
        vd["%s.proj_b" % p] = to_tt(expand_bias(sd["%s.attn.proj.bias" % p].to(torch.bfloat16), VAE_SEQ_LEN), tt_device)
        vd["%s.fc1_w" % p] = to_tt(sd["%s.mlp.fc1.weight" % p].T.contiguous().to(torch.bfloat16), tt_device)
        vd["%s.fc1_b" % p] = to_tt(expand_bias(sd["%s.mlp.fc1.bias" % p].to(torch.bfloat16), VAE_SEQ_LEN), tt_device)
        vd["%s.fc2_w" % p] = to_tt(sd["%s.mlp.fc2.weight" % p].T.contiguous().to(torch.bfloat16), tt_device)
        vd["%s.fc2_b" % p] = to_tt(expand_bias(sd["%s.mlp.fc2.bias" % p].to(torch.bfloat16), VAE_SEQ_LEN), tt_device)
    vd["dec_norm_w"] = to_tt(expand_bias(sd["dec_norm.weight"].to(torch.bfloat16), VAE_SEQ_LEN), tt_device)
    vd["dec_norm_b"] = to_tt(expand_bias(sd["dec_norm.bias"].to(torch.bfloat16), VAE_SEQ_LEN), tt_device)
    VAE_PATCH_DIM_PAD = ((VAE_PATCH_DIM + TILE - 1) // TILE) * TILE
    pred_w_raw = sd["predictor.weight"].T.contiguous().to(torch.bfloat16)
    pred_w_padded = torch.zeros(VAE_DEC_DIM, VAE_PATCH_DIM_PAD, dtype=torch.bfloat16)
    pred_w_padded[:, :VAE_PATCH_DIM] = pred_w_raw
    vd["pred_w"] = to_tt(pred_w_padded, tt_device)
    pred_b_padded = torch.zeros(VAE_PATCH_DIM_PAD, dtype=torch.bfloat16)
    pred_b_padded[:VAE_PATCH_DIM] = sd["predictor.bias"].to(torch.bfloat16)
    vd["pred_b"] = to_tt(expand_bias(pred_b_padded, VAE_SEQ_LEN), tt_device)
    vd["vae_cos"], vd["vae_sin_perm"] = precompute_vae_rope(tt_device)

    # Scratch
    vs = {}
    vs["z_a"] = zeros_tt((VAE_SEQ_LEN, VAE_DEC_DIM), tt_device)
    vs["z_b"] = zeros_tt((VAE_SEQ_LEN, VAE_DEC_DIM), tt_device)
    vs["normed"] = zeros_tt((VAE_SEQ_LEN, VAE_DEC_DIM), tt_device)
    vs["qkv_full"] = zeros_tt((VAE_SEQ_LEN, 5 * VAE_DEC_DIM), tt_device)
    vs["o_proj"] = zeros_tt((VAE_SEQ_LEN, VAE_DEC_DIM), tt_device)
    vs["fc1"] = zeros_tt((VAE_SEQ_LEN, VAE_D_MLP), tt_device)
    vs["gelu"] = zeros_tt((VAE_SEQ_LEN, VAE_D_MLP), tt_device)
    vs["fc2"] = zeros_tt((VAE_SEQ_LEN, VAE_DEC_DIM), tt_device)
    vs["pred_out"] = zeros_tt((VAE_SEQ_LEN, VAE_PATCH_DIM_PAD), tt_device)
    SDPA_ROWS = VAE_DEC_HEADS * VAE_SEQ_LEN
    vs["q_sdpa"] = zeros_tt((SDPA_ROWS, VAE_D_HEAD), tt_device)
    vs["k_sdpa"] = zeros_tt((SDPA_ROWS, VAE_D_HEAD), tt_device)
    vs["v_sdpa"] = zeros_tt((SDPA_ROWS, VAE_D_HEAD), tt_device)

    # Forward
    if pqc_f32:
        z_padded = torch.zeros(VAE_SEQ_LEN, 32, dtype=torch.float32)
        z_padded[:, :VAE_LATENT_DIM] = z_flat.float()
        z_tt = to_tt(z_padded, tt_device, dtype=ttnn.float32)
        pqc_out = ttnn.matmul(z_tt, vd["pqc_w"], compute_kernel_config=hifi4_config)
        pqc_out = ttnn.add(pqc_out, vd["pqc_b"])
        # Convert back to bf16 for rest of pipeline
        pqc_bf16 = ttnn.typecast(pqc_out, ttnn.bfloat16)
        ttnn.copy(pqc_bf16, vs["z_a"])
        ttnn.deallocate(z_tt)
        ttnn.deallocate(pqc_out)
        ttnn.deallocate(pqc_bf16)
    else:
        z_padded = torch.zeros(VAE_SEQ_LEN, 32, dtype=torch.bfloat16)
        z_padded[:, :VAE_LATENT_DIM] = z_flat.to(torch.bfloat16)
        z_tt = to_tt(z_padded, tt_device)
        ttnn.matmul(z_tt, vd["pqc_w"], optional_output_tensor=vs["z_a"])
        ttnn.add(vs["z_a"], vd["pqc_b"], output_tensor=vs["z_a"])
        ttnn.deallocate(z_tt)

    for i in range(VAE_DEC_DEPTH):
        p = "decoder.%d" % i
        layernorm_d1024(vs["z_a"], vd["%s.norm1_w" % p], vd["%s.norm1_b" % p],
                        scaler, mean_scale, vs["normed"])
        ttnn.matmul(vs["normed"], vd["%s.qkv_full_w" % p], optional_output_tensor=vs["qkv_full"])
        ttnn.add(vs["qkv_full"], vd["%s.qkv_full_b" % p], output_tensor=vs["qkv_full"])
        vae_rope_fused(vs["qkv_full"], vd["vae_cos"], vd["vae_sin_perm"],
                       vs["q_sdpa"], vs["k_sdpa"], vs["v_sdpa"])
        q_s = ttnn.reshape(vs["q_sdpa"], [VAE_DEC_HEADS, 1, VAE_SEQ_LEN, VAE_D_HEAD])
        k_s = ttnn.reshape(vs["k_sdpa"], [VAE_DEC_HEADS, 1, VAE_SEQ_LEN, VAE_D_HEAD])
        v_s = ttnn.reshape(vs["v_sdpa"], [VAE_DEC_HEADS, 1, VAE_SEQ_LEN, VAE_D_HEAD])
        attn_out = ttnn.transformer.scaled_dot_product_attention(q_s, k_s, v_s, is_causal=False)
        attn_out = ttnn.permute(attn_out, [1, 2, 0, 3])
        attn_2d = ttnn.reshape(attn_out, [VAE_SEQ_LEN, VAE_DEC_DIM])
        ttnn.matmul(attn_2d, vd["%s.proj_w" % p], optional_output_tensor=vs["o_proj"])
        ttnn.add(vs["o_proj"], vd["%s.proj_b" % p], output_tensor=vs["o_proj"])
        ttnn.add(vs["z_a"], vs["o_proj"], output_tensor=vs["z_b"])
        layernorm_d1024(vs["z_b"], vd["%s.norm2_w" % p], vd["%s.norm2_b" % p],
                        scaler, mean_scale, vs["normed"])
        ttnn.matmul(vs["normed"], vd["%s.fc1_w" % p], optional_output_tensor=vs["fc1"])
        ttnn.add(vs["fc1"], vd["%s.fc1_b" % p], output_tensor=vs["fc1"])
        ttnn.gelu(vs["fc1"], output_tensor=vs["gelu"])
        ttnn.matmul(vs["gelu"], vd["%s.fc2_w" % p], optional_output_tensor=vs["fc2"])
        ttnn.add(vs["fc2"], vd["%s.fc2_b" % p], output_tensor=vs["fc2"])
        ttnn.add(vs["z_b"], vs["fc2"], output_tensor=vs["z_a"])

    layernorm_d1024(vs["z_a"], vd["dec_norm_w"], vd["dec_norm_b"],
                    scaler, mean_scale, vs["normed"])
    ttnn.matmul(vs["normed"], vd["pred_w"], optional_output_tensor=vs["pred_out"])
    ttnn.add(vs["pred_out"], vd["pred_b"], output_tensor=vs["pred_out"])

    result = readback(vs["pred_out"]).float()[:, :VAE_PATCH_DIM]
    return result


if __name__ == "__main__":
    tt_device = ttnn.open_device(device_id=0)
    torch.manual_seed(42)

    scaler = to_tt_l1(torch.ones(TILE, TILE, dtype=torch.bfloat16), tt_device)
    mean_scale = to_tt_l1(torch.full((TILE, TILE), 1.0 / D_MODEL, dtype=torch.bfloat16), tt_device)

    arch = tt_device.arch()
    hifi4_config = ttnn.init_device_compute_kernel_config(
        arch, math_fidelity=ttnn.MathFidelity.HiFi4,
        math_approx_mode=False, fp32_dest_acc_en=True, packer_l1_acc=True)

    print("Loading VAE...")
    vae = load_vae_cpu()

    # Encode prompt
    from PIL import Image
    prompt_img = Image.open("/tmp/sample_image_0.png").convert("RGB").resize((640, 360))
    prompt_tensor = torch.tensor(list(prompt_img.getdata()), dtype=torch.float32)
    prompt_tensor = prompt_tensor.reshape(1, 360, 640, 3).permute(0, 3, 1, 2) / 255.0
    with torch.no_grad():
        prompt_latent = vae.encode(prompt_tensor * 2 - 1).mean * SCALING_FACTOR
    z_latent = rearrange(prompt_latent, "b (h w) c -> b 1 c h w", h=18, w=32)
    z_flat = rearrange(z_latent[0, 0].float(), "c h w -> (h w) c") / SCALING_FACTOR

    # CPU reference
    print("\n=== CPU decode (fp32) ===")
    with torch.no_grad():
        cpu_out = cpu_decode(z_flat, vae)
    print("CPU range: [%.4f, %.4f]" % (cpu_out.min().item(), cpu_out.max().item()))

    # Test 1: Baseline (all bf16)
    print("\n=== Test 1: Baseline (all bf16) ===")
    dev_out = dev_decode(z_flat, vae, tt_device, scaler, mean_scale, hifi4_config, pqc_f32=False)
    diff = (dev_out - cpu_out).abs()
    print("  max_err=%.4f  mean_err=%.6f" % (diff.max().item(), diff.mean().item()))
    print("  Dev range: [%.4f, %.4f]" % (dev_out.min().item(), dev_out.max().item()))

    # Test 2: post_quant_conv in fp32
    print("\n=== Test 2: post_quant_conv in fp32 ===")
    dev_out2 = dev_decode(z_flat, vae, tt_device, scaler, mean_scale, hifi4_config, pqc_f32=True)
    diff2 = (dev_out2 - cpu_out).abs()
    print("  max_err=%.4f  mean_err=%.6f" % (diff2.max().item(), diff2.mean().item()))
    print("  Dev range: [%.4f, %.4f]" % (dev_out2.min().item(), dev_out2.max().item()))

    # Summary
    print("\n=== Summary ===")
    print("  Baseline:       max=%.4f mean=%.6f" % ((dev_out - cpu_out).abs().max().item(), (dev_out - cpu_out).abs().mean().item()))
    print("  pqc_f32:        max=%.4f mean=%.6f" % ((dev_out2 - cpu_out).abs().max().item(), (dev_out2 - cpu_out).abs().mean().item()))

    ttnn.close_device(tt_device)
    print("\nDone!")
