"""
Per-block triage: Feed reference input, compare block-by-block output.
Uses reference PyTorch model to get ground truth after each block.
"""
import sys, types

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

_timm_models_vit.Mlp = _Mlp
_timm_layers_helpers.to_2tuple = lambda x: (x, x) if not isinstance(x, (tuple, list)) else tuple(x)

import torch.nn.functional as F
import ttnn
import ttl
import os
from safetensors.torch import load_model
from einops import rearrange

sys.path.insert(0, "/tmp")
from dit import DiT_models

TILE = 32
D_MODEL = 1024
N_HEADS = 16
D_HEAD = 64
N_PATCHES = 9 * 16  # 144
N_PATCH_PAD = 160
IN_CHANNELS = 16
PATCH_SIZE = 2
FRAME_H = 9
FRAME_W = 16

def find_dit_weights():
    blob_dir = "/root/.cache/huggingface/hub/models--Etched--oasis-500m/blobs/"
    from safetensors import safe_open
    for f in sorted(os.listdir(blob_dir)):
        path = blob_dir + f
        with safe_open(path, framework="pt") as st:
            if any(k.startswith("blocks.") for k in st.keys()):
                return path

def compare(name, ref, tt_val):
    ref_f = ref.float().contiguous()
    tt_f = tt_val.float() if isinstance(tt_val, torch.Tensor) else ttnn.to_torch(tt_val).float()
    min_shape = [min(r, t) for r, t in zip(ref_f.shape, tt_f.shape)]
    for d in range(len(min_shape)):
        ref_f = ref_f.narrow(d, 0, min_shape[d])
        tt_f = tt_f.narrow(d, 0, min_shape[d])
    diff = (ref_f - tt_f).abs()
    max_err = diff.max().item()
    mean_err = diff.mean().item()
    ref_range = ref_f.max().item() - ref_f.min().item()
    print("  %-45s max=%.4f mean=%.4f rel=%.4f ref=[%.2f,%.2f]" % (
        name, max_err, mean_err, max_err / (ref_range + 1e-8),
        ref_f.min().item(), ref_f.max().item()))
    return max_err

if __name__ == "__main__":
    tt_device = ttnn.open_device(device_id=0)

    # Load reference model
    print("Loading reference model...")
    ref_model = DiT_models["DiT-S/2"]()
    load_model(ref_model, find_dit_weights())
    ref_model = ref_model.eval()

    # Load TT pipeline
    from oasis_inference import (preload_dit_weights, prealloc_scratch, to_tt, to_tt_l1,
                                  zeros_tt, run_sub_block, compute_cond_for_frame,
                                  patch_embed_host, expand_bias, timestep_embedding,
                                  precompute_spatial_rope_freqs, precompute_temporal_rope_freqs,
                                  apply_rotary_emb, unpatchify_host)
    import oasis_inference

    scaler = to_tt_l1(torch.ones(TILE, TILE, dtype=torch.bfloat16), tt_device)
    mean_scale = to_tt_l1(torch.full((TILE, TILE), 1.0 / D_MODEL, dtype=torch.bfloat16), tt_device)
    dev = preload_dit_weights(tt_device, n_frames=1)
    scr = prealloc_scratch(tt_device, n_frames=1)

    # Use T=1 for simplicity - just test the DiT forward accuracy
    torch.manual_seed(42)
    x_latent = torch.randn(1, 1, IN_CHANNELS, 18, 32) * 0.5
    timesteps = torch.tensor([[500]], dtype=torch.long)
    actions = torch.zeros(1, 1, 25)

    print("\n" + "=" * 60)
    print("Per-block comparison (T=1)")
    print("=" * 60)

    # ---- Reference forward (manual, capturing intermediates) ----
    B, T = 1, 1
    x_ref = rearrange(x_latent, "b t c h w -> (b t) c h w")
    x_ref_emb = ref_model.x_embedder(x_ref)
    x_ref_5d = rearrange(x_ref_emb, "(b t) h w d -> b t h w d", t=T)

    t_flat = rearrange(timesteps, "b t -> (b t)")
    c_ref = ref_model.t_embedder(t_flat)
    c_ref = rearrange(c_ref, "(b t) d -> b t d", t=T)
    c_ref += ref_model.external_cond(actions)

    # ---- TT forward setup ----
    x_tt_host = patch_embed_host(x_ref, dev["x_emb_conv_w"], dev["x_emb_conv_b"])
    x_pad = torch.zeros(N_PATCH_PAD, D_MODEL, dtype=torch.bfloat16)
    x_pad[:N_PATCHES] = x_tt_host
    z_cur = to_tt(x_pad, tt_device)

    cond_tt = compute_cond_for_frame(500, actions[0, 0], dev, scr, tt_device)

    # Compare conditioning
    compare("conditioning", c_ref[0, 0], ttnn.to_torch(cond_tt)[0, :D_MODEL])

    # ---- Per-block comparison ----
    x_ref_cur = x_ref_5d.clone()

    for block_idx in range(16):
        block = ref_model.blocks[block_idx]

        # Reference: run one block
        x_ref_cur = block(x_ref_cur, c_ref)
        ref_2d = x_ref_cur[0, 0].reshape(N_PATCHES, D_MODEL)

        # TT: run one block (spatial + temporal)
        cond_list = [cond_tt]
        z_cur = run_sub_block(
            "blocks.%d.s" % block_idx, z_cur, cond_list,
            dev, scr, tt_device, scaler, mean_scale, attn_type="spatial"
        )
        z_cur = run_sub_block(
            "blocks.%d.t" % block_idx, z_cur, cond_list,
            dev, scr, tt_device, scaler, mean_scale, attn_type="temporal"
        )

        tt_2d = ttnn.to_torch(z_cur)[:N_PATCHES]
        err = compare("block %d output" % block_idx, ref_2d, tt_2d)

        # If error is growing fast, also compare with RESET:
        # feed reference output to TT for next block
        if block_idx < 3 or err > 5.0:
            # Reset TT to reference output for next block
            x_pad_reset = torch.zeros(N_PATCH_PAD, D_MODEL, dtype=torch.bfloat16)
            x_pad_reset[:N_PATCHES] = ref_2d.to(torch.bfloat16)
            z_reset = to_tt(x_pad_reset, tt_device)

            # Run TT block with reference input
            z_reset = run_sub_block(
                "blocks.%d.s" % block_idx, z_reset, cond_list,
                dev, scr, tt_device, scaler, mean_scale, attn_type="spatial"
            )
            z_reset = run_sub_block(
                "blocks.%d.t" % block_idx, z_reset, cond_list,
                dev, scr, tt_device, scaler, mean_scale, attn_type="temporal"
            )
            reset_2d = ttnn.to_torch(z_reset)[:N_PATCHES]
            compare("block %d (reset-to-ref input)" % block_idx, ref_2d, reset_2d)

    # Final layer
    from dit import modulate
    c_final_ref = ref_model.final_layer(x_ref_cur, c_ref)
    ref_final_2d = c_final_ref[0, 0].reshape(N_PATCHES, -1)  # (144, 64)

    # TT final layer
    cond_host = ttnn.to_torch(cond_tt)[0, :D_MODEL].float()
    cond_silu = cond_host * torch.sigmoid(cond_host)
    mod_host = cond_silu @ dev["final_adaln_w_host"] + dev["final_adaln_b_host"]
    shift_f = expand_bias(mod_host[:D_MODEL].to(torch.bfloat16), N_PATCH_PAD)
    scale_f = expand_bias(mod_host[D_MODEL:].to(torch.bfloat16), N_PATCH_PAD)

    from oasis_inference import layernorm_d1024, adaln_modulate_kernel, linear_k32
    layernorm_d1024(z_cur, dev["ln_w_ones"], dev["ln_b_zeros"], scaler, mean_scale, scr["normed"])
    adaln_modulate_kernel(scr["normed"], to_tt(shift_f, tt_device), to_tt(scale_f, tt_device), scr["modulated"])
    linear_k32(scr["modulated"], dev["final_linear_w"], scr["final_out"])
    scr["final_out"] = ttnn.add(scr["final_out"], dev["final_linear_b"])

    tt_final = ttnn.to_torch(scr["final_out"])[:N_PATCHES]
    compare("final_layer", ref_final_2d, tt_final)

    # Unpatchify and compare
    ref_img = ref_model.unpatchify(rearrange(c_final_ref, "b t h w d -> (b t) h w d"))
    ref_img = rearrange(ref_img, "(b t) c h w -> b t c h w", t=T)

    tt_img = unpatchify_host(tt_final.float(), PATCH_SIZE, IN_CHANNELS, FRAME_H, FRAME_W)
    compare("unpatchified_output", ref_img[0, 0], tt_img[0])

    print("\nDone!")
    ttnn.close_device(tt_device)
