"""
Compare single-step DiT velocity: PyTorch reference vs TT.
Uses saved inputs from triage_full.py.
Also: run TT with same DDIM loop to see if error is per-step or accumulated.
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
import time
import os
from einops import rearrange

sys.path.insert(0, "/tmp")

def compare(name, ref, tt_val):
    ref_f = ref.float().contiguous()
    if not isinstance(tt_val, torch.Tensor):
        tt_f = ttnn.to_torch(tt_val).float()
    else:
        tt_f = tt_val.float()
    # Trim to match
    min_shape = [min(r, t) for r, t in zip(ref_f.shape, tt_f.shape)]
    for d in range(len(min_shape)):
        ref_f = ref_f.narrow(d, 0, min_shape[d])
        tt_f = tt_f.narrow(d, 0, min_shape[d])
    diff = (ref_f - tt_f).abs()
    max_err = diff.max().item()
    mean_err = diff.mean().item()
    ref_range = ref_f.max().item() - ref_f.min().item()
    nans = tt_f.isnan().sum().item()
    infs = tt_f.isinf().sum().item()
    print("  %-40s max=%.4f mean=%.4f rel=%.4f ref=[%.2f,%.2f] nan=%d inf=%d" % (
        name, max_err, mean_err, max_err / (ref_range + 1e-8),
        ref_f.min().item(), ref_f.max().item(), nans, infs))
    return max_err

if __name__ == "__main__":
    tt_device = ttnn.open_device(device_id=0)

    # Load saved reference data
    ref_velocity = torch.load("/tmp/ref_velocity.pt")  # (1, 2, 16, 18, 32)
    ref_input = torch.load("/tmp/ref_test_input.pt")    # (1, 2, 16, 18, 32)
    ref_prompt_latent = torch.load("/tmp/ref_prompt_latent.pt")  # (1, 1, 16, 18, 32)

    print("Reference velocity shape:", ref_velocity.shape,
          "range: [%.4f, %.4f]" % (ref_velocity.min().item(), ref_velocity.max().item()))
    print("Reference input shape:", ref_input.shape)

    # ================================================================
    # TEST 1: Single-step velocity comparison
    # ================================================================
    print("\n" + "=" * 60)
    print("TEST 1: Single forward pass velocity comparison")
    print("=" * 60)

    from oasis_inference import (preload_dit_weights, prealloc_scratch, dit_forward,
                                  to_tt, to_tt_l1, zeros_tt, TILE, D_MODEL)

    scaler = to_tt_l1(torch.ones(TILE, TILE, dtype=torch.bfloat16), tt_device)
    mean_scale = to_tt_l1(torch.full((TILE, TILE), 1.0 / D_MODEL, dtype=torch.bfloat16), tt_device)

    N_FRAMES = 2
    dev = preload_dit_weights(tt_device, n_frames=N_FRAMES)
    scr = prealloc_scratch(tt_device, n_frames=N_FRAMES)

    # Same input as reference
    test_t = torch.tensor([[14, 999]], dtype=torch.long)  # stabilization-1, noise
    test_actions = torch.zeros(1, 2, 25)

    tt_velocity = dit_forward(ref_input, test_t, test_actions, dev, scr, tt_device, scaler, mean_scale)

    print("\nTT velocity shape:", tt_velocity.shape,
          "range: [%.4f, %.4f]" % (tt_velocity.min().item(), tt_velocity.max().item()))

    compare("full_velocity (both frames)", ref_velocity, tt_velocity)
    compare("velocity frame 0 (prompt)", ref_velocity[:, 0], tt_velocity[:, 0])
    compare("velocity frame 1 (generated)", ref_velocity[:, 1], tt_velocity[:, 1])

    # Per-channel comparison for generated frame
    for ch in range(16):
        compare("  v1 channel %d" % ch, ref_velocity[0, 1, ch], tt_velocity[0, 1, ch])

    # ================================================================
    # TEST 2: Full DDIM loop comparison with same random seed
    # ================================================================
    print("\n" + "=" * 60)
    print("TEST 2: Full DDIM loop - TT vs reference latent")
    print("=" * 60)

    from oasis_inference import sigmoid_beta_schedule

    max_noise_level = 1000
    ddim_steps = 10
    noise_range = torch.linspace(-1, max_noise_level - 1, ddim_steps + 1)
    noise_abs_max = 20
    stabilization_level = 15

    betas = sigmoid_beta_schedule(max_noise_level).float()
    alphas_cumprod = torch.cumprod(1.0 - betas, dim=0)
    alphas_cumprod = rearrange(alphas_cumprod, "T -> T 1 1 1")

    torch.manual_seed(42)
    chunk = torch.randn(1, 1, 16, 18, 32)
    chunk = torch.clamp(chunk, -noise_abs_max, noise_abs_max)
    actions = torch.zeros(1, 2, 25)

    for noise_idx in reversed(range(1, ddim_steps + 1)):
        noise_level = int(noise_range[noise_idx].item())
        timesteps = torch.tensor([[stabilization_level - 1, noise_level]], dtype=torch.long)

        x_input = torch.cat([ref_prompt_latent, chunk], dim=1)
        v_all = dit_forward(x_input, timesteps, actions, dev, scr, tt_device, scaler, mean_scale)
        v = v_all[:, -1:]

        t_next_noise = int(noise_range[noise_idx - 1].item())
        t_val_long = torch.full((1, 1), noise_level, dtype=torch.long)
        t_next_long = torch.full((1, 1), max(t_next_noise, noise_level if t_next_noise < 0 else 0), dtype=torch.long)

        alpha_t = alphas_cumprod[t_val_long]
        alpha_next = alphas_cumprod[t_next_long]
        if noise_idx == 1:
            alpha_next = torch.ones_like(alpha_next)

        x_start = alpha_t.sqrt() * chunk - (1 - alpha_t).sqrt() * v
        x_noise = ((1 / alpha_t).sqrt() * chunk - x_start) / (1 / alpha_t - 1).sqrt()
        chunk = alpha_next.sqrt() * x_start + x_noise * (1 - alpha_next).sqrt()

        print("  Step %d noise=%d  v:[%.2f,%.2f]  chunk:[%.2f,%.2f]" % (
            ddim_steps - noise_idx + 1, noise_level,
            v.min().item(), v.max().item(),
            chunk.min().item(), chunk.max().item()))

    # Compare final latent with reference
    ref_gen_latent = torch.load("/tmp/ref_gen_latent.pt")
    compare("final_latent", ref_gen_latent, chunk)

    print("\nDone!")
    ttnn.close_device(tt_device)
