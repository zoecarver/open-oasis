"""
Full triage: Run PyTorch reference end-to-end, save its output image,
then compare TT output at key checkpoints.
"""
import sys, types

# Stub timm
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
import os
import time
from safetensors.torch import load_model
from einops import rearrange
from PIL import Image

sys.path.insert(0, "/tmp")
from dit import DiT_models
from vae import VAE_models

SCALING_FACTOR = 0.07843137255

def find_weights(key):
    blob_dir = "/root/.cache/huggingface/hub/models--Etched--oasis-500m/blobs/"
    from safetensors import safe_open
    for f in sorted(os.listdir(blob_dir)):
        path = blob_dir + f
        with safe_open(path, framework="pt") as st:
            keys = list(st.keys())
            if any(k.startswith(key) for k in keys):
                return path
    raise FileNotFoundError("Weights not found for key: " + key)

def sigmoid_beta_schedule(timesteps, start=-3, end=3, tau=1, clamp_min=1e-5):
    steps = timesteps + 1
    t = torch.linspace(0, timesteps, steps, dtype=torch.float64) / timesteps
    v_start = torch.tensor(start / tau).sigmoid()
    v_end = torch.tensor(end / tau).sigmoid()
    alphas_cumprod = (-((t * (end - start) + start) / tau).sigmoid() + v_end) / (v_end - v_start)
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)

if __name__ == "__main__":
    torch.manual_seed(42)

    print("=" * 60)
    print("PURE PYTORCH REFERENCE (no TT)")
    print("=" * 60)

    # Load models
    print("\nLoading VAE...")
    vae = VAE_models["vit-l-20-shallow-encoder"]()
    load_model(vae, find_weights("encoder."))
    vae = vae.eval()

    print("Loading DiT...")
    model = DiT_models["DiT-S/2"]()
    load_model(model, find_weights("blocks."))
    model = model.eval()

    # Encode prompt
    prompt_path = "/tmp/sample_image_0.png"
    print("Encoding prompt:", prompt_path)
    prompt_img = Image.open(prompt_path).convert("RGB").resize((640, 360))
    prompt_tensor = torch.tensor(list(prompt_img.getdata()), dtype=torch.float32)
    prompt_tensor = prompt_tensor.reshape(1, 360, 640, 3).permute(0, 3, 1, 2) / 255.0
    with torch.no_grad():
        prompt_latent = vae.encode(prompt_tensor * 2 - 1).mean * SCALING_FACTOR
    prompt_latent = rearrange(prompt_latent, "b (h w) c -> b 1 c h w", h=18, w=32)
    print("Prompt latent:", prompt_latent.shape, "range: [%.3f, %.3f]" % (
        prompt_latent.min().item(), prompt_latent.max().item()))

    # Diffusion schedule
    max_noise_level = 1000
    ddim_steps = 10
    noise_range = torch.linspace(-1, max_noise_level - 1, ddim_steps + 1)
    noise_abs_max = 20
    stabilization_level = 15

    betas = sigmoid_beta_schedule(max_noise_level).float()
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    alphas_cumprod = rearrange(alphas_cumprod, "T -> T 1 1 1")

    # Generate frame (matching reference generate.py logic)
    chunk = torch.randn(1, 1, 16, 18, 32)
    chunk = torch.clamp(chunk, -noise_abs_max, noise_abs_max)
    x_full = torch.cat([prompt_latent, chunk], dim=1)  # (1, 2, 16, 18, 32)
    actions = torch.zeros(1, 2, 25)

    print("\nRunning DDIM (%d steps, T=2)..." % ddim_steps)
    t_total = time.time()

    for noise_idx in reversed(range(1, ddim_steps + 1)):
        t_step = time.time()

        # Timesteps: [stabilization for prompt, noise_level for gen]
        noise_level = int(noise_range[noise_idx].item())
        t_ctx = torch.full((1, 1), stabilization_level - 1, dtype=torch.long)
        t_cur = torch.full((1, 1), noise_level, dtype=torch.long)
        t_full = torch.cat([t_ctx, t_cur], dim=1)  # (1, 2)

        t_next_noise = int(noise_range[noise_idx - 1].item())
        t_next_cur = torch.full((1, 1), max(t_next_noise, noise_level) if t_next_noise < 0 else t_next_noise, dtype=torch.long)
        t_next_ctx = torch.full((1, 1), stabilization_level - 1, dtype=torch.long)
        t_next_full = torch.cat([t_next_ctx, t_next_cur], dim=1)

        with torch.no_grad():
            v = model(x_full, t_full, actions)

        # DDIM update (matching reference generate.py)
        alpha_t = alphas_cumprod[t_full]
        alpha_next = alphas_cumprod[t_next_full]
        alpha_next[:, :1] = torch.ones_like(alpha_next[:, :1])  # prompt frames get alpha=1
        if noise_idx == 1:
            alpha_next[:, -1:] = torch.ones_like(alpha_next[:, -1:])

        x_start = alpha_t.sqrt() * x_full - (1 - alpha_t).sqrt() * v
        x_noise = ((1 / alpha_t).sqrt() * x_full - x_start) / (1 / alpha_t - 1).sqrt()
        x_pred = alpha_next.sqrt() * x_start + x_noise * (1 - alpha_next).sqrt()
        x_full[:, -1:] = x_pred[:, -1:]

        elapsed = time.time() - t_step
        print("  Step %d/%d (noise=%d) %.2fs  v:[%.2f,%.2f]" % (
            ddim_steps - noise_idx + 1, ddim_steps, noise_level, elapsed,
            v[:, -1:].min().item(), v[:, -1:].max().item()))

    total_time = time.time() - t_total
    print("DDIM complete: %.1fs" % total_time)

    # Save the generated frame latent for TT comparison
    gen_latent = x_full[:, -1:]  # (1, 1, 16, 18, 32)
    print("Generated latent range: [%.3f, %.3f]" % (gen_latent.min().item(), gen_latent.max().item()))
    torch.save(gen_latent, "/tmp/ref_gen_latent.pt")
    torch.save(prompt_latent, "/tmp/ref_prompt_latent.pt")

    # Also save a single-step velocity for comparison
    # Re-run one forward with fixed noise to compare against TT
    torch.manual_seed(42)
    test_chunk = torch.randn(1, 1, 16, 18, 32)
    test_chunk = torch.clamp(test_chunk, -noise_abs_max, noise_abs_max)
    test_input = torch.cat([prompt_latent, test_chunk], dim=1)
    test_t = torch.tensor([[stabilization_level - 1, 999]], dtype=torch.long)
    test_actions = torch.zeros(1, 2, 25)
    with torch.no_grad():
        test_v = model(test_input, test_t, test_actions)
    torch.save(test_v, "/tmp/ref_velocity.pt")
    torch.save(test_input, "/tmp/ref_test_input.pt")
    print("\nSaved reference velocity shape:", test_v.shape,
          "range: [%.4f, %.4f]" % (test_v.min().item(), test_v.max().item()))

    # VAE decode
    print("\nDecoding with VAE...")
    z = rearrange(gen_latent, "b t c h w -> (b t) (h w) c")
    with torch.no_grad():
        decoded = (vae.decode(z.float() / SCALING_FACTOR) + 1) / 2
    decoded = torch.clamp(decoded, 0, 1)
    frame = rearrange(decoded, "b c h w -> b h w c")[0]
    frame_np = (frame * 255).byte().numpy()
    img = Image.fromarray(frame_np)
    img.save("/tmp/ref_frame.png")
    print("Saved REFERENCE frame to /tmp/ref_frame.png")
    print("Frame shape:", frame_np.shape, "range: [%d, %d]" % (frame_np.min(), frame_np.max()))

    # Also decode the prompt for visual comparison
    z_prompt = rearrange(prompt_latent, "b t c h w -> (b t) (h w) c")
    with torch.no_grad():
        decoded_prompt = (vae.decode(z_prompt.float() / SCALING_FACTOR) + 1) / 2
    decoded_prompt = torch.clamp(decoded_prompt, 0, 1)
    frame_prompt = rearrange(decoded_prompt, "b c h w -> b h w c")[0]
    img_prompt = Image.fromarray((frame_prompt * 255).byte().numpy())
    img_prompt.save("/tmp/ref_prompt_decoded.png")
    print("Saved decoded prompt to /tmp/ref_prompt_decoded.png")

    print("\nDone! Now run TT inference with same inputs and compare.")
