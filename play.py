#!/usr/bin/env python3
"""
Oasis interactive frame generator. Runs on TT device in a loop:
  1. Reads action from /tmp/oasis_action.json (written by play_server.py)
  2. Generates frame via traced DiT + VAE
  3. Writes frame to /tmp/oasis_live_frame.png (atomic rename)
  4. Writes status to /tmp/oasis_status.json

This is a single-threaded script like oasis_inference.py.
All device calls happen in the main thread. Ctrl+C exits cleanly.

Usage: run via run-test.sh --hw play.py
"""
import sys
import os
import json
import time
import torch
import ttnn
from PIL import Image
from einops import rearrange

# Import oasis_inference for all model setup
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
import importlib
oasis = importlib.import_module("oasis_inference")

# Constants
TILE = oasis.TILE
D_MODEL = oasis.D_MODEL
N_PATCH_PAD = oasis.N_PATCH_PAD
N_PATCHES = oasis.N_PATCHES
OUT_DIM = oasis.OUT_DIM
IN_CHANNELS = oasis.IN_CHANNELS
INPUT_H = oasis.INPUT_H
INPUT_W = oasis.INPUT_W
PATCH_SIZE = oasis.PATCH_SIZE
FRAME_H = oasis.FRAME_H
FRAME_W = oasis.FRAME_W
EXT_COND_DIM = oasis.EXT_COND_DIM
SCALING_FACTOR = oasis.SCALING_FACTOR
N_CHIPS = oasis.N_CHIPS
FREQ_DIM = oasis.FREQ_DIM
VAE_SEQ_LEN = oasis.VAE_SEQ_LEN
VAE_LATENT_DIM = oasis.VAE_LATENT_DIM
VAE_PATCH_DIM = oasis.VAE_PATCH_DIM

# File paths for IPC with play_server.py
FRAME_PATH = "/tmp/oasis_live_frame.png"
ACTION_PATH = "/tmp/oasis_action.json"
STATUS_PATH = "/tmp/oasis_status.json"

N_FRAMES = 2
ddim_steps = 4
noise_abs_max = 20
stabilization_level = 15


def read_action():
    """Read action vector from file. Returns (25,) torch tensor."""
    try:
        with open(ACTION_PATH) as f:
            data = json.load(f)
        vec = data.get("action", [0] * EXT_COND_DIM)
        return torch.tensor(vec[:EXT_COND_DIM], dtype=torch.float32)
    except Exception:
        return torch.zeros(EXT_COND_DIM)


def write_status(frame_index, fps):
    """Atomic write of status JSON."""
    tmp = STATUS_PATH + ".tmp"
    with open(tmp, "w") as f:
        json.dump({"frame_index": frame_index, "fps": round(fps, 1)}, f)
    os.rename(tmp, STATUS_PATH)


def write_frame_png(frame_hwc_float):
    """Atomic write of frame PNG. frame_hwc_float: (H, W, 3) float [0,1]."""
    frame_rgb = (frame_hwc_float * 255).byte().numpy()
    img = Image.fromarray(frame_rgb)
    tmp = FRAME_PATH + ".tmp.png"
    img.save(tmp)
    os.rename(tmp, FRAME_PATH)


if __name__ == "__main__":
    # === Identical to oasis_inference.py setup ===

    if N_CHIPS > 1:
        ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
        tt_device = ttnn.open_mesh_device(ttnn.MeshShape(1, N_CHIPS),
                                           trace_region_size=100000000)
        oasis._MESH_DEVICE = tt_device
    else:
        tt_device = ttnn.open_device(device_id=0, trace_region_size=100000000)
    oasis.init_compute_configs(tt_device)
    torch.manual_seed(42)

    print("=" * 60)
    print("Oasis Interactive Generator")
    print("=" * 60)

    scaler = oasis.to_tt_l1(torch.ones(TILE, TILE, dtype=torch.bfloat16), tt_device)
    mean_scale = oasis.to_tt_l1(torch.full((TILE, TILE), 1.0 / D_MODEL, dtype=torch.bfloat16), tt_device)

    # Load VAE for prompt encoding only (no device VAE decode in interactive mode)
    print("\nLoading VAE (encode only)...")
    vae = oasis.load_vae_cpu()

    prompt_path = "/tmp/sample_image_0.png"
    print("Encoding prompt image:", prompt_path)
    prompt_img = Image.open(prompt_path).convert("RGB").resize((640, 360))
    prompt_tensor = torch.tensor(list(prompt_img.getdata()), dtype=torch.float32)
    prompt_tensor = prompt_tensor.reshape(1, 360, 640, 3).permute(0, 3, 1, 2) / 255.0
    with torch.no_grad():
        prompt_latent = vae.encode(prompt_tensor * 2 - 1).mean * SCALING_FACTOR
    prompt_latent = rearrange(prompt_latent, "b (h w) c -> b 1 c h w",
                              h=INPUT_H, w=INPUT_W)
    print("Prompt latent shape:", prompt_latent.shape)

    dev = oasis.preload_dit_weights(tt_device, n_frames=N_FRAMES)
    scr = oasis.prealloc_scratch(tt_device, n_frames=N_FRAMES)

    max_noise_level = 1000
    noise_range = torch.linspace(-1, max_noise_level - 1, ddim_steps + 1)
    betas = oasis.sigmoid_beta_schedule(max_noise_level).float()
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)

    prompt_action = torch.zeros(EXT_COND_DIM)
    prompt_z_pad = torch.zeros(N_PATCH_PAD, D_MODEL, dtype=torch.bfloat16)
    prompt_z_pad[:N_PATCHES] = oasis.patch_embed_host(
        prompt_latent[0, 0].unsqueeze(0), dev["x_emb_conv_w"], dev["x_emb_conv_b"])
    prompt_z_dev = oasis.to_tt(prompt_z_pad, tt_device)

    prompt_cond_dev = oasis.compute_cond_for_frame(
        stabilization_level - 1, prompt_action, dev, scr, tt_device)

    conv_w = dev["x_emb_conv_w"].float()
    W_rt = conv_w.permute(0, 2, 3, 1).reshape(D_MODEL, OUT_DIM).T.contiguous()
    W_rt_dev = oasis.to_tt(W_rt.to(torch.bfloat16), tt_device)
    b_rt = dev["x_emb_conv_b"].float().to(torch.bfloat16)
    b_rt_pad = b_rt.unsqueeze(0).expand(N_PATCH_PAD, -1).contiguous()
    b_rt_dev = oasis.to_tt(b_rt_pad, tt_device)

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
    for noise_idx in ddim_coeffs_host:
        for k in ddim_coeffs_host[noise_idx]:
            ddim_coeffs_host[noise_idx][k] = ttnn.from_torch(
                ddim_coeffs_host[noise_idx][k], dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)

    ddim_coeff_dev = {}
    for k in ["at_sqrt", "1mat_sqrt", "inv_at_sqrt", "inv_sigma", "an_sqrt", "1man_sqrt"]:
        ddim_coeff_dev[k] = oasis.to_tt(torch.zeros(*CHUNK_SHAPE, dtype=torch.bfloat16), tt_device)

    cond_traced = []
    for f in range(N_FRAMES):
        cond_traced.append(oasis.to_tt(torch.zeros(TILE, D_MODEL, dtype=torch.bfloat16), tt_device))

    context_z_dev = oasis.to_tt(torch.zeros((N_FRAMES - 1) * N_PATCH_PAD, D_MODEL, dtype=torch.bfloat16), tt_device)
    trace_chunk = oasis.to_tt(torch.zeros(*CHUNK_SHAPE, dtype=torch.bfloat16), tt_device)

    def ddim_step_fn(chunk):
        gen_z = ttnn.linear(chunk, W_rt_dev, bias=b_rt_dev)
        z_cur = ttnn.concat([context_z_dev, gen_z], dim=0)
        final_out = oasis.dit_forward_device(z_cur, cond_traced, dev, scr, tt_device, scaler, mean_scale)
        gen_start = (N_FRAMES - 1) * N_PATCH_PAD
        v_dev = ttnn.slice(final_out, [gen_start, 0], [gen_start + N_PATCH_PAD, OUT_DIM])
        ttnn.multiply(chunk, ddim_coeff_dev["at_sqrt"], output_tensor=scr["ddim_tmp"])
        ttnn.multiply(v_dev, ddim_coeff_dev["1mat_sqrt"], output_tensor=scr["ddim_x_start"])
        ttnn.subtract(scr["ddim_tmp"], scr["ddim_x_start"], output_tensor=scr["ddim_x_start"])
        ttnn.multiply(chunk, ddim_coeff_dev["inv_at_sqrt"], output_tensor=scr["ddim_tmp"])
        ttnn.subtract(scr["ddim_tmp"], scr["ddim_x_start"], output_tensor=scr["ddim_x_noise"])
        ttnn.multiply(scr["ddim_x_noise"], ddim_coeff_dev["inv_sigma"], output_tensor=scr["ddim_x_noise"])
        ttnn.multiply(scr["ddim_x_start"], ddim_coeff_dev["an_sqrt"], output_tensor=scr["ddim_tmp"])
        ttnn.multiply(scr["ddim_x_noise"], ddim_coeff_dev["1man_sqrt"], output_tensor=scr["ddim_x_noise"])
        ttnn.add(scr["ddim_tmp"], scr["ddim_x_noise"], output_tensor=chunk)
        return chunk

    gen_cond_per_step = oasis.precompute_gen_cond(prompt_action, ddim_steps, noise_range, dev, scr, tt_device)

    # Compile
    print("Compiling DDIM step...")
    t_compile = time.time()
    first_noise_idx = ddim_steps
    for k in ddim_coeff_dev:
        ttnn.copy_host_to_device_tensor(ddim_coeffs_host[first_noise_idx][k], ddim_coeff_dev[k])
    for f in range(N_FRAMES - 1):
        host_t = ttnn.from_torch(
            oasis.readback_torch(prompt_cond_dev),
            dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
        ttnn.copy_host_to_device_tensor(host_t, cond_traced[f])
    first_cond_host = ttnn.from_torch(
        oasis.readback_torch(gen_cond_per_step[first_noise_idx]),
        dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    ttnn.copy_host_to_device_tensor(first_cond_host, cond_traced[N_FRAMES - 1])
    context_host = ttnn.from_torch(
        torch.cat([oasis.readback_torch(prompt_z_dev)] * (N_FRAMES - 1), dim=0),
        dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    ttnn.copy_host_to_device_tensor(context_host, context_z_dev)
    ddim_step_fn(trace_chunk)
    ttnn.synchronize_device(tt_device)
    print("Compile done in %.1fs" % (time.time() - t_compile))

    # Capture DiT trace
    print("Capturing DiT trace...")
    t_trace = time.time()
    trace_id = ttnn.begin_trace_capture(tt_device, cq_id=0)
    ddim_step_fn(trace_chunk)
    ttnn.end_trace_capture(tt_device, trace_id, cq_id=0)
    ttnn.synchronize_device(tt_device)
    print("DiT trace captured in %.1fs" % (time.time() - t_trace))

    gen_cond_host = {}
    for noise_idx in gen_cond_per_step:
        gen_cond_host[noise_idx] = ttnn.from_torch(
            oasis.readback_torch(gen_cond_per_step[noise_idx]),
            dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    prompt_cond_host = ttnn.from_torch(
        oasis.readback_torch(prompt_cond_dev),
        dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)

    def run_ddim_loop(chunk_in_host, context_cond_hosts=None, gen_cond_map=None):
        if gen_cond_map is None:
            gen_cond_map = gen_cond_host
        if context_cond_hosts is None:
            context_cond_hosts = [prompt_cond_host] * (N_FRAMES - 1)
        ttnn.copy_host_to_device_tensor(chunk_in_host, trace_chunk)
        for f in range(N_FRAMES - 1):
            ttnn.copy_host_to_device_tensor(context_cond_hosts[f], cond_traced[f])
        for noise_idx in reversed(range(1, ddim_steps + 1)):
            for k in ddim_coeff_dev:
                ttnn.copy_host_to_device_tensor(ddim_coeffs_host[noise_idx][k], ddim_coeff_dev[k])
            ttnn.copy_host_to_device_tensor(gen_cond_map[noise_idx], cond_traced[N_FRAMES - 1])
            ttnn.execute_trace(tt_device, trace_id, cq_id=0, blocking=False)
        ttnn.synchronize_device(tt_device)
        return trace_chunk

    # Warmup
    chunk_img = torch.randn(1, IN_CHANNELS, INPUT_H, INPUT_W)
    chunk_img = torch.clamp(chunk_img, -noise_abs_max, noise_abs_max)
    chunk_patches = oasis.patchify_to_output_space(chunk_img)
    chunk_pad = torch.zeros(N_PATCH_PAD, OUT_DIM, dtype=torch.bfloat16)
    chunk_pad[:N_PATCHES] = chunk_patches.to(torch.bfloat16)
    chunk_host_tilized = ttnn.from_torch(chunk_pad, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    print("\nWarmup...")
    _ = run_ddim_loop(chunk_host_tilized)
    print("Warmup done")

    # VAE skipped - any device calls (even untraced) allocate with active DiT trace.
    # We visualize raw latents instead. Add VAE back once trace conflict is resolved.

    # Context window
    context_window_z = [prompt_z_pad.clone() for _ in range(N_FRAMES - 1)]
    context_cond_window = [prompt_cond_host] * (N_FRAMES - 1)

    # === Interactive loop (same structure as oasis_inference.py video loop) ===
    print("\n" + "=" * 60)
    print("Generating frames. Run play_server.py separately for the UI.")
    print("Ctrl+C to stop.")
    print("=" * 60)

    frame_idx = 0
    frame_times = []
    try:
        while True:
            t_frame = time.time()

            # Read action from file
            action_vec = read_action()

            # Fresh noise
            chunk_img = torch.randn(1, IN_CHANNELS, INPUT_H, INPUT_W)
            chunk_img = torch.clamp(chunk_img, -noise_abs_max, noise_abs_max)
            chunk_patches = oasis.patchify_to_output_space(chunk_img)
            chunk_pad_f = torch.zeros(N_PATCH_PAD, OUT_DIM, dtype=torch.bfloat16)
            chunk_pad_f[:N_PATCHES] = chunk_patches.to(torch.bfloat16)
            chunk_host_f = ttnn.from_torch(chunk_pad_f, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)

            # Update context
            context_cat = torch.cat(context_window_z, dim=0)
            context_host = ttnn.from_torch(context_cat, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
            ttnn.copy_host_to_device_tensor(context_host, context_z_dev)

            # Conditioning (same as oasis_inference.py video loop)
            frame_cond_dev = oasis.precompute_gen_cond(action_vec, ddim_steps, noise_range, dev, scr, tt_device)
            frame_cond_host = {}
            for noise_idx in frame_cond_dev:
                frame_cond_host[noise_idx] = ttnn.from_torch(
                    oasis.readback_torch(frame_cond_dev[noise_idx]),
                    dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)

            # DiT
            result_dev = run_ddim_loop(chunk_host_f,
                                       context_cond_hosts=context_cond_window,
                                       gen_cond_map=frame_cond_host)

            # Readback latent
            chunk_result = oasis.readback_torch(result_dev)[:N_PATCHES].float()
            gen_latent = oasis.unpatchify_host(chunk_result, PATCH_SIZE, IN_CHANNELS, FRAME_H, FRAME_W)

            # Update context window
            new_z = torch.zeros(N_PATCH_PAD, D_MODEL, dtype=torch.bfloat16)
            new_z[:N_PATCHES] = oasis.patch_embed_host(
                gen_latent, dev["x_emb_conv_w"], dev["x_emb_conv_b"])
            context_window_z.pop(0)
            context_window_z.append(new_z)

            new_context_cond = oasis.compute_cond_for_frame(
                stabilization_level - 1, action_vec, dev, scr, tt_device)
            new_cond_entry = ttnn.from_torch(
                oasis.readback_torch(new_context_cond), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
            context_cond_window.pop(0)
            context_cond_window.append(new_cond_entry)

            # Visualize raw latent (no VAE - avoids device alloc with active trace)
            # gen_latent: (1, 16, 9, 16) -> take first 3 channels as RGB, upscale
            lat = gen_latent.squeeze(0).float()  # (16, 9, 16)
            rgb = lat[:3]  # (3, 9, 16)
            rgb = (rgb - rgb.min()) / (rgb.max() - rgb.min() + 1e-6)
            # Upscale to 360x640 via repeat
            rgb = rgb.unsqueeze(0)  # (1, 3, 9, 16)
            frame_hwc = torch.nn.functional.interpolate(rgb, size=(360, 640), mode='nearest')
            frame_hwc = frame_hwc.squeeze(0).permute(1, 2, 0)  # (360, 640, 3)

            # Write frame + status
            write_frame_png(frame_hwc)

            elapsed = time.time() - t_frame
            frame_times.append(elapsed)
            if len(frame_times) > 10:
                frame_times.pop(0)
            fps = len(frame_times) / sum(frame_times)
            write_status(frame_idx, fps)

            frame_idx += 1
            print("Frame %d: %.2fs (%.1f FPS)" % (frame_idx, elapsed, fps))

    except KeyboardInterrupt:
        print("\nStopping...")

    print("Closing device...")
    ttnn.close_device(tt_device)
    print("Done!")
