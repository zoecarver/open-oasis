#!/usr/bin/env python3
"""
Oasis interactive frame generator. Single trace: 4 DDIM steps + bridge + VAE.
  1. Reads action from /tmp/oasis_action.json (written by play_server.py)
  2. Generates frame via single traced execution (DiT + unpatchify + VAE)
  3. Writes decoded frame to /tmp/oasis_live_frame.png (atomic rename)
  4. Writes status to /tmp/oasis_status.json

All device calls happen in the main thread. Ctrl+C exits cleanly.
Usage: run via run-test.sh --hw play.py
"""
import sys
import os
import json
import time
import gc
import signal
import torch
import ttnn
from PIL import Image
from einops import rearrange

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

FRAME_PATH = "/tmp/oasis_live_frame.png"
ACTION_PATH = "/tmp/oasis_action.json"
STATUS_PATH = "/tmp/oasis_status.json"

N_FRAMES = 2
ddim_steps = 4
noise_abs_max = 20
stabilization_level = 15

COEFF_KEYS = ["at_sqrt", "1mat_sqrt", "inv_at_sqrt", "inv_sigma", "an_sqrt", "1man_sqrt"]

_shutdown_done = False


def safe_shutdown(tt_device):
    global _shutdown_done
    if _shutdown_done:
        return
    _shutdown_done = True
    print("\nClosing device...")
    try:
        ttnn.close_device(tt_device)
    except Exception:
        pass
    print("Done!")
    os._exit(0)


def read_action():
    try:
        with open(ACTION_PATH) as f:
            data = json.load(f)
        vec = data.get("action", [0] * EXT_COND_DIM)
        return torch.tensor(vec[:EXT_COND_DIM], dtype=torch.float32)
    except Exception:
        return torch.zeros(EXT_COND_DIM)


def write_status(frame_index, fps):
    tmp = STATUS_PATH + ".tmp"
    with open(tmp, "w") as f:
        json.dump({"frame_index": frame_index, "fps": round(fps, 1)}, f)
    os.rename(tmp, STATUS_PATH)


def write_frame_bmp(frame_hwc_float):
    frame_rgb = (frame_hwc_float * 255).byte().numpy()
    img = Image.fromarray(frame_rgb)
    tmp = FRAME_PATH + ".tmp"
    img.save(tmp, format="BMP")
    os.rename(tmp, FRAME_PATH)


if __name__ == "__main__":
    if N_CHIPS > 1:
        ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
        tt_device = ttnn.open_mesh_device(ttnn.MeshShape(1, N_CHIPS),
                                           trace_region_size=100000000)
        oasis._MESH_DEVICE = tt_device
    else:
        tt_device = ttnn.open_device(device_id=0, trace_region_size=100000000)
    torch.manual_seed(42)

    signal.signal(signal.SIGINT, lambda *_: safe_shutdown(tt_device))
    signal.signal(signal.SIGTERM, lambda *_: safe_shutdown(tt_device))

    print("=" * 60)
    print("Oasis Interactive Generator (single trace: DiT+VAE)")
    print("=" * 60)

    scaler = oasis.to_tt_l1(torch.ones(TILE, TILE, dtype=torch.bfloat16), tt_device)
    mean_scale = oasis.to_tt_l1(torch.full((TILE, TILE), 1.0 / D_MODEL, dtype=torch.bfloat16), tt_device)

    # Load VAE
    print("\nLoading VAE...")
    vae = oasis.load_vae_cpu()
    print("Loading VAE decoder weights onto device...")
    vae_dev = oasis.preload_vae_decoder_weights(vae, tt_device)
    vae_scr = oasis.prealloc_vae_scratch(tt_device)

    # Encode all prompt images
    def encode_prompt(path):
        img = Image.open(path).convert("RGB").resize((640, 360))
        t = torch.tensor(list(img.getdata()), dtype=torch.float32)
        t = t.reshape(1, 360, 640, 3).permute(0, 3, 1, 2) / 255.0
        with torch.no_grad():
            latent = vae.encode(t * 2 - 1).mean * SCALING_FACTOR
        latent = rearrange(latent, "b (h w) c -> b 1 c h w", h=INPUT_H, w=INPUT_W)
        return latent, t

    import glob
    prompt_paths = ["/tmp/sample_image_0.png"] + sorted(glob.glob("/tmp/prompts/*.png"))
    all_prompts = []
    for p in prompt_paths:
        if os.path.exists(p):
            print("Encoding prompt:", p)
            latent, tensor = encode_prompt(p)
            all_prompts.append({"path": p, "latent": latent, "tensor": tensor})
    print("Encoded %d prompts" % len(all_prompts))
    prompt_idx = 0
    prompt_latent = all_prompts[0]["latent"]
    prompt_tensor = all_prompts[0]["tensor"]

    # Load DiT weights and scratch
    dev = oasis.preload_dit_weights(tt_device, n_frames=N_FRAMES)
    scr = oasis.prealloc_scratch(tt_device, n_frames=N_FRAMES)

    # Diffusion schedule
    max_noise_level = 1000
    noise_range = torch.linspace(-1, max_noise_level - 1, ddim_steps + 1)
    betas = oasis.sigmoid_beta_schedule(max_noise_level).float()
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)

    # Prompt data: pre-compute patch embeddings for all prompts
    prompt_action = torch.zeros(EXT_COND_DIM)
    for p in all_prompts:
        z_pad = torch.zeros(N_PATCH_PAD, D_MODEL, dtype=torch.bfloat16)
        z_pad[:N_PATCHES] = oasis.patch_embed_host(
            p["latent"][0, 0].unsqueeze(0), dev["x_emb_conv_w"], dev["x_emb_conv_b"])
        p["z_pad"] = z_pad
    prompt_z_pad = all_prompts[0]["z_pad"]
    prompt_z_dev = oasis.to_tt(prompt_z_pad, tt_device)
    prompt_cond_dev = oasis.compute_cond_for_frame(
        stabilization_level - 1, prompt_action, dev, scr, tt_device)

    # Round-trip weight
    conv_w = dev["x_emb_conv_w"].float()
    W_rt = conv_w.permute(0, 2, 3, 1).reshape(D_MODEL, OUT_DIM).T.contiguous()
    W_rt_dev = oasis.to_tt(W_rt.to(torch.bfloat16), tt_device)
    b_rt = dev["x_emb_conv_b"].float().to(torch.bfloat16)
    b_rt_pad = b_rt.unsqueeze(0).expand(N_PATCH_PAD, -1).contiguous()
    b_rt_dev = oasis.to_tt(b_rt_pad, tt_device)

    # DDIM coefficients (host tensors, tilized)
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

    # Per-step device coefficient tensors (4 sets)
    ddim_coeff_per_step = []
    for _ in range(ddim_steps):
        step_coeffs = {}
        for k in COEFF_KEYS:
            step_coeffs[k] = oasis.to_tt(torch.zeros(*CHUNK_SHAPE, dtype=torch.bfloat16), tt_device)
        ddim_coeff_per_step.append(step_coeffs)

    # Shared context conditioning + per-step gen conditioning
    cond_context = []
    for f in range(N_FRAMES - 1):
        cond_context.append(oasis.to_tt(torch.zeros(TILE, D_MODEL, dtype=torch.bfloat16), tt_device))
    gen_cond_step_dev = []
    for _ in range(ddim_steps):
        gen_cond_step_dev.append(oasis.to_tt(torch.zeros(TILE, D_MODEL, dtype=torch.bfloat16), tt_device))
    cond_traced_per_step = []
    for step_idx in range(ddim_steps):
        cond_traced_per_step.append(cond_context + [gen_cond_step_dev[step_idx]])

    context_z_dev = oasis.to_tt(
        torch.zeros((N_FRAMES - 1) * N_PATCH_PAD, D_MODEL, dtype=torch.bfloat16), tt_device)
    trace_chunk = oasis.to_tt(torch.zeros(*CHUNK_SHAPE, dtype=torch.bfloat16), tt_device)

    # Bridge matrices
    print("Building bridge matrices...")
    bridge = oasis.build_bridge_matrices(tt_device)
    bridge_tmp = oasis.zeros_tt((VAE_SEQ_LEN, OUT_DIM), tt_device)

    def ddim_step_fn(chunk, step_coeffs, cond_list):
        gen_z = ttnn.linear(chunk, W_rt_dev, bias=b_rt_dev)
        z_cur = ttnn.concat([context_z_dev, gen_z], dim=0)
        final_out = oasis.dit_forward_device(z_cur, cond_list, dev, scr, tt_device, scaler, mean_scale)
        gen_start = (N_FRAMES - 1) * N_PATCH_PAD
        v_dev = ttnn.slice(final_out, [gen_start, 0], [gen_start + N_PATCH_PAD, OUT_DIM])
        ttnn.multiply(chunk, step_coeffs["at_sqrt"], output_tensor=scr["ddim_tmp"])
        ttnn.multiply(v_dev, step_coeffs["1mat_sqrt"], output_tensor=scr["ddim_x_start"])
        ttnn.subtract(scr["ddim_tmp"], scr["ddim_x_start"], output_tensor=scr["ddim_x_start"])
        ttnn.multiply(chunk, step_coeffs["inv_at_sqrt"], output_tensor=scr["ddim_tmp"])
        ttnn.subtract(scr["ddim_tmp"], scr["ddim_x_start"], output_tensor=scr["ddim_x_noise"])
        ttnn.multiply(scr["ddim_x_noise"], step_coeffs["inv_sigma"], output_tensor=scr["ddim_x_noise"])
        ttnn.multiply(scr["ddim_x_start"], step_coeffs["an_sqrt"], output_tensor=scr["ddim_tmp"])
        ttnn.multiply(scr["ddim_x_noise"], step_coeffs["1man_sqrt"], output_tensor=scr["ddim_x_noise"])
        ttnn.add(scr["ddim_tmp"], scr["ddim_x_noise"], output_tensor=chunk)
        return chunk

    # Host-only conditioning (no device ops, safe with active trace)
    def compute_cond_host(t_scalar, action_vec):
        t_freq = oasis.timestep_embedding(torch.tensor([t_scalar]), FREQ_DIM)
        h = (t_freq[0].float() @ dev["t_emb_w0_host"]) + dev["t_emb_b0_host"]
        h = h * torch.sigmoid(h)
        cond = (h @ dev["t_emb_w2_host"]) + dev["t_emb_b2_host"]
        if action_vec is not None:
            ext = (action_vec.float() @ dev["ext_cond_w_host"]) + dev["ext_cond_b_host"]
            cond = cond + ext
        cond_pad = cond.to(torch.bfloat16).unsqueeze(0).expand(TILE, -1).contiguous()
        return ttnn.from_torch(cond_pad, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)

    def precompute_gen_cond_host(action_vec):
        cond_per_step = {}
        for noise_idx in reversed(range(1, ddim_steps + 1)):
            noise_level = int(noise_range[noise_idx].item())
            cond_per_step[noise_idx] = compute_cond_host(noise_level, action_vec)
        return cond_per_step

    # === Compile full pipeline ===
    print("Compiling full pipeline (4 DDIM steps + bridge + VAE)...")
    t_compile = time.time()
    prompt_cond_host_til = ttnn.from_torch(
        oasis.readback_torch(prompt_cond_dev),
        dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    context_host = ttnn.from_torch(
        torch.cat([oasis.readback_torch(prompt_z_dev)] * (N_FRAMES - 1), dim=0),
        dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    ttnn.copy_host_to_device_tensor(context_host, context_z_dev)
    for f in range(N_FRAMES - 1):
        ttnn.copy_host_to_device_tensor(prompt_cond_host_til, cond_context[f])
    gen_cond_per_step = oasis.precompute_gen_cond(
        prompt_action, ddim_steps, noise_range, dev, scr, tt_device)
    for step_idx in range(ddim_steps):
        noise_idx = ddim_steps - step_idx
        for k in COEFF_KEYS:
            ttnn.copy_host_to_device_tensor(ddim_coeffs_host[noise_idx][k],
                                            ddim_coeff_per_step[step_idx][k])
        gen_cond_til = ttnn.from_torch(
            oasis.readback_torch(gen_cond_per_step[noise_idx]),
            dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
        ttnn.copy_host_to_device_tensor(gen_cond_til, gen_cond_step_dev[step_idx])
    for step_idx in range(ddim_steps):
        ddim_step_fn(trace_chunk, ddim_coeff_per_step[step_idx],
                     cond_traced_per_step[step_idx])
    oasis.bridge_unpatchify(trace_chunk, bridge, bridge_tmp, vae_scr["vae_input"])
    oasis.vae_decode_forward(vae_dev, vae_scr, scaler, mean_scale)
    ttnn.synchronize_device(tt_device)
    print("Compile done in %.1fs" % (time.time() - t_compile))

    # === Capture single trace ===
    print("Capturing single trace...")
    t_trace = time.time()
    trace_id = ttnn.begin_trace_capture(tt_device, cq_id=0)
    for step_idx in range(ddim_steps):
        ddim_step_fn(trace_chunk, ddim_coeff_per_step[step_idx],
                     cond_traced_per_step[step_idx])
    oasis.bridge_unpatchify(trace_chunk, bridge, bridge_tmp, vae_scr["vae_input"])
    oasis.vae_decode_forward(vae_dev, vae_scr, scaler, mean_scale)
    ttnn.end_trace_capture(tt_device, trace_id, cq_id=0)
    ttnn.synchronize_device(tt_device)
    print("Single trace captured in %.1fs" % (time.time() - t_trace))

    # Prepare host prompt conditioning
    prompt_cond_host = ttnn.from_torch(
        oasis.readback_torch(prompt_cond_dev),
        dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)

    def run_full_frame(chunk_in_host, context_cond_hosts, gen_cond_map):
        ttnn.copy_host_to_device_tensor(chunk_in_host, trace_chunk)
        for f in range(N_FRAMES - 1):
            ttnn.copy_host_to_device_tensor(context_cond_hosts[f], cond_context[f])
        for step_idx in range(ddim_steps):
            noise_idx = ddim_steps - step_idx
            for k in COEFF_KEYS:
                ttnn.copy_host_to_device_tensor(ddim_coeffs_host[noise_idx][k],
                                                ddim_coeff_per_step[step_idx][k])
            ttnn.copy_host_to_device_tensor(gen_cond_map[noise_idx],
                                            gen_cond_step_dev[step_idx])
        ttnn.execute_trace(tt_device, trace_id, cq_id=0, blocking=False)
        ttnn.synchronize_device(tt_device)

    # Warmup
    chunk_img = torch.randn(1, IN_CHANNELS, INPUT_H, INPUT_W)
    chunk_img = torch.clamp(chunk_img, -noise_abs_max, noise_abs_max)
    chunk_patches = oasis.patchify_to_output_space(chunk_img)
    chunk_pad = torch.zeros(N_PATCH_PAD, OUT_DIM, dtype=torch.bfloat16)
    chunk_pad[:N_PATCHES] = chunk_patches.to(torch.bfloat16)
    chunk_host = ttnn.from_torch(chunk_pad, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    warmup_cond = precompute_gen_cond_host(prompt_action)
    print("\nWarmup...")
    run_full_frame(chunk_host, [prompt_cond_host] * (N_FRAMES - 1), warmup_cond)
    print("Warmup done")

    # Write prompt frame
    prompt_lat = prompt_latent[0, 0]
    z_flat = rearrange(prompt_lat.float(), "c h w -> (h w) c") / SCALING_FACTOR
    z_padded = torch.zeros(VAE_SEQ_LEN, 32, dtype=torch.bfloat16)
    z_padded[:, :VAE_LATENT_DIM] = z_flat.to(torch.bfloat16)
    # The warmup just ran the full trace, so vae_scr["pred_out"] has some output.
    # For the prompt frame, we already have the prompt image - just write it directly.
    prompt_img_tensor = prompt_tensor.squeeze(0).permute(1, 2, 0)  # (360, 640, 3)
    write_frame_bmp(prompt_img_tensor)
    print("Wrote prompt frame")

    # Context window
    context_window_z = [prompt_z_pad.clone() for _ in range(N_FRAMES - 1)]
    context_cond_window = [prompt_cond_host] * (N_FRAMES - 1)

    # === Interactive loop ===
    print("\n" + "=" * 60)
    print("Generating frames. Run play_server.py separately for the UI.")
    print("Ctrl+C to stop.")
    print("=" * 60)

    frame_idx = 0
    frame_times = []
    try:
        while True:
            t_frame = time.time()

            action_vec = read_action()

            # Fresh noise
            _t0 = time.time()
            chunk_img = torch.randn(1, IN_CHANNELS, INPUT_H, INPUT_W)
            chunk_img = torch.clamp(chunk_img, -noise_abs_max, noise_abs_max)
            chunk_patches = oasis.patchify_to_output_space(chunk_img)
            chunk_pad_f = torch.zeros(N_PATCH_PAD, OUT_DIM, dtype=torch.bfloat16)
            chunk_pad_f[:N_PATCHES] = chunk_patches.to(torch.bfloat16)
            chunk_host_f = ttnn.from_torch(chunk_pad_f, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)

            # Update context
            context_cat = torch.cat(context_window_z, dim=0)
            context_host_t = ttnn.from_torch(context_cat, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
            ttnn.copy_host_to_device_tensor(context_host_t, context_z_dev)

            # Conditioning (host-only, no device alloc)
            frame_cond_host = precompute_gen_cond_host(action_vec)
            _t1 = time.time()

            # Single trace: 4 DDIM + bridge + VAE
            run_full_frame(chunk_host_f, context_cond_window, frame_cond_host)
            _t2 = time.time()

            # Free host tilized tensors immediately after copy to device
            del chunk_host_f, context_host_t, frame_cond_host

            # Readback VAE output
            result = oasis.readback_torch(vae_scr["pred_out"]).float()
            patches = result[:, :VAE_PATCH_DIM]
            image = oasis.vae_unpatchify(patches)
            decoded = torch.clamp((image + 1) / 2, 0, 1)
            frame_hwc = decoded.squeeze(0).permute(1, 2, 0)  # (360, 640, 3)
            _t3 = time.time()
            write_frame_bmp(frame_hwc)
            _t4 = time.time()
            del result, patches, image, decoded

            # Update context window (readback DiT output for patch embedding)
            chunk_result = oasis.readback_torch(trace_chunk)[:N_PATCHES].float()
            gen_latent = oasis.unpatchify_host(chunk_result, PATCH_SIZE, IN_CHANNELS, FRAME_H, FRAME_W)
            new_z = torch.zeros(N_PATCH_PAD, D_MODEL, dtype=torch.bfloat16)
            new_z[:N_PATCHES] = oasis.patch_embed_host(
                gen_latent, dev["x_emb_conv_w"], dev["x_emb_conv_b"])
            context_window_z.pop(0)
            context_window_z.append(new_z)
            del chunk_result, gen_latent

            new_cond_entry = compute_cond_host(stabilization_level - 1, action_vec)
            context_cond_window.pop(0)
            context_cond_window.append(new_cond_entry)
            _t5 = time.time()

            # Reset context to initial prompt every 120 frames
            if frame_idx % 120 == 0 and frame_idx > 0:
                context_window_z = [all_prompts[0]["z_pad"].clone() for _ in range(N_FRAMES - 1)]
                context_cond_window = [prompt_cond_host] * (N_FRAMES - 1)
                print("  [reset context]")

            # Periodic GC to prevent ttnn host tensor accumulation
            if frame_idx % 10 == 0:
                gc.collect()

            # Timing breakdown every 20 frames
            if frame_idx % 20 == 0:
                import tracemalloc
                if not tracemalloc.is_tracing():
                    tracemalloc.start()
                cur, peak = tracemalloc.get_traced_memory()
                print("  [profile] prep=%.0fms trace=%.0fms readback=%.0fms png=%.0fms ctx=%.0fms | mem cur=%.0fMB peak=%.0fMB" % (
                    (_t1-_t0)*1000, (_t2-_t1)*1000, (_t3-_t2)*1000,
                    (_t4-_t3)*1000, (_t5-_t4)*1000,
                    cur/1e6, peak/1e6))

            elapsed = time.time() - t_frame
            frame_times.append(elapsed)
            if len(frame_times) > 10:
                frame_times.pop(0)
            fps = len(frame_times) / sum(frame_times)
            write_status(frame_idx, fps)

            frame_idx += 1
            print("Frame %d: %.2fs (%.1f FPS)" % (frame_idx, elapsed, fps))

    except KeyboardInterrupt:
        pass

    safe_shutdown(tt_device)
