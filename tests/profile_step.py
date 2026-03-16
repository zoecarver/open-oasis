"""Profile one DiT forward pass with per-phase timing (no trace)."""
import sys
import os
import time
import torch
import ttnn

# Import everything from oasis_inference
sys.path.insert(0, os.path.dirname(__file__))

# Patch the module before importing
import oasis_inference as oi

# Override profiling flags
oi.PROFILE_BLOCKS = True
oi.PROFILE_BLOCK_DEVICE = "blocks.1.s"

N_CHIPS = oi.N_CHIPS
N_PATCH_PAD = oi.N_PATCH_PAD
D_MODEL = oi.D_MODEL
N_FRAMES = 2
SEQ = N_PATCH_PAD * N_FRAMES

if N_CHIPS > 1:
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    tt_device = ttnn.open_mesh_device(ttnn.MeshShape(1, N_CHIPS), trace_region_size=100000000)
    oi._MESH_DEVICE = tt_device
else:
    tt_device = ttnn.open_device(device_id=0, trace_region_size=100000000)
oi.init_compute_configs(tt_device)
torch.manual_seed(42)

print("Loading weights...")
dev = oi.preload_dit_weights(tt_device, N_FRAMES)
scr = oi.prealloc_scratch(tt_device, N_FRAMES)
scaler = oi.to_tt_l1(torch.ones(1, 32, dtype=torch.bfloat16), tt_device)
mean_scale = oi.to_tt_l1(torch.full((1, 32), 1.0/1024, dtype=torch.bfloat16), tt_device)

# Create dummy inputs
z_cur = oi.to_tt(torch.randn(SEQ, D_MODEL, dtype=torch.bfloat16), tt_device)
cond_list = [oi.to_tt(torch.randn(32, D_MODEL, dtype=torch.bfloat16), tt_device) for _ in range(N_FRAMES)]

# Warmup (compile all ops)
print("\nWarmup run...")
_ = oi.dit_forward_device(z_cur, cond_list, dev, scr, tt_device, scaler, mean_scale)
ttnn.synchronize_device(tt_device)

# Profile run with per-phase timing
print("\n" + "=" * 60)
print("PROFILE RUN (no trace, per-block sync)")
print("=" * 60)
_ = oi.dit_forward_device(z_cur, cond_list, dev, scr, tt_device, scaler, mean_scale, profile_step=True)
ttnn.synchronize_device(tt_device)

# Also time the full step
print("\n" + "=" * 60)
print("FULL STEP TIMING (no trace, no per-block sync)")
print("=" * 60)
oi.PROFILE_BLOCKS = False
oi.PROFILE_BLOCK_DEVICE = None

# 5 iterations
times = []
for i in range(5):
    ttnn.synchronize_device(tt_device)
    t0 = time.time()
    _ = oi.dit_forward_device(z_cur, cond_list, dev, scr, tt_device, scaler, mean_scale)
    ttnn.synchronize_device(tt_device)
    t1 = time.time()
    times.append((t1 - t0) * 1000)
    print("  iter %d: %.1fms" % (i, times[-1]))

avg = sum(times[1:]) / len(times[1:])
print("Average (excl first): %.1fms" % avg)

if N_CHIPS > 1:
    ttnn.close_mesh_device(tt_device)
else:
    ttnn.close_device(tt_device)
