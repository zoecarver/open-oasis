# Oasis-500M TT-Lang Optimization Log

## Target
2+ FPS at 5 DDIM steps on Wormhole n150 (72 cores, 1.0 GHz, 74 TFLOPS BF16).

## Architecture
- 16 DiT blocks, each with spatial + temporal sub-block
- Per sub-block: adaLN params, LN+modulate, QKV+RoPE+SDPA, O proj+LN+FC1+GELU+FC2+residual
- T=2 frames (1 prompt + 1 generated), N_PATCHES=144, N_PATCH_PAD=160, D_MODEL=1024

## Current Status (as of 2026-03-14)
- **5 DDIM steps**: ~TBD (measuring)
- **10 DDIM steps**: ~0.5s/step steady-state, ~5s total (first step has compilation overhead)
- **DDIM loop**: Fully on-device, zero host transfers between steps

## What Worked

### Mega Kernel A: QKV + RoPE + SDPA (spatial)
- Fused QKV matmul (K-accumulation), RoPE, and SDPA into one TT-Lang kernel
- 19 DFBs, parallelized over heads (32 total for T=2)
- Spatial sub-block QKV+SDPA: **0.2ms** (down from ~1.5ms with separate ops)
- File: `oasis_inference.py`, `make_qkv_rope_sdpa_kernel()`

### Mega Kernel B: O proj + LN + modulate + FC1 + GELU + FC2 + residual
- Fused entire MLP sub-block into one kernel with DRAM scratch phases
- 18 DFBs, parallelized over rows
- Phase A: O proj via (1, dim_tiles) @ (dim_tiles, 1) per output column
- Phase B: LN (3-pass) + adaLN modulate
- Phase C: FC1 with K-accumulation + GELU
- Phase D: FC2 with K-accumulation + bias + gate + residual
- Post-attn: **0.3ms** (down from 0.9ms temporal, 0.8ms spatial with separate ops)
- Applied to BOTH spatial and temporal paths

### On-Device DDIM Loop
- Chunk kept in output (patch) space as device tensor
- Round-trip matmul: output space -> input space via reshaped conv weights
- DDIM arithmetic with ttnn elementwise ops (multiply, add, subtract)
- Prompt frame patch embed and conditioning precomputed once
- Only readback after final step for VAE decode

### Temporal Attention Bug Fix
- Combined QKV weight left `scr["qkv"]` unpopulated; temporal path read stale data
- Fix: slice Q/K/V directly from `qkv_full` tensor
- Result: DDIM 7.8s -> 4.7s, correct output

## What Didn't Work

### Fused LN + QKV + RoPE + SDPA (kernel too large)
- Attempted to fuse LayerNorm + adaLN modulate into Mega Kernel A
- Program binary: 72,240 bytes vs 70,672 byte kernel config buffer limit
- 1,568 bytes over limit even after eliminating 3 DFBs via self-cycling
- Root cause: LN adds 3 passes of reduce/accumulate/broadcast per row, too much compiled code
- **Saved kernel**: `_wip_ln_qkv_rope_sdpa.py` in this directory
- Could revisit if kernel config buffer limit is raised, or with RMSNorm (2 passes instead of 3)

## Performance Breakdown (per step, steady-state)

| Component | Spatial (x16) | Temporal (x16) | Total |
|---|---|---|---|
| adaln params (ttnn) | 0.8ms x 16 | 0.8ms x 16 | 25.6ms |
| norm+mod (TT-Lang LN) | 0.2ms x 16 | 0.2ms x 16 | 6.4ms |
| qkv+sdpa (mega kernel A) | 0.2ms x 16 | - | 3.2ms |
| sdpa (temporal ttnn) | - | 0.8ms x 16 | 12.8ms |
| post_attn (mega kernel B) | 0.3ms x 16 | 0.3ms x 16 | 9.6ms |
| **Sub-block totals** | **25.6ms** | **35.2ms** | **60.8ms** |

Non-block overhead per step: ~1ms patch, ~1ms cond, ~2ms final layer, ~1ms DDIM
Total measured dispatch: ~65ms. True device execution: ~520ms.
**Gap: ~455ms of kernel launch/dispatch overhead.**

## Kernel Count Per Step (estimated)
- adaln params: ~10 ttnn ops x 32 sub-blocks = ~320 dispatches
- norm+mod: 1 TT-Lang kernel x 32 = 32
- spatial mega A: 1 x 16 = 16
- temporal QKV+reshape+sdpa: ~15 ttnn ops x 16 = ~240
- mega B: 1 x 32 = 32
- final layer: ~10 ttnn ops
- DDIM arithmetic: ~10 ttnn ops
- round-trip matmul + concat: ~3 ttnn ops
- **Total: ~660+ kernel dispatches per step**

## Optimization Opportunities (ranked by impact)

### 1. Fuse adaLN params into TT-Lang kernel
- Currently 10 ttnn ops per sub-block x 32 sub-blocks = 320 dispatches
- Pattern: SiLU(cond) @ adaln_w + adaln_b -> slice into 6 params -> expand
- Could be one TT-Lang kernel per sub-block: 32 dispatches (save ~288)

### 2. Fuse temporal QKV+RoPE into TT-Lang kernel
- Currently ttnn.matmul + 5 slices + 2 RoPE kernels = ~8 ops per block
- Same pattern as spatial mega kernel A but different RoPE tables
- Save ~112 dispatches (16 blocks x 7 ops)

### 3. Fuse temporal reshape+permute+SDPA
- Currently: 9 reshapes/permutes + ttnn SDPA per block = ~10 ops
- Harder to fuse (ttnn SDPA is optimized C++ kernel)
- Could write TT-Lang SDPA for temporal (T=2, causal, batched)

### 4. Reduce adaln expand cost
- `ttnn.concat([adaln_raw] * 5, dim=0)` creates 5 copies
- Could use broadcast in the consuming kernel instead

### 5. Move conditioning MLP to device
- Currently host-side: timestep_embedding + 2 matmuls + SiLU
- Small tensors but adds to_tt() call per step
- Minor: ~1ms savings

## Key Learnings

1. **Kernel dispatch overhead dominates**: 660+ dispatches x ~0.7ms each = ~460ms. Actual compute is ~60ms.
2. **ttnn.to_torch() forces full device sync**: The 455ms "readback" was actually waiting for all pending async ops.
3. **Async dispatch hides latency for 2-3 steps**: Steps 2-3 of DDIM run in 0.1s because dispatch runs ahead of execution.
4. **Kernel config buffer limit (70,672 bytes)**: Hard limit on compiled kernel binary size. Constrains how much we can fuse into one TT-Lang kernel.
5. **BF16 precision compounds**: Over 320 attention passes (16 blocks x 2 sub x 10 steps), BF16 softmax errors compound. Output recognizable but noisier than FP32.
6. **Per-head vs per-row parallelization**: Attention parallelizes over heads (32 items), MLP over rows (10 items). Different work distribution prevents fusing attention+MLP without pipes/barriers.
