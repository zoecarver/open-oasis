# Temporal KV-cache across DDIM steps

Big perf lever. Theoretical ~1.7-1.9× speedup on `trace_exec`. May span
multiple sessions; this doc tracks design + progress.

## Target

Replace 12× full-forward per frame with: 1× past-frame precompute pass +
12× current-frame-only step. Cuts per-step compute roughly in half.

Baseline at start of work: commit `c97397f`, sterling, HiFi2, addcmul
modulate. trace_exec ~412 ms, per-frame steady-state ~2.4 FPS, overall
~2.0 FPS.

## Architectural invariant (proved by inspection)

For T=2 (1 context frame + 1 generated frame), the past frame's residual
stream after every block is **invariant across the 12 DDIM steps for one
video frame**. Argument:

- `z_cur = concat(context_z_dev, gen_z)`. `context_z_dev` is fixed for
  a video frame; only `gen_z` changes per DDIM step.
- `cond_list = [cond_context, gen_cond_step_dev[step_idx]]`. First entry
  fixed.
- LN, modulate, QKV matmul, o_proj, FC1, FC2, residuals, biases, gates
  are all **per-row** ops. Past-frame rows depend only on past-frame
  inputs and past-frame conditioning → invariant.
- Spatial attention only mixes within a frame → past-frame attention out
  depends only on past-frame Q/K/V → invariant.
- Temporal attention with the existing block-diagonal causal mask: past
  position 0 attends only to position 0 (itself); current position 1
  attends to both. Past attention out depends only on past Q/K/V →
  invariant.
- adaln_packed rows for past frame come from `silu(cond_context)` →
  invariant.

Therefore past frame can be precomputed once per video frame; only
current frame needs to run 12×. Current frame's temporal attention needs
past K/V at every temporal sub-block → that's the cache.

## Plan

1. **Survey** existing kernels and confirm row-position semantics:
   - `src/rope_layout_kernel.py` (or wherever rope_temporal lives) — does
     it index cos/sin by row, and can we restrict to bottom-half rows?
   - `src/sdpa_causal.py` — packed (BATCH, T_PADDED, D_HEAD) layout. With
     cached past K/V we keep T_PADDED=32 (16 past cached + 16 current);
     Q at past positions is unused (will discard those output rows).
   - `run_sub_block` body — list every scratch and op so we can mirror
     it at half size.

2. **Allocate scratches**:
   - Half-sized (160-row) versions of: `modulated_f32`, `z_scratch`,
     `z_a`, `o_proj`, `qkv_full_f32`, `gelu_f32`, `fc2`,
     `t_q_scratch`, `t_k_scratch`, `t_v_scratch`, `sdpa_temp_*`.
   - 16 cached past-K + 16 cached past-V tensors, each
     `(N_PATCH_PAD=160, D_MODEL_TP=256)` f32.

3. **`precompute_past_state(past_z, past_cond, ...)`**:
   - T=1 forward through 32 sub-blocks on past_z (160 rows).
   - At each temporal sub-block, save K,V projections to cache.
   - Causal-T=1 temporal SDPA degenerates to `attn = V` (softmax over a
     single position = 1). Can short-circuit attention or use a thin
     reshape.
   - For RoPE on temporal Q/K, use cos/sin tables for the **past-frame
     row range** (top half of full table).

4. **`dit_forward_currentonly(current_z, current_cond, cached_kv, ...)`**:
   - T=1 forward on current frame; structurally same as precompute pass.
   - For temporal sub-block:
     - Apply RoPE with **bottom-half** cos/sin (current-frame positions).
     - Build full K = concat(cached_past_K, current_K) → (320, 256).
     - Build full V = concat(cached_past_V, current_V) → (320, 256).
     - Build full Q with junk at past positions, current Q at the bottom
       (320, 256). The original packed reshape interleaves by (p,h)
       group, then within-group by frame position; need to confirm the
       cache-concat layout matches.
     - Run existing SDPA kernel on packed shape.
     - Slice out current-frame rows (160) from SDPA out.
   - For spatial sub-block, just run on (160, 1024) with no cache.
   - o_proj, all_reduce, bias, gate, residual, LN, FC1, GELU, FC2,
     all_reduce, bias, gate, residual on half-sized rows.

5. **Final layer**: only operates on current frame's rows now.
   `final_adaln`, `ttnn.layer_norm`, `addcmul`, `linear`, `add` all on
   (160, ...).

6. **Trace structure**: per frame the trace replays
   `precompute_past_state` → 12× `dit_forward_currentonly` → bridge →
   VAE. Bump `trace_region_size` to 200MB if needed (user confirmed
   unlimited headroom).

7. **PCC validation** against the existing full-forward path:
   - Save `v_dev` from full forward; save `v_dev` from KV-cache forward.
   - Compare PCC > 0.9999 (or check element-wise diff).
   - If mismatch, bisect by comparing residual at each block boundary.

8. **FPS validation** on sterling, frame visual check, commit or revert.

## Pitfalls to watch

- **RoPE table indexing**: positions 0..159 are past, 160..319 are
  current. `rope_temporal` needs to know which rows it's processing.
  Likely we slice the cos/sin tables once at setup into "past_*" and
  "current_*" copies.
- **Packed SDPA layout interleaves frames within (p,h) groups**. Cache
  must be in the right pre-pack layout, OR we cache in (160, 256) 2D
  layout and let the existing pack-reshape work after concat.
- **adaLN bias** is pre-expanded to `(SEQ=320, ...)`. Need (160-row)
  versions for the half path. Either re-expand to 160 rows at load
  time, or pre-expand to TILE rows (32) and rely on broadcast.
- **fc1/fc2/o_proj biases** similarly pre-expanded to 320. Either slice
  at runtime or store 160-row versions at load.
- **qkv_full_w_f32** is `(D_MODEL=1024, 5*D_MODEL_TP=1280)`. Same weight
  used for both passes — no change needed.
- **Trace replay**: precompute pass and step pass run on different
  device scratches. Need separate input scratches (`past_z_scratch`
  vs `current_z_scratch`) so the host loads the right data per replay.

## Progress log

- [x] Plan written, invariant proof sketched.
- [x] Task #25: Survey kernels.
- [x] Task #26: Allocate scratches.
- [x] Task #27: precompute_past_state.
- [x] Task #28: dit_forward_currentonly.
- [x] Task #29: Wire into ddim loop + trace.
- [x] Task #30: PCC validation. PCC = 0.999992, max_diff=3.6e-2, mean_diff=2.8e-3 (ref_max=4.4).
- [x] Task #31: FPS validation. Sterling, HiFi2: trace_exec 412ms→323ms (-22%), overall 2.13→2.68 FPS (+25%). Frames visually identical, committed.

## Result

Theoretical lower bound was ~50% (replace 12 full forwards with 1 + 12*0.5 ≈ 6.5 forwards).
Achieved 78%. Gap comes from: still packing T=2 layout in temporal SDPA (needed because
existing kernel handles block-diagonal causal mask), padding Q with zeros at past positions,
and slicing past-position output rows away. Future optimization: replace the T=2-with-zeroed-Q
path with a true T=1 cross-attention against cached K/V (skip the zero-padded Q half entirely).
