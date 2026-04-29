# Temporal SDPA perf levers

Steady-state (TP=1) per-block budget for the temporal sub-block:

```
rope_temporal:    0.6 ms
sdpa_layout_fwd:  1.7 ms   (15 ttnn reshape/permute/pad ops)
sdpa_kernel:      2.6 ms
post-sdpa layout: 0.4 ms
total attn core:  ~5.3 ms
```

Sub-block runs 16× per DDIM step × 12 DDIM steps × per frame, so a 1 ms saving here is ~190 ms/frame.

## 1. Block-diagonal batch packing (biggest single perf lever)

The kernel runs 640 batches/chip (TP=4) of `(32 × 32)` causal attention but only the first `T=2` queries are real — 30/32 rows are masked padding. ~94% of the compute is wasted.

Fix: pack 16 `(p, h)` groups into one batch with a block-diagonal causal mask: rows 0–1 attend only to keys 0–1 (causal), rows 2–3 only to keys 2–3, etc. Bias = 0 within-block (causal-masked), `-1e4` cross-block. Output identical, with 16× fewer kernel iterations and proper DST utilization. Likely 3–5× speedup of the kernel itself. Localized to `src/sdpa_causal.py` plus a host-side bias builder. The layout transform can stay or be subsumed into the next item.

## 2. Mega-kernel fusion (rope + layout + SDPA)

Today qkv_full → `rope_temporal` (DRAM write) → 15 ttnn ops (DRAM round-trip) → `sdpa_temporal_causal` (DRAM read). The layout step alone is 1.7 ms.

Fuse all three into a single TT-Lang kernel (analog of `make_rope_layout_kernel` for spatial in `src/rope_layout_kernel.py`). Read qkv_full once, do RoPE, write Q/K/V tiles directly into SDPA layout in L1, and run attention on them inline. Eliminates ~2.3 ms of DRAM/layout work per sub-block.

Hardest of the four: the layout transform requires per-row scatter across tile boundaries (T=2 valid in T_PADDED=32 destination), which TT-Lang doesn't expose cleanly. Likely needs a redesigned data-movement pattern, similar in spirit to how spatial fuses but with extra packing logic. See `src/rope_layout_kernel.py` for the spatial precedent and deepseek's `tt-lang-kernels/fused_matmul_rmsnorm.py` for a SUMMA-grid fused-kernel pattern.

## 3. Fused softmax/reduce/normalize (deepseek "compressor" pattern)

Today the kernel has 4 sequential compute stages for softmax: `reduce_max → broadcast → exp → reduce_sum → broadcast → divide`. Each stage incurs DFB synchronization and cannot fuse with neighbors due to broadcast/reduce DFB constraints.

A single-pass fused stage (online softmax: track running max + running sum, normalize at the end) collapses these into 1–2 compute stages. Reference: deepseek's `tt-lang-kernels/compressor_softmax_sum_norm.py` does exactly this idiom for compressor outputs and could be adapted. Smaller win than (1) or (2) — maybe 20–30% of the kernel's softmax cost — but easy and cleanly local to the kernel.

## 4. L1-sharded SDPA inputs

Currently Q/K/V tensors live in DRAM. rope writes them, SDPA reads them, pure DRAM round-trip with no reuse. Per-chip TP=4 sizes: spatial 960 KB total Q+K+V (fits in one core's L1), temporal 15 MB total (234 KB/core when sharded across 64 cores, fits comfortably).

Allocate Q/K/V with `ttnn.create_sharded_memory_config` aligned to the SDPA grid (spatial: 8 batches → 8 cores, 1 batch/core; temporal: 640 batches → 64 cores, 10 batches/core). The DFB likeness in the kernel inherits the memory config, so `ttl.copy(Q_all[...], dfb.reserve())` becomes L1→L1 instead of DRAM round-trip. The rope kernel also needs to write into the same sharded layout.

Risk: if shard alignment doesn't match the SDPA per-core work assignment, NoC traffic replaces DRAM traffic and we gain nothing or regress. Honest expected gain: 10–30% on the SDPA kernel (kernel is more latency-bound than bandwidth-bound for these small shapes), so 5–15% on sub-block total. Best paired with (2) since the mega-kernel naturally produces L1-resident Q/K/V.

## 5. Pipe-mcast hot read-only data

cos/sin tables and `silu_cond` are already L1 (`to_tt_l1` / `to_tt_l1_f32`), but every consumer core does its own NoC read from the producer's L1. A pipe-mcast variant has one DM core read each tile once and broadcast it to all consumers via a multicast tree. Pattern: `pipe_examples/test_broadcast_2d.py` (rectangular mcast from one source core to a grid of destinations).

Three concrete targets, in order of payoff:

- **cos/sin tables in the fused mega-kernel (lever #2)**: 64 cores × 16 sub-blocks × 12 ddim steps all read the same per-position cos/sin tile. Mcast once per tile, every core latches it. Compounds with #2.
- **silu_cond per block**: `(TILE, D_MODEL)` = 128 KB, computed once per `(block, frame)` and read by every core's adaLN linear. Mcast from the producer to all adaLN consumers cuts a hot-path read.
- **K/V mcast inside block-diagonal SDPA (with #1)**: after batch-packing, multiple query tiles within a packed batch attend to the same K/V tiles. Read K/V once per packed batch and mcast to the cores doing the per-query work. Pairs naturally with #1's redesign.

Pipes are intra-chip only — they don't replace inter-chip all_reduce. ttnn's matmul already does 2D mcast internally (`pipe_examples/test_mcast_matmul.py`-style), so qkv/fc1/fc2 don't have headroom from explicit pipes.

## 6. Sequence-parallel residual stream

Today the row-parallel matmuls (`o_proj`, `fc2`) emit per-chip partials, `all_reduce` replicates the sum across chips, and then every chip redundantly runs the bias + gate-multiply + add-residual + LayerNorm work on the full `[SEQ, D_MODEL]` tensor. With TP=4 that residual-stream work is 4× redundant.

Lever: keep the residual stream sharded along the sequence dimension between row-parallel and column-parallel matmuls.

- After `fc2 @ W_row`, do `reduce_scatter` along seq instead of `all_reduce`. Each chip gets its own `[SEQ/4, D_MODEL]` slice.
- Run bias add, gated residual, LayerNorm, and adaLN modulate on the local slice (4× less work per chip, no algorithmic change since these are all row-wise ops).
- Before the next column-parallel matmul (e.g. qkv or fc1), `all_gather` along seq to re-replicate.

Net comm cost is the same (`reduce_scatter` + `all_gather` ≡ `all_reduce` on the ring), but the elementwise/LN/modulate work between them is 4× smaller. With our current `bias_gated_residual_kernel` running redundantly on 4 chips, this should ~4× the residual-stream throughput. Same idea applies symmetrically to the `o_proj` + gate + LN + modulate stretch on the attention side.

References: gemma and lingbot-world both use this pattern; deepseek's `inference.py` has the persistent-buffer + scatter/gather collectives idiom worth lifting. Risk: two collectives per stretch instead of one; need to confirm `reduce_scatter` + `all_gather` on FABRIC_1D_RING is at parity with `all_reduce` on our shapes.
