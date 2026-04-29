# Oasis-500M perf optimization on Tenstorrent

Model is already brought up end-to-end at f32 precision and producing coherent video. Goal: maximize FPS.

## Autonomous task

Make perf improvements without pausing for confirmation. Pick the next lever, prototype, measure, commit if it improves FPS, revert otherwise. Document non-obvious decisions in code or notes. Pause only for: device hangs, ambiguity that meaningfully changes architecture, or destructive actions.

The FPS from the last committed run is the bar to beat. Find it by reading the most recent commit message or running once first to establish current state.

## Setup (every session)

Load these skills: `tt-connect-remote-device`, `ttnn`, `tt-lang`.

**Always use `sterling-all.conf`. Do not use `all.conf` or any other remote — they are shared with other people's work and you would disturb them.** If sterling is unreachable, pause and ask; do not silently fall back.

```
export TT_REMOTE_CONF=/Users/zcarver/.claude/skills/tt-connect-remote-device/scripts/sterling-all.conf
/Users/zcarver/.claude/skills/tt-connect-remote-device/scripts/smoke-test.sh
```

Run inference end-to-end on hardware (functional sim doesn't run the full model):

```
$TT_REMOTE_CONF/../scripts/run-test.sh --hw /Users/zcarver/Developer/oasis-ttlang/oasis_inference.py
```

Pull frames or video back for visual checks:

```
$TT_REMOTE_CONF/../scripts/copy-from-remote.sh /tmp/oasis_video_12step_T2/frame_NNNN.png /tmp/some_dir/
```

## Validation

There's no automated PCC gate that covers everything (the existing `tests/pcc_block0.py` only covers spatial block-0). Before committing a perf improvement, either:
- Pull a few frames and eyeball them locally, or
- Ask the user to validate.

Don't ship something that visibly regresses output quality.

## What you can change

- TT-Lang kernels: write more, fuse them, micro-optimize, mcast/pipe between cores.
- ttnn op replacements with TT-Lang fusions.
- Sharding, memory layout (DRAM/L1, sharded/interleaved), CCL params (`num_links`, etc).
- Trace structure, tensor pre-allocation, host orchestration.

**You have 4 cards — use them.** How work is distributed across the mesh is a big lever. Today we're at TP=4 with vanilla `all_reduce` and getting ~1.78× single-card. There's headroom: try alternative sharding schemes (sequence-parallel for some blocks, expert-style splits, fused matmul+all_reduce, async overlap of comms with compute). `../deepseek/inference.py` has battle-tested patterns for cluster-axis gathers and persistent-buffer collectives worth lifting.

## What you cannot change without asking

- **Don't move logic out of TT-Lang** (to ttnn or to host), even temporarily. Pause and ask.
- **Don't move anything out of f32** (no bf16 / bfp8 / bfp4 weights or activations) for now.
- **Don't change `ddim_steps`** (currently 12) — that's a quality lever the user controls.
- Anything destructive: force push, wiping caches, removing weights.

## Hang recovery

Don't run `pkill`, `tt-smi -r`, or any recovery commands. If you suspect a hang or device error, stop and ask the user.

## Where to start

`docs/perf_levers_temporal_sdpa.md` enumerates the 5 biggest known temporal-SDPA opportunities, ranked, with pointers to relevant existing kernels (`src/rope_layout_kernel.py`, `pipe_examples/test_broadcast_2d.py`) and to deepseek references.

Fusing more into tt-lang is a great idea. A sub-goal of this is to show tt-lang powering the model. You can write tt-lang mega kernels, and the user is happy to help with that. tt-lang will often immediately give perf wins, but also has a suit of perf tools that make optimization easier, if you want help optimizing tt-lang, ask the user. Once code is in tt-lang, it's much easier to optimize as there are more levers and more tools to assist.

You can also grep for commits that improved perf a lot. You can also reference ../deepseek and the referenced projects there. Deepseek has a number of great commits that are self contained perf improvements, that might help you identify targets and give you a good reference to replicate. 

Tracing might not be optimal, think about if we could trace bigger blocks.

Weight loading might not be optimal, think about if there's anything we can push to load time or any way to store weights more efficiently (eg sharded).

Another idea: look at how deepseek does tt-lang kernel caching, I think looking at the logs we are OK, but you could ensure that tt-lang kernels are not getting compiled eg per diffusion step (you'd see verbose cxx kernel emit logs between frames generated). If current kernel caching works, don't update it.

Another idea: you can explore ../tt-metal to see how models improve perf there. Specifically, I think there's something called minimal_matmul which overlaps matmul and ccls. You could look for this and look for similar ops that might help improve perf. There has been a lot of work done on dit / video gen models (sources are in ../tt-metal) to get perf/utilization above 50%.

VAE improvements (lower priority): currently not the bottleneck but may become eventually and we may want to lower diffusion steps which would increase how long this takes. See if you can improve the perf of the VAE decoder, metal should also have some references here.

## Prototyping kernels

Prototype new kernels in isolation first. Drop a self-contained test in `/tmp/test_<thing>.py`, copy with `copy-file.sh`, run with `run-test.sh --hw`. Promote to `src/` only once it's correct against a torch reference and integrates cleanly.

## References

`../deepseek` has battle-tested multi-chip TP, fused matmul+norm kernels, persistent-buffer trace patterns, and online-softmax/compressor kernels. Read `../deepseek/tt-lang-kernels/` for kernel idioms and `../deepseek/inference.py` for mesh+trace patterns. Don't copy verbatim — adapt to our shapes.

Only these sibling directories are in scope for exploration:

- `../tt-metal`
- `../tt-lang`
- `../engram`
- `../nanochat`
- `../tt-lang-import`
- `../gemma` — multi-chip mesh reference (mesh setup, shard/replicate patterns)
- `../lingbot-world` — multi-chip mesh reference

## Other notes

* We have a LOT of cores and L1, each core has about 1.5 MB of l1, we have about 130 cores on 4 chips, so 780 MB of L1 total, try to use it! In isolation you can measure how L1 sharded vs interleaved affects perf, specifically for datamovement (read/write) and ccls (eg all reduce).
* Pipes can give some ops 2-3x perf boost or even more. You can see how ../tt-lang/benchmarks does this, it has an optimal matmul, if any of your kernels need a matmul, you can use this as a template, and use the mcast pattern throughout any time you're reading input tensors in tt-lang.