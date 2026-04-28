"""Multi-head scaled dot-product attention kernel (non-causal).

Factory parameterized by (sq_tiles, skv_tiles, head_tiles, scale_val).
Grid="auto": each core runs `heads_per_core` head computations.

Input layout: heads stacked along dim 0.
  Q_all:    (total_heads * sq_tiles * TILE, head_tiles * TILE)
  K_all:    (total_heads * skv_tiles * TILE, head_tiles * TILE)
  V_all:    (total_heads * skv_tiles * TILE, head_tiles * TILE)
  attn_scratch: (sq_tiles * TILE, skv_tiles * TILE)  - dtype template only
  out:      (total_heads * sq_tiles * TILE, head_tiles * TILE)

attn_scratch exists only because make_dataflow_buffer_like requires a
likeness tensor whose tile grid is >= the DFB shape; K_all and Q_all are
too narrow along head_dim to back the (sq_tiles, skv_tiles) intermediates.
Its contents are never read or written.

DFB dtype is inherited from the input tensors, so the same kernel runs on
bf16 or f32 depending on caller-provided tensor dtypes. attn_scratch must
match Q/K/V dtype.
"""
import ttl

TILE = 32


def make_sdpa_kernel(sq_tiles, skv_tiles, head_tiles, scale_val):
    # Q is sliced one row-tile at a time so the per-iter intermediates are
    # (1, skv_tiles) instead of (sq_tiles, skv_tiles). This both fits f32 in
    # L1 (~530KB vs ~1.7MB) and avoids the multi-row matmul->reduce bug that
    # forced fp32_dest_acc_en in the flat version (single-row matmul output
    # has no per-tile/per-row ambiguity).
    # --no-ttl-reduce-full-fp32: workaround for tenstorrent/tt-lang#533, where
    # reduce_max/reduce_sum dims=[1] silently returns zeros for f32 tiles in
    # the default (full-fp32 reduce) lowering.
    # TODO: enable fp32_dest_acc_en=True for full f32 SDPA precision. Currently
    # blocked because the (1, skv_tiles=5) matmul output exceeds the 4-tile f32
    # DST capacity. The compiler suggests "enable maximize_dst to auto-subblock"
    # but adding "--ttl-maximize-dst" doesn't help; we likely need to either
    # sub-block the matmul down to (1, 1) per iter or find the correct flag.
    @ttl.operation(grid="auto", options="--no-ttl-reduce-full-fp32")
    def sdpa_kernel(Q_all, K_all, V_all, scaler, attn_scratch, out):
        grid_cols, _ = ttl.grid_size(dims=2)
        total_heads = Q_all.shape[0] // TILE // sq_tiles
        heads_per_core = -(-total_heads // grid_cols)

        q_dfb = ttl.make_dataflow_buffer_like(Q_all, shape=(1, head_tiles), block_count=2)
        k_dfb = ttl.make_dataflow_buffer_like(K_all, shape=(skv_tiles, head_tiles), block_count=2)
        v_dfb = ttl.make_dataflow_buffer_like(V_all, shape=(skv_tiles, head_tiles), block_count=2)
        sc_dfb = ttl.make_dataflow_buffer_like(scaler, shape=(1, 1), block_count=1)

        kt_dfb = ttl.make_dataflow_buffer_like(attn_scratch, shape=(head_tiles, skv_tiles), block_count=2)
        qk_dfb = ttl.make_dataflow_buffer_like(attn_scratch, shape=(1, skv_tiles), block_count=2)
        scaled_dfb = ttl.make_dataflow_buffer_like(attn_scratch, shape=(1, skv_tiles), block_count=2)
        max_dfb = ttl.make_dataflow_buffer_like(attn_scratch, shape=(1, 1), block_count=2)
        max_bc_dfb = ttl.make_dataflow_buffer_like(attn_scratch, shape=(1, skv_tiles), block_count=2)
        exp_dfb = ttl.make_dataflow_buffer_like(attn_scratch, shape=(1, skv_tiles), block_count=2)
        sum_dfb = ttl.make_dataflow_buffer_like(attn_scratch, shape=(1, 1), block_count=2)
        sum_bc_dfb = ttl.make_dataflow_buffer_like(attn_scratch, shape=(1, skv_tiles), block_count=2)
        attn_dfb = ttl.make_dataflow_buffer_like(attn_scratch, shape=(1, skv_tiles), block_count=2)
        out_dfb = ttl.make_dataflow_buffer_like(out, shape=(1, head_tiles), block_count=2)

        @ttl.compute()
        def compute():
            core_x, _ = ttl.node(dims=2)
            sc = sc_dfb.wait()
            for local_h in range(heads_per_core):
                h = core_x * heads_per_core + local_h
                if h < total_heads:
                    kt_dfb.reserve().store(ttl.transpose(k_dfb.wait()))
                    kt = kt_dfb.wait()
                    v = v_dfb.wait()
                    for s in range(sq_tiles):
                        qk_dfb.reserve().store(q_dfb.wait() @ kt)
                        qk = qk_dfb.wait()
                        scaled_dfb.reserve().store(qk * ttl.math.fill(qk, scale_val))
                        sdv = scaled_dfb.wait()
                        max_dfb.reserve().store(ttl.math.reduce_max(sdv, sc, dims=[1]))
                        mxb = max_bc_dfb.reserve()
                        mxb.store(ttl.math.broadcast(max_dfb.wait(), mxb, dims=[1]))
                        mxbv = max_bc_dfb.wait()
                        exp_dfb.reserve().store(ttl.math.exp(sdv - mxbv))
                        exv = exp_dfb.wait()
                        sum_dfb.reserve().store(ttl.math.reduce_sum(exv, sc, dims=[1]))
                        smb = sum_bc_dfb.reserve()
                        smb.store(ttl.math.broadcast(sum_dfb.wait(), smb, dims=[1]))
                        attn_dfb.reserve().store(exv / sum_bc_dfb.wait())
                        out_dfb.reserve().store(attn_dfb.wait() @ v)

        @ttl.datamovement()
        def dm_read():
            core_x, _ = ttl.node(dims=2)
            ttl.copy(scaler[0, 0], sc_dfb.reserve()).wait()
            for local_h in range(heads_per_core):
                h = core_x * heads_per_core + local_h
                if h < total_heads:
                    q_off = h * sq_tiles
                    kv_off = h * skv_tiles
                    ttl.copy(K_all[kv_off:kv_off + skv_tiles, 0:head_tiles], k_dfb.reserve()).wait()
                    ttl.copy(V_all[kv_off:kv_off + skv_tiles, 0:head_tiles], v_dfb.reserve()).wait()
                    for s in range(sq_tiles):
                        ttl.copy(Q_all[q_off + s:q_off + s + 1, 0:head_tiles], q_dfb.reserve()).wait()

        @ttl.datamovement()
        def dm_write():
            core_x, _ = ttl.node(dims=2)
            for local_h in range(heads_per_core):
                h = core_x * heads_per_core + local_h
                if h < total_heads:
                    q_off = h * sq_tiles
                    for s in range(sq_tiles):
                        ttl.copy(out_dfb.wait(), out[q_off + s:q_off + s + 1, 0:head_tiles]).wait()

    return sdpa_kernel
