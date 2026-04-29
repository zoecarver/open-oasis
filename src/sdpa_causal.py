"""Multi-head SDPA with additive bias (for causal/padding mask).

Fork of sdpa.py with one extra input: a (sq_tiles*TILE, skv_tiles*TILE)
bias tensor that is added to the scaled QK before softmax. Caller fills
bias with a large negative value at masked positions. Bias is shared
across heads (one mask, many heads), so each head reads the same bias.

Used for the temporal SDPA path in oasis: T frames as the sequence, with
causal masking (key position s > query position q is masked) plus padding
masking when T < TILE.
"""
import ttl

TILE = 32


def make_sdpa_causal_kernel(sq_tiles, skv_tiles, head_tiles, scale_val, total_heads=None):
    """total_heads: pass explicitly when the input tensor's logical
    shape[0] does not match physical (e.g., a TILE_LAYOUT tensor with
    seq_len < TILE that gets interpreted as more heads than the logical
    leading dim suggests). Defaults to inferring from Q_all.shape[0]."""
    @ttl.operation(grid="auto", options="--no-ttl-reduce-full-fp32")
    def sdpa_causal_kernel(Q_all, K_all, V_all, scaler, bias, attn_scratch, out):
        grid_cols, _ = ttl.grid_size(dims=2)
        if total_heads is None:
            n_heads = Q_all.shape[0] // TILE // sq_tiles
        else:
            n_heads = total_heads
        heads_per_core = -(-n_heads // grid_cols)

        q_dfb = ttl.make_dataflow_buffer_like(Q_all, shape=(1, head_tiles), block_count=2)
        k_dfb = ttl.make_dataflow_buffer_like(K_all, shape=(skv_tiles, head_tiles), block_count=2)
        v_dfb = ttl.make_dataflow_buffer_like(V_all, shape=(skv_tiles, head_tiles), block_count=2)
        sc_dfb = ttl.make_dataflow_buffer_like(scaler, shape=(1, 1), block_count=1)
        bias_dfb = ttl.make_dataflow_buffer_like(bias, shape=(1, skv_tiles), block_count=2)

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
                if h < n_heads:
                    kt_dfb.reserve().store(ttl.transpose(k_dfb.wait()))
                    kt = kt_dfb.wait()
                    v = v_dfb.wait()
                    for s in range(sq_tiles):
                        qk_dfb.reserve().store(q_dfb.wait() @ kt)
                        qk = qk_dfb.wait()
                        scaled_dfb.reserve().store(qk * ttl.math.fill(qk, scale_val) + bias_dfb.wait())
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
                if h < n_heads:
                    q_off = h * sq_tiles
                    kv_off = h * skv_tiles
                    ttl.copy(K_all[kv_off:kv_off + skv_tiles, 0:head_tiles], k_dfb.reserve()).wait()
                    ttl.copy(V_all[kv_off:kv_off + skv_tiles, 0:head_tiles], v_dfb.reserve()).wait()
                    for s in range(sq_tiles):
                        ttl.copy(Q_all[q_off + s:q_off + s + 1, 0:head_tiles], q_dfb.reserve()).wait()
                        ttl.copy(bias[s:s + 1, 0:skv_tiles], bias_dfb.reserve()).wait()

        @ttl.datamovement()
        def dm_write():
            core_x, _ = ttl.node(dims=2)
            for local_h in range(heads_per_core):
                h = core_x * heads_per_core + local_h
                if h < n_heads:
                    q_off = h * sq_tiles
                    for s in range(sq_tiles):
                        ttl.copy(out_dfb.wait(), out[q_off + s:q_off + s + 1, 0:head_tiles]).wait()

    return sdpa_causal_kernel
