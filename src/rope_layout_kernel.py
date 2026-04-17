"""Fused RoPE + layout transform kernel.

Reads from qkv_full (SEQ, 5*D_MODEL), applies RoPE to Q and K,
writes Q,K,V directly in SDPA format: (BATCH, 1, N_PATCH_PAD, D_HEAD).

Eliminates: 5 slices + 2 RoPE kernels + 9 reshape/permute ops.
Output goes directly to ttnn.scaled_dot_product_attention (f32 accumulation).

Grid="auto" parallelizes over (frame, head) pairs.
"""
import ttl

TILE = 32


def make_rope_layout_kernel(seq_tiles, head_tiles, n_heads_val):
    """Fused RoPE + layout for spatial attention.
    Reads qkv_full (SEQ, 5*D_MODEL), produces q_out/k_out/v_out in SDPA layout.
    Each output is (T*N_HEADS, 1, N_PATCH_PAD, D_HEAD) stored as (T*N_HEADS, seq_tiles, head_tiles).
    """
    @ttl.operation(grid="auto")
    def rope_layout(qkv_full, cos_tab, sin_tab, q_out, k_out, v_out):
        grid_cols, _ = ttl.grid_size(dims=2)
        n_frames = qkv_full.shape[0] // TILE // seq_tiles
        total_heads = n_frames * n_heads_val
        heads_per_core = -(-total_heads // grid_cols)

        q_dfb = ttl.make_dataflow_buffer_like(qkv_full, shape=(seq_tiles, head_tiles), block_count=2)
        qs_dfb = ttl.make_dataflow_buffer_like(qkv_full, shape=(seq_tiles, head_tiles), block_count=2)
        k_dfb = ttl.make_dataflow_buffer_like(qkv_full, shape=(seq_tiles, head_tiles), block_count=2)
        ks_dfb = ttl.make_dataflow_buffer_like(qkv_full, shape=(seq_tiles, head_tiles), block_count=2)
        v_dfb = ttl.make_dataflow_buffer_like(qkv_full, shape=(seq_tiles, head_tiles), block_count=2)
        cos_dfb = ttl.make_dataflow_buffer_like(cos_tab, shape=(seq_tiles, head_tiles), block_count=2)
        sin_dfb = ttl.make_dataflow_buffer_like(sin_tab, shape=(seq_tiles, head_tiles), block_count=2)
        qr_dfb = ttl.make_dataflow_buffer_like(qkv_full, shape=(seq_tiles, head_tiles), block_count=2)
        kr_dfb = ttl.make_dataflow_buffer_like(qkv_full, shape=(seq_tiles, head_tiles), block_count=2)
        vo_dfb = ttl.make_dataflow_buffer_like(qkv_full, shape=(seq_tiles, head_tiles), block_count=2)

        @ttl.compute()
        def compute():
            core_x, _ = ttl.node(dims=2)
            for local_h in range(heads_per_core):
                head_idx = core_x * heads_per_core + local_h
                if head_idx < total_heads:
                    cv = cos_dfb.wait()
                    sv = sin_dfb.wait()
                    q = q_dfb.wait()
                    qs = qs_dfb.wait()
                    qr_dfb.reserve().store(q * cv + qs * sv)
                    k = k_dfb.wait()
                    ks = ks_dfb.wait()
                    kr_dfb.reserve().store(k * cv + ks * sv)
                    vo_dfb.reserve().store(v_dfb.wait())

        @ttl.datamovement()
        def dm_read():
            core_x, _ = ttl.node(dims=2)
            d_tiles = n_heads_val * head_tiles
            for local_h in range(heads_per_core):
                head_idx = core_x * heads_per_core + local_h
                if head_idx < total_heads:
                    frame = head_idx // n_heads_val
                    h = head_idx % n_heads_val
                    row_start = frame * seq_tiles
                    hc = h * head_tiles
                    ttl.copy(qkv_full[row_start:row_start + seq_tiles, hc:hc + head_tiles], q_dfb.reserve()).wait()
                    ttl.copy(qkv_full[row_start:row_start + seq_tiles, 3 * d_tiles + hc:3 * d_tiles + hc + head_tiles], qs_dfb.reserve()).wait()
                    ttl.copy(qkv_full[row_start:row_start + seq_tiles, d_tiles + hc:d_tiles + hc + head_tiles], k_dfb.reserve()).wait()
                    ttl.copy(qkv_full[row_start:row_start + seq_tiles, 4 * d_tiles + hc:4 * d_tiles + hc + head_tiles], ks_dfb.reserve()).wait()
                    ttl.copy(qkv_full[row_start:row_start + seq_tiles, 2 * d_tiles + hc:2 * d_tiles + hc + head_tiles], v_dfb.reserve()).wait()
                    ttl.copy(cos_tab[row_start:row_start + seq_tiles, hc:hc + head_tiles], cos_dfb.reserve()).wait()
                    ttl.copy(sin_tab[row_start:row_start + seq_tiles, hc:hc + head_tiles], sin_dfb.reserve()).wait()

        @ttl.datamovement()
        def dm_write():
            core_x, _ = ttl.node(dims=2)
            for local_h in range(heads_per_core):
                head_idx = core_x * heads_per_core + local_h
                if head_idx < total_heads:
                    # Output: (BATCH, N_PATCH_PAD, D_HEAD) as 2D: row = head_idx * seq_tiles
                    out_row = head_idx * seq_tiles
                    ttl.copy(qr_dfb.wait(), q_out[out_row:out_row + seq_tiles, 0:head_tiles]).wait()
                    ttl.copy(kr_dfb.wait(), k_out[out_row:out_row + seq_tiles, 0:head_tiles]).wait()
                    ttl.copy(vo_dfb.wait(), v_out[out_row:out_row + seq_tiles, 0:head_tiles]).wait()

    return rope_layout


def make_rope_temporal_kernel(head_tiles, n_heads_val):
    """Fused RoPE for temporal attention (no layout transform).
    Reads qkv_full (SEQ, 5*D_MODEL), applies RoPE to Q and K,
    writes q_out/k_out/v_out in (SEQ, D_MODEL) format.
    Eliminates: 5 slices + 2 RoPE kernels.
    Parallelizes over (row_tile, head) pairs.
    """
    @ttl.operation(grid="auto")
    def rope_temporal(qkv_full, cos_tab, sin_tab, q_out, k_out, v_out):
        grid_cols, _ = ttl.grid_size(dims=2)
        row_tiles = qkv_full.shape[0] // TILE
        total_units = row_tiles * n_heads_val
        units_per_core = -(-total_units // grid_cols)

        q_dfb = ttl.make_dataflow_buffer_like(qkv_full, shape=(1, head_tiles), block_count=2)
        qs_dfb = ttl.make_dataflow_buffer_like(qkv_full, shape=(1, head_tiles), block_count=2)
        k_dfb = ttl.make_dataflow_buffer_like(qkv_full, shape=(1, head_tiles), block_count=2)
        ks_dfb = ttl.make_dataflow_buffer_like(qkv_full, shape=(1, head_tiles), block_count=2)
        v_dfb = ttl.make_dataflow_buffer_like(qkv_full, shape=(1, head_tiles), block_count=2)
        cos_dfb = ttl.make_dataflow_buffer_like(cos_tab, shape=(1, head_tiles), block_count=2)
        sin_dfb = ttl.make_dataflow_buffer_like(sin_tab, shape=(1, head_tiles), block_count=2)
        qr_dfb = ttl.make_dataflow_buffer_like(qkv_full, shape=(1, head_tiles), block_count=2)
        kr_dfb = ttl.make_dataflow_buffer_like(qkv_full, shape=(1, head_tiles), block_count=2)
        vo_dfb = ttl.make_dataflow_buffer_like(qkv_full, shape=(1, head_tiles), block_count=2)

        @ttl.compute()
        def compute():
            core_x, _ = ttl.node(dims=2)
            for local_u in range(units_per_core):
                uid = core_x * units_per_core + local_u
                if uid < total_units:
                    cv = cos_dfb.wait()
                    sv = sin_dfb.wait()
                    q = q_dfb.wait()
                    qs = qs_dfb.wait()
                    qr_dfb.reserve().store(q * cv + qs * sv)
                    k = k_dfb.wait()
                    ks = ks_dfb.wait()
                    kr_dfb.reserve().store(k * cv + ks * sv)
                    vo_dfb.reserve().store(v_dfb.wait())

        @ttl.datamovement()
        def dm_read():
            core_x, _ = ttl.node(dims=2)
            d_tiles = n_heads_val * head_tiles
            for local_u in range(units_per_core):
                uid = core_x * units_per_core + local_u
                if uid < total_units:
                    row = uid // n_heads_val
                    h = uid % n_heads_val
                    hc = h * head_tiles
                    ttl.copy(qkv_full[row, hc:hc + head_tiles], q_dfb.reserve()).wait()
                    ttl.copy(qkv_full[row, 3 * d_tiles + hc:3 * d_tiles + hc + head_tiles], qs_dfb.reserve()).wait()
                    ttl.copy(qkv_full[row, d_tiles + hc:d_tiles + hc + head_tiles], k_dfb.reserve()).wait()
                    ttl.copy(qkv_full[row, 4 * d_tiles + hc:4 * d_tiles + hc + head_tiles], ks_dfb.reserve()).wait()
                    ttl.copy(qkv_full[row, 2 * d_tiles + hc:2 * d_tiles + hc + head_tiles], v_dfb.reserve()).wait()
                    ttl.copy(cos_tab[row, hc:hc + head_tiles], cos_dfb.reserve()).wait()
                    ttl.copy(sin_tab[row, hc:hc + head_tiles], sin_dfb.reserve()).wait()

        @ttl.datamovement()
        def dm_write():
            core_x, _ = ttl.node(dims=2)
            for local_u in range(units_per_core):
                uid = core_x * units_per_core + local_u
                if uid < total_units:
                    row = uid // n_heads_val
                    h = uid % n_heads_val
                    hc = h * head_tiles
                    ttl.copy(qr_dfb.wait(), q_out[row, hc:hc + head_tiles]).wait()
                    ttl.copy(kr_dfb.wait(), k_out[row, hc:hc + head_tiles]).wait()
                    ttl.copy(vo_dfb.wait(), v_out[row, hc:hc + head_tiles]).wait()

    return rope_temporal
