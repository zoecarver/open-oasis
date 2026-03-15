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
    @ttl.kernel(grid="auto")
    def rope_layout(qkv_full, cos_tab, sin_tab, q_out, k_out, v_out):
        grid_cols, _ = ttl.grid_size(dims=2)
        n_frames = qkv_full.shape[0] // TILE // seq_tiles
        total_heads = n_frames * n_heads_val
        heads_per_core = -(-total_heads // grid_cols)

        q_dfb = ttl.make_dataflow_buffer_like(qkv_full, shape=(seq_tiles, head_tiles), buffer_factor=2)
        qs_dfb = ttl.make_dataflow_buffer_like(qkv_full, shape=(seq_tiles, head_tiles), buffer_factor=2)
        k_dfb = ttl.make_dataflow_buffer_like(qkv_full, shape=(seq_tiles, head_tiles), buffer_factor=2)
        ks_dfb = ttl.make_dataflow_buffer_like(qkv_full, shape=(seq_tiles, head_tiles), buffer_factor=2)
        v_dfb = ttl.make_dataflow_buffer_like(qkv_full, shape=(seq_tiles, head_tiles), buffer_factor=2)
        cos_dfb = ttl.make_dataflow_buffer_like(cos_tab, shape=(seq_tiles, head_tiles), buffer_factor=2)
        sin_dfb = ttl.make_dataflow_buffer_like(sin_tab, shape=(seq_tiles, head_tiles), buffer_factor=2)
        # Compute in 2D, output in 3D (matching output tensor rank)
        qr_dfb = ttl.make_dataflow_buffer_like(qkv_full, shape=(seq_tiles, head_tiles), buffer_factor=2)
        kr_dfb = ttl.make_dataflow_buffer_like(qkv_full, shape=(seq_tiles, head_tiles), buffer_factor=2)
        vo_dfb = ttl.make_dataflow_buffer_like(qkv_full, shape=(seq_tiles, head_tiles), buffer_factor=2)

        @ttl.compute()
        def compute():
            core_x, _ = ttl.core(dims=2)
            for local_h in range(heads_per_core):
                head_idx = core_x * heads_per_core + local_h
                if head_idx < total_heads:
                    with cos_dfb.wait() as cv, sin_dfb.wait() as sv:
                        with q_dfb.wait() as q, qs_dfb.wait() as qs, qr_dfb.reserve() as qr:
                            qr.store(q * cv + qs * sv)
                        with k_dfb.wait() as k, ks_dfb.wait() as ks, kr_dfb.reserve() as kr:
                            kr.store(k * cv + ks * sv)
                    with v_dfb.wait() as vv, vo_dfb.reserve() as vo:
                        vo.store(vv)

        @ttl.datamovement()
        def dm_read():
            core_x, _ = ttl.core(dims=2)
            d_tiles = n_heads_val * head_tiles
            for local_h in range(heads_per_core):
                head_idx = core_x * heads_per_core + local_h
                if head_idx < total_heads:
                    frame = head_idx // n_heads_val
                    h = head_idx % n_heads_val
                    row_start = frame * seq_tiles
                    hc = h * head_tiles
                    with q_dfb.reserve() as blk:
                        tx = ttl.copy(qkv_full[row_start:row_start + seq_tiles, hc:hc + head_tiles], blk); tx.wait()
                    with qs_dfb.reserve() as blk:
                        tx = ttl.copy(qkv_full[row_start:row_start + seq_tiles, 3 * d_tiles + hc:3 * d_tiles + hc + head_tiles], blk); tx.wait()
                    with k_dfb.reserve() as blk:
                        tx = ttl.copy(qkv_full[row_start:row_start + seq_tiles, d_tiles + hc:d_tiles + hc + head_tiles], blk); tx.wait()
                    with ks_dfb.reserve() as blk:
                        tx = ttl.copy(qkv_full[row_start:row_start + seq_tiles, 4 * d_tiles + hc:4 * d_tiles + hc + head_tiles], blk); tx.wait()
                    with v_dfb.reserve() as blk:
                        tx = ttl.copy(qkv_full[row_start:row_start + seq_tiles, 2 * d_tiles + hc:2 * d_tiles + hc + head_tiles], blk); tx.wait()
                    with cos_dfb.reserve() as blk:
                        tx = ttl.copy(cos_tab[row_start:row_start + seq_tiles, hc:hc + head_tiles], blk); tx.wait()
                    with sin_dfb.reserve() as blk:
                        tx = ttl.copy(sin_tab[row_start:row_start + seq_tiles, hc:hc + head_tiles], blk); tx.wait()

        @ttl.datamovement()
        def dm_write():
            core_x, _ = ttl.core(dims=2)
            for local_h in range(heads_per_core):
                head_idx = core_x * heads_per_core + local_h
                if head_idx < total_heads:
                    # Output: (BATCH, N_PATCH_PAD, D_HEAD) as 2D: row = head_idx * seq_tiles
                    out_row = head_idx * seq_tiles
                    with qr_dfb.wait() as blk:
                        tx = ttl.copy(blk, q_out[out_row:out_row + seq_tiles, 0:head_tiles]); tx.wait()
                    with kr_dfb.wait() as blk:
                        tx = ttl.copy(blk, k_out[out_row:out_row + seq_tiles, 0:head_tiles]); tx.wait()
                    with vo_dfb.wait() as blk:
                        tx = ttl.copy(blk, v_out[out_row:out_row + seq_tiles, 0:head_tiles]); tx.wait()

    return rope_layout
