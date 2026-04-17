"""Fused RoPE for VAE decoder: reads qkv_full, writes Q_roped, K_roped, V
in heads-first layout (N_HEADS * SEQ, D_HEAD) ready for single reshape to SDPA.
Eliminates 5 slices + 2 fused_rope calls + 3 reshape + 3 permute = 13 ops -> 1."""
import ttl

TILE = 32

def make_vae_rope_kernel(n_heads, head_tiles):
    @ttl.operation(grid="auto")
    def vae_rope_kernel(qkv_full, cos_tab, sin_tab, q_out, k_out, v_out):
        grid_cols, _ = ttl.grid_size(dims=2)
        seq_tiles = qkv_full.shape[0] // TILE
        total = seq_tiles * n_heads
        tiles_per_core = -(-total // grid_cols)
        d_tiles = n_heads * head_tiles

        q_dfb = ttl.make_dataflow_buffer_like(qkv_full, shape=(1, head_tiles), block_count=2)
        qs_dfb = ttl.make_dataflow_buffer_like(qkv_full, shape=(1, head_tiles), block_count=2)
        k_dfb = ttl.make_dataflow_buffer_like(qkv_full, shape=(1, head_tiles), block_count=2)
        ks_dfb = ttl.make_dataflow_buffer_like(qkv_full, shape=(1, head_tiles), block_count=2)
        v_dfb = ttl.make_dataflow_buffer_like(qkv_full, shape=(1, head_tiles), block_count=2)
        cos_dfb = ttl.make_dataflow_buffer_like(cos_tab, shape=(1, head_tiles), block_count=2)
        sin_dfb = ttl.make_dataflow_buffer_like(sin_tab, shape=(1, head_tiles), block_count=2)
        qr_dfb = ttl.make_dataflow_buffer_like(q_out, shape=(1, head_tiles), block_count=2)
        kr_dfb = ttl.make_dataflow_buffer_like(k_out, shape=(1, head_tiles), block_count=2)
        vo_dfb = ttl.make_dataflow_buffer_like(v_out, shape=(1, head_tiles), block_count=2)

        @ttl.compute()
        def compute():
            core_x, _ = ttl.node(dims=2)
            for local_t in range(tiles_per_core):
                t = core_x * tiles_per_core + local_t
                if t < total:
                    with cos_dfb.wait() as cv, sin_dfb.wait() as sv:
                        with q_dfb.wait() as q, qs_dfb.wait() as qs, qr_dfb.reserve() as qr:
                            qr.store(q * cv + qs * sv)
                        with k_dfb.wait() as k, ks_dfb.wait() as ks, kr_dfb.reserve() as kr:
                            kr.store(k * cv + ks * sv)
                    with v_dfb.wait() as vv, vo_dfb.reserve() as vo:
                        vo.store(vv)

        @ttl.datamovement()
        def dm_read():
            core_x, _ = ttl.node(dims=2)
            for local_t in range(tiles_per_core):
                t = core_x * tiles_per_core + local_t
                if t < total:
                    row = t // n_heads
                    h = t % n_heads
                    hc = h * head_tiles
                    with q_dfb.reserve() as blk:
                        tx = ttl.copy(qkv_full[row, hc:hc + head_tiles], blk); tx.wait()
                    with qs_dfb.reserve() as blk:
                        tx = ttl.copy(qkv_full[row, 3 * d_tiles + hc:3 * d_tiles + hc + head_tiles], blk); tx.wait()
                    with k_dfb.reserve() as blk:
                        tx = ttl.copy(qkv_full[row, d_tiles + hc:d_tiles + hc + head_tiles], blk); tx.wait()
                    with ks_dfb.reserve() as blk:
                        tx = ttl.copy(qkv_full[row, 4 * d_tiles + hc:4 * d_tiles + hc + head_tiles], blk); tx.wait()
                    with v_dfb.reserve() as blk:
                        tx = ttl.copy(qkv_full[row, 2 * d_tiles + hc:2 * d_tiles + hc + head_tiles], blk); tx.wait()
                    with cos_dfb.reserve() as blk:
                        tx = ttl.copy(cos_tab[row, hc:hc + head_tiles], blk); tx.wait()
                    with sin_dfb.reserve() as blk:
                        tx = ttl.copy(sin_tab[row, hc:hc + head_tiles], blk); tx.wait()

        @ttl.datamovement()
        def dm_write():
            core_x, _ = ttl.node(dims=2)
            for local_t in range(tiles_per_core):
                t = core_x * tiles_per_core + local_t
                if t < total:
                    row = t // n_heads
                    h = t % n_heads
                    out_row = h * seq_tiles + row
                    with qr_dfb.wait() as blk:
                        tx = ttl.copy(blk, q_out[out_row, 0:head_tiles]); tx.wait()
                    with kr_dfb.wait() as blk:
                        tx = ttl.copy(blk, k_out[out_row, 0:head_tiles]); tx.wait()
                    with vo_dfb.wait() as blk:
                        tx = ttl.copy(blk, v_out[out_row, 0:head_tiles]); tx.wait()

    return vae_rope_kernel
