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
            for local_t in range(tiles_per_core):
                t = core_x * tiles_per_core + local_t
                if t < total:
                    row = t // n_heads
                    h = t % n_heads
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
            for local_t in range(tiles_per_core):
                t = core_x * tiles_per_core + local_t
                if t < total:
                    row = t // n_heads
                    h = t % n_heads
                    out_row = h * seq_tiles + row
                    ttl.copy(qr_dfb.wait(), q_out[out_row, 0:head_tiles]).wait()
                    ttl.copy(kr_dfb.wait(), k_out[out_row, 0:head_tiles]).wait()
                    ttl.copy(vo_dfb.wait(), v_out[out_row, 0:head_tiles]).wait()

    return vae_rope_kernel
