"""Fused adaLN params: matmul + bias + row-expand in one kernel.

Computes: out = expand(silu_cond @ adaln_w + adaln_b, n_repeat)

Grid="auto" parallelizes over output columns. Each core computes
K-accumulated matmul for its columns, adds bias, writes to N_REPEAT rows.
"""
import ttl

TILE = 32


def make_adaln_matmul_expand_kernel(k_tiles, n_repeat):
    @ttl.operation(grid="auto")
    def adaln_matmul_expand(silu_cond, adaln_w, adaln_b, out):
        grid_cols, _ = ttl.grid_size(dims=2)
        out_col_tiles = adaln_w.shape[1] // TILE
        cols_per_core = -(-out_col_tiles // grid_cols)

        cond_dfb = ttl.make_dataflow_buffer_like(silu_cond, shape=(1, 1), block_count=2)
        w_dfb = ttl.make_dataflow_buffer_like(adaln_w, shape=(1, 1), block_count=2)
        mm_dfb = ttl.make_dataflow_buffer_like(out, shape=(1, 1), block_count=2)
        acc_dfb = ttl.make_dataflow_buffer_like(out, shape=(1, 1), block_count=2)
        b_dfb = ttl.make_dataflow_buffer_like(adaln_b, shape=(1, 1), block_count=2)
        out_dfb = ttl.make_dataflow_buffer_like(out, shape=(1, 1), block_count=2)

        @ttl.compute()
        def compute():
            core_x, _ = ttl.node(dims=2)
            for local_c in range(cols_per_core):
                col = core_x * cols_per_core + local_c
                if col < out_col_tiles:
                    # K-accumulate: silu_cond @ adaln_w[:, col]
                    cv = cond_dfb.wait()
                    wv = w_dfb.wait()
                    acc_dfb.reserve().store(cv @ wv)
                    for k in range(k_tiles - 1):
                        cv = cond_dfb.wait()
                        wv = w_dfb.wait()
                        mm_dfb.reserve().store(cv @ wv)
                        mv = mm_dfb.wait()
                        prev = acc_dfb.wait()
                        acc_dfb.reserve().store(prev + mv)
                    av = acc_dfb.wait()
                    bv = b_dfb.wait()
                    for rep in range(n_repeat):
                        out_dfb.reserve().store(av + bv)

        @ttl.datamovement()
        def dm_read():
            core_x, _ = ttl.node(dims=2)
            for local_c in range(cols_per_core):
                col = core_x * cols_per_core + local_c
                if col < out_col_tiles:
                    for k in range(k_tiles):
                        ttl.copy(silu_cond[0, k], cond_dfb.reserve()).wait()
                        ttl.copy(adaln_w[k, col], w_dfb.reserve()).wait()
                    ttl.copy(adaln_b[0, col], b_dfb.reserve()).wait()

        @ttl.datamovement()
        def dm_write():
            core_x, _ = ttl.node(dims=2)
            for local_c in range(cols_per_core):
                col = core_x * cols_per_core + local_c
                if col < out_col_tiles:
                    for rep in range(n_repeat):
                        ttl.copy(out_dfb.wait(), out[rep, col]).wait()

    return adaln_matmul_expand
