"""Linear (matmul) kernel: out = x @ w.

Factory function parameterized by k_chunk (K dimension tile count).
Uses compiler K-accumulation for the matmul."""
import ttl

TILE = 32


def make_linear_kernel(k_chunk):
    @ttl.operation(grid="auto")
    def linear_kernel(x, w, out):
        grid_cols, _ = ttl.grid_size(dims=2)
        m_tiles = x.shape[0] // TILE
        n_tiles = w.shape[1] // TILE
        total_out = m_tiles * n_tiles
        tiles_per_core = -(-total_out // grid_cols)
        x_dfb = ttl.make_dataflow_buffer_like(x, shape=(1, k_chunk), block_count=2)
        w_dfb = ttl.make_dataflow_buffer_like(w, shape=(k_chunk, 1), block_count=2)
        out_dfb = ttl.make_dataflow_buffer_like(out, shape=(1, 1), block_count=2)

        @ttl.compute()
        def compute():
            core_x, _ = ttl.node(dims=2)
            for local_t in range(tiles_per_core):
                idx = core_x * tiles_per_core + local_t
                if idx < total_out:
                    xv = x_dfb.wait()
                    wv = w_dfb.wait()
                    out_dfb.reserve().store(xv @ wv)

        @ttl.datamovement()
        def dm_read():
            core_x, _ = ttl.node(dims=2)
            for local_t in range(tiles_per_core):
                idx = core_x * tiles_per_core + local_t
                if idx < total_out:
                    row = idx // n_tiles
                    col = idx % n_tiles
                    ttl.copy(x[row, 0:k_chunk], x_dfb.reserve()).wait()
                    ttl.copy(w[0:k_chunk, col], w_dfb.reserve()).wait()

        @ttl.datamovement()
        def dm_write():
            core_x, _ = ttl.node(dims=2)
            for local_t in range(tiles_per_core):
                idx = core_x * tiles_per_core + local_t
                if idx < total_out:
                    row = idx // n_tiles
                    col = idx % n_tiles
                    ttl.copy(out_dfb.wait(), out[row, col]).wait()

    return linear_kernel
