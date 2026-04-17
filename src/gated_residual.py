"""Gated residual kernel: out = residual + x * gate."""
import ttl

TILE = 32
ELEM_GRAN = 8


@ttl.operation(grid="auto")
def gated_residual_kernel(residual, x, gate, out):
    grid_cols, _ = ttl.grid_size(dims=2)
    row_tiles = residual.shape[0] // TILE
    col_blocks = residual.shape[1] // TILE // ELEM_GRAN
    total = row_tiles * col_blocks
    tiles_per_core = -(-total // grid_cols)
    res_dfb = ttl.make_dataflow_buffer_like(residual, shape=(1, ELEM_GRAN), block_count=2)
    x_dfb = ttl.make_dataflow_buffer_like(x, shape=(1, ELEM_GRAN), block_count=2)
    g_dfb = ttl.make_dataflow_buffer_like(gate, shape=(1, ELEM_GRAN), block_count=2)
    out_dfb = ttl.make_dataflow_buffer_like(out, shape=(1, ELEM_GRAN), block_count=2)

    @ttl.compute()
    def compute():
        core_x, _ = ttl.node(dims=2)
        for local_t in range(tiles_per_core):
            t = core_x * tiles_per_core + local_t
            if t < total:
                rv = res_dfb.wait()
                xv = x_dfb.wait()
                gv = g_dfb.wait()
                o = out_dfb.reserve()
                o.store(rv + xv * gv)

    @ttl.datamovement()
    def dm_read():
        core_x, _ = ttl.node(dims=2)
        for local_t in range(tiles_per_core):
            t = core_x * tiles_per_core + local_t
            if t < total:
                row = t // col_blocks
                cb = t % col_blocks
                sc = cb * ELEM_GRAN
                ttl.copy(residual[row, sc:sc + ELEM_GRAN], res_dfb.reserve()).wait()
                ttl.copy(x[row, sc:sc + ELEM_GRAN], x_dfb.reserve()).wait()
                ttl.copy(gate[row, sc:sc + ELEM_GRAN], g_dfb.reserve()).wait()

    @ttl.datamovement()
    def dm_write():
        core_x, _ = ttl.node(dims=2)
        for local_t in range(tiles_per_core):
            t = core_x * tiles_per_core + local_t
            if t < total:
                row = t // col_blocks
                cb = t % col_blocks
                sc = cb * ELEM_GRAN
                ttl.copy(out_dfb.wait(), out[row, sc:sc + ELEM_GRAN]).wait()
