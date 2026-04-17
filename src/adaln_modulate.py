"""adaLN modulate kernel: out = x * (scale + 1) + shift."""
import ttl

TILE = 32
ELEM_GRAN = 8


@ttl.operation(grid="auto")
def adaln_modulate_kernel(x, shift, scale, out):
    grid_cols, _ = ttl.grid_size(dims=2)
    row_tiles = x.shape[0] // TILE
    col_blocks = x.shape[1] // TILE // ELEM_GRAN
    total = row_tiles * col_blocks
    tiles_per_core = -(-total // grid_cols)
    x_dfb = ttl.make_dataflow_buffer_like(x, shape=(1, ELEM_GRAN), block_count=2)
    sh_dfb = ttl.make_dataflow_buffer_like(shift, shape=(1, ELEM_GRAN), block_count=2)
    sc_dfb = ttl.make_dataflow_buffer_like(scale, shape=(1, ELEM_GRAN), block_count=2)
    out_dfb = ttl.make_dataflow_buffer_like(out, shape=(1, ELEM_GRAN), block_count=2)

    @ttl.compute()
    def compute():
        core_x, _ = ttl.node(dims=2)
        for local_t in range(tiles_per_core):
            t = core_x * tiles_per_core + local_t
            if t < total:
                xv = x_dfb.wait()
                shv = sh_dfb.wait()
                scv = sc_dfb.wait()
                out_dfb.reserve().store(xv * (scv + ttl.math.fill(scv, 1.0)) + shv)

    @ttl.datamovement()
    def dm_read():
        core_x, _ = ttl.node(dims=2)
        for local_t in range(tiles_per_core):
            t = core_x * tiles_per_core + local_t
            if t < total:
                row = t // col_blocks
                cb = t % col_blocks
                sc = cb * ELEM_GRAN
                ttl.copy(x[row, sc:sc + ELEM_GRAN], x_dfb.reserve()).wait()
                ttl.copy(shift[row, sc:sc + ELEM_GRAN], sh_dfb.reserve()).wait()
                ttl.copy(scale[row, sc:sc + ELEM_GRAN], sc_dfb.reserve()).wait()

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
