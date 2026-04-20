"""LayerNorm kernel with affine transform: out = (x - mean) / std * weight + bias.

Three-pass streaming: mean, variance, normalize.
Factory function parameterized by dim_tiles (number of tiles in the reduction dimension).

Intermediate DFBs (red_dfb, sq_dfb, mean_dfb, istd_dfb, bcast_dfb) are required:
reduce and broadcast outputs must be materialized to a DFB before any elementwise
consumer, per TT-Lang's DFB-arg-op fusion rules.
"""
import ttl

TILE = 32


def make_layernorm_kernel(dim_tiles):
    @ttl.operation(grid="auto")
    def layernorm_kernel(x, weight, ln_bias, scaler, mean_scale, out):
        grid_cols, _ = ttl.grid_size(dims=2)
        seq_tiles = x.shape[0] // TILE
        tiles_per_core = -(-seq_tiles // grid_cols)
        x_dfb = ttl.make_dataflow_buffer_like(x, shape=(1, 1), block_count=2)
        sc_dfb = ttl.make_dataflow_buffer_like(scaler, shape=(1, 1), block_count=1)
        ms_dfb = ttl.make_dataflow_buffer_like(mean_scale, shape=(1, 1), block_count=1)
        red_dfb = ttl.make_dataflow_buffer_like(scaler, shape=(1, 1), block_count=2)
        acc_dfb = ttl.make_dataflow_buffer_like(scaler, shape=(1, 1), block_count=2)
        sq_dfb = ttl.make_dataflow_buffer_like(x, shape=(1, 1), block_count=2)
        bcast_dfb = ttl.make_dataflow_buffer_like(x, shape=(1, 1), block_count=2)
        mean_dfb = ttl.make_dataflow_buffer_like(x, shape=(1, 1), block_count=2)
        istd_dfb = ttl.make_dataflow_buffer_like(x, shape=(1, 1), block_count=2)
        w_dfb = ttl.make_dataflow_buffer_like(weight, shape=(1, 1), block_count=2)
        b_dfb = ttl.make_dataflow_buffer_like(ln_bias, shape=(1, 1), block_count=2)
        out_dfb = ttl.make_dataflow_buffer_like(out, shape=(1, 1), block_count=2)

        @ttl.compute()
        def compute():
            core_x, _ = ttl.node(dims=2)
            sc = sc_dfb.wait()
            ms = ms_dfb.wait()
            for local_t in range(tiles_per_core):
                tile_idx = core_x * tiles_per_core + local_t
                if tile_idx < seq_tiles:
                    # Pass 1: mean
                    red_dfb.reserve().store(ttl.math.reduce_sum(x_dfb.wait(), sc, dims=[1]))
                    acc_dfb.reserve().store(red_dfb.wait())
                    for j in range(dim_tiles - 1):
                        red_dfb.reserve().store(ttl.math.reduce_sum(x_dfb.wait(), sc, dims=[1]))
                        acc_dfb.reserve().store(acc_dfb.wait() + red_dfb.wait())
                    bc1 = bcast_dfb.reserve()
                    bc1.store(ttl.math.broadcast(acc_dfb.wait(), bc1, dims=[1]))
                    mean_dfb.reserve().store(bcast_dfb.wait() * ms)
                    mean_val = mean_dfb.wait()
                    # Pass 2: variance
                    diff0 = x_dfb.wait() - mean_val
                    sq_dfb.reserve().store(diff0 * diff0)
                    red_dfb.reserve().store(ttl.math.reduce_sum(sq_dfb.wait(), sc, dims=[1]))
                    acc_dfb.reserve().store(red_dfb.wait())
                    for j in range(dim_tiles - 1):
                        diffj = x_dfb.wait() - mean_val
                        sq_dfb.reserve().store(diffj * diffj)
                        red_dfb.reserve().store(ttl.math.reduce_sum(sq_dfb.wait(), sc, dims=[1]))
                        acc_dfb.reserve().store(acc_dfb.wait() + red_dfb.wait())
                    bc2 = bcast_dfb.reserve()
                    bc2.store(ttl.math.broadcast(acc_dfb.wait(), bc2, dims=[1]))
                    istd_dfb.reserve().store(ttl.math.rsqrt(bcast_dfb.wait() * ms + ttl.math.fill(ms, 1e-6)))
                    inv_std = istd_dfb.wait()
                    # Pass 3: normalize + affine
                    for j in range(dim_tiles):
                        out_dfb.reserve().store((x_dfb.wait() - mean_val) * inv_std * w_dfb.wait() + b_dfb.wait())

        @ttl.datamovement()
        def dm_read():
            core_x, _ = ttl.node(dims=2)
            ttl.copy(scaler[0, 0], sc_dfb.reserve()).wait()
            ttl.copy(mean_scale[0, 0], ms_dfb.reserve()).wait()
            for local_t in range(tiles_per_core):
                tile_idx = core_x * tiles_per_core + local_t
                if tile_idx < seq_tiles:
                    for j in range(dim_tiles):
                        ttl.copy(x[tile_idx, j], x_dfb.reserve()).wait()
                    for j in range(dim_tiles):
                        ttl.copy(x[tile_idx, j], x_dfb.reserve()).wait()
                    for j in range(dim_tiles):
                        ttl.copy(x[tile_idx, j], x_dfb.reserve()).wait()
                        ttl.copy(weight[tile_idx, j], w_dfb.reserve()).wait()
                        ttl.copy(ln_bias[tile_idx, j], b_dfb.reserve()).wait()

        @ttl.datamovement()
        def dm_write():
            core_x, _ = ttl.node(dims=2)
            for local_t in range(tiles_per_core):
                tile_idx = core_x * tiles_per_core + local_t
                if tile_idx < seq_tiles:
                    for j in range(dim_tiles):
                        ttl.copy(out_dfb.wait(), out[tile_idx, j]).wait()

    return layernorm_kernel
