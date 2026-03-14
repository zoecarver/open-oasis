"""Profile layernorm_d1024 kernel in isolation."""
import torch
import ttnn
import ttl

TILE = 32
D_MODEL = 1024
D_TILES = D_MODEL // TILE  # 32

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def to_tt(t, device):
    return ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                           device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)

def to_tt_l1(t, device):
    return ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                           device=device, memory_config=ttnn.L1_MEMORY_CONFIG)

def from_tt(t):
    return ttnn.to_torch(t)

def zeros_tt(shape, device):
    return to_tt(torch.zeros(shape, dtype=torch.bfloat16), device)

# ---------------------------------------------------------------------------
# Kernel: LayerNorm (3-pass: mean, var, normalize)
# ---------------------------------------------------------------------------

def make_layernorm_kernel(dim_tiles):
    @ttl.kernel(grid="auto")
    def layernorm_kernel(x, weight, ln_bias, scaler, mean_scale, out):
        grid_cols, _ = ttl.grid_size(dims=2)
        seq_tiles = x.shape[0] // TILE
        tiles_per_core = -(-seq_tiles // grid_cols)

        x_dfb = ttl.make_dataflow_buffer_like(x, shape=(1, 1), buffer_factor=2)
        sc_dfb = ttl.make_dataflow_buffer_like(scaler, shape=(1, 1), buffer_factor=1)
        ms_dfb = ttl.make_dataflow_buffer_like(mean_scale, shape=(1, 1), buffer_factor=1)
        red_dfb = ttl.make_dataflow_buffer_like(scaler, shape=(1, 1), buffer_factor=2)
        acc_dfb = ttl.make_dataflow_buffer_like(scaler, shape=(1, 1), buffer_factor=2)
        bcast_dfb = ttl.make_dataflow_buffer_like(x, shape=(1, 1), buffer_factor=2)
        sq_dfb = ttl.make_dataflow_buffer_like(x, shape=(1, 1), buffer_factor=2)
        mean_dfb = ttl.make_dataflow_buffer_like(x, shape=(1, 1), buffer_factor=2)
        istd_dfb = ttl.make_dataflow_buffer_like(x, shape=(1, 1), buffer_factor=2)
        w_dfb = ttl.make_dataflow_buffer_like(weight, shape=(1, 1), buffer_factor=2)
        b_dfb = ttl.make_dataflow_buffer_like(ln_bias, shape=(1, 1), buffer_factor=2)
        out_dfb = ttl.make_dataflow_buffer_like(out, shape=(1, 1), buffer_factor=2)

        @ttl.compute()
        def compute():
            core_x, _ = ttl.core(dims=2)
            with sc_dfb.wait() as sc, ms_dfb.wait() as ms:
                for local_t in range(tiles_per_core):
                    tile_idx = core_x * tiles_per_core + local_t
                    if tile_idx < seq_tiles:
                        # Pass 1: compute mean = sum(x) / N
                        with x_dfb.wait() as x0:
                            with red_dfb.reserve() as r:
                                r.store(ttl.math.reduce_sum(x0, sc, dims=[1]))
                        with red_dfb.wait() as rv, acc_dfb.reserve() as acc:
                            acc.store(rv)
                        for j in range(dim_tiles - 1):
                            with x_dfb.wait() as xj:
                                with red_dfb.reserve() as r:
                                    r.store(ttl.math.reduce_sum(xj, sc, dims=[1]))
                            with red_dfb.wait() as rv, acc_dfb.wait() as av, acc_dfb.reserve() as acc:
                                acc.store(av + rv)
                        with acc_dfb.wait() as sum_x, bcast_dfb.reserve() as bc:
                            bc.store(ttl.math.broadcast(sum_x, dims=[1]))
                        with bcast_dfb.wait() as sum_x_bc, mean_dfb.reserve() as mean_out:
                            mean_out.store(sum_x_bc * ms)

                        # Pass 2: compute var = sum((x - mean)^2) / N, then inv_std
                        with mean_dfb.wait() as mean_val:
                            with x_dfb.wait() as x0:
                                diff = x0 - mean_val
                                with sq_dfb.reserve() as sq:
                                    sq.store(diff * diff)
                            with sq_dfb.wait() as sqv, red_dfb.reserve() as r:
                                r.store(ttl.math.reduce_sum(sqv, sc, dims=[1]))
                            with red_dfb.wait() as rv, acc_dfb.reserve() as acc:
                                acc.store(rv)
                            for j in range(dim_tiles - 1):
                                with x_dfb.wait() as xj:
                                    diff = xj - mean_val
                                    with sq_dfb.reserve() as sq:
                                        sq.store(diff * diff)
                                with sq_dfb.wait() as sqv, red_dfb.reserve() as r:
                                    r.store(ttl.math.reduce_sum(sqv, sc, dims=[1]))
                                with red_dfb.wait() as rv, acc_dfb.wait() as av, acc_dfb.reserve() as acc:
                                    acc.store(av + rv)
                            with acc_dfb.wait() as sum_sq, bcast_dfb.reserve() as bc:
                                bc.store(ttl.math.broadcast(sum_sq, dims=[1]))
                            with bcast_dfb.wait() as var_bc, istd_dfb.reserve() as istd:
                                istd.store(ttl.math.rsqrt(var_bc * ms + ttl.math.fill(var_bc, 1e-6)))

                            # Pass 3: normalize = (x - mean) * inv_std * weight + bias
                            with istd_dfb.wait() as inv_std:
                                for j in range(dim_tiles):
                                    with x_dfb.wait() as xj, w_dfb.wait() as wj, b_dfb.wait() as bj, out_dfb.reserve() as o:
                                        o.store((xj - mean_val) * inv_std * wj + bj)

        @ttl.datamovement()
        def dm_read():
            core_x, _ = ttl.core(dims=2)
            with sc_dfb.reserve() as blk:
                tx = ttl.copy(scaler[0, 0], blk); tx.wait()
            with ms_dfb.reserve() as blk:
                tx = ttl.copy(mean_scale[0, 0], blk); tx.wait()
            for local_t in range(tiles_per_core):
                tile_idx = core_x * tiles_per_core + local_t
                if tile_idx < seq_tiles:
                    # Pass 1: x tiles for mean
                    for j in range(dim_tiles):
                        with x_dfb.reserve() as blk:
                            tx = ttl.copy(x[tile_idx, j], blk); tx.wait()
                    # Pass 2: x tiles for variance
                    for j in range(dim_tiles):
                        with x_dfb.reserve() as blk:
                            tx = ttl.copy(x[tile_idx, j], blk); tx.wait()
                    # Pass 3: x + weight + bias tiles for normalize
                    for j in range(dim_tiles):
                        with x_dfb.reserve() as blk:
                            tx = ttl.copy(x[tile_idx, j], blk); tx.wait()
                        with w_dfb.reserve() as blk:
                            tx = ttl.copy(weight[tile_idx, j], blk); tx.wait()
                        with b_dfb.reserve() as blk:
                            tx = ttl.copy(ln_bias[tile_idx, j], blk); tx.wait()

        @ttl.datamovement()
        def dm_write():
            core_x, _ = ttl.core(dims=2)
            for local_t in range(tiles_per_core):
                tile_idx = core_x * tiles_per_core + local_t
                if tile_idx < seq_tiles:
                    for j in range(dim_tiles):
                        with out_dfb.wait() as blk:
                            tx = ttl.copy(blk, out[tile_idx, j]); tx.wait()

    return layernorm_kernel

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    device = ttnn.open_device(device_id=0)

    SEQ = 320
    layernorm_d1024 = make_layernorm_kernel(D_TILES)

    x = torch.randn(SEQ, D_MODEL, dtype=torch.bfloat16)
    weight = torch.ones(D_MODEL, dtype=torch.bfloat16).unsqueeze(0).expand(SEQ, -1).contiguous()
    bias = torch.zeros(D_MODEL, dtype=torch.bfloat16).unsqueeze(0).expand(SEQ, -1).contiguous()
    scaler = torch.ones(TILE, TILE, dtype=torch.bfloat16)
    mean_scale = torch.full((TILE, TILE), 1.0 / D_MODEL, dtype=torch.bfloat16)

    x_tt = to_tt(x, device)
    w_tt = to_tt(weight, device)
    b_tt = to_tt(bias, device)
    sc_tt = to_tt_l1(scaler, device)
    ms_tt = to_tt_l1(mean_scale, device)
    out_tt = zeros_tt((SEQ, D_MODEL), device)

    layernorm_d1024(x_tt, w_tt, b_tt, sc_tt, ms_tt, out_tt)

    result = from_tt(out_tt)
    print(f"Result shape: {result.shape}")
    print(f"Result[0, :8]: {result[0, :8]}")
    print(f"Result mean: {result.float().mean().item():.6f}")
    print(f"Result std: {result.float().std().item():.6f}")

    ttnn.close_device(device)
