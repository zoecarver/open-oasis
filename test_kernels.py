"""
Test basic TT-Lang kernels for Oasis DiT against PyTorch reference.
Each kernel is tested in isolation with small tensors, comparing allclose.

Oasis model dims:
  D_MODEL=1024, N_HEADS=16, D_HEAD=64, D_MLP=4096
  N_PATCHES=144 (9x16 grid), padded to 160 (5 tiles)
  Patch size=2, input latent=18x32x16
"""
import torch
import ttnn
import ttl

TILE = 32

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

def check(name, result, expected, rows=None, cols=None, atol=0.5, rtol=0.1):
    r = result.float()
    e = expected.float()
    if rows is not None:
        r = r[:rows]
        e = e[:rows]
    if cols is not None:
        r = r[:, :cols]
        e = e[:, :cols]
    max_err = (r - e).abs().max().item()
    mean_err = (r - e).abs().mean().item()
    ok = torch.allclose(r, e, atol=atol, rtol=rtol)
    status = "PASS" if ok else "FAIL"
    print(f"  [{status}] {name}: max_err={max_err:.6f} mean_err={mean_err:.6f}")
    if not ok:
        print(f"    expected range: [{e.min().item():.4f}, {e.max().item():.4f}]")
        print(f"    got range:      [{r.min().item():.4f}, {r.max().item():.4f}]")
    return ok

# ---------------------------------------------------------------------------
# Kernel: Linear (single-pass K, no bias)
# ---------------------------------------------------------------------------

def make_linear_kernel(k_chunk):
    @ttl.kernel(grid="auto")
    def linear_kernel(x, w, out):
        grid_cols, _ = ttl.grid_size(dims=2)
        m_tiles = x.shape[0] // TILE
        n_tiles = w.shape[1] // TILE
        total_out = m_tiles * n_tiles
        tiles_per_core = -(-total_out // grid_cols)
        x_dfb = ttl.make_dataflow_buffer_like(x, shape=(1, k_chunk), buffer_factor=2)
        w_dfb = ttl.make_dataflow_buffer_like(w, shape=(k_chunk, 1), buffer_factor=2)
        out_dfb = ttl.make_dataflow_buffer_like(out, shape=(1, 1), buffer_factor=2)
        @ttl.compute()
        def compute():
            core_x, _ = ttl.core(dims=2)
            for local_t in range(tiles_per_core):
                idx = core_x * tiles_per_core + local_t
                if idx < total_out:
                    with x_dfb.wait() as xv, w_dfb.wait() as wv, out_dfb.reserve() as o:
                        o.store(xv @ wv)
        @ttl.datamovement()
        def dm_read():
            core_x, _ = ttl.core(dims=2)
            for local_t in range(tiles_per_core):
                idx = core_x * tiles_per_core + local_t
                if idx < total_out:
                    row = idx // n_tiles
                    col = idx % n_tiles
                    with x_dfb.reserve() as blk:
                        tx = ttl.copy(x[row, 0:k_chunk], blk); tx.wait()
                    with w_dfb.reserve() as blk:
                        tx = ttl.copy(w[0:k_chunk, col], blk); tx.wait()
        @ttl.datamovement()
        def dm_write():
            core_x, _ = ttl.core(dims=2)
            for local_t in range(tiles_per_core):
                idx = core_x * tiles_per_core + local_t
                if idx < total_out:
                    row = idx // n_tiles
                    col = idx % n_tiles
                    with out_dfb.wait() as blk:
                        tx = ttl.copy(blk, out[row, col]); tx.wait()
    return linear_kernel

# ---------------------------------------------------------------------------
# Kernel: Linear with bias (single-pass K)
# ---------------------------------------------------------------------------

def make_linear_bias_kernel(k_chunk, n_chunk=1):
    @ttl.kernel(grid="auto")
    def linear_bias_kernel(x, w, bias, out):
        grid_cols, _ = ttl.grid_size(dims=2)
        m_tiles = x.shape[0] // TILE
        n_blocks = w.shape[1] // TILE // n_chunk
        total_out = m_tiles * n_blocks
        tiles_per_core = -(-total_out // grid_cols)
        x_dfb = ttl.make_dataflow_buffer_like(x, shape=(1, k_chunk), buffer_factor=2)
        w_dfb = ttl.make_dataflow_buffer_like(w, shape=(k_chunk, n_chunk), buffer_factor=2)
        b_dfb = ttl.make_dataflow_buffer_like(bias, shape=(1, n_chunk), buffer_factor=2)
        mm_dfb = ttl.make_dataflow_buffer_like(out, shape=(1, n_chunk), buffer_factor=2)
        out_dfb = ttl.make_dataflow_buffer_like(out, shape=(1, n_chunk), buffer_factor=2)
        @ttl.compute()
        def compute():
            core_x, _ = ttl.core(dims=2)
            for local_t in range(tiles_per_core):
                idx = core_x * tiles_per_core + local_t
                if idx < total_out:
                    with x_dfb.wait() as xv, w_dfb.wait() as wv, mm_dfb.reserve() as mm:
                        mm.store(xv @ wv)
                    with mm_dfb.wait() as mmv, b_dfb.wait() as bv, out_dfb.reserve() as o:
                        o.store(mmv + bv)
        @ttl.datamovement()
        def dm_read():
            core_x, _ = ttl.core(dims=2)
            for local_t in range(tiles_per_core):
                idx = core_x * tiles_per_core + local_t
                if idx < total_out:
                    row = idx // n_blocks
                    cb = idx % n_blocks
                    sc = cb * n_chunk
                    with x_dfb.reserve() as blk:
                        tx = ttl.copy(x[row, 0:k_chunk], blk); tx.wait()
                    with w_dfb.reserve() as blk:
                        tx = ttl.copy(w[0:k_chunk, sc:sc + n_chunk], blk); tx.wait()
                    with b_dfb.reserve() as blk:
                        tx = ttl.copy(bias[row, sc:sc + n_chunk], blk); tx.wait()
        @ttl.datamovement()
        def dm_write():
            core_x, _ = ttl.core(dims=2)
            for local_t in range(tiles_per_core):
                idx = core_x * tiles_per_core + local_t
                if idx < total_out:
                    row = idx // n_blocks
                    cb = idx % n_blocks
                    sc = cb * n_chunk
                    with out_dfb.wait() as blk:
                        tx = ttl.copy(blk, out[row, sc:sc + n_chunk]); tx.wait()
    return linear_bias_kernel

# ---------------------------------------------------------------------------
# Kernel: Linear with K accumulation (multi-pass for large K like MLP down)
# ---------------------------------------------------------------------------

def make_linear_accum_kernel(k_chunk, k_iters):
    @ttl.kernel(grid="auto")
    def linear_accum_kernel(x, w, out):
        grid_cols, _ = ttl.grid_size(dims=2)
        m_tiles = x.shape[0] // TILE
        n_tiles = w.shape[1] // TILE
        total_out = m_tiles * n_tiles
        tiles_per_core = -(-total_out // grid_cols)
        x_dfb = ttl.make_dataflow_buffer_like(x, shape=(1, k_chunk), buffer_factor=2)
        w_dfb = ttl.make_dataflow_buffer_like(w, shape=(k_chunk, 1), buffer_factor=2)
        mm_dfb = ttl.make_dataflow_buffer_like(out, shape=(1, 1), buffer_factor=2)
        acc_dfb = ttl.make_dataflow_buffer_like(out, shape=(1, 1), buffer_factor=2)
        out_dfb = ttl.make_dataflow_buffer_like(out, shape=(1, 1), buffer_factor=2)
        @ttl.compute()
        def compute():
            core_x, _ = ttl.core(dims=2)
            for local_t in range(tiles_per_core):
                idx = core_x * tiles_per_core + local_t
                if idx < total_out:
                    # First K chunk: matmul directly into accumulator
                    with x_dfb.wait() as xv, w_dfb.wait() as wv, acc_dfb.reserve() as acc:
                        acc.store(xv @ wv)
                    # Remaining K chunks: matmul into mm, accumulate
                    for ki in range(k_iters - 1):
                        with x_dfb.wait() as xv, w_dfb.wait() as wv, mm_dfb.reserve() as mm:
                            mm.store(xv @ wv)
                        with mm_dfb.wait() as mmv, acc_dfb.wait() as av, acc_dfb.reserve() as acc:
                            acc.store(av + mmv)
                    with acc_dfb.wait() as final, out_dfb.reserve() as o:
                        o.store(final)
        @ttl.datamovement()
        def dm_read():
            core_x, _ = ttl.core(dims=2)
            for local_t in range(tiles_per_core):
                idx = core_x * tiles_per_core + local_t
                if idx < total_out:
                    row = idx // n_tiles
                    col = idx % n_tiles
                    for ki in range(k_iters):
                        k_start = ki * k_chunk
                        with x_dfb.reserve() as blk:
                            tx = ttl.copy(x[row, k_start:k_start + k_chunk], blk); tx.wait()
                        with w_dfb.reserve() as blk:
                            tx = ttl.copy(w[k_start:k_start + k_chunk, col], blk); tx.wait()
        @ttl.datamovement()
        def dm_write():
            core_x, _ = ttl.core(dims=2)
            for local_t in range(tiles_per_core):
                idx = core_x * tiles_per_core + local_t
                if idx < total_out:
                    row = idx // n_tiles
                    col = idx % n_tiles
                    with out_dfb.wait() as blk:
                        tx = ttl.copy(blk, out[row, col]); tx.wait()
    return linear_accum_kernel

# ---------------------------------------------------------------------------
# Kernel: Elementwise ops (GELU, SiLU, adaLN modulate, gated residual)
# All use same ELEM_GRAN blocking pattern from toy-wm
# ---------------------------------------------------------------------------

ELEM_GRAN = 8  # column tiles per block (divides 32 for d=1024)

@ttl.kernel(grid="auto")
def silu_kernel(x, out):
    grid_cols, _ = ttl.grid_size(dims=2)
    row_tiles = x.shape[0] // TILE
    col_blocks = x.shape[1] // TILE // ELEM_GRAN
    total = row_tiles * col_blocks
    tiles_per_core = -(-total // grid_cols)
    x_dfb = ttl.make_dataflow_buffer_like(x, shape=(1, ELEM_GRAN), buffer_factor=2)
    out_dfb = ttl.make_dataflow_buffer_like(out, shape=(1, ELEM_GRAN), buffer_factor=2)
    @ttl.compute()
    def compute():
        core_x, _ = ttl.core(dims=2)
        for local_t in range(tiles_per_core):
            t = core_x * tiles_per_core + local_t
            if t < total:
                with x_dfb.wait() as xv, out_dfb.reserve() as o:
                    o.store(xv * ttl.math.sigmoid(xv))
    @ttl.datamovement()
    def dm_read():
        core_x, _ = ttl.core(dims=2)
        for local_t in range(tiles_per_core):
            t = core_x * tiles_per_core + local_t
            if t < total:
                row = t // col_blocks
                cb = t % col_blocks
                sc = cb * ELEM_GRAN
                with x_dfb.reserve() as blk:
                    tx = ttl.copy(x[row, sc:sc + ELEM_GRAN], blk); tx.wait()
    @ttl.datamovement()
    def dm_write():
        core_x, _ = ttl.core(dims=2)
        for local_t in range(tiles_per_core):
            t = core_x * tiles_per_core + local_t
            if t < total:
                row = t // col_blocks
                cb = t % col_blocks
                sc = cb * ELEM_GRAN
                with out_dfb.wait() as blk:
                    tx = ttl.copy(blk, out[row, sc:sc + ELEM_GRAN]); tx.wait()

@ttl.kernel(grid="auto")
def gelu_approx_kernel(x, out):
    """GELU with tanh approximation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))"""
    grid_cols, _ = ttl.grid_size(dims=2)
    row_tiles = x.shape[0] // TILE
    col_blocks = x.shape[1] // TILE // ELEM_GRAN
    total = row_tiles * col_blocks
    tiles_per_core = -(-total // grid_cols)
    x_dfb = ttl.make_dataflow_buffer_like(x, shape=(1, ELEM_GRAN), buffer_factor=2)
    out_dfb = ttl.make_dataflow_buffer_like(out, shape=(1, ELEM_GRAN), buffer_factor=2)
    @ttl.compute()
    def compute():
        core_x, _ = ttl.core(dims=2)
        for local_t in range(tiles_per_core):
            t = core_x * tiles_per_core + local_t
            if t < total:
                with x_dfb.wait() as xv, out_dfb.reserve() as o:
                    x3 = xv * xv * xv
                    inner = ttl.math.fill(xv, 0.7978845608) * (xv + ttl.math.fill(xv, 0.044715) * x3)
                    o.store(ttl.math.fill(xv, 0.5) * xv * (ttl.math.fill(xv, 1.0) + ttl.math.tanh(inner)))
    @ttl.datamovement()
    def dm_read():
        core_x, _ = ttl.core(dims=2)
        for local_t in range(tiles_per_core):
            t = core_x * tiles_per_core + local_t
            if t < total:
                row = t // col_blocks
                cb = t % col_blocks
                sc = cb * ELEM_GRAN
                with x_dfb.reserve() as blk:
                    tx = ttl.copy(x[row, sc:sc + ELEM_GRAN], blk); tx.wait()
    @ttl.datamovement()
    def dm_write():
        core_x, _ = ttl.core(dims=2)
        for local_t in range(tiles_per_core):
            t = core_x * tiles_per_core + local_t
            if t < total:
                row = t // col_blocks
                cb = t % col_blocks
                sc = cb * ELEM_GRAN
                with out_dfb.wait() as blk:
                    tx = ttl.copy(blk, out[row, sc:sc + ELEM_GRAN]); tx.wait()

@ttl.kernel(grid="auto")
def adaln_modulate_kernel(x, shift, scale, out):
    """AdaLN: out = x * (1 + scale) + shift"""
    grid_cols, _ = ttl.grid_size(dims=2)
    row_tiles = x.shape[0] // TILE
    col_blocks = x.shape[1] // TILE // ELEM_GRAN
    total = row_tiles * col_blocks
    tiles_per_core = -(-total // grid_cols)
    x_dfb = ttl.make_dataflow_buffer_like(x, shape=(1, ELEM_GRAN), buffer_factor=2)
    sh_dfb = ttl.make_dataflow_buffer_like(shift, shape=(1, ELEM_GRAN), buffer_factor=2)
    sc_dfb = ttl.make_dataflow_buffer_like(scale, shape=(1, ELEM_GRAN), buffer_factor=2)
    out_dfb = ttl.make_dataflow_buffer_like(out, shape=(1, ELEM_GRAN), buffer_factor=2)
    @ttl.compute()
    def compute():
        core_x, _ = ttl.core(dims=2)
        for local_t in range(tiles_per_core):
            t = core_x * tiles_per_core + local_t
            if t < total:
                with x_dfb.wait() as xv, sh_dfb.wait() as shv, sc_dfb.wait() as scv, out_dfb.reserve() as o:
                    o.store(xv * (scv + ttl.math.fill(scv, 1.0)) + shv)
    @ttl.datamovement()
    def dm_read():
        core_x, _ = ttl.core(dims=2)
        for local_t in range(tiles_per_core):
            t = core_x * tiles_per_core + local_t
            if t < total:
                row = t // col_blocks
                cb = t % col_blocks
                sc = cb * ELEM_GRAN
                with x_dfb.reserve() as blk:
                    tx = ttl.copy(x[row, sc:sc + ELEM_GRAN], blk); tx.wait()
                with sh_dfb.reserve() as blk:
                    tx = ttl.copy(shift[row, sc:sc + ELEM_GRAN], blk); tx.wait()
                with sc_dfb.reserve() as blk:
                    tx = ttl.copy(scale[row, sc:sc + ELEM_GRAN], blk); tx.wait()
    @ttl.datamovement()
    def dm_write():
        core_x, _ = ttl.core(dims=2)
        for local_t in range(tiles_per_core):
            t = core_x * tiles_per_core + local_t
            if t < total:
                row = t // col_blocks
                cb = t % col_blocks
                sc = cb * ELEM_GRAN
                with out_dfb.wait() as blk:
                    tx = ttl.copy(blk, out[row, sc:sc + ELEM_GRAN]); tx.wait()

@ttl.kernel(grid="auto")
def gated_residual_kernel(residual, x, gate, out):
    """Gated residual: out = residual + gate * x"""
    grid_cols, _ = ttl.grid_size(dims=2)
    row_tiles = residual.shape[0] // TILE
    col_blocks = residual.shape[1] // TILE // ELEM_GRAN
    total = row_tiles * col_blocks
    tiles_per_core = -(-total // grid_cols)
    res_dfb = ttl.make_dataflow_buffer_like(residual, shape=(1, ELEM_GRAN), buffer_factor=2)
    x_dfb = ttl.make_dataflow_buffer_like(x, shape=(1, ELEM_GRAN), buffer_factor=2)
    g_dfb = ttl.make_dataflow_buffer_like(gate, shape=(1, ELEM_GRAN), buffer_factor=2)
    out_dfb = ttl.make_dataflow_buffer_like(out, shape=(1, ELEM_GRAN), buffer_factor=2)
    @ttl.compute()
    def compute():
        core_x, _ = ttl.core(dims=2)
        for local_t in range(tiles_per_core):
            t = core_x * tiles_per_core + local_t
            if t < total:
                with res_dfb.wait() as rv, x_dfb.wait() as xv, g_dfb.wait() as gv, out_dfb.reserve() as o:
                    o.store(rv + xv * gv)
    @ttl.datamovement()
    def dm_read():
        core_x, _ = ttl.core(dims=2)
        for local_t in range(tiles_per_core):
            t = core_x * tiles_per_core + local_t
            if t < total:
                row = t // col_blocks
                cb = t % col_blocks
                sc = cb * ELEM_GRAN
                with res_dfb.reserve() as blk:
                    tx = ttl.copy(residual[row, sc:sc + ELEM_GRAN], blk); tx.wait()
                with x_dfb.reserve() as blk:
                    tx = ttl.copy(x[row, sc:sc + ELEM_GRAN], blk); tx.wait()
                with g_dfb.reserve() as blk:
                    tx = ttl.copy(gate[row, sc:sc + ELEM_GRAN], blk); tx.wait()
    @ttl.datamovement()
    def dm_write():
        core_x, _ = ttl.core(dims=2)
        for local_t in range(tiles_per_core):
            t = core_x * tiles_per_core + local_t
            if t < total:
                row = t // col_blocks
                cb = t % col_blocks
                sc = cb * ELEM_GRAN
                with out_dfb.wait() as blk:
                    tx = ttl.copy(blk, out[row, sc:sc + ELEM_GRAN]); tx.wait()

# ---------------------------------------------------------------------------
# Kernel: LayerNorm (3-pass: mean, var, normalize)
# ---------------------------------------------------------------------------

def make_layernorm_kernel(dim_tiles):
    """LayerNorm: out = (x - mean) / sqrt(var + eps) * weight + bias
    3-pass approach: pass 1 computes mean, pass 2 computes var, pass 3 normalizes."""
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
# Tests
# ---------------------------------------------------------------------------

def expand_bias(bias_1d, seq_len):
    """Expand (D,) bias to (seq_padded, D) for tile-aligned processing."""
    seq_padded = ((seq_len + TILE - 1) // TILE) * TILE
    return bias_1d.unsqueeze(0).expand(seq_padded, -1).contiguous().to(torch.bfloat16)

def test_linear(device):
    print("\n=== Test: Linear (no bias) ===")
    # Small test: (32, 64) @ (64, 32)
    M, K, N = 32, 64, 32
    k_chunk = K // TILE
    x = torch.randn(M, K, dtype=torch.bfloat16)
    w = torch.randn(K, N, dtype=torch.bfloat16)
    expected = x.float() @ w.float()
    out_tt = zeros_tt((M, N), device)
    kernel = make_linear_kernel(k_chunk)
    kernel(to_tt(x, device), to_tt(w, device), out_tt)
    result = from_tt(out_tt)
    check("linear_32x64_64x32", result, expected, atol=1.0, rtol=0.1)

    # Oasis-sized: (160, 1024) @ (1024, 3072) - QKV projection
    print("  Testing Oasis QKV size (160, 1024) @ (1024, 3072)...")
    M, K, N = 160, 1024, 3072
    k_chunk = K // TILE  # 32
    x = torch.randn(M, K, dtype=torch.bfloat16)
    w = torch.randn(K, N, dtype=torch.bfloat16)
    expected = x.float() @ w.float()
    out_tt = zeros_tt((M, N), device)
    kernel32 = make_linear_kernel(k_chunk)
    kernel32(to_tt(x, device), to_tt(w, device), out_tt)
    result = from_tt(out_tt)
    check("linear_160x1024_1024x3072", result, expected, rows=M, cols=N, atol=5.0, rtol=0.15)

def test_linear_bias(device):
    print("\n=== Test: Linear with bias ===")
    M, K, N = 32, 64, 32
    k_chunk = K // TILE
    x = torch.randn(M, K, dtype=torch.bfloat16)
    w = torch.randn(K, N, dtype=torch.bfloat16)
    bias = torch.randn(N, dtype=torch.bfloat16)
    bias_expanded = expand_bias(bias, M)
    expected = x.float() @ w.float() + bias.float()
    out_tt = zeros_tt((M, N), device)
    kernel = make_linear_bias_kernel(k_chunk, n_chunk=1)
    kernel(to_tt(x, device), to_tt(w, device), to_tt(bias_expanded, device), out_tt)
    result = from_tt(out_tt)
    check("linear_bias_32x64_64x32", result, expected, atol=1.0, rtol=0.1)

def test_linear_accum(device):
    print("\n=== Test: Linear with K accumulation ===")
    # MLP down: (32, 128) @ (128, 32) with k_chunk=2, k_iters=2
    M, K, N = 32, 128, 32
    k_chunk = 2
    k_iters = K // TILE // k_chunk  # 128/32/2 = 2
    x = torch.randn(M, K, dtype=torch.bfloat16)
    w = torch.randn(K, N, dtype=torch.bfloat16)
    expected = x.float() @ w.float()
    out_tt = zeros_tt((M, N), device)
    kernel = make_linear_accum_kernel(k_chunk, k_iters)
    kernel(to_tt(x, device), to_tt(w, device), out_tt)
    result = from_tt(out_tt)
    check("linear_accum_32x128_128x32", result, expected, atol=2.0, rtol=0.15)

def test_silu(device):
    print("\n=== Test: SiLU ===")
    M, D = 32, 256  # 8 col tiles, ELEM_GRAN=8 divides evenly
    x = torch.randn(M, D, dtype=torch.bfloat16)
    expected = torch.nn.functional.silu(x.float())
    out_tt = zeros_tt((M, D), device)
    silu_kernel(to_tt(x, device), out_tt)
    result = from_tt(out_tt)
    check("silu_32x256", result, expected, atol=0.1, rtol=0.05)

def test_gelu(device):
    print("\n=== Test: GELU (tanh approx) ===")
    M, D = 32, 256
    x = torch.randn(M, D, dtype=torch.bfloat16)
    expected = torch.nn.functional.gelu(x.float(), approximate="tanh")
    out_tt = zeros_tt((M, D), device)
    gelu_approx_kernel(to_tt(x, device), out_tt)
    result = from_tt(out_tt)
    check("gelu_approx_32x256", result, expected, atol=0.15, rtol=0.1)

def test_adaln_modulate(device):
    print("\n=== Test: AdaLN Modulate ===")
    M, D = 32, 256
    x = torch.randn(M, D, dtype=torch.bfloat16)
    shift = torch.randn(M, D, dtype=torch.bfloat16)
    scale = torch.randn(M, D, dtype=torch.bfloat16)
    expected = x.float() * (1.0 + scale.float()) + shift.float()
    out_tt = zeros_tt((M, D), device)
    adaln_modulate_kernel(to_tt(x, device), to_tt(shift, device), to_tt(scale, device), out_tt)
    result = from_tt(out_tt)
    check("adaln_modulate_32x256", result, expected, atol=0.5, rtol=0.1)

def test_gated_residual(device):
    print("\n=== Test: Gated Residual ===")
    M, D = 32, 256
    residual = torch.randn(M, D, dtype=torch.bfloat16)
    x = torch.randn(M, D, dtype=torch.bfloat16)
    gate = torch.randn(M, D, dtype=torch.bfloat16)
    expected = residual.float() + x.float() * gate.float()
    out_tt = zeros_tt((M, D), device)
    gated_residual_kernel(to_tt(residual, device), to_tt(x, device), to_tt(gate, device), out_tt)
    result = from_tt(out_tt)
    check("gated_residual_32x256", result, expected, atol=0.5, rtol=0.1)

def test_layernorm(device):
    print("\n=== Test: LayerNorm ===")
    M, D = 32, 64  # 2 dim tiles for small test
    dim_tiles = D // TILE
    x = torch.randn(M, D, dtype=torch.bfloat16)
    weight = torch.randn(D, dtype=torch.bfloat16)
    bias = torch.randn(D, dtype=torch.bfloat16)

    # PyTorch reference
    ln = torch.nn.LayerNorm(D, eps=1e-6)
    ln.weight.data = weight.float()
    ln.bias.data = bias.float()
    expected = ln(x.float())

    # Expand weight and bias to (M, D) for tile-aligned processing
    weight_expanded = weight.unsqueeze(0).expand(M, -1).contiguous().to(torch.bfloat16)
    bias_expanded = bias.unsqueeze(0).expand(M, -1).contiguous().to(torch.bfloat16)

    scaler = torch.ones(TILE, TILE, dtype=torch.bfloat16)
    mean_scale = torch.full((TILE, TILE), 1.0 / D, dtype=torch.bfloat16)

    out_tt = zeros_tt((M, D), device)
    kernel = make_layernorm_kernel(dim_tiles)
    kernel(
        to_tt(x, device),
        to_tt(weight_expanded, device),
        to_tt(bias_expanded, device),
        to_tt_l1(scaler, device),
        to_tt_l1(mean_scale, device),
        out_tt,
    )
    result = from_tt(out_tt)
    check("layernorm_32x64", result, expected, atol=0.5, rtol=0.1)


if __name__ == "__main__":
    device = ttnn.open_device(device_id=0)

    print("=" * 60)
    print("Oasis TT-Lang Kernel Tests")
    print("=" * 60)

    test_linear(device)
    test_linear_bias(device)
    test_linear_accum(device)
    test_silu(device)
    test_gelu(device)
    test_adaln_modulate(device)
    test_gated_residual(device)
    test_layernorm(device)

    print("\n" + "=" * 60)
    print("All tests complete!")
    print("=" * 60)

    ttnn.close_device(device)
