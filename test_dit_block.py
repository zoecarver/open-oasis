"""
Test a single SpatioTemporalDiTBlock on TT hardware against PyTorch reference.

For T=1 (single frame), this tests:
  - Spatial attention: LayerNorm -> adaLN -> QKV(no bias) -> RoPE -> SDPA -> O proj -> gated residual
  - Spatial MLP: LayerNorm -> adaLN -> Linear -> GELU -> Linear -> gated residual
  - Temporal attention: same structure (trivial at T=1, single-token self-attention)
  - Temporal MLP: same structure

We test spatial sub-block first, then add temporal.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import ttnn
import ttl
import sys
import os

# Add open-oasis to path for reference model
sys.path.insert(0, "/Users/zcarver/Developer/open-oasis")

TILE = 32

# Oasis dimensions
D_MODEL = 1024
N_HEADS = 16
D_HEAD = 64
D_MLP = 4096
N_PATCHES = 144  # 9x16 spatial grid
N_PATCH_PAD = 160  # padded to 5 tiles
FRAME_H = 9
FRAME_W = 16

D_MODEL_TILES = D_MODEL // TILE  # 32
D_MLP_TILES = D_MLP // TILE  # 128
D_3QKV_TILES = 3 * D_MODEL // TILE  # 96

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

def expand_to_seq(vec_1d, seq_pad):
    """Expand (D,) to (seq_pad, D) by broadcasting."""
    return vec_1d.unsqueeze(0).expand(seq_pad, -1).contiguous().to(torch.bfloat16)

def check(name, result, expected, rows=None, cols=None, atol=1.0, rtol=0.15):
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
    print(f"  [{status}] {name}: max_err={max_err:.4f} mean_err={mean_err:.4f}")
    if not ok:
        # Show where the biggest errors are
        diff = (r - e).abs()
        flat_idx = diff.argmax()
        row_idx = flat_idx // e.shape[1] if e.dim() > 1 else flat_idx
        col_idx = flat_idx % e.shape[1] if e.dim() > 1 else 0
        print(f"    worst at [{row_idx}, {col_idx}]: got={r.flatten()[flat_idx]:.4f} expected={e.flatten()[flat_idx]:.4f}")
    return ok

ELEM_GRAN = 8

# ---------------------------------------------------------------------------
# Import kernels from test_kernels.py
# ---------------------------------------------------------------------------

# Linear (no bias, single-pass K)
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

# Linear with bias
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

# Linear with K accumulation
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
                    with x_dfb.wait() as xv, w_dfb.wait() as wv, acc_dfb.reserve() as acc:
                        acc.store(xv @ wv)
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

# Elementwise kernels
@ttl.kernel(grid="auto")
def gelu_approx_kernel(x, out):
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

# LayerNorm kernel
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
                    for j in range(dim_tiles):
                        with x_dfb.reserve() as blk:
                            tx = ttl.copy(x[tile_idx, j], blk); tx.wait()
                    for j in range(dim_tiles):
                        with x_dfb.reserve() as blk:
                            tx = ttl.copy(x[tile_idx, j], blk); tx.wait()
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
# Instantiate kernels
# ---------------------------------------------------------------------------

linear_k32 = make_linear_kernel(D_MODEL_TILES)  # (seq, 1024) @ (1024, N)
linear_bias_k32 = make_linear_bias_kernel(D_MODEL_TILES, n_chunk=4)  # for MLP up (1024->4096)
linear_accum_k32_4 = make_linear_accum_kernel(D_MODEL_TILES, 4)  # for MLP down (4096->1024)
layernorm_d1024 = make_layernorm_kernel(D_MODEL_TILES)

# ---------------------------------------------------------------------------
# Test: Spatial sub-block (LayerNorm -> adaLN -> Attn -> MLP -> gated residual)
# ---------------------------------------------------------------------------

def layernorm_host(x, weight, bias, eps=1e-6):
    return F.layer_norm(x.float(), [x.shape[-1]], weight.float(), bias.float(), eps).to(torch.bfloat16)

def gelu_host(x):
    return F.gelu(x.float(), approximate="tanh").to(torch.bfloat16)

def test_spatial_sub_block(device):
    """Test the spatial sub-block of a SpatioTemporalDiTBlock.
    For T=1: input is (1, 1, 9, 16, 1024), spatial attention on (1, 16, 144, 64)."""

    print("\n=== Test: Spatial Sub-Block ===")
    torch.manual_seed(42)

    # Create random weights with proper Xavier initialization to keep values in reasonable range
    xavier_std = (2.0 / (D_MODEL + D_MODEL)) ** 0.5
    s_norm1_w = torch.ones(D_MODEL, dtype=torch.bfloat16)  # LayerNorm weight init=1
    s_norm1_b = torch.zeros(D_MODEL, dtype=torch.bfloat16)
    s_norm2_w = torch.ones(D_MODEL, dtype=torch.bfloat16)
    s_norm2_b = torch.zeros(D_MODEL, dtype=torch.bfloat16)
    s_qkv_w = (torch.randn(D_MODEL, 3 * D_MODEL) * xavier_std).to(torch.bfloat16)
    s_out_w = (torch.randn(D_MODEL, D_MODEL) * xavier_std).to(torch.bfloat16)
    s_out_b = torch.zeros(D_MODEL, dtype=torch.bfloat16)
    mlp_xavier = (2.0 / (D_MODEL + D_MLP)) ** 0.5
    s_mlp_fc1_w = (torch.randn(D_MODEL, D_MLP) * mlp_xavier).to(torch.bfloat16)
    s_mlp_fc1_b = torch.zeros(D_MLP, dtype=torch.bfloat16)
    s_mlp_fc2_w = (torch.randn(D_MLP, D_MODEL) * mlp_xavier).to(torch.bfloat16)
    s_mlp_fc2_b = torch.zeros(D_MODEL, dtype=torch.bfloat16)

    # adaLN params initialized to 0 (like real model), with small perturbation for test
    adaln_params = torch.randn(6, D_MODEL, dtype=torch.bfloat16) * 0.01

    # Input: (N_PATCH_PAD, D_MODEL) = (160, 1024) flattened 2D
    x_2d = torch.randn(N_PATCH_PAD, D_MODEL, dtype=torch.bfloat16)

    # ---- PyTorch Reference ----
    x_ref = x_2d[:N_PATCHES].float()  # only valid patches

    # adaLN modulation
    shift_msa = adaln_params[0].float()
    scale_msa = adaln_params[1].float()
    gate_msa = adaln_params[2].float()
    shift_mlp = adaln_params[3].float()
    scale_mlp = adaln_params[4].float()
    gate_mlp = adaln_params[5].float()

    # Attention path: norm -> modulate -> qkv -> sdpa -> o_proj -> gated residual
    normed = F.layer_norm(x_ref, [D_MODEL], s_norm1_w.float(), s_norm1_b.float(), 1e-6)
    modulated = normed * (1 + scale_msa) + shift_msa

    # QKV projection (no bias)
    qkv = modulated @ s_qkv_w.float()
    q, k, v = qkv.chunk(3, dim=-1)

    # Reshape for multi-head attention: (N_PATCHES, D_MODEL) -> (1, N_HEADS, N_PATCHES, D_HEAD)
    q = q.view(N_PATCHES, N_HEADS, D_HEAD).permute(1, 0, 2).unsqueeze(0)
    k = k.view(N_PATCHES, N_HEADS, D_HEAD).permute(1, 0, 2).unsqueeze(0)
    v = v.view(N_PATCHES, N_HEADS, D_HEAD).permute(1, 0, 2).unsqueeze(0)

    # Pad to tile alignment (same as TT version) so attention matches
    q = F.pad(q, [0, 0, 0, N_PATCH_PAD - N_PATCHES])
    k = F.pad(k, [0, 0, 0, N_PATCH_PAD - N_PATCHES])
    v = F.pad(v, [0, 0, 0, N_PATCH_PAD - N_PATCHES])

    # Skip RoPE for now (test without it first)
    attn_out = F.scaled_dot_product_attention(q, k, v, is_causal=False)
    attn_out = attn_out.squeeze(0).permute(1, 0, 2).reshape(N_PATCH_PAD, D_MODEL)[:N_PATCHES]

    # O projection (on valid patches only)
    attn_projected = attn_out @ s_out_w.float()

    # Gated residual
    x_after_attn = x_ref + gate_msa * attn_projected

    # MLP path: norm -> modulate -> fc1 -> gelu -> fc2 -> gated residual
    normed2 = F.layer_norm(x_after_attn, [D_MODEL], s_norm2_w.float(), s_norm2_b.float(), 1e-6)
    modulated2 = normed2 * (1 + scale_mlp) + shift_mlp
    fc1_out = F.gelu(modulated2 @ s_mlp_fc1_w.float() + s_mlp_fc1_b.float(), approximate="tanh")
    fc2_out = fc1_out @ s_mlp_fc2_w.float() + s_mlp_fc2_b.float()
    x_after_mlp = x_after_attn + gate_mlp * fc2_out

    expected = x_after_mlp
    # Save intermediate references for later checks
    ref_attn_projected = attn_projected
    ref_x_after_attn = x_after_attn
    ref_fc1_gelu = fc1_out
    ref_fc2_bias = fc2_out

    # ---- TT-Lang Implementation ----
    # Constants
    scaler = to_tt_l1(torch.ones(TILE, TILE, dtype=torch.bfloat16), device)
    mean_scale = to_tt_l1(torch.full((TILE, TILE), 1.0 / D_MODEL, dtype=torch.bfloat16), device)

    # Weights on device
    norm1_w_tt = to_tt(expand_to_seq(s_norm1_w, N_PATCH_PAD), device)
    norm1_b_tt = to_tt(expand_to_seq(s_norm1_b, N_PATCH_PAD), device)
    norm2_w_tt = to_tt(expand_to_seq(s_norm2_w, N_PATCH_PAD), device)
    norm2_b_tt = to_tt(expand_to_seq(s_norm2_b, N_PATCH_PAD), device)
    qkv_w_tt = to_tt(s_qkv_w, device)
    out_w_tt = to_tt(s_out_w, device)
    mlp_fc1_w_tt = to_tt(s_mlp_fc1_w, device)
    mlp_fc1_b_tt = to_tt(expand_to_seq(s_mlp_fc1_b, N_PATCH_PAD), device)
    mlp_fc2_w_tt = to_tt(s_mlp_fc2_w, device)
    mlp_fc2_b_tt = to_tt(expand_to_seq(s_mlp_fc2_b, N_PATCH_PAD), device)

    # adaLN params expanded to (N_PATCH_PAD, D_MODEL) - broadcast from conditioning
    shift_msa_tt = to_tt(expand_to_seq(adaln_params[0], N_PATCH_PAD), device)
    scale_msa_tt = to_tt(expand_to_seq(adaln_params[1], N_PATCH_PAD), device)
    gate_msa_tt = to_tt(expand_to_seq(adaln_params[2], N_PATCH_PAD), device)
    shift_mlp_tt = to_tt(expand_to_seq(adaln_params[3], N_PATCH_PAD), device)
    scale_mlp_tt = to_tt(expand_to_seq(adaln_params[4], N_PATCH_PAD), device)
    gate_mlp_tt = to_tt(expand_to_seq(adaln_params[5], N_PATCH_PAD), device)

    # Scratch buffers
    x_tt = to_tt(x_2d, device)
    normed_tt = zeros_tt((N_PATCH_PAD, D_MODEL), device)
    modulated_tt = zeros_tt((N_PATCH_PAD, D_MODEL), device)
    qkv_tt = zeros_tt((N_PATCH_PAD, 3 * D_MODEL), device)
    attn_2d_tt = zeros_tt((N_PATCH_PAD, D_MODEL), device)
    o_proj_tt = zeros_tt((N_PATCH_PAD, D_MODEL), device)
    x_after_attn_tt = zeros_tt((N_PATCH_PAD, D_MODEL), device)
    normed2_tt = zeros_tt((N_PATCH_PAD, D_MODEL), device)
    modulated2_tt = zeros_tt((N_PATCH_PAD, D_MODEL), device)
    fc1_tt = zeros_tt((N_PATCH_PAD, D_MLP), device)
    gelu_tt = zeros_tt((N_PATCH_PAD, D_MLP), device)
    fc2_tt = zeros_tt((N_PATCH_PAD, D_MODEL), device)
    x_after_mlp_tt = zeros_tt((N_PATCH_PAD, D_MODEL), device)

    # Step 1: LayerNorm
    print("  Running LayerNorm 1...")
    layernorm_d1024(x_tt, norm1_w_tt, norm1_b_tt, scaler, mean_scale, normed_tt)
    normed_host = from_tt(normed_tt)[:N_PATCHES]
    check("layernorm1", normed_host, F.layer_norm(x_2d[:N_PATCHES].float(), [D_MODEL], s_norm1_w.float(), s_norm1_b.float(), 1e-6))

    # Step 2: adaLN modulate
    print("  Running adaLN modulate...")
    adaln_modulate_kernel(normed_tt, shift_msa_tt, scale_msa_tt, modulated_tt)

    # Step 3: QKV projection (no bias)
    print("  Running QKV projection...")
    linear_k32(modulated_tt, qkv_w_tt, qkv_tt)

    # Step 4: Reshape to multi-head, SDPA, reshape back (using ttnn)
    print("  Running SDPA via ttnn...")
    qkv_host = from_tt(qkv_tt)[:N_PATCHES].float()
    q_h, k_h, v_h = qkv_host.chunk(3, dim=-1)
    q_sdpa = q_h.view(N_PATCHES, N_HEADS, D_HEAD).permute(1, 0, 2).unsqueeze(0).to(torch.bfloat16)
    k_sdpa = k_h.view(N_PATCHES, N_HEADS, D_HEAD).permute(1, 0, 2).unsqueeze(0).to(torch.bfloat16)
    v_sdpa = v_h.view(N_PATCHES, N_HEADS, D_HEAD).permute(1, 0, 2).unsqueeze(0).to(torch.bfloat16)

    # Pad seq to tile alignment for SDPA
    def pad_to_tile(t, dim, target):
        pad_size = target - t.shape[dim]
        if pad_size <= 0:
            return t
        pad_spec = [0] * (2 * (t.dim() - dim - 1)) + [0, pad_size]
        return F.pad(t, pad_spec)

    q_pad = pad_to_tile(q_sdpa, 2, N_PATCH_PAD)
    k_pad = pad_to_tile(k_sdpa, 2, N_PATCH_PAD)
    v_pad = pad_to_tile(v_sdpa, 2, N_PATCH_PAD)

    q_tt_sdpa = to_tt(q_pad, device)
    k_tt_sdpa = to_tt(k_pad, device)
    v_tt_sdpa = to_tt(v_pad, device)

    attn_out_tt = ttnn.transformer.scaled_dot_product_attention(
        q_tt_sdpa, k_tt_sdpa, v_tt_sdpa, is_causal=False)

    # Reshape back: (1, N_HEADS, N_PATCH_PAD, D_HEAD) -> (N_PATCH_PAD, D_MODEL)
    attn_perm = ttnn.permute(attn_out_tt, [0, 2, 1, 3])
    attn_2d_tt = ttnn.reshape(attn_perm, [N_PATCH_PAD, N_HEADS * D_HEAD])

    # Step 5: O projection (no bias for simplicity, real model has to_out linear)
    print("  Running O projection...")
    o_proj_tt = zeros_tt((N_PATCH_PAD, D_MODEL), device)
    linear_k32(attn_2d_tt, out_w_tt, o_proj_tt)

    # Check QKV intermediate
    qkv_ref = (F.layer_norm(x_2d[:N_PATCHES].float(), [D_MODEL], s_norm1_w.float(), s_norm1_b.float(), 1e-6) * (1 + scale_msa) + shift_msa) @ s_qkv_w.float()
    qkv_result = from_tt(qkv_tt)[:N_PATCHES]
    check("qkv_projection", qkv_result, qkv_ref, atol=5.0, rtol=0.2)

    # Check SDPA output (compare against padded reference)
    attn_2d_result = from_tt(attn_2d_tt)[:N_PATCHES]
    check("sdpa_output", attn_2d_result, attn_out, atol=2.0, rtol=0.15)

    # Check O projection
    o_proj_result = from_tt(o_proj_tt)[:N_PATCHES]
    check("o_projection", o_proj_result, ref_attn_projected, atol=5.0, rtol=0.3)

    # Step 6: Gated residual
    print("  Running gated residual (attention)...")
    gated_residual_kernel(x_tt, o_proj_tt, gate_msa_tt, x_after_attn_tt)

    # Check after attention gated residual
    x_after_attn_result = from_tt(x_after_attn_tt)[:N_PATCHES]
    check("after_attn_gated_res", x_after_attn_result, ref_x_after_attn, atol=5.0, rtol=0.3)

    # Step 7: LayerNorm 2
    print("  Running LayerNorm 2...")
    layernorm_d1024(x_after_attn_tt, norm2_w_tt, norm2_b_tt, scaler, mean_scale, normed2_tt)

    # Step 8: adaLN modulate 2
    print("  Running adaLN modulate 2...")
    adaln_modulate_kernel(normed2_tt, shift_mlp_tt, scale_mlp_tt, modulated2_tt)

    # Step 9: MLP fc1 + GELU
    print("  Running MLP fc1 + GELU...")
    linear_bias_k32(modulated2_tt, mlp_fc1_w_tt, mlp_fc1_b_tt, fc1_tt)
    gelu_approx_kernel(fc1_tt, gelu_tt)

    # Check fc1+gelu intermediate
    gelu_result = from_tt(gelu_tt)[:N_PATCHES]
    check("fc1_gelu", gelu_result, ref_fc1_gelu, atol=10.0, rtol=0.3)

    # Step 10: MLP fc2 (4096->1024, needs accumulation)
    print("  Running MLP fc2 (accum)...")
    fc2_tt = zeros_tt((N_PATCH_PAD, D_MODEL), device)
    linear_accum_k32_4(gelu_tt, mlp_fc2_w_tt, fc2_tt)
    # Add bias manually
    fc2_with_bias = ttnn.add(fc2_tt, mlp_fc2_b_tt)

    # Check fc2 intermediate
    fc2_result = from_tt(fc2_with_bias)[:N_PATCHES]
    check("fc2_with_bias", fc2_result, ref_fc2_bias, atol=20.0, rtol=0.3)

    # Step 11: Gated residual (MLP)
    print("  Running gated residual (MLP)...")
    gated_residual_kernel(x_after_attn_tt, fc2_with_bias, gate_mlp_tt, x_after_mlp_tt)

    # Compare final output
    result = from_tt(x_after_mlp_tt)[:N_PATCHES]
    ok = check("spatial_sub_block", result, expected, atol=5.0, rtol=0.2)

    return ok


if __name__ == "__main__":
    device = ttnn.open_device(device_id=0)

    print("=" * 60)
    print("Oasis DiT Block Test")
    print("=" * 60)

    test_spatial_sub_block(device)

    print("\n" + "=" * 60)
    print("Test complete!")
    print("=" * 60)

    ttnn.close_device(device)
