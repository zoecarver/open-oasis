"""
Oasis-500M inference on Tenstorrent hardware using TT-Lang kernels.

DiT runs on TT device. VAE encode/decode runs on CPU (PyTorch).
Single-frame generation for initial testing (T=1).

Usage: run via run-test.sh --hw oasis_inference.py
"""
import sys, types
import torch
import torch.nn as nn

# Stub out timm entirely -- vae.py/dit.py only need Mlp and to_2tuple.
# Trying to stub torchvision (timm's transitive dep) is endless whack-a-mole.

def _to_2tuple(x):
    if isinstance(x, (tuple, list)):
        return tuple(x)
    return (x, x)

class _Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None,
                 act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

def _make_stub(name, parent=None):
    m = types.ModuleType(name)
    sys.modules[name] = m
    if parent is not None:
        setattr(parent, name.split('.')[-1], m)
    return m

_timm = _make_stub('timm')
_timm_models = _make_stub('timm.models', _timm)
_timm_models_vit = _make_stub('timm.models.vision_transformer', _timm_models)
_timm_layers = _make_stub('timm.layers', _timm)
_timm_layers_helpers = _make_stub('timm.layers.helpers', _timm_layers)

_timm_models_vit.Mlp = _Mlp
_timm_layers_helpers.to_2tuple = _to_2tuple

import torch.nn.functional as F
import ttnn
import ttl
import math
import time
import os
from safetensors import safe_open
from einops import rearrange

# ============================================================
# Constants
# ============================================================

TILE = 32
D_MODEL = 1024
N_HEADS = 16
D_HEAD = 64  # D_MODEL // N_HEADS
D_MLP = 4096
N_BLOCKS = 16
PATCH_SIZE = 2
IN_CHANNELS = 16
INPUT_H = 18
INPUT_W = 32
FRAME_H = INPUT_H // PATCH_SIZE  # 9
FRAME_W = INPUT_W // PATCH_SIZE  # 16
N_PATCHES = FRAME_H * FRAME_W  # 144
N_PATCH_PAD = ((N_PATCHES + TILE - 1) // TILE) * TILE  # 160
FREQ_DIM = 256
EXT_COND_DIM = 25
EXT_COND_PAD = 32  # padded to tile
OUT_DIM = PATCH_SIZE * PATCH_SIZE * IN_CHANNELS  # 64
SCALING_FACTOR = 0.07843137255

D_TILES = D_MODEL // TILE  # 32
D_MLP_TILES = D_MLP // TILE  # 128

ELEM_GRAN = 8  # divides 32 (D_MODEL tiles) and 16 (128/8 for D_MLP)

# ============================================================
# TT-Lang Kernels
# ============================================================

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

# Instantiate kernel variants
linear_k32 = make_linear_kernel(D_TILES)
linear_k8 = make_linear_kernel(FREQ_DIM // TILE)  # for timestep embed (256->1024)
linear_k1 = make_linear_kernel(1)  # for external_cond (32->1024)
linear_bias_k32 = make_linear_bias_kernel(D_TILES, n_chunk=4)
linear_bias_k8 = make_linear_bias_kernel(FREQ_DIM // TILE, n_chunk=4)
linear_accum_k32_4 = make_linear_accum_kernel(D_TILES, 4)  # MLP down 4096->1024
layernorm_d1024 = make_layernorm_kernel(D_TILES)

# ============================================================
# Host helpers
# ============================================================

def to_tt(t, device):
    return ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                           device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)

def to_tt_l1(t, device):
    return ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                           device=device, memory_config=ttnn.L1_MEMORY_CONFIG)

def zeros_tt(shape, device):
    return to_tt(torch.zeros(shape, dtype=torch.bfloat16), device)

def expand_bias(bias_1d, seq_pad):
    return bias_1d.unsqueeze(0).expand(seq_pad, -1).contiguous().to(torch.bfloat16)

def timestep_embedding(t, dim, max_period=10000):
    half = dim // 2
    freqs = torch.exp(-math.log(max_period) * torch.arange(0, half, dtype=torch.float32) / half)
    args = t[:, None].float() * freqs[None]
    return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)

def sigmoid_beta_schedule(timesteps, start=-3, end=3, tau=1, clamp_min=1e-5):
    steps = timesteps + 1
    t = torch.linspace(0, timesteps, steps, dtype=torch.float64) / timesteps
    v_start = torch.tensor(start / tau).sigmoid()
    v_end = torch.tensor(end / tau).sigmoid()
    alphas_cumprod = (-((t * (end - start) + start) / tau).sigmoid() + v_end) / (v_end - v_start)
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)

# ============================================================
# Weight loading
# ============================================================

def find_dit_weights():
    blob_dir = "/root/.cache/huggingface/hub/models--Etched--oasis-500m/blobs/"
    for f in sorted(os.listdir(blob_dir)):
        path = blob_dir + f
        with safe_open(path, framework="pt") as st:
            keys = list(st.keys())
            if any(k.startswith("blocks.") for k in keys):
                return path
    raise FileNotFoundError("DiT weights not found")

def find_vae_weights():
    blob_dir = "/root/.cache/huggingface/hub/models--Etched--oasis-500m/blobs/"
    for f in sorted(os.listdir(blob_dir)):
        path = blob_dir + f
        with safe_open(path, framework="pt") as st:
            keys = list(st.keys())
            if any(k.startswith("encoder.") for k in keys):
                return path
    raise FileNotFoundError("VAE weights not found")

def preload_dit_weights(tt_device):
    t0 = time.time()
    path = find_dit_weights()
    print("Loading DiT weights from:", path)
    dev = {}

    with safe_open(path, framework="pt") as st:
        # Timestep embedder: Linear(256, 1024) -> SiLU -> Linear(1024, 1024)
        dev["t_emb_w0"] = to_tt(st.get_tensor("t_embedder.mlp.0.weight").T.contiguous(), tt_device)
        dev["t_emb_b0"] = to_tt(expand_bias(st.get_tensor("t_embedder.mlp.0.bias"), TILE), tt_device)
        dev["t_emb_w2"] = to_tt(st.get_tensor("t_embedder.mlp.2.weight").T.contiguous(), tt_device)
        dev["t_emb_b2"] = to_tt(expand_bias(st.get_tensor("t_embedder.mlp.2.bias"), TILE), tt_device)

        # External conditioning: Linear(25, 1024) - pad input from 25 to 32
        ext_w = st.get_tensor("external_cond.weight").T.contiguous()  # (25, 1024)
        ext_w_pad = torch.zeros(EXT_COND_PAD, D_MODEL, dtype=torch.float32)
        ext_w_pad[:EXT_COND_DIM] = ext_w
        dev["ext_cond_w"] = to_tt(ext_w_pad.to(torch.bfloat16), tt_device)
        dev["ext_cond_b"] = to_tt(expand_bias(st.get_tensor("external_cond.bias"), TILE), tt_device)

        # Patch embedder: Conv2d(16, 1024, 2, 2) - keep as host tensor for now
        dev["x_emb_conv_w"] = st.get_tensor("x_embedder.proj.weight")  # (1024, 16, 2, 2)
        dev["x_emb_conv_b"] = st.get_tensor("x_embedder.proj.bias")  # (1024,)

        # Final layer
        dev["final_adaln_w"] = to_tt(st.get_tensor("final_layer.adaLN_modulation.1.weight").T.contiguous().to(torch.bfloat16), tt_device)
        dev["final_adaln_b"] = to_tt(expand_bias(st.get_tensor("final_layer.adaLN_modulation.1.bias").to(torch.bfloat16), TILE), tt_device)
        dev["final_linear_w"] = to_tt(st.get_tensor("final_layer.linear.weight").T.contiguous().to(torch.bfloat16), tt_device)
        dev["final_linear_b"] = to_tt(expand_bias(st.get_tensor("final_layer.linear.bias").to(torch.bfloat16), N_PATCH_PAD), tt_device)

        # Per-block weights
        for i in range(N_BLOCKS):
            for prefix in ["s", "t"]:
                p = "blocks.%d.%s" % (i, prefix)

                # adaLN modulation: Linear(1024, 6144) with bias
                adaln_w = st.get_tensor("%s_adaLN_modulation.1.weight" % p).T.contiguous().to(torch.bfloat16)
                dev["%s.adaln_w" % p] = to_tt(adaln_w, tt_device)
                adaln_b = st.get_tensor("%s_adaLN_modulation.1.bias" % p).to(torch.bfloat16)
                dev["%s.adaln_b" % p] = to_tt(expand_bias(adaln_b, TILE), tt_device)

                # QKV: Linear(1024, 3072) NO bias
                qkv_w = st.get_tensor("%s_attn.to_qkv.weight" % p).T.contiguous().to(torch.bfloat16)
                dev["%s.qkv_w" % p] = to_tt(qkv_w, tt_device)

                # Output projection: Linear(1024, 1024) with bias
                out_w = st.get_tensor("%s_attn.to_out.weight" % p).T.contiguous().to(torch.bfloat16)
                dev["%s.out_w" % p] = to_tt(out_w, tt_device)
                out_b = st.get_tensor("%s_attn.to_out.bias" % p).to(torch.bfloat16)
                dev["%s.out_b" % p] = to_tt(expand_bias(out_b, N_PATCH_PAD), tt_device)

                # MLP: fc1 Linear(1024, 4096) + fc2 Linear(4096, 1024)
                fc1_w = st.get_tensor("%s_mlp.fc1.weight" % p).T.contiguous().to(torch.bfloat16)
                dev["%s.fc1_w" % p] = to_tt(fc1_w, tt_device)
                fc1_b = st.get_tensor("%s_mlp.fc1.bias" % p).to(torch.bfloat16)
                dev["%s.fc1_b" % p] = to_tt(expand_bias(fc1_b, N_PATCH_PAD), tt_device)
                fc2_w = st.get_tensor("%s_mlp.fc2.weight" % p).T.contiguous().to(torch.bfloat16)
                dev["%s.fc2_w" % p] = to_tt(fc2_w, tt_device)
                fc2_b = st.get_tensor("%s_mlp.fc2.bias" % p).to(torch.bfloat16)
                dev["%s.fc2_b" % p] = to_tt(expand_bias(fc2_b, N_PATCH_PAD), tt_device)

        # LayerNorm weights: final_layer uses elementwise_affine=False, so no LN weights
        # Block LayerNorms also use elementwise_affine=False (eps=1e-6)
        # So norm weight = 1, norm bias = 0 (expanded to seq dim)
        dev["ln_w_ones"] = to_tt(torch.ones(N_PATCH_PAD, D_MODEL, dtype=torch.bfloat16), tt_device)
        dev["ln_b_zeros"] = to_tt(torch.zeros(N_PATCH_PAD, D_MODEL, dtype=torch.bfloat16), tt_device)

    elapsed = time.time() - t0
    print("Preloaded %d tensors in %.1fs" % (len(dev), elapsed))
    return dev

# ============================================================
# Scratch buffers
# ============================================================

def prealloc_scratch(tt_device):
    t0 = time.time()
    s = {}
    s["z_a"] = zeros_tt((N_PATCH_PAD, D_MODEL), tt_device)
    s["z_b"] = zeros_tt((N_PATCH_PAD, D_MODEL), tt_device)
    s["normed"] = zeros_tt((N_PATCH_PAD, D_MODEL), tt_device)
    s["modulated"] = zeros_tt((N_PATCH_PAD, D_MODEL), tt_device)
    s["qkv"] = zeros_tt((N_PATCH_PAD, 3 * D_MODEL), tt_device)
    s["o_proj"] = zeros_tt((N_PATCH_PAD, D_MODEL), tt_device)
    s["fc1"] = zeros_tt((N_PATCH_PAD, D_MLP), tt_device)
    s["gelu"] = zeros_tt((N_PATCH_PAD, D_MLP), tt_device)
    s["fc2"] = zeros_tt((N_PATCH_PAD, D_MODEL), tt_device)
    # Conditioning scratch
    s["t_emb_a"] = zeros_tt((TILE, D_MODEL), tt_device)
    s["t_emb_b"] = zeros_tt((TILE, D_MODEL), tt_device)
    s["cond"] = zeros_tt((TILE, D_MODEL), tt_device)
    # adaLN scratch: (TILE, 6*D_MODEL) = (32, 6144)
    s["adaln_out"] = zeros_tt((TILE, 6 * D_MODEL), tt_device)
    # Final layer
    s["final_adaln"] = zeros_tt((TILE, 2 * D_MODEL), tt_device)
    s["final_out"] = zeros_tt((N_PATCH_PAD, OUT_DIM), tt_device)
    elapsed = time.time() - t0
    print("Pre-allocated %d scratch tensors in %.1fs" % (len(s), elapsed))
    return s

# ============================================================
# DiT forward pass
# ============================================================

def patch_embed_host(x_latent, conv_w, conv_b):
    """x_latent: (B, C, H, W) = (1, 16, 18, 32) -> (N_PATCHES, D_MODEL)"""
    x = F.conv2d(x_latent.float(), conv_w.float(), conv_b.float(), stride=PATCH_SIZE)
    # x: (1, 1024, 9, 16)
    x = rearrange(x, "b d h w -> b h w d")
    # x: (1, 9, 16, 1024) -> flatten spatial
    x = x.reshape(1, N_PATCHES, D_MODEL)
    return x.squeeze(0).to(torch.bfloat16)  # (144, 1024)

def unpatchify_host(x, patch_size, out_channels, h, w):
    """x: (N_PATCHES, patch_size^2 * out_channels) -> (1, out_channels, H, W)"""
    x = x.float().reshape(1, h, w, patch_size, patch_size, out_channels)
    x = torch.einsum("nhwpqc->nchpwq", x)
    return x.reshape(1, out_channels, h * patch_size, w * patch_size)

def run_sub_block(prefix, x_tt, cond_tt, dev, scr, tt_device, scaler, mean_scale):
    """Run one spatial or temporal sub-block (norm -> modulate -> attn -> gated_res -> norm -> modulate -> MLP -> gated_res).
    prefix: e.g. "blocks.0.s" or "blocks.0.t"
    x_tt: (N_PATCH_PAD, D_MODEL) device tensor
    cond_tt: (TILE, D_MODEL) conditioning device tensor
    """
    # Step 1: Compute adaLN params: adaLN_modulation = Sequential(SiLU, Linear(1024, 6144))
    # cond_tt is the raw conditioning (already computed), apply SiLU then linear
    silu_kernel(cond_tt, scr["t_emb_a"])
    linear_bias_k32(scr["t_emb_a"], dev["%s.adaln_w" % prefix], dev["%s.adaln_b" % prefix], scr["adaln_out"])

    # Read adaLN params back to host for chunking and broadcasting
    adaln_host = ttnn.to_torch(scr["adaln_out"])[0, :6 * D_MODEL].float()  # row 0
    chunks = adaln_host.reshape(6, D_MODEL)

    # Broadcast each chunk to (N_PATCH_PAD, D_MODEL) and send to device
    shift_msa = to_tt(expand_bias(chunks[0].to(torch.bfloat16), N_PATCH_PAD), tt_device)
    scale_msa = to_tt(expand_bias(chunks[1].to(torch.bfloat16), N_PATCH_PAD), tt_device)
    gate_msa = to_tt(expand_bias(chunks[2].to(torch.bfloat16), N_PATCH_PAD), tt_device)
    shift_mlp = to_tt(expand_bias(chunks[3].to(torch.bfloat16), N_PATCH_PAD), tt_device)
    scale_mlp = to_tt(expand_bias(chunks[4].to(torch.bfloat16), N_PATCH_PAD), tt_device)
    gate_mlp = to_tt(expand_bias(chunks[5].to(torch.bfloat16), N_PATCH_PAD), tt_device)

    # Attention path
    layernorm_d1024(x_tt, dev["ln_w_ones"], dev["ln_b_zeros"], scaler, mean_scale, scr["normed"])
    adaln_modulate_kernel(scr["normed"], shift_msa, scale_msa, scr["modulated"])
    linear_k32(scr["modulated"], dev["%s.qkv_w" % prefix], scr["qkv"])

    # SDPA on host via ttnn (reshape Q/K/V)
    qkv_host = ttnn.to_torch(scr["qkv"])[:N_PATCHES].to(torch.bfloat16)
    q, k, v = qkv_host.float().chunk(3, dim=-1)
    q = q.view(N_PATCHES, N_HEADS, D_HEAD).permute(1, 0, 2).unsqueeze(0).to(torch.bfloat16)
    k = k.view(N_PATCHES, N_HEADS, D_HEAD).permute(1, 0, 2).unsqueeze(0).to(torch.bfloat16)
    v = v.view(N_PATCHES, N_HEADS, D_HEAD).permute(1, 0, 2).unsqueeze(0).to(torch.bfloat16)

    # Pad to tile alignment
    q = F.pad(q, [0, 0, 0, N_PATCH_PAD - N_PATCHES])
    k = F.pad(k, [0, 0, 0, N_PATCH_PAD - N_PATCHES])
    v = F.pad(v, [0, 0, 0, N_PATCH_PAD - N_PATCHES])

    q_tt = to_tt(q, tt_device)
    k_tt = to_tt(k, tt_device)
    v_tt = to_tt(v, tt_device)

    attn_out = ttnn.transformer.scaled_dot_product_attention(q_tt, k_tt, v_tt, is_causal=False)
    attn_perm = ttnn.permute(attn_out, [0, 2, 1, 3])
    attn_2d = ttnn.reshape(attn_perm, [N_PATCH_PAD, N_HEADS * D_HEAD])

    # O projection + bias + gated residual
    linear_k32(attn_2d, dev["%s.out_w" % prefix], scr["o_proj"])
    o_with_bias = ttnn.add(scr["o_proj"], dev["%s.out_b" % prefix])
    gated_residual_kernel(x_tt, o_with_bias, gate_msa, scr["z_b"])

    # Swap z_a and z_b (z_b becomes new x)
    x_tt = scr["z_b"]

    # MLP path
    layernorm_d1024(x_tt, dev["ln_w_ones"], dev["ln_b_zeros"], scaler, mean_scale, scr["normed"])
    adaln_modulate_kernel(scr["normed"], shift_mlp, scale_mlp, scr["modulated"])
    linear_bias_k32(scr["modulated"], dev["%s.fc1_w" % prefix], dev["%s.fc1_b" % prefix], scr["fc1"])
    gelu_approx_kernel(scr["fc1"], scr["gelu"])
    linear_accum_k32_4(scr["gelu"], dev["%s.fc2_w" % prefix], scr["fc2"])
    fc2_with_bias = ttnn.add(scr["fc2"], dev["%s.fc2_b" % prefix])
    gated_residual_kernel(x_tt, fc2_with_bias, gate_mlp, scr["z_a"])

    return scr["z_a"]

def dit_forward(x_latent, timesteps, actions, dev, scr, tt_device, scaler, mean_scale):
    """Full DiT forward pass.
    x_latent: (B, T, C, H, W) = (1, 1, 16, 18, 32)
    timesteps: (B, T) = (1, 1) long tensor
    actions: (B, T, 25) float tensor
    Returns: (B, T, C, H, W) velocity prediction
    """
    B, T = 1, 1

    # Patch embedding on host - ensure 4D input (1, C, H, W)
    x_for_conv = x_latent.reshape(-1, IN_CHANNELS, INPUT_H, INPUT_W)[:1]  # (1, 16, 18, 32)
    x_2d_host = patch_embed_host(
        x_for_conv,
        dev["x_emb_conv_w"], dev["x_emb_conv_b"]
    )  # (144, 1024)

    # Pad to tile alignment
    x_pad = torch.zeros(N_PATCH_PAD, D_MODEL, dtype=torch.bfloat16)
    x_pad[:N_PATCHES] = x_2d_host
    z_cur = to_tt(x_pad, tt_device)

    # Timestep conditioning
    t_freq = timestep_embedding(timesteps.flatten(), FREQ_DIM)  # (1, 256)
    t_freq_pad = torch.zeros(TILE, FREQ_DIM, dtype=torch.bfloat16)
    t_freq_pad[0] = t_freq[0].to(torch.bfloat16)
    t_freq_tt = to_tt(t_freq_pad, tt_device)

    # t_embedder: Linear(256, 1024) -> SiLU -> Linear(1024, 1024)
    linear_bias_k8(t_freq_tt, dev["t_emb_w0"], dev["t_emb_b0"], scr["t_emb_a"])
    silu_kernel(scr["t_emb_a"], scr["t_emb_b"])
    linear_bias_k32(scr["t_emb_b"], dev["t_emb_w2"], dev["t_emb_b2"], scr["cond"])

    # External conditioning: Linear(25, 1024) + add to cond
    # ext_cond_w is (32, 1024), K=1 tile, so use linear_k1
    if actions is not None:
        act = actions[0, 0]  # (25,)
        act_pad = torch.zeros(TILE, EXT_COND_PAD, dtype=torch.bfloat16)
        act_pad[0, :EXT_COND_DIM] = act.to(torch.bfloat16)
        act_tt = to_tt(act_pad, tt_device)
        ext_out = zeros_tt((TILE, D_MODEL), tt_device)
        linear_k1(act_tt, dev["ext_cond_w"], ext_out)
        ext_out = ttnn.add(ext_out, dev["ext_cond_b"])
        scr["cond"] = ttnn.add(scr["cond"], ext_out)

    # 16 blocks: each has spatial + temporal sub-block
    for block_idx in range(N_BLOCKS):
        # Spatial sub-block
        z_cur = run_sub_block(
            "blocks.%d.s" % block_idx, z_cur, scr["cond"],
            dev, scr, tt_device, scaler, mean_scale
        )
        # Temporal sub-block (at T=1, attention is trivial but MLP still applies)
        z_cur = run_sub_block(
            "blocks.%d.t" % block_idx, z_cur, scr["cond"],
            dev, scr, tt_device, scaler, mean_scale
        )
        print("  Block %d done" % block_idx)

    # Final layer: SiLU(cond) -> Linear(1024, 2048) -> chunk(shift, scale)
    silu_kernel(scr["cond"], scr["t_emb_a"])
    linear_bias_k32(scr["t_emb_a"], dev["final_adaln_w"], dev["final_adaln_b"], scr["final_adaln"])

    final_mod_host = ttnn.to_torch(scr["final_adaln"])[0, :2 * D_MODEL].float()
    shift_f = expand_bias(final_mod_host[:D_MODEL].to(torch.bfloat16), N_PATCH_PAD)
    scale_f = expand_bias(final_mod_host[D_MODEL:].to(torch.bfloat16), N_PATCH_PAD)
    shift_tt = to_tt(shift_f, tt_device)
    scale_tt = to_tt(scale_f, tt_device)

    # LayerNorm + adaLN modulate
    layernorm_d1024(z_cur, dev["ln_w_ones"], dev["ln_b_zeros"], scaler, mean_scale, scr["normed"])
    adaln_modulate_kernel(scr["normed"], shift_tt, scale_tt, scr["modulated"])

    # Final linear: (N_PATCH_PAD, 1024) @ (1024, 64) = (N_PATCH_PAD, 64)
    # OUT_DIM=64=2 tiles, use linear_k32 (no bias) + ttnn.add since n_chunk must divide n_tiles
    linear_k32(scr["modulated"], dev["final_linear_w"], scr["final_out"])
    scr["final_out"] = ttnn.add(scr["final_out"], dev["final_linear_b"])

    # Read back and unpatchify on host
    out_host = ttnn.to_torch(scr["final_out"])[:N_PATCHES].float()
    out_img = unpatchify_host(out_host, PATCH_SIZE, IN_CHANNELS, FRAME_H, FRAME_W)
    return out_img.unsqueeze(1)  # (1, 1, 16, 18, 32)

# ============================================================
# VAE (CPU only)
# ============================================================

def load_vae_cpu():
    """Load VAE model on CPU."""
    import sys
    sys.path.insert(0, "/tmp")
    from vae import VAE_models
    from safetensors.torch import load_model

    vae = VAE_models["vit-l-20-shallow-encoder"]()
    path = find_vae_weights()
    print("Loading VAE weights from:", path)
    load_model(vae, path)
    vae = vae.eval()
    return vae

def vae_decode_cpu(z_latent, vae):
    """Decode latent to image using VAE on CPU.
    z_latent: (B, T, C, H, W) = (1, 1, 16, 18, 32)
    Returns: (B, T, 3, 360, 640) in [0, 1]
    """
    B, T = z_latent.shape[:2]
    z = rearrange(z_latent, "b t c h w -> (b t) (h w) c")
    with torch.no_grad():
        decoded = (vae.decode(z.float() / SCALING_FACTOR) + 1) / 2
    decoded = torch.clamp(decoded, 0, 1)
    return rearrange(decoded, "(b t) c h w -> b t h w c", b=B, t=T)

# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    tt_device = ttnn.open_device(device_id=0)
    torch.manual_seed(42)

    print("=" * 60)
    print("Oasis-500M DiT Inference on Tenstorrent")
    print("=" * 60)

    # L1 constants
    scaler = to_tt_l1(torch.ones(TILE, TILE, dtype=torch.bfloat16), tt_device)
    mean_scale = to_tt_l1(torch.full((TILE, TILE), 1.0 / D_MODEL, dtype=torch.bfloat16), tt_device)

    # Load weights
    dev = preload_dit_weights(tt_device)
    scr = prealloc_scratch(tt_device)

    # Diffusion schedule
    max_noise_level = 1000
    ddim_steps = 10
    noise_range = torch.linspace(-1, max_noise_level - 1, ddim_steps + 1)
    noise_abs_max = 20
    stabilization_level = 15

    betas = sigmoid_beta_schedule(max_noise_level).float()
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    alphas_cumprod = rearrange(alphas_cumprod, "T -> T 1 1 1")

    # For initial test: random latent as prompt (skip VAE encode)
    print("\nGenerating with random latent prompt...")
    x = torch.randn(1, 1, IN_CHANNELS, INPUT_H, INPUT_W)
    x = torch.clamp(x, -noise_abs_max, noise_abs_max)

    # Single frame: generate frame index 1 (denoise from noise)
    chunk = torch.randn(1, 1, IN_CHANNELS, INPUT_H, INPUT_W)
    chunk = torch.clamp(chunk, -noise_abs_max, noise_abs_max)
    actions = torch.zeros(1, 1, EXT_COND_DIM)  # no-op action

    print("\nRunning DDIM denoising (%d steps)..." % ddim_steps)
    t_total = time.time()

    for noise_idx in reversed(range(1, ddim_steps + 1)):
        t_step = time.time()
        t_val = torch.full((1, 1), noise_range[noise_idx], dtype=torch.long)

        print("Step %d/%d (noise_level=%d)..." % (ddim_steps - noise_idx + 1, ddim_steps, t_val.item()))

        v = dit_forward(chunk, t_val, actions, dev, scr, tt_device, scaler, mean_scale)

        # DDIM update
        t_next_val = torch.full((1, 1), noise_range[noise_idx - 1], dtype=torch.long)
        t_next_val = torch.where(t_next_val < 0, t_val, t_next_val)

        # alphas_cumprod[t_val] gives (1, 1, 1, 1, 1) which broadcasts with (1, 1, 16, 18, 32)
        alpha_t = alphas_cumprod[t_val]
        alpha_next = alphas_cumprod[t_next_val]
        if noise_idx == 1:
            alpha_next = torch.ones_like(alpha_next)

        x_start = alpha_t.sqrt() * chunk - (1 - alpha_t).sqrt() * v
        x_noise = ((1 / alpha_t).sqrt() * chunk - x_start) / (1 / alpha_t - 1).sqrt()
        chunk = alpha_next.sqrt() * x_start + x_noise * (1 - alpha_next).sqrt()

        print("  Step time: %.1fs, output range: [%.2f, %.2f]" % (
            time.time() - t_step, v.min().item(), v.max().item()))

    total_time = time.time() - t_total
    print("\nDDIM complete in %.1fs" % total_time)
    print("Final latent range: [%.2f, %.2f]" % (chunk.min().item(), chunk.max().item()))

    # Save raw latent
    print("\nSaving output latent to /tmp/oasis_output_latent.pt")
    torch.save(chunk, "/tmp/oasis_output_latent.pt")

    # Latent preview (3 of 16 channels)
    from PIL import Image
    latent_vis = chunk[0, 0, :3]
    latent_vis = (latent_vis - latent_vis.min()) / (latent_vis.max() - latent_vis.min() + 1e-8)
    latent_vis = (latent_vis * 255).byte()
    latent_vis = F.interpolate(latent_vis.unsqueeze(0).float(), size=(180, 320), mode="nearest").byte()
    img = Image.fromarray(latent_vis[0].permute(1, 2, 0).numpy())
    img.save("/tmp/oasis_latent_preview.png")
    print("Saved latent preview to /tmp/oasis_latent_preview.png")

    # VAE decode
    print("\nLoading VAE for decode...")
    vae = load_vae_cpu()
    print("Running VAE decode...")
    t_vae = time.time()
    frames = vae_decode_cpu(chunk, vae)  # (1, 1, 360, 640, 3) in [0, 1]
    print("VAE decode: %.1fs" % (time.time() - t_vae))

    # Save as PNG
    frame = frames[0, 0]  # (360, 640, 3)
    frame = (frame * 255).byte().numpy()
    img = Image.fromarray(frame)
    img.save("/tmp/oasis_frame.png")
    print("Saved frame to /tmp/oasis_frame.png")
    print("Frame shape:", frame.shape, "range: [%d, %d]" % (frame.min(), frame.max()))

    print("\nDone!")
    ttnn.close_device(tt_device)
