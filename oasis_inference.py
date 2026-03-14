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

# RoPE dimensions (from model weights)
# Spatial: 16 freqs * 2 (interleave) * 2 (H+W axes) = 64
SPATIAL_ROPE_DIM = 64
# Temporal: 32 freqs * 2 (interleave) = 64
TEMPORAL_ROPE_DIM = 64

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
# RoPE (Rotary Positional Embeddings) - applied on host
# ============================================================

def rotate_half(x):
    """Rotary embedding helper: rearrange pairs and negate."""
    x = rearrange(x, "... (d r) -> ... d r", r=2)
    x1, x2 = x.unbind(dim=-1)
    x = torch.stack((-x2, x1), dim=-1)
    return rearrange(x, "... d r -> ... (d r)")

def apply_rotary_emb(freqs, t):
    """Apply rotary embeddings to tensor t using precomputed freqs.
    freqs: (..., rot_dim)  t: (..., d_head)
    Only the first rot_dim dims of t are rotated; rest pass through."""
    rot_dim = freqs.shape[-1]
    t_rot = t[..., :rot_dim]
    t_pass = t[..., rot_dim:]
    t_rot = (t_rot * freqs.cos()) + (rotate_half(t_rot) * freqs.sin())
    return torch.cat((t_rot, t_pass), dim=-1)

def precompute_spatial_rope_freqs(freqs_param):
    """Precompute spatial axial RoPE freqs for H_GRID x W_GRID grid.
    freqs_param: learned freqs tensor shape (half_dim,) from weights.
    Returns: (N_PATCHES, rot_dim) float tensor.
    rot_dim = 2 axes * 2 * half_dim = 4 * half_dim (e.g. 4*16=64 for Oasis)."""
    from einops import repeat as einops_repeat
    h_pos = torch.linspace(-1, 1, steps=FRAME_H)
    w_pos = torch.linspace(-1, 1, steps=FRAME_W)

    # Per-axis: outer product then interleave-repeat (f0,f0,f1,f1,...)
    h_freqs = torch.einsum("i, f -> i f", h_pos, freqs_param)
    h_freqs = einops_repeat(h_freqs, "... n -> ... (n r)", r=2)  # (9, 32)
    w_freqs = torch.einsum("i, f -> i f", w_pos, freqs_param)
    w_freqs = einops_repeat(w_freqs, "... n -> ... (n r)", r=2)  # (16, 32)

    # Broadcast to (H, W, 32) each, then cat along last dim -> (H, W, 64)
    h_broad = h_freqs[:, None, :].expand(FRAME_H, FRAME_W, -1)
    w_broad = w_freqs[None, :, :].expand(FRAME_H, FRAME_W, -1)
    axial_freqs = torch.cat([h_broad, w_broad], dim=-1)

    return axial_freqs.reshape(N_PATCHES, -1)  # (144, 64)

def precompute_temporal_rope_freqs(freqs_param, max_t=32):
    """Precompute temporal RoPE freqs.
    freqs_param: learned freqs tensor shape (half_dim,) from weights.
    Returns: (max_t, rot_dim) float tensor. rot_dim = 2 * half_dim."""
    from einops import repeat as einops_repeat
    positions = torch.arange(max_t, dtype=torch.float32)
    freqs = torch.einsum("i, f -> i f", positions, freqs_param)
    return einops_repeat(freqs, "... n -> ... (n r)", r=2)  # (max_t, 64)

# Global RoPE freq tables (set during weight loading)
SPATIAL_ROPE_FREQS = None   # (144, 64) float
TEMPORAL_ROPE_FREQS = None  # (max_t, 64) float

def swap_adjacent_columns(w):
    """Swap adjacent column pairs: col 2k <-> col 2k+1.
    Used to bake rotate_half permutation into weight matrices for device RoPE."""
    w_swap = w.clone()
    w_swap[:, 0::2] = w[:, 1::2]
    w_swap[:, 1::2] = w[:, 0::2]
    return w_swap

def build_spatial_rope_device_tables(spatial_freqs, n_frames, tt_device):
    """Build device-side cos/sin_perm tables for spatial RoPE.
    spatial_freqs: (N_PATCHES, 64) float.
    Returns cos_tt, sin_perm_tt each (SEQ, D_MODEL) on device.
    sin_perm bakes in the rotate_half sign: sin_perm[2k] = -sin, sin_perm[2k+1] = +sin.
    """
    SEQ = N_PATCH_PAD * n_frames
    cos_per_pos = spatial_freqs.cos()  # (N_PATCHES, 64)
    sin_per_pos = spatial_freqs.sin()  # (N_PATCHES, 64)

    # Bake rotate_half sign into sin: [-, +, -, +, ...] pattern
    sign = torch.ones(SPATIAL_ROPE_DIM)
    sign[0::2] = -1
    sin_perm_per_pos = sin_per_pos * sign.unsqueeze(0)  # (N_PATCHES, 64)

    # Repeat across all heads: (N_PATCHES, 64) -> (N_PATCHES, 1024)
    cos_expanded = cos_per_pos.repeat(1, N_HEADS)
    sin_expanded = sin_perm_per_pos.repeat(1, N_HEADS)

    # Pad to N_PATCH_PAD rows, repeat for T frames
    cos_full = torch.zeros(SEQ, D_MODEL, dtype=torch.bfloat16)
    sin_full = torch.zeros(SEQ, D_MODEL, dtype=torch.bfloat16)
    for t in range(n_frames):
        start = t * N_PATCH_PAD
        cos_full[start:start + N_PATCHES] = cos_expanded.to(torch.bfloat16)
        sin_full[start:start + N_PATCHES] = sin_expanded.to(torch.bfloat16)

    return to_tt(cos_full, tt_device), to_tt(sin_full, tt_device)

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

def preload_dit_weights(tt_device, n_frames=2):
    t0 = time.time()
    path = find_dit_weights()
    print("Loading DiT weights from:", path)
    dev = {}

    with safe_open(path, framework="pt") as st:
        # Timestep embedder + external cond: computed on HOST in fp32
        dev["t_emb_w0_host"] = st.get_tensor("t_embedder.mlp.0.weight").T.contiguous().float()  # (256, 1024)
        dev["t_emb_b0_host"] = st.get_tensor("t_embedder.mlp.0.bias").float()  # (1024,)
        dev["t_emb_w2_host"] = st.get_tensor("t_embedder.mlp.2.weight").T.contiguous().float()  # (1024, 1024)
        dev["t_emb_b2_host"] = st.get_tensor("t_embedder.mlp.2.bias").float()  # (1024,)

        dev["ext_cond_w_host"] = st.get_tensor("external_cond.weight").T.contiguous().float()  # (25, 1024)
        dev["ext_cond_b_host"] = st.get_tensor("external_cond.bias").float()  # (1024,)

        # Patch embedder: Conv2d(16, 1024, 2, 2) - keep as host tensor for now
        dev["x_emb_conv_w"] = st.get_tensor("x_embedder.proj.weight")  # (1024, 16, 2, 2)
        dev["x_emb_conv_b"] = st.get_tensor("x_embedder.proj.bias")  # (1024,)

        # Final layer adaLN: on device with ttnn.matmul for precision
        dev["final_adaln_w"] = to_tt(st.get_tensor("final_layer.adaLN_modulation.1.weight").T.contiguous().to(torch.bfloat16), tt_device)
        final_adaln_b = st.get_tensor("final_layer.adaLN_modulation.1.bias").to(torch.bfloat16)
        dev["final_adaln_b"] = to_tt(final_adaln_b.unsqueeze(0).expand(TILE, -1).contiguous(), tt_device)
        dev["final_linear_w"] = to_tt(st.get_tensor("final_layer.linear.weight").T.contiguous().to(torch.bfloat16), tt_device)
        dev["final_linear_b"] = to_tt(expand_bias(st.get_tensor("final_layer.linear.bias").to(torch.bfloat16), N_PATCH_PAD * n_frames), tt_device)

        # Per-block weights
        for i in range(N_BLOCKS):
            for prefix in ["s", "t"]:
                p = "blocks.%d.%s" % (i, prefix)

                # adaLN modulation: on device with ttnn.matmul for precision
                dev["%s.adaln_w" % p] = to_tt(st.get_tensor("%s_adaLN_modulation.1.weight" % p).T.contiguous().to(torch.bfloat16), tt_device)
                adaln_b_raw = st.get_tensor("%s_adaLN_modulation.1.bias" % p).to(torch.bfloat16)
                dev["%s.adaln_b" % p] = to_tt(adaln_b_raw.unsqueeze(0).expand(TILE, -1).contiguous(), tt_device)

                # QKV: Linear(1024, 3072) NO bias
                qkv_w = st.get_tensor("%s_attn.to_qkv.weight" % p).T.contiguous().to(torch.bfloat16)
                dev["%s.qkv_w" % p] = to_tt(qkv_w, tt_device)

                # Pair-swapped Q/K weights for device-side RoPE (rotate_half baked in)
                q_w = qkv_w[:, :D_MODEL]
                k_w = qkv_w[:, D_MODEL:2*D_MODEL]
                qk_swap_w = torch.cat([swap_adjacent_columns(q_w),
                                       swap_adjacent_columns(k_w)], dim=1)
                dev["%s.qk_swap_w" % p] = to_tt(qk_swap_w, tt_device)

                # Output projection: Linear(1024, 1024) with bias
                SEQ = N_PATCH_PAD * n_frames
                out_w = st.get_tensor("%s_attn.to_out.weight" % p).T.contiguous().to(torch.bfloat16)
                dev["%s.out_w" % p] = to_tt(out_w, tt_device)
                out_b = st.get_tensor("%s_attn.to_out.bias" % p).to(torch.bfloat16)
                dev["%s.out_b" % p] = to_tt(expand_bias(out_b, SEQ), tt_device)

                # MLP: fc1 Linear(1024, 4096) + fc2 Linear(4096, 1024)
                fc1_w = st.get_tensor("%s_mlp.fc1.weight" % p).T.contiguous().to(torch.bfloat16)
                dev["%s.fc1_w" % p] = to_tt(fc1_w, tt_device)
                fc1_b = st.get_tensor("%s_mlp.fc1.bias" % p).to(torch.bfloat16)
                dev["%s.fc1_b" % p] = to_tt(expand_bias(fc1_b, SEQ), tt_device)
                fc2_w = st.get_tensor("%s_mlp.fc2.weight" % p).T.contiguous().to(torch.bfloat16)
                dev["%s.fc2_w" % p] = to_tt(fc2_w, tt_device)
                fc2_b = st.get_tensor("%s_mlp.fc2.bias" % p).to(torch.bfloat16)
                dev["%s.fc2_b" % p] = to_tt(expand_bias(fc2_b, SEQ), tt_device)

        SEQ = N_PATCH_PAD * n_frames
        dev["ln_w_ones"] = to_tt(torch.ones(SEQ, D_MODEL, dtype=torch.bfloat16), tt_device)
        dev["ln_b_zeros"] = to_tt(torch.zeros(SEQ, D_MODEL, dtype=torch.bfloat16), tt_device)

        # RoPE: load learned freqs from block 0 (shared across all blocks)
        global SPATIAL_ROPE_FREQS, TEMPORAL_ROPE_FREQS
        s_freqs = st.get_tensor("blocks.0.s_attn.rotary_emb.freqs").float()  # (16,)
        t_freqs = st.get_tensor("blocks.0.t_attn.rotary_emb.freqs").float()  # (32,)
        SPATIAL_ROPE_FREQS = precompute_spatial_rope_freqs(s_freqs)  # (144, 64)
        TEMPORAL_ROPE_FREQS = precompute_temporal_rope_freqs(t_freqs)  # (max_t, 64)
        print("Precomputed RoPE freqs: spatial %s, temporal %s" % (
            tuple(SPATIAL_ROPE_FREQS.shape), tuple(TEMPORAL_ROPE_FREQS.shape)))

        # Device-side spatial RoPE tables
        dev["spatial_cos"], dev["spatial_sin_perm"] = \
            build_spatial_rope_device_tables(SPATIAL_ROPE_FREQS, n_frames, tt_device)

    elapsed = time.time() - t0
    print("Preloaded %d tensors in %.1fs" % (len(dev), elapsed))
    return dev

# ============================================================
# Scratch buffers
# ============================================================

def prealloc_scratch(tt_device, n_frames=2):
    """Pre-allocate scratch buffers. n_frames=2 for prompt+generated frame."""
    t0 = time.time()
    s = {}
    SEQ = N_PATCH_PAD * n_frames  # 320 for T=2
    s["z_a"] = zeros_tt((SEQ, D_MODEL), tt_device)
    s["z_b"] = zeros_tt((SEQ, D_MODEL), tt_device)
    s["normed"] = zeros_tt((SEQ, D_MODEL), tt_device)
    s["modulated"] = zeros_tt((SEQ, D_MODEL), tt_device)
    s["qkv"] = zeros_tt((SEQ, 3 * D_MODEL), tt_device)
    s["o_proj"] = zeros_tt((SEQ, D_MODEL), tt_device)
    s["fc1"] = zeros_tt((SEQ, D_MLP), tt_device)
    s["gelu"] = zeros_tt((SEQ, D_MLP), tt_device)
    s["fc2"] = zeros_tt((SEQ, D_MODEL), tt_device)
    # Conditioning scratch (per-frame, so TILE * n_frames)
    s["t_emb_a"] = zeros_tt((TILE, D_MODEL), tt_device)
    s["t_emb_b"] = zeros_tt((TILE, D_MODEL), tt_device)
    s["cond"] = zeros_tt((TILE, D_MODEL), tt_device)
    # adaLN scratch
    s["adaln_out"] = zeros_tt((TILE, 6 * D_MODEL), tt_device)
    s["silu_cond"] = zeros_tt((TILE, D_MODEL), tt_device)
    # Final layer
    s["final_adaln"] = zeros_tt((TILE, 2 * D_MODEL), tt_device)
    s["final_out"] = zeros_tt((SEQ, OUT_DIM), tt_device)
    s["n_frames"] = n_frames
    elapsed = time.time() - t0
    print("Pre-allocated %d scratch tensors (T=%d) in %.1fs" % (len(s) - 1, n_frames, elapsed))
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

def build_per_frame_adaln(cond_list, prefix, dev, scr, tt_device):
    """Compute adaLN params on DEVICE using ttnn.matmul.
    cond_list: list of T conditioning device tensors, each (TILE, D_MODEL) with all rows filled.
    Returns: [shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp] as device tensors.
    """
    T = len(cond_list)
    SEQ = N_PATCH_PAD * T
    N_REPEAT = N_PATCH_PAD // TILE  # 5

    per_frame_expanded = []
    for cond_tt in cond_list:
        # SiLU on device
        silu_kernel(cond_tt, scr["silu_cond"])
        # Linear(1024, 6144) on device via ttnn.matmul
        adaln_raw = ttnn.matmul(scr["silu_cond"], dev["%s.adaln_w" % prefix])
        adaln_raw = ttnn.add(adaln_raw, dev["%s.adaln_b" % prefix])
        # adaln_raw: (TILE, 6*D_MODEL) = (32, 6144), all rows have same values
        # Expand to (N_PATCH_PAD, 6*D_MODEL) by repeating the tile block
        expanded = ttnn.concat([adaln_raw] * N_REPEAT, dim=0)
        per_frame_expanded.append(expanded)

    # Combine frames: (SEQ, 6*D_MODEL)
    if T == 1:
        full = per_frame_expanded[0]
    else:
        full = ttnn.concat(per_frame_expanded, dim=0)

    # Slice into 6 params of (SEQ, D_MODEL) each
    results = []
    for i in range(6):
        start_col = i * D_MODEL
        end_col = (i + 1) * D_MODEL
        param = ttnn.slice(full, [0, start_col], [SEQ, end_col])
        results.append(param)
    return results

PROFILE_BLOCKS = True  # Set True for timing breakdown

def run_sub_block(prefix, x_tt, cond_list, dev, scr, tt_device, scaler, mean_scale, attn_type="spatial"):
    """Run one spatial or temporal sub-block.
    prefix: e.g. "blocks.0.s" or "blocks.0.t"
    cond_list: list of T conditioning device tensors, each (TILE, D_MODEL)
    attn_type: "spatial" or "temporal"
    x_tt: (N_PATCH_PAD * T, D_MODEL) device tensor
    """
    T = len(cond_list)
    SEQ = N_PATCH_PAD * T

    if PROFILE_BLOCKS:
        import time as _t
        _p = lambda label: None
        class _Timer:
            def __init__(self):
                self.t = _t.time()
                self.times = []
            def mark(self, label):
                now = _t.time()
                self.times.append((label, (now - self.t) * 1000))
                self.t = now
            def report(self, prefix, attn_type):
                parts = " | ".join("%s:%.1fms" % (l, t) for l, t in self.times)
                total = sum(t for _, t in self.times)
                print("    [%s %s] %.1fms: %s" % (prefix, attn_type, total, parts))
        _timer = _Timer()
    else:
        class _DummyTimer:
            def mark(self, label): pass
            def report(self, prefix, attn_type): pass
        _timer = _DummyTimer()

    # Compute per-frame adaLN params
    shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = \
        build_per_frame_adaln(cond_list, prefix, dev, scr, tt_device)
    _timer.mark("adaln")

    # Attention path: LayerNorm -> adaLN modulate -> QKV
    layernorm_d1024(x_tt, dev["ln_w_ones"], dev["ln_b_zeros"], scaler, mean_scale, scr["normed"])
    adaln_modulate_kernel(scr["normed"], shift_msa, scale_msa, scr["modulated"])
    _timer.mark("norm+mod")
    linear_k32(scr["modulated"], dev["%s.qkv_w" % prefix], scr["qkv"])
    _timer.mark("qkv")

    if attn_type == "spatial":
        # All-device spatial attention: RoPE + SDPA without host round-trip
        # Slice Q, K, V from QKV on device
        q = ttnn.slice(scr["qkv"], [0, 0], [SEQ, D_MODEL])
        k = ttnn.slice(scr["qkv"], [0, D_MODEL], [SEQ, 2 * D_MODEL])
        v = ttnn.slice(scr["qkv"], [0, 2 * D_MODEL], [SEQ, 3 * D_MODEL])

        # Compute pair-swapped Q/K projections for rotate_half
        qk_swap = ttnn.matmul(scr["modulated"], dev["%s.qk_swap_w" % prefix])
        q_swap = ttnn.slice(qk_swap, [0, 0], [SEQ, D_MODEL])
        k_swap = ttnn.slice(qk_swap, [0, D_MODEL], [SEQ, 2 * D_MODEL])

        # Apply spatial RoPE: q_roped = q * cos + q_swap * sin_perm
        q_roped = ttnn.add(ttnn.multiply(q, dev["spatial_cos"]),
                           ttnn.multiply(q_swap, dev["spatial_sin_perm"]))
        k_roped = ttnn.add(ttnn.multiply(k, dev["spatial_cos"]),
                           ttnn.multiply(k_swap, dev["spatial_sin_perm"]))

        # Reshape for SDPA: (SEQ, D_MODEL) -> (T, N_HEADS, N_PATCH_PAD, D_HEAD)
        q_4d = ttnn.reshape(q_roped, [T, N_PATCH_PAD, N_HEADS, D_HEAD])
        q_4d = ttnn.permute(q_4d, [0, 2, 1, 3])
        k_4d = ttnn.reshape(k_roped, [T, N_PATCH_PAD, N_HEADS, D_HEAD])
        k_4d = ttnn.permute(k_4d, [0, 2, 1, 3])
        v_4d = ttnn.reshape(v, [T, N_PATCH_PAD, N_HEADS, D_HEAD])
        v_4d = ttnn.permute(v_4d, [0, 2, 1, 3])

        attn_out = ttnn.transformer.scaled_dot_product_attention(q_4d, k_4d, v_4d, is_causal=False)
        attn_perm = ttnn.permute(attn_out, [0, 2, 1, 3])  # (T, N_PATCH_PAD, heads, d_head)
        attn_2d = ttnn.reshape(attn_perm, [SEQ, D_MODEL])
        _timer.mark("sdpa")

    else:
        # Temporal: read QKV to host for CPU SDPA (T=2 too small for device)
        qkv_host_full = ttnn.to_torch(scr["qkv"])  # (SEQ, 3*D_MODEL)
        _timer.mark("qkv_read")
        # Temporal: (B*H*W, heads, T, d_head) with causal attention
        # Gather per-frame patches, reshape for temporal
        frames_qkv = []
        for t_idx in range(T):
            start = t_idx * N_PATCH_PAD
            qkv_frame = qkv_host_full[start:start + N_PATCHES].to(torch.bfloat16).float()
            frames_qkv.append(qkv_frame)

        # Stack: (N_PATCHES, T, D_MODEL*3) then split Q/K/V
        stacked = torch.stack(frames_qkv, dim=1)  # (N_PATCHES, T, 3*D_MODEL)
        q, k, v = stacked.chunk(3, dim=-1)  # each (N_PATCHES, T, D_MODEL)

        # Reshape to multi-head: (N_PATCHES, T, N_HEADS, D_HEAD)
        q_mh = q.view(N_PATCHES, T, N_HEADS, D_HEAD)
        k_mh = k.view(N_PATCHES, T, N_HEADS, D_HEAD)
        v_mh = v.view(N_PATCHES, T, N_HEADS, D_HEAD)

        # Apply temporal RoPE: freqs are per-timestep
        t_freqs = TEMPORAL_ROPE_FREQS[:T].unsqueeze(1)  # (T, 1, 64) broadcast over heads
        q_mh = apply_rotary_emb(t_freqs.unsqueeze(0), q_mh)  # broadcast over N_PATCHES
        k_mh = apply_rotary_emb(t_freqs.unsqueeze(0), k_mh)

        # Reshape to SDPA format: (N_PATCHES, heads, T, d_head)
        q_sdpa = q_mh.permute(0, 2, 1, 3).to(torch.bfloat16)
        k_sdpa = k_mh.permute(0, 2, 1, 3).to(torch.bfloat16)
        v_sdpa = v_mh.permute(0, 2, 1, 3).to(torch.bfloat16)

        # Pad N_PATCHES -> N_PATCH_PAD in batch dim for tile alignment
        q_sdpa = F.pad(q_sdpa, [0, 0, 0, 0, 0, 0, 0, N_PATCH_PAD - N_PATCHES])
        k_sdpa = F.pad(k_sdpa, [0, 0, 0, 0, 0, 0, 0, N_PATCH_PAD - N_PATCHES])
        v_sdpa = F.pad(v_sdpa, [0, 0, 0, 0, 0, 0, 0, N_PATCH_PAD - N_PATCHES])

        # T=2 is not tile-aligned (needs 32). For small T, do SDPA on host CPU
        attn_host = F.scaled_dot_product_attention(
            q_sdpa.float(), k_sdpa.float(), v_sdpa.float(), is_causal=True
        )  # (N_PATCH_PAD, heads, T, d_head)

        # Reshape back to (SEQ, D_MODEL): permute to (N_PATCH_PAD, T, heads, d_head)
        attn_host = attn_host.permute(0, 2, 1, 3)  # (N_PATCH_PAD, T, heads, d_head)
        attn_host = attn_host.reshape(N_PATCH_PAD, T, D_MODEL)
        # Interleave frames: frame0 rows then frame1 rows
        attn_2d_host = torch.zeros(SEQ, D_MODEL, dtype=torch.bfloat16)
        for t_idx in range(T):
            attn_2d_host[t_idx * N_PATCH_PAD:(t_idx + 1) * N_PATCH_PAD] = \
                attn_host[:, t_idx].to(torch.bfloat16)
        attn_2d = to_tt(attn_2d_host, tt_device)
        _timer.mark("sdpa")

    # O projection + bias + gated residual
    linear_k32(attn_2d, dev["%s.out_w" % prefix], scr["o_proj"])
    o_with_bias = ttnn.add(scr["o_proj"], dev["%s.out_b" % prefix])
    gated_residual_kernel(x_tt, o_with_bias, gate_msa, scr["z_b"])
    _timer.mark("o_proj+res")

    x_tt = scr["z_b"]

    # MLP path
    layernorm_d1024(x_tt, dev["ln_w_ones"], dev["ln_b_zeros"], scaler, mean_scale, scr["normed"])
    adaln_modulate_kernel(scr["normed"], shift_mlp, scale_mlp, scr["modulated"])
    _timer.mark("mlp_norm+mod")
    linear_bias_k32(scr["modulated"], dev["%s.fc1_w" % prefix], dev["%s.fc1_b" % prefix], scr["fc1"])
    gelu_approx_kernel(scr["fc1"], scr["gelu"])
    _timer.mark("fc1+gelu")
    # TODO: investigate why tt-lang linear_accum kernel has much worse precision than
    # ttnn.matmul for large-K accumulation (4096->1024, k_chunk=32, k_iters=4).
    # Our kernel: max_err=11.7, ttnn.matmul: max_err=0.34 on same inputs.
    scr["fc2"] = ttnn.matmul(scr["gelu"], dev["%s.fc2_w" % prefix])
    fc2_with_bias = ttnn.add(scr["fc2"], dev["%s.fc2_b" % prefix])
    gated_residual_kernel(x_tt, fc2_with_bias, gate_mlp, scr["z_a"])
    _timer.mark("fc2+res")

    _timer.report(prefix, attn_type)
    return scr["z_a"]

def compute_cond_for_frame(t_scalar, action_vec, dev, scr, tt_device):
    """Compute conditioning vector on HOST in fp32 for precision.
    t_scalar: integer timestep
    action_vec: (25,) action tensor or None
    Returns: (TILE, D_MODEL) device tensor
    """
    t_freq = timestep_embedding(torch.tensor([t_scalar]), FREQ_DIM)  # (1, 256)

    # t_embedder MLP on host in fp32: Linear(256,1024) -> SiLU -> Linear(1024,1024)
    h = (t_freq[0].float() @ dev["t_emb_w0_host"]) + dev["t_emb_b0_host"]
    h = h * torch.sigmoid(h)  # SiLU
    cond = (h @ dev["t_emb_w2_host"]) + dev["t_emb_b2_host"]

    # External conditioning on host
    if action_vec is not None:
        ext = (action_vec.float() @ dev["ext_cond_w_host"]) + dev["ext_cond_b_host"]
        cond = cond + ext

    # Package as (TILE, D_MODEL) device tensor - fill ALL rows for device-side broadcast
    cond_bf16 = cond.to(torch.bfloat16)
    cond_pad = cond_bf16.unsqueeze(0).expand(TILE, -1).contiguous()
    return to_tt(cond_pad, tt_device)

def dit_forward(x_latent, timesteps, actions, dev, scr, tt_device, scaler, mean_scale):
    """Full DiT forward pass.
    x_latent: (B, T, C, H, W) = (1, T, 16, 18, 32)
    timesteps: (B, T) = (1, T) long tensor
    actions: (B, T, 25) float tensor
    Returns: (B, T, C, H, W) velocity prediction
    """
    B = 1
    T = x_latent.shape[1]
    SEQ = N_PATCH_PAD * T

    # Patch embedding on host for each frame
    x_pad = torch.zeros(SEQ, D_MODEL, dtype=torch.bfloat16)
    for t_idx in range(T):
        x_frame = x_latent[0, t_idx].unsqueeze(0)  # (1, C, H, W)
        x_2d = patch_embed_host(x_frame, dev["x_emb_conv_w"], dev["x_emb_conv_b"])
        start = t_idx * N_PATCH_PAD
        x_pad[start:start + N_PATCHES] = x_2d
    z_cur = to_tt(x_pad, tt_device)

    # Per-frame conditioning
    cond_list = []
    for t_idx in range(T):
        t_val = timesteps[0, t_idx].item()
        act = actions[0, t_idx] if actions is not None else None
        cond_list.append(compute_cond_for_frame(t_val, act, dev, scr, tt_device))

    # 16 blocks: each has spatial + temporal sub-block
    for block_idx in range(N_BLOCKS):
        z_cur = run_sub_block(
            "blocks.%d.s" % block_idx, z_cur, cond_list,
            dev, scr, tt_device, scaler, mean_scale, attn_type="spatial"
        )
        z_cur = run_sub_block(
            "blocks.%d.t" % block_idx, z_cur, cond_list,
            dev, scr, tt_device, scaler, mean_scale, attn_type="temporal"
        )
        print("  Block %d done" % block_idx)

    # Final layer: per-frame adaLN (on device)
    N_REPEAT = N_PATCH_PAD // TILE  # 5
    per_frame_final = []
    for t_idx in range(T):
        silu_kernel(cond_list[t_idx], scr["silu_cond"])
        final_raw = ttnn.matmul(scr["silu_cond"], dev["final_adaln_w"])
        final_raw = ttnn.add(final_raw, dev["final_adaln_b"])
        expanded = ttnn.concat([final_raw] * N_REPEAT, dim=0)
        per_frame_final.append(expanded)
    if T == 1:
        full_final = per_frame_final[0]
    else:
        full_final = ttnn.concat(per_frame_final, dim=0)

    shift_tt = ttnn.slice(full_final, [0, 0], [SEQ, D_MODEL])
    scale_tt = ttnn.slice(full_final, [0, D_MODEL], [SEQ, 2 * D_MODEL])

    layernorm_d1024(z_cur, dev["ln_w_ones"], dev["ln_b_zeros"], scaler, mean_scale, scr["normed"])
    adaln_modulate_kernel(scr["normed"], shift_tt, scale_tt, scr["modulated"])
    linear_k32(scr["modulated"], dev["final_linear_w"], scr["final_out"])
    scr["final_out"] = ttnn.add(scr["final_out"], dev["final_linear_b"])

    # Read back all frames and unpatchify
    out_host_full = ttnn.to_torch(scr["final_out"])
    out_frames = []
    for t_idx in range(T):
        start = t_idx * N_PATCH_PAD
        out_host = out_host_full[start:start + N_PATCHES].float()
        out_img = unpatchify_host(out_host, PATCH_SIZE, IN_CHANNELS, FRAME_H, FRAME_W)
        out_frames.append(out_img)
    return torch.stack(out_frames, dim=1)  # (1, T, C, H, W)

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

    N_FRAMES = 2  # prompt + generated

    # Load VAE first (needed for encoding prompt)
    print("\nLoading VAE...")
    vae = load_vae_cpu()

    # Encode prompt image
    prompt_path = "/tmp/sample_image_0.png"
    print("Encoding prompt image:", prompt_path)
    from PIL import Image
    prompt_img = Image.open(prompt_path).convert("RGB").resize((640, 360))
    prompt_tensor = torch.tensor(list(prompt_img.getdata()), dtype=torch.float32)
    prompt_tensor = prompt_tensor.reshape(1, 360, 640, 3).permute(0, 3, 1, 2) / 255.0  # (1, 3, 360, 640)
    with torch.no_grad():
        prompt_latent = vae.encode(prompt_tensor * 2 - 1).mean * SCALING_FACTOR
    # prompt_latent: (1, H*W, C) -> reshape to (1, 1, C, H, W)
    prompt_latent = rearrange(prompt_latent, "b (h w) c -> b 1 c h w",
                              h=INPUT_H, w=INPUT_W)
    print("Prompt latent shape:", prompt_latent.shape,
          "range: [%.2f, %.2f]" % (prompt_latent.min().item(), prompt_latent.max().item()))

    # Load weights and scratch (sized for T=2)
    dev = preload_dit_weights(tt_device, n_frames=N_FRAMES)
    scr = prealloc_scratch(tt_device, n_frames=N_FRAMES)

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

    # x = [prompt_latent, noisy_chunk] with T=2
    chunk = torch.randn(1, 1, IN_CHANNELS, INPUT_H, INPUT_W)
    chunk = torch.clamp(chunk, -noise_abs_max, noise_abs_max)

    # Actions: no-op for both frames
    actions = torch.zeros(1, N_FRAMES, EXT_COND_DIM)

    print("\nRunning DDIM denoising (%d steps, T=%d)..." % (ddim_steps, N_FRAMES))
    t_total = time.time()

    for noise_idx in reversed(range(1, ddim_steps + 1)):
        t_step = time.time()

        # Timesteps: [stabilization_level for prompt, noise_level for generated]
        noise_level = int(noise_range[noise_idx].item())
        timesteps = torch.tensor([[stabilization_level - 1, noise_level]], dtype=torch.long)

        print("Step %d/%d (noise_level=%d)..." % (ddim_steps - noise_idx + 1, ddim_steps, noise_level))

        # Concatenate prompt + noisy chunk: (1, 2, C, H, W)
        x_input = torch.cat([prompt_latent, chunk], dim=1)
        v_all = dit_forward(x_input, timesteps, actions, dev, scr, tt_device, scaler, mean_scale)

        # Only use the velocity for the generated frame (index 1)
        v = v_all[:, -1:]  # (1, 1, C, H, W)

        # DDIM update on generated frame only
        t_next_noise = int(noise_range[noise_idx - 1].item())
        t_val_long = torch.full((1, 1), noise_level, dtype=torch.long)
        t_next_long = torch.full((1, 1), max(t_next_noise, noise_level if t_next_noise < 0 else 0), dtype=torch.long)

        alpha_t = alphas_cumprod[t_val_long]
        alpha_next = alphas_cumprod[t_next_long]
        if noise_idx == 1:
            alpha_next = torch.ones_like(alpha_next)

        x_start = alpha_t.sqrt() * chunk - (1 - alpha_t).sqrt() * v
        x_noise = ((1 / alpha_t).sqrt() * chunk - x_start) / (1 / alpha_t - 1).sqrt()
        chunk = alpha_next.sqrt() * x_start + x_noise * (1 - alpha_next).sqrt()

        print("  Step time: %.1fs, v range: [%.2f, %.2f]" % (
            time.time() - t_step, v.min().item(), v.max().item()))

    total_time = time.time() - t_total
    print("\nDDIM complete in %.1fs" % total_time)
    print("Final latent range: [%.2f, %.2f]" % (chunk.min().item(), chunk.max().item()))

    # VAE decode the generated frame
    print("\nRunning VAE decode...")
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
