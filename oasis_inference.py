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

def make_fused_linear_bias_gated_res_kernel(k_chunk):
    """Fused: out = residual + (x @ w + bias) * gate.
    Eliminates DRAM round-trips for intermediate matmul result and bias-add."""
    @ttl.kernel(grid="auto")
    def fused_lbgr_kernel(x, w, bias, gate, residual, out):
        grid_cols, _ = ttl.grid_size(dims=2)
        m_tiles = x.shape[0] // TILE
        n_tiles = w.shape[1] // TILE
        total_out = m_tiles * n_tiles
        tiles_per_core = -(-total_out // grid_cols)
        x_dfb = ttl.make_dataflow_buffer_like(x, shape=(1, k_chunk), buffer_factor=2)
        w_dfb = ttl.make_dataflow_buffer_like(w, shape=(k_chunk, 1), buffer_factor=2)
        b_dfb = ttl.make_dataflow_buffer_like(bias, shape=(1, 1), buffer_factor=2)
        g_dfb = ttl.make_dataflow_buffer_like(gate, shape=(1, 1), buffer_factor=2)
        r_dfb = ttl.make_dataflow_buffer_like(residual, shape=(1, 1), buffer_factor=2)
        mm_dfb = ttl.make_dataflow_buffer_like(out, shape=(1, 1), buffer_factor=2)
        gb_dfb = ttl.make_dataflow_buffer_like(out, shape=(1, 1), buffer_factor=2)
        out_dfb = ttl.make_dataflow_buffer_like(out, shape=(1, 1), buffer_factor=2)
        @ttl.compute()
        def compute():
            core_x, _ = ttl.core(dims=2)
            for local_t in range(tiles_per_core):
                idx = core_x * tiles_per_core + local_t
                if idx < total_out:
                    with x_dfb.wait() as xv, w_dfb.wait() as wv, mm_dfb.reserve() as mm:
                        mm.store(xv @ wv)
                    with mm_dfb.wait() as mmv, b_dfb.wait() as bv, g_dfb.wait() as gv, gb_dfb.reserve() as gb:
                        gb.store((mmv + bv) * gv)
                    with gb_dfb.wait() as gbv, r_dfb.wait() as rv, out_dfb.reserve() as o:
                        o.store(rv + gbv)
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
                    with b_dfb.reserve() as blk:
                        tx = ttl.copy(bias[row, col], blk); tx.wait()
                    with g_dfb.reserve() as blk:
                        tx = ttl.copy(gate[row, col], blk); tx.wait()
                    with r_dfb.reserve() as blk:
                        tx = ttl.copy(residual[row, col], blk); tx.wait()
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
    return fused_lbgr_kernel

def make_fused_linear_bias_gelu_kernel(k_chunk, n_chunk=1):
    """Fused: out = gelu_approx(x @ w + bias). Saves DRAM round-trip between FC1 and GELU."""
    @ttl.kernel(grid="auto")
    def fused_lbg_kernel(x, w, bias, out):
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
                        h = mmv + bv
                        x3 = h * h * h
                        inner = ttl.math.fill(h, 0.7978845608) * (h + ttl.math.fill(h, 0.044715) * x3)
                        o.store(ttl.math.fill(h, 0.5) * h * (ttl.math.fill(h, 1.0) + ttl.math.tanh(inner)))
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
    return fused_lbg_kernel

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
def fused_rope_kernel(a, b, cos_tab, sin_tab, out):
    """out = a * cos + b * sin. Fused multiply-add for RoPE.
    Eliminates 3 ttnn intermediates (2 multiply + 1 add)."""
    grid_cols, _ = ttl.grid_size(dims=2)
    row_tiles = a.shape[0] // TILE
    col_blocks = a.shape[1] // TILE // ELEM_GRAN
    total = row_tiles * col_blocks
    tiles_per_core = -(-total // grid_cols)
    a_dfb = ttl.make_dataflow_buffer_like(a, shape=(1, ELEM_GRAN), buffer_factor=2)
    b_dfb = ttl.make_dataflow_buffer_like(b, shape=(1, ELEM_GRAN), buffer_factor=2)
    c_dfb = ttl.make_dataflow_buffer_like(cos_tab, shape=(1, ELEM_GRAN), buffer_factor=2)
    s_dfb = ttl.make_dataflow_buffer_like(sin_tab, shape=(1, ELEM_GRAN), buffer_factor=2)
    tmp_dfb = ttl.make_dataflow_buffer_like(out, shape=(1, ELEM_GRAN), buffer_factor=2)
    out_dfb = ttl.make_dataflow_buffer_like(out, shape=(1, ELEM_GRAN), buffer_factor=2)
    @ttl.compute()
    def compute():
        core_x, _ = ttl.core(dims=2)
        for local_t in range(tiles_per_core):
            t = core_x * tiles_per_core + local_t
            if t < total:
                with a_dfb.wait() as av, c_dfb.wait() as cv, tmp_dfb.reserve() as tmp:
                    tmp.store(av * cv)
                with tmp_dfb.wait() as tv, b_dfb.wait() as bv, s_dfb.wait() as sv, out_dfb.reserve() as o:
                    o.store(tv + bv * sv)
    @ttl.datamovement()
    def dm_read():
        core_x, _ = ttl.core(dims=2)
        for local_t in range(tiles_per_core):
            t = core_x * tiles_per_core + local_t
            if t < total:
                row = t // col_blocks
                cb = t % col_blocks
                sc = cb * ELEM_GRAN
                with a_dfb.reserve() as blk:
                    tx = ttl.copy(a[row, sc:sc + ELEM_GRAN], blk); tx.wait()
                with c_dfb.reserve() as blk:
                    tx = ttl.copy(cos_tab[row, sc:sc + ELEM_GRAN], blk); tx.wait()
                with b_dfb.reserve() as blk:
                    tx = ttl.copy(b[row, sc:sc + ELEM_GRAN], blk); tx.wait()
                with s_dfb.reserve() as blk:
                    tx = ttl.copy(sin_tab[row, sc:sc + ELEM_GRAN], blk); tx.wait()
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

N_PATCH_PAD_TILES = N_PATCH_PAD // TILE  # 5
D_HEAD_TILES = D_HEAD // TILE  # 2

def make_rope_sdpa_kernel(seq_tiles, head_tiles, n_heads_val, scale_val):
    """Fused RoPE + SDPA kernel. Reads from combined QKV tensor (SEQ, 5*D_MODEL)
    directly, applies RoPE to Q and K, then computes spatial SDPA per head.
    Layout of qkv_full columns: [Q | K | V | Q_swap | K_swap], each D_MODEL wide.
    Eliminates 5 ttnn.slice + 2 RoPE kernels + all reshape/permute ops."""
    @ttl.kernel(grid="auto")
    def rope_sdpa(qkv_full, cos_tab, sin_tab, scaler, out):
        grid_cols, _ = ttl.grid_size(dims=2)
        n_frames = qkv_full.shape[0] // TILE // seq_tiles
        total_heads = n_frames * n_heads_val
        heads_per_core = -(-total_heads // grid_cols)

        # Input DFBs from DRAM
        q_dfb = ttl.make_dataflow_buffer_like(qkv_full, shape=(seq_tiles, head_tiles), buffer_factor=2)
        qs_dfb = ttl.make_dataflow_buffer_like(qkv_full, shape=(seq_tiles, head_tiles), buffer_factor=2)
        k_dfb = ttl.make_dataflow_buffer_like(qkv_full, shape=(seq_tiles, head_tiles), buffer_factor=2)
        ks_dfb = ttl.make_dataflow_buffer_like(qkv_full, shape=(seq_tiles, head_tiles), buffer_factor=2)
        v_dfb = ttl.make_dataflow_buffer_like(qkv_full, shape=(seq_tiles, head_tiles), buffer_factor=2)
        cos_dfb = ttl.make_dataflow_buffer_like(cos_tab, shape=(seq_tiles, head_tiles), buffer_factor=2)
        sin_dfb = ttl.make_dataflow_buffer_like(sin_tab, shape=(seq_tiles, head_tiles), buffer_factor=2)
        scaler_dfb = ttl.make_dataflow_buffer_like(scaler, shape=(1, 1), buffer_factor=2)

        # RoPE intermediate DFBs
        qr_dfb = ttl.make_dataflow_buffer_like(qkv_full, shape=(seq_tiles, head_tiles), buffer_factor=2)
        kr_dfb = ttl.make_dataflow_buffer_like(qkv_full, shape=(seq_tiles, head_tiles), buffer_factor=2)

        # SDPA DFBs
        kt_dfb = ttl.make_dataflow_buffer_like(qkv_full, shape=(head_tiles, seq_tiles), buffer_factor=2)
        a_dfb = ttl.make_dataflow_buffer_like(scaler, shape=(seq_tiles, seq_tiles), buffer_factor=2)
        b_dfb = ttl.make_dataflow_buffer_like(scaler, shape=(seq_tiles, seq_tiles), buffer_factor=2)
        c_dfb = ttl.make_dataflow_buffer_like(scaler, shape=(seq_tiles, seq_tiles), buffer_factor=2)
        row_dfb = ttl.make_dataflow_buffer_like(scaler, shape=(seq_tiles, 1), buffer_factor=2)
        row_bc_dfb = ttl.make_dataflow_buffer_like(scaler, shape=(seq_tiles, seq_tiles), buffer_factor=2)
        out_dfb = ttl.make_dataflow_buffer_like(out, shape=(seq_tiles, head_tiles), buffer_factor=2)

        @ttl.compute()
        def compute():
            core_x, _ = ttl.core(dims=2)
            for local_h in range(heads_per_core):
                head_idx = core_x * heads_per_core + local_h
                if head_idx < total_heads:
                    with cos_dfb.wait() as cv, sin_dfb.wait() as sv, v_dfb.wait() as vv, scaler_dfb.wait() as sc:
                        with q_dfb.wait() as q, qs_dfb.wait() as qs, qr_dfb.reserve() as qr:
                            qr.store(q * cv + qs * sv)
                        with k_dfb.wait() as k, ks_dfb.wait() as ks, kr_dfb.reserve() as kr:
                            kr.store(k * cv + ks * sv)
                        with qr_dfb.wait() as qrv, kr_dfb.wait() as krv:
                            with kt_dfb.reserve() as kt:
                                kt.store(ttl.transpose(krv))
                            with kt_dfb.wait() as ktv, a_dfb.reserve() as qk:
                                qk.store(qrv @ ktv)
                            with a_dfb.wait() as qkv, b_dfb.reserve() as scaled:
                                scaled.store(qkv * ttl.math.fill(qkv, scale_val))
                            with b_dfb.wait() as sdv:
                                with row_dfb.reserve() as mx:
                                    mx.store(ttl.math.reduce_max(sdv, sc, dims=[1]))
                                with row_dfb.wait() as mxv, row_bc_dfb.reserve() as mxb:
                                    mxb.store(ttl.math.broadcast(mxv, dims=[1]))
                                with row_bc_dfb.wait() as mxbv:
                                    with a_dfb.reserve() as ex:
                                        ex.store(ttl.math.exp(sdv - mxbv))
                                with a_dfb.wait() as exv:
                                    with row_dfb.reserve() as sm:
                                        sm.store(ttl.math.reduce_sum(exv, sc, dims=[1]))
                                    with row_dfb.wait() as smv, row_bc_dfb.reserve() as smb:
                                        smb.store(ttl.math.broadcast(smv, dims=[1]))
                                    with row_bc_dfb.wait() as smbv, c_dfb.reserve() as attn:
                                        attn.store(exv * ttl.math.recip(smbv))
                            with c_dfb.wait() as av, out_dfb.reserve() as o:
                                o.store(av @ vv)

        @ttl.datamovement()
        def dm_read():
            core_x, _ = ttl.core(dims=2)
            d_tiles = n_heads_val * head_tiles
            for local_h in range(heads_per_core):
                head_idx = core_x * heads_per_core + local_h
                if head_idx < total_heads:
                    frame = head_idx // n_heads_val
                    h = head_idx % n_heads_val
                    row_start = frame * seq_tiles
                    hc = h * head_tiles
                    # Q at offset 0, K at d_tiles, V at 2*d_tiles
                    # Q_swap at 3*d_tiles, K_swap at 4*d_tiles
                    with q_dfb.reserve() as blk:
                        tx = ttl.copy(qkv_full[row_start:row_start + seq_tiles, hc:hc + head_tiles], blk); tx.wait()
                    with qs_dfb.reserve() as blk:
                        tx = ttl.copy(qkv_full[row_start:row_start + seq_tiles, 3 * d_tiles + hc:3 * d_tiles + hc + head_tiles], blk); tx.wait()
                    with k_dfb.reserve() as blk:
                        tx = ttl.copy(qkv_full[row_start:row_start + seq_tiles, d_tiles + hc:d_tiles + hc + head_tiles], blk); tx.wait()
                    with ks_dfb.reserve() as blk:
                        tx = ttl.copy(qkv_full[row_start:row_start + seq_tiles, 4 * d_tiles + hc:4 * d_tiles + hc + head_tiles], blk); tx.wait()
                    with v_dfb.reserve() as blk:
                        tx = ttl.copy(qkv_full[row_start:row_start + seq_tiles, 2 * d_tiles + hc:2 * d_tiles + hc + head_tiles], blk); tx.wait()
                    with cos_dfb.reserve() as blk:
                        tx = ttl.copy(cos_tab[row_start:row_start + seq_tiles, hc:hc + head_tiles], blk); tx.wait()
                    with sin_dfb.reserve() as blk:
                        tx = ttl.copy(sin_tab[row_start:row_start + seq_tiles, hc:hc + head_tiles], blk); tx.wait()
                    with scaler_dfb.reserve() as blk:
                        tx = ttl.copy(scaler[0, 0], blk); tx.wait()

        @ttl.datamovement()
        def dm_write():
            core_x, _ = ttl.core(dims=2)
            for local_h in range(heads_per_core):
                head_idx = core_x * heads_per_core + local_h
                if head_idx < total_heads:
                    frame = head_idx // n_heads_val
                    h = head_idx % n_heads_val
                    row_start = frame * seq_tiles
                    col_start = h * head_tiles
                    with out_dfb.wait() as blk:
                        tx = ttl.copy(blk, out[row_start:row_start + seq_tiles, col_start:col_start + head_tiles]); tx.wait()

    return rope_sdpa

rope_sdpa_kernel = make_rope_sdpa_kernel(N_PATCH_PAD_TILES, D_HEAD_TILES, N_HEADS, 0.125)

def make_qkv_rope_sdpa_kernel(dim_tiles, seq_tiles, head_tiles, n_heads_val, scale_val):
    """Mega-fused QKV matmul + RoPE + SDPA kernel.
    Reads pre-modulated input and QKV weights, does K-accumulation matmul per head,
    applies RoPE, then SDPA. Q/K/V stay in L1 between matmul and attention."""
    @ttl.kernel(grid="auto")
    def qkv_rope_sdpa(modulated, qkv_w, cos_tab, sin_tab, scaler, out):
        grid_cols, _ = ttl.grid_size(dims=2)
        n_frames = modulated.shape[0] // TILE // seq_tiles
        total_heads = n_frames * n_heads_val
        heads_per_core = -(-total_heads // grid_cols)
        d_tiles = n_heads_val * head_tiles

        mod_dfb = ttl.make_dataflow_buffer_like(modulated, shape=(seq_tiles, 1), buffer_factor=2)
        w_dfb = ttl.make_dataflow_buffer_like(qkv_w, shape=(1, head_tiles), buffer_factor=2)
        acc_dfb = ttl.make_dataflow_buffer_like(modulated, shape=(seq_tiles, head_tiles), buffer_factor=2)
        mm_dfb = ttl.make_dataflow_buffer_like(modulated, shape=(seq_tiles, head_tiles), buffer_factor=2)
        q_dfb = ttl.make_dataflow_buffer_like(modulated, shape=(seq_tiles, head_tiles), buffer_factor=2)
        k_dfb = ttl.make_dataflow_buffer_like(modulated, shape=(seq_tiles, head_tiles), buffer_factor=2)
        v_dfb = ttl.make_dataflow_buffer_like(modulated, shape=(seq_tiles, head_tiles), buffer_factor=2)
        cos_dfb = ttl.make_dataflow_buffer_like(cos_tab, shape=(seq_tiles, head_tiles), buffer_factor=2)
        sin_dfb = ttl.make_dataflow_buffer_like(sin_tab, shape=(seq_tiles, head_tiles), buffer_factor=2)
        qr_dfb = ttl.make_dataflow_buffer_like(modulated, shape=(seq_tiles, head_tiles), buffer_factor=2)
        kr_dfb = ttl.make_dataflow_buffer_like(modulated, shape=(seq_tiles, head_tiles), buffer_factor=2)
        kt_dfb = ttl.make_dataflow_buffer_like(modulated, shape=(head_tiles, seq_tiles), buffer_factor=2)
        a_dfb = ttl.make_dataflow_buffer_like(scaler, shape=(seq_tiles, seq_tiles), buffer_factor=2)
        b_dfb = ttl.make_dataflow_buffer_like(scaler, shape=(seq_tiles, seq_tiles), buffer_factor=2)
        c_dfb = ttl.make_dataflow_buffer_like(scaler, shape=(seq_tiles, seq_tiles), buffer_factor=2)
        row_dfb = ttl.make_dataflow_buffer_like(scaler, shape=(seq_tiles, 1), buffer_factor=2)
        row_bc_dfb = ttl.make_dataflow_buffer_like(scaler, shape=(seq_tiles, seq_tiles), buffer_factor=2)
        scaler_dfb = ttl.make_dataflow_buffer_like(scaler, shape=(1, 1), buffer_factor=2)
        out_dfb = ttl.make_dataflow_buffer_like(out, shape=(seq_tiles, head_tiles), buffer_factor=2)

        @ttl.compute()
        def compute():
            core_x, _ = ttl.core(dims=2)
            for local_h in range(heads_per_core):
                head_idx = core_x * heads_per_core + local_h
                if head_idx < total_heads:
                    # Q matmul
                    with mod_dfb.wait() as m0, w_dfb.wait() as w0, acc_dfb.reserve() as a:
                        a.store(m0 @ w0)
                    for k_idx in range(dim_tiles - 1):
                        with mod_dfb.wait() as mk, w_dfb.wait() as wk, mm_dfb.reserve() as mm:
                            mm.store(mk @ wk)
                        with mm_dfb.wait() as mmv, acc_dfb.wait() as prev, acc_dfb.reserve() as a:
                            a.store(prev + mmv)
                    with acc_dfb.wait() as qr, q_dfb.reserve() as q:
                        q.store(qr)
                    # Qs matmul + RoPE Q
                    with mod_dfb.wait() as m0, w_dfb.wait() as w0, acc_dfb.reserve() as a:
                        a.store(m0 @ w0)
                    for k_idx in range(dim_tiles - 1):
                        with mod_dfb.wait() as mk, w_dfb.wait() as wk, mm_dfb.reserve() as mm:
                            mm.store(mk @ wk)
                        with mm_dfb.wait() as mmv, acc_dfb.wait() as prev, acc_dfb.reserve() as a:
                            a.store(prev + mmv)
                    with acc_dfb.wait() as qs, q_dfb.wait() as qv, cos_dfb.wait() as cv, sin_dfb.wait() as sv:
                        with qr_dfb.reserve() as qr:
                            qr.store(qv * cv + qs * sv)
                    # K matmul
                    with mod_dfb.wait() as m0, w_dfb.wait() as w0, acc_dfb.reserve() as a:
                        a.store(m0 @ w0)
                    for k_idx in range(dim_tiles - 1):
                        with mod_dfb.wait() as mk, w_dfb.wait() as wk, mm_dfb.reserve() as mm:
                            mm.store(mk @ wk)
                        with mm_dfb.wait() as mmv, acc_dfb.wait() as prev, acc_dfb.reserve() as a:
                            a.store(prev + mmv)
                    with acc_dfb.wait() as kr, k_dfb.reserve() as k:
                        k.store(kr)
                    # Ks matmul + RoPE K
                    with mod_dfb.wait() as m0, w_dfb.wait() as w0, acc_dfb.reserve() as a:
                        a.store(m0 @ w0)
                    for k_idx in range(dim_tiles - 1):
                        with mod_dfb.wait() as mk, w_dfb.wait() as wk, mm_dfb.reserve() as mm:
                            mm.store(mk @ wk)
                        with mm_dfb.wait() as mmv, acc_dfb.wait() as prev, acc_dfb.reserve() as a:
                            a.store(prev + mmv)
                    with acc_dfb.wait() as ks, k_dfb.wait() as kv, cos_dfb.wait() as cv, sin_dfb.wait() as sv:
                        with kr_dfb.reserve() as kr:
                            kr.store(kv * cv + ks * sv)
                    # V matmul
                    with mod_dfb.wait() as m0, w_dfb.wait() as w0, acc_dfb.reserve() as a:
                        a.store(m0 @ w0)
                    for k_idx in range(dim_tiles - 1):
                        with mod_dfb.wait() as mk, w_dfb.wait() as wk, mm_dfb.reserve() as mm:
                            mm.store(mk @ wk)
                        with mm_dfb.wait() as mmv, acc_dfb.wait() as prev, acc_dfb.reserve() as a:
                            a.store(prev + mmv)
                    with acc_dfb.wait() as vr, v_dfb.reserve() as v:
                        v.store(vr)
                    # SDPA
                    with qr_dfb.wait() as qrv, kr_dfb.wait() as krv, v_dfb.wait() as vv, scaler_dfb.wait() as sc:
                        with kt_dfb.reserve() as kt:
                            kt.store(ttl.transpose(krv))
                        with kt_dfb.wait() as ktv, a_dfb.reserve() as qk:
                            qk.store(qrv @ ktv)
                        with a_dfb.wait() as qkv, b_dfb.reserve() as scaled:
                            scaled.store(qkv * ttl.math.fill(qkv, scale_val))
                        with b_dfb.wait() as sdv:
                            with row_dfb.reserve() as mx:
                                mx.store(ttl.math.reduce_max(sdv, sc, dims=[1]))
                            with row_dfb.wait() as mxv, row_bc_dfb.reserve() as mxb:
                                mxb.store(ttl.math.broadcast(mxv, dims=[1]))
                            with row_bc_dfb.wait() as mxbv:
                                with a_dfb.reserve() as ex:
                                    ex.store(ttl.math.exp(sdv - mxbv))
                            with a_dfb.wait() as exv:
                                with row_dfb.reserve() as sm:
                                    sm.store(ttl.math.reduce_sum(exv, sc, dims=[1]))
                                with row_dfb.wait() as smv, row_bc_dfb.reserve() as smb:
                                    smb.store(ttl.math.broadcast(smv, dims=[1]))
                                with row_bc_dfb.wait() as smbv, c_dfb.reserve() as attn:
                                    attn.store(exv * ttl.math.recip(smbv))
                        with c_dfb.wait() as av, out_dfb.reserve() as o:
                            o.store(av @ vv)

        @ttl.datamovement()
        def dm_read():
            core_x, _ = ttl.core(dims=2)
            for local_h in range(heads_per_core):
                head_idx = core_x * heads_per_core + local_h
                if head_idx < total_heads:
                    frame = head_idx // n_heads_val
                    h = head_idx % n_heads_val
                    row_start = frame * seq_tiles
                    hc = h * head_tiles
                    d_t = n_heads_val * head_tiles
                    # Q matmul: weights at column hc
                    for k in range(dim_tiles):
                        with mod_dfb.reserve() as blk:
                            tx = ttl.copy(modulated[row_start:row_start + seq_tiles, k], blk); tx.wait()
                        with w_dfb.reserve() as blk:
                            tx = ttl.copy(qkv_w[k, hc:hc + head_tiles], blk); tx.wait()
                    # Qs matmul: weights at column 3*d_t+hc
                    qs_col = 3 * d_t + hc
                    for k in range(dim_tiles):
                        with mod_dfb.reserve() as blk:
                            tx = ttl.copy(modulated[row_start:row_start + seq_tiles, k], blk); tx.wait()
                        with w_dfb.reserve() as blk:
                            tx = ttl.copy(qkv_w[k, qs_col:qs_col + head_tiles], blk); tx.wait()
                    with cos_dfb.reserve() as blk:
                        tx = ttl.copy(cos_tab[row_start:row_start + seq_tiles, hc:hc + head_tiles], blk); tx.wait()
                    with sin_dfb.reserve() as blk:
                        tx = ttl.copy(sin_tab[row_start:row_start + seq_tiles, hc:hc + head_tiles], blk); tx.wait()
                    # K matmul: weights at column d_t+hc
                    k_col = d_t + hc
                    for k in range(dim_tiles):
                        with mod_dfb.reserve() as blk:
                            tx = ttl.copy(modulated[row_start:row_start + seq_tiles, k], blk); tx.wait()
                        with w_dfb.reserve() as blk:
                            tx = ttl.copy(qkv_w[k, k_col:k_col + head_tiles], blk); tx.wait()
                    # Ks matmul: weights at column 4*d_t+hc
                    ks_col = 4 * d_t + hc
                    for k in range(dim_tiles):
                        with mod_dfb.reserve() as blk:
                            tx = ttl.copy(modulated[row_start:row_start + seq_tiles, k], blk); tx.wait()
                        with w_dfb.reserve() as blk:
                            tx = ttl.copy(qkv_w[k, ks_col:ks_col + head_tiles], blk); tx.wait()
                    with cos_dfb.reserve() as blk:
                        tx = ttl.copy(cos_tab[row_start:row_start + seq_tiles, hc:hc + head_tiles], blk); tx.wait()
                    with sin_dfb.reserve() as blk:
                        tx = ttl.copy(sin_tab[row_start:row_start + seq_tiles, hc:hc + head_tiles], blk); tx.wait()
                    # V matmul: weights at column 2*d_t+hc
                    v_col = 2 * d_t + hc
                    for k in range(dim_tiles):
                        with mod_dfb.reserve() as blk:
                            tx = ttl.copy(modulated[row_start:row_start + seq_tiles, k], blk); tx.wait()
                        with w_dfb.reserve() as blk:
                            tx = ttl.copy(qkv_w[k, v_col:v_col + head_tiles], blk); tx.wait()
                    # Scaler for SDPA
                    with scaler_dfb.reserve() as blk:
                        tx = ttl.copy(scaler[0, 0], blk); tx.wait()

        @ttl.datamovement()
        def dm_write():
            core_x, _ = ttl.core(dims=2)
            for local_h in range(heads_per_core):
                head_idx = core_x * heads_per_core + local_h
                if head_idx < total_heads:
                    frame = head_idx // n_heads_val
                    h = head_idx % n_heads_val
                    row_start = frame * seq_tiles
                    col_start = h * head_tiles
                    with out_dfb.wait() as blk:
                        tx = ttl.copy(blk, out[row_start:row_start + seq_tiles, col_start:col_start + head_tiles]); tx.wait()

    return qkv_rope_sdpa

mega_qkv_rope_sdpa = make_qkv_rope_sdpa_kernel(D_TILES, N_PATCH_PAD_TILES, D_HEAD_TILES, N_HEADS, 0.125)


def make_temporal_qkv_rope_sdpa_kernel(dim_tiles, seq_tiles, head_tiles, n_heads_val, scale_val):
    """Temporal QKV+RoPE+SDPA: fused kernel for T=2 causal temporal attention.
    Parallelize over heads. Each head processes both frames.
    For T=2 causal: frame0 output = V0, frame1 output = softmax(Q1@K) @ V.
    Uses DRAM scratch for Q/K/V between QKV matmul and SDPA phases."""
    @ttl.kernel(grid="auto")
    def temporal_qkv_rope_sdpa(modulated, qkv_w, cos_tab, sin_tab, scaler, q_scratch, k_scratch, v_scratch, out):
        grid_cols, _ = ttl.grid_size(dims=2)
        heads_per_core = -(-n_heads_val // grid_cols)
        d_tiles = n_heads_val * head_tiles

        # Phase 1: QKV matmul + RoPE (same DFBs as spatial)
        mod_dfb = ttl.make_dataflow_buffer_like(modulated, shape=(seq_tiles, 1), buffer_factor=2)
        w_dfb = ttl.make_dataflow_buffer_like(qkv_w, shape=(1, head_tiles), buffer_factor=2)
        acc_dfb = ttl.make_dataflow_buffer_like(modulated, shape=(seq_tiles, head_tiles), buffer_factor=2)
        mm_dfb = ttl.make_dataflow_buffer_like(modulated, shape=(seq_tiles, head_tiles), buffer_factor=2)
        q_dfb = ttl.make_dataflow_buffer_like(modulated, shape=(seq_tiles, head_tiles), buffer_factor=2)
        k_dfb = ttl.make_dataflow_buffer_like(modulated, shape=(seq_tiles, head_tiles), buffer_factor=2)
        v_dfb = ttl.make_dataflow_buffer_like(modulated, shape=(seq_tiles, head_tiles), buffer_factor=2)
        cos_dfb = ttl.make_dataflow_buffer_like(cos_tab, shape=(seq_tiles, head_tiles), buffer_factor=2)
        sin_dfb = ttl.make_dataflow_buffer_like(sin_tab, shape=(seq_tiles, head_tiles), buffer_factor=2)
        qr_dfb = ttl.make_dataflow_buffer_like(modulated, shape=(seq_tiles, head_tiles), buffer_factor=2)
        kr_dfb = ttl.make_dataflow_buffer_like(modulated, shape=(seq_tiles, head_tiles), buffer_factor=2)
        # Phase 2: temporal SDPA per tile row
        # Read Q1, K0, K1, V0, V1 per tile row (1, head_tiles) each
        tq1_dfb = ttl.make_dataflow_buffer_like(modulated, shape=(1, head_tiles), buffer_factor=2)
        tk0_dfb = ttl.make_dataflow_buffer_like(modulated, shape=(1, head_tiles), buffer_factor=2)
        tk1_dfb = ttl.make_dataflow_buffer_like(modulated, shape=(1, head_tiles), buffer_factor=2)
        tv0_dfb = ttl.make_dataflow_buffer_like(modulated, shape=(1, head_tiles), buffer_factor=2)
        tv1_dfb = ttl.make_dataflow_buffer_like(modulated, shape=(1, head_tiles), buffer_factor=2)
        # Intermediates for temporal SDPA
        prod_dfb = ttl.make_dataflow_buffer_like(modulated, shape=(1, head_tiles), buffer_factor=2)
        scaler_dfb = ttl.make_dataflow_buffer_like(scaler, shape=(1, 1), buffer_factor=2)
        s0_dfb = ttl.make_dataflow_buffer_like(scaler, shape=(1, 1), buffer_factor=2)
        s1_dfb = ttl.make_dataflow_buffer_like(scaler, shape=(1, 1), buffer_factor=2)
        a_dfb = ttl.make_dataflow_buffer_like(scaler, shape=(1, 1), buffer_factor=2)
        a_bc_dfb = ttl.make_dataflow_buffer_like(modulated, shape=(1, head_tiles), buffer_factor=2)
        out_dfb = ttl.make_dataflow_buffer_like(out, shape=(1, head_tiles), buffer_factor=2)

        @ttl.compute()
        def compute():
            core_x, _ = ttl.core(dims=2)
            for local_h in range(heads_per_core):
                head_idx = core_x * heads_per_core + local_h
                if head_idx < n_heads_val:
                    # Phase 1: QKV matmul + RoPE for BOTH frames
                    for frame in range(2):
                        # Q matmul (K-accumulation)
                        with mod_dfb.wait() as m0, w_dfb.wait() as w0, acc_dfb.reserve() as a:
                            a.store(m0 @ w0)
                        for k_idx in range(dim_tiles - 1):
                            with mod_dfb.wait() as mk, w_dfb.wait() as wk, mm_dfb.reserve() as mm:
                                mm.store(mk @ wk)
                            with mm_dfb.wait() as mmv, acc_dfb.wait() as prev, acc_dfb.reserve() as a:
                                a.store(prev + mmv)
                        with acc_dfb.wait() as qr, q_dfb.reserve() as q:
                            q.store(qr)
                        # Qs matmul + RoPE Q
                        with mod_dfb.wait() as m0, w_dfb.wait() as w0, acc_dfb.reserve() as a:
                            a.store(m0 @ w0)
                        for k_idx in range(dim_tiles - 1):
                            with mod_dfb.wait() as mk, w_dfb.wait() as wk, mm_dfb.reserve() as mm:
                                mm.store(mk @ wk)
                            with mm_dfb.wait() as mmv, acc_dfb.wait() as prev, acc_dfb.reserve() as a:
                                a.store(prev + mmv)
                        with acc_dfb.wait() as qs, q_dfb.wait() as qv, cos_dfb.wait() as cv, sin_dfb.wait() as sv:
                            with qr_dfb.reserve() as qr:
                                qr.store(qv * cv + qs * sv)
                        # K matmul
                        with mod_dfb.wait() as m0, w_dfb.wait() as w0, acc_dfb.reserve() as a:
                            a.store(m0 @ w0)
                        for k_idx in range(dim_tiles - 1):
                            with mod_dfb.wait() as mk, w_dfb.wait() as wk, mm_dfb.reserve() as mm:
                                mm.store(mk @ wk)
                            with mm_dfb.wait() as mmv, acc_dfb.wait() as prev, acc_dfb.reserve() as a:
                                a.store(prev + mmv)
                        with acc_dfb.wait() as kr, k_dfb.reserve() as k:
                            k.store(kr)
                        # Ks matmul + RoPE K
                        with mod_dfb.wait() as m0, w_dfb.wait() as w0, acc_dfb.reserve() as a:
                            a.store(m0 @ w0)
                        for k_idx in range(dim_tiles - 1):
                            with mod_dfb.wait() as mk, w_dfb.wait() as wk, mm_dfb.reserve() as mm:
                                mm.store(mk @ wk)
                            with mm_dfb.wait() as mmv, acc_dfb.wait() as prev, acc_dfb.reserve() as a:
                                a.store(prev + mmv)
                        with acc_dfb.wait() as ks, k_dfb.wait() as kv, cos_dfb.wait() as cv, sin_dfb.wait() as sv:
                            with kr_dfb.reserve() as kr:
                                kr.store(kv * cv + ks * sv)
                        # V matmul
                        with mod_dfb.wait() as m0, w_dfb.wait() as w0, acc_dfb.reserve() as a:
                            a.store(m0 @ w0)
                        for k_idx in range(dim_tiles - 1):
                            with mod_dfb.wait() as mk, w_dfb.wait() as wk, mm_dfb.reserve() as mm:
                                mm.store(mk @ wk)
                            with mm_dfb.wait() as mmv, acc_dfb.wait() as prev, acc_dfb.reserve() as a:
                                a.store(prev + mmv)
                        with acc_dfb.wait() as vr, v_dfb.reserve() as v:
                            v.store(vr)

                    # Phase 2: temporal SDPA per tile row
                    # T=2 causal: frame0 out = V0, frame1 out = softmax(Q1@K0/K1) @ V
                    with scaler_dfb.wait() as sc:
                        for r in range(seq_tiles):
                            # Frame 0: output = V0 (causal, only attends to self)
                            with tv0_dfb.wait() as v0:
                                with out_dfb.reserve() as o:
                                    o.store(v0)
                            # Frame 1: score0 = Q1@K0, score1 = Q1@K1
                            with tq1_dfb.wait() as q1, tk0_dfb.wait() as k0, tk1_dfb.wait() as k1:
                                # score0 = scale * reduce_sum(Q1 * K0, dim=1)
                                with prod_dfb.reserve() as p:
                                    p.store(q1 * k0 * ttl.math.fill(q1, scale_val))
                                with prod_dfb.wait() as pv, s0_dfb.reserve() as s0:
                                    s0.store(ttl.math.reduce_sum(pv, sc, dims=[1]))
                                # score1 = scale * reduce_sum(Q1 * K1, dim=1)
                                with prod_dfb.reserve() as p:
                                    p.store(q1 * k1 * ttl.math.fill(q1, scale_val))
                                with prod_dfb.wait() as pv, s1_dfb.reserve() as s1:
                                    s1.store(ttl.math.reduce_sum(pv, sc, dims=[1]))
                            # Softmax of 2 scores per row
                            with s0_dfb.wait() as s0v, s1_dfb.wait() as s1v:
                                mx = ttl.math.max(s0v, s1v)
                                e0 = ttl.math.exp(s0v - mx)
                                e1 = ttl.math.exp(s1v - mx)
                                inv_sum = ttl.math.recip(e0 + e1)
                                with a_dfb.reserve() as a0:
                                    a0.store(e0 * inv_sum)
                            # Broadcast a0 to (1, head_tiles), compute out1
                            with a_dfb.wait() as a0v, a_bc_dfb.reserve() as a0bc:
                                a0bc.store(ttl.math.broadcast(a0v, dims=[1]))
                            with a_bc_dfb.wait() as a0_bc, tv0_dfb.wait() as v0, tv1_dfb.wait() as v1:
                                # out1 = a0 * V0 + (1 - a0) * V1
                                with out_dfb.reserve() as o:
                                    o.store(a0_bc * v0 + (ttl.math.fill(v0, 1.0) - a0_bc) * v1)

        @ttl.datamovement()
        def dm_read():
            core_x, _ = ttl.core(dims=2)
            for local_h in range(heads_per_core):
                head_idx = core_x * heads_per_core + local_h
                if head_idx < n_heads_val:
                    h = head_idx
                    hc = h * head_tiles
                    d_t = n_heads_val * head_tiles
                    # Phase 1: QKV matmul data for each frame
                    for frame in range(2):
                        row_start = frame * seq_tiles
                        # Q weights at column hc
                        for k in range(dim_tiles):
                            with mod_dfb.reserve() as blk:
                                tx = ttl.copy(modulated[row_start:row_start + seq_tiles, k], blk); tx.wait()
                            with w_dfb.reserve() as blk:
                                tx = ttl.copy(qkv_w[k, hc:hc + head_tiles], blk); tx.wait()
                        # Qs weights at 3*d_t+hc
                        qs_col = 3 * d_t + hc
                        for k in range(dim_tiles):
                            with mod_dfb.reserve() as blk:
                                tx = ttl.copy(modulated[row_start:row_start + seq_tiles, k], blk); tx.wait()
                            with w_dfb.reserve() as blk:
                                tx = ttl.copy(qkv_w[k, qs_col:qs_col + head_tiles], blk); tx.wait()
                        with cos_dfb.reserve() as blk:
                            tx = ttl.copy(cos_tab[row_start:row_start + seq_tiles, hc:hc + head_tiles], blk); tx.wait()
                        with sin_dfb.reserve() as blk:
                            tx = ttl.copy(sin_tab[row_start:row_start + seq_tiles, hc:hc + head_tiles], blk); tx.wait()
                        # K weights at d_t+hc
                        k_col = d_t + hc
                        for k in range(dim_tiles):
                            with mod_dfb.reserve() as blk:
                                tx = ttl.copy(modulated[row_start:row_start + seq_tiles, k], blk); tx.wait()
                            with w_dfb.reserve() as blk:
                                tx = ttl.copy(qkv_w[k, k_col:k_col + head_tiles], blk); tx.wait()
                        # Ks weights at 4*d_t+hc
                        ks_col = 4 * d_t + hc
                        for k in range(dim_tiles):
                            with mod_dfb.reserve() as blk:
                                tx = ttl.copy(modulated[row_start:row_start + seq_tiles, k], blk); tx.wait()
                            with w_dfb.reserve() as blk:
                                tx = ttl.copy(qkv_w[k, ks_col:ks_col + head_tiles], blk); tx.wait()
                        with cos_dfb.reserve() as blk:
                            tx = ttl.copy(cos_tab[row_start:row_start + seq_tiles, hc:hc + head_tiles], blk); tx.wait()
                        with sin_dfb.reserve() as blk:
                            tx = ttl.copy(sin_tab[row_start:row_start + seq_tiles, hc:hc + head_tiles], blk); tx.wait()
                        # V weights at 2*d_t+hc
                        v_col = 2 * d_t + hc
                        for k in range(dim_tiles):
                            with mod_dfb.reserve() as blk:
                                tx = ttl.copy(modulated[row_start:row_start + seq_tiles, k], blk); tx.wait()
                            with w_dfb.reserve() as blk:
                                tx = ttl.copy(qkv_w[k, v_col:v_col + head_tiles], blk); tx.wait()
                    # Phase 2: read Q/K/V from scratch for temporal SDPA
                    with scaler_dfb.reserve() as blk:
                        tx = ttl.copy(scaler[0, 0], blk); tx.wait()
                    for r in range(seq_tiles):
                        # V0 for frame 0 output
                        with tv0_dfb.reserve() as blk:
                            tx = ttl.copy(v_scratch[r, hc:hc + head_tiles], blk); tx.wait()
                        # Q1, K0, K1, V0, V1 for frame 1 attention
                        with tq1_dfb.reserve() as blk:
                            tx = ttl.copy(q_scratch[seq_tiles + r, hc:hc + head_tiles], blk); tx.wait()
                        with tk0_dfb.reserve() as blk:
                            tx = ttl.copy(k_scratch[r, hc:hc + head_tiles], blk); tx.wait()
                        with tk1_dfb.reserve() as blk:
                            tx = ttl.copy(k_scratch[seq_tiles + r, hc:hc + head_tiles], blk); tx.wait()
                        with tv0_dfb.reserve() as blk:
                            tx = ttl.copy(v_scratch[r, hc:hc + head_tiles], blk); tx.wait()
                        with tv1_dfb.reserve() as blk:
                            tx = ttl.copy(v_scratch[seq_tiles + r, hc:hc + head_tiles], blk); tx.wait()

        @ttl.datamovement()
        def dm_write():
            core_x, _ = ttl.core(dims=2)
            for local_h in range(heads_per_core):
                head_idx = core_x * heads_per_core + local_h
                if head_idx < n_heads_val:
                    h = head_idx
                    hc = h * head_tiles
                    # Phase 1: write Q_roped, K_roped, V to scratch
                    for frame in range(2):
                        row_start = frame * seq_tiles
                        with qr_dfb.wait() as blk:
                            tx = ttl.copy(blk, q_scratch[row_start:row_start + seq_tiles, hc:hc + head_tiles]); tx.wait()
                        with kr_dfb.wait() as blk:
                            tx = ttl.copy(blk, k_scratch[row_start:row_start + seq_tiles, hc:hc + head_tiles]); tx.wait()
                        with v_dfb.wait() as blk:
                            tx = ttl.copy(blk, v_scratch[row_start:row_start + seq_tiles, hc:hc + head_tiles]); tx.wait()
                    # Phase 2: write temporal SDPA output
                    for r in range(seq_tiles):
                        # Frame 0 output
                        with out_dfb.wait() as blk:
                            tx = ttl.copy(blk, out[r, hc:hc + head_tiles]); tx.wait()
                        # Frame 1 output
                        with out_dfb.wait() as blk:
                            tx = ttl.copy(blk, out[seq_tiles + r, hc:hc + head_tiles]); tx.wait()

    return temporal_qkv_rope_sdpa

mega_temporal_qkv_rope_sdpa = make_temporal_qkv_rope_sdpa_kernel(
    D_TILES, N_PATCH_PAD_TILES, D_HEAD_TILES, N_HEADS, 0.125)

def make_mega_post_attn_kernel(dim_tiles, mlp_dim_tiles):
    """Mega kernel B: O proj + residual + LN + modulate + FC1 + GELU + FC2 + residual.
    Per-row processing. Uses DRAM scratch between phases."""
    @ttl.kernel(grid="auto")
    def mega_post_attn(attn_out, x_residual, adaln_packed,
                       out_w, out_b,
                       fc1_w, fc1_b, fc2_w, fc2_b,
                       scaler, mean_scale,
                       z_scratch, gelu_scratch, final_out):
        grid_cols, _ = ttl.grid_size(dims=2)
        seq_tiles = attn_out.shape[0] // TILE
        tiles_per_core = -(-seq_tiles // grid_cols)
        attn_dfb = ttl.make_dataflow_buffer_like(attn_out, shape=(1, dim_tiles), buffer_factor=2)
        wcol_dfb = ttl.make_dataflow_buffer_like(out_w, shape=(dim_tiles, 1), buffer_factor=2)
        x_dfb = ttl.make_dataflow_buffer_like(attn_out, shape=(1, 1), buffer_factor=2)
        w_dfb = ttl.make_dataflow_buffer_like(out_w, shape=(1, 1), buffer_factor=2)
        p1_dfb = ttl.make_dataflow_buffer_like(attn_out, shape=(1, 1), buffer_factor=2)
        p2_dfb = ttl.make_dataflow_buffer_like(attn_out, shape=(1, 1), buffer_factor=2)
        p3_dfb = ttl.make_dataflow_buffer_like(attn_out, shape=(1, 1), buffer_factor=2)
        scaler_dfb = ttl.make_dataflow_buffer_like(scaler, shape=(1, 1), buffer_factor=1)
        ms_dfb = ttl.make_dataflow_buffer_like(mean_scale, shape=(1, 1), buffer_factor=1)
        mm_dfb = ttl.make_dataflow_buffer_like(final_out, shape=(1, 1), buffer_factor=2)
        acc_dfb = ttl.make_dataflow_buffer_like(final_out, shape=(1, 1), buffer_factor=2)
        red_dfb = ttl.make_dataflow_buffer_like(scaler, shape=(1, 1), buffer_factor=2)
        bcast_dfb = ttl.make_dataflow_buffer_like(final_out, shape=(1, 1), buffer_factor=2)
        sq_dfb = ttl.make_dataflow_buffer_like(final_out, shape=(1, 1), buffer_factor=2)
        mean_dfb = ttl.make_dataflow_buffer_like(final_out, shape=(1, 1), buffer_factor=2)
        istd_dfb = ttl.make_dataflow_buffer_like(final_out, shape=(1, 1), buffer_factor=2)
        out_dfb = ttl.make_dataflow_buffer_like(final_out, shape=(1, 1), buffer_factor=2)
        @ttl.compute()
        def compute():
            core_x, _ = ttl.core(dims=2)
            with scaler_dfb.wait() as sclr, ms_dfb.wait() as ms:
                for local_t in range(tiles_per_core):
                    tile_idx = core_x * tiles_per_core + local_t
                    if tile_idx < seq_tiles:
                        with attn_dfb.wait() as a_row:
                            for col in range(dim_tiles):
                                with wcol_dfb.wait() as w_col, mm_dfb.reserve() as mm:
                                    mm.store(a_row @ w_col)
                                with mm_dfb.wait() as oproj, p1_dfb.wait() as bv, p2_dfb.wait() as gv, p3_dfb.wait() as rv:
                                    with out_dfb.reserve() as o:
                                        o.store(rv + (oproj + bv) * gv)
                        with x_dfb.wait() as z0:
                            with red_dfb.reserve() as r:
                                r.store(ttl.math.reduce_sum(z0, sclr, dims=[1]))
                        with red_dfb.wait() as rv, acc_dfb.reserve() as a:
                            a.store(rv)
                        for j in range(dim_tiles - 1):
                            with x_dfb.wait() as zj:
                                with red_dfb.reserve() as r:
                                    r.store(ttl.math.reduce_sum(zj, sclr, dims=[1]))
                            with red_dfb.wait() as rv, acc_dfb.wait() as prev, acc_dfb.reserve() as a:
                                a.store(prev + rv)
                        with acc_dfb.wait() as sum_x, bcast_dfb.reserve() as bc:
                            bc.store(ttl.math.broadcast(sum_x, dims=[1]))
                        with bcast_dfb.wait() as sum_bc, mean_dfb.reserve() as mn:
                            mn.store(sum_bc * ms)
                        with mean_dfb.wait() as mean_val:
                            with x_dfb.wait() as z0:
                                with sq_dfb.reserve() as sq:
                                    sq.store((z0 - mean_val) * (z0 - mean_val))
                            with sq_dfb.wait() as sqv, red_dfb.reserve() as r:
                                r.store(ttl.math.reduce_sum(sqv, sclr, dims=[1]))
                            with red_dfb.wait() as rv, acc_dfb.reserve() as a:
                                a.store(rv)
                            for j in range(dim_tiles - 1):
                                with x_dfb.wait() as zj:
                                    with sq_dfb.reserve() as sq:
                                        sq.store((zj - mean_val) * (zj - mean_val))
                                with sq_dfb.wait() as sqv, red_dfb.reserve() as r:
                                    r.store(ttl.math.reduce_sum(sqv, sclr, dims=[1]))
                                with red_dfb.wait() as rv, acc_dfb.wait() as prev, acc_dfb.reserve() as a:
                                    a.store(prev + rv)
                            with acc_dfb.wait() as sum_sq, bcast_dfb.reserve() as bc:
                                bc.store(ttl.math.broadcast(sum_sq, dims=[1]))
                            with bcast_dfb.wait() as var_bc, istd_dfb.reserve() as istd:
                                istd.store(ttl.math.rsqrt(var_bc * ms + ttl.math.fill(var_bc, 1e-6)))
                            with istd_dfb.wait() as inv_std:
                                for j in range(dim_tiles):
                                    with x_dfb.wait() as zj, p1_dfb.wait() as shv, p2_dfb.wait() as sclv:
                                        normed = (zj - mean_val) * inv_std
                                        with out_dfb.reserve() as o:
                                            o.store(normed * (sclv + ttl.math.fill(sclv, 1.0)) + shv)
                        for fc1_col in range(mlp_dim_tiles):
                            with x_dfb.wait() as m0, w_dfb.wait() as fw0, acc_dfb.reserve() as a:
                                a.store(m0 @ fw0)
                            for k in range(dim_tiles - 1):
                                with x_dfb.wait() as mk, w_dfb.wait() as fwk, mm_dfb.reserve() as mm:
                                    mm.store(mk @ fwk)
                                with mm_dfb.wait() as mmv, acc_dfb.wait() as prev, acc_dfb.reserve() as a:
                                    a.store(prev + mmv)
                            with acc_dfb.wait() as fc1r, p1_dfb.wait() as fb:
                                h = fc1r + fb
                                x3 = h * h * h
                                inner = ttl.math.fill(h, 0.7978845608) * (h + ttl.math.fill(h, 0.044715) * x3)
                                with out_dfb.reserve() as o:
                                    o.store(ttl.math.fill(h, 0.5) * h * (ttl.math.fill(h, 1.0) + ttl.math.tanh(inner)))
                        for col in range(dim_tiles):
                            with x_dfb.wait() as g0, w_dfb.wait() as fw0, acc_dfb.reserve() as a:
                                a.store(g0 @ fw0)
                            for k in range(mlp_dim_tiles - 1):
                                with x_dfb.wait() as gk, w_dfb.wait() as fwk, mm_dfb.reserve() as mm:
                                    mm.store(gk @ fwk)
                                with mm_dfb.wait() as mmv, acc_dfb.wait() as prev, acc_dfb.reserve() as a:
                                    a.store(prev + mmv)
                            with acc_dfb.wait() as fc2r, p1_dfb.wait() as fb, p2_dfb.wait() as gv, p3_dfb.wait() as zv:
                                with out_dfb.reserve() as o:
                                    o.store(zv + (fc2r + fb) * gv)
        @ttl.datamovement()
        def dm_read():
            core_x, _ = ttl.core(dims=2)
            with scaler_dfb.reserve() as blk:
                tx = ttl.copy(scaler[0, 0], blk); tx.wait()
            with ms_dfb.reserve() as blk:
                tx = ttl.copy(mean_scale[0, 0], blk); tx.wait()
            for local_t in range(tiles_per_core):
                tile_idx = core_x * tiles_per_core + local_t
                if tile_idx < seq_tiles:
                    with attn_dfb.reserve() as blk:
                        tx = ttl.copy(attn_out[tile_idx, 0:dim_tiles], blk); tx.wait()
                    for col in range(dim_tiles):
                        with wcol_dfb.reserve() as blk:
                            tx = ttl.copy(out_w[0:dim_tiles, col], blk); tx.wait()
                        with p1_dfb.reserve() as blk:
                            tx = ttl.copy(out_b[tile_idx, col], blk); tx.wait()
                        with p2_dfb.reserve() as blk:
                            tx = ttl.copy(adaln_packed[tile_idx, 2 * dim_tiles + col], blk); tx.wait()
                        with p3_dfb.reserve() as blk:
                            tx = ttl.copy(x_residual[tile_idx, col], blk); tx.wait()
                    for j in range(dim_tiles):
                        with x_dfb.reserve() as blk:
                            tx = ttl.copy(z_scratch[tile_idx, j], blk); tx.wait()
                    for j in range(dim_tiles):
                        with x_dfb.reserve() as blk:
                            tx = ttl.copy(z_scratch[tile_idx, j], blk); tx.wait()
                    for j in range(dim_tiles):
                        with x_dfb.reserve() as blk:
                            tx = ttl.copy(z_scratch[tile_idx, j], blk); tx.wait()
                        with p1_dfb.reserve() as blk:
                            tx = ttl.copy(adaln_packed[tile_idx, 3 * dim_tiles + j], blk); tx.wait()
                        with p2_dfb.reserve() as blk:
                            tx = ttl.copy(adaln_packed[tile_idx, 4 * dim_tiles + j], blk); tx.wait()
                    for fc1_col in range(mlp_dim_tiles):
                        for k in range(dim_tiles):
                            with x_dfb.reserve() as blk:
                                tx = ttl.copy(final_out[tile_idx, k], blk); tx.wait()
                            with w_dfb.reserve() as blk:
                                tx = ttl.copy(fc1_w[k, fc1_col], blk); tx.wait()
                        with p1_dfb.reserve() as blk:
                            tx = ttl.copy(fc1_b[tile_idx, fc1_col], blk); tx.wait()
                    for col in range(dim_tiles):
                        for k in range(mlp_dim_tiles):
                            with x_dfb.reserve() as blk:
                                tx = ttl.copy(gelu_scratch[tile_idx, k], blk); tx.wait()
                            with w_dfb.reserve() as blk:
                                tx = ttl.copy(fc2_w[k, col], blk); tx.wait()
                        with p1_dfb.reserve() as blk:
                            tx = ttl.copy(fc2_b[tile_idx, col], blk); tx.wait()
                        with p2_dfb.reserve() as blk:
                            tx = ttl.copy(adaln_packed[tile_idx, 5 * dim_tiles + col], blk); tx.wait()
                        with p3_dfb.reserve() as blk:
                            tx = ttl.copy(z_scratch[tile_idx, col], blk); tx.wait()
        @ttl.datamovement()
        def dm_write():
            core_x, _ = ttl.core(dims=2)
            for local_t in range(tiles_per_core):
                tile_idx = core_x * tiles_per_core + local_t
                if tile_idx < seq_tiles:
                    for col in range(dim_tiles):
                        with out_dfb.wait() as blk:
                            tx = ttl.copy(blk, z_scratch[tile_idx, col]); tx.wait()
                    for col in range(dim_tiles):
                        with out_dfb.wait() as blk:
                            tx = ttl.copy(blk, final_out[tile_idx, col]); tx.wait()
                    for fc1_col in range(mlp_dim_tiles):
                        with out_dfb.wait() as blk:
                            tx = ttl.copy(blk, gelu_scratch[tile_idx, fc1_col]); tx.wait()
                    for col in range(dim_tiles):
                        with out_dfb.wait() as blk:
                            tx = ttl.copy(blk, final_out[tile_idx, col]); tx.wait()
    return mega_post_attn

mega_post_attn_kernel = make_mega_post_attn_kernel(D_TILES, D_MLP_TILES)

def make_fused_ln_adaln_kernel(dim_tiles):
    """Fused LayerNorm + adaLN modulate: out = layernorm(x) * (1 + scale) + shift.
    adaln_packed: (SEQ, 6*D_MODEL) with shift at cols [0:D], scale at cols [D:2D].
    Eliminates DRAM round-trip for the intermediate normed tensor."""
    @ttl.kernel(grid="auto")
    def fused_ln_adaln(x, scaler, mean_scale, adaln_packed, out):
        grid_cols, _ = ttl.grid_size(dims=2)
        seq_tiles = x.shape[0] // TILE
        tiles_per_core = -(-seq_tiles // grid_cols)
        x_dfb = ttl.make_dataflow_buffer_like(x, shape=(1, 1), buffer_factor=2)
        sc_dfb = ttl.make_dataflow_buffer_like(scaler, shape=(1, 1), buffer_factor=1)
        ms_dfb = ttl.make_dataflow_buffer_like(mean_scale, shape=(1, 1), buffer_factor=1)
        sh_dfb = ttl.make_dataflow_buffer_like(adaln_packed, shape=(1, 1), buffer_factor=2)
        scl_dfb = ttl.make_dataflow_buffer_like(adaln_packed, shape=(1, 1), buffer_factor=2)
        red_dfb = ttl.make_dataflow_buffer_like(scaler, shape=(1, 1), buffer_factor=2)
        acc_dfb = ttl.make_dataflow_buffer_like(scaler, shape=(1, 1), buffer_factor=2)
        bcast_dfb = ttl.make_dataflow_buffer_like(x, shape=(1, 1), buffer_factor=2)
        sq_dfb = ttl.make_dataflow_buffer_like(x, shape=(1, 1), buffer_factor=2)
        mean_dfb = ttl.make_dataflow_buffer_like(x, shape=(1, 1), buffer_factor=2)
        istd_dfb = ttl.make_dataflow_buffer_like(x, shape=(1, 1), buffer_factor=2)
        out_dfb = ttl.make_dataflow_buffer_like(out, shape=(1, 1), buffer_factor=2)
        @ttl.compute()
        def compute():
            core_x, _ = ttl.core(dims=2)
            with sc_dfb.wait() as sc, ms_dfb.wait() as ms:
                for local_t in range(tiles_per_core):
                    tile_idx = core_x * tiles_per_core + local_t
                    if tile_idx < seq_tiles:
                        # Pass 1: compute mean
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
                        # Pass 2: compute variance
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
                            # Pass 3: normalize + adaLN modulate (fused)
                            with istd_dfb.wait() as inv_std:
                                for j in range(dim_tiles):
                                    with x_dfb.wait() as xj, sh_dfb.wait() as shv, scl_dfb.wait() as sclv, out_dfb.reserve() as o:
                                        normed = (xj - mean_val) * inv_std
                                        o.store(normed * (sclv + ttl.math.fill(sclv, 1.0)) + shv)
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
                    # Pass 1: mean
                    for j in range(dim_tiles):
                        with x_dfb.reserve() as blk:
                            tx = ttl.copy(x[tile_idx, j], blk); tx.wait()
                    # Pass 2: variance
                    for j in range(dim_tiles):
                        with x_dfb.reserve() as blk:
                            tx = ttl.copy(x[tile_idx, j], blk); tx.wait()
                    # Pass 3: normalize + modulate (shift at cols 0:D, scale at cols D:2D)
                    for j in range(dim_tiles):
                        with x_dfb.reserve() as blk:
                            tx = ttl.copy(x[tile_idx, j], blk); tx.wait()
                        with sh_dfb.reserve() as blk:
                            tx = ttl.copy(adaln_packed[tile_idx, j], blk); tx.wait()
                        with scl_dfb.reserve() as blk:
                            tx = ttl.copy(adaln_packed[tile_idx, dim_tiles + j], blk); tx.wait()
        @ttl.datamovement()
        def dm_write():
            core_x, _ = ttl.core(dims=2)
            for local_t in range(tiles_per_core):
                tile_idx = core_x * tiles_per_core + local_t
                if tile_idx < seq_tiles:
                    for j in range(dim_tiles):
                        with out_dfb.wait() as blk:
                            tx = ttl.copy(blk, out[tile_idx, j]); tx.wait()
    return fused_ln_adaln

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
fused_lbgr_k32 = make_fused_linear_bias_gated_res_kernel(D_TILES)  # O proj + bias + gated residual
fused_lbg_k32 = make_fused_linear_bias_gelu_kernel(D_TILES, n_chunk=4)  # FC1 + bias + GELU
fused_ln_adaln_d1024 = make_fused_ln_adaln_kernel(D_TILES)  # LayerNorm + adaLN modulate
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

def build_rope_device_tables(freqs_per_position, n_positions, n_patch_pad, n_heads, n_frames, tt_device):
    """Build cos/sin_perm device tables from per-position freq table.
    freqs_per_position: (n_positions, freq_dim) float tensor.
    Returns cos_tt, sin_perm_tt each (n_patch_pad * n_frames, n_heads * freq_dim) on device.
    """
    freq_dim = freqs_per_position.shape[-1]
    d_model = n_heads * freq_dim
    SEQ = n_patch_pad * n_frames

    cos_vals = freqs_per_position.cos()  # (n_positions, freq_dim)
    sin_vals = freqs_per_position.sin()
    sign = torch.ones(freq_dim)
    sign[0::2] = -1
    sin_perm_vals = sin_vals * sign.unsqueeze(0)

    cos_expanded = cos_vals.repeat(1, n_heads)  # (n_positions, d_model)
    sin_expanded = sin_perm_vals.repeat(1, n_heads)

    cos_full = torch.zeros(SEQ, d_model, dtype=torch.bfloat16)
    sin_full = torch.zeros(SEQ, d_model, dtype=torch.bfloat16)
    for t in range(n_frames):
        start = t * n_patch_pad
        n = min(n_positions, n_patch_pad)
        cos_full[start:start + n] = cos_expanded[:n].to(torch.bfloat16)
        sin_full[start:start + n] = sin_expanded[:n].to(torch.bfloat16)

    return to_tt(cos_full, tt_device), to_tt(sin_full, tt_device)

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

                # Combined QKV + QK_swap: (1024, 5120) via single ttnn.matmul
                # Layout: [Q | K | V | Q_swap | K_swap] each 1024 cols
                qkv_w = st.get_tensor("%s_attn.to_qkv.weight" % p).T.contiguous().to(torch.bfloat16)
                q_w = qkv_w[:, :D_MODEL]
                k_w = qkv_w[:, D_MODEL:2*D_MODEL]
                v_w = qkv_w[:, 2*D_MODEL:]
                qkv_full_w = torch.cat([q_w, k_w, v_w,
                                        swap_adjacent_columns(q_w),
                                        swap_adjacent_columns(k_w)], dim=1)
                dev["%s.qkv_full_w" % p] = to_tt(qkv_full_w, tt_device)
                # Separate QK_swap weights still needed for temporal path
                qk_swap_w = torch.cat([swap_adjacent_columns(q_w),
                                       swap_adjacent_columns(k_w)], dim=1)
                dev["%s.qk_swap_w" % p] = to_tt(qk_swap_w, tt_device)
                dev["%s.qkv_w" % p] = to_tt(qkv_w, tt_device)

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
                # 1D bias for ttnn.linear
                dev["%s.fc2_b_1d" % p] = to_tt(fc2_b.unsqueeze(0).contiguous(), tt_device)

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

        # Device-side temporal RoPE tables: per-frame freqs broadcast to all patches
        # For T frames: frame t uses temporal_freqs[t], broadcast across N_PATCH_PAD rows
        temporal_per_frame = TEMPORAL_ROPE_FREQS[:n_frames]  # (T, 64)
        # Expand each frame's freqs to fill N_PATCH_PAD rows
        temporal_expanded = temporal_per_frame.unsqueeze(1).expand(
            n_frames, N_PATCH_PAD, TEMPORAL_ROPE_DIM).reshape(
            n_frames * N_PATCH_PAD, TEMPORAL_ROPE_DIM)  # (SEQ, 64)
        dev["temporal_cos"], dev["temporal_sin_perm"] = \
            build_rope_device_tables(temporal_expanded, n_frames * N_PATCH_PAD,
                                     N_PATCH_PAD, N_HEADS, 1, tt_device)

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
    s["qkv_full"] = zeros_tt((SEQ, 5 * D_MODEL), tt_device)
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
    # Packed adaln: (SEQ, 6*D_MODEL) reused across blocks
    s["adaln_packed"] = zeros_tt((SEQ, 6 * D_MODEL), tt_device)
    # Per-frame adaln expanded: (N_PATCH_PAD, 6*D_MODEL) for building packed tensor
    for f in range(n_frames):
        s["adaln_frame_%d" % f] = zeros_tt((N_PATCH_PAD, 6 * D_MODEL), tt_device)
    # RoPE scratch (reused for spatial and temporal)
    s["q_roped"] = zeros_tt((SEQ, D_MODEL), tt_device)
    s["k_roped"] = zeros_tt((SEQ, D_MODEL), tt_device)
    # Temporal mega kernel scratch (Q/K roped + V)
    s["t_q_scratch"] = zeros_tt((SEQ, D_MODEL), tt_device)
    s["t_k_scratch"] = zeros_tt((SEQ, D_MODEL), tt_device)
    s["t_v_scratch"] = zeros_tt((SEQ, D_MODEL), tt_device)
    # SDPA scratch: (SEQ, D_MODEL) for TT-Lang spatial SDPA output
    s["sdpa_out"] = zeros_tt((SEQ, D_MODEL), tt_device)
    # Mega kernel B scratch
    s["z_scratch"] = zeros_tt((SEQ, D_MODEL), tt_device)
    s["gelu_scratch"] = zeros_tt((SEQ, D_MLP), tt_device)
    # Final layer
    s["final_adaln"] = zeros_tt((TILE, 2 * D_MODEL), tt_device)
    s["final_out"] = zeros_tt((SEQ, OUT_DIM), tt_device)
    s["n_frames"] = n_frames
    # DDIM arithmetic scratch (N_PATCH_PAD, OUT_DIM)
    s["ddim_x_start"] = zeros_tt((N_PATCH_PAD, OUT_DIM), tt_device)
    s["ddim_x_noise"] = zeros_tt((N_PATCH_PAD, OUT_DIM), tt_device)
    s["ddim_tmp"] = zeros_tt((N_PATCH_PAD, OUT_DIM), tt_device)
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

def patchify_to_output_space(x_img):
    """Inverse of unpatchify: (1, C, H, W) -> (N_PATCHES, OUT_DIM).
    Matches the pixel ordering that unpatchify_host expects."""
    x = x_img.float().reshape(1, IN_CHANNELS, FRAME_H, PATCH_SIZE, FRAME_W, PATCH_SIZE)
    x = torch.einsum("nchpwq->nhwpqc", x)
    return x.reshape(N_PATCHES, OUT_DIM)

def unpatchify_host(x, patch_size, out_channels, h, w):
    """x: (N_PATCHES, patch_size^2 * out_channels) -> (1, out_channels, H, W)"""
    x = x.float().reshape(1, h, w, patch_size, patch_size, out_channels)
    x = torch.einsum("nhwpqc->nchpwq", x)
    return x.reshape(1, out_channels, h * patch_size, w * patch_size)

def build_per_frame_adaln(silu_cond_list, prefix, dev, scr, tt_device):
    """Compute adaLN params on DEVICE using ttnn.matmul.
    silu_cond_list: list of T pre-computed SiLU(cond) device tensors, each (TILE, D_MODEL).
    Returns: packed (SEQ, 6*D_MODEL) device tensor.
    """
    T = len(silu_cond_list)
    N_REPEAT = N_PATCH_PAD // TILE  # 5

    per_frame_expanded = []
    for silu_cond in silu_cond_list:
        # Linear(1024, 6144) -> reuse scratch for matmul+add
        ttnn.matmul(silu_cond, dev["%s.adaln_w" % prefix],
                    optional_output_tensor=scr["adaln_out"])
        ttnn.add(scr["adaln_out"], dev["%s.adaln_b" % prefix],
                 output_tensor=scr["adaln_out"])
        # Expand to (N_PATCH_PAD, 6*D_MODEL) - concat allocates but matmul/add reuse scratch
        expanded = ttnn.concat([scr["adaln_out"]] * N_REPEAT, dim=0)
        per_frame_expanded.append(expanded)

    # Combine frames: (SEQ, 6*D_MODEL)
    if T == 1:
        return per_frame_expanded[0]
    else:
        return ttnn.concat(per_frame_expanded, dim=0)

PROFILE_BLOCKS = False
PROFILE_BLOCK_DEVICE = "blocks.1.s"  # sync-profile this specific block

def run_sub_block(prefix, x_tt, silu_cond_list, dev, scr, tt_device, scaler, mean_scale, attn_type="spatial"):
    """Run one spatial or temporal sub-block.
    prefix: e.g. "blocks.0.s" or "blocks.0.t"
    silu_cond_list: list of T pre-computed SiLU(cond) device tensors, each (TILE, D_MODEL)
    attn_type: "spatial" or "temporal"
    x_tt: (N_PATCH_PAD * T, D_MODEL) device tensor
    """
    T = len(silu_cond_list)
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

    _do_dev_profile = (prefix == PROFILE_BLOCK_DEVICE)
    if _do_dev_profile:
        import time as _dt
        ttnn.synchronize_device(tt_device)
        _t0 = _dt.time()

    # Compute per-frame adaLN params (SiLU already applied to cond) - packed (SEQ, 6*D_MODEL)
    adaln_packed = build_per_frame_adaln(silu_cond_list, prefix, dev, scr, tt_device)
    _timer.mark("adaln")
    if _do_dev_profile:
        ttnn.synchronize_device(tt_device)
        _t1 = _dt.time(); print("      %s adaln: %.1fms" % (prefix, (_t1 - _t0) * 1000)); _t0 = _t1

    # Fused LayerNorm + adaLN modulate (reads shift/scale from packed tensor)
    fused_ln_adaln_d1024(x_tt, scaler, mean_scale, adaln_packed, scr["modulated"])
    _timer.mark("norm+mod")
    if _do_dev_profile:
        ttnn.synchronize_device(tt_device)
        _t1 = _dt.time(); print("      %s norm+mod: %.1fms" % (prefix, (_t1 - _t0) * 1000)); _t0 = _t1

    # Hybrid: ttnn.matmul for QKV (72-core parallel), TT-Lang for RoPE+SDPA
    # NOTE: mega_qkv_rope_sdpa fuses QKV+RoPE+SDPA into one kernel but only uses
    # 32 cores for K-accumulation matmul. ttnn.matmul uses all 72 cores (30x faster).
    # To restore fused version: mega_qkv_rope_sdpa(scr["modulated"], qkv_w, cos, sin, scaler, out)
    ttnn.matmul(scr["modulated"], dev["%s.qkv_full_w" % prefix],
                optional_output_tensor=scr["qkv_full"])
    if _do_dev_profile:
        ttnn.synchronize_device(tt_device)
        _t1 = _dt.time(); print("      %s qkv_matmul: %.1fms" % (prefix, (_t1 - _t0) * 1000)); _t0 = _t1

    # Slice Q/K/V/Q_swap/K_swap, apply RoPE, then ttnn SDPA (f32 accumulation)
    # TODO: investigate bf16 accumulation in rope_sdpa_kernel (TT-Lang) - it causes
    # quality degradation over 320 attention passes. ttnn SDPA uses f32 accumulation.
    # Replaced: rope_sdpa_kernel(qkv_full, cos, sin, scaler, out)
    q = ttnn.slice(scr["qkv_full"], [0, 0], [SEQ, D_MODEL])
    k = ttnn.slice(scr["qkv_full"], [0, D_MODEL], [SEQ, 2 * D_MODEL])
    v = ttnn.slice(scr["qkv_full"], [0, 2 * D_MODEL], [SEQ, 3 * D_MODEL])
    q_swap = ttnn.slice(scr["qkv_full"], [0, 3 * D_MODEL], [SEQ, 4 * D_MODEL])
    k_swap = ttnn.slice(scr["qkv_full"], [0, 4 * D_MODEL], [SEQ, 5 * D_MODEL])

    if attn_type == "spatial":
        cos_key, sin_key = "spatial_cos", "spatial_sin_perm"
    else:
        cos_key, sin_key = "temporal_cos", "temporal_sin_perm"
    fused_rope_kernel(q, q_swap, dev[cos_key], dev[sin_key], scr["q_roped"])
    fused_rope_kernel(k, k_swap, dev[cos_key], dev[sin_key], scr["k_roped"])

    if attn_type == "spatial":
        # Spatial: each frame attends within itself, batch over (frame, head)
        BATCH_S = T * N_HEADS
        q_s = ttnn.reshape(scr["q_roped"], [T, N_PATCH_PAD, N_HEADS, D_HEAD])
        q_s = ttnn.permute(q_s, [0, 2, 1, 3])
        q_s = ttnn.reshape(q_s, [BATCH_S, 1, N_PATCH_PAD, D_HEAD])
        k_s = ttnn.reshape(scr["k_roped"], [T, N_PATCH_PAD, N_HEADS, D_HEAD])
        k_s = ttnn.permute(k_s, [0, 2, 1, 3])
        k_s = ttnn.reshape(k_s, [BATCH_S, 1, N_PATCH_PAD, D_HEAD])
        v_s = ttnn.reshape(v, [T, N_PATCH_PAD, N_HEADS, D_HEAD])
        v_s = ttnn.permute(v_s, [0, 2, 1, 3])
        v_s = ttnn.reshape(v_s, [BATCH_S, 1, N_PATCH_PAD, D_HEAD])
        attn_out = ttnn.transformer.scaled_dot_product_attention(q_s, k_s, v_s, is_causal=False)
        attn_out = ttnn.reshape(attn_out, [T, N_HEADS, N_PATCH_PAD, D_HEAD])
        attn_out = ttnn.permute(attn_out, [0, 2, 1, 3])
        attn_2d = ttnn.reshape(attn_out, [SEQ, D_MODEL])
        _timer.mark("qkv+sdpa")
    else:
        # Temporal: each patch attends across frames, batch over (patch, head)
        BATCH_T = N_PATCH_PAD * N_HEADS
        q_t = ttnn.reshape(scr["q_roped"], [T, BATCH_T, D_HEAD])
        q_t = ttnn.permute(q_t, [1, 0, 2])
        q_t = ttnn.reshape(q_t, [BATCH_T, 1, T, D_HEAD])
        k_t = ttnn.reshape(scr["k_roped"], [T, BATCH_T, D_HEAD])
        k_t = ttnn.permute(k_t, [1, 0, 2])
        k_t = ttnn.reshape(k_t, [BATCH_T, 1, T, D_HEAD])
        v_t = ttnn.reshape(v, [T, BATCH_T, D_HEAD])
        v_t = ttnn.permute(v_t, [1, 0, 2])
        v_t = ttnn.reshape(v_t, [BATCH_T, 1, T, D_HEAD])
        attn_out = ttnn.transformer.scaled_dot_product_attention(q_t, k_t, v_t, is_causal=True)
        attn_out = ttnn.reshape(attn_out, [BATCH_T, T, D_HEAD])
        attn_out = ttnn.permute(attn_out, [1, 0, 2])
        attn_2d = ttnn.reshape(attn_out, [SEQ, D_MODEL])
        _timer.mark("sdpa")
    if _do_dev_profile:
        ttnn.synchronize_device(tt_device)
        _t1 = _dt.time(); print("      %s rope+sdpa: %.1fms" % (prefix, (_t1 - _t0) * 1000)); _t0 = _t1

    # Separate ops: ttnn.matmul for large matmuls, TT-Lang for fused elementwise
    # NOTE: mega_post_attn_kernel fuses O proj+LN+FC1+GELU+FC2+residual but only uses
    # 10 cores (per-row). ttnn.matmul on 72 cores is 7x faster for the matmuls.
    # To restore fused version: mega_post_attn_kernel(attn_2d, x_tt, adaln_packed,
    #   out_w, out_b, fc1_w, fc1_b, fc2_w, fc2_b, scaler, mean_scale,
    #   z_scratch, gelu_scratch, z_a)
    # Phase A: O proj + bias + gated residual
    ttnn.matmul(attn_2d, dev["%s.out_w" % prefix], optional_output_tensor=scr["o_proj"])
    ttnn.add(scr["o_proj"], dev["%s.out_b" % prefix], output_tensor=scr["o_proj"])
    gate_msa = ttnn.slice(adaln_packed, [0, 2 * D_MODEL], [SEQ, 3 * D_MODEL])
    gated_residual_kernel(x_tt, scr["o_proj"], gate_msa, scr["z_scratch"])
    if _do_dev_profile:
        ttnn.synchronize_device(tt_device)
        _t1 = _dt.time(); print("      %s o_proj+res: %.1fms" % (prefix, (_t1 - _t0) * 1000)); _t0 = _t1

    # Phase B: LN + modulate
    shift_mlp = ttnn.slice(adaln_packed, [0, 3 * D_MODEL], [SEQ, 4 * D_MODEL])
    scale_mlp = ttnn.slice(adaln_packed, [0, 4 * D_MODEL], [SEQ, 5 * D_MODEL])
    layernorm_d1024(scr["z_scratch"], dev["ln_w_ones"], dev["ln_b_zeros"], scaler, mean_scale, scr["normed"])
    adaln_modulate_kernel(scr["normed"], shift_mlp, scale_mlp, scr["modulated"])
    if _do_dev_profile:
        ttnn.synchronize_device(tt_device)
        _t1 = _dt.time(); print("      %s ln+mod: %.1fms" % (prefix, (_t1 - _t0) * 1000)); _t0 = _t1

    # Phase C: FC1 + bias + GELU
    ttnn.matmul(scr["modulated"], dev["%s.fc1_w" % prefix], optional_output_tensor=scr["fc1"])
    ttnn.add(scr["fc1"], dev["%s.fc1_b" % prefix], output_tensor=scr["fc1"])
    ttnn.gelu(scr["fc1"], output_tensor=scr["gelu"])
    if _do_dev_profile:
        ttnn.synchronize_device(tt_device)
        _t1 = _dt.time(); print("      %s fc1+gelu: %.1fms" % (prefix, (_t1 - _t0) * 1000)); _t0 = _t1

    # Phase D: FC2 + bias + gated residual
    ttnn.matmul(scr["gelu"], dev["%s.fc2_w" % prefix], optional_output_tensor=scr["fc2"])
    ttnn.add(scr["fc2"], dev["%s.fc2_b" % prefix], output_tensor=scr["fc2"])
    gate_mlp = ttnn.slice(adaln_packed, [0, 5 * D_MODEL], [SEQ, 6 * D_MODEL])
    gated_residual_kernel(scr["z_scratch"], scr["fc2"], gate_mlp, scr["z_a"])
    _timer.mark("post_attn")
    if _do_dev_profile:
        ttnn.synchronize_device(tt_device)
        _t1 = _dt.time(); print("      %s fc2+res: %.1fms" % (prefix, (_t1 - _t0) * 1000)); _t0 = _t1

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

def dit_forward_device(z_cur, cond_list, dev, scr, tt_device, scaler, mean_scale, profile_step=False):
    """DiT forward pass, all on device. Returns final_out as device tensor.
    z_cur: (SEQ, D_MODEL) device tensor
    cond_list: list of T device tensors for conditioning
    Returns: device tensor (SEQ, OUT_DIM)
    """
    T = len(cond_list)
    SEQ = N_PATCH_PAD * T

    if profile_step:
        import time as _t
        _pt = _t.time
        ttnn.synchronize_device(tt_device)

    # Pre-compute SiLU(cond) once per step, reuse across all 32 sub-blocks + final layer
    silu_cond_list = []
    for t_idx in range(T):
        silu_out = zeros_tt((TILE, D_MODEL), tt_device)
        silu_kernel(cond_list[t_idx], silu_out)
        silu_cond_list.append(silu_out)

    if profile_step:
        ttnn.synchronize_device(tt_device)
        print("  [PROFILE] silu: %.1fms" % ((_pt() - _pt.__self__) if False else 0))

    # 16 blocks: each has spatial + temporal sub-block
    if profile_step:
        t_blocks = _pt()
    for block_idx in range(N_BLOCKS):
        z_cur = run_sub_block(
            "blocks.%d.s" % block_idx, z_cur, silu_cond_list,
            dev, scr, tt_device, scaler, mean_scale, attn_type="spatial"
        )
        if profile_step:
            ttnn.synchronize_device(tt_device)
            t1 = _pt()
            print("      block %d.s: %.1fms" % (block_idx, (t1 - t_blocks) * 1000))
            t_blocks = t1
        z_cur = run_sub_block(
            "blocks.%d.t" % block_idx, z_cur, silu_cond_list,
            dev, scr, tt_device, scaler, mean_scale, attn_type="temporal"
        )
        if profile_step:
            ttnn.synchronize_device(tt_device)
            t1 = _pt()
            print("      block %d.t: %.1fms" % (block_idx, (t1 - t_blocks) * 1000))
            t_blocks = t1

    if profile_step:
        print("      === final layer ===")
        t_blocks = _pt()
    # Final layer: per-frame adaLN (SiLU already computed)
    N_REPEAT = N_PATCH_PAD // TILE  # 5
    per_frame_final = []
    for t_idx in range(T):
        final_raw = ttnn.matmul(silu_cond_list[t_idx], dev["final_adaln_w"])
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
    result = ttnn.add(scr["final_out"], dev["final_linear_b"])
    if profile_step:
        ttnn.synchronize_device(tt_device)
        print("      final_layer: %.1fms" % ((_pt() - t_blocks) * 1000))
    return result

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
    alphas_cumprod = torch.cumprod(alphas, dim=0)  # (1000,)

    # Actions: no-op for both frames
    actions = torch.zeros(1, N_FRAMES, EXT_COND_DIM)

    # === Pre-compute device-resident data for on-device DDIM loop ===

    # 1. Prompt frame: patch embed once, keep on device permanently
    prompt_z_pad = torch.zeros(N_PATCH_PAD, D_MODEL, dtype=torch.bfloat16)
    prompt_z_pad[:N_PATCHES] = patch_embed_host(
        prompt_latent[0, 0].unsqueeze(0), dev["x_emb_conv_w"], dev["x_emb_conv_b"])
    prompt_z_dev = to_tt(prompt_z_pad, tt_device)

    # 2. Prompt conditioning (constant across all DDIM steps)
    prompt_cond_dev = compute_cond_for_frame(
        stabilization_level - 1, actions[0, 0], dev, scr, tt_device)

    # 3. Round-trip weight: output_space -> input_space (patch_embed composed with unpatchify)
    # Conv weight (D_MODEL, C, ps, ps) reordered to match patchify pixel order (p, q, c)
    conv_w = dev["x_emb_conv_w"].float()  # (1024, 16, 2, 2)
    W_rt = conv_w.permute(0, 2, 3, 1).reshape(D_MODEL, OUT_DIM).T.contiguous()  # (64, 1024)
    W_rt_dev = to_tt(W_rt.to(torch.bfloat16), tt_device)
    # Bias: expand to (N_PATCH_PAD, D_MODEL) for broadcast add
    b_rt = dev["x_emb_conv_b"].float().to(torch.bfloat16)  # (D_MODEL,)
    b_rt_pad = b_rt.unsqueeze(0).expand(N_PATCH_PAD, -1).contiguous()
    b_rt_dev = to_tt(b_rt_pad, tt_device)

    # 4. Initial chunk: random noise in output space on device
    chunk_img = torch.randn(1, IN_CHANNELS, INPUT_H, INPUT_W)
    chunk_img = torch.clamp(chunk_img, -noise_abs_max, noise_abs_max)
    chunk_patches = patchify_to_output_space(chunk_img)  # (144, 64)
    chunk_pad = torch.zeros(N_PATCH_PAD, OUT_DIM, dtype=torch.bfloat16)
    chunk_pad[:N_PATCHES] = chunk_patches.to(torch.bfloat16)
    chunk_dev = to_tt(chunk_pad, tt_device)

    # Pre-compute conditioning for ALL DDIM steps (noise levels known ahead of time)
    print("Pre-computing conditioning for %d steps..." % ddim_steps)
    t_precond = time.time()
    gen_cond_per_step = {}
    for noise_idx in reversed(range(1, ddim_steps + 1)):
        noise_level = int(noise_range[noise_idx].item())
        gen_cond_per_step[noise_idx] = compute_cond_for_frame(
            noise_level, actions[0, 1], dev, scr, tt_device)
    print("Pre-computed conditioning in %.1fs" % (time.time() - t_precond))

    # Pre-compute DDIM scalar coefficients as device tensors
    ddim_coeffs = {}
    for noise_idx in reversed(range(1, ddim_steps + 1)):
        noise_level = int(noise_range[noise_idx].item())
        t_next_noise = int(noise_range[noise_idx - 1].item())
        at = float(alphas_cumprod[noise_level].item())
        an = float(alphas_cumprod[max(t_next_noise, 0)].item())
        if noise_idx == 1:
            an = 1.0
        ddim_coeffs[noise_idx] = {
            "at_sqrt": float(at ** 0.5),
            "1mat_sqrt": float((1 - at) ** 0.5),
            "inv_at_sqrt": float((1.0 / at) ** 0.5),
            "inv_sigma": float(1.0 / ((1.0 / at - 1) ** 0.5)),
            "an_sqrt": float(an ** 0.5),
            "1man_sqrt": float((1 - an) ** 0.5),
        }

    def run_ddim_loop(chunk_in, label, profile=False):
        """Run full DDIM loop. Returns final chunk_dev."""
        chunk = chunk_in
        t_total = time.time()
        for noise_idx in reversed(range(1, ddim_steps + 1)):
            t_step = time.time()
            step_num = ddim_steps - noise_idx + 1

            gen_z = ttnn.matmul(chunk, W_rt_dev)
            ttnn.add(gen_z, b_rt_dev, output_tensor=gen_z)
            z_cur = ttnn.concat([prompt_z_dev, gen_z], dim=0)
            cond_list = [prompt_cond_dev, gen_cond_per_step[noise_idx]]

            if profile:
                ttnn.synchronize_device(tt_device)
                t_dit = time.time()
                print("    step %d pre-dit: %.1fms" % (step_num, (t_dit - t_step) * 1000))

            final_out = dit_forward_device(z_cur, cond_list, dev, scr, tt_device, scaler, mean_scale,
                                            profile_step=(profile and step_num == 1))

            if profile:
                ttnn.synchronize_device(tt_device)
                t_post = time.time()
                print("    step %d dit_forward: %.1fms" % (step_num, (t_post - t_dit) * 1000))

            v_dev = ttnn.slice(final_out, [N_PATCH_PAD, 0], [2 * N_PATCH_PAD, OUT_DIM])
            coeffs = ddim_coeffs[noise_idx]
            # x_start = at_sqrt * chunk - (1-at)_sqrt * v
            ttnn.multiply(chunk, coeffs["at_sqrt"], output_tensor=scr["ddim_tmp"])
            ttnn.multiply(v_dev, coeffs["1mat_sqrt"], output_tensor=scr["ddim_x_start"])
            ttnn.subtract(scr["ddim_tmp"], scr["ddim_x_start"], output_tensor=scr["ddim_x_start"])
            # x_noise = (inv_at_sqrt * chunk - x_start) / sigma
            ttnn.multiply(chunk, coeffs["inv_at_sqrt"], output_tensor=scr["ddim_tmp"])
            ttnn.subtract(scr["ddim_tmp"], scr["ddim_x_start"], output_tensor=scr["ddim_x_noise"])
            ttnn.multiply(scr["ddim_x_noise"], coeffs["inv_sigma"], output_tensor=scr["ddim_x_noise"])
            # chunk = an_sqrt * x_start + (1-an)_sqrt * x_noise
            ttnn.multiply(scr["ddim_x_start"], coeffs["an_sqrt"], output_tensor=scr["ddim_tmp"])
            ttnn.multiply(scr["ddim_x_noise"], coeffs["1man_sqrt"], output_tensor=scr["ddim_x_noise"])
            ttnn.add(scr["ddim_tmp"], scr["ddim_x_noise"], output_tensor=chunk)

            if profile:
                ttnn.synchronize_device(tt_device)
                t_end = time.time()
                print("    step %d ddim_arith: %.1fms" % (step_num, (t_end - t_post) * 1000))
                print("    step %d TOTAL: %.1fms" % (step_num, (t_end - t_step) * 1000))

        ttnn.synchronize_device(tt_device)
        total_time = time.time() - t_total
        print("%s: %.3fs total (%.0fms/step, %.2f FPS)" % (
            label, total_time, total_time * 1000 / ddim_steps, ddim_steps / total_time))
        return chunk

    # Warmup frame (compile all kernels)
    print("\n=== WARMUP (compiling kernels) ===")
    _ = run_ddim_loop(chunk_dev, "Warmup")

    # === Generate 30-frame video ===
    # Following reference: autoregressive in latent space. Generated frame's latent
    # feeds directly as prompt for next frame (no VAE decode/re-encode between frames).
    # VAE decode happens once at the end for all frames.
    N_VIDEO_FRAMES = 30
    print("\n=== GENERATING %d-FRAME VIDEO ===" % N_VIDEO_FRAMES)
    t_video_start = time.time()

    # Collect all latents for batch VAE decode at the end
    all_latents = [prompt_latent[0, 0].clone()]  # frame 0 = prompt

    for frame_idx in range(1, N_VIDEO_FRAMES):
        t_frame = time.time()

        # Fresh noise for generated frame (in output space on device)
        chunk_img = torch.randn(1, IN_CHANNELS, INPUT_H, INPUT_W)
        chunk_img = torch.clamp(chunk_img, -noise_abs_max, noise_abs_max)
        chunk_patches = patchify_to_output_space(chunk_img)
        chunk_pad_f = torch.zeros(N_PATCH_PAD, OUT_DIM, dtype=torch.bfloat16)
        chunk_pad_f[:N_PATCHES] = chunk_patches.to(torch.bfloat16)
        chunk_dev = to_tt(chunk_pad_f, tt_device)

        # DDIM denoise
        chunk_dev = run_ddim_loop(chunk_dev, "Frame %d" % frame_idx)

        # Readback generated latent (output space -> image space)
        chunk_host = ttnn.to_torch(chunk_dev)[:N_PATCHES].float()
        gen_latent = unpatchify_host(chunk_host, PATCH_SIZE, IN_CHANNELS, FRAME_H, FRAME_W)
        all_latents.append(gen_latent.squeeze(0))  # (C, H, W)

        # Update prompt: patch-embed the generated latent directly (no VAE round-trip)
        prompt_z_pad_new = torch.zeros(N_PATCH_PAD, D_MODEL, dtype=torch.bfloat16)
        prompt_z_pad_new[:N_PATCHES] = patch_embed_host(
            gen_latent, dev["x_emb_conv_w"], dev["x_emb_conv_b"])
        prompt_z_dev = to_tt(prompt_z_pad_new, tt_device)

        elapsed = time.time() - t_frame
        print("  Frame %d: %.2fs" % (frame_idx, elapsed))

    total_video = time.time() - t_video_start
    print("\nDiT generation: %.1fs for %d frames (%.2f FPS)" % (
        total_video, N_VIDEO_FRAMES - 1, (N_VIDEO_FRAMES - 1) / total_video))

    # Batch VAE decode all frames
    print("\nVAE decoding %d frames..." % len(all_latents))
    t_vae = time.time()
    all_latent_tensor = torch.stack(all_latents, dim=0).unsqueeze(0)  # (1, T, C, H, W)
    all_decoded = vae_decode_cpu(all_latent_tensor, vae)  # (1, T, 360, 640, 3)
    print("VAE decode: %.1fs" % (time.time() - t_vae))

    # Save as individual PNGs
    os.makedirs("/tmp/oasis_video", exist_ok=True)
    for i in range(all_decoded.shape[1]):
        frame_rgb = (all_decoded[0, i] * 255).byte().numpy()
        Image.fromarray(frame_rgb).save("/tmp/oasis_video/frame_%04d.png" % i)
    print("Saved %d frames to /tmp/oasis_video/" % all_decoded.shape[1])

    # Save first generated frame
    first_gen = (all_decoded[0, 1] * 255).byte().numpy()
    Image.fromarray(first_gen).save("/tmp/oasis_frame.png")

    print("\nDone!")
    ttnn.close_device(tt_device)
