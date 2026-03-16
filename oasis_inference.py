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

# Multi-chip tensor parallelism: set to 1 for single-chip, 2+ for TP
N_CHIPS = 1

# VAE decoder constants
VAE_DEC_DEPTH = 12
VAE_DEC_DIM = 1024
VAE_DEC_HEADS = 16
VAE_LATENT_DIM = 16
VAE_PATCH_SIZE_VAE = 20
VAE_PATCH_DIM = 3 * VAE_PATCH_SIZE_VAE ** 2  # 1200
VAE_SEQ_H = 18   # 360 / 20
VAE_SEQ_W = 32   # 640 / 20
VAE_SEQ_LEN = VAE_SEQ_H * VAE_SEQ_W  # 576
VAE_SEQ_TILES = VAE_SEQ_LEN // TILE  # 18
VAE_D_HEAD = VAE_DEC_DIM // VAE_DEC_HEADS  # 64
VAE_D_MLP = VAE_DEC_DIM * 4  # 4096
VAE_ROPE_DIM = VAE_D_HEAD // 2  # 32 (first 32 dims of each head rotated)

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

def make_vae_rope_kernel(n_heads, head_tiles):
    """Fused RoPE for VAE decoder: reads qkv_full, writes Q_roped, K_roped, V
    in heads-first layout (N_HEADS * SEQ, D_HEAD) ready for single reshape to SDPA.
    Eliminates 5 slices + 2 fused_rope calls + 3 reshape + 3 permute = 13 ops -> 1."""
    @ttl.kernel(grid="auto")
    def vae_rope_kernel(qkv_full, cos_tab, sin_tab, q_out, k_out, v_out):
        grid_cols, _ = ttl.grid_size(dims=2)
        seq_tiles = qkv_full.shape[0] // TILE
        total = seq_tiles * n_heads
        tiles_per_core = -(-total // grid_cols)
        d_tiles = n_heads * head_tiles

        q_dfb = ttl.make_dataflow_buffer_like(qkv_full, shape=(1, head_tiles), buffer_factor=2)
        qs_dfb = ttl.make_dataflow_buffer_like(qkv_full, shape=(1, head_tiles), buffer_factor=2)
        k_dfb = ttl.make_dataflow_buffer_like(qkv_full, shape=(1, head_tiles), buffer_factor=2)
        ks_dfb = ttl.make_dataflow_buffer_like(qkv_full, shape=(1, head_tiles), buffer_factor=2)
        v_dfb = ttl.make_dataflow_buffer_like(qkv_full, shape=(1, head_tiles), buffer_factor=2)
        cos_dfb = ttl.make_dataflow_buffer_like(cos_tab, shape=(1, head_tiles), buffer_factor=2)
        sin_dfb = ttl.make_dataflow_buffer_like(sin_tab, shape=(1, head_tiles), buffer_factor=2)
        qr_dfb = ttl.make_dataflow_buffer_like(q_out, shape=(1, head_tiles), buffer_factor=2)
        kr_dfb = ttl.make_dataflow_buffer_like(k_out, shape=(1, head_tiles), buffer_factor=2)
        vo_dfb = ttl.make_dataflow_buffer_like(v_out, shape=(1, head_tiles), buffer_factor=2)

        @ttl.compute()
        def compute():
            core_x, _ = ttl.core(dims=2)
            for local_t in range(tiles_per_core):
                t = core_x * tiles_per_core + local_t
                if t < total:
                    with cos_dfb.wait() as cv, sin_dfb.wait() as sv:
                        with q_dfb.wait() as q, qs_dfb.wait() as qs, qr_dfb.reserve() as qr:
                            qr.store(q * cv + qs * sv)
                        with k_dfb.wait() as k, ks_dfb.wait() as ks, kr_dfb.reserve() as kr:
                            kr.store(k * cv + ks * sv)
                    with v_dfb.wait() as vv, vo_dfb.reserve() as vo:
                        vo.store(vv)

        @ttl.datamovement()
        def dm_read():
            core_x, _ = ttl.core(dims=2)
            for local_t in range(tiles_per_core):
                t = core_x * tiles_per_core + local_t
                if t < total:
                    row = t // n_heads
                    h = t % n_heads
                    hc = h * head_tiles
                    with q_dfb.reserve() as blk:
                        tx = ttl.copy(qkv_full[row, hc:hc + head_tiles], blk); tx.wait()
                    with qs_dfb.reserve() as blk:
                        tx = ttl.copy(qkv_full[row, 3 * d_tiles + hc:3 * d_tiles + hc + head_tiles], blk); tx.wait()
                    with k_dfb.reserve() as blk:
                        tx = ttl.copy(qkv_full[row, d_tiles + hc:d_tiles + hc + head_tiles], blk); tx.wait()
                    with ks_dfb.reserve() as blk:
                        tx = ttl.copy(qkv_full[row, 4 * d_tiles + hc:4 * d_tiles + hc + head_tiles], blk); tx.wait()
                    with v_dfb.reserve() as blk:
                        tx = ttl.copy(qkv_full[row, 2 * d_tiles + hc:2 * d_tiles + hc + head_tiles], blk); tx.wait()
                    with cos_dfb.reserve() as blk:
                        tx = ttl.copy(cos_tab[row, hc:hc + head_tiles], blk); tx.wait()
                    with sin_dfb.reserve() as blk:
                        tx = ttl.copy(sin_tab[row, hc:hc + head_tiles], blk); tx.wait()

        @ttl.datamovement()
        def dm_write():
            core_x, _ = ttl.core(dims=2)
            for local_t in range(tiles_per_core):
                t = core_x * tiles_per_core + local_t
                if t < total:
                    row = t // n_heads
                    h = t % n_heads
                    out_row = h * seq_tiles + row
                    with qr_dfb.wait() as blk:
                        tx = ttl.copy(blk, q_out[out_row, 0:head_tiles]); tx.wait()
                    with kr_dfb.wait() as blk:
                        tx = ttl.copy(blk, k_out[out_row, 0:head_tiles]); tx.wait()
                    with vo_dfb.wait() as blk:
                        tx = ttl.copy(blk, v_out[out_row, 0:head_tiles]); tx.wait()

    return vae_rope_kernel

def make_fused_gated_res_ln_adaln_kernel(dim_tiles):
    """Fused gated_residual + LayerNorm + adaLN modulate.
    Computes: gated_res_out = residual + x * gate (also written to gated_res_out for downstream use)
              modulated = LN(gated_res) * (1 + scale) + shift
    Eliminates 1 DRAM intermediate (normed: 640KB) per call.
    Still writes gated_res to DRAM since it's needed as residual for FC2.
    Called 32 times per DDIM step (once per sub-block MLP path).
    Recomputes gated_res in LN passes 2 and 3 to avoid extra DRAM reads.
    adaln_packed has shift at cols [3D:4D] and scale at cols [4D:5D]."""
    @ttl.kernel(grid="auto")
    def fused_kernel(residual, x, gate, scaler, mean_scale, adaln_packed, gated_res_out, out):
        grid_cols, _ = ttl.grid_size(dims=2)
        seq_tiles = residual.shape[0] // TILE
        tiles_per_core = -(-seq_tiles // grid_cols)
        # 3 sets of input DFBs (one per LN pass, each recomputes gated_res)
        res_dfb = ttl.make_dataflow_buffer_like(residual, shape=(1, 1), buffer_factor=2)
        x_dfb = ttl.make_dataflow_buffer_like(x, shape=(1, 1), buffer_factor=2)
        g_dfb = ttl.make_dataflow_buffer_like(gate, shape=(1, 1), buffer_factor=2)
        gr_dfb = ttl.make_dataflow_buffer_like(residual, shape=(1, 1), buffer_factor=2)  # gated_res temp for reduce
        sc_dfb = ttl.make_dataflow_buffer_like(scaler, shape=(1, 1), buffer_factor=1)
        ms_dfb = ttl.make_dataflow_buffer_like(mean_scale, shape=(1, 1), buffer_factor=1)
        red_dfb = ttl.make_dataflow_buffer_like(scaler, shape=(1, 1), buffer_factor=2)
        acc_dfb = ttl.make_dataflow_buffer_like(scaler, shape=(1, 1), buffer_factor=2)
        bcast_dfb = ttl.make_dataflow_buffer_like(residual, shape=(1, 1), buffer_factor=2)
        sq_dfb = ttl.make_dataflow_buffer_like(residual, shape=(1, 1), buffer_factor=2)
        mean_dfb = ttl.make_dataflow_buffer_like(residual, shape=(1, 1), buffer_factor=2)
        istd_dfb = ttl.make_dataflow_buffer_like(residual, shape=(1, 1), buffer_factor=2)
        sh_dfb = ttl.make_dataflow_buffer_like(adaln_packed, shape=(1, 1), buffer_factor=2)
        scl_dfb = ttl.make_dataflow_buffer_like(adaln_packed, shape=(1, 1), buffer_factor=2)
        gro_dfb = ttl.make_dataflow_buffer_like(gated_res_out, shape=(1, 1), buffer_factor=2)
        out_dfb = ttl.make_dataflow_buffer_like(out, shape=(1, 1), buffer_factor=2)

        @ttl.compute()
        def compute():
            core_x, _ = ttl.core(dims=2)
            with sc_dfb.wait() as sc, ms_dfb.wait() as ms:
                for local_t in range(tiles_per_core):
                    tile_idx = core_x * tiles_per_core + local_t
                    if tile_idx < seq_tiles:
                        # Pass 1: compute gated_res, write to output, and accumulate mean
                        with res_dfb.wait() as rv, x_dfb.wait() as xv, g_dfb.wait() as gv:
                            with gr_dfb.reserve() as gr:
                                gr.store(rv + xv * gv)
                        with gr_dfb.wait() as grv:
                            with gro_dfb.reserve() as gro:
                                gro.store(grv)
                            with red_dfb.reserve() as r:
                                r.store(ttl.math.reduce_sum(grv, sc, dims=[1]))
                        with red_dfb.wait() as rv, acc_dfb.reserve() as acc:
                            acc.store(rv)
                        for j in range(dim_tiles - 1):
                            with res_dfb.wait() as rv, x_dfb.wait() as xv, g_dfb.wait() as gv:
                                with gr_dfb.reserve() as gr:
                                    gr.store(rv + xv * gv)
                            with gr_dfb.wait() as grv:
                                with gro_dfb.reserve() as gro:
                                    gro.store(grv)
                                with red_dfb.reserve() as r:
                                    r.store(ttl.math.reduce_sum(grv, sc, dims=[1]))
                            with red_dfb.wait() as rv, acc_dfb.wait() as av, acc_dfb.reserve() as acc:
                                acc.store(av + rv)
                        with acc_dfb.wait() as sum_x, bcast_dfb.reserve() as bc:
                            bc.store(ttl.math.broadcast(sum_x, dims=[1]))
                        with bcast_dfb.wait() as sum_x_bc, mean_dfb.reserve() as mean_out:
                            mean_out.store(sum_x_bc * ms)
                        # Pass 2: compute variance (recompute gated_res)
                        with mean_dfb.wait() as mean_val:
                            with res_dfb.wait() as rv, x_dfb.wait() as xv, g_dfb.wait() as gv:
                                with gr_dfb.reserve() as gr:
                                    gr.store(rv + xv * gv)
                            with gr_dfb.wait() as grv:
                                with sq_dfb.reserve() as sq:
                                    sq.store((grv - mean_val) * (grv - mean_val))
                            with sq_dfb.wait() as sqv, red_dfb.reserve() as r:
                                r.store(ttl.math.reduce_sum(sqv, sc, dims=[1]))
                            with red_dfb.wait() as rv, acc_dfb.reserve() as acc:
                                acc.store(rv)
                            for j in range(dim_tiles - 1):
                                with res_dfb.wait() as rv, x_dfb.wait() as xv, g_dfb.wait() as gv:
                                    with gr_dfb.reserve() as gr:
                                        gr.store(rv + xv * gv)
                                with gr_dfb.wait() as grv:
                                    with sq_dfb.reserve() as sq:
                                        sq.store((grv - mean_val) * (grv - mean_val))
                                with sq_dfb.wait() as sqv, red_dfb.reserve() as r:
                                    r.store(ttl.math.reduce_sum(sqv, sc, dims=[1]))
                                with red_dfb.wait() as rv, acc_dfb.wait() as av, acc_dfb.reserve() as acc:
                                    acc.store(av + rv)
                            with acc_dfb.wait() as sum_sq, bcast_dfb.reserve() as bc:
                                bc.store(ttl.math.broadcast(sum_sq, dims=[1]))
                            with bcast_dfb.wait() as var_bc, istd_dfb.reserve() as istd:
                                istd.store(ttl.math.rsqrt(var_bc * ms + ttl.math.fill(var_bc, 1e-6)))
                            # Pass 3: normalize + adaln modulate (recompute gated_res)
                            with istd_dfb.wait() as inv_std:
                                for j in range(dim_tiles):
                                    with res_dfb.wait() as rv, x_dfb.wait() as xv, g_dfb.wait() as gv:
                                        with gr_dfb.reserve() as gr:
                                            gr.store(rv + xv * gv)
                                    with gr_dfb.wait() as grv, sh_dfb.wait() as shv, scl_dfb.wait() as sclv, out_dfb.reserve() as o:
                                        normed = (grv - mean_val) * inv_std
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
                    # Pass 1: mean (read residual, x, gate)
                    for j in range(dim_tiles):
                        with res_dfb.reserve() as blk:
                            tx = ttl.copy(residual[tile_idx, j], blk); tx.wait()
                        with x_dfb.reserve() as blk:
                            tx = ttl.copy(x[tile_idx, j], blk); tx.wait()
                        with g_dfb.reserve() as blk:
                            tx = ttl.copy(gate[tile_idx, j], blk); tx.wait()
                    # Pass 2: variance (re-read residual, x, gate)
                    for j in range(dim_tiles):
                        with res_dfb.reserve() as blk:
                            tx = ttl.copy(residual[tile_idx, j], blk); tx.wait()
                        with x_dfb.reserve() as blk:
                            tx = ttl.copy(x[tile_idx, j], blk); tx.wait()
                        with g_dfb.reserve() as blk:
                            tx = ttl.copy(gate[tile_idx, j], blk); tx.wait()
                    # Pass 3: normalize + modulate (re-read residual, x, gate + shift, scale)
                    for j in range(dim_tiles):
                        with res_dfb.reserve() as blk:
                            tx = ttl.copy(residual[tile_idx, j], blk); tx.wait()
                        with x_dfb.reserve() as blk:
                            tx = ttl.copy(x[tile_idx, j], blk); tx.wait()
                        with g_dfb.reserve() as blk:
                            tx = ttl.copy(gate[tile_idx, j], blk); tx.wait()
                        with sh_dfb.reserve() as blk:
                            tx = ttl.copy(adaln_packed[tile_idx, 3 * dim_tiles + j], blk); tx.wait()
                        with scl_dfb.reserve() as blk:
                            tx = ttl.copy(adaln_packed[tile_idx, 4 * dim_tiles + j], blk); tx.wait()

        @ttl.datamovement()
        def dm_write():
            core_x, _ = ttl.core(dims=2)
            for local_t in range(tiles_per_core):
                tile_idx = core_x * tiles_per_core + local_t
                if tile_idx < seq_tiles:
                    for j in range(dim_tiles):
                        with gro_dfb.wait() as blk:
                            tx = ttl.copy(blk, gated_res_out[tile_idx, j]); tx.wait()
                    for j in range(dim_tiles):
                        with out_dfb.wait() as blk:
                            tx = ttl.copy(blk, out[tile_idx, j]); tx.wait()
    return fused_kernel

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
fused_gated_res_ln_adaln_d1024 = make_fused_gated_res_ln_adaln_kernel(D_TILES)  # gated_res + LN + adaLN (5-core seq-only)
# TODO: pipe-based version produces incorrect values on repeated calls (many-to-one
# pipe runtime args caching bug). Re-enable once resolved.
# from pipe_fused_ln import make_pipe_fused_gated_res_ln_adaln
# pipe_fused_gated_res_ln_adaln_d1024 = make_pipe_fused_gated_res_ln_adaln(D_TILES, 8, N_PATCH_PAD // TILE)
# Fused adaLN params: matmul + bias + expand in one kernel (replaces ~10 ttnn ops)
from adaln_matmul_expand import make_adaln_matmul_expand_kernel
adaln_matmul_expand_kernel = make_adaln_matmul_expand_kernel(D_TILES, N_PATCH_PAD // TILE)
# Fused RoPE + layout transform: eliminates 5 slices + 2 RoPE + 9 reshape/permute
from rope_layout_kernel import make_rope_layout_kernel, make_rope_temporal_kernel
N_HEADS_TP = N_HEADS // N_CHIPS
rope_layout_spatial = make_rope_layout_kernel(N_PATCH_PAD // TILE, D_HEAD // TILE, N_HEADS_TP)
# Fused temporal RoPE: eliminates 5 slices + 2 RoPE kernels
rope_temporal = make_rope_temporal_kernel(D_HEAD // TILE, N_HEADS_TP)
# VAE decoder RoPE + layout (2D pixel-based, seq=576 tokens)
vae_rope_layout = make_rope_layout_kernel(VAE_SEQ_TILES, VAE_D_HEAD // TILE, VAE_DEC_HEADS)
# Fused VAE RoPE: reads qkv_full, writes Q/K/V in heads-first SDPA layout
# Uses (1, 2) tile DFBs (~160KB/core) vs vae_rope_layout's (18, 2) (~1.4MB/core OOM)
vae_rope_fused = make_vae_rope_kernel(VAE_DEC_HEADS, VAE_D_HEAD // TILE)

# Compute kernel configs (following tt-metal DiT patterns)
# Initialized lazily after device is opened (need device.arch())
MATMUL_COMPUTE_CONFIG = None
SDPA_COMPUTE_CONFIG = None
SDPA_PROGRAM_CONFIG = None

def init_compute_configs(device):
    """Initialize compute kernel configs. Call after device is opened."""
    global MATMUL_COMPUTE_CONFIG, SDPA_COMPUTE_CONFIG, SDPA_PROGRAM_CONFIG
    arch = device.arch()
    MATMUL_COMPUTE_CONFIG = ttnn.init_device_compute_kernel_config(
        arch,
        math_fidelity=ttnn.MathFidelity.HiFi2,
        math_approx_mode=False,
        fp32_dest_acc_en=True,
        packer_l1_acc=True,
    )
    SDPA_COMPUTE_CONFIG = ttnn.init_device_compute_kernel_config(
        arch,
        math_fidelity=ttnn.MathFidelity.HiFi2,
        math_approx_mode=False,
        fp32_dest_acc_en=False,
    )
    grid = device.compute_with_storage_grid_size()
    SDPA_PROGRAM_CONFIG = ttnn.SDPAProgramConfig(
        compute_with_storage_grid_size=(grid.x, grid.y - 1),
        q_chunk_size=128,
        k_chunk_size=128,
    )

# ============================================================
# Host helpers
# ============================================================

def _mesh_kwargs(device):
    """Return mesh_mapper kwarg if device is a MeshDevice."""
    if isinstance(device, ttnn.MeshDevice):
        return {"mesh_mapper": ttnn.ReplicateTensorToMesh(device)}
    return {}

def to_tt(t, device):
    return ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                           device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG,
                           **_mesh_kwargs(device))

def to_tt_l1(t, device):
    return ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                           device=device, memory_config=ttnn.L1_MEMORY_CONFIG,
                           **_mesh_kwargs(device))

def shard_tt(t, device, dim):
    """Load tensor sharded across mesh devices along given dimension."""
    return ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                           device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG,
                           mesh_mapper=ttnn.ShardTensorToMesh(device, dim=dim))

def shard_tt_l1(t, device, dim):
    """Load tensor sharded across mesh devices along given dimension (L1)."""
    return ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                           device=device, memory_config=ttnn.L1_MEMORY_CONFIG,
                           mesh_mapper=ttnn.ShardTensorToMesh(device, dim=dim))

def zeros_tt(shape, device):
    return to_tt(torch.zeros(shape, dtype=torch.bfloat16), device)

def zeros_l1(shape, device):
    return to_tt_l1(torch.zeros(shape, dtype=torch.bfloat16), device)

# Global mesh device reference for readback (set in main)
_MESH_DEVICE = None

def readback_torch(t):
    """Read tensor from device/mesh to torch. For mesh, reads chip 0."""
    if _MESH_DEVICE is not None:
        return ttnn.to_torch(t, mesh_composer=ttnn.ConcatMeshToTensor(_MESH_DEVICE, dim=0))[:t.shape[0]]
    return ttnn.to_torch(t)

def expand_bias(bias_1d, seq_pad):
    return bias_1d.unsqueeze(0).expand(seq_pad, -1).contiguous().to(torch.bfloat16)

def interleave_qkv_for_tp(qkv_full_w, n_chips, d_model, d_head):
    """Rearrange QKV weight for per-head sharding across chips.
    Input: (d_model, 5*d_model) layout [Q|K|V|Q_swap|K_swap] each d_model cols.
    Output: (d_model, 5*d_model) rearranged so ShardTensorToMesh(dim=1)
    gives each chip n_heads/n_chips heads with [Q|K|V|Q_swap|K_swap] layout."""
    if n_chips == 1:
        return qkv_full_w
    heads_per_chip = d_model // d_head // n_chips
    cols_per_chip = heads_per_chip * d_head  # per section
    sections = torch.split(qkv_full_w, d_model, dim=1)  # 5 sections
    chip_parts = []
    for chip in range(n_chips):
        sc = chip * cols_per_chip
        ec = sc + cols_per_chip
        for section in sections:
            chip_parts.append(section[:, sc:ec])
    return torch.cat(chip_parts, dim=1)

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

def swap_adjacent_elements(b):
    """1D version of swap_adjacent_columns for bias vectors."""
    b_swap = b.clone()
    b_swap[0::2] = b[1::2]
    b_swap[1::2] = b[0::2]
    return b_swap

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
        src_start = t * n_patch_pad
        n = min(n_positions - src_start, n_patch_pad)
        cos_full[start:start + n] = cos_expanded[src_start:src_start + n].to(torch.bfloat16)
        sin_full[start:start + n] = sin_expanded[src_start:src_start + n].to(torch.bfloat16)

    # L1 for RoPE tables: read 16x per step by each spatial/temporal sub-block
    return to_tt_l1(cos_full, tt_device), to_tt_l1(sin_full, tt_device)

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

    # Repeat across TP heads: (N_PATCHES, 64) -> (N_PATCHES, N_HEADS_TP * D_HEAD)
    d_model_tp = N_HEADS_TP * D_HEAD
    cos_expanded = cos_per_pos.repeat(1, N_HEADS_TP)
    sin_expanded = sin_perm_per_pos.repeat(1, N_HEADS_TP)

    # Pad to N_PATCH_PAD rows, repeat for T frames
    cos_full = torch.zeros(SEQ, d_model_tp, dtype=torch.bfloat16)
    sin_full = torch.zeros(SEQ, d_model_tp, dtype=torch.bfloat16)
    for t in range(n_frames):
        start = t * N_PATCH_PAD
        cos_full[start:start + N_PATCHES] = cos_expanded.to(torch.bfloat16)
        sin_full[start:start + N_PATCHES] = sin_expanded.to(torch.bfloat16)

    # L1 for RoPE tables: read 16x per step by each spatial/temporal sub-block
    return to_tt_l1(cos_full, tt_device), to_tt_l1(sin_full, tt_device)

def precompute_vae_rope(vae, tt_device):
    """Extract 2D pixel-based RoPE tables from CPU VAE model.
    Uses the actual rotary_freqs buffer from RotaryEmbedding to match CPU exactly.
    Returns cos_tt, sin_perm_tt each (VAE_SEQ_LEN, VAE_DEC_DIM) on device."""
    # Extract the actual rotary_freqs from the model (same for all decoder blocks)
    cpu_freqs = vae.decoder[0].attn.rotary_freqs  # (H, W, rot_dim) = (18, 32, 32)
    freqs_flat = cpu_freqs.reshape(VAE_SEQ_LEN, -1)  # (576, 32)
    rot_dim = freqs_flat.shape[1]  # 32

    # Build per-head tables: first rot_dim dims rotated, rest identity
    cos_per_head = torch.ones(VAE_SEQ_LEN, VAE_D_HEAD)
    sin_per_head = torch.zeros(VAE_SEQ_LEN, VAE_D_HEAD)
    cos_per_head[:, :rot_dim] = freqs_flat.cos()
    sin_per_head[:, :rot_dim] = freqs_flat.sin()

    # Bake rotate_half sign into sin: [-, +, -, +, ...] for rotated dims
    sign = torch.ones(VAE_D_HEAD)
    sign[:rot_dim:2] = -1
    sin_perm_per_head = sin_per_head * sign.unsqueeze(0)

    # Repeat across all 16 heads: (576, 64) -> (576, 1024)
    cos_full = cos_per_head.repeat(1, VAE_DEC_HEADS).to(torch.bfloat16)
    sin_full = sin_perm_per_head.repeat(1, VAE_DEC_HEADS).to(torch.bfloat16)

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
                # TP: interleave heads so ShardTensorToMesh(dim=1) gives each chip its heads
                if N_CHIPS > 1:
                    qkv_full_w = interleave_qkv_for_tp(qkv_full_w, N_CHIPS, D_MODEL, D_HEAD)
                    dev["%s.qkv_full_w" % p] = shard_tt(qkv_full_w, tt_device, dim=1)
                else:
                    dev["%s.qkv_full_w" % p] = to_tt(qkv_full_w, tt_device)
                # Separate QK_swap weights still needed for temporal path
                qk_swap_w = torch.cat([swap_adjacent_columns(q_w),
                                       swap_adjacent_columns(k_w)], dim=1)
                dev["%s.qk_swap_w" % p] = to_tt(qk_swap_w, tt_device)
                dev["%s.qkv_w" % p] = to_tt(qkv_w, tt_device)

                # Output projection: row-parallel (shard input dim)
                SEQ = N_PATCH_PAD * n_frames
                out_w = st.get_tensor("%s_attn.to_out.weight" % p).T.contiguous().to(torch.bfloat16)
                if N_CHIPS > 1:
                    dev["%s.out_w" % p] = shard_tt(out_w, tt_device, dim=0)
                else:
                    dev["%s.out_w" % p] = to_tt(out_w, tt_device)
                out_b = st.get_tensor("%s_attn.to_out.bias" % p).to(torch.bfloat16)
                dev["%s.out_b" % p] = to_tt(expand_bias(out_b, SEQ), tt_device)

                # MLP: fc1 column-parallel (shard output dim), fc2 row-parallel (shard input dim)
                fc1_w = st.get_tensor("%s_mlp.fc1.weight" % p).T.contiguous().to(torch.bfloat16)
                fc1_b = st.get_tensor("%s_mlp.fc1.bias" % p).to(torch.bfloat16)
                fc2_w = st.get_tensor("%s_mlp.fc2.weight" % p).T.contiguous().to(torch.bfloat16)
                fc2_b = st.get_tensor("%s_mlp.fc2.bias" % p).to(torch.bfloat16)
                if N_CHIPS > 1:
                    dev["%s.fc1_w" % p] = shard_tt(fc1_w, tt_device, dim=1)
                    dev["%s.fc1_b" % p] = shard_tt(expand_bias(fc1_b, SEQ), tt_device, dim=1)
                    dev["%s.fc2_w" % p] = shard_tt(fc2_w, tt_device, dim=0)
                else:
                    dev["%s.fc1_w" % p] = to_tt(fc1_w, tt_device)
                    dev["%s.fc1_b" % p] = to_tt(expand_bias(fc1_b, SEQ), tt_device)
                    dev["%s.fc2_w" % p] = to_tt(fc2_w, tt_device)
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

        # Device-side spatial RoPE tables (sized for N_HEADS_TP heads per chip)
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
                                     N_PATCH_PAD, N_HEADS_TP, n_frames, tt_device)

    elapsed = time.time() - t0
    print("Preloaded %d tensors in %.1fs" % (len(dev), elapsed))
    return dev

def preload_vae_decoder_weights(vae, tt_device):
    """Extract VAE decoder weights from CPU model and load onto device."""
    t0 = time.time()
    vd = {}
    sd = vae.state_dict()

    # post_quant_conv: Linear(16, 1024)
    pqc_w = sd["post_quant_conv.weight"].T.contiguous().to(torch.bfloat16)  # (16, 1024)
    # Pad input dim to 32 for tile alignment
    pqc_w_padded = torch.zeros(32, VAE_DEC_DIM, dtype=torch.bfloat16)
    pqc_w_padded[:VAE_LATENT_DIM] = pqc_w
    vd["pqc_w"] = to_tt(pqc_w_padded, tt_device)
    vd["pqc_b"] = to_tt(expand_bias(sd["post_quant_conv.bias"].to(torch.bfloat16), VAE_SEQ_LEN), tt_device)

    for i in range(VAE_DEC_DEPTH):
        p = "decoder.%d" % i

        # LN weight/bias expanded to (VAE_SEQ_LEN, D) for layernorm kernel
        for ln_idx in ["1", "2"]:
            w = sd["%s.norm%s.weight" % (p, ln_idx)].to(torch.bfloat16)
            b = sd["%s.norm%s.bias" % (p, ln_idx)].to(torch.bfloat16)
            vd["%s.norm%s_w" % (p, ln_idx)] = to_tt(expand_bias(w, VAE_SEQ_LEN), tt_device)
            vd["%s.norm%s_b" % (p, ln_idx)] = to_tt(expand_bias(b, VAE_SEQ_LEN), tt_device)

        # Combined QKV+swap: [Q, K, V, Q_swap, K_swap] = (1024, 5120)
        qkv_w = sd["%s.attn.qkv.weight" % p].T.contiguous().to(torch.bfloat16)
        q_w = qkv_w[:, :VAE_DEC_DIM]
        k_w = qkv_w[:, VAE_DEC_DIM:2*VAE_DEC_DIM]
        v_w = qkv_w[:, 2*VAE_DEC_DIM:]
        qkv_full_w = torch.cat([q_w, k_w, v_w,
                                swap_adjacent_columns(q_w),
                                swap_adjacent_columns(k_w)], dim=1)
        vd["%s.qkv_full_w" % p] = to_tt(qkv_full_w, tt_device)

        # QKV bias with swap for RoPE trick
        qkv_b = sd["%s.attn.qkv.bias" % p].to(torch.bfloat16)
        b_q = qkv_b[:VAE_DEC_DIM]
        b_k = qkv_b[VAE_DEC_DIM:2*VAE_DEC_DIM]
        b_v = qkv_b[2*VAE_DEC_DIM:]
        full_b = torch.cat([b_q, b_k, b_v,
                           swap_adjacent_elements(b_q),
                           swap_adjacent_elements(b_k)])
        vd["%s.qkv_full_b" % p] = to_tt(expand_bias(full_b, VAE_SEQ_LEN), tt_device)

        # Output projection
        vd["%s.proj_w" % p] = to_tt(
            sd["%s.attn.proj.weight" % p].T.contiguous().to(torch.bfloat16), tt_device)
        vd["%s.proj_b" % p] = to_tt(
            expand_bias(sd["%s.attn.proj.bias" % p].to(torch.bfloat16), VAE_SEQ_LEN), tt_device)

        # MLP
        vd["%s.fc1_w" % p] = to_tt(
            sd["%s.mlp.fc1.weight" % p].T.contiguous().to(torch.bfloat16), tt_device)
        vd["%s.fc1_b" % p] = to_tt(
            expand_bias(sd["%s.mlp.fc1.bias" % p].to(torch.bfloat16), VAE_SEQ_LEN), tt_device)
        vd["%s.fc2_w" % p] = to_tt(
            sd["%s.mlp.fc2.weight" % p].T.contiguous().to(torch.bfloat16), tt_device)
        vd["%s.fc2_b" % p] = to_tt(
            expand_bias(sd["%s.mlp.fc2.bias" % p].to(torch.bfloat16), VAE_SEQ_LEN), tt_device)

    # Final LN
    vd["dec_norm_w"] = to_tt(
        expand_bias(sd["dec_norm.weight"].to(torch.bfloat16), VAE_SEQ_LEN), tt_device)
    vd["dec_norm_b"] = to_tt(
        expand_bias(sd["dec_norm.bias"].to(torch.bfloat16), VAE_SEQ_LEN), tt_device)

    # Predictor: Linear(1024, 1200) - pad to 1216 for tile alignment
    VAE_PATCH_DIM_PAD = ((VAE_PATCH_DIM + TILE - 1) // TILE) * TILE  # 1216
    pred_w_raw = sd["predictor.weight"].T.contiguous().to(torch.bfloat16)  # (1024, 1200)
    pred_w_padded = torch.zeros(VAE_DEC_DIM, VAE_PATCH_DIM_PAD, dtype=torch.bfloat16)
    pred_w_padded[:, :VAE_PATCH_DIM] = pred_w_raw
    vd["pred_w"] = to_tt(pred_w_padded, tt_device)
    pred_b_raw = sd["predictor.bias"].to(torch.bfloat16)  # (1200,)
    pred_b_padded = torch.zeros(VAE_PATCH_DIM_PAD, dtype=torch.bfloat16)
    pred_b_padded[:VAE_PATCH_DIM] = pred_b_raw
    vd["pred_b"] = to_tt(expand_bias(pred_b_padded, VAE_SEQ_LEN), tt_device)

    # VAE RoPE tables
    vd["vae_cos"], vd["vae_sin_perm"] = precompute_vae_rope(vae, tt_device)

    elapsed = time.time() - t0
    print("Preloaded %d VAE decoder tensors in %.1fs" % (len(vd), elapsed))
    return vd

def prealloc_vae_scratch(tt_device):
    """Pre-allocate VAE decoder scratch buffers. All DRAM to avoid L1 pressure
    (DiT already uses ~25MB of L1). Can promote hot tensors to L1 later."""
    t0 = time.time()
    vs = {}
    vs["vae_input"] = zeros_tt((VAE_SEQ_LEN, 32), tt_device)  # padded latent for trace
    vs["z_a"] = zeros_tt((VAE_SEQ_LEN, VAE_DEC_DIM), tt_device)
    vs["z_b"] = zeros_tt((VAE_SEQ_LEN, VAE_DEC_DIM), tt_device)
    vs["normed"] = zeros_tt((VAE_SEQ_LEN, VAE_DEC_DIM), tt_device)
    vs["qkv_full"] = zeros_tt((VAE_SEQ_LEN, 5 * VAE_DEC_DIM), tt_device)
    vs["o_proj"] = zeros_tt((VAE_SEQ_LEN, VAE_DEC_DIM), tt_device)
    vs["fc1"] = zeros_tt((VAE_SEQ_LEN, VAE_D_MLP), tt_device)
    vs["gelu"] = zeros_tt((VAE_SEQ_LEN, VAE_D_MLP), tt_device)
    vs["fc2"] = zeros_tt((VAE_SEQ_LEN, VAE_DEC_DIM), tt_device)
    VAE_PATCH_DIM_PAD = ((VAE_PATCH_DIM + TILE - 1) // TILE) * TILE  # 1216
    vs["pred_out"] = zeros_tt((VAE_SEQ_LEN, VAE_PATCH_DIM_PAD), tt_device)
    # SDPA buffers: heads-first layout for fused RoPE kernel output
    SDPA_ROWS = VAE_DEC_HEADS * VAE_SEQ_LEN  # 16 * 576 = 9216
    vs["q_sdpa"] = zeros_tt((SDPA_ROWS, VAE_D_HEAD), tt_device)
    vs["k_sdpa"] = zeros_tt((SDPA_ROWS, VAE_D_HEAD), tt_device)
    vs["v_sdpa"] = zeros_tt((SDPA_ROWS, VAE_D_HEAD), tt_device)
    elapsed = time.time() - t0
    print("Pre-allocated %d VAE scratch tensors in %.1fs" % (len(vs), elapsed))
    return vs

# ============================================================
# Scratch buffers
# ============================================================

def prealloc_scratch(tt_device, n_frames=2):
    """Pre-allocate scratch buffers. n_frames=2 for prompt+generated frame."""
    t0 = time.time()
    s = {}
    SEQ = N_PATCH_PAD * n_frames  # 320 for T=2
    D_MODEL_TP = D_MODEL // N_CHIPS  # per-chip model dim for sharded tensors
    D_MLP_TP = D_MLP // N_CHIPS
    # L1 intermediates: eliminates DRAM round-trips between operations
    s["z_a"] = zeros_l1((SEQ, D_MODEL), tt_device)
    s["z_b"] = zeros_l1((SEQ, D_MODEL), tt_device)
    s["normed"] = zeros_l1((SEQ, D_MODEL), tt_device)
    s["modulated"] = zeros_l1((SEQ, D_MODEL), tt_device)
    s["qkv"] = zeros_l1((SEQ, 3 * D_MODEL_TP), tt_device)
    s["qkv_full"] = zeros_l1((SEQ, 5 * D_MODEL_TP), tt_device)
    s["o_proj"] = zeros_l1((SEQ, D_MODEL), tt_device)
    s["fc1"] = zeros_l1((SEQ, D_MLP_TP), tt_device)
    s["gelu"] = zeros_l1((SEQ, D_MLP_TP), tt_device)
    s["fc2"] = zeros_l1((SEQ, D_MODEL), tt_device)
    # Conditioning scratch (per-frame, so TILE * n_frames)
    s["t_emb_a"] = zeros_tt((TILE, D_MODEL), tt_device)
    s["t_emb_b"] = zeros_tt((TILE, D_MODEL), tt_device)
    s["cond"] = zeros_tt((TILE, D_MODEL), tt_device)
    # adaLN scratch (replicated, not sharded)
    s["adaln_out"] = to_tt_l1(torch.zeros(TILE, 6 * D_MODEL, dtype=torch.bfloat16), tt_device)
    s["silu_cond"] = zeros_tt((TILE, D_MODEL), tt_device)
    # Packed adaln: (SEQ, 6*D_MODEL) reused across blocks (replicated)
    s["adaln_packed"] = zeros_l1((SEQ, 6 * D_MODEL), tt_device)
    # Per-frame adaln expanded: (N_PATCH_PAD, 6*D_MODEL) for building packed tensor
    for f in range(n_frames):
        s["adaln_frame_%d" % f] = zeros_l1((N_PATCH_PAD, 6 * D_MODEL), tt_device)
    # RoPE scratch: sized for N_HEADS_TP heads per chip
    s["q_roped"] = zeros_l1((SEQ, D_MODEL_TP), tt_device)
    s["k_roped"] = zeros_l1((SEQ, D_MODEL_TP), tt_device)
    # SDPA-format scratch for fused RoPE+layout kernel (spatial)
    # Stored as 2D: (BATCH_S * N_PATCH_PAD, D_HEAD), reshaped to 4D for SDPA
    BATCH_S = n_frames * N_HEADS_TP
    s["q_sdpa"] = to_tt_l1(torch.zeros(BATCH_S * N_PATCH_PAD, D_HEAD, dtype=torch.bfloat16), tt_device)
    s["k_sdpa"] = to_tt_l1(torch.zeros(BATCH_S * N_PATCH_PAD, D_HEAD, dtype=torch.bfloat16), tt_device)
    s["v_sdpa"] = to_tt_l1(torch.zeros(BATCH_S * N_PATCH_PAD, D_HEAD, dtype=torch.bfloat16), tt_device)
    # Temporal mega kernel scratch (Q/K roped + V) - sized for TP heads
    s["t_q_scratch"] = zeros_l1((SEQ, D_MODEL_TP), tt_device)
    s["t_k_scratch"] = zeros_l1((SEQ, D_MODEL_TP), tt_device)
    s["t_v_scratch"] = zeros_l1((SEQ, D_MODEL_TP), tt_device)
    # SDPA scratch: (SEQ, D_MODEL) for TT-Lang spatial SDPA output
    s["sdpa_out"] = zeros_l1((SEQ, D_MODEL), tt_device)
    # Mega kernel B scratch
    s["z_scratch"] = zeros_l1((SEQ, D_MODEL), tt_device)
    s["gelu_scratch"] = zeros_l1((SEQ, D_MLP), tt_device)
    # Final layer
    s["final_adaln"] = zeros_l1((TILE, 2 * D_MODEL), tt_device)
    s["final_out"] = zeros_l1((SEQ, OUT_DIM), tt_device)
    # Pre-allocated SiLU output buffers per frame (L1 for fast access, read 32x/step)
    for f in range(n_frames):
        s["silu_out_%d" % f] = to_tt_l1(torch.zeros(TILE, D_MODEL, dtype=torch.bfloat16), tt_device)
    s["n_frames"] = n_frames
    # DDIM arithmetic scratch (N_PATCH_PAD, OUT_DIM)
    s["ddim_x_start"] = zeros_l1((N_PATCH_PAD, OUT_DIM), tt_device)
    s["ddim_x_noise"] = zeros_l1((N_PATCH_PAD, OUT_DIM), tt_device)
    s["ddim_tmp"] = zeros_l1((N_PATCH_PAD, OUT_DIM), tt_device)
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
        ttnn.linear(silu_cond, dev["%s.adaln_w" % prefix],
                    bias=dev["%s.adaln_b" % prefix],
                    optional_output_tensor=scr["adaln_out"])
        expanded = ttnn.concat([scr["adaln_out"]] * N_REPEAT, dim=0)
        per_frame_expanded.append(expanded)

    if T == 1:
        return per_frame_expanded[0]
    else:
        return ttnn.concat(per_frame_expanded, dim=0)

PROFILE_BLOCKS = False
PROFILE_BLOCK_DEVICE = None  # set to "blocks.1.s" for per-phase profiling (breaks trace)

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

    # LayerNorm + adaLN modulate via ttnn ops
    # TODO: revisit TT-Lang fused_ln_adaln_d1024 kernel (0.3ms vs 0.15ms ttnn, 2x slower)
    # Was: fused_ln_adaln_d1024(x_tt, scaler, mean_scale, adaln_packed, scr["modulated"])
    normed_a = ttnn.layer_norm(x_tt)
    shift_a = ttnn.slice(adaln_packed, [0, 0], [SEQ, D_MODEL])
    scale_a = ttnn.slice(adaln_packed, [0, D_MODEL], [SEQ, 2 * D_MODEL])
    ttnn.add(scale_a, 1.0, output_tensor=scr["normed"])
    ttnn.multiply(normed_a, scr["normed"], output_tensor=scr["modulated"])
    ttnn.add(scr["modulated"], shift_a, output_tensor=scr["modulated"])
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

    D_MODEL_TP = D_MODEL // N_CHIPS

    if attn_type == "spatial":
        cos_key, sin_key = "spatial_cos", "spatial_sin_perm"
        # Fused RoPE + layout: reads qkv_full, writes Q,K,V in SDPA format directly
        rope_layout_spatial(scr["qkv_full"], dev[cos_key], dev[sin_key],
                           scr["q_sdpa"], scr["k_sdpa"], scr["v_sdpa"])
        BATCH_S = T * N_HEADS_TP
        # Already in (BATCH_S * N_PATCH_PAD, D_HEAD) layout, just reshape for SDPA
        q_s = ttnn.reshape(scr["q_sdpa"], [BATCH_S, 1, N_PATCH_PAD, D_HEAD])
        k_s = ttnn.reshape(scr["k_sdpa"], [BATCH_S, 1, N_PATCH_PAD, D_HEAD])
        v_s = ttnn.reshape(scr["v_sdpa"], [BATCH_S, 1, N_PATCH_PAD, D_HEAD])
        attn_out = ttnn.transformer.scaled_dot_product_attention(q_s, k_s, v_s, is_causal=False)
        attn_out = ttnn.reshape(attn_out, [T, N_HEADS_TP, N_PATCH_PAD, D_HEAD])
        attn_out = ttnn.permute(attn_out, [0, 2, 1, 3])
        attn_2d = ttnn.reshape(attn_out, [SEQ, D_MODEL_TP])
        _timer.mark("qkv+sdpa")
    else:
        # Fused temporal RoPE: reads qkv_full, writes q/k/v in (SEQ, D_MODEL_TP)
        cos_key, sin_key = "temporal_cos", "temporal_sin_perm"
        rope_temporal(scr["qkv_full"], dev[cos_key], dev[sin_key],
                      scr["q_roped"], scr["k_roped"], scr["t_v_scratch"])
        # Temporal: each patch attends across frames, batch over (patch, head)
        BATCH_T = N_PATCH_PAD * N_HEADS_TP
        q_t = ttnn.reshape(scr["q_roped"], [T, BATCH_T, D_HEAD])
        q_t = ttnn.permute(q_t, [1, 0, 2])
        q_t = ttnn.reshape(q_t, [BATCH_T, 1, T, D_HEAD])
        k_t = ttnn.reshape(scr["k_roped"], [T, BATCH_T, D_HEAD])
        k_t = ttnn.permute(k_t, [1, 0, 2])
        k_t = ttnn.reshape(k_t, [BATCH_T, 1, T, D_HEAD])
        v_t = ttnn.reshape(scr["t_v_scratch"], [T, BATCH_T, D_HEAD])
        v_t = ttnn.permute(v_t, [1, 0, 2])
        v_t = ttnn.reshape(v_t, [BATCH_T, 1, T, D_HEAD])
        attn_out = ttnn.transformer.scaled_dot_product_attention(q_t, k_t, v_t, is_causal=True)
        attn_out = ttnn.reshape(attn_out, [BATCH_T, T, D_HEAD])
        attn_out = ttnn.permute(attn_out, [1, 0, 2])
        attn_2d = ttnn.reshape(attn_out, [SEQ, D_MODEL_TP])
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
    # Phase A: O proj (row-parallel) + all_reduce + bias
    # TODO: fuse matmul+bias via ttnn.linear when N_CHIPS==1 (no all_reduce between them)
    o_proj = ttnn.matmul(attn_2d, dev["%s.out_w" % prefix])
    if N_CHIPS > 1:
        o_proj = ttnn.all_reduce(o_proj)
    ttnn.add(o_proj, dev["%s.out_b" % prefix], output_tensor=scr["o_proj"])
    gate_msa = ttnn.slice(adaln_packed, [0, 2 * D_MODEL], [SEQ, 3 * D_MODEL])

    # Phase B: gated_residual + LN + adaLN modulate via ttnn ops
    # TODO: revisit TT-Lang fused_gated_res_ln_adaln_d1024 kernel (0.8ms vs 0.15ms ttnn, 5.5x slower)
    # Was: fused_gated_res_ln_adaln_d1024(x_tt, scr["o_proj"], gate_msa, scaler, mean_scale,
    #          adaln_packed, scr["z_scratch"], scr["modulated"])
    ttnn.multiply(scr["o_proj"], gate_msa, output_tensor=scr["normed"])
    ttnn.add(x_tt, scr["normed"], output_tensor=scr["z_scratch"])
    normed = ttnn.layer_norm(scr["z_scratch"])
    shift_b = ttnn.slice(adaln_packed, [0, 3 * D_MODEL], [SEQ, 4 * D_MODEL])
    scale_b = ttnn.slice(adaln_packed, [0, 4 * D_MODEL], [SEQ, 5 * D_MODEL])
    ttnn.add(scale_b, 1.0, output_tensor=scr["normed"])
    ttnn.multiply(normed, scr["normed"], output_tensor=scr["modulated"])
    ttnn.add(scr["modulated"], shift_b, output_tensor=scr["modulated"])
    if _do_dev_profile:
        ttnn.synchronize_device(tt_device)
        _t1 = _dt.time(); print("      %s ln+mod: %.1fms" % (prefix, (_t1 - _t0) * 1000)); _t0 = _t1

    # Phase C: Fused FC1 + bias + GELU (single ttnn.linear call)
    ttnn.linear(scr["modulated"], dev["%s.fc1_w" % prefix], bias=dev["%s.fc1_b" % prefix],
                activation="gelu", optional_output_tensor=scr["gelu"])
    if _do_dev_profile:
        ttnn.synchronize_device(tt_device)
        _t1 = _dt.time(); print("      %s fc1+gelu: %.1fms" % (prefix, (_t1 - _t0) * 1000)); _t0 = _t1

    # Phase D: FC2 (row-parallel) + all_reduce + bias + gated residual
    # TODO: fuse matmul+bias via ttnn.linear when N_CHIPS==1 (no all_reduce between them)
    fc2_out = ttnn.matmul(scr["gelu"], dev["%s.fc2_w" % prefix])
    if N_CHIPS > 1:
        fc2_out = ttnn.all_reduce(fc2_out)
    ttnn.add(fc2_out, dev["%s.fc2_b" % prefix], output_tensor=scr["fc2"])
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

def precompute_gen_cond(action_vec, ddim_steps, noise_range, dev, scr, tt_device):
    """Pre-compute conditioning for all DDIM noise levels with given action."""
    cond_per_step = {}
    for noise_idx in reversed(range(1, ddim_steps + 1)):
        noise_level = int(noise_range[noise_idx].item())
        cond_per_step[noise_idx] = compute_cond_for_frame(
            noise_level, action_vec, dev, scr, tt_device)
    return cond_per_step

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
    # Uses pre-allocated buffers from scratch (no allocation during trace)
    silu_cond_list = []
    for t_idx in range(T):
        silu_out = scr["silu_out_%d" % t_idx]
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
        final_raw = ttnn.linear(silu_cond_list[t_idx], dev["final_adaln_w"],
                                bias=dev["final_adaln_b"])
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
    # TODO: fuse linear+bias via ttnn.linear (currently uses TT-Lang linear_k32 kernel)
    linear_k32(scr["modulated"], dev["final_linear_w"], scr["final_out"])
    ttnn.add(scr["final_out"], dev["final_linear_b"], output_tensor=scr["final_out"])
    result = scr["final_out"]
    if profile_step:
        ttnn.synchronize_device(tt_device)
        print("      final_layer: %.1fms" % ((_pt() - t_blocks) * 1000))
    return result

# ============================================================
# VAE
# ============================================================

def load_vae_cpu():
    """Load VAE model on CPU (needed for encoding prompt image)."""
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
    """Decode latent to image using VAE on CPU (fallback).
    z_latent: (B, T, C, H, W) = (1, 1, 16, 18, 32)
    Returns: (B, T, 3, 360, 640) in [0, 1]
    """
    B, T = z_latent.shape[:2]
    z = rearrange(z_latent, "b t c h w -> (b t) (h w) c")
    with torch.no_grad():
        decoded = (vae.decode(z.float() / SCALING_FACTOR) + 1) / 2
    decoded = torch.clamp(decoded, 0, 1)
    return rearrange(decoded, "(b t) c h w -> b t h w c", b=B, t=T)

def vae_decode_forward(vae_dev, vae_scr, scaler, mean_scale):
    """VAE forward pass on device (trace-compatible, no allocations).
    Input: vae_scr["vae_input"] already populated with padded latent.
    Output: vae_scr["pred_out"] contains predictions."""
    # post_quant_conv: (576, 32) @ (32, 1024) + bias -> (576, 1024) in z_a
    ttnn.linear(vae_scr["vae_input"], vae_dev["pqc_w"], bias=vae_dev["pqc_b"],
                optional_output_tensor=vae_scr["z_a"])

    for i in range(VAE_DEC_DEPTH):
        p = "decoder.%d" % i

        # === Attention path: x + attn(LN(x)) ===
        layernorm_d1024(vae_scr["z_a"], vae_dev["%s.norm1_w" % p], vae_dev["%s.norm1_b" % p],
                        scaler, mean_scale, vae_scr["normed"])

        ttnn.linear(vae_scr["normed"], vae_dev["%s.qkv_full_w" % p],
                    bias=vae_dev["%s.qkv_full_b" % p],
                    optional_output_tensor=vae_scr["qkv_full"])

        # Fused RoPE: reads qkv_full, writes Q/K/V in heads-first SDPA layout
        vae_rope_fused(vae_scr["qkv_full"], vae_dev["vae_cos"], vae_dev["vae_sin_perm"],
                       vae_scr["q_sdpa"], vae_scr["k_sdpa"], vae_scr["v_sdpa"])

        # Single reshape to SDPA format (data already in heads-first order)
        q_s = ttnn.reshape(vae_scr["q_sdpa"], [VAE_DEC_HEADS, 1, VAE_SEQ_LEN, VAE_D_HEAD])
        k_s = ttnn.reshape(vae_scr["k_sdpa"], [VAE_DEC_HEADS, 1, VAE_SEQ_LEN, VAE_D_HEAD])
        v_s = ttnn.reshape(vae_scr["v_sdpa"], [VAE_DEC_HEADS, 1, VAE_SEQ_LEN, VAE_D_HEAD])
        attn_out = ttnn.transformer.scaled_dot_product_attention(
            q_s, k_s, v_s, is_causal=False)

        # Reshape back to (576, 1024): (heads, 1, seq, dim) -> (1, seq, heads, dim) -> (seq, dim*heads)
        attn_out = ttnn.permute(attn_out, [1, 2, 0, 3])  # (1, 576, 16, 64)
        attn_2d = ttnn.reshape(attn_out, [VAE_SEQ_LEN, VAE_DEC_DIM])

        # Out proj + bias
        ttnn.linear(attn_2d, vae_dev["%s.proj_w" % p], bias=vae_dev["%s.proj_b" % p],
                    optional_output_tensor=vae_scr["o_proj"])

        # Residual: z_b = z_a + o_proj
        ttnn.add(vae_scr["z_a"], vae_scr["o_proj"], output_tensor=vae_scr["z_b"])

        # === MLP path: x + MLP(LN(x)) ===
        layernorm_d1024(vae_scr["z_b"], vae_dev["%s.norm2_w" % p], vae_dev["%s.norm2_b" % p],
                        scaler, mean_scale, vae_scr["normed"])

        # Fused FC1 + bias + GELU
        ttnn.linear(vae_scr["normed"], vae_dev["%s.fc1_w" % p],
                    bias=vae_dev["%s.fc1_b" % p], activation="gelu",
                    optional_output_tensor=vae_scr["gelu"])

        # FC2 + bias
        ttnn.linear(vae_scr["gelu"], vae_dev["%s.fc2_w" % p],
                    bias=vae_dev["%s.fc2_b" % p],
                    optional_output_tensor=vae_scr["fc2"])

        # Residual: z_a = z_b + fc2
        ttnn.add(vae_scr["z_b"], vae_scr["fc2"], output_tensor=vae_scr["z_a"])

    # Final LN
    layernorm_d1024(vae_scr["z_a"], vae_dev["dec_norm_w"], vae_dev["dec_norm_b"],
                    scaler, mean_scale, vae_scr["normed"])

    # Predictor: (576, 1024) @ (1024, 1200) + bias
    ttnn.linear(vae_scr["normed"], vae_dev["pred_w"], bias=vae_dev["pred_b"],
                optional_output_tensor=vae_scr["pred_out"])
    return vae_scr["pred_out"]

def vae_decode_device(z_flat, vae_dev, vae_scr, tt_device, scaler, mean_scale):
    """Decode latent on TT device. 12 transformer blocks with 2D RoPE.
    z_flat: (576, 16) torch bf16 - flattened latent (already divided by scaling factor).
    Returns: (576, 1200) torch float - patch predictions for unpatchify on host.
    """
    # Pad latent from 16 to 32 cols and copy into pre-allocated buffer
    z_padded = torch.zeros(VAE_SEQ_LEN, 32, dtype=torch.bfloat16)
    z_padded[:, :VAE_LATENT_DIM] = z_flat.to(torch.bfloat16)
    z_host = ttnn.from_torch(z_padded, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    ttnn.copy_host_to_device_tensor(z_host, vae_scr["vae_input"])

    vae_decode_forward(vae_dev, vae_scr, scaler, mean_scale)

    result = readback_torch(vae_scr["pred_out"]).float()
    return result[:, :VAE_PATCH_DIM]  # strip tile padding (1216 -> 1200)

def vae_unpatchify(patches):
    """Convert VAE patch predictions to image.
    patches: (576, 1200) float tensor.
    Returns: (1, 3, 360, 640) float tensor in [-1, 1].
    """
    x = patches.reshape(1, VAE_SEQ_H, VAE_SEQ_W, VAE_PATCH_DIM)
    x = x.permute(0, 3, 1, 2)  # (1, 1200, 18, 32)
    x = x.reshape(1, 3, VAE_PATCH_SIZE_VAE, VAE_PATCH_SIZE_VAE, VAE_SEQ_H, VAE_SEQ_W)
    x = x.permute(0, 1, 4, 2, 5, 3)  # (1, 3, 18, 20, 32, 20)
    return x.reshape(1, 3, 360, 640)

# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    # Multi-chip: enable fabric for inter-chip collectives
    if N_CHIPS > 1:
        ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
        tt_device = ttnn.open_mesh_device(ttnn.MeshShape(1, N_CHIPS),
                                           trace_region_size=100000000)
        _MESH_DEVICE = tt_device
    else:
        tt_device = ttnn.open_device(device_id=0, trace_region_size=100000000)
    init_compute_configs(tt_device)
    torch.manual_seed(42)

    print("=" * 60)
    print("Oasis-500M DiT Inference on Tenstorrent")
    print("=" * 60)

    # L1 constants
    scaler = to_tt_l1(torch.ones(TILE, TILE, dtype=torch.bfloat16), tt_device)
    mean_scale = to_tt_l1(torch.full((TILE, TILE), 1.0 / D_MODEL, dtype=torch.bfloat16), tt_device)

    N_FRAMES = 2  # sliding window: T-1 context + 1 generated

    # Load VAE (CPU for encoding, device for decoding)
    print("\nLoading VAE...")
    vae = load_vae_cpu()
    print("Loading VAE decoder weights onto device...")
    vae_dev = preload_vae_decoder_weights(vae, tt_device)
    vae_scr = prealloc_vae_scratch(tt_device)

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
    ddim_steps = 4
    noise_range = torch.linspace(-1, max_noise_level - 1, ddim_steps + 1)
    noise_abs_max = 20
    stabilization_level = 15

    betas = sigmoid_beta_schedule(max_noise_level).float()
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)  # (1000,)

    # Load real sample actions for better quality
    actions_path = "/tmp/sample_actions_0.one_hot_actions.pt"
    if os.path.exists(actions_path):
        sample_actions = torch.load(actions_path, weights_only=True)  # (T, 25)
        # Prepend zeros for the prompt frame (reference does this)
        sample_actions = torch.cat([torch.zeros_like(sample_actions[:1]), sample_actions], dim=0)
        print("Loaded %d sample actions from %s" % (sample_actions.shape[0], actions_path))
    else:
        print("WARNING: sample actions not found at %s, using zero actions" % actions_path)
        sample_actions = torch.zeros(960, EXT_COND_DIM)
    # Frame 0 = prompt (always zero action), frame 1+ = real actions
    prompt_action = torch.zeros(EXT_COND_DIM)

    # === Pre-compute device-resident data for on-device DDIM loop ===

    # 1. Prompt frame: patch embed once, keep on device permanently
    prompt_z_pad = torch.zeros(N_PATCH_PAD, D_MODEL, dtype=torch.bfloat16)
    prompt_z_pad[:N_PATCHES] = patch_embed_host(
        prompt_latent[0, 0].unsqueeze(0), dev["x_emb_conv_w"], dev["x_emb_conv_b"])
    prompt_z_dev = to_tt(prompt_z_pad, tt_device)

    # 2. Prompt conditioning (constant across all DDIM steps)
    prompt_cond_dev = compute_cond_for_frame(
        stabilization_level - 1, prompt_action, dev, scr, tt_device)

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

    # Default: zero action for warmup and single-frame test
    print("Pre-computing conditioning for %d steps..." % ddim_steps)
    t_precond = time.time()
    gen_cond_per_step = precompute_gen_cond(prompt_action, ddim_steps, noise_range, dev, scr, tt_device)
    print("Pre-computed conditioning in %.1fs" % (time.time() - t_precond))

    # Pre-compute DDIM scalar coefficients as host tensors for trace-compatible updates
    CHUNK_SHAPE = (N_PATCH_PAD, OUT_DIM)
    ddim_coeffs_host = {}
    for noise_idx in reversed(range(1, ddim_steps + 1)):
        noise_level = int(noise_range[noise_idx].item())
        t_next_noise = int(noise_range[noise_idx - 1].item())
        at = float(alphas_cumprod[noise_level].item())
        an = float(alphas_cumprod[max(t_next_noise, 0)].item())
        if noise_idx == 1:
            an = 1.0
        ddim_coeffs_host[noise_idx] = {
            "at_sqrt": torch.full(CHUNK_SHAPE, at ** 0.5, dtype=torch.bfloat16),
            "1mat_sqrt": torch.full(CHUNK_SHAPE, (1 - at) ** 0.5, dtype=torch.bfloat16),
            "inv_at_sqrt": torch.full(CHUNK_SHAPE, (1.0 / at) ** 0.5, dtype=torch.bfloat16),
            "inv_sigma": torch.full(CHUNK_SHAPE, 1.0 / ((1.0 / at - 1) ** 0.5), dtype=torch.bfloat16),
            "an_sqrt": torch.full(CHUNK_SHAPE, an ** 0.5, dtype=torch.bfloat16),
            "1man_sqrt": torch.full(CHUNK_SHAPE, (1 - an) ** 0.5, dtype=torch.bfloat16),
        }
    # Convert to tilized host tensors (ready for copy_host_to_device_tensor)
    for noise_idx in ddim_coeffs_host:
        for k in ddim_coeffs_host[noise_idx]:
            ddim_coeffs_host[noise_idx][k] = ttnn.from_torch(
                ddim_coeffs_host[noise_idx][k], dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)

    # Device coefficient tensors (updated before each traced step)
    ddim_coeff_dev = {}
    for k in ["at_sqrt", "1mat_sqrt", "inv_at_sqrt", "inv_sigma", "an_sqrt", "1man_sqrt"]:
        ddim_coeff_dev[k] = to_tt(torch.zeros(*CHUNK_SHAPE, dtype=torch.bfloat16), tt_device)

    # Mutable conditioning tensors for trace (one per frame in window, updated before each step)
    cond_traced = []
    for f in range(N_FRAMES):
        cond_traced.append(to_tt(torch.zeros(TILE, D_MODEL, dtype=torch.bfloat16), tt_device))

    # Context frames: T-1 frames of patch-embedded latents (updated between frames)
    context_z_dev = to_tt(torch.zeros((N_FRAMES - 1) * N_PATCH_PAD, D_MODEL, dtype=torch.bfloat16), tt_device)

    # Traced chunk buffer: the trace operates on this tensor in-place
    trace_chunk = to_tt(torch.zeros(*CHUNK_SHAPE, dtype=torch.bfloat16), tt_device)

    def ddim_step_fn(chunk):
        """One DDIM step: round-trip + DiT forward + DDIM arithmetic.
        Uses cond_traced and ddim_coeff_dev which are updated before each call."""
        gen_z = ttnn.linear(chunk, W_rt_dev, bias=b_rt_dev)
        z_cur = ttnn.concat([context_z_dev, gen_z], dim=0)

        final_out = dit_forward_device(z_cur, cond_traced, dev, scr, tt_device, scaler, mean_scale)

        # Extract velocity for the last frame (the one being generated)
        gen_start = (N_FRAMES - 1) * N_PATCH_PAD
        v_dev = ttnn.slice(final_out, [gen_start, 0], [gen_start + N_PATCH_PAD, OUT_DIM])
        # x_start = at_sqrt * chunk - (1-at)_sqrt * v
        ttnn.multiply(chunk, ddim_coeff_dev["at_sqrt"], output_tensor=scr["ddim_tmp"])
        ttnn.multiply(v_dev, ddim_coeff_dev["1mat_sqrt"], output_tensor=scr["ddim_x_start"])
        ttnn.subtract(scr["ddim_tmp"], scr["ddim_x_start"], output_tensor=scr["ddim_x_start"])
        # x_noise = (inv_at_sqrt * chunk - x_start) / sigma
        ttnn.multiply(chunk, ddim_coeff_dev["inv_at_sqrt"], output_tensor=scr["ddim_tmp"])
        ttnn.subtract(scr["ddim_tmp"], scr["ddim_x_start"], output_tensor=scr["ddim_x_noise"])
        ttnn.multiply(scr["ddim_x_noise"], ddim_coeff_dev["inv_sigma"], output_tensor=scr["ddim_x_noise"])
        # chunk = an_sqrt * x_start + (1-an)_sqrt * x_noise
        ttnn.multiply(scr["ddim_x_start"], ddim_coeff_dev["an_sqrt"], output_tensor=scr["ddim_tmp"])
        ttnn.multiply(scr["ddim_x_noise"], ddim_coeff_dev["1man_sqrt"], output_tensor=scr["ddim_x_noise"])
        ttnn.add(scr["ddim_tmp"], scr["ddim_x_noise"], output_tensor=chunk)
        return chunk

    # === Trace capture ===
    print("Compiling DDIM step...")
    t_compile = time.time()
    first_noise_idx = ddim_steps
    for k in ddim_coeff_dev:
        ttnn.copy_host_to_device_tensor(ddim_coeffs_host[first_noise_idx][k], ddim_coeff_dev[k])
    for f in range(N_FRAMES - 1):
        prompt_cond_host = ttnn.from_torch(
            readback_torch(prompt_cond_dev),
            dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
        ttnn.copy_host_to_device_tensor(prompt_cond_host, cond_traced[f])
    first_cond_host = ttnn.from_torch(
        readback_torch(gen_cond_per_step[first_noise_idx]),
        dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    ttnn.copy_host_to_device_tensor(first_cond_host, cond_traced[N_FRAMES - 1])
    context_host = ttnn.from_torch(
        torch.cat([readback_torch(prompt_z_dev)] * (N_FRAMES - 1), dim=0),
        dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    ttnn.copy_host_to_device_tensor(context_host, context_z_dev)
    ddim_step_fn(trace_chunk)
    ttnn.synchronize_device(tt_device)
    print("Compile done in %.1fs" % (time.time() - t_compile))
    print("Capturing trace...")
    t_trace = time.time()
    trace_id = ttnn.begin_trace_capture(tt_device, cq_id=0)
    traced_output = ddim_step_fn(trace_chunk)
    ttnn.end_trace_capture(tt_device, trace_id, cq_id=0)
    ttnn.synchronize_device(tt_device)
    print("Trace captured in %.1fs" % (time.time() - t_trace))
    gen_cond_host = {}
    for noise_idx in gen_cond_per_step:
        gen_cond_host[noise_idx] = ttnn.from_torch(
            readback_torch(gen_cond_per_step[noise_idx]),
            dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    prompt_cond_host = ttnn.from_torch(
        readback_torch(prompt_cond_dev),
        dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)

    def run_ddim_loop(chunk_in_host, label, context_cond_hosts=None, gen_cond_map=None):
        if gen_cond_map is None:
            gen_cond_map = gen_cond_host
        if context_cond_hosts is None:
            context_cond_hosts = [prompt_cond_host] * (N_FRAMES - 1)
        ttnn.copy_host_to_device_tensor(chunk_in_host, trace_chunk)
        for f in range(N_FRAMES - 1):
            ttnn.copy_host_to_device_tensor(context_cond_hosts[f], cond_traced[f])
        t_total = time.time()
        for noise_idx in reversed(range(1, ddim_steps + 1)):
            for k in ddim_coeff_dev:
                ttnn.copy_host_to_device_tensor(ddim_coeffs_host[noise_idx][k], ddim_coeff_dev[k])
            ttnn.copy_host_to_device_tensor(gen_cond_map[noise_idx], cond_traced[N_FRAMES - 1])
            ttnn.execute_trace(tt_device, trace_id, cq_id=0, blocking=False)
        ttnn.synchronize_device(tt_device)
        total_time = time.time() - t_total
        print("%s: %.3fs total (%.0fms/step, %.2f FPS)" % (
            label, total_time, total_time * 1000 / ddim_steps, ddim_steps / total_time))
        return trace_chunk

    chunk_host_tilized = ttnn.from_torch(chunk_pad, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    print("\n=== WARMUP (traced execution, %d chips) ===" % N_CHIPS)
    _ = run_ddim_loop(chunk_host_tilized, "Warmup")
    print("\n=== TIMED SINGLE FRAME ===")
    result_dev = run_ddim_loop(chunk_host_tilized, "Timed")

    # === Generate video ===
    # Sliding window: T-1 context frames + 1 generated frame.
    # Generated frame's latent feeds into context window for next frame.
    # VAE decode happens once at the end for all frames.
    N_VIDEO_FRAMES = 30
    print("\n=== GENERATING %d-FRAME VIDEO (T=%d window) ===" % (N_VIDEO_FRAMES, N_FRAMES))
    t_video_start = time.time()

    # Collect all latents for batch VAE decode at the end
    all_latents = [prompt_latent[0, 0].clone()]  # frame 0 = prompt

    # Sliding window of patch-embedded frames (host tensors, each (N_PATCH_PAD, D_MODEL))
    # Start with prompt replicated to fill T-1 context slots
    prompt_z_torch = torch.zeros(N_PATCH_PAD, D_MODEL, dtype=torch.bfloat16)
    prompt_z_torch[:N_PATCHES] = patch_embed_host(
        prompt_latent[0, 0].unsqueeze(0), dev["x_emb_conv_w"], dev["x_emb_conv_b"])
    context_window_z = [prompt_z_torch.clone() for _ in range(N_FRAMES - 1)]

    # Sliding window of per-frame conditioning (host tensors for trace updates)
    context_cond_window = [prompt_cond_host] * (N_FRAMES - 1)

    for frame_idx in range(1, N_VIDEO_FRAMES):
        t_frame = time.time()

        # Fresh noise for generated frame (in output space)
        chunk_img = torch.randn(1, IN_CHANNELS, INPUT_H, INPUT_W)
        chunk_img = torch.clamp(chunk_img, -noise_abs_max, noise_abs_max)
        chunk_patches = patchify_to_output_space(chunk_img)
        chunk_pad_f = torch.zeros(N_PATCH_PAD, OUT_DIM, dtype=torch.bfloat16)
        chunk_pad_f[:N_PATCHES] = chunk_patches.to(torch.bfloat16)
        chunk_host_f = ttnn.from_torch(chunk_pad_f, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)

        # Update context_z_dev with current sliding window
        context_cat = torch.cat(context_window_z, dim=0)  # ((T-1)*N_PATCH_PAD, D_MODEL)
        context_host = ttnn.from_torch(context_cat, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
        ttnn.copy_host_to_device_tensor(context_host, context_z_dev)

        # Pre-compute gen conditioning with this frame's action
        action_idx = min(frame_idx, sample_actions.shape[0] - 1)
        frame_cond_dev = precompute_gen_cond(sample_actions[action_idx], ddim_steps, noise_range, dev, scr, tt_device)

        frame_cond_host = {}
        for noise_idx in frame_cond_dev:
            frame_cond_host[noise_idx] = ttnn.from_torch(
                readback_torch(frame_cond_dev[noise_idx]),
                dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
        result_dev = run_ddim_loop(chunk_host_f, "Frame %d" % frame_idx,
                                   context_cond_hosts=context_cond_window,
                                   gen_cond_map=frame_cond_host)

        # Readback generated latent (output space -> image space)
        chunk_result = readback_torch(result_dev)[:N_PATCHES].float()
        gen_latent = unpatchify_host(chunk_result, PATCH_SIZE, IN_CHANNELS, FRAME_H, FRAME_W)
        all_latents.append(gen_latent.squeeze(0))  # (C, H, W)

        # Patch-embed generated frame for context window
        new_z = torch.zeros(N_PATCH_PAD, D_MODEL, dtype=torch.bfloat16)
        new_z[:N_PATCHES] = patch_embed_host(
            gen_latent, dev["x_emb_conv_w"], dev["x_emb_conv_b"])

        # Slide window: drop oldest, append newest
        context_window_z.pop(0)
        context_window_z.append(new_z)

        # Slide conditioning: context frames all use stabilization_level
        new_context_cond = compute_cond_for_frame(
            stabilization_level - 1, sample_actions[action_idx], dev, scr, tt_device)
        new_cond_entry = ttnn.from_torch(
            readback_torch(new_context_cond), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
        context_cond_window.pop(0)
        context_cond_window.append(new_cond_entry)

        elapsed = time.time() - t_frame
        print("  Frame %d: %.2fs" % (frame_idx, elapsed))

    total_video = time.time() - t_video_start
    print("\nDiT generation: %.1fs for %d frames (%.2f FPS)" % (
        total_video, N_VIDEO_FRAMES - 1, (N_VIDEO_FRAMES - 1) / total_video))

    # VAE decode all frames on device (traced)
    print("\nVAE: compiling forward pass...")
    t_vae_compile = time.time()
    # Warmup compile with dummy input
    dummy_z = torch.zeros(VAE_SEQ_LEN, 32, dtype=torch.bfloat16)
    dummy_host = ttnn.from_torch(dummy_z, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    ttnn.copy_host_to_device_tensor(dummy_host, vae_scr["vae_input"])
    vae_decode_forward(vae_dev, vae_scr, scaler, mean_scale)
    ttnn.synchronize_device(tt_device)
    print("VAE compile: %.1fs" % (time.time() - t_vae_compile))

    # Capture VAE trace
    print("VAE: capturing trace...")
    t_vae_trace = time.time()
    vae_trace_id = ttnn.begin_trace_capture(tt_device, cq_id=0)
    vae_decode_forward(vae_dev, vae_scr, scaler, mean_scale)
    ttnn.end_trace_capture(tt_device, vae_trace_id, cq_id=0)
    ttnn.synchronize_device(tt_device)
    print("VAE trace captured in %.1fs" % (time.time() - t_vae_trace))

    print("\nVAE decoding %d frames on device (traced)..." % len(all_latents))
    t_vae = time.time()
    all_decoded_frames = []
    for fi, lat in enumerate(all_latents):
        # lat: (C, H, W) = (16, 18, 32) -> flatten to (576, 16)
        z_flat = rearrange(lat.float(), "c h w -> (h w) c") / SCALING_FACTOR
        z_padded = torch.zeros(VAE_SEQ_LEN, 32, dtype=torch.bfloat16)
        z_padded[:, :VAE_LATENT_DIM] = z_flat.to(torch.bfloat16)
        z_host = ttnn.from_torch(z_padded, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
        ttnn.copy_host_to_device_tensor(z_host, vae_scr["vae_input"])
        ttnn.execute_trace(tt_device, vae_trace_id, cq_id=0, blocking=False)
        ttnn.synchronize_device(tt_device)
        result = readback_torch(vae_scr["pred_out"]).float()
        patches = result[:, :VAE_PATCH_DIM]
        image = vae_unpatchify(patches)  # (1, 3, 360, 640) in [-1, 1]
        decoded = torch.clamp((image + 1) / 2, 0, 1)  # [0, 1]
        # (1, 3, H, W) -> (H, W, 3)
        all_decoded_frames.append(decoded.squeeze(0).permute(1, 2, 0))
    vae_time = time.time() - t_vae
    print("VAE decode: %.2fs (%.0fms/frame)" % (vae_time, vae_time * 1000 / len(all_latents)))

    # Save as individual PNGs
    video_dir = "/tmp/oasis_video_%dstep_T%d" % (ddim_steps, N_FRAMES)
    os.makedirs(video_dir, exist_ok=True)
    for i, frame_hwc in enumerate(all_decoded_frames):
        frame_rgb = (frame_hwc * 255).byte().numpy()
        Image.fromarray(frame_rgb).save("%s/frame_%04d.png" % (video_dir, i))
    print("Saved %d frames to %s/" % (len(all_decoded_frames), video_dir))

    # Save first generated frame
    first_gen = (all_decoded_frames[1] * 255).byte().numpy()
    Image.fromarray(first_gen).save("/tmp/oasis_frame.png")

    # Stitch frames into mp4 with ffmpeg
    import subprocess
    mp4_path = "%s/output.mp4" % video_dir
    ffmpeg_cmd = [
        "ffmpeg", "-y",
        "-framerate", "20",
        "-i", "%s/frame_%%04d.png" % video_dir,
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-crf", "18",
        mp4_path
    ]
    result = subprocess.run(ffmpeg_cmd, capture_output=True, text=True)
    if result.returncode == 0:
        print("Saved video to %s" % mp4_path)
    else:
        print("ffmpeg failed: %s" % result.stderr[-200:])

    print("\nDone!")
    if isinstance(tt_device, ttnn.MeshDevice):
        ttnn.close_mesh_device(tt_device)
    else:
        ttnn.close_device(tt_device)
