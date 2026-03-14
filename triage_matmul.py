"""
Isolate matmul precision: test QKV linear (160x1024 @ 1024x3072) and
MLP fc2 (160x4096 @ 4096x1024) with known inputs to measure bf16 error.
"""
import sys, types

def _make_stub(name, parent=None):
    m = types.ModuleType(name)
    sys.modules[name] = m
    if parent is not None:
        setattr(parent, name.split('.')[-1], m)
    return m

import torch
import torch.nn as nn

_timm = _make_stub('timm')
_timm_models = _make_stub('timm.models', _timm)
_timm_models_vit = _make_stub('timm.models.vision_transformer', _timm_models)
_timm_layers = _make_stub('timm.layers', _timm)
_timm_layers_helpers = _make_stub('timm.layers.helpers', _timm_layers)

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
        x = self.drop(self.act(self.fc1(x)))
        x = self.drop(self.fc2(x))
        return x

_timm_models_vit.Mlp = _Mlp
_timm_layers_helpers.to_2tuple = lambda x: (x, x) if not isinstance(x, (tuple, list)) else tuple(x)

import torch.nn.functional as F
import ttnn
import ttl
import os
from safetensors import safe_open
from einops import rearrange

TILE = 32
D_MODEL = 1024
N_PATCH_PAD = 160
N_PATCHES = 144
D_MLP = 4096
D_TILES = D_MODEL // TILE
ELEM_GRAN = 8

sys.path.insert(0, "/tmp")

def to_tt(t, device):
    return ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                           device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)

def zeros_tt(shape, device):
    return to_tt(torch.zeros(shape, dtype=torch.bfloat16), device)

def compare(name, ref, tt_val):
    ref_f = ref.float()
    tt_f = tt_val.float() if isinstance(tt_val, torch.Tensor) else ttnn.to_torch(tt_val).float()
    min_shape = [min(r, t) for r, t in zip(ref_f.shape, tt_f.shape)]
    for d in range(len(min_shape)):
        ref_f = ref_f.narrow(d, 0, min_shape[d])
        tt_f = tt_f.narrow(d, 0, min_shape[d])
    diff = (ref_f - tt_f).abs()
    max_err = diff.max().item()
    mean_err = diff.mean().item()
    ref_range = ref_f.max().item() - ref_f.min().item()
    print("  %-45s max=%.4f mean=%.4f rel=%.4f ref=[%.2f,%.2f]" % (
        name, max_err, mean_err, max_err / (ref_range + 1e-8),
        ref_f.min().item(), ref_f.max().item()))
    return max_err

if __name__ == "__main__":
    tt_device = ttnn.open_device(device_id=0)
    torch.manual_seed(42)

    from oasis_inference import make_linear_kernel, make_linear_accum_kernel

    linear_k32 = make_linear_kernel(D_TILES)
    linear_accum_k32_4 = make_linear_accum_kernel(D_TILES, 4)

    # Load real weights
    blob_dir = "/root/.cache/huggingface/hub/models--Etched--oasis-500m/blobs/"
    files = sorted(os.listdir(blob_dir))
    dit_path = blob_dir + files[1]

    with safe_open(dit_path, framework="pt") as st:
        qkv_w_fp32 = st.get_tensor("blocks.0.s_attn.to_qkv.weight").T.contiguous()  # (1024, 3072) fp32
        fc1_w_fp32 = st.get_tensor("blocks.0.s_mlp.fc1.weight").T.contiguous()  # (1024, 4096)
        fc2_w_fp32 = st.get_tensor("blocks.0.s_mlp.fc2.weight").T.contiguous()  # (4096, 1024)

    # Create realistic input (from layernorm output range)
    x_input = torch.randn(N_PATCH_PAD, D_MODEL) * 2.0  # typical post-LN range

    print("=" * 60)
    print("TEST 1: QKV linear (160x1024 @ 1024x3072)")
    print("=" * 60)

    # Reference: fp32
    ref_qkv = (x_input.float() @ qkv_w_fp32.float())

    # bf16 on CPU (what bf16 precision gives you)
    bf16_qkv = (x_input.to(torch.bfloat16).float() @ qkv_w_fp32.to(torch.bfloat16).float())

    # TT kernel
    qkv_w_tt = to_tt(qkv_w_fp32.to(torch.bfloat16), tt_device)
    x_tt = to_tt(x_input.to(torch.bfloat16), tt_device)
    out_tt = zeros_tt((N_PATCH_PAD, 3 * D_MODEL), tt_device)
    linear_k32(x_tt, qkv_w_tt, out_tt)

    compare("QKV ref(fp32) vs bf16_cpu", ref_qkv[:N_PATCHES], bf16_qkv[:N_PATCHES])
    compare("QKV ref(fp32) vs TT(bf16)", ref_qkv[:N_PATCHES], ttnn.to_torch(out_tt)[:N_PATCHES])
    compare("QKV bf16_cpu vs TT(bf16)", bf16_qkv[:N_PATCHES], ttnn.to_torch(out_tt)[:N_PATCHES])

    print("\n" + "=" * 60)
    print("TEST 2: MLP fc2 (160x4096 @ 4096x1024) with K-accumulation")
    print("=" * 60)

    gelu_input = torch.randn(N_PATCH_PAD, D_MLP) * 1.5

    ref_fc2 = (gelu_input.float() @ fc2_w_fp32.float())
    bf16_fc2 = (gelu_input.to(torch.bfloat16).float() @ fc2_w_fp32.to(torch.bfloat16).float())

    fc2_w_tt = to_tt(fc2_w_fp32.to(torch.bfloat16), tt_device)
    gelu_tt = to_tt(gelu_input.to(torch.bfloat16), tt_device)
    fc2_out_tt = zeros_tt((N_PATCH_PAD, D_MODEL), tt_device)
    linear_accum_k32_4(gelu_tt, fc2_w_tt, fc2_out_tt)

    compare("FC2 ref(fp32) vs bf16_cpu", ref_fc2[:N_PATCHES], bf16_fc2[:N_PATCHES])
    compare("FC2 ref(fp32) vs TT(bf16)", ref_fc2[:N_PATCHES], ttnn.to_torch(fc2_out_tt)[:N_PATCHES])
    compare("FC2 bf16_cpu vs TT(bf16)", bf16_fc2[:N_PATCHES], ttnn.to_torch(fc2_out_tt)[:N_PATCHES])

    print("\n" + "=" * 60)
    print("TEST 3: ttnn.matmul comparison (does ttnn do better?)")
    print("=" * 60)

    # Compare our kernel vs ttnn.matmul
    ttnn_qkv = ttnn.matmul(x_tt, qkv_w_tt)
    compare("QKV ref(fp32) vs ttnn.matmul", ref_qkv[:N_PATCHES], ttnn.to_torch(ttnn_qkv)[:N_PATCHES])
    compare("QKV TT_kernel vs ttnn.matmul", ttnn.to_torch(out_tt)[:N_PATCHES], ttnn.to_torch(ttnn_qkv)[:N_PATCHES])

    ttnn_fc2 = ttnn.matmul(gelu_tt, fc2_w_tt)
    compare("FC2 ref(fp32) vs ttnn.matmul", ref_fc2[:N_PATCHES], ttnn.to_torch(ttnn_fc2)[:N_PATCHES])
    compare("FC2 TT_kernel vs ttnn.matmul", ttnn.to_torch(fc2_out_tt)[:N_PATCHES], ttnn.to_torch(ttnn_fc2)[:N_PATCHES])

    print("\n" + "=" * 60)
    print("TEST 4: Baseline - what does pure bf16 matmul error look like?")
    print("=" * 60)

    # Small K (should be accurate)
    x_small = torch.randn(N_PATCH_PAD, 32) * 2.0  # K=1 tile
    w_small = torch.randn(32, D_MODEL) * 0.1
    ref_small = x_small.float() @ w_small.float()
    x_small_tt = to_tt(x_small.to(torch.bfloat16), tt_device)
    w_small_tt = to_tt(w_small.to(torch.bfloat16), tt_device)

    linear_k1 = make_linear_kernel(1)
    out_small_tt = zeros_tt((N_PATCH_PAD, D_MODEL), tt_device)
    linear_k1(x_small_tt, w_small_tt, out_small_tt)
    compare("Small matmul (K=32, 1 tile)", ref_small[:N_PATCHES], ttnn.to_torch(out_small_tt)[:N_PATCHES])

    print("\nDone!")
    ttnn.close_device(tt_device)
