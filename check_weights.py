from safetensors import safe_open
import torch
import os

blob_dir = "/root/.cache/huggingface/hub/models--Etched--oasis-500m/blobs/"
files = sorted(os.listdir(blob_dir))

# DiT file (302 keys)
dit_path = blob_dir + files[1]
with safe_open(dit_path, framework="pt") as st:
    # Check if key weights are zero
    for key in [
        "final_layer.linear.weight",
        "final_layer.linear.bias",
        "final_layer.adaLN_modulation.1.weight",
        "final_layer.adaLN_modulation.1.bias",
        "blocks.0.s_adaLN_modulation.1.weight",
        "blocks.0.s_adaLN_modulation.1.bias",
        "blocks.0.s_attn.to_qkv.weight",
        "blocks.0.s_mlp.fc1.weight",
    ]:
        t = st.get_tensor(key)
        nz = (t.abs() > 1e-8).sum().item()
        total = t.numel()
        print("%s: shape=%s nonzero=%d/%d max=%.6f" % (key, tuple(t.shape), nz, total, t.abs().max().item()))
