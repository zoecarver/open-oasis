from safetensors import safe_open
import os

blob_dir = "/root/.cache/huggingface/hub/models--Etched--oasis-500m/blobs/"
files = sorted(os.listdir(blob_dir))

# Second file is DiT (302 keys)
dit_path = blob_dir + files[1]
with safe_open(dit_path, framework="pt") as st:
    keys = sorted(st.keys())
    # Show non-block keys (top-level model structure)
    non_block = [k for k in keys if not k.startswith("blocks.")]
    print("DiT non-block keys:")
    for k in non_block:
        t = st.get_tensor(k)
        print(" ", k, tuple(t.shape), str(t.dtype))

    # Count per-block keys
    block_counts = {}
    for k in keys:
        if k.startswith("blocks."):
            block_num = k.split(".")[1]
            block_counts[block_num] = block_counts.get(block_num, 0) + 1
    print("\nBlocks:", dict(sorted(block_counts.items())))

# First file is VAE (228 keys)
vae_path = blob_dir + files[0]
with safe_open(vae_path, framework="pt") as st:
    keys = sorted(st.keys())
    non_block = [k for k in keys if not k.startswith("encoder.") and not k.startswith("decoder.")]
    print("\nVAE non-encoder/decoder keys:")
    for k in non_block:
        t = st.get_tensor(k)
        print(" ", k, tuple(t.shape), str(t.dtype))
