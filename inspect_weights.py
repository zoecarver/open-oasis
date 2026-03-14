from safetensors import safe_open
import os

blob_dir = "/root/.cache/huggingface/hub/models--Etched--oasis-500m/blobs/"
files = sorted(os.listdir(blob_dir))
for f in files:
    path = blob_dir + f
    try:
        with safe_open(path, framework="pt") as st:
            keys = list(st.keys())
            print("File:", f, "Keys:", len(keys))
            for k in sorted(keys)[:15]:
                t = st.get_tensor(k)
                print(" ", k, tuple(t.shape), str(t.dtype))
            if len(keys) > 15:
                print("  ...", len(keys), "total")
            bk = [k for k in keys if k.startswith("blocks.0.")]
            if bk:
                print("  Block 0:")
                for k in sorted(bk):
                    t = st.get_tensor(k)
                    print(" ", k, tuple(t.shape), str(t.dtype))
    except Exception as e:
        print("File:", f, "Error:", e)
    print()
