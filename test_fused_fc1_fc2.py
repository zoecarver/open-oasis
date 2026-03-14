"""
Test fused FC1+GELU+FC2+bias+gate+residual kernel.
Eliminates the GELU DRAM intermediate by recomputing FC1+GELU per FC2 output col.

The modulated input (from LN+adaLN) is read from DRAM. For each FC2 output col,
the modulated row is held in scope and FC1+GELU is computed on-the-fly for each
FC2 K chunk. The GELU intermediate never touches DRAM.

Pattern from md_cell_list.py: hold data in scope via nested `with` blocks
and reuse across iterations (like `cg` in xy_conv_kernel).
"""
import torch
import torch.nn as nn
import ttnn
import ttl
import time

TILE = 32
D_MODEL = 1024
D_MLP = 4096
D_TILES = D_MODEL // TILE  # 32
D_MLP_TILES = D_MLP // TILE  # 128
FC2_K_CHUNK = 4  # tiles per FC2 K chunk (smaller to fit L1)
FC2_K_ITERS = D_MLP_TILES // FC2_K_CHUNK  # 32


def make_fused_fc1_gelu_fc2_gate_res_kernel(dim_tiles, fc2_k_chunk, fc2_k_iters):
    """Fused: out = residual + gate * (fc2(gelu(modulated @ fc1_w + fc1_b)) + fc2_b)

    For each output tile (row, col):
      Hold modulated[row, :] as (1, dim_tiles) in scope.
      For each FC2 K chunk: FC1 matmul + GELU -> FC2 partial accumulate.
      Then add bias, multiply gate, add residual.

    Eliminates GELU DRAM write (2.5MB) + read (2.5MB) = 5MB per sub-block.
    Cost: re-reads modulated row and FC1 weights per FC2 output col.
    """
    @ttl.kernel(grid="auto")
    def kernel(modulated, fc1_w, fc1_b, fc2_w, fc2_b, gate, residual, out):
        grid_cols, _ = ttl.grid_size(dims=2)
        seq_tiles = modulated.shape[0] // TILE
        n_out_tiles = fc2_w.shape[1] // TILE
        # Each core processes seq rows. Within each row, we iterate over output cols.
        tiles_per_core = -(-seq_tiles // grid_cols)

        # Input DFBs
        mod_dfb = ttl.make_dataflow_buffer_like(modulated, shape=(1, dim_tiles), buffer_factor=2)
        fc1w_dfb = ttl.make_dataflow_buffer_like(fc1_w, shape=(dim_tiles, fc2_k_chunk), buffer_factor=2)
        fc1b_dfb = ttl.make_dataflow_buffer_like(fc1_b, shape=(1, fc2_k_chunk), buffer_factor=2)
        fc2w_dfb = ttl.make_dataflow_buffer_like(fc2_w, shape=(fc2_k_chunk, 1), buffer_factor=2)
        fc2b_dfb = ttl.make_dataflow_buffer_like(fc2_b, shape=(1, 1), buffer_factor=2)
        g_dfb = ttl.make_dataflow_buffer_like(gate, shape=(1, 1), buffer_factor=2)
        r_dfb = ttl.make_dataflow_buffer_like(residual, shape=(1, 1), buffer_factor=2)

        # Intermediate DFBs
        fc1out_dfb = ttl.make_dataflow_buffer_like(fc1_b, shape=(1, fc2_k_chunk), buffer_factor=2)
        gelu_dfb = ttl.make_dataflow_buffer_like(fc1_b, shape=(1, fc2_k_chunk), buffer_factor=2)
        mm_dfb = ttl.make_dataflow_buffer_like(fc2_b, shape=(1, 1), buffer_factor=2)
        fc2acc_dfb = ttl.make_dataflow_buffer_like(fc2_b, shape=(1, 1), buffer_factor=2)
        gb_dfb = ttl.make_dataflow_buffer_like(fc2_b, shape=(1, 1), buffer_factor=2)
        out_dfb = ttl.make_dataflow_buffer_like(out, shape=(1, 1), buffer_factor=2)

        @ttl.compute()
        def compute():
            core_x, _ = ttl.core(dims=2)
            for local_t in range(tiles_per_core):
                row = core_x * tiles_per_core + local_t
                if row < seq_tiles:
                    for out_col in range(n_out_tiles):
                        # Hold modulated row in scope across all K chunks
                        with mod_dfb.wait() as modv:
                            # First K chunk: FC1 matmul -> +bias+GELU -> FC2 -> init acc
                            with fc1w_dfb.wait() as fw, fc1out_dfb.reserve() as fc1o:
                                fc1o.store(modv @ fw)
                            with fc1out_dfb.wait() as fc1v, fc1b_dfb.wait() as fb, gelu_dfb.reserve() as gl:
                                h = fc1v + fb
                                x3 = h * h * h
                                inner = ttl.math.fill(h, 0.7978845608) * (h + ttl.math.fill(h, 0.044715) * x3)
                                gl.store(ttl.math.fill(h, 0.5) * h * (ttl.math.fill(h, 1.0) + ttl.math.tanh(inner)))
                            with gelu_dfb.wait() as glv, fc2w_dfb.wait() as f2w, fc2acc_dfb.reserve() as acc:
                                acc.store(glv @ f2w)

                            # Remaining K chunks: accumulate
                            for ki in range(fc2_k_iters - 1):
                                with fc1w_dfb.wait() as fw, fc1out_dfb.reserve() as fc1o:
                                    fc1o.store(modv @ fw)
                                with fc1out_dfb.wait() as fc1v, fc1b_dfb.wait() as fb, gelu_dfb.reserve() as gl:
                                    h = fc1v + fb
                                    x3 = h * h * h
                                    inner = ttl.math.fill(h, 0.7978845608) * (h + ttl.math.fill(h, 0.044715) * x3)
                                    gl.store(ttl.math.fill(h, 0.5) * h * (ttl.math.fill(h, 1.0) + ttl.math.tanh(inner)))
                                with gelu_dfb.wait() as glv, fc2w_dfb.wait() as f2w, mm_dfb.reserve() as m:
                                    m.store(glv @ f2w)
                                with mm_dfb.wait() as mv, fc2acc_dfb.wait() as av, fc2acc_dfb.reserve() as acc:
                                    acc.store(av + mv)

                        # (acc + bias) * gate + residual
                        with fc2acc_dfb.wait() as final, fc2b_dfb.wait() as b2, gb_dfb.reserve() as gb:
                            gb.store(final + b2)
                        with gb_dfb.wait() as gbv, g_dfb.wait() as gv, r_dfb.wait() as rv, out_dfb.reserve() as o:
                            o.store(rv + gbv * gv)

        @ttl.datamovement()
        def dm_read():
            core_x, _ = ttl.core(dims=2)
            for local_t in range(tiles_per_core):
                row = core_x * tiles_per_core + local_t
                if row < seq_tiles:
                    for out_col in range(n_out_tiles):
                        # Modulated row: (1, dim_tiles) - re-read per output col
                        with mod_dfb.reserve() as blk:
                            tx = ttl.copy(modulated[row, 0:dim_tiles], blk); tx.wait()
                        # FC1 + FC2 weights for each K chunk
                        for ki in range(fc2_k_iters):
                            k_start = ki * fc2_k_chunk
                            with fc1w_dfb.reserve() as blk:
                                tx = ttl.copy(fc1_w[0:dim_tiles, k_start:k_start + fc2_k_chunk], blk); tx.wait()
                            with fc1b_dfb.reserve() as blk:
                                tx = ttl.copy(fc1_b[row, k_start:k_start + fc2_k_chunk], blk); tx.wait()
                            with fc2w_dfb.reserve() as blk:
                                tx = ttl.copy(fc2_w[k_start:k_start + fc2_k_chunk, out_col], blk); tx.wait()
                        # Bias + gate + residual
                        with fc2b_dfb.reserve() as blk:
                            tx = ttl.copy(fc2_b[row, out_col], blk); tx.wait()
                        with g_dfb.reserve() as blk:
                            tx = ttl.copy(gate[row, out_col], blk); tx.wait()
                        with r_dfb.reserve() as blk:
                            tx = ttl.copy(residual[row, out_col], blk); tx.wait()

        @ttl.datamovement()
        def dm_write():
            core_x, _ = ttl.core(dims=2)
            for local_t in range(tiles_per_core):
                row = core_x * tiles_per_core + local_t
                if row < seq_tiles:
                    for out_col in range(n_out_tiles):
                        with out_dfb.wait() as blk:
                            tx = ttl.copy(blk, out[row, out_col]); tx.wait()

    return kernel


def to_tt(t, device):
    return ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                           device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)

def expand_bias(bias_1d, seq_pad):
    return bias_1d.unsqueeze(0).expand(seq_pad, -1).contiguous().to(torch.bfloat16)


if __name__ == "__main__":
    device = ttnn.open_device(device_id=0)
    torch.manual_seed(42)

    TEST_SEQ = 320  # Match inference: N_PATCH_PAD * 2 frames
    print("Testing fused FC1+GELU+FC2+gate+res (SEQ=%d, D=%d, MLP=%d)" % (TEST_SEQ, D_MODEL, D_MLP))

    # Inputs
    modulated = torch.randn(TEST_SEQ, D_MODEL, dtype=torch.float32) * 0.5
    gate_vals = torch.randn(TEST_SEQ, D_MODEL, dtype=torch.float32) * 0.1 + 0.5
    residual = torch.randn(TEST_SEQ, D_MODEL, dtype=torch.float32) * 0.5
    fc1_w = torch.randn(D_MODEL, D_MLP, dtype=torch.float32) * 0.02
    fc1_b_1d = torch.randn(D_MLP, dtype=torch.float32) * 0.01
    fc2_w = torch.randn(D_MLP, D_MODEL, dtype=torch.float32) * 0.02
    fc2_b_1d = torch.randn(D_MODEL, dtype=torch.float32) * 0.01

    # PyTorch reference
    fc1_out = modulated @ fc1_w + fc1_b_1d
    gelu_out = 0.5 * fc1_out * (1 + torch.tanh(0.7978845608 * (fc1_out + 0.044715 * fc1_out ** 3)))
    fc2_out = gelu_out @ fc2_w + fc2_b_1d
    ref_out = residual + gate_vals * fc2_out
    print("Reference range: [%.4f, %.4f]" % (ref_out.min().item(), ref_out.max().item()))

    # Device tensors
    mod_tt = to_tt(modulated.to(torch.bfloat16), device)
    gate_tt = to_tt(gate_vals.to(torch.bfloat16), device)
    res_tt = to_tt(residual.to(torch.bfloat16), device)
    fc1_w_tt = to_tt(fc1_w.to(torch.bfloat16), device)
    fc1_b_tt = to_tt(expand_bias(fc1_b_1d.to(torch.bfloat16), TEST_SEQ), device)
    fc2_w_tt = to_tt(fc2_w.to(torch.bfloat16), device)
    fc2_b_tt = to_tt(expand_bias(fc2_b_1d.to(torch.bfloat16), TEST_SEQ), device)
    out_tt = to_tt(torch.zeros(TEST_SEQ, D_MODEL, dtype=torch.bfloat16), device)

    print("Building kernel...")
    fused_k = make_fused_fc1_gelu_fc2_gate_res_kernel(D_TILES, FC2_K_CHUNK, FC2_K_ITERS)

    # Warmup run (compilation)
    print("Warmup run (compilation)...")
    fused_k(mod_tt, fc1_w_tt, fc1_b_tt, fc2_w_tt, fc2_b_tt, gate_tt, res_tt, out_tt)

    # Timed run
    print("Running on HW (timed)...")
    ttnn.synchronize_device(device)
    t0 = time.time()
    fused_k(mod_tt, fc1_w_tt, fc1_b_tt, fc2_w_tt, fc2_b_tt, gate_tt, res_tt, out_tt)
    ttnn.synchronize_device(device)
    elapsed = time.time() - t0
    print("Kernel time: %.3fs (%.1fms)" % (elapsed, elapsed * 1000))

    result = ttnn.to_torch(out_tt).float()
    diff = (ref_out - result).abs()
    max_err = diff.max().item()
    mean_err = diff.mean().item()
    ref_range = ref_out.max().item() - ref_out.min().item()
    print("Max error: %.4f, Mean: %.4f, Rel: %.4f" % (max_err, mean_err, max_err / (ref_range + 1e-8)))
    print("Result range: [%.4f, %.4f]" % (result.min().item(), result.max().item()))

    if max_err < 10.0:
        print("PASS")
    else:
        print("FAIL")

    ttnn.close_device(device)
