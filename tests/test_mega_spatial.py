"""Test mega-fused QKV matmul + RoPE + SDPA kernel.

Reads pre-modulated input (SEQ, D_MODEL) from DRAM, computes:
  - QKV matmul via K-accumulation per head: mod @ w → Q, K, V, Q_swap, K_swap
  - RoPE: q_roped = Q * cos + Q_swap * sin, same for K
  - SDPA: softmax(Q_roped @ K_roped^T * scale) @ V
  - Writes attention output back to (SEQ, D_MODEL)

All per-head, parallelized across cores. Q/K/V stay in L1 between matmul and SDPA.
"""
import torch
import torch.nn.functional as F
import ttnn
import ttl
import time

TILE = 32
D_MODEL = 1024
N_HEADS = 16
D_HEAD = 64
N_PATCH_PAD = 160
D_TILES = D_MODEL // TILE        # 32
SEQ_TILES = N_PATCH_PAD // TILE   # 5
HEAD_TILES = D_HEAD // TILE       # 2
SCALE = 0.125  # 1/sqrt(64)


def make_qkv_rope_sdpa_kernel(dim_tiles, seq_tiles, head_tiles, n_heads_val, scale_val):
    """Fused QKV matmul + RoPE + SDPA.

    Inputs:
      modulated: (SEQ, D_MODEL) - pre-computed LN + adaLN output
      qkv_w: (D_MODEL, 5*D_MODEL) - combined [Q|K|V|Q_swap|K_swap] weights
      cos_tab, sin_tab: (SEQ, D_MODEL) - RoPE tables
      scaler: (TILE, TILE) - all 1s
      out: (SEQ, D_MODEL) - output

    Per head h, the QKV matmul computes:
      Q_h = modulated @ qkv_w[:, h*head_tiles:(h+1)*head_tiles]
    via K-accumulation: sum over k of mod[:, k] @ w[k, h_col]
    """
    @ttl.kernel(grid="auto")
    def qkv_rope_sdpa(modulated, qkv_w, cos_tab, sin_tab, scaler, out):
        grid_cols, _ = ttl.grid_size(dims=2)
        n_frames = modulated.shape[0] // TILE // seq_tiles
        total_heads = n_frames * n_heads_val
        heads_per_core = -(-total_heads // grid_cols)
        d_tiles = n_heads_val * head_tiles

        # Input streaming DFBs
        mod_dfb = ttl.make_dataflow_buffer_like(modulated, shape=(seq_tiles, 1), buffer_factor=2)
        w_dfb = ttl.make_dataflow_buffer_like(qkv_w, shape=(1, head_tiles), buffer_factor=2)

        # Matmul accumulator and QKV storage
        acc_dfb = ttl.make_dataflow_buffer_like(modulated, shape=(seq_tiles, head_tiles), buffer_factor=2)
        q_dfb = ttl.make_dataflow_buffer_like(modulated, shape=(seq_tiles, head_tiles), buffer_factor=2)
        k_dfb = ttl.make_dataflow_buffer_like(modulated, shape=(seq_tiles, head_tiles), buffer_factor=2)
        v_dfb = ttl.make_dataflow_buffer_like(modulated, shape=(seq_tiles, head_tiles), buffer_factor=2)

        # RoPE
        cos_dfb = ttl.make_dataflow_buffer_like(cos_tab, shape=(seq_tiles, head_tiles), buffer_factor=2)
        sin_dfb = ttl.make_dataflow_buffer_like(sin_tab, shape=(seq_tiles, head_tiles), buffer_factor=2)
        qr_dfb = ttl.make_dataflow_buffer_like(modulated, shape=(seq_tiles, head_tiles), buffer_factor=2)
        kr_dfb = ttl.make_dataflow_buffer_like(modulated, shape=(seq_tiles, head_tiles), buffer_factor=2)

        # Matmul intermediate (for accumulation: can't fuse add + matmul)
        mm_dfb = ttl.make_dataflow_buffer_like(modulated, shape=(seq_tiles, head_tiles), buffer_factor=2)

        # SDPA
        kt_dfb = ttl.make_dataflow_buffer_like(modulated, shape=(head_tiles, seq_tiles), buffer_factor=2)
        a_dfb = ttl.make_dataflow_buffer_like(scaler, shape=(seq_tiles, seq_tiles), buffer_factor=2)
        b_dfb = ttl.make_dataflow_buffer_like(scaler, shape=(seq_tiles, seq_tiles), buffer_factor=2)
        c_dfb = ttl.make_dataflow_buffer_like(scaler, shape=(seq_tiles, seq_tiles), buffer_factor=2)
        row_dfb = ttl.make_dataflow_buffer_like(scaler, shape=(seq_tiles, 1), buffer_factor=2)
        row_bc_dfb = ttl.make_dataflow_buffer_like(scaler, shape=(seq_tiles, seq_tiles), buffer_factor=2)
        scaler_dfb = ttl.make_dataflow_buffer_like(scaler, shape=(1, 1), buffer_factor=2)
        out_dfb = ttl.make_dataflow_buffer_like(out, shape=(seq_tiles, head_tiles), buffer_factor=2)
        # 18 DFBs total

        @ttl.compute()
        def compute():
            core_x, _ = ttl.core(dims=2)
            for local_h in range(heads_per_core):
                head_idx = core_x * heads_per_core + local_h
                if head_idx < total_heads:
                    # ---- Q matmul: K-accumulation over dim_tiles ----
                    with mod_dfb.wait() as m0, w_dfb.wait() as w0, acc_dfb.reserve() as a:
                        a.store(m0 @ w0)
                    for k_idx in range(dim_tiles - 1):
                        with mod_dfb.wait() as mk, w_dfb.wait() as wk, mm_dfb.reserve() as mm:
                            mm.store(mk @ wk)
                        with mm_dfb.wait() as mmv, acc_dfb.wait() as prev, acc_dfb.reserve() as a:
                            a.store(prev + mmv)
                    with acc_dfb.wait() as q_result, q_dfb.reserve() as q:
                        q.store(q_result)

                    # ---- Q_swap matmul → acc_dfb, then RoPE Q ----
                    with mod_dfb.wait() as m0, w_dfb.wait() as w0, acc_dfb.reserve() as a:
                        a.store(m0 @ w0)
                    for k_idx in range(dim_tiles - 1):
                        with mod_dfb.wait() as mk, w_dfb.wait() as wk, mm_dfb.reserve() as mm:
                            mm.store(mk @ wk)
                        with mm_dfb.wait() as mmv, acc_dfb.wait() as prev, acc_dfb.reserve() as a:
                            a.store(prev + mmv)
                    # RoPE Q: q * cos + qs * sin
                    with acc_dfb.wait() as qs, q_dfb.wait() as qv, cos_dfb.wait() as cv, sin_dfb.wait() as sv:
                        with qr_dfb.reserve() as qr:
                            qr.store(qv * cv + qs * sv)

                    # ---- K matmul ----
                    with mod_dfb.wait() as m0, w_dfb.wait() as w0, acc_dfb.reserve() as a:
                        a.store(m0 @ w0)
                    for k_idx in range(dim_tiles - 1):
                        with mod_dfb.wait() as mk, w_dfb.wait() as wk, mm_dfb.reserve() as mm:
                            mm.store(mk @ wk)
                        with mm_dfb.wait() as mmv, acc_dfb.wait() as prev, acc_dfb.reserve() as a:
                            a.store(prev + mmv)
                    with acc_dfb.wait() as k_result, k_dfb.reserve() as k:
                        k.store(k_result)

                    # ---- K_swap matmul → acc_dfb, then RoPE K ----
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

                    # ---- V matmul ----
                    with mod_dfb.wait() as m0, w_dfb.wait() as w0, acc_dfb.reserve() as a:
                        a.store(m0 @ w0)
                    for k_idx in range(dim_tiles - 1):
                        with mod_dfb.wait() as mk, w_dfb.wait() as wk, mm_dfb.reserve() as mm:
                            mm.store(mk @ wk)
                        with mm_dfb.wait() as mmv, acc_dfb.wait() as prev, acc_dfb.reserve() as a:
                            a.store(prev + mmv)
                    with acc_dfb.wait() as v_result, v_dfb.reserve() as v:
                        v.store(v_result)

                    # ---- SDPA ----
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

                    # 5 matmuls in order: Q, Qs, K, Ks, V
                    # Column offsets: Q=hc, Qs=3*d_t+hc, K=d_t+hc, Ks=4*d_t+hc, V=2*d_t+hc

                    # Q matmul
                    for k in range(dim_tiles):
                        with mod_dfb.reserve() as blk:
                            tx = ttl.copy(modulated[row_start:row_start + seq_tiles, k], blk); tx.wait()
                        with w_dfb.reserve() as blk:
                            tx = ttl.copy(qkv_w[k, hc:hc + head_tiles], blk); tx.wait()

                    # Qs matmul
                    qs_col = 3 * d_t + hc
                    for k in range(dim_tiles):
                        with mod_dfb.reserve() as blk:
                            tx = ttl.copy(modulated[row_start:row_start + seq_tiles, k], blk); tx.wait()
                        with w_dfb.reserve() as blk:
                            tx = ttl.copy(qkv_w[k, qs_col:qs_col + head_tiles], blk); tx.wait()
                    # cos/sin for RoPE Q
                    with cos_dfb.reserve() as blk:
                        tx = ttl.copy(cos_tab[row_start:row_start + seq_tiles, hc:hc + head_tiles], blk); tx.wait()
                    with sin_dfb.reserve() as blk:
                        tx = ttl.copy(sin_tab[row_start:row_start + seq_tiles, hc:hc + head_tiles], blk); tx.wait()

                    # K matmul
                    k_col = d_t + hc
                    for k in range(dim_tiles):
                        with mod_dfb.reserve() as blk:
                            tx = ttl.copy(modulated[row_start:row_start + seq_tiles, k], blk); tx.wait()
                        with w_dfb.reserve() as blk:
                            tx = ttl.copy(qkv_w[k, k_col:k_col + head_tiles], blk); tx.wait()

                    # Ks matmul
                    ks_col = 4 * d_t + hc
                    for k in range(dim_tiles):
                        with mod_dfb.reserve() as blk:
                            tx = ttl.copy(modulated[row_start:row_start + seq_tiles, k], blk); tx.wait()
                        with w_dfb.reserve() as blk:
                            tx = ttl.copy(qkv_w[k, ks_col:ks_col + head_tiles], blk); tx.wait()
                    # cos/sin for RoPE K
                    with cos_dfb.reserve() as blk:
                        tx = ttl.copy(cos_tab[row_start:row_start + seq_tiles, hc:hc + head_tiles], blk); tx.wait()
                    with sin_dfb.reserve() as blk:
                        tx = ttl.copy(sin_tab[row_start:row_start + seq_tiles, hc:hc + head_tiles], blk); tx.wait()

                    # V matmul
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


def to_tt(t, device):
    return ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                           device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)

def to_tt_l1(t, device):
    return ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                           device=device, memory_config=ttnn.L1_MEMORY_CONFIG)


if __name__ == "__main__":
    device = ttnn.open_device(device_id=0)
    torch.manual_seed(42)

    T = 1
    SEQ = N_PATCH_PAD * T

    print("Testing mega QKV + RoPE + SDPA kernel")
    print("  SEQ=%d, D_MODEL=%d, N_HEADS=%d" % (SEQ, D_MODEL, N_HEADS))

    # Pre-modulated input (simulates LN + adaLN output)
    modulated = torch.randn(SEQ, D_MODEL, dtype=torch.float32) * 0.3
    # QKV weight: (D_MODEL, 5*D_MODEL) = [Q|K|V|Qs|Ks]
    qkv_w = torch.randn(D_MODEL, 5 * D_MODEL, dtype=torch.float32) * (2.0 / D_MODEL) ** 0.5

    # cos=1, sin=0 → no RoPE effect (identity), easier to validate
    cos_tab = torch.ones(SEQ, D_MODEL, dtype=torch.float32)
    sin_tab = torch.zeros(SEQ, D_MODEL, dtype=torch.float32)

    # PyTorch reference
    qkv_out = modulated.float() @ qkv_w.float()
    ref_heads = []
    for h in range(N_HEADS):
        q_h = qkv_out[:, h * D_HEAD:(h + 1) * D_HEAD]
        k_h = qkv_out[:, D_MODEL + h * D_HEAD:D_MODEL + (h + 1) * D_HEAD]
        v_h = qkv_out[:, 2 * D_MODEL + h * D_HEAD:2 * D_MODEL + (h + 1) * D_HEAD]
        # With cos=1, sin=0: q_roped = q*1 + qs*0 = q (no swap effect)
        scores = (q_h @ k_h.T) * SCALE
        attn = F.softmax(scores, dim=-1)
        ref_heads.append(attn @ v_h)
    ref_out = torch.zeros(SEQ, D_MODEL)
    for h in range(N_HEADS):
        ref_out[:, h * D_HEAD:(h + 1) * D_HEAD] = ref_heads[h]
    print("  Ref range: [%.4f, %.4f]" % (ref_out.min().item(), ref_out.max().item()))

    # Device tensors
    mod_tt = to_tt(modulated.to(torch.bfloat16), device)
    qkv_w_tt = to_tt(qkv_w.to(torch.bfloat16), device)
    cos_tt = to_tt(cos_tab.to(torch.bfloat16), device)
    sin_tt = to_tt(sin_tab.to(torch.bfloat16), device)
    out_tt = to_tt(torch.zeros(SEQ, D_MODEL, dtype=torch.bfloat16), device)
    scaler = to_tt_l1(torch.ones(TILE, TILE, dtype=torch.bfloat16), device)

    print("  Building kernel...")
    mega_k = make_qkv_rope_sdpa_kernel(D_TILES, SEQ_TILES, HEAD_TILES, N_HEADS, SCALE)

    print("  Running...")
    mega_k(mod_tt, qkv_w_tt, cos_tt, sin_tt, scaler, out_tt)

    result = ttnn.to_torch(out_tt).float()

    # Compare per-head
    max_err = 0
    for h in range(N_HEADS):
        h_result = result[:N_PATCH_PAD, h * D_HEAD:(h + 1) * D_HEAD]
        h_ref = ref_out[:N_PATCH_PAD, h * D_HEAD:(h + 1) * D_HEAD]
        diff = (h_ref - h_result).abs()
        h_max = diff.max().item()
        h_mean = diff.mean().item()
        if h < 4 or h_max > 1.0:
            print("  Head %d: max_err=%.4f mean=%.4f" % (h, h_max, h_mean))
        max_err = max(max_err, h_max)

    mean_err = (ref_out[:N_PATCH_PAD] - result[:N_PATCH_PAD]).abs().mean().item()
    print("  Overall max_err=%.4f mean=%.4f" % (max_err, mean_err))
    print("  Result range: [%.4f, %.4f]" % (result[:N_PATCH_PAD].min().item(), result[:N_PATCH_PAD].max().item()))
    print("  Result zeros?", (result[:N_PATCH_PAD] == 0).all().item())
    print("  PASS" if max_err < 5.0 else "  FAIL")

    ttnn.close_device(device)
