"""WIP: Fused LN + adaLN modulate + QKV matmul + RoPE + SDPA kernel.

Status: Exceeds kernel config buffer limit by ~1,568 bytes (72,240 vs 70,672).
27 DFBs. Uses self-cycling on ln_mean_dfb, ln_istd_dfb, ln_red_dfb to save 3 DFBs.

Could revisit if:
- Kernel config buffer limit is raised
- RMSNorm used instead of LayerNorm (eliminates mean pass, ~1/3 less code)
- Compiler generates more compact code
"""
import ttl

TILE = 32

def make_ln_qkv_rope_sdpa_kernel(dim_tiles, seq_tiles, head_tiles, n_heads_val, scale_val):
    @ttl.kernel(grid="auto")
    def ln_qkv_rope_sdpa(x, shift_msa, scale_msa, qkv_w, cos_tab, sin_tab,
                          scaler, mean_scale, mod_scratch, out):
        grid_cols, _ = ttl.grid_size(dims=2)
        n_frames = x.shape[0] // TILE // seq_tiles
        total_heads = n_frames * n_heads_val
        heads_per_core = -(-total_heads // grid_cols)
        d_tiles = n_heads_val * head_tiles

        # LN DFBs (all 1x1, self-cycling to save DFBs)
        ln_x_dfb = ttl.make_dataflow_buffer_like(x, shape=(1, 1), buffer_factor=2)
        ln_red_dfb = ttl.make_dataflow_buffer_like(scaler, shape=(1, 1), buffer_factor=2)
        ln_acc_dfb = ttl.make_dataflow_buffer_like(scaler, shape=(1, 1), buffer_factor=2)
        ln_mean_dfb = ttl.make_dataflow_buffer_like(x, shape=(1, 1), buffer_factor=2)
        ln_istd_dfb = ttl.make_dataflow_buffer_like(x, shape=(1, 1), buffer_factor=2)
        sh_dfb = ttl.make_dataflow_buffer_like(shift_msa, shape=(1, 1), buffer_factor=2)
        scl_dfb = ttl.make_dataflow_buffer_like(scale_msa, shape=(1, 1), buffer_factor=2)
        ln_out_dfb = ttl.make_dataflow_buffer_like(x, shape=(1, 1), buffer_factor=2)
        scaler_dfb = ttl.make_dataflow_buffer_like(scaler, shape=(1, 1), buffer_factor=1)
        ms_dfb = ttl.make_dataflow_buffer_like(mean_scale, shape=(1, 1), buffer_factor=1)
        # QKV + SDPA DFBs
        mod_dfb = ttl.make_dataflow_buffer_like(x, shape=(seq_tiles, 1), buffer_factor=2)
        w_dfb = ttl.make_dataflow_buffer_like(qkv_w, shape=(1, head_tiles), buffer_factor=2)
        acc_dfb = ttl.make_dataflow_buffer_like(x, shape=(seq_tiles, head_tiles), buffer_factor=2)
        mm_dfb = ttl.make_dataflow_buffer_like(x, shape=(seq_tiles, head_tiles), buffer_factor=2)
        q_dfb = ttl.make_dataflow_buffer_like(x, shape=(seq_tiles, head_tiles), buffer_factor=2)
        k_dfb = ttl.make_dataflow_buffer_like(x, shape=(seq_tiles, head_tiles), buffer_factor=2)
        v_dfb = ttl.make_dataflow_buffer_like(x, shape=(seq_tiles, head_tiles), buffer_factor=2)
        cos_dfb = ttl.make_dataflow_buffer_like(cos_tab, shape=(seq_tiles, head_tiles), buffer_factor=2)
        sin_dfb = ttl.make_dataflow_buffer_like(sin_tab, shape=(seq_tiles, head_tiles), buffer_factor=2)
        qr_dfb = ttl.make_dataflow_buffer_like(x, shape=(seq_tiles, head_tiles), buffer_factor=2)
        kr_dfb = ttl.make_dataflow_buffer_like(x, shape=(seq_tiles, head_tiles), buffer_factor=2)
        kt_dfb = ttl.make_dataflow_buffer_like(x, shape=(head_tiles, seq_tiles), buffer_factor=2)
        a_dfb = ttl.make_dataflow_buffer_like(scaler, shape=(seq_tiles, seq_tiles), buffer_factor=2)
        b_dfb = ttl.make_dataflow_buffer_like(scaler, shape=(seq_tiles, seq_tiles), buffer_factor=2)
        row_dfb = ttl.make_dataflow_buffer_like(scaler, shape=(seq_tiles, 1), buffer_factor=2)
        row_bc_dfb = ttl.make_dataflow_buffer_like(scaler, shape=(seq_tiles, seq_tiles), buffer_factor=2)
        out_dfb = ttl.make_dataflow_buffer_like(out, shape=(seq_tiles, head_tiles), buffer_factor=2)

        @ttl.compute()
        def compute():
            core_x, _ = ttl.core(dims=2)
            with scaler_dfb.wait() as sc, ms_dfb.wait() as ms:
                for local_h in range(heads_per_core):
                    head_idx = core_x * heads_per_core + local_h
                    if head_idx < total_heads:
                        # --- LN + modulate phase (per row) ---
                        for row in range(seq_tiles):
                            with ln_x_dfb.wait() as x0:
                                with ln_red_dfb.reserve() as r:
                                    r.store(ttl.math.reduce_sum(x0, sc, dims=[1]))
                            with ln_red_dfb.wait() as rv, ln_acc_dfb.reserve() as acc:
                                acc.store(rv)
                            for j in range(dim_tiles - 1):
                                with ln_x_dfb.wait() as xj:
                                    with ln_red_dfb.reserve() as r:
                                        r.store(ttl.math.reduce_sum(xj, sc, dims=[1]))
                                with ln_red_dfb.wait() as rv, ln_acc_dfb.wait() as av, ln_acc_dfb.reserve() as acc:
                                    acc.store(av + rv)
                            with ln_acc_dfb.wait() as sum_x, ln_mean_dfb.reserve() as bc:
                                bc.store(ttl.math.broadcast(sum_x, dims=[1]))
                            with ln_mean_dfb.wait() as sum_bc, ln_mean_dfb.reserve() as mean_out:
                                mean_out.store(sum_bc * ms)
                            with ln_mean_dfb.wait() as mean_val:
                                with ln_x_dfb.wait() as x0, ln_red_dfb.reserve() as sq:
                                    sq.store((x0 - mean_val) * (x0 - mean_val))
                                with ln_red_dfb.wait() as sqv, ln_red_dfb.reserve() as r:
                                    r.store(ttl.math.reduce_sum(sqv, sc, dims=[1]))
                                with ln_red_dfb.wait() as rv, ln_acc_dfb.reserve() as acc:
                                    acc.store(rv)
                                for j in range(dim_tiles - 1):
                                    with ln_x_dfb.wait() as xj, ln_red_dfb.reserve() as sq:
                                        sq.store((xj - mean_val) * (xj - mean_val))
                                    with ln_red_dfb.wait() as sqv, ln_red_dfb.reserve() as r:
                                        r.store(ttl.math.reduce_sum(sqv, sc, dims=[1]))
                                    with ln_red_dfb.wait() as rv, ln_acc_dfb.wait() as av, ln_acc_dfb.reserve() as acc:
                                        acc.store(av + rv)
                                with ln_acc_dfb.wait() as sum_sq, ln_istd_dfb.reserve() as bc:
                                    bc.store(ttl.math.broadcast(sum_sq, dims=[1]))
                                with ln_istd_dfb.wait() as var_bc, ln_istd_dfb.reserve() as istd:
                                    istd.store(ttl.math.rsqrt(var_bc * ms + ttl.math.fill(var_bc, 1e-6)))
                                with ln_istd_dfb.wait() as inv_std:
                                    for j in range(dim_tiles):
                                        with ln_x_dfb.wait() as xj, sh_dfb.wait() as shv, scl_dfb.wait() as sclv, ln_out_dfb.reserve() as o:
                                            normed = (xj - mean_val) * inv_std
                                            o.store(normed * (sclv + ttl.math.fill(sclv, 1.0)) + shv)
                        # --- QKV + RoPE + SDPA phase ---
                        with mod_dfb.wait() as m0, w_dfb.wait() as w0, acc_dfb.reserve() as a:
                            a.store(m0 @ w0)
                        for k_idx in range(dim_tiles - 1):
                            with mod_dfb.wait() as mk, w_dfb.wait() as wk, mm_dfb.reserve() as mm:
                                mm.store(mk @ wk)
                            with mm_dfb.wait() as mmv, acc_dfb.wait() as prev, acc_dfb.reserve() as a:
                                a.store(prev + mmv)
                        with acc_dfb.wait() as qr, q_dfb.reserve() as q:
                            q.store(qr)
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
                        with mod_dfb.wait() as m0, w_dfb.wait() as w0, acc_dfb.reserve() as a:
                            a.store(m0 @ w0)
                        for k_idx in range(dim_tiles - 1):
                            with mod_dfb.wait() as mk, w_dfb.wait() as wk, mm_dfb.reserve() as mm:
                                mm.store(mk @ wk)
                            with mm_dfb.wait() as mmv, acc_dfb.wait() as prev, acc_dfb.reserve() as a:
                                a.store(prev + mmv)
                        with acc_dfb.wait() as kr, k_dfb.reserve() as k:
                            k.store(kr)
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
                        with qr_dfb.wait() as qrv, kr_dfb.wait() as krv, v_dfb.wait() as vv:
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
                                    with row_bc_dfb.wait() as smbv, a_dfb.reserve() as attn:
                                        attn.store(exv * ttl.math.recip(smbv))
                            with a_dfb.wait() as av, out_dfb.reserve() as o:
                                o.store(av @ vv)

        @ttl.datamovement()
        def dm_read():
            core_x, _ = ttl.core(dims=2)
            with scaler_dfb.reserve() as blk:
                tx = ttl.copy(scaler[0, 0], blk); tx.wait()
            with ms_dfb.reserve() as blk:
                tx = ttl.copy(mean_scale[0, 0], blk); tx.wait()
            for local_h in range(heads_per_core):
                head_idx = core_x * heads_per_core + local_h
                if head_idx < total_heads:
                    frame = head_idx // n_heads_val
                    h = head_idx % n_heads_val
                    row_start = frame * seq_tiles
                    hc = h * head_tiles
                    d_t = n_heads_val * head_tiles
                    for row in range(seq_tiles):
                        r = row_start + row
                        for j in range(dim_tiles):
                            with ln_x_dfb.reserve() as blk:
                                tx = ttl.copy(x[r, j], blk); tx.wait()
                        for j in range(dim_tiles):
                            with ln_x_dfb.reserve() as blk:
                                tx = ttl.copy(x[r, j], blk); tx.wait()
                        for j in range(dim_tiles):
                            with ln_x_dfb.reserve() as blk:
                                tx = ttl.copy(x[r, j], blk); tx.wait()
                            with sh_dfb.reserve() as blk:
                                tx = ttl.copy(shift_msa[r, j], blk); tx.wait()
                            with scl_dfb.reserve() as blk:
                                tx = ttl.copy(scale_msa[r, j], blk); tx.wait()
                    for k in range(dim_tiles):
                        with mod_dfb.reserve() as blk:
                            tx = ttl.copy(mod_scratch[row_start:row_start + seq_tiles, k], blk); tx.wait()
                        with w_dfb.reserve() as blk:
                            tx = ttl.copy(qkv_w[k, hc:hc + head_tiles], blk); tx.wait()
                    qs_col = 3 * d_t + hc
                    for k in range(dim_tiles):
                        with mod_dfb.reserve() as blk:
                            tx = ttl.copy(mod_scratch[row_start:row_start + seq_tiles, k], blk); tx.wait()
                        with w_dfb.reserve() as blk:
                            tx = ttl.copy(qkv_w[k, qs_col:qs_col + head_tiles], blk); tx.wait()
                    with cos_dfb.reserve() as blk:
                        tx = ttl.copy(cos_tab[row_start:row_start + seq_tiles, hc:hc + head_tiles], blk); tx.wait()
                    with sin_dfb.reserve() as blk:
                        tx = ttl.copy(sin_tab[row_start:row_start + seq_tiles, hc:hc + head_tiles], blk); tx.wait()
                    k_col = d_t + hc
                    for k in range(dim_tiles):
                        with mod_dfb.reserve() as blk:
                            tx = ttl.copy(mod_scratch[row_start:row_start + seq_tiles, k], blk); tx.wait()
                        with w_dfb.reserve() as blk:
                            tx = ttl.copy(qkv_w[k, k_col:k_col + head_tiles], blk); tx.wait()
                    ks_col = 4 * d_t + hc
                    for k in range(dim_tiles):
                        with mod_dfb.reserve() as blk:
                            tx = ttl.copy(mod_scratch[row_start:row_start + seq_tiles, k], blk); tx.wait()
                        with w_dfb.reserve() as blk:
                            tx = ttl.copy(qkv_w[k, ks_col:ks_col + head_tiles], blk); tx.wait()
                    with cos_dfb.reserve() as blk:
                        tx = ttl.copy(cos_tab[row_start:row_start + seq_tiles, hc:hc + head_tiles], blk); tx.wait()
                    with sin_dfb.reserve() as blk:
                        tx = ttl.copy(sin_tab[row_start:row_start + seq_tiles, hc:hc + head_tiles], blk); tx.wait()
                    v_col = 2 * d_t + hc
                    for k in range(dim_tiles):
                        with mod_dfb.reserve() as blk:
                            tx = ttl.copy(mod_scratch[row_start:row_start + seq_tiles, k], blk); tx.wait()
                        with w_dfb.reserve() as blk:
                            tx = ttl.copy(qkv_w[k, v_col:v_col + head_tiles], blk); tx.wait()

        @ttl.datamovement()
        def dm_write():
            core_x, _ = ttl.core(dims=2)
            for local_h in range(heads_per_core):
                head_idx = core_x * heads_per_core + local_h
                if head_idx < total_heads:
                    frame = head_idx // n_heads_val
                    h = head_idx % n_heads_val
                    row_start = frame * seq_tiles
                    for row in range(seq_tiles):
                        r = row_start + row
                        for j in range(dim_tiles):
                            with ln_out_dfb.wait() as blk:
                                tx = ttl.copy(blk, mod_scratch[r, j]); tx.wait()
                    col_start = h * head_tiles
                    with out_dfb.wait() as blk:
                        tx = ttl.copy(blk, out[row_start:row_start + seq_tiles, col_start:col_start + head_tiles]); tx.wait()

    return ln_qkv_rope_sdpa
