"""Pipe-based fused gated_residual + LayerNorm + adaLN modulate.

Grid: (D_CORES, SEQ_TILES) = (8, 5) = 40 cores.
X dimension: D-cores for gather+scatter reduction.
Y dimension: independent sequence rows.

Computes:
  gated_res = residual + x * gate  (written to gated_res_out for FC2 residual)
  modulated = LN(gated_res) * (1 + scale) + shift

Recomputes gated_res in passes 2 and 3 to avoid DRAM intermediate.
adaln_packed has shift at cols [3D:4D] and scale at cols [4D:5D].
"""
import ttl

TILE = 32

def make_pipe_fused_gated_res_ln_adaln(d_tiles, d_cores, seq_tiles):
    tiles_per_core = d_tiles // d_cores
    assert d_tiles % d_cores == 0

    @ttl.kernel(grid=(d_cores, seq_tiles))
    def pipe_fused_kernel(residual, x, gate, scaler, mean_scale,
                          adaln_packed, gated_res_out, out):
        grid_x, grid_y = ttl.grid_size(dims=2)
        tpc = d_tiles // grid_x

        gather_mean = ttl.PipeNet([
            ttl.Pipe(src=(cx, cy), dst=(0, cy))
            for cx in range(1, grid_x) for cy in range(grid_y)])
        scatter_mean = ttl.PipeNet([
            ttl.Pipe(src=(0, cy), dst=(slice(1, grid_x), cy))
            for cy in range(grid_y)])
        gather_var = ttl.PipeNet([
            ttl.Pipe(src=(cx, cy), dst=(0, cy))
            for cx in range(1, grid_x) for cy in range(grid_y)])
        scatter_var = ttl.PipeNet([
            ttl.Pipe(src=(0, cy), dst=(slice(1, grid_x), cy))
            for cy in range(grid_y)])

        res_dfb = ttl.make_dataflow_buffer_like(residual, shape=(1, 1), buffer_factor=2)
        x_dfb = ttl.make_dataflow_buffer_like(x, shape=(1, 1), buffer_factor=2)
        g_dfb = ttl.make_dataflow_buffer_like(gate, shape=(1, 1), buffer_factor=2)
        gr_dfb = ttl.make_dataflow_buffer_like(residual, shape=(1, 1), buffer_factor=2)
        sc_dfb = ttl.make_dataflow_buffer_like(scaler, shape=(1, 1), buffer_factor=2)
        ms_dfb = ttl.make_dataflow_buffer_like(mean_scale, shape=(1, 1), buffer_factor=2)
        red_dfb = ttl.make_dataflow_buffer_like(scaler, shape=(1, 1), buffer_factor=2)
        acc_dfb = ttl.make_dataflow_buffer_like(scaler, shape=(1, 1), buffer_factor=2)
        send_dfb = ttl.make_dataflow_buffer_like(scaler, shape=(1, 1), buffer_factor=2)
        recv_dfb = ttl.make_dataflow_buffer_like(scaler, shape=(1, 1), buffer_factor=grid_x)
        mean_out_dfb = ttl.make_dataflow_buffer_like(residual, shape=(1, 1), buffer_factor=2)
        mean_in_dfb = ttl.make_dataflow_buffer_like(residual, shape=(1, 1), buffer_factor=2)
        send2_dfb = ttl.make_dataflow_buffer_like(scaler, shape=(1, 1), buffer_factor=2)
        recv2_dfb = ttl.make_dataflow_buffer_like(scaler, shape=(1, 1), buffer_factor=grid_x)
        istd_out_dfb = ttl.make_dataflow_buffer_like(residual, shape=(1, 1), buffer_factor=2)
        istd_in_dfb = ttl.make_dataflow_buffer_like(residual, shape=(1, 1), buffer_factor=2)
        sq_dfb = ttl.make_dataflow_buffer_like(residual, shape=(1, 1), buffer_factor=2)
        bcast_dfb = ttl.make_dataflow_buffer_like(residual, shape=(1, 1), buffer_factor=2)
        mean_dfb = ttl.make_dataflow_buffer_like(residual, shape=(1, 1), buffer_factor=2)
        istd_dfb = ttl.make_dataflow_buffer_like(residual, shape=(1, 1), buffer_factor=2)
        sh_dfb = ttl.make_dataflow_buffer_like(adaln_packed, shape=(1, 1), buffer_factor=2)
        scl_dfb = ttl.make_dataflow_buffer_like(adaln_packed, shape=(1, 1), buffer_factor=2)
        gro_dfb = ttl.make_dataflow_buffer_like(gated_res_out, shape=(1, 1), buffer_factor=2)
        out_dfb = ttl.make_dataflow_buffer_like(out, shape=(1, 1), buffer_factor=2)

        @ttl.compute()
        def compute():
            cx, cy = ttl.core(dims=2)
            with sc_dfb.wait() as sc, ms_dfb.wait() as ms:
                # === Pass 1: gated_res, write to DRAM, reduce for mean ===
                with res_dfb.wait() as rv, x_dfb.wait() as xv, g_dfb.wait() as gv:
                    with gr_dfb.reserve() as gr:
                        gr.store(rv + xv * gv)
                with gr_dfb.wait() as grv:
                    with gro_dfb.reserve() as gro:
                        gro.store(grv)
                    with red_dfb.reserve() as r:
                        r.store(ttl.math.reduce_sum(grv, sc, dims=[1]))
                with red_dfb.wait() as rv, acc_dfb.reserve() as a:
                    a.store(rv)
                for j in range(tpc - 1):
                    with res_dfb.wait() as rv, x_dfb.wait() as xv, g_dfb.wait() as gv:
                        with gr_dfb.reserve() as gr:
                            gr.store(rv + xv * gv)
                    with gr_dfb.wait() as grv:
                        with gro_dfb.reserve() as gro:
                            gro.store(grv)
                        with red_dfb.reserve() as r:
                            r.store(ttl.math.reduce_sum(grv, sc, dims=[1]))
                    with red_dfb.wait() as rv, acc_dfb.wait() as prev, acc_dfb.reserve() as a:
                        a.store(prev + rv)

                # Gather mean
                if cx == 0:
                    with acc_dfb.wait() as local_sum:
                        with recv_dfb.wait() as p0:
                            with acc_dfb.reserve() as a:
                                a.store(local_sum + p0)
                    for g in range(grid_x - 2):
                        with acc_dfb.wait() as prev, recv_dfb.wait() as pg:
                            with acc_dfb.reserve() as a:
                                a.store(prev + pg)
                    with acc_dfb.wait() as total, bcast_dfb.reserve() as bc:
                        bc.store(ttl.math.broadcast(total, dims=[1]))
                    with bcast_dfb.wait() as bc_val, mean_dfb.reserve() as m:
                        m.store(bc_val * ms)
                    with mean_dfb.wait() as mv:
                        with mean_out_dfb.reserve() as mo:
                            mo.store(mv)
                        with mean_dfb.reserve() as m2:
                            m2.store(mv)
                else:
                    with acc_dfb.wait() as rv, send_dfb.reserve() as s:
                        s.store(rv)
                    with mean_in_dfb.wait() as mv, mean_dfb.reserve() as m:
                        m.store(mv)

                # === Pass 2: recompute gated_res, variance ===
                with mean_dfb.wait() as mean_val:
                    with res_dfb.wait() as rv, x_dfb.wait() as xv, g_dfb.wait() as gv:
                        with gr_dfb.reserve() as gr:
                            gr.store(rv + xv * gv)
                    with gr_dfb.wait() as grv:
                        with sq_dfb.reserve() as sq:
                            sq.store((grv - mean_val) * (grv - mean_val))
                    with sq_dfb.wait() as sqv, red_dfb.reserve() as r:
                        r.store(ttl.math.reduce_sum(sqv, sc, dims=[1]))
                    with red_dfb.wait() as rv, acc_dfb.reserve() as a:
                        a.store(rv)
                    for j in range(tpc - 1):
                        with res_dfb.wait() as rv, x_dfb.wait() as xv, g_dfb.wait() as gv:
                            with gr_dfb.reserve() as gr:
                                gr.store(rv + xv * gv)
                        with gr_dfb.wait() as grv:
                            with sq_dfb.reserve() as sq:
                                sq.store((grv - mean_val) * (grv - mean_val))
                        with sq_dfb.wait() as sqv, red_dfb.reserve() as r:
                            r.store(ttl.math.reduce_sum(sqv, sc, dims=[1]))
                        with red_dfb.wait() as rv, acc_dfb.wait() as prev, acc_dfb.reserve() as a:
                            a.store(prev + rv)

                    # Gather variance
                    if cx == 0:
                        with acc_dfb.wait() as local_sq:
                            with recv2_dfb.wait() as p0:
                                with acc_dfb.reserve() as a:
                                    a.store(local_sq + p0)
                        for g in range(grid_x - 2):
                            with acc_dfb.wait() as prev, recv2_dfb.wait() as pg:
                                with acc_dfb.reserve() as a:
                                    a.store(prev + pg)
                        with acc_dfb.wait() as total, bcast_dfb.reserve() as bc:
                            bc.store(ttl.math.broadcast(total, dims=[1]))
                        with bcast_dfb.wait() as var_bc, istd_dfb.reserve() as istd:
                            istd.store(ttl.math.rsqrt(var_bc * ms + ttl.math.fill(var_bc, 1e-6)))
                        with istd_dfb.wait() as iv:
                            with istd_out_dfb.reserve() as io:
                                io.store(iv)
                            with istd_dfb.reserve() as i2:
                                i2.store(iv)
                    else:
                        with acc_dfb.wait() as rv, send2_dfb.reserve() as s:
                            s.store(rv)
                        with istd_in_dfb.wait() as iv, istd_dfb.reserve() as istd:
                            istd.store(iv)

                    # === Pass 3: recompute gated_res, normalize + adaLN ===
                    with istd_dfb.wait() as inv_std:
                        for j in range(tpc):
                            with res_dfb.wait() as rv, x_dfb.wait() as xv, g_dfb.wait() as gv:
                                with gr_dfb.reserve() as gr:
                                    gr.store(rv + xv * gv)
                            with gr_dfb.wait() as grv, sh_dfb.wait() as shv, scl_dfb.wait() as sclv, out_dfb.reserve() as o:
                                normed = (grv - mean_val) * inv_std
                                o.store(normed * (sclv + ttl.math.fill(sclv, 1.0)) + shv)

        @ttl.datamovement()
        def dm_read():
            cx, cy = ttl.core(dims=2)
            col_start = cx * tpc

            with sc_dfb.reserve() as blk:
                tx = ttl.copy(scaler[0, 0], blk); tx.wait()
            with ms_dfb.reserve() as blk:
                tx = ttl.copy(mean_scale[0, 0], blk); tx.wait()

            # Pass 1: load residual, x, gate for each tile
            for j in range(tpc):
                with res_dfb.reserve() as blk:
                    tx = ttl.copy(residual[cy, col_start + j], blk); tx.wait()
                with x_dfb.reserve() as blk:
                    tx = ttl.copy(x[cy, col_start + j], blk); tx.wait()
                with g_dfb.reserve() as blk:
                    tx = ttl.copy(gate[cy, col_start + j], blk); tx.wait()

            # Gather+scatter mean
            def send_p1(pipe):
                with send_dfb.wait() as blk:
                    xf = ttl.copy(blk, pipe); xf.wait()
            gather_mean.if_src(send_p1)

            def recv_p1(pipe):
                with recv_dfb.reserve() as blk:
                    xf = ttl.copy(pipe, blk); xf.wait()
            gather_mean.if_dst(recv_p1)

            def send_mean(pipe):
                with mean_out_dfb.wait() as blk:
                    xf = ttl.copy(blk, pipe); xf.wait()
            scatter_mean.if_src(send_mean)

            def recv_mean(pipe):
                with mean_in_dfb.reserve() as blk:
                    xf = ttl.copy(pipe, blk); xf.wait()
            scatter_mean.if_dst(recv_mean)

            # Pass 2: reload residual, x, gate
            for j in range(tpc):
                with res_dfb.reserve() as blk:
                    tx = ttl.copy(residual[cy, col_start + j], blk); tx.wait()
                with x_dfb.reserve() as blk:
                    tx = ttl.copy(x[cy, col_start + j], blk); tx.wait()
                with g_dfb.reserve() as blk:
                    tx = ttl.copy(gate[cy, col_start + j], blk); tx.wait()

            # Gather+scatter variance
            def send_p2(pipe):
                with send2_dfb.wait() as blk:
                    xf = ttl.copy(blk, pipe); xf.wait()
            gather_var.if_src(send_p2)

            def recv_p2(pipe):
                with recv2_dfb.reserve() as blk:
                    xf = ttl.copy(pipe, blk); xf.wait()
            gather_var.if_dst(recv_p2)

            def send_istd(pipe):
                with istd_out_dfb.wait() as blk:
                    xf = ttl.copy(blk, pipe); xf.wait()
            scatter_var.if_src(send_istd)

            def recv_istd(pipe):
                with istd_in_dfb.reserve() as blk:
                    xf = ttl.copy(pipe, blk); xf.wait()
            scatter_var.if_dst(recv_istd)

            # Pass 3: reload residual, x, gate + shift, scale
            for j in range(tpc):
                with res_dfb.reserve() as blk:
                    tx = ttl.copy(residual[cy, col_start + j], blk); tx.wait()
                with x_dfb.reserve() as blk:
                    tx = ttl.copy(x[cy, col_start + j], blk); tx.wait()
                with g_dfb.reserve() as blk:
                    tx = ttl.copy(gate[cy, col_start + j], blk); tx.wait()
                with sh_dfb.reserve() as blk:
                    tx = ttl.copy(adaln_packed[cy, 3 * d_tiles + col_start + j], blk); tx.wait()
                with scl_dfb.reserve() as blk:
                    tx = ttl.copy(adaln_packed[cy, 4 * d_tiles + col_start + j], blk); tx.wait()

        @ttl.datamovement()
        def dm_write():
            cx, cy = ttl.core(dims=2)
            col_start = cx * tpc
            # Write gated_res (from pass 1)
            for j in range(tpc):
                with gro_dfb.wait() as blk:
                    tx = ttl.copy(blk, gated_res_out[cy, col_start + j]); tx.wait()
            # Write modulated output (from pass 3)
            for j in range(tpc):
                with out_dfb.wait() as blk:
                    tx = ttl.copy(blk, out[cy, col_start + j]); tx.wait()

    return pipe_fused_kernel
