
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