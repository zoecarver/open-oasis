"""Test trace + all_reduce with pre-allocated tensors (mimicking Oasis pattern)."""
import torch
import ttnn

N_CHIPS = 2
TILE = 32

ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
mesh_device = ttnn.open_mesh_device(ttnn.MeshShape(1, N_CHIPS),
                                     trace_region_size=100000000)

def replicate(t):
    return ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                           device=mesh_device,
                           mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
                           memory_config=ttnn.DRAM_MEMORY_CONFIG)

def shard(t, dim):
    return ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                           device=mesh_device,
                           mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=dim),
                           memory_config=ttnn.DRAM_MEMORY_CONFIG)

# Mimic Oasis TP pattern:
# 1. replicated activation @ col-sharded weight -> col-sharded output (QKV)
# 2. col-sharded intermediate @ row-sharded weight -> partial (O proj)
# 3. all_reduce(partial) -> replicated result
# 4. replicated + replicated bias

# Pre-allocate ALL tensors before trace
x = replicate(torch.randn(160, 1024, dtype=torch.bfloat16))
w_col = shard(torch.randn(1024, 2048, dtype=torch.bfloat16), dim=1)  # each chip: (1024, 1024)
w_row = shard(torch.randn(2048, 1024, dtype=torch.bfloat16), dim=0)  # each chip: (1024, 1024)
bias = replicate(torch.randn(160, 1024, dtype=torch.bfloat16))
# Pre-allocate output buffers
col_out = replicate(torch.zeros(160, 1024, dtype=torch.bfloat16))  # col-parallel result per chip
result = replicate(torch.zeros(160, 1024, dtype=torch.bfloat16))

# Test 1: Check if all_reduce has optional_output_tensor
print("=== Test 1: all_reduce with output_tensor ===")
import inspect
try:
    # Try the pattern: matmul -> all_reduce
    partial = ttnn.matmul(x, w_col, optional_output_tensor=col_out)
    row_out = ttnn.matmul(partial, w_row)
    # Try all_reduce - does it support output_tensor?
    try:
        reduced = ttnn.all_reduce(row_out, output_tensor=result)
        print("all_reduce with output_tensor WORKS")
    except TypeError as e:
        print("all_reduce doesn't support output_tensor:", e)
        # Try without output_tensor
        reduced = ttnn.all_reduce(row_out)
        print("all_reduce without output_tensor works, returns new tensor")
    ttnn.add(reduced, bias, output_tensor=result)
    print("Full pattern works")
except Exception as e:
    import traceback
    traceback.print_exc()

# Test 2: Trace capture with the full TP pattern
print("\n=== Test 2: Trace with matmul + all_reduce ===")
try:
    # Compile run
    col_out2 = ttnn.matmul(x, w_col)
    row_out2 = ttnn.matmul(col_out2, w_row)
    reduced2 = ttnn.all_reduce(row_out2)
    ttnn.add(reduced2, bias, output_tensor=result)
    ttnn.synchronize_device(mesh_device)
    print("Compile run done")

    # Capture trace
    trace_id = ttnn.begin_trace_capture(mesh_device, cq_id=0)
    col_out2 = ttnn.matmul(x, w_col)
    row_out2 = ttnn.matmul(col_out2, w_row)
    reduced2 = ttnn.all_reduce(row_out2)
    ttnn.add(reduced2, bias, output_tensor=result)
    ttnn.end_trace_capture(mesh_device, trace_id, cq_id=0)
    ttnn.synchronize_device(mesh_device)
    print("Trace captured!")

    # Execute trace
    ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=True)
    print("Trace executed!")

    # Verify result
    r = ttnn.to_torch(result, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0))
    print("Result shape:", r.shape)
    print("Result range: [%.2f, %.2f]" % (r[:160].min().item(), r[:160].max().item()))
    print("PASS: Trace + all_reduce works!")
except Exception as e:
    import traceback
    traceback.print_exc()
    print("FAIL:", e)

# Test 3: Trace with all_gather too (full SDPA pattern)
print("\n=== Test 3: Trace with all_gather + all_reduce ===")
try:
    # Mimic: SDPA output (sharded) -> all_gather -> ... no, we removed all_gather
    # Actually in the fixed code, we DON'T all_gather. We go:
    # col_sharded_attn @ row_sharded_out_w -> partial -> all_reduce
    # That's the same as Test 2. Let's also test the FC path:
    # replicated @ col_sharded_fc1 -> col_sharded -> gelu -> col_sharded @ row_sharded_fc2 -> partial -> all_reduce

    fc1_w = shard(torch.randn(1024, 4096, dtype=torch.bfloat16), dim=1)
    fc2_w = shard(torch.randn(4096, 1024, dtype=torch.bfloat16), dim=0)
    fc1_b = shard(torch.randn(160, 4096, dtype=torch.bfloat16), dim=1)
    fc2_b = replicate(torch.randn(160, 1024, dtype=torch.bfloat16))
    fc1_out = replicate(torch.zeros(160, 2048, dtype=torch.bfloat16))  # sharded
    gelu_out = replicate(torch.zeros(160, 2048, dtype=torch.bfloat16))
    fc2_result = replicate(torch.zeros(160, 1024, dtype=torch.bfloat16))

    # Compile
    f1 = ttnn.matmul(x, fc1_w)
    ttnn.add(f1, fc1_b, output_tensor=f1)
    g = ttnn.gelu(f1)
    f2 = ttnn.matmul(g, fc2_w)
    f2r = ttnn.all_reduce(f2)
    ttnn.add(f2r, fc2_b, output_tensor=fc2_result)
    ttnn.synchronize_device(mesh_device)
    print("FC compile done")

    # Trace
    trace_id2 = ttnn.begin_trace_capture(mesh_device, cq_id=0)
    f1 = ttnn.matmul(x, fc1_w)
    ttnn.add(f1, fc1_b, output_tensor=f1)
    g = ttnn.gelu(f1)
    f2 = ttnn.matmul(g, fc2_w)
    f2r = ttnn.all_reduce(f2)
    ttnn.add(f2r, fc2_b, output_tensor=fc2_result)
    ttnn.end_trace_capture(mesh_device, trace_id2, cq_id=0)
    ttnn.synchronize_device(mesh_device)
    print("FC trace captured!")

    ttnn.execute_trace(mesh_device, trace_id2, cq_id=0, blocking=True)
    print("FC trace executed!")
    print("PASS: Full FC TP pattern traces!")
except Exception as e:
    import traceback
    traceback.print_exc()
    print("FAIL:", e)

# Test 4: TT-Lang kernel in trace on mesh
print("\n=== Test 4: TT-Lang kernel in trace on mesh ===")
try:
    import ttl
    @ttl.kernel(grid=(1, 1))
    def add_kern(a, b, out):
        a_dfb = ttl.make_dataflow_buffer_like(a, shape=(1, 1), buffer_factor=2)
        b_dfb = ttl.make_dataflow_buffer_like(b, shape=(1, 1), buffer_factor=2)
        out_dfb = ttl.make_dataflow_buffer_like(out, shape=(1, 1), buffer_factor=2)
        @ttl.compute()
        def compute():
            with a_dfb.wait() as av, b_dfb.wait() as bv, out_dfb.reserve() as o:
                o.store(av + bv)
        @ttl.datamovement()
        def dm_read():
            with a_dfb.reserve() as blk:
                tx = ttl.copy(a[0, 0], blk); tx.wait()
            with b_dfb.reserve() as blk:
                tx = ttl.copy(b[0, 0], blk); tx.wait()
        @ttl.datamovement()
        def dm_write():
            with out_dfb.wait() as blk:
                tx = ttl.copy(blk, out[0, 0]); tx.wait()

    ta = replicate(torch.randn(32, 32, dtype=torch.bfloat16))
    tb = replicate(torch.randn(32, 32, dtype=torch.bfloat16))
    tc = replicate(torch.zeros(32, 32, dtype=torch.bfloat16))
    # Compile
    add_kern(ta, tb, tc)
    ttnn.synchronize_device(mesh_device)
    # Trace
    trace_id3 = ttnn.begin_trace_capture(mesh_device, cq_id=0)
    add_kern(ta, tb, tc)
    ttnn.end_trace_capture(mesh_device, trace_id3, cq_id=0)
    ttnn.synchronize_device(mesh_device)
    ttnn.execute_trace(mesh_device, trace_id3, cq_id=0, blocking=True)
    print("PASS: TT-Lang kernel traces on mesh!")
except Exception as e:
    import traceback
    traceback.print_exc()
    print("FAIL:", e)

print("\n=== ALL TRACE TESTS DONE ===")
ttnn.close_mesh_device(mesh_device)
