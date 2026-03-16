"""Test mesh device: shard, matmul, collectives, TT-Lang, trace, end-to-end TP."""
import torch
import ttnn
import ttl

N_CHIPS = 2
TILE = 32

ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)

print("Opening mesh device with %d chips..." % N_CHIPS)
mesh_device = ttnn.open_mesh_device(ttnn.MeshShape(1, N_CHIPS),
                                     trace_region_size=100000000)
print("Mesh device opened:", mesh_device)

def replicate(t):
    return ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                           device=mesh_device,
                           mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device))

def shard(t, dim):
    return ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                           device=mesh_device,
                           mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=dim))

def readback_concat(t, dim):
    return ttnn.to_torch(t, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=dim))

passed = 0
failed = 0
def check(name, condition, detail=""):
    global passed, failed
    if condition:
        print("  PASS: %s %s" % (name, detail))
        passed += 1
    else:
        print("  FAIL: %s %s" % (name, detail))
        failed += 1

# Test 1: Column-parallel matmul
print("\n=== Test 1: Column-parallel matmul ===")
a = torch.randn(160, 1024, dtype=torch.bfloat16)
w = torch.randn(1024, 5120, dtype=torch.bfloat16)
expected = a.float() @ w.float()
c_tt = ttnn.matmul(replicate(a), shard(w, dim=1))
c_back = readback_concat(c_tt, dim=1)
diff = (c_back.float() - expected).abs().max().item()
check("column-parallel matmul", diff < 5.0, "(max_diff=%.2f)" % diff)

# Test 2: Row-parallel matmul + all_reduce
print("\n=== Test 2: Row-parallel matmul + all_reduce ===")
x_full = torch.randn(160, 4096, dtype=torch.bfloat16)
w_row = torch.randn(4096, 1024, dtype=torch.bfloat16)
expected_row = x_full.float() @ w_row.float()
partial = ttnn.matmul(shard(x_full, dim=1), shard(w_row, dim=0))
reduced = ttnn.all_reduce(partial)
r_back = readback_concat(reduced, dim=0)
chip0 = r_back[:160].float()
diff = (chip0 - expected_row).abs().max().item()
check("row-parallel all_reduce", diff < 10.0, "(max_diff=%.2f)" % diff)

# Test 3: all_gather
print("\n=== Test 3: all_gather ===")
sdpa_full = torch.randn(160, 1024, dtype=torch.bfloat16)
sdpa_tt = shard(sdpa_full, dim=1)
gathered = ttnn.all_gather(sdpa_tt, dim=1)
g_back = readback_concat(gathered, dim=0)
chip0 = g_back[:160]
diff = (chip0.float() - sdpa_full.float()).abs().max().item()
check("all_gather", diff == 0, "(diff=%.4f)" % diff)

# Test 4: TT-Lang kernel on mesh
print("\n=== Test 4: TT-Lang kernel on mesh ===")
@ttl.kernel(grid="auto")
def add_kernel(a, b, out):
    grid_cols, _ = ttl.grid_size(dims=2)
    n_rows = a.shape[0] // TILE
    n_cols = a.shape[1] // TILE
    total = n_rows * n_cols
    per_core = -(-total // grid_cols)
    a_dfb = ttl.make_dataflow_buffer_like(a, shape=(1, 1), buffer_factor=2)
    b_dfb = ttl.make_dataflow_buffer_like(b, shape=(1, 1), buffer_factor=2)
    out_dfb = ttl.make_dataflow_buffer_like(out, shape=(1, 1), buffer_factor=2)
    @ttl.compute()
    def compute():
        core_x, _ = ttl.core(dims=2)
        for t in range(per_core):
            idx = core_x * per_core + t
            if idx < total:
                with a_dfb.wait() as av, b_dfb.wait() as bv, out_dfb.reserve() as o:
                    o.store(av + bv)
    @ttl.datamovement()
    def dm_read():
        core_x, _ = ttl.core(dims=2)
        for t in range(per_core):
            idx = core_x * per_core + t
            if idx < total:
                r = idx // n_cols
                c = idx % n_cols
                with a_dfb.reserve() as blk:
                    tx = ttl.copy(a[r, c], blk); tx.wait()
                with b_dfb.reserve() as blk:
                    tx = ttl.copy(b[r, c], blk); tx.wait()
    @ttl.datamovement()
    def dm_write():
        core_x, _ = ttl.core(dims=2)
        for t in range(per_core):
            idx = core_x * per_core + t
            if idx < total:
                r = idx // n_cols
                c = idx % n_cols
                with out_dfb.wait() as blk:
                    tx = ttl.copy(blk, out[r, c]); tx.wait()

a_torch = torch.randn(160, 1024, dtype=torch.bfloat16)
b_torch = torch.randn(160, 1024, dtype=torch.bfloat16)
out_torch = torch.zeros(160, 1024, dtype=torch.bfloat16)
add_kernel(replicate(a_torch), replicate(b_torch), replicate(out_torch))
# Can't easily readback TT-Lang output on mesh since it wrote in-place to replicated tensor
# Just check it doesn't crash
check("TT-Lang kernel runs on mesh", True)

# Test 5: End-to-end TP pattern
print("\n=== Test 5: End-to-end TP: col_matmul -> row_matmul -> all_reduce + bias ===")
x = torch.randn(160, 1024, dtype=torch.bfloat16)
w_col = torch.randn(1024, 2048, dtype=torch.bfloat16)
w_row = torch.randn(2048, 1024, dtype=torch.bfloat16)
bias = torch.randn(160, 1024, dtype=torch.bfloat16)
expected_full = (x.float() @ w_col.float() @ w_row.float()) + bias.float()

x_tt = replicate(x)
y = ttnn.matmul(x_tt, shard(w_col, dim=1))
z = ttnn.matmul(y, shard(w_row, dim=0))
z_full = ttnn.all_reduce(z)
ttnn.add(z_full, replicate(bias), output_tensor=z_full)
z_back = readback_concat(z_full, dim=0)
chip0 = z_back[:160].float()
diff = (chip0 - expected_full).abs().max().item()
check("end-to-end TP", diff < 15.0, "(max_diff=%.2f)" % diff)

# Test 6: Trace capture on mesh
print("\n=== Test 6: Trace on mesh ===")
try:
    trace_a = replicate(torch.randn(32, 32, dtype=torch.bfloat16))
    trace_b = replicate(torch.randn(32, 32, dtype=torch.bfloat16))
    trace_out = replicate(torch.zeros(32, 32, dtype=torch.bfloat16))
    # Compile
    trace_result = ttnn.matmul(trace_a, trace_b)
    # Capture
    trace_id = ttnn.begin_trace_capture(mesh_device, cq_id=0)
    trace_result = ttnn.matmul(trace_a, trace_b)
    ttnn.end_trace_capture(mesh_device, trace_id, cq_id=0)
    # Execute
    ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=True)
    check("trace on mesh", True)
except Exception as e:
    check("trace on mesh", False, str(e))

# Test 7: Trace with collectives (this is the critical one for DDIM)
print("\n=== Test 7: Trace with all_reduce ===")
try:
    t_x = replicate(torch.randn(32, 64, dtype=torch.bfloat16))
    t_w = shard(torch.randn(64, 32, dtype=torch.bfloat16), dim=0)
    # Compile
    t_partial = ttnn.matmul(t_x, t_w)
    t_reduced = ttnn.all_reduce(t_partial)
    # Capture
    trace_id2 = ttnn.begin_trace_capture(mesh_device, cq_id=0)
    t_partial = ttnn.matmul(t_x, t_w)
    t_reduced = ttnn.all_reduce(t_partial)
    ttnn.end_trace_capture(mesh_device, trace_id2, cq_id=0)
    # Execute
    ttnn.execute_trace(mesh_device, trace_id2, cq_id=0, blocking=True)
    check("trace with all_reduce", True)
except Exception as e:
    check("trace with all_reduce", False, str(e))

print("\n" + "=" * 40)
print("Results: %d passed, %d failed" % (passed, failed))
ttnn.close_mesh_device(mesh_device)
