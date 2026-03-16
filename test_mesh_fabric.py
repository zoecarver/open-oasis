"""Investigate fabric initialization for collectives."""
import torch
import ttnn
import inspect

N_CHIPS = 2

# Check FabricConfig options
print("FabricConfig members:")
for m in dir(ttnn.FabricConfig):
    if not m.startswith('_'):
        print(" ", m)

print("\nFabricManagerMode members:")
for m in dir(ttnn.FabricManagerMode):
    if not m.startswith('_'):
        print(" ", m)

print("\nset_fabric_config signature:")
try:
    sig = inspect.signature(ttnn.set_fabric_config)
    print(" ", sig)
except:
    help(ttnn.set_fabric_config)

# Try to initialize fabric before opening mesh
print("\nTrying fabric init...")
try:
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    print("  set_fabric_config(FABRIC_1D) succeeded")
except Exception as e:
    print("  FABRIC_1D failed:", e)

try:
    mesh_device = ttnn.open_mesh_device(ttnn.MeshShape(1, N_CHIPS))
    print("Mesh device opened")

    # Now try all_reduce
    x = torch.randn(32, 32, dtype=torch.bfloat16)
    x_tt = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                            device=mesh_device,
                            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device))
    print("Trying all_reduce...")
    try:
        result = ttnn.all_reduce(x_tt)
        print("all_reduce SUCCEEDED!")
        result_back = ttnn.to_torch(result, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0))
        print("Result shape:", result_back.shape)
    except Exception as e:
        print("all_reduce failed:", e)

    ttnn.close_mesh_device(mesh_device)
except Exception as e:
    import traceback
    traceback.print_exc()
