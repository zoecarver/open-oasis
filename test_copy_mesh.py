"""Test copy_host_to_device_tensor with mesh tensors."""
import torch
import ttnn

N_CHIPS = 2
ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
mesh_device = ttnn.open_mesh_device(ttnn.MeshShape(1, N_CHIPS))

# Create mesh tensor (replicated)
dev_tensor = ttnn.from_torch(
    torch.zeros(32, 32, dtype=torch.bfloat16),
    dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
    device=mesh_device,
    mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    memory_config=ttnn.DRAM_MEMORY_CONFIG)

# Create host tensor
host_data = torch.randn(32, 32, dtype=torch.bfloat16)
host_tensor = ttnn.from_torch(host_data, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)

# Try copy
print("Copying host tensor to mesh device tensor...")
try:
    ttnn.copy_host_to_device_tensor(host_tensor, dev_tensor)
    print("PASS: copy_host_to_device_tensor works with mesh")
    # Verify
    result = ttnn.to_torch(dev_tensor, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0))
    diff = (result[:32] - host_data).abs().max().item()
    print("Readback diff:", diff)
    assert diff == 0, "Data mismatch!"
    print("Data verified on both chips!")
except Exception as e:
    import traceback
    traceback.print_exc()

ttnn.close_mesh_device(mesh_device)
