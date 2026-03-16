"""Compare 1D vs 2D pipe grids called twice.
Both do the same thing: core 0 loads a, core 1 loads b, pipe b to core 0, add.
Only difference: grid=(2,1) vs grid=(2,2) with 2 independent Y rows.
"""
import torch
import ttnn
import ttl

TILE = 32


def to_tt(t, device):
    return ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                           device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)


# === 1D version: grid=(2,1) ===
@ttl.kernel(grid=(2, 1))
def pipe_1d(a, b, out):
    net = ttl.PipeNet([ttl.Pipe((1, 0), (0, 0))])
    a_dfb = ttl.make_dataflow_buffer_like(a, shape=(1, 1), buffer_factor=2)
    recv_dfb = ttl.make_dataflow_buffer_like(a, shape=(1, 1), buffer_factor=2)
    out_dfb = ttl.make_dataflow_buffer_like(out, shape=(1, 1), buffer_factor=2)

    @ttl.compute()
    def compute():
        cx, _ = ttl.core(dims=2)
        if cx == 0:
            with a_dfb.wait() as av, recv_dfb.wait() as rv, out_dfb.reserve() as o:
                o.store(av + rv)

    @ttl.datamovement()
    def dm_read():
        cx, _ = ttl.core(dims=2)
        if cx == 0:
            with a_dfb.reserve() as blk:
                tx = ttl.copy(a[0, 0], blk); tx.wait()
            with recv_dfb.reserve() as blk:
                def recv(pipe):
                    xf = ttl.copy(pipe, blk); xf.wait()
                net.if_dst(recv)
        if cx == 1:
            with a_dfb.reserve() as blk:
                tx = ttl.copy(b[0, 0], blk); tx.wait()
                def send(pipe):
                    xf = ttl.copy(blk, pipe); xf.wait()
                net.if_src(send)

    @ttl.datamovement()
    def dm_write():
        cx, _ = ttl.core(dims=2)
        if cx == 0:
            with out_dfb.wait() as blk:
                tx = ttl.copy(blk, out[0, 0]); tx.wait()


# === 2D version: grid=(2,2) ===
@ttl.kernel(grid=(2, 2))
def pipe_2d(a, b, out):
    net = ttl.PipeNet([
        ttl.Pipe((1, 0), (0, 0)),
        ttl.Pipe((1, 1), (0, 1))])
    a_dfb = ttl.make_dataflow_buffer_like(a, shape=(1, 1), buffer_factor=2)
    recv_dfb = ttl.make_dataflow_buffer_like(a, shape=(1, 1), buffer_factor=2)
    out_dfb = ttl.make_dataflow_buffer_like(out, shape=(1, 1), buffer_factor=2)

    @ttl.compute()
    def compute():
        cx, cy = ttl.core(dims=2)
        if cx == 0:
            with a_dfb.wait() as av, recv_dfb.wait() as rv, out_dfb.reserve() as o:
                o.store(av + rv)

    @ttl.datamovement()
    def dm_read():
        cx, cy = ttl.core(dims=2)
        if cx == 0:
            with a_dfb.reserve() as blk:
                tx = ttl.copy(a[cy, 0], blk); tx.wait()
            with recv_dfb.reserve() as blk:
                def recv(pipe):
                    xf = ttl.copy(pipe, blk); xf.wait()
                net.if_dst(recv)
        if cx == 1:
            with a_dfb.reserve() as blk:
                tx = ttl.copy(b[cy, 0], blk); tx.wait()
                def send(pipe):
                    xf = ttl.copy(blk, pipe); xf.wait()
                net.if_src(send)

    @ttl.datamovement()
    def dm_write():
        cx, cy = ttl.core(dims=2)
        if cx == 0:
            with out_dfb.wait() as blk:
                tx = ttl.copy(blk, out[cy, 0]); tx.wait()


if __name__ == "__main__":
    device = ttnn.open_device(device_id=0)

    a_t = torch.full((2 * TILE, TILE), 3.0, dtype=torch.bfloat16)
    b_t = torch.full((2 * TILE, TILE), 4.0, dtype=torch.bfloat16)

    print("=== 1D pipe: grid=(2,1) ===")
    for call in range(3):
        a1 = to_tt(a_t, device)
        b1 = to_tt(b_t, device)
        o1 = to_tt(torch.zeros(2 * TILE, TILE, dtype=torch.bfloat16), device)
        pipe_1d(a1, b1, o1)
        r = ttnn.to_torch(o1).float()
        print("  call %d: out[0,0]=%.1f (expect 7.0)" % (call + 1, r[0, 0]))

    print("\n=== 2D pipe: grid=(2,2) ===")
    for call in range(3):
        a2 = to_tt(a_t, device)
        b2 = to_tt(b_t, device)
        o2 = to_tt(torch.zeros(2 * TILE, TILE, dtype=torch.bfloat16), device)
        pipe_2d(a2, b2, o2)
        r = ttnn.to_torch(o2).float()
        print("  call %d: out[0,0]=%.1f out[32,0]=%.1f (expect 7.0, 7.0)" % (
            call + 1, r[0, 0], r[TILE, 0]))

    ttnn.close_device(device)
