"""Minimal reproducer: many-to-one gather pipe, called twice.
3 cores send to core 0. grid=(4,1).
"""
import torch
import ttnn
import ttl

TILE = 32


def to_tt(t, device):
    return ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                           device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)


@ttl.kernel(grid=(4, 1))
def many_to_one(inp, out):
    """Cores 1,2,3 each send their tile to core 0. Core 0 sums all."""
    gather = ttl.PipeNet([
        ttl.Pipe(src=(1, 0), dst=(0, 0)),
        ttl.Pipe(src=(2, 0), dst=(0, 0)),
        ttl.Pipe(src=(3, 0), dst=(0, 0))])

    inp_dfb = ttl.make_dataflow_buffer_like(inp, shape=(1, 1), buffer_factor=2)
    recv_dfb = ttl.make_dataflow_buffer_like(inp, shape=(1, 1), buffer_factor=4)
    acc_dfb = ttl.make_dataflow_buffer_like(inp, shape=(1, 1), buffer_factor=2)
    out_dfb = ttl.make_dataflow_buffer_like(out, shape=(1, 1), buffer_factor=2)

    @ttl.compute()
    def compute():
        cx, _ = ttl.core(dims=2)
        if cx == 0:
            # Own value
            with inp_dfb.wait() as own, acc_dfb.reserve() as a:
                a.store(own)
            # Add 3 received values
            for i in range(3):
                with recv_dfb.wait() as rv, acc_dfb.wait() as prev, acc_dfb.reserve() as a:
                    a.store(prev + rv)
            with acc_dfb.wait() as total, out_dfb.reserve() as o:
                o.store(total)

    @ttl.datamovement()
    def dm_read():
        cx, _ = ttl.core(dims=2)
        # Everyone loads their tile
        with inp_dfb.reserve() as blk:
            tx = ttl.copy(inp[cx, 0], blk); tx.wait()
        # Non-zero cores send
        if cx > 0:
            with inp_dfb.wait() as blk:
                def send(pipe):
                    xf = ttl.copy(blk, pipe); xf.wait()
                gather.if_src(send)
        # Core 0 receives
        if cx == 0:
            def recv(pipe):
                with recv_dfb.reserve() as blk:
                    xf = ttl.copy(pipe, blk); xf.wait()
            gather.if_dst(recv)

    @ttl.datamovement()
    def dm_write():
        cx, _ = ttl.core(dims=2)
        if cx == 0:
            with out_dfb.wait() as blk:
                tx = ttl.copy(blk, out[0, 0]); tx.wait()


if __name__ == "__main__":
    device = ttnn.open_device(device_id=0)

    # Each core gets a tile filled with its index: 0, 1, 2, 3
    # Sum should be 0+1+2+3 = 6
    inp_t = torch.zeros(4 * TILE, TILE, dtype=torch.bfloat16)
    for i in range(4):
        inp_t[i*TILE:(i+1)*TILE, :] = float(i)

    for call in range(3):
        inp_tt = to_tt(inp_t, device)
        out_tt = to_tt(torch.zeros(4 * TILE, TILE, dtype=torch.bfloat16), device)
        many_to_one(inp_tt, out_tt)
        r = ttnn.to_torch(out_tt).float()
        print("call %d: out[0,0]=%.1f (expect 6.0)" % (call + 1, r[0, 0]))

    ttnn.close_device(device)
