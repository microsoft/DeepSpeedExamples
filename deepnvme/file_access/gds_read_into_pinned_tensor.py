import torch
import os
import timeit
import deepspeed
from deepspeed.ops.op_builder import GDSBuilder


input_file = os.path.join("/mnt", "nvme03", "aio", "test_1GB.pt")
gds_handle = GDSBuilder().load().gds_handle(1024**2, 128, True, True, 1)
gds_buffer = torch.empty(os.path.getsize(input_file), dtype=torch.uint8, device='cuda', requires_grad=False)

def gds_read(inp_f, h, t):
    read_status = h.sync_pread(t, inp_f)
    # print(f'{read_status=}')

if __name__ == "__main__":
    cnt = 3
    file_sz = os.path.getsize(input_file)
    gds_handle.new_device_locked_tensor(gds_buffer)
    gds_pinned_t = timeit.timeit('gds_read(input_file, gds_handle, gds_buffer)', setup="from __main__ import gds_read", globals=globals(), number=cnt)
    gds_handle.free_device_locked_tensor(gds_buffer)
    gds_pinned_gbs = (cnt*file_sz)/gds_pinned_t/1e9
    print(f'gds_pinned: {gds_pinned_gbs:5.2f} GB/sec, {gds_pinned_t:5.2f} secs')
