import torch
import os
import timeit
import deepspeed
from deepspeed.ops.op_builder import AsyncIOBuilder


input_file = os.path.join("/mnt", "nvme03", "aio", "test_1GB.pt")
aio_handle = AsyncIOBuilder().load().aio_handle(1024**2, 128, True, True, 1)
aio_buffer = torch.empty(os.path.getsize(input_file), dtype=torch.uint8).pin_memory()

def aio_read(inp_f, h, t):
    read_status = h.sync_pread(t, inp_f)
    # print(f'{read_status=}')


if __name__ == "__main__":
    cnt = 3
    file_sz = os.path.getsize(input_file)
    aio_pinned_t = timeit.timeit('aio_read(input_file, aio_handle, aio_buffer)', setup="from __main__ import aio_read", globals=globals(), number=cnt)
    aio_pinned_gbs = (cnt*file_sz)/aio_pinned_t/1e9
    print(f'aio_pinned: {aio_pinned_gbs:5.2f} GB/sec, {aio_pinned_t:5.2f} secs')
