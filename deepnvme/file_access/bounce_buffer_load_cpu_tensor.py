import torch
import os
import timeit, functools
from utils import parse_read_arguments
import deepspeed
from deepspeed.ops.op_builder import AsyncIOBuilder

def file_read(inp_f, h, bounce_buffer):
    read_status = h.sync_pread(bounce_buffer, inp_f)
    t = bounce_buffer.cpu()

def main():
    cnt = 3
    args = parse_read_arguments()

    input_file = args.input_file
    aio_handle = AsyncIOBuilder().load().aio_handle(1024**2, 128, True, True, 2)
    bounce_buffer = torch.empty(os.path.getsize(input_file), dtype=torch.uint8).pin_memory()

    t = timeit.Timer(functools.partial(file_read, input_file, aio_handle, bounce_buffer))
    bb_t = t.timeit(cnt)
    bb_gbs = (cnt*os.path.getsize(input_file))/bb_t/1e9
    print(f'bbuf load_cpu: {bb_gbs:5.2f} GB/sec, {bb_t:5.2f} secs')

if __name__ == "__main__":
    main()
