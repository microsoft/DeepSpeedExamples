import torch
import os, timeit, functools
from deepspeed.ops.op_builder import AsyncIOBuilder
from utils import parse_read_arguments

def file_read(inp_f, h, bounce_buffer):
    h.sync_pread(bounce_buffer, inp_f)
    return bounce_buffer.cpu()

def main():
    args = parse_read_arguments()
    input_file = args.input_file
    file_sz = os.path.getsize(input_file)
    cnt = args.loop

    aio_handle = AsyncIOBuilder().load().aio_handle(1024**2, 128, True, True, 1)
    bounce_buffer = torch.empty(os.path.getsize(input_file), dtype=torch.uint8).pin_memory()

    t = timeit.Timer(functools.partial(file_read, input_file, aio_handle, bounce_buffer))
    aio_t = t.timeit(cnt)
    aio_gbs = (cnt*file_sz)/aio_t/1e9
    print(f'bbuf load_cpu: {file_sz/(1024**3)}GB, {aio_gbs:5.2f} GB/sec, {aio_t:5.2f} secs')

    if args.validate: 
        from py_load_cpu_tensor import file_read as py_file_read 
        aio_tensor = file_read(input_file, aio_handle, bounce_buffer)
        py_tensor = py_file_read(input_file)
        print(f'Validation success = {aio_tensor.equal(py_tensor)}')

if __name__ == "__main__":
    main()
