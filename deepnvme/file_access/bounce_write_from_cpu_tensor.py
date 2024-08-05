import torch
import os
import timeit, functools
import pathlib
from utils import parse_write_arguments
import deepspeed
from deepspeed.ops.op_builder import AsyncIOBuilder

def aio_bounce_buffer_write(out_f, t, h, bounce_buffer):
    bounce_buffer.copy_(t)
    h.sync_pwrite(bounce_buffer, out_f)

def main():
    cnt = 3
    args = parse_write_arguments()
    output_file = os.path.join(args.output_folder, f'test_ouput_{args.mb_size}MB.pt')
    pathlib.Path(output_file).unlink(missing_ok=True)
    file_sz = args.mb_size*(1024**2)
    app_tensor = torch.empty(file_sz, dtype=torch.uint8, device='cpu', requires_grad=False)

    aio_handle = AsyncIOBuilder().load().aio_handle(1024**2, 128, True, True, 2)
    bounce_buffer = torch.empty(file_sz, dtype=torch.uint8, requires_grad=False).pin_memory()


    t = timeit.Timer(functools.partial(aio_bounce_buffer_write, output_file, app_tensor, aio_handle, bounce_buffer))

    bb_t = t.timeit(cnt)
    bb_gbs = (cnt*file_sz)/bb_t/1e9
    print(f'bb write from cpu: {bb_gbs:5.2f} GB/sec, {bb_t:5.2f} secs')
    pathlib.Path(output_file).unlink(missing_ok=True)

if __name__ == "__main__":
    main()
