import torch
import os
import timeit, functools
import pathlib
from utils import parse_write_arguments
import deepspeed
from deepspeed.ops.op_builder import GDSBuilder

def gds_write(out_f, t, h, gpu_buffer):
    gpu_buffer.copy_(t)
    h.sync_pwrite(gpu_buffer, out_f)

def main():
    cnt = 3
    args = parse_write_arguments()
    output_file = os.path.join(args.output_folder, f'test_ouput_{args.mb_size}MB.pt')
    pathlib.Path(output_file).unlink(missing_ok=True)
    file_sz = args.mb_size*(1024**2)
    app_tensor = torch.empty(file_sz, dtype=torch.uint8, device='cuda', requires_grad=False)

    gds_handle = GDSBuilder().load().gds_handle(1024**2, 128, True, True, 1)
    gds_buffer = torch.empty(file_sz, dtype=torch.uint8, device='cuda', requires_grad=False)

    t = timeit.Timer(functools.partial(gds_write, output_file, app_tensor, gds_handle, gds_buffer))

    gds_t = t.timeit(cnt)
    gds_gbs = (cnt*file_sz)/gds_t/1e9
    print(f'gds write from gpu: {gds_gbs:5.2f} GB/sec, {gds_t:5.2f} secs')

if __name__ == "__main__":
    main()
