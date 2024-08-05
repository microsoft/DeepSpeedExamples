import torch
import os
import timeit, functools
from utils import parse_read_arguments
import deepspeed
from deepspeed.ops.op_builder import GDSBuilder

def gds_read(inp_f, h, gpu_buffer):
    read_status = h.sync_pread(gpu_buffer, inp_f)
    t = gpu_buffer.cuda()

def main():
    cnt = 3
    args = parse_read_arguments()

    input_file = args.input_file
    file_sz = os.path.getsize(input_file)
    gds_handle = GDSBuilder().load().gds_handle(1024**2, 128, True, True, 1)
    gds_buffer = torch.empty(file_sz, dtype=torch.uint8, device='cuda', requires_grad=False)

    t = timeit.Timer(functools.partial(gds_read, input_file, gds_handle, gds_buffer))
    gds_t = t.timeit(cnt)
    gds_gbs = (cnt*os.path.getsize(input_file))/gds_t/1e9
    print(f'gds read into gpu: {gds_gbs:5.2f} GB/sec, {gds_t:5.2f} secs')

if __name__ == "__main__":
    main()
