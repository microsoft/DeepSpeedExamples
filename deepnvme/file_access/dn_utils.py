import os 
import torch
import timeit
import functools
import pathlib
from deepspeed.ops.op_builder import AsyncIOBuilder
from utils import parse_nvme_arguments, create_file

def aio_file_read(inp_f, h, bounce_buffer, device):
    h.sync_pread(bounce_buffer, inp_f)
    t = bounce_buffer.to(device)

def timed_aio_tensor_load(device:torch.device, tag:str):
    args = parse_nvme_arguments()    
    input_file = args.file_path
    create_file(input_file, args.mb_size * (1024 **2))
    file_sz = os.path.getsize(input_file)
    aio_handle = AsyncIOBuilder().load().aio_handle(1024**2, 128, True, True, 2)
    bounce_buffer = torch.empty(os.path.getsize(input_file), dtype=torch.uint8).pin_memory()
    t = timeit.Timer(functools.partial(aio_file_read, input_file, aio_handle, bounce_buffer, device))    
    aio_t = t.timeit(args.loop)   

    aio_gbs = (args.loop*file_sz)/aio_t/1e9
    print(f'aio {tag}: {args.mb_size/1024}GB, {aio_gbs:5.2f} GB/sec, {aio_t:5.2f} secs')
    pathlib.Path(input_file).unlink(missing_ok=True)
