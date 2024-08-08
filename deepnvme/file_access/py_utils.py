import os 
import torch
import timeit
import functools
import pathlib
from utils import parse_nvme_arguments, create_file

def py_file_read(inp_f, device):
    with open(inp_f, 'rb') as f:
       t = torch.frombuffer(f.read(), dtype=torch.uint8).to(device)

def timed_py_tensor_load(device:torch.device, tag:str):
    args = parse_nvme_arguments()    
    input_file = args.file_path
    create_file(input_file, args.mb_size * (1024 **2))
    file_sz = os.path.getsize(input_file)
    t = timeit.Timer(functools.partial(py_file_read, input_file, device))    
    py_t = t.timeit(args.loop)   

    py_gbs = (args.loop*file_sz)/py_t/1e9
    print(f'py {tag}: {args.mb_size/1024}GB, {py_gbs:5.2f} GB/sec, {py_t:5.2f} secs')
    pathlib.Path(input_file).unlink(missing_ok=True)


def py_file_write(out_f, t):
    with open(out_f, 'wb') as f:
       f.write(t.numpy(force=True))


def timed_py_tensor_store(device:torch.device, tag:str):
    args = parse_nvme_arguments()    
    input_file = args.file_path
    file_sz = args.mb_size*(1024**2)
    data_tensor = torch.empty(file_sz, dtype=torch.uint8, device=device, requires_grad=False)
    pathlib.Path(args.file_path).unlink(missing_ok=True)

    t = timeit.Timer(functools.partial(py_file_write, input_file, data_tensor))    
    py_t = t.timeit(args.loop)   

    py_gbs = (args.loop*file_sz)/py_t/1e9
    print(f'py {tag}: {args.mb_size/1024}GB, {py_gbs:5.2f} GB/sec, {py_t:5.2f} secs')
    pathlib.Path(input_file).unlink(missing_ok=True)
