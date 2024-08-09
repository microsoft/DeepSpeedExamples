import torch
import os, timeit, functools
from utils import parse_read_arguments

def file_read(inp_f):
    with open(inp_f, 'rb') as f:
       t = torch.frombuffer(f.read(), dtype=torch.uint8)
    return t 

def main():
    args = parse_read_arguments()
    input_file = args.input_file
    file_sz = os.path.getsize(input_file)
    cnt = args.loop

    t = timeit.Timer(functools.partial(file_read, input_file))
    py_t = t.timeit(cnt)
    py_gbs = (cnt*file_sz)/py_t/1e9
    print(f'py load_cpu: {file_sz/(1024**3)}GB, {py_gbs:5.2f} GB/sec, {py_t:5.2f} secs')

if __name__ == "__main__":
    main()
