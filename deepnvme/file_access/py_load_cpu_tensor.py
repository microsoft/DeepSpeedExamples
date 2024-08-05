import torch
import os
import timeit, functools
from utils import parse_read_arguments

def file_read(inp_f):
    with open(inp_f, 'rb') as f:
       t = torch.frombuffer(f.read(), dtype=torch.uint8)

def main():
    cnt = 3
    args = parse_read_arguments()
    input_file = args.input_file
    t = timeit.Timer(functools.partial(file_read, input_file))
    py_t = t.timeit(cnt)
    py_gbs = (cnt*os.path.getsize(input_file))/py_t/1e9
    print(f'py load_cpu: {py_gbs:5.2f} GB/sec, {py_t:5.2f} secs')

if __name__ == "__main__":
    main()
