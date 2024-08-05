import torch
import numpy
import os
import timeit, functools
from utils import parse_write_arguments
import pathlib

def py_write(out_f, t):
    with open(out_f, 'wb') as f:
       f.write(t.numpy(force=True))

def main():
    cnt = 3
    args = parse_write_arguments()
    output_file = os.path.join(args.output_folder, f'test_ouput_{args.mb_size}MB.pt')
    pathlib.Path(output_file).unlink(missing_ok=True)
    file_sz = args.mb_size*(1024**2)
    cpu_tensor = torch.empty(file_sz, dtype=torch.uint8, device='cpu', requires_grad=False)

    t = timeit.Timer(functools.partial(py_write, output_file, cpu_tensor))

    py_t = t.timeit(cnt)
    py_gbs = (cnt*file_sz)/py_t/1e9
    print(f'py write from cpu: {py_gbs:5.2f} GB/sec, {py_t:5.2f} secs')
    pathlib.Path(output_file).unlink(missing_ok=True)

if __name__ == "__main__":
    main()
