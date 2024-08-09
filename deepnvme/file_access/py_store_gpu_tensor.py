import torch
import os, timeit, functools
import pathlib
from utils import parse_write_arguments

def file_write(out_f, t):
    with open(out_f, 'wb') as f:
       f.write(t.numpy(force=True))

def main():
    args = parse_write_arguments()
    cnt = args.loop
    output_file = os.path.join(args.nvme_folder, f'test_ouput_{args.mb_size}MB.pt')
    pathlib.Path(output_file).unlink(missing_ok=True)
    file_sz = args.mb_size*(1024**2)
    gpu_tensor = torch.empty(file_sz, dtype=torch.uint8, device='cuda', requires_grad=False)

    t = timeit.Timer(functools.partial(file_write, output_file, gpu_tensor))

    py_t = t.timeit(cnt)
    py_gbs = (cnt*file_sz)/py_t/1e9
    print(f'py store_gpu: {file_sz/(1024**3)}GB, {py_gbs:5.2f} GB/sec, {py_t:5.2f} secs')
    pathlib.Path(output_file).unlink(missing_ok=True)


if __name__ == "__main__":
    main()
