import torch
import os
import timeit

input_file = os.path.join("/mnt", "nvme03", "aio", "test_1GB.pt")

def py_read(inp_f):
    with open(inp_f, 'rb') as f:
       t = torch.frombuffer(f.read(), dtype=torch.uint8)
    # print(f'{t.size()=}')

if __name__ == "__main__":
    cnt = 3
    file_sz = os.path.getsize(input_file)
    py_t = timeit.timeit('py_read(input_file)', setup="from __main__ import py_read", globals=globals(), number=cnt)
    py_gbs = (cnt*file_sz)/py_t/1e9
    print(f'py: {py_gbs:5.2f} GB/sec, {py_t:5.2f} secs')
