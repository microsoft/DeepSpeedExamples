import torch
from py_utils import timed_py_tensor_store

if __name__ == "__main__":
    timed_py_tensor_store(torch.device('cuda'), "store_gpu")