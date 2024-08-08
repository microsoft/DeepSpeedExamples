import torch
from py_utils import timed_py_tensor_load

if __name__ == "__main__":
    timed_py_tensor_load(torch.device('cpu'), "load_cpu")
