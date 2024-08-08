import torch
from dn_utils import timed_aio_tensor_load

if __name__ == "__main__":
    timed_aio_tensor_load(torch.device('cuda'), 'load_gpu')

