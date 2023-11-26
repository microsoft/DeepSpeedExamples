import os
from typing import Any, Callable
import datetime
import torch.distributed as dist


def is_distributed():
    return dist.is_initialized()


def get_rank():
    if not dist.is_initialized():
        return 0
    else:
        return dist.get_rank()


def rank_zero_only(fn: Callable) -> Callable:
    """Function that can be used as a decorator to enable a function/method being called only on
    global rank 0."""

    def wrapped_fn(*args: Any, **kwargs: Any):
        rank = get_rank()
        if rank == 0:
            return fn(*args, **kwargs)
        return None

    return wrapped_fn


def dist_init(rank, backend="nccl"):
    env_dict = {
        key: os.environ[key] for key in ("MASTER_ADDR", "MASTER_PORT", "RANK", "WORLD_SIZE")
    }
    print(f"[{os.getpid()}] Initializing process group with: {env_dict}")
    dist.init_process_group(backend="nccl", init_method="env://", timeout=datetime.timedelta(seconds=5400))
    assert rank == dist.get_rank()
    print(
        f"[{os.getpid()}]: world_size = {dist.get_world_size()}, "
        + f"rank = {dist.get_rank()}, backend={dist.get_backend()}"
    )
    return dist.get_world_size()


def dist_barrier():
    if dist.is_initialized():
        dist.barrier()


def dist_destroy():
    if dist.is_initialized():
        dist.destroy_process_group()