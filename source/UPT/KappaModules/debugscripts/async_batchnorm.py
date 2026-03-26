import os

import torch
import torch.distributed as dist
from torch.multiprocessing import spawn

from kappamodules.layers import AsyncBatchNorm


def main_single(rank, world_size):
    if rank == 0:
        print(f"world_size: {world_size}")
    x = torch.rand(4, 5, generator=torch.manual_seed(843), requires_grad=True)
    assert len(x) % 2 == 0
    if world_size == 2:
        n = len(x) // 2
        x = x[rank * n:rank * n + n]
        assert len(x) == n
    abn = AsyncBatchNorm(dim=x.size(1), affine=False, whiten=False)
    for i in range(3):
        y = abn(x)
        if rank == 0:
            print(f"{i} y   : {y[[0]]}")
            # print(f"{i} mean: {abn.mean}")
            # print(f"{i} var : {abn.var}")
    abn.finish()


def run_multi(rank):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "55555"
    dist.init_process_group(backend="gloo", init_method="env://", world_size=2, rank=rank)
    main_single(rank=rank, world_size=2)
    dist.destroy_process_group()


def main_multi():
    spawn(run_multi, nprocs=2)


if __name__ == "__main__":
    # main_single(rank=0, world_size=1)
    main_multi()
