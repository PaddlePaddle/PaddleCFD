import os

import matplotlib.pyplot as plt
import torch
import torch.distributed as dist
from torch.multiprocessing import spawn
from torch.utils.data import DataLoader
# noinspection PyPackageRequirements
from tqdm import tqdm

from kappamodules.layers import DataNorm


def main_single(rank, world_size):
    if rank == 0:
        print(f"world_size: {world_size}")
    # norm = DataNorm(dim=3)
    # from torchvision.datasets import CIFAR10
    # from torchvision.transforms import ToTensor
    # ds = CIFAR10(root=".", train=True, download=True, transform=ToTensor())
    from torch.utils.data import TensorDataset
    norm = DataNorm(dim=3, channel_first=False, gather_mode="none")
    ds = TensorDataset(torch.randn(10000, 3), torch.randn(10000))

    dl = DataLoader(ds, batch_size=31 + rank)
    mean_deltas = []
    var_deltas = []
    for x, _ in tqdm(dl):
        pre_mean = norm.mean.clone()
        pre_var = norm.var.clone()
        norm(x)
        with torch.no_grad():
            mean_deltas.append((pre_mean - norm.mean).abs().mean())
            var_deltas.append((pre_var - norm.var).abs().mean())
    sd = norm.state_dict()
    print(f"mean: {sd['mean']}")
    print(f"var: {sd['var']}")
    print(f"std: {sd['var'].sqrt()}")

    plt.plot(range(len(mean_deltas)), mean_deltas, label="mean delta")
    plt.plot(range(len(var_deltas)), var_deltas, label="var delta")
    plt.legend()
    plt.grid()
    plt.yscale("log")
    plt.tight_layout()
    if os.name == "nt":
        plt.show()
    else:
        plt.savefig("datanorm.svg")


def run_multi(rank):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "55455"
    dist.init_process_group(backend="gloo", init_method="env://", world_size=2, rank=rank)
    main_single(rank=rank, world_size=2)
    dist.destroy_process_group()


def main_multi():
    spawn(run_multi, nprocs=2)


if __name__ == "__main__":
    if os.name == "nt":
        os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    # main_single(rank=0, world_size=1)
    main_multi()
