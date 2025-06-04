import argparse

import paddle


class Options(object):

    def __init__(self):
        parser = argparse.ArgumentParser()
        parser.add_argument("--no_cuda", action="store_true", default=False, help="Disables CUDA training.")
        parser.add_argument("--cuda_index", type=int, default=0, help="Cuda index you want to choose.")
        parser.add_argument("--seed", type=int, default=200, help="Number ofchs to train")
        parser.add_argument(
            "--domain",
            nargs="+",
            type=float,
            default=[-1, 1, -1, 1],
            help="domain you choose, such as [-1, 1, -1, 1] for square, washer and [-1, 1, 0, 4]  for Lshape",
        )
        parser.add_argument(
            "--pde_case",
            type=str,
            default="Stokes",
            help="type of PDE you choose, such as Poisson, DiffusionReaction.",
        )
        parser.add_argument("--solution_case", type=int, default=1, help="type of PDE solution you choose")
        parser.add_argument(
            "--blocks_num",
            nargs="+",
            type=int,
            default=[4, 4],
            help="the number of blocks (tuple) for xi samples during domain decomposition training",
        )
        parser.add_argument("--scale", nargs="+", type=int, default=[5, 10], help="levels of locally refined samples")
        parser.add_argument(
            "--scale_resume", nargs="+", type=int, default=[3, 6], help="levels of locally refined samples (resumed)"
        )
        parser.add_argument("--lam", type=float, default=400, help="weight in loss function")
        parser.add_argument("--lr", type=float, default=0.05, help="Initial learning rate")
        parser.add_argument(
            "--epochs_Adam", type=int, default=50000, help="Number of epochs for Adam optimizer to train"
        )
        parser.add_argument(
            "--tol", type=float, default=1e-12, help="tolerance of train loss to stop the training process"
        )
        parser.add_argument(
            "--tol_change", type=float, default=0.0003, help="tolerance of train loss to stop the training process"
        )
        parser.add_argument("--resume", type=bool, default=False, help="put the path to resuming file if needed")
        parser.add_argument(
            "--ngs_boundary", type=int, default=4, help="number of Gauss quadrature points on boundary"
        )
        parser.add_argument("--ngs_interior", type=int, default=4, help="number of Gauss quadrature points on mesh")
        self.parser = parser

    def parse(self):
        args = self.parser.parse_args()
        args.cuda = not args.no_cuda and paddle.device.cuda.device_count() >= 1
        args.device = str(f"cuda:{args.cuda_index}" if paddle.device.cuda.device_count() >= 1 else "cpu").replace(
            "cuda", "gpu"
        )
        args.layers = [4, 6, 50, 50, 1]
        return args


if __name__ == "__main__":
    args = Options().parse()
    print(args)
