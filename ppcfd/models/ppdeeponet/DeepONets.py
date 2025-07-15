import paddle


try:
    from FunActivation import FunActivation
except ImportError:
    from .FunActivation import FunActivation
try:
    from Networks.DeepONets_strategy import IndependentStrategy
    from Networks.DeepONets_strategy import SingleOutputStrategy
    from Networks.DeepONets_strategy import SplitBothStrategy
    from Networks.DeepONets_strategy import SplitBranchStrategy
    from Networks.DeepONets_strategy import SplitTrunkStrategy
except ImportError:
    from DeepONets_strategy import IndependentStrategy
    from DeepONets_strategy import SingleOutputStrategy
    from DeepONets_strategy import SplitBothStrategy
    from DeepONets_strategy import SplitBranchStrategy
    from DeepONets_strategy import SplitTrunkStrategy
try:
    from Networks.FCNet import FCNet
except ImportError:
    from FCNet import FCNet


class DeepONetBatch(paddle.nn.Layer):

    def __init__(
        self,
        num_output,
        layers_branch,
        layers_trunk,
        activation_branch: str = None,
        activation_trunk: str = None,
        multi_output_strategy: str = None,
        kernel_init=None,
        dtype=None,
        device="cpu",
    ) -> None:
        super(DeepONetBatch, self).__init__()
        """Deep operator network for dataset in the format of Batch product.
        """
        self.num_output = num_output
        self.kernel_init = kernel_init
        self.dtype = dtype
        self.device = device
        self.activation_branch = FunActivation()(activation_branch)
        self.activation_trunk = FunActivation()(activation_trunk)
        if self.num_output == 1:
            if multi_output_strategy is not None:
                raise ValueError("num_outputs is set to 1,but multi_output_strategy is not None.")
        elif multi_output_strategy is None:
            multi_output_strategy = "independent"
            print(
                f'Warning: There are {num_output} outputs, but no multi_output_strategy selected.Use "independent" as the multi_output_strategy.'
            )
        self.multi_output_strategy = {
            None: SingleOutputStrategy,
            "independent": IndependentStrategy,
            "split_both": SplitBothStrategy,
            "split_branch": SplitBranchStrategy,
            "split_trunk": SplitTrunkStrategy,
        }[multi_output_strategy](self)
        self.branch, self.trunk = self.multi_output_strategy.build(layers_branch, layers_trunk)
        self.b = paddle.nn.ParameterList(
            parameters=[
                paddle.base.framework.EagerParamBase.from_tensor(tensor=paddle.to_tensor(data=0.0))
                for _ in range(self.num_output)
            ]
        )

    def build_branch_net(self, layers_branch):
        if callable(layers_branch[0]):
            return layers_branch[0].to(self.device)
        else:
            return FCNet(layers_branch, self.activation_branch, self.dtype, self.kernel_init).to(self.device)

    def build_trunk_net(self, layer_sizes_trunk):
        return FCNet(layer_sizes_trunk, self.activation_trunk, self.dtype, self.kernel_init).to(self.device)

    def merge_branch_trunk(self, x_func, x_loc, index):
        """
        Input:
            x_loc: size(n_batch, n_mesh, out_size)
            x_func: size(n_batch, out_size)
        """
        y = paddle.einsum("bmi,bi->bm", x_loc, x_func)
        y += self.b[index]
        return y.unsqueeze(axis=-1)

    @staticmethod
    def concatenate_outputs(ys):
        return paddle.concat(x=ys, axis=-1)

    def forward(self, x_loc, x_func):
        """
        Input:
            x_loc: size(n_batch, n_mesh, dx)
            x_func: size(n_batch, n_mesh)
        """
        x = self.multi_output_strategy.call(x_func, x_loc)
        return x


class DeepONetCartesianProd(paddle.nn.Layer):

    def __init__(
        self,
        num_output,
        layers_branch,
        layers_trunk,
        activation_branch: str = None,
        activation_trunk: str = None,
        multi_output_strategy: str = None,
        kernel_init=None,
        dtype=None,
        device="cpu",
    ) -> None:
        super(DeepONetCartesianProd, self).__init__()
        """Deep operator network for dataset in the format of Cartesian product.
        """
        self.num_output = num_output
        self.kernel_init = kernel_init
        self.dtype = dtype
        self.device = device
        self.activation_branch = FunActivation()(activation_branch)
        self.activation_trunk = FunActivation()(activation_trunk)
        if self.num_output == 1:
            if multi_output_strategy is not None:
                raise ValueError("num_outputs is set to 1,but multi_output_strategy is not None.")
        elif multi_output_strategy is None:
            multi_output_strategy = "independent"
            print(
                f'Warning: There are {num_output} outputs, but no multi_output_strategy selected.Use "independent" as the multi_output_strategy.'
            )
        self.multi_output_strategy = {
            None: SingleOutputStrategy,
            "independent": IndependentStrategy,
            "split_both": SplitBothStrategy,
            "split_branch": SplitBranchStrategy,
            "split_trunk": SplitTrunkStrategy,
        }[multi_output_strategy](self)
        self.branch, self.trunk = self.multi_output_strategy.build(layers_branch, layers_trunk)
        self.b = paddle.nn.ParameterList(
            parameters=[
                paddle.base.framework.EagerParamBase.from_tensor(tensor=paddle.to_tensor(data=0.0))
                for _ in range(self.num_output)
            ]
        )

    def build_branch_net(self, layers_branch):
        if callable(layers_branch[0]):
            return layers_branch[0].to(self.device)
        else:
            return FCNet(layers_branch, self.activation_branch, self.dtype, self.kernel_init).to(self.device)

    def build_trunk_net(self, layer_sizes_trunk):
        return FCNet(layer_sizes_trunk, self.activation_trunk, self.dtype, self.kernel_init).to(self.device)

    def merge_branch_trunk(self, x_func, x_loc, index):
        """
        Input:
            x_func: size(n_batch, out_size)
            x_loc: size(n_mesh, out_size)
        """
        y = paddle.einsum("bi,ni->bn", x_func, x_loc)
        y += self.b[index]
        return y.unsqueeze(axis=-1)

    @staticmethod
    def concatenate_outputs(ys):
        return paddle.stack(x=ys, axis=2)

    def forward(self, x_loc, x_func):
        """
        Input:
            x_loc: size(n_mesh, dx)
            x_func: size(n_batch, n_mesh)
        """
        x = self.multi_output_strategy.call(x_func, x_loc)
        return x
