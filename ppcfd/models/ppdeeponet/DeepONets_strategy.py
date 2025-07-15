from abc import ABC
from abc import abstractmethod


class DeepONetStrategy(ABC):
    """DeepONet building strategy."""

    def __init__(self, net):
        self.net = net

    @abstractmethod
    def build(self, layers_branch, layers_trunk):
        """Build branch and trunk nets."""

    @abstractmethod
    def call(self, x_func, x_loc):
        """Forward pass."""


class SingleOutputStrategy(DeepONetStrategy):
    """Single output build strategy is the standard build method."""

    def build(self, layers_branch, layers_trunk):
        if layers_branch[-1] != layers_trunk[-1]:
            raise AssertionError("Output sizes of branch net and trunk netdo not match.")
        branch = self.net.build_branch_net(layers_branch)
        trunk = self.net.build_trunk_net(layers_trunk)
        return branch, trunk

    def call(self, x_func, x_loc):
        x_func = self.net.branch(x_func)
        x_loc = self.net.activation_trunk(self.net.trunk(x_loc))
        if tuple(x_func.shape)[-1] != tuple(x_loc.shape)[-1]:
            raise AssertionError("Output sizes of branch net and trunk net do not match.")
        x = self.net.merge_branch_trunk(x_func, x_loc, 0)
        return x


class IndependentStrategy(DeepONetStrategy):
    """Directly use n independent DeepONets,
    and each DeepONet outputs only one function.
    """

    def build(self, layers_branch, layers_trunk):
        single_output_strategy = SingleOutputStrategy(self.net)
        branch, trunk = [], []
        for _ in range(self.net.num_output):
            branch_, trunk_ = single_output_strategy.build(layers_branch, layers_trunk)
            branch.append(branch_)
            trunk.append(trunk_)
        return branch, trunk

    def call(self, x_func, x_loc):
        xs = []
        for i in range(self.net.num_output):
            x_func_ = self.net.branch[i](x_func)
            x_loc_ = self.net.activation_trunk(self.net.trunk[i](x_loc))
            x = self.net.merge_branch_trunk(x_func_, x_loc_, i)
            xs.append(x)
        return self.net.concatenate_outputs(xs)


class SplitBranchStrategy(DeepONetStrategy):
    """Split the branch net and share the trunk net."""

    def build(self, layers_branch, layers_trunk):
        if layers_branch[-1] % self.net.num_output != 0:
            raise AssertionError(f"Output size of the branch net is notevenly divisible by {self.net.num_output}.")
        if layers_branch[-1] / self.net.num_output != layers_trunk[-1]:
            raise AssertionError(
                f"Output size of the trunk net does notequal to {layers_branch[-1] // self.net.num_output}."
            )
        return self.net.build_branch_net(layers_branch), self.net.build_trunk_net(layers_trunk)

    def call(self, x_func, x_loc):
        x_func = self.net.branch(x_func)
        x_loc = self.net.activation_trunk(self.net.trunk(x_loc))
        shift = 0
        size = tuple(x_loc.shape)[1]
        xs = []
        for i in range(self.net.num_output):
            x_func_ = x_func[:, shift : shift + size]
            x = self.net.merge_branch_trunk(x_func_, x_loc, i)
            xs.append(x)
            shift += size
        return self.net.concatenate_outputs(xs)


class SplitTrunkStrategy(DeepONetStrategy):
    """Split the trunk net and share the branch net."""

    def build(self, layers_branch, layers_trunk):
        if layers_trunk[-1] % self.net.num_output != 0:
            raise AssertionError(f"Output size of the trunk net is notevenly divisible by {self.net.num_output}.")
        if layers_trunk[-1] / self.net.num_output != layers_branch[-1]:
            raise AssertionError(
                f"Output size of the branch net does notequal to {layers_trunk[-1] // self.net.num_output}."
            )
        return self.net.build_branch_net(layers_branch), self.net.build_trunk_net(layers_trunk)

    def call(self, x_func, x_loc):
        x_func = self.net.branch(x_func)
        x_loc = self.net.activation_trunk(self.net.trunk(x_loc))
        shift = 0
        size = tuple(x_func.shape)[1]
        xs = []
        for i in range(self.net.num_output):
            x_loc_ = x_loc[:, shift : shift + size]
            x = self.net.merge_branch_trunk(x_func, x_loc_, i)
            xs.append(x)
            shift += size
        return self.net.concatenate_outputs(xs)


class SplitBothStrategy(DeepONetStrategy):
    """Split the outputs of both the branch net and
    the trunk net into n groups, and then the kth group
    outputs the kth solution.
    """

    def build(self, layers_branch, layers_trunk):
        if layers_branch[-1] != layers_trunk[-1]:
            raise AssertionError("Output sizes of branch net and trunk netdo not match.")
        if layers_branch[-1] % self.net.num_output != 0:
            raise AssertionError(f"Output size of the branch net is notevenly divisible by {self.net.num_output}.")
        single_output_strategy = SingleOutputStrategy(self.net)
        return single_output_strategy.build(layers_branch, layers_trunk)

    def call(self, x_func, x_loc):
        x_func = self.net.branch(x_func)
        x_loc = self.net.activation_trunk(self.net.trunk(x_loc))
        shift = 0
        size = tuple(x_func.shape)[1] // self.net.num_output
        xs = []
        for i in range(self.net.num_output):
            x_func_ = x_func[:, shift : shift + size]
            x_loc_ = x_loc[:, shift : shift + size]
            x = self.net.merge_branch_trunk(x_func_, x_loc_, i)
            xs.append(x)
            shift += size
        return self.net.concatenate_outputs(xs)
