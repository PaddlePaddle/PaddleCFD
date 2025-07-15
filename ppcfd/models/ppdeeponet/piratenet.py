from typing import Callable
from typing import Dict
from typing import Optional
from typing import Tuple
from typing import Union

import numpy as np
import paddle
import paddle.nn as nn


act_func_dict = {
    "elu": nn.ELU(),
    "relu": nn.ReLU(),
    "selu": nn.SELU(),
    "gelu": nn.GELU(),
    "leaky_relu": nn.LeakyReLU(),
    "sigmoid": nn.Sigmoid(),
    "silu": nn.Silu(),
    "tanh": nn.Tanh(),
    "identity": nn.Identity(),
}


def get_activation(act_name: str) -> Callable:
    """Get activation function according to act_name.

    Args:
        act_name (str): Name of activation, such as "tanh".

    Returns:
        Callable: Paddle activation function.
    """
    if act_name.lower() not in act_func_dict:
        raise ValueError(f"act_name({act_name}) not found in act_func_dict")

    act_layer = act_func_dict[act_name.lower()]
    if isinstance(act_layer, type) and act_name != "stan":
        # Is a activation class but not a instance of it, instantiate manually(except for 'Stan')
        return act_layer()

    return act_layer


def concat_to_tensor(data_dict: Dict[str, paddle.Tensor], keys: Tuple[str, ...], axis=-1) -> Tuple[paddle.Tensor, ...]:
    """Concatenate tensors from dict in the order of given keys.

    Args:
        data_dict (Dict[str, paddle.Tensor]): Dict contains tensor.
        keys (Tuple[str, ...]): Keys tensor fetched from.
        axis (int, optional): Axis concatenate at. Defaults to -1.

    Returns:
        Tuple[paddle.Tensor, ...]: Concatenated tensor.

    Examples:
        >>> import paddle
        >>> import ppsci
        >>> model = ppsci.arch.Arch()
        >>> # fetch one tensor
        >>> out = model.concat_to_tensor({'x':paddle.rand([64, 64, 1])}, ('x',))
        >>> print(out.dtype, out.shape)
        paddle.float32 [64, 64, 1]
        >>> # fetch more tensors
        >>> out = model.concat_to_tensor(
        ...     {'x1':paddle.rand([64, 64, 1]), 'x2':paddle.rand([64, 64, 1])},
        ...     ('x1', 'x2'),
        ...     axis=2)
        >>> print(out.dtype, out.shape)
        paddle.float32 [64, 64, 2]

    """
    if len(keys) == 1:
        return data_dict[keys[0]]
    data = [data_dict[key] for key in keys]
    return paddle.concat(data, axis)


def split_to_dict(data_tensor: paddle.Tensor, keys: Tuple[str, ...], axis=-1) -> Dict[str, paddle.Tensor]:
    """Split tensor and wrap into a dict by given keys.

    Args:
        data_tensor (paddle.Tensor): Tensor to be split.
        keys (Tuple[str, ...]): Keys tensor mapping to.
        axis (int, optional): Axis split at. Defaults to -1.

    Returns:
        Dict[str, paddle.Tensor]: Dict contains tensor.

    Examples:
        >>> import paddle
        >>> import ppsci
        >>> model = ppsci.arch.Arch()
        >>> # split one tensor
        >>> out = model.split_to_dict(paddle.rand([64, 64, 1]), ('x',))
        >>> for k, v in out.items():
        ...     print(f"{k} {v.dtype} {v.shape}")
        x paddle.float32 [64, 64, 1]
        >>> # split more tensors
        >>> out = model.split_to_dict(paddle.rand([64, 64, 2]), ('x1', 'x2'), axis=2)
        >>> for k, v in out.items():
        ...     print(f"{k} {v.dtype} {v.shape}")
        x1 paddle.float32 [64, 64, 1]
        x2 paddle.float32 [64, 64, 1]

    """
    if len(keys) == 1:
        return {keys[0]: data_tensor}
    data = paddle.split(data_tensor, len(keys), axis=axis)
    return {key: data[i] for i, key in enumerate(keys)}


class FourierEmbedding(nn.Layer):
    def __init__(self, in_features, out_features, scale):
        super().__init__()
        if out_features % 2 != 0:
            raise ValueError(f"out_features must be even, but got {out_features}.")

        self.kernel = self.create_parameter(
            [in_features, out_features // 2],
            default_initializer=nn.initializer.Normal(std=scale),
        )

    def forward(self, x: paddle.Tensor):
        y = paddle.concat(
            [
                paddle.cos(x @ self.kernel),
                paddle.sin(x @ self.kernel),
            ],
            axis=-1,
        )
        return y


class PirateNetBlock(nn.Layer):
    r"""Basic block of PirateNet.

    $$
    \begin{align*}
        \Phi(\mathbf{x})=\left[\begin{array}{l}
        \cos (\mathbf{B} \mathbf{x}) \\
        \sin (\mathbf{B} \mathbf{x})
        \end{array}\right] \\
        \mathbf{f}^{(l)} & =\sigma\left(\mathbf{W}_1^{(l)} \mathbf{x}^{(l)}+\mathbf{b}_1^{(l)}\right) \\
        \mathbf{z}_1^{(l)} & =\mathbf{f}^{(l)} \odot \mathbf{U}+\left(1-\mathbf{f}^{(l)}\right) \odot \mathbf{V} \\
        \mathbf{g}^{(l)} & =\sigma\left(\mathbf{W}_2^{(l)} \mathbf{z}_1^{(l)}+\mathbf{b}_2^{(l)}\right) \\
        \mathbf{z}_2^{(l)} & =\mathbf{g}^{(l)} \odot \mathbf{U}+\left(1-\mathbf{g}^{(l)}\right) \odot \mathbf{V} \\
        \mathbf{h}^{(l)} & =\sigma\left(\mathbf{W}_3^{(l)} \mathbf{z}_2^{(l)}+\mathbf{b}_3^{(l)}\right) \\
        \mathbf{x}^{(l+1)} & =\alpha^{(l)} \cdot \mathbf{h}^{(l)}+\left(1-\alpha^{(l)}\right) \cdot \mathbf{x}^{(l)}
    \end{align*}
    $$

    Args:
        input_dim (int): Input dimension.
        embed_dim (int): Embedding dimension.
        activation (str, optional): Name of activation function. Defaults to "tanh".
        random_weight (Optional[Dict[str, float]]): Mean and std of random weight
            factorization layer, e.g. {"mean": 0.5, "std: 0.1"}. Defaults to None.
    """

    def __init__(
        self,
        input_dim: int,
        embed_dim: int,
        activation: str = "tanh",
        random_weight: Optional[Dict[str, float]] = None,
    ):
        super().__init__()
        weight_attr = paddle.ParamAttr(
            initializer=paddle.nn.initializer.Uniform(low=-np.sqrt(1 / input_dim), high=np.sqrt(1 / input_dim))
        )
        bias_attr = paddle.ParamAttr(
            initializer=paddle.nn.initializer.Uniform(low=-np.sqrt(1 / input_dim), high=np.sqrt(1 / input_dim))
        )
        self.linear1 = (
            nn.Linear(input_dim, embed_dim, weight_attr=weight_attr, bias_attr=bias_attr)
            if random_weight is None
            else RandomWeightFactorization(
                input_dim,
                embed_dim,
                mean=random_weight["mean"],
                std=random_weight["std"],
            )
        )
        self.linear2 = (
            nn.Linear(embed_dim, embed_dim, weight_attr=weight_attr, bias_attr=bias_attr)
            if random_weight is None
            else RandomWeightFactorization(
                embed_dim,
                embed_dim,
                mean=random_weight["mean"],
                std=random_weight["std"],
            )
        )
        self.linear3 = (
            nn.Linear(embed_dim, embed_dim, weight_attr=weight_attr, bias_attr=bias_attr)
            if random_weight is None
            else RandomWeightFactorization(
                embed_dim,
                embed_dim,
                mean=random_weight["mean"],
                std=random_weight["std"],
            )
        )
        self.alpha = self.create_parameter(
            [
                1,
            ],
            default_initializer=nn.initializer.Constant(0),
        )
        self.act1 = activation
        self.act2 = activation
        self.act3 = activation

    def forward(self, x, u, v):
        f = self.act1(self.linear1(x))
        z1 = f * u + (1 - f) * v
        g = self.act2(self.linear2(z1))
        z2 = g * u + (1 - g) * v
        h = self.act3(self.linear3(z2))
        out = self.alpha * h + (1 - self.alpha) * x
        return out


class PirateNet(nn.Layer):
    r"""PirateNet.

    [PIRATENETS: PHYSICS-INFORMED DEEP LEARNING WITHRESIDUAL ADAPTIVE NETWORKS](https://arxiv.org/pdf/2402.00326.pdf)

    $$
    \begin{align*}
        \Phi(\mathbf{x}) &= \left[\begin{array}{l}
        \cos (\mathbf{B} \mathbf{x}) \\
        \sin (\mathbf{B} \mathbf{x})
        \end{array}\right] \\
        \mathbf{f}^{(l)} &= \sigma\left(\mathbf{W}_1^{(l)} \mathbf{x}^{(l)}+\mathbf{b}_1^{(l)}\right) \\
        \mathbf{z}_1^{(l)} &= \mathbf{f}^{(l)} \odot \mathbf{U}+\left(1-\mathbf{f}^{(l)}\right) \odot \mathbf{V} \\
        \mathbf{g}^{(l)} &= \sigma\left(\mathbf{W}_2^{(l)} \mathbf{z}_1^{(l)}+\mathbf{b}_2^{(l)}\right) \\
        \mathbf{z}_2^{(l)} &= \mathbf{g}^{(l)} \odot \mathbf{U}+\left(1-\mathbf{g}^{(l)}\right) \odot \mathbf{V} \\
        \mathbf{h}^{(l)} &= \sigma\left(\mathbf{W}_3^{(l)} \mathbf{z}_2^{(l)}+\mathbf{b}_3^{(l)}\right) \\
        \mathbf{x}^{(l+1)} &= \text{PirateBlock}^{(l)}\left(\mathbf{x}^{(l)}\right), l=1...L-1\\
        \mathbf{u}_\theta &= \mathbf{W}^{(L+1)} \mathbf{x}^{(L)}
    \end{align*}
    $$

    Args:
        input_keys (Tuple[str, ...]): Name of input keys, such as ("x", "y", "z").
        output_keys (Tuple[str, ...]): Name of output keys, such as ("u", "v", "w").
        num_blocks (int): Number of PirateBlocks.
        hidden_size (Union[int, Tuple[int, ...]]): Number of hidden size.
            An integer for all layers, or list of integer specify each layer's size.
        activation (str, optional): Name of activation function. Defaults to "tanh".
        weight_norm (bool, optional): Whether to apply weight norm on parameter(s). Defaults to False.
        input_dim (Optional[int]): Number of input's dimension. Defaults to None.
        output_dim (Optional[int]): Number of output's dimension. Defaults to None.
        periods (Optional[Dict[int, Tuple[float, bool]]]): Period of each input key,
            input in given channel will be period embedded if specified, each tuple of
            periods list is [period, trainable]. Defaults to None.
        fourier (Optional[Dict[str, Union[float, int]]]): Random fourier feature embedding,
            e.g. {'dim': 256, 'scale': 1.0}. Defaults to None.
        random_weight (Optional[Dict[str, float]]): Mean and std of random weight
            factorization layer, e.g. {"mean": 0.5, "std: 0.1"}. Defaults to None.

    Examples:
        >>> import paddle
        >>> import ppsci
        >>> model = ppsci.arch.PirateNet(
        ...     input_keys=("x", "y"),
        ...     output_keys=("u", "v"),
        ...     num_blocks=3,
        ...     hidden_size=256,
        ...     fourier={'dim': 256, 'scale': 1.0},
        ... )
        >>> input_dict = {"x": paddle.rand([64, 1]),
        ...               "y": paddle.rand([64, 1])}
        >>> output_dict = model(input_dict)
        >>> print(output_dict["u"].shape)
        [64, 1]
        >>> print(output_dict["v"].shape)
        [64, 1]
    """

    def __init__(
        self,
        input_keys: Tuple[str, ...],
        output_keys: Tuple[str, ...],
        num_blocks: int,
        hidden_size: int,
        activation: str = "tanh",
        weight_norm: bool = False,
        input_dim: Optional[int] = None,
        output_dim: Optional[int] = None,
        periods: Optional[Dict[int, Tuple[float, bool]]] = None,
        fourier: Optional[Dict[str, Union[float, int]]] = None,
        random_weight: Optional[Dict[str, float]] = None,
    ):
        super().__init__()
        self.input_keys = input_keys
        self.output_keys = output_keys
        self.blocks = []
        self.periods = periods
        self.fourier = fourier
        if periods:
            self.period_emb = PeriodEmbedding(periods)

        if isinstance(hidden_size, int):
            if not isinstance(num_blocks, int):
                raise ValueError("num_blocks should be an int")
            hidden_size = [hidden_size] * num_blocks
        else:
            raise ValueError(f"hidden_size should be int, but got {type(hidden_size)}")

        # initialize FC layer(s)
        cur_size = len(self.input_keys) if input_dim is None else input_dim
        if input_dim is None and periods:
            # period embedded channel(s) will be doubled automatically
            # if input_dim is not specified
            cur_size += len(periods)

        if fourier:
            self.fourier_emb = FourierEmbedding(cur_size, fourier["dim"], fourier["scale"])
            cur_size = fourier["dim"]
        else:
            self.linear_emb = nn.Linear(cur_size, hidden_size[0])
            cur_size = hidden_size[0]

        self.embed_u = nn.Sequential(
            (
                WeightNormLinear(cur_size, hidden_size[0])
                if weight_norm
                else (
                    nn.Linear(cur_size, hidden_size[0])
                    if random_weight is None
                    else RandomWeightFactorization(
                        cur_size,
                        hidden_size[0],
                        mean=random_weight["mean"],
                        std=random_weight["std"],
                    )
                )
            ),
            (get_activation(activation)),
        )
        self.embed_v = nn.Sequential(
            (
                WeightNormLinear(cur_size, hidden_size[0])
                if weight_norm
                else (
                    nn.Linear(cur_size, hidden_size[0])
                    if random_weight is None
                    else RandomWeightFactorization(
                        cur_size,
                        hidden_size[0],
                        mean=random_weight["mean"],
                        std=random_weight["std"],
                    )
                )
            ),
            (get_activation(activation)),
        )

        for i, _size in enumerate(hidden_size):
            self.blocks.append(
                PirateNetBlock(
                    cur_size,
                    _size,
                    activation=activation,
                    random_weight=random_weight,
                )
            )
            cur_size = _size

        self.blocks = nn.LayerList(self.blocks)
        if random_weight:
            self.last_fc = RandomWeightFactorization(
                cur_size,
                len(self.output_keys) if output_dim is None else output_dim,
                mean=random_weight["mean"],
                std=random_weight["std"],
            )
        else:
            self.last_fc = nn.Linear(
                cur_size,
                len(self.output_keys) if output_dim is None else output_dim,
            )

    def forward_tensor(self, x):
        u = self.embed_u(x)
        v = self.embed_v(x)

        y = x
        for i, block in enumerate(self.blocks):
            y = block(y, u, v)

        y = self.last_fc(y)
        return y

    def forward(self, x):

        if self.periods:
            x = self.period_emb(x)

        y = concat_to_tensor(x, self.input_keys, axis=-1)

        if self.fourier:
            y = self.fourier_emb(y)
        else:
            y = self.linear_emb(y)

        y = self.forward_tensor(y)
        y = split_to_dict(y, self.output_keys, axis=-1)

        return y
