from functools import partial

import paddle
from paddle_utils import *

from kappamodules.init import init_with_scheme
from kappamodules.layers import DropPath, LayerNorm1d, LayerNorm2d, LayerNorm3d
from kappamodules.mlp import Mlp


class ConvNextBlock(paddle.nn.Layer):
    def __init__(
        self,
        dim,
        drop_path=0.0,
        conv_ctor=paddle.nn.Conv2D,
        norm_ctor=LayerNorm2d,
        kernel_size=7,
        depthwise=True,
        global_response_norm=True,
    ):
        super().__init__()
        assert kernel_size % 2 == 1
        self.drop_path = DropPath(drop_prob=drop_path)
        self.conv = conv_ctor(
            in_channels=dim,
            out_channels=dim,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            groups=dim if depthwise else 1,
        )
        if isinstance(self.conv, paddle.nn.Conv1D):
            ndim = 1
        elif isinstance(self.conv, paddle.nn.Conv2D):
            ndim = 2
        elif isinstance(self.conv, paddle.nn.Conv3D):
            ndim = 3
        else:
            raise NotImplementedError
        self.norm = norm_ctor(dim)
        self.mlp = Mlp(
            in_dim=dim,
            hidden_dim=dim * 4,
            ndim=ndim,
            use_global_response_norm=global_response_norm,
        )

    def _forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.mlp(x)
        return x

    def forward(self, x):
        return self.drop_path(x, self._forward)


class ConvNextStage(paddle.nn.Layer):
    def __init__(
        self,
        input_dim,
        output_dim,
        depth,
        drop_path_rates=None,
        conv_ctor=paddle.nn.Conv2D,
        norm_ctor=LayerNorm2d,
        kernel_size=7,
        depthwise=True,
        global_response_norm=True,
    ):
        super().__init__()
        if input_dim != output_dim:
            self.downsampling = paddle.nn.Sequential(
                norm_ctor(input_dim),
                conv_ctor(input_dim, output_dim, kernel_size=2, stride=2),
            )
        else:
            self.downsampling = paddle.nn.Identity()
        drop_path_rates = drop_path_rates or [0.0 * depth]
        self.blocks = paddle.nn.Sequential(
            *[
                ConvNextBlock(
                    dim=output_dim,
                    conv_ctor=conv_ctor,
                    norm_ctor=norm_ctor,
                    drop_path=drop_path_rates[i],
                    kernel_size=kernel_size,
                    depthwise=depthwise,
                    global_response_norm=global_response_norm,
                )
                for i in range(depth)
            ]
        )

    def forward(self, x):
        x = self.downsampling(x)
        x = self.blocks(x)
        return x


class ConvNext(paddle.nn.Layer):
    """minimal version of timm.models.convnext.ConvNext that works with 3d"""

    def __init__(
        self,
        patch_size,
        input_dim,
        depths,
        dims,
        drop_path_rate=0.0,
        drop_path_decay=True,
        kernel_size=7,
        depthwise=True,
        global_response_norm=True,
        ndim=2,
        eps=1e-06,
        init_weights="truncnormal002",
    ):
        super().__init__()
        assert len(dims) == len(depths)
        self.init_weights = init_weights
        if ndim == 1:
            conv_ctor = paddle.nn.Conv1D
            norm_ctor = partial(LayerNorm1d, eps=eps)
        elif ndim == 2:
            conv_ctor = paddle.nn.Conv2D
            norm_ctor = partial(LayerNorm2d, eps=eps)
        elif ndim == 3:
            conv_ctor = paddle.nn.Conv3D
            norm_ctor = partial(LayerNorm3d, eps=eps)
        else:
            raise NotImplementedError
        self.stem = paddle.nn.Sequential(
            conv_ctor(input_dim, dims[0], kernel_size=patch_size, stride=patch_size),
            norm_ctor(dims[0]),
        )
        if drop_path_decay:
            dprs = [
                dpr.tolist()
                for dpr in paddle.linspace(0, drop_path_rate, sum(depths)).split(depths)
            ]
        else:
            dprs = [
                dpr.tolist()
                for dpr in paddle.to_tensor(drop_path_rate)
                .tile([sum(depths)])
                .split(depths)
            ]
        self.stages = paddle.nn.LayerList(
            [
                ConvNextStage(
                    input_dim=dims[max(0, i - 1)],
                    output_dim=dims[i],
                    depth=depths[i],
                    drop_path_rates=dprs[i],
                    conv_ctor=conv_ctor,
                    norm_ctor=norm_ctor,
                    kernel_size=kernel_size,
                    depthwise=depthwise,
                    global_response_norm=global_response_norm,
                )
                for i in range(len(dims))
            ]
        )
        self.reset_parameters()

    def reset_parameters(self):
        init_with_scheme(self, scheme=self.init_weights)

    def forward(self, x):
        x = self.stem(x)
        for stage in self.stages:
            x = stage(x)
        return x
