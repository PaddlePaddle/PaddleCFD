import einops
import numpy as np
import paddle

from kappamodules.init import init_truncnormal_zero_bias
from kappamodules.utils.param_checking import to_ntuple


class VitPatchEmbed(paddle.nn.Layer):
    def __init__(
        self,
        dim,
        num_channels,
        resolution,
        patch_size,
        stride=None,
        norm_ctor=None,
        flatten=False,
        init_weights="xavier_uniform",
    ):
        super().__init__()
        self.dim = dim
        self.num_channels = num_channels
        self.resolution = resolution
        self.init_weights = init_weights
        self.flatten = flatten
        self.ndim = len(resolution)
        self.patch_size = to_ntuple(patch_size, n=self.ndim)
        if stride is None:
            self.stride = self.patch_size
        else:
            self.stride = to_ntuple(stride, n=self.ndim)
        for i in range(self.ndim):
            assert (
                resolution[i] % self.patch_size[i] == 0
            ), f"resolution[{i}] % patch_size[{i}] != 0 (resolution={resolution} patch_size={patch_size})"
        self.seqlens = [(resolution[i] // self.patch_size[i]) for i in range(self.ndim)]
        if self.patch_size == self.stride:
            self.num_patches = int(np.prod(self.seqlens))
        else:
            if self.ndim == 1:
                conv_func = paddle.nn.functional.conv1d
            elif self.ndim == 2:
                conv_func = paddle.nn.functional.conv2d
            elif self.ndim == 3:
                conv_func = paddle.nn.functional.conv3d
            else:
                raise NotImplementedError
            self.num_patches = conv_func(
                input=paddle.zeros(1, 1, *resolution),
                weight=paddle.zeros(1, 1, *self.patch_size),
                stride=self.stride,
            ).size
        if self.ndim == 1:
            conv_ctor = paddle.nn.Conv1d
        elif self.ndim == 2:
            conv_ctor = paddle.nn.Conv2d
        elif self.ndim == 3:
            conv_ctor = paddle.nn.Conv3d
        else:
            raise NotImplementedError
        self.proj = conv_ctor(
            num_channels, dim, kernel_size=self.patch_size, stride=self.stride
        )
        self.norm = paddle.nn.Identity() if norm_ctor is None else norm_ctor(dim)
        self.reset_parameters()

    def reset_parameters(self):
        if self.init_weights == "paddle":
            pass
        elif self.init_weights == "xavier_uniform":
            w = self.proj.weight.data
            paddle.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
            paddle.nn.init.zeros_(self.proj.bias)
        elif self.init_weights in ["truncnormal", "truncnormal002"]:
            self.apply(init_truncnormal_zero_bias)
        else:
            raise NotImplementedError

    def interpolate_weights(self, weights):
        return self.interpolate_weights_from_different_patchsize(weights)

    def interpolate_weights_from_different_patchsize(self, weights):
        assert (
            weights.ndim == self.proj.weight.ndim
        ), f"can't interpolate from different ndim (e.g. cant interpolate 2D PatchEmbed to 3D PatchEmbed)"
        src_patch_size = weights.shape[2:]
        if self.patch_size == src_patch_size:
            return weights
        if self.ndim == 1:
            weights = paddle.nn.functional.interpolate(
                weights, size=self.patch_size, mode="linear"
            )
        elif self.ndim == 2:
            weights = paddle.nn.functional.interpolate(
                weights, size=self.patch_size, mode="bicubic"
            )
        elif self.ndim == 3:
            weights = paddle.nn.functional.interpolate(
                weights, size=self.patch_size, mode="trilinear"
            )
        else:
            raise NotImplementedError
        return weights

    def forward(self, x):
        assert all(
            x.size(i + 2) % self.patch_size[i] == 0 for i in range(self.ndim)
        ), f"x.shape={x.shape} incompatible with patch_size={self.patch_size}"
        x = self.proj(x)
        if self.flatten:
            x = einops.rearrange(x, "b c ... -> b (...) c")
        else:
            x = einops.rearrange(x, "b c ... -> b ... c")
        x = self.norm(x)
        return x
