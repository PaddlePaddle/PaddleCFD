from functools import partial

import einops
import numpy as np
import paddle
from .paddle_utils import *

from .mixed_modules import SpaceTimeBlock, build_spacetime_block
from .spatial_modules import SubsampledLinear, hMLP_output, hMLP_stem


def build_avit(params):
    """Builds model from parameter file.

    General recipe is to build the spatial and temporal modules separately and then
    combine them in a model. Eventually the "stem" and "destem" should
    also be parameterized.
    """
    space_time_block = build_spacetime_block(params)
    model = AViT(
        patch_size=params.patch_size,
        embed_dim=params.embed_dim,
        processor_blocks=params.processor_blocks,
        n_states=params.n_states,
        override_block=space_time_block,
    )
    return model


class AViT(paddle.nn.Layer):
    """
    Naive model that interweaves spatial and temporal attention blocks. Temporal attention
    acts only on the time dimension.

    Args:
        patch_size (tuple): Size of the input patch
        embed_dim (int): Dimension of the embedding
        processor_blocks (int): Number of blocks (consisting of spatial mixing - temporal attention)
        n_states (int): Number of input state variables.
    """

    def __init__(
        self,
        patch_size=(16, 16),
        embed_dim=768,
        processor_blocks=8,
        n_states=6,
        override_block=None,
        drop_path=0.2,
    ):
        super().__init__()
        self.drop_path = drop_path
        self.dp = np.linspace(0, drop_path, processor_blocks)
        self.space_bag = SubsampledLinear(n_states, embed_dim // 4)
        self.embed = hMLP_stem(
            patch_size=patch_size, in_chans=embed_dim // 4, embed_dim=embed_dim
        )
        if override_block is not None:
            inner_block = override_block
        else:
            inner_block = partial(SpaceTimeBlock, hidden_dim=embed_dim)
        self.blocks = paddle.nn.LayerList(
            [inner_block(drop_path=self.dp[i]) for i in range(processor_blocks)]
        )
        self.debed = hMLP_output(
            patch_size=patch_size, embed_dim=embed_dim, out_chans=n_states
        )

    def expand_projections(self, expansion_amount):
        """Appends addition embeddings for finetuning on new data"""
        with paddle.no_grad():
            temp_space_bag = SubsampledLinear(
                dim_in=self.space_bag.dim_in + expansion_amount,
                dim_out=self.space_bag.dim_out,
            )
            temp_space_bag.weight[:, : self.space_bag.dim_in] = self.space_bag.weight
            temp_space_bag.bias[:] = self.space_bag.bias[:]
            self.space_bag = temp_space_bag
            out_head = paddle.nn.Conv2DTranspose(
                in_channels=self.debed.embed_dim // 4,
                out_channels=self.debed.out_chans + expansion_amount,
                kernel_size=4,
                stride=4,
            )
            temp_out_kernel = out_head.weight
            temp_out_bias = out_head.bias
            temp_out_kernel[:, : self.debed.out_chans, :, :] = self.debed.out_kernel
            temp_out_bias[: self.debed.out_chans] = self.debed.out_bias
            self.debed.out_kernel = temp_out_kernel
            self.debed.out_bias = temp_out_bias

    def freeze_middle(self):
        for param in self.parameters():
            param.stop_gradient = not False
        for param in self.space_bag.parameters():
            param.stop_gradient = not True
        self.debed.out_kernel.stop_gradient = not True
        self.debed.out_bias.stop_gradient = not True

    def freeze_processor(self):
        for param in self.parameters():
            param.stop_gradient = not False
        for param in self.space_bag.parameters():
            param.stop_gradient = not True
        for param in self.debed.parameters():
            param.stop_gradient = not True
        for param in self.embed.parameters():
            param.stop_gradient = not True

    def unfreeze(self):
        for param in self.parameters():
            param.stop_gradient = not True

    def forward(self, x, state_labels, bcs):
        T, B, C = x.shape[:3]
        with paddle.no_grad():
            """data_std, data_mean = torch.std_mean(x, dim=(0, -2, -1), keepdims=True)
            paddle has no std_mean api"""
            data_std = paddle.std(x=x, axis=(0, -2, -1), keepdim=True)
            data_mean = paddle.mean(x=x, axis=(0, -2, -1), keepdim=True)
            data_std = data_std + 1e-07
        x = (x - data_mean) / data_std
        x = einops.rearrange(x, "t b c h w -> t b h w c")
        x = self.space_bag(x, state_labels)
        x = einops.rearrange(x, "t b h w c -> (t b) c h w")
        x = self.embed(x)
        x = einops.rearrange(x, "(t b) c h w -> t b c h w", t=T)
        for blk in self.blocks:
            x = blk(x, bcs)
        x = einops.rearrange(x, "t b c h w -> (t b) c h w")
        x = self.debed(x, state_labels[0])
        x = einops.rearrange(x, "(t b) c h w -> t b c h w", t=T)
        x = x * data_std + data_mean
        return x[-1]


if __name__ == "__main__":
    print(paddle.device.is_compiled_with_cuda())
    model = AViT().cuda()
    for n, p in model.debed.named_parameters():
        print(n, p.shape)
    model.expand_projections(2)
    for n, p in model.debed.named_parameters():
        print(n, p.shape)
    T = 10
    bs = 4
    nx = 128
    ny = 128
    x = paddle.randn(T, bs, 2, nx, ny).cuda()
    print("xshape", x.shape)
    labels = [0, 1]
    y = model(x, labels)
    print("yshape", y.shape)
