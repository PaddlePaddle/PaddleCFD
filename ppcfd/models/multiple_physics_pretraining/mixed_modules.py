import math
from functools import partial

import einops
import numpy as np
import paddle

from .spatial_modules import build_space_block
from .time_modules import AttentionBlock, build_time_block


def build_spacetime_block(params):
    """
    Builds a spacetime block from the parameter file.
    """
    if params.block_type == "axial":
        space_block = build_space_block(params)
        time_block = build_time_block(params)
        return partial(
            SpaceTimeBlock,
            params.embed_dim,
            params.num_heads,
            space_override=space_block,
            time_override=time_block,
            gradient_checkpointing=params.gradient_checkpointing,
        )
    else:
        raise NotImplementedError


class SpaceTimeBlock(paddle.nn.Layer):
    """
    Alternates spatial and temporal processing. Current code base uses
    1D attention over each axis. Spatial axes share weights.

    Note: MLP is in spatial block.
    """

    def __init__(
        self,
        hidden_dim=768,
        num_heads=12,
        drop_path=0.0,
        space_override=None,
        time_override=None,
        gradient_checkpointing=False,
    ):
        super().__init__()
        self.gradient_checkpointing = gradient_checkpointing
        if space_override is not None:
            self.spatial = space_override(drop_path=drop_path)
        if time_override is not None:
            self.temporal = time_override(drop_path=drop_path)
        else:
            self.temporal = AttentionBlock(hidden_dim, num_heads, drop_path=drop_path)

    def forward(self, x, bcs):
        T, B, C, H, W = x.shape
        if self.gradient_checkpointing:
            wrapped_temporal = partial(self.temporal)
            x = paddle.distributed.fleet.utils.recompute(
                wrapped_temporal, x, use_reentrant=False
            )
        else:
            x = self.temporal(x)
        x = einops.rearrange(x, "t b c h w -> (t b) c h w")
        if self.gradient_checkpointing:
            wrapped_spatial = partial(self.spatial)
            x = paddle.distributed.fleet.utils.recompute(
                wrapped_spatial, x, bcs, use_reentrant=False
            )
        else:
            x = self.spatial(x, bcs)
        x = einops.rearrange(x, "(t b) c h w -> t b c h w", t=T)
        return x
