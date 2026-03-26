import paddle

from kappamodules.init.functional import init_norms_as_noaffine

from .vit_block import VitBlock


class VitBlocks(paddle.nn.Layer):
    def __init__(
        self, depth, dim, num_heads, pooling=None, use_last_norm=True, eps=1e-06
    ):
        super().__init__()
        self.depth = depth
        self.dim = dim
        self.num_heads = num_heads
        self.pooling = pooling
        self.use_last_norm = use_last_norm
        self.blocks = paddle.nn.Sequential(
            *[VitBlock(dim=dim, num_heads=num_heads, eps=eps) for _ in range(depth)]
        )
        if use_last_norm:
            self.norm = paddle.nn.LayerNorm(dim, eps=eps)
        else:
            self.norm = paddle.nn.Identity()
        self.reset_parameters()

    def reset_parameters(self):
        if self.blocks[-1].init_norms == "torch":
            pass
        elif self.blocks[-1].init_norms == "nonaffine":
            init_norms_as_noaffine(self.norm)
        else:
            raise NotImplementedError

    def forward(self, x):
        x = self.blocks(x)
        if self.pooling is not None:
            x = self.pooling(x)
        x = self.norm(x)
        return x
