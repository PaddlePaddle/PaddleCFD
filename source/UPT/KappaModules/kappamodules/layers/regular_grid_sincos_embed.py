import paddle

from kappamodules.functional.pos_embed import get_sincos_pos_embed_from_seqlens


class RegularGridSincosEmbed(paddle.nn.Layer):
    def __init__(self, seqlens, dim: int, is_learnable: bool = False):
        super().__init__()
        self.seqlens = seqlens
        self.dim = dim
        self.is_learnable = is_learnable
        if is_learnable:
            self.embed = paddle.nn.Parameter(paddle.zeros(1, *seqlens, dim))
        else:
            self.register_buffer(
                "embed",
                get_sincos_pos_embed_from_seqlens(seqlens=seqlens, dim=dim).unsqueeze(
                    0
                ),
            )
        self.reset_parameters()

    def reset_parameters(self):
        if self.is_learnable:
            paddle.nn.init.trunc_normal_(self.embed, std=0.02)

    def forward(self):
        return self.embed
