import einops
import paddle
from models.base.single_model_base import SingleModelBase


class RansInterpolated(SingleModelBase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        _, input_dim = self.input_shape
        _, output_dim = self.output_shape
        self.pred = paddle.nn.Sequential(
            paddle.compat.nn.Linear(input_dim, input_dim),
            paddle.nn.GELU(),
            paddle.compat.nn.Linear(input_dim, output_dim),
        )

    def forward(self, x, query_pos):
        assert paddle.all(-1 <= query_pos)
        assert paddle.all(query_pos <= 1)
        x = x.reshape(len(x), *self.static_ctx["grid_resolution"], -1)
        x = einops.rearrange(x, "batch_size ... dim -> batch_size dim ...")
        query_pos = einops.rearrange(
            query_pos,
            "(batch_size num_query_pos) ndim -> batch_size num_query_pos 1 1 ndim",
            num_query_pos=3586,
        )
        x_hat = paddle.nn.functional.grid_sample(
            input=x, grid=query_pos, align_corners=False
        )
        x_hat = einops.rearrange(
            x_hat, "batch_size dim num_query_pos 1 1 -> (batch_size num_query_pos) dim "
        )
        x_hat = self.pred(x_hat)
        return x_hat
