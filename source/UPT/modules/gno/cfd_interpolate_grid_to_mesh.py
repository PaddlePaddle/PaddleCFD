import einops
import paddle
from kappamodules.layers import ContinuousSincosEmbed
from torch_scatter import segment_csr


class CfdInterpolateGridToMesh(paddle.nn.Layer):
    @staticmethod
    def forward(x, query_pos):
        assert paddle.all(query_pos.abs() <= 1)
        if query_pos.ndim == 3:
            query_pos = einops.rearrange(
                query_pos,
                "batch_size num_query_pos ndim -> batch_size num_query_pos 1 ndim",
            )
        else:
            raise NotImplementedError
        query_pos = paddle.stack(list(reversed(query_pos.unbind(-1))), dim=-1)
        x = paddle.nn.functional.grid_sample(
            input=x, grid=query_pos, align_corners=False
        )
        x = einops.rearrange(
            x, "batch_size dim num_query_pos 1 -> (batch_size num_query_pos) dim "
        )
        return x
