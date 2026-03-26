import einops
import paddle
from kappamodules.init.functional import init_xavier_uniform_zero_bias
from kappamodules.layers import ContinuousSincosEmbed
from torch_scatter import segment_csr


class RansGinoGridToMeshOg(paddle.nn.Layer):
    def __init__(self, input_dim, hidden_dim, bottleneck_dim, output_dim):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.bottleneck_dim = bottleneck_dim
        self.output_dim = output_dim
        self.pos_embed = ContinuousSincosEmbed(dim=hidden_dim * 4, ndim=4)
        self.message = paddle.nn.Sequential(
            paddle.compat.nn.Linear(input_dim + 4 * hidden_dim, 512),
            paddle.nn.GELU(),
            paddle.compat.nn.Linear(512, 256),
            paddle.nn.GELU(),
            paddle.compat.nn.Linear(256, bottleneck_dim),
        )
        self.pred = paddle.nn.Sequential(
            paddle.compat.nn.Linear(bottleneck_dim, 256),
            paddle.nn.GELU(),
            paddle.compat.nn.Linear(256, output_dim),
        )

    def forward(self, x, query_pos, grid_to_query_edges):
        assert query_pos.ndim == 2
        assert grid_to_query_edges.ndim == 2
        query_pos = query_pos / 100 - 1
        x = einops.rearrange(x, "batch_size seqlen dim -> (batch_size seqlen) dim")
        ones = paddle.ones(
            size=(len(query_pos),), dtype=query_pos.dtype, device=query_pos.device
        ).unsqueeze(1)
        query_pos = paddle.concat([query_pos, ones], dim=1)
        query_pos = self.pos_embed(query_pos)
        query_idx, grid_idx = grid_to_query_edges.unbind(1)
        x = paddle.concat([x[grid_idx], query_pos[query_idx]], dim=1)
        x = self.message(x)
        dst_indices, counts = query_idx.unique(return_counts=True)
        padded_counts = paddle.zeros(
            len(query_pos) + 1, device=counts.device, dtype=counts.dtype
        )
        padded_counts[dst_indices + 1] = counts
        indptr = padded_counts.cumsum(dim=0)
        x = segment_csr(src=x, indptr=indptr, reduce="mean")
        x = self.pred(x)
        return x
