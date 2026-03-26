import einops
import paddle
from kappamodules.init.functional import init_xavier_uniform_zero_bias
from kappamodules.layers import ContinuousSincosEmbed
from torch_scatter import segment_csr


class GinoGridToMesh(paddle.nn.Layer):
    def __init__(
        self,
        input_dim,
        hidden_dim,
        output_dim,
        ndim,
        bottleneck_dim=None,
        embed_dim=None,
        pred_hidden_dim=None,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.ndim = ndim
        if isinstance(hidden_dim, int):
            self.bottleneck_dim = bottleneck_dim or hidden_dim
            self.embed_dim = embed_dim or hidden_dim
            self.pred_hidden_dim = pred_hidden_dim or hidden_dim
            self.proj = paddle.compat.nn.Linear(input_dim, self.embed_dim)
            self.pos_embed = ContinuousSincosEmbed(dim=self.embed_dim, ndim=ndim)
            self.message = paddle.nn.Sequential(
                paddle.compat.nn.Linear(2 * self.embed_dim, 2 * hidden_dim),
                paddle.nn.GELU(),
                paddle.compat.nn.Linear(2 * hidden_dim, 2 * hidden_dim),
                paddle.nn.GELU(),
                paddle.compat.nn.Linear(2 * hidden_dim, self.bottleneck_dim),
            )
        else:
            assert bottleneck_dim is None
            assert embed_dim is None
            self.bottleneck_dim = hidden_dim[-1]
            self.embed_dim = hidden_dim[0]
            assert self.embed_dim % 2 == 0
            self.pred_hidden_dim = pred_hidden_dim or hidden_dim[-1]
            self.proj = paddle.compat.nn.Linear(input_dim, hidden_dim[0] // 2)
            self.pos_embed = ContinuousSincosEmbed(dim=hidden_dim[0] // 2, ndim=ndim)
            layers = []
            for i in range(len(hidden_dim) - 1):
                layers.append(paddle.compat.nn.Linear(hidden_dim[i], hidden_dim[i + 1]))
                if i < len(hidden_dim) - 2:
                    layers.append(paddle.nn.GELU())
            self.message = paddle.nn.Sequential(*layers)
        self.pred = paddle.nn.Sequential(
            paddle.compat.nn.Linear(self.bottleneck_dim, self.pred_hidden_dim),
            paddle.nn.GELU(),
            paddle.compat.nn.Linear(self.pred_hidden_dim, output_dim),
        )

    def forward(self, x, query_pos, grid_to_query_edges):
        assert query_pos.ndim == 2
        assert grid_to_query_edges.ndim == 2
        x = einops.rearrange(x, "batch_size seqlen dim -> (batch_size seqlen) dim")
        x = self.proj(x)
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
