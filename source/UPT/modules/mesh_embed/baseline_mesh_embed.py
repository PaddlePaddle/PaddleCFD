import einops
import paddle
from kappamodules.init import init_xavier_uniform_zero_bias
from kappamodules.layers import ContinuousSincosEmbed
from modules.graph.gnn_layer import GNNLayer


class BaselineMeshEmbed(paddle.nn.Layer):
    def __init__(self, dim, depth, resolution, input_dim):
        super().__init__()
        self.dim = dim
        assert depth >= 1
        self.depth = depth
        assert len(resolution) == 2
        self.resolution = resolution
        self.num_grid_points = self.resolution[0] * self.resolution[1]
        self.register_buffer(
            "grid_points_arange", paddle.arange(self.num_grid_points), persistent=False
        )
        self.proj = paddle.compat.nn.Linear(input_dim, dim)
        self.pos_embed = ContinuousSincosEmbed(dim=dim, ndim=len(resolution))
        self.pos_mlp = paddle.nn.Sequential(
            paddle.compat.nn.Linear(dim, dim),
            paddle.nn.SiLU(),
            paddle.compat.nn.Linear(dim, dim),
        )
        self.gnn_layers = paddle.nn.LayerList(
            [GNNLayer(input_dim=dim, hidden_dim=dim) for _ in range(depth)]
        )
        self.reset_parameters()

    def reset_parameters(self):
        init_xavier_uniform_zero_bias(self.proj)
        self.pos_mlp.apply(init_xavier_uniform_zero_bias)

    def forward(self, x, pos, batch_idx, edge_index):
        _, counts = batch_idx.unique(return_counts=True)
        start = (counts.cumsum(dim=0) - counts[0]).repeat_interleave(
            self.num_grid_points
        )
        grid_pos_idx = self.grid_points_arange.repeat(len(counts)) + start
        x = self.proj(x)
        pos_embed = self.pos_embed(pos)
        x[grid_pos_idx] = self.pos_mlp(pos_embed[grid_pos_idx])
        x = x + pos_embed
        for gnn_layer in self.gnn_layers:
            x = gnn_layer(x, pos, edge_index.T)
        x = x[grid_pos_idx]
        x = einops.rearrange(
            x,
            "(batch_size num_grid_points) dim -> batch_size num_grid_points dim",
            num_grid_points=self.num_grid_points,
        )
        return x
