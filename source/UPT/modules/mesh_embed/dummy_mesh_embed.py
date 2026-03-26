import math

import einops
import paddle
from kappamodules.init import init_xavier_uniform_zero_bias
from kappamodules.layers import ContinuousSincosEmbed
from torch_geometric.nn.pool import SAGPooling


class DummyMeshEmbed(paddle.nn.Layer):
    def __init__(self, in_features, hidden_features, pool_ratio):
        super().__init__()
        self.in_features = in_features
        self.hidden_features = hidden_features
        self.pool_ratio = pool_ratio
        self.proj = paddle.compat.nn.Linear(in_features, hidden_features)
        self.pool = SAGPooling(hidden_features, ratio=pool_ratio)
        self.reset_parameters()

    def reset_parameters(self):
        init_xavier_uniform_zero_bias(self.proj)

    def forward(self, x, pos, edge_index, batch_idx):
        pool_result = self.pool(self.proj(x), edge_index.T, batch=batch_idx)
        x_pool, _, _, batch_pool, _, _ = pool_result
        return x_pool, batch_pool
