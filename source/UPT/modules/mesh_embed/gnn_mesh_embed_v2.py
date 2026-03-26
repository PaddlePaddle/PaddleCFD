import paddle
from kappamodules.layers import ContinuousSincosEmbed
from modules.graph.gnn_layer import GNNLayer
from modules.graph.sag_pool import SAGPoolingFixedNumNodes
from torch_geometric.data import Data


class GNNMeshEmbedV2(paddle.nn.Layer):
    def __init__(self, input_dim, gnn_dim, pool_dim, output_dim, num_output_nodes):
        super().__init__()
        self.input_dim = input_dim
        self.gnn_dim = gnn_dim
        self.pool_dim = pool_dim
        self.output_dim = output_dim or pool_dim
        self.num_output_nodes = num_output_nodes
        self.gnn_proj = paddle.compat.nn.Linear(input_dim, gnn_dim)
        self.pos_embed = ContinuousSincosEmbed(dim=gnn_dim, ndim=2)
        self.gnn_layer = GNNLayer(input_dim=gnn_dim, hidden_dim=gnn_dim)
        self.pool_proj = paddle.compat.nn.Linear(gnn_dim, pool_dim)
        self.pool = SAGPoolingFixedNumNodes(pool_dim, num_output_nodes=num_output_nodes)
        self.out_proj = paddle.compat.nn.Linear(pool_dim, self.output_dim)

    def forward(self, x, pos, edge_index, batch_idx):
        x = self.gnn_proj(x)
        x = x + self.pos_embed(pos)
        x = self.gnn_layer(x, pos, edge_index.T)
        x = self.pool_proj(x)
        pool_result = self.pool(x, edge_index.T, batch=batch_idx)
        x_pool, _, _, batch_pool, _, _ = pool_result
        x_pool = self.out_proj(x_pool)
        return x_pool, batch_pool
