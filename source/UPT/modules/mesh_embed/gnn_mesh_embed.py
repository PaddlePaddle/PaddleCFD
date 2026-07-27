import paddle
from kappamodules.layers import ContinuousSincosEmbed
from modules.graph.sag_pool import SAGPoolingFixedNumNodes
from paddle_geometric.data import Data
from paddle_geometric.nn.conv import MessagePassing
from paddle_geometric.nn.pool import SAGPooling


class GNNLayer(MessagePassing):
    def __init__(self, in_features, hidden_features):
        super().__init__()
        self.message_net = paddle.nn.Sequential(
            paddle.compat.nn.Linear(2 * in_features + 1, hidden_features),
            paddle.nn.SiLU(),
        )
        self.update_net = paddle.nn.Sequential(
            paddle.compat.nn.Linear(in_features + hidden_features, hidden_features),
            paddle.nn.SiLU(),
        )

    def forward(self, x, pos, edge_index):
        """Propagate messages along edges"""
        x = self.propagate(edge_index, x=x, pos=pos)
        return x

    def message(self, x_i, x_j, pos_i, pos_j):
        """Message update"""
        msg_input = paddle.cat(
            (
                x_i,
                x_j,
                paddle.sqrt(paddle.sum((pos_i - pos_j) ** 2, dim=1)).unsqueeze(dim=1),
            ),
            dim=-1,
        )
        message = self.message_net(msg_input)
        return message

    def update(self, message, x, pos):
        """Node update"""
        x = x + self.update_net(paddle.cat((x, message), dim=-1))
        return x

    def message_and_aggregate(self, adj_t):
        raise NotImplementedError

    def edge_update(self):
        raise NotImplementedError


class GNNMeshEmbed(paddle.nn.Layer):
    def __init__(
        self,
        in_features=3,
        out_features=None,
        hidden_features=32,
        use_gnn=True,
        pool_ratio=None,
        num_output_nodes=None,
    ):
        super().__init__()
        assert (pool_ratio is None) ^ (
            num_output_nodes is None
        ), "GnnMeshEmbed requires pool_ratio or num_output_nodes"
        self.in_features = in_features
        self.out_features = out_features or hidden_features
        self.hidden_features = hidden_features
        self.use_gnn = use_gnn
        self.pool_ratio = pool_ratio
        self.num_output_nodes = num_output_nodes
        if use_gnn:
            self.gnn_layer = GNNLayer(
                in_features=self.hidden_features, hidden_features=self.hidden_features
            )
        else:
            self.gnn_layer = None
        self.pos_embed = ContinuousSincosEmbed(dim=self.hidden_features, ndim=2)
        self.embedding_proj = paddle.compat.nn.Linear(
            self.in_features, self.hidden_features
        )
        self.output_proj = paddle.compat.nn.Linear(
            self.hidden_features, self.out_features
        )
        if num_output_nodes is not None:
            self.pool = SAGPoolingFixedNumNodes(
                self.hidden_features, num_output_nodes=self.num_output_nodes
            )
        else:
            self.pool = SAGPooling(self.hidden_features, ratio=pool_ratio)

    def forward(self, x, pos, edge_index, batch_idx):
        x = self.embedding_proj(x)
        x = x + self.pos_embed(pos)
        if self.gnn_layer is not None:
            x = self.gnn_layer(x, pos, edge_index.T)
        pool_result = self.pool(x, edge_index.T, batch=batch_idx)
        x_pool, _, _, batch_pool, _, _ = pool_result
        x_pool = self.output_proj(x_pool)
        return x_pool, batch_pool
