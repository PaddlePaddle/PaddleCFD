import paddle
from torch_geometric.nn.conv import MessagePassing


class GNNLayer(MessagePassing):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.message_net = paddle.nn.Sequential(
            paddle.compat.nn.Linear(2 * input_dim + 1, hidden_dim), paddle.nn.SiLU()
        )
        self.update_net = paddle.nn.Sequential(
            paddle.compat.nn.Linear(input_dim + hidden_dim, hidden_dim),
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
