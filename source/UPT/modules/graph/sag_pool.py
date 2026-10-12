from typing import Callable, Optional, Tuple, Union

import paddle
from paddle_geometric.nn import GraphConv
from paddle_geometric.nn.pool.connect import FilterEdges
from paddle_geometric.typing import OptTensor

from .topk import SelectTopK


class SAGPoolingFixedNumNodes(paddle.nn.Layer):
    """paddle_geometric.nn.pool.SAGPooling with a dynamic ratio such that the number of output nodes is constant"""

    def __init__(
        self,
        in_channels: int,
        num_output_nodes: int = 5,
        GNN: paddle.nn.Layer = GraphConv,
        multiplier: float = 1.0,
        nonlinearity: Union[str, Callable] = "tanh",
        aggr: str = "sum",
    ):
        super().__init__()
        self.in_channels = in_channels
        self.num_output_nodes = num_output_nodes
        self.multiplier = multiplier
        self.aggr = aggr
        self.gnn = GNN(in_channels, 1, aggr=aggr)
        self.select = SelectTopK(1, num_output_nodes, nonlinearity)
        self.connect = FilterEdges()
        self.reset_parameters()

    def reset_parameters(self):
        """Resets all learnable parameters of the module."""
        self.gnn.reset_parameters()
        self.select.reset_parameters()

    def forward(
        self,
        x: paddle.Tensor,
        edge_index: paddle.Tensor,
        edge_attr: OptTensor = None,
        batch: OptTensor = None,
        attn: OptTensor = None,
    ) -> Tuple[paddle.Tensor, paddle.Tensor, OptTensor, OptTensor, paddle.Tensor, paddle.Tensor]:
        """
        Args:
            x (paddle.Tensor): The node feature matrix.
            edge_index (paddle.Tensor): The edge indices.
            edge_attr (paddle.Tensor, optional): The edge features.
                (default: :obj:`None`)
            batch (paddle.Tensor, optional): The batch vector
                :math:`\\mathbf{b} \\in {\\{ 0, \\ldots, B-1\\}}^N`, which assigns
                each node to a specific example. (default: :obj:`None`)
            attn (paddle.Tensor, optional): Optional node-level matrix to use
                for computing attention scores instead of using the node
                feature matrix :obj:`x`. (default: :obj:`None`)
        """
        if batch is None:
            batch = edge_index.new_zeros(x.size(0))
        attn = x if attn is None else attn
        attn = attn.view(-1, 1) if attn.dim() == 1 else attn
        attn = self.gnn(attn, edge_index)
        select_out = self.select(attn, batch)
        perm = select_out.node_index
        score = select_out.weight
        assert score is not None
        x = x[perm] * score.view(-1, 1)
        x = self.multiplier * x if self.multiplier != 1 else x
        connect_out = self.connect(select_out, edge_index, edge_attr, batch)
        return (
            x,
            connect_out.edge_index,
            connect_out.edge_attr,
            connect_out.batch,
            perm,
            score,
        )

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(gnn={self.gnn.__class__.__name__}, in_channels={self.in_channels}, num_output_nodes={self.num_output_nodes}, multiplier={self.multiplier})"
