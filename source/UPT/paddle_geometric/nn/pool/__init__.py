import math

import paddle

from ..._compat import knn_graph, radius, radius_graph
from ..conv import GraphConv
from .connect import FilterEdges
from .select import SelectOutput


class SAGPooling(paddle.nn.Layer):
    def __init__(self, in_channels, ratio=0.5, GNN=GraphConv, multiplier=1.0, nonlinearity='tanh', aggr='sum'):
        super().__init__()
        self.in_channels = in_channels
        self.ratio = ratio
        self.multiplier = multiplier
        self.gnn = GNN(in_channels, 1, aggr=aggr)
        self.connect = FilterEdges()

    def reset_parameters(self):
        self.gnn.reset_parameters()

    def _select(self, score, batch):
        if batch is None:
            batch = paddle.zeros([score.shape[0]], dtype='int64')
        perms = []
        for batch_id in paddle.unique(batch).numpy().tolist():
            mask = batch == batch_id
            idx = paddle.nonzero(mask).flatten()
            local_score = score[idx]
            k = max(1, int(math.ceil(float(len(idx)) * float(self.ratio))))
            _, top_idx = paddle.topk(local_score, k=k)
            perms.append(idx[top_idx])
        perm = paddle.concat(perms) if perms else paddle.zeros([0], dtype='int64')
        return SelectOutput(
            node_index=perm,
            num_nodes=int(score.shape[0]),
            cluster_index=paddle.arange(len(perm), dtype='int64'),
            num_clusters=int(len(perm)),
            weight=score[perm],
        )

    def forward(self, x, edge_index, edge_attr=None, batch=None, attn=None):
        if batch is None:
            batch = paddle.zeros([x.shape[0]], dtype='int64')
        attn = x if attn is None else attn
        attn = attn.reshape([-1, 1]) if attn.ndim == 1 else attn
        score = paddle.tanh(self.gnn(attn, edge_index)).reshape([-1])
        select_out = self._select(score, batch)
        perm = select_out.node_index
        weight = select_out.weight
        x = x[perm] * weight.unsqueeze(-1)
        if self.multiplier != 1:
            x = x * self.multiplier
        connect_out = self.connect(select_out, edge_index, edge_attr, batch)
        return x, connect_out.edge_index, connect_out.edge_attr, connect_out.batch, perm, weight


__all__ = ["FilterEdges", "SAGPooling", "knn_graph", "radius", "radius_graph"]
