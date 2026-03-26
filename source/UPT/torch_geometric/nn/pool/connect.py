import paddle

from ..._compat import ConnectOutput


class FilterEdges:
    def __call__(self, select_out, edge_index, edge_attr=None, batch=None):
        perm = select_out.node_index
        num_nodes = int(select_out.num_nodes)
        mapping = paddle.full([num_nodes], -1, dtype='int64')
        mapping[perm] = paddle.arange(len(perm), dtype='int64')
        src = mapping[edge_index[0]]
        dst = mapping[edge_index[1]]
        mask = (src >= 0) & (dst >= 0)
        new_edge_index = paddle.stack([src[mask], dst[mask]], axis=0)
        new_edge_attr = edge_attr[mask] if edge_attr is not None else None
        new_batch = batch[perm] if batch is not None else None
        return ConnectOutput(edge_index=new_edge_index, edge_attr=new_edge_attr, batch=new_batch)
