from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Optional

import paddle


def _ensure_long(x):
    return x.astype('int64') if x.dtype != paddle.int64 else x


def _default_batch(num_nodes, place=None):
    return paddle.zeros([num_nodes], dtype='int64')


def _as_batch(batch, num_nodes, place=None):
    if batch is None:
        return _default_batch(num_nodes, place=place)
    return _ensure_long(batch)


def _unique_batches(*batches):
    values = []
    for batch in batches:
        if batch is not None:
            values.append(batch)
    if not values:
        return paddle.to_tensor([0], dtype='int64')
    merged = paddle.concat(values)
    return paddle.unique(merged)


def cumsum(x):
    x = _ensure_long(x.reshape([-1]))
    zero = paddle.zeros([1], dtype=x.dtype)
    return paddle.concat([zero, paddle.cumsum(x)], axis=0)


def scatter(src, index, dim=0, dim_size=None, reduce='sum'):
    if dim != 0:
        raise NotImplementedError('only dim=0 is supported')
    index = _ensure_long(index.reshape([-1]))
    if index.numel() == 0:
        if dim_size is None:
            dim_size = 0
        out_shape = [dim_size, *src.shape[1:]]
        return paddle.zeros(out_shape, dtype=src.dtype)
    if dim_size is None:
        dim_size = int(index.max()) + 1
    src_shape = list(src.shape)
    out_shape = [dim_size, *src_shape[1:]]
    if reduce in ('sum', 'add', 'mean'):
        out = paddle.zeros(out_shape, dtype=src.dtype)
        out = paddle.scatter_nd_add(out, index.unsqueeze(-1), src)
        if reduce == 'mean':
            ones = paddle.ones([src.shape[0]], dtype=src.dtype)
            counts = paddle.zeros([dim_size], dtype=src.dtype)
            counts = paddle.scatter_nd_add(counts, index.unsqueeze(-1), ones)
            counts = paddle.where(counts == 0, paddle.ones_like(counts), counts)
            reshape = [dim_size] + [1] * (src.ndim - 1)
            out = out / counts.reshape(reshape)
        return out
    if reduce in ('max', 'min'):
        src_np = src.numpy()
        index_np = index.numpy()
        out = None
        for i, idx in enumerate(index_np.tolist()):
            value = src_np[i]
            if out is None:
                fill = -math.inf if reduce == 'max' else math.inf
                out = paddle.full(out_shape, fill_value=fill, dtype=src.dtype).numpy()
            if reduce == 'max':
                out[idx] = value if out[idx].shape == () else paddle.maximum(paddle.to_tensor(out[idx]), paddle.to_tensor(value)).numpy()
            else:
                out[idx] = value if out[idx].shape == () else paddle.minimum(paddle.to_tensor(out[idx]), paddle.to_tensor(value)).numpy()
        if out is None:
            fill = -math.inf if reduce == 'max' else math.inf
            out = paddle.full(out_shape, fill_value=fill, dtype=src.dtype).numpy()
        return paddle.to_tensor(out, dtype=src.dtype)
    raise NotImplementedError(f'reduce={reduce!r} is not supported')


def unbatch(x, batch):
    batch = _ensure_long(batch.reshape([-1]))
    if batch.numel() == 0:
        return []
    batch_size = int(batch.max()) + 1
    return [x[batch == idx] for idx in range(batch_size)]


def to_dense_batch(x, batch, fill_value=0, max_num_nodes=None):
    batch = _ensure_long(batch.reshape([-1]))
    if batch.numel() == 0:
        out_shape = [0, 0, *x.shape[1:]]
        return paddle.full(out_shape, fill_value, dtype=x.dtype), paddle.zeros([0, 0], dtype='bool')
    batch_size = int(batch.max()) + 1
    num_nodes = paddle.bincount(batch, minlength=batch_size)
    if max_num_nodes is None:
        max_num_nodes = int(num_nodes.max())
    out_shape = [batch_size, max_num_nodes, *x.shape[1:]]
    out = paddle.full(out_shape, fill_value, dtype=x.dtype)
    base = paddle.concat([paddle.zeros([1], dtype='int64'), paddle.cumsum(num_nodes)[:-1]])
    idx_in_batch = paddle.arange(len(batch), dtype='int64') - paddle.gather(base, batch)
    mask = idx_in_batch < max_num_nodes
    safe_batch = paddle.masked_select(batch, mask)
    safe_idx = paddle.masked_select(idx_in_batch, mask)
    safe_x = x[mask]
    indices = paddle.stack([safe_batch, safe_idx], axis=1)
    out = paddle.scatter_nd_add(out, indices, safe_x)
    out_mask = paddle.arange(max_num_nodes, dtype='int64').unsqueeze(0) < num_nodes.unsqueeze(1)
    return out, out_mask


def _limit_neighbors(order, distances, max_num_neighbors):
    if max_num_neighbors is None:
        return order
    if max_num_neighbors >= len(order):
        return order
    dist = distances[order]
    _, perm = paddle.topk(-dist, k=max_num_neighbors)
    return order[perm]


def radius(x, y, r, batch_x=None, batch_y=None, max_num_neighbors=32):
    batch_x = _as_batch(batch_x, len(x), place=x.place)
    batch_y = _as_batch(batch_y, len(y), place=y.place)
    x_ids = []
    y_ids = []
    for batch_id in _unique_batches(batch_x, batch_y).numpy().tolist():
        x_mask = batch_x == batch_id
        y_mask = batch_y == batch_id
        if int(x_mask.astype('int64').sum()) == 0 or int(y_mask.astype('int64').sum()) == 0:
            continue
        x_idx = paddle.nonzero(x_mask).flatten()
        y_idx = paddle.nonzero(y_mask).flatten()
        xb = x[x_idx]
        yb = y[y_idx]
        dists = paddle.cdist(yb, xb)
        for local_y in range(dists.shape[0]):
            neighbor_idx = paddle.nonzero(dists[local_y] <= r).flatten()
            if neighbor_idx.numel() == 0:
                continue
            neighbor_idx = _limit_neighbors(neighbor_idx, dists[local_y], max_num_neighbors)
            count = int(neighbor_idx.numel())
            x_ids.append(x_idx[neighbor_idx])
            y_ids.append(paddle.full([count], int(y_idx[local_y]), dtype='int64'))
    if not x_ids:
        return paddle.zeros([2, 0], dtype='int64')
    return paddle.stack([paddle.concat(x_ids), paddle.concat(y_ids)], axis=0)


def radius_graph(x, r, batch=None, loop=False, max_num_neighbors=32, flow='source_to_target'):
    edges = radius(x=x, y=x, r=r, batch_x=batch, batch_y=batch, max_num_neighbors=max_num_neighbors)
    if not loop and edges.shape[1] > 0:
        mask = edges[0] != edges[1]
        edges = edges[:, mask]
    if flow == 'source_to_target':
        return edges
    if flow == 'target_to_source':
        return edges[[1, 0]]
    raise ValueError(f'unsupported flow: {flow}')


def knn_graph(x, k, batch=None, loop=False, flow='source_to_target', force_undirected=False):
    batch = _as_batch(batch, len(x), place=x.place)
    src_ids = []
    dst_ids = []
    for batch_id in _unique_batches(batch).numpy().tolist():
        mask = batch == batch_id
        node_idx = paddle.nonzero(mask).flatten()
        xb = x[node_idx]
        dists = paddle.cdist(xb, xb)
        if not loop:
            inf = paddle.full([xb.shape[0]], float('inf'), dtype=dists.dtype)
            dists = dists + paddle.diag(inf)
        for local_dst in range(dists.shape[0]):
            kk = min(int(k), int(dists.shape[1]))
            _, neighbors = paddle.topk(-dists[local_dst], k=kk)
            src_ids.append(node_idx[neighbors])
            dst_ids.append(paddle.full([kk], int(node_idx[local_dst]), dtype='int64'))
    if not src_ids:
        edges = paddle.zeros([2, 0], dtype='int64')
    else:
        edges = paddle.stack([paddle.concat(src_ids), paddle.concat(dst_ids)], axis=0)
    if force_undirected and edges.shape[1] > 0:
        rev = edges[[1, 0]]
        edges = paddle.concat([edges, rev], axis=1)
        edges = paddle.unique(edges.T, axis=0).T
    if flow == 'source_to_target':
        return edges
    if flow == 'target_to_source':
        return edges[[1, 0]]
    raise ValueError(f'unsupported flow: {flow}')


def knn_interpolate(x, pos_x, pos_y, batch_x=None, batch_y=None, k=3, eps=1e-8):
    batch_x = _as_batch(batch_x, len(pos_x), place=pos_x.place)
    batch_y = _as_batch(batch_y, len(pos_y), place=pos_y.place)
    outputs = []
    out_indices = []
    for batch_id in _unique_batches(batch_x, batch_y).numpy().tolist():
        x_mask = batch_x == batch_id
        y_mask = batch_y == batch_id
        if int(x_mask.astype('int64').sum()) == 0 or int(y_mask.astype('int64').sum()) == 0:
            continue
        x_idx = paddle.nonzero(x_mask).flatten()
        y_idx = paddle.nonzero(y_mask).flatten()
        xb = x[x_idx]
        pos_xb = pos_x[x_idx]
        pos_yb = pos_y[y_idx]
        dists = paddle.cdist(pos_yb, pos_xb)
        kk = min(int(k), int(dists.shape[1]))
        values, nn_idx = paddle.topk(-dists, k=kk)
        neighbor_dist = -values
        weights = 1.0 / paddle.clip(neighbor_dist, min=eps)
        weights = weights / weights.sum(axis=1, keepdim=True)
        gathered = xb[nn_idx]
        interp = (gathered * weights.unsqueeze(-1)).sum(axis=1)
        outputs.append(interp)
        out_indices.append(y_idx)
    if not outputs:
        return paddle.zeros([len(pos_y), x.shape[1]], dtype=x.dtype)
    out = paddle.zeros([len(pos_y), x.shape[1]], dtype=x.dtype)
    out[paddle.concat(out_indices)] = paddle.concat(outputs)
    return out


@dataclass
class ConnectOutput:
    edge_index: paddle.Tensor
    edge_attr: Optional[paddle.Tensor]
    batch: Optional[paddle.Tensor]
