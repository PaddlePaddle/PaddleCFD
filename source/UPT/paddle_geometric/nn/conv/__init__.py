import inspect

import paddle

from ..._compat import scatter


class MessagePassing(paddle.nn.Layer):
    def __init__(self, aggr='add', flow='source_to_target'):
        super().__init__()
        self.aggr = 'sum' if aggr == 'add' else aggr
        self.flow = flow

    def reset_parameters(self):
        pass

    def propagate(self, edge_index, size=None, **kwargs):
        if self.flow == 'source_to_target':
            j, i = edge_index[0], edge_index[1]
        elif self.flow == 'target_to_source':
            i, j = edge_index[0], edge_index[1]
        else:
            raise ValueError(f'unsupported flow: {self.flow}')
        message_kwargs = {}
        for name in inspect.signature(self.message).parameters:
            if name == 'self':
                continue
            if name.endswith('_i'):
                key = name[:-2]
                message_kwargs[name] = kwargs[key][i]
            elif name.endswith('_j'):
                key = name[:-2]
                message_kwargs[name] = kwargs[key][j]
            elif name in kwargs:
                message_kwargs[name] = kwargs[name]
        messages = self.message(**message_kwargs)
        dim_size = size[1] if size is not None else None
        if dim_size is None:
            for value in kwargs.values():
                if isinstance(value, paddle.Tensor) and value.ndim > 0:
                    dim_size = value.shape[0]
                    break
            if dim_size is None:
                dim_size = int(i.max()) + 1 if i.numel() > 0 else 0
        aggregated = scatter(messages, i, dim=0, dim_size=dim_size, reduce=self.aggr)
        update_kwargs = {}
        for name in inspect.signature(self.update).parameters:
            if name in ('self', 'inputs'):
                continue
            if name in kwargs:
                update_kwargs[name] = kwargs[name]
        return self.update(aggregated, **update_kwargs)

    def message(self, x_j):
        return x_j

    def update(self, inputs, **kwargs):
        return inputs


class GraphConv(MessagePassing):
    def __init__(self, in_channels, out_channels, aggr='add'):
        super().__init__(aggr=aggr)
        self.lin_rel = paddle.nn.Linear(in_channels, out_channels)
        self.lin_root = paddle.nn.Linear(in_channels, out_channels)
        self.reset_parameters()

    def reset_parameters(self):
        self.lin_rel.reset_parameters()
        self.lin_root.reset_parameters()

    def forward(self, x, edge_index):
        out = self.propagate(edge_index=edge_index, x=x)
        return self.lin_rel(out) + self.lin_root(x)

    def message(self, x_j):
        return x_j
