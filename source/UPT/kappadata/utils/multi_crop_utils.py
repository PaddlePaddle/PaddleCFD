from collections import defaultdict

import paddle


class SplitForwardModule(paddle.nn.Layer):
    def __init__(self, module, batch_size):
        super().__init__()
        self.module = module
        self.batch_size = batch_size

    def forward(self, *args, **kwargs):
        return split_forward(self.module, *args, batch_size=self.batch_size, **kwargs)


def split_forward(model, x, batch_size=None):
    # chunk if input is tensor
    if paddle.is_tensor(x):
        assert batch_size is not None and len(x) % batch_size == 0
        x = x.chunk(len(x) // batch_size)
    else:
        assert isinstance(x, (list, tuple)) and batch_size is None

    # split forward
    results = []
    for chunk in x:
        results.append(model(chunk))

    # concat if input was tensor
    if batch_size is not None:
        results = paddle.concat(results)

    return results


def concat_same_shape_inputs(x):
    if paddle.is_tensor(x):
        return [x], len(x)
    results = defaultdict(list)
    for xx in x:
        results[tuple(xx.shape[1:])].append(xx)
    return [paddle.concat(v) for v in results.values()], len(x[0])


def split_same_shape_inputs(x, batch_size):
    if paddle.is_tensor(x):
        return [x]
    assert isinstance(x, list)
    results = []
    for xx in x:
        assert len(xx) % batch_size == 0
        results += xx.chunk(len(xx) // batch_size)
    return results
