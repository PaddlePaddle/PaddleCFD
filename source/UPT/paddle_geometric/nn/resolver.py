import paddle


def activation_resolver(act):
    if callable(act):
        return act
    if act in ('tanh', None):
        return paddle.tanh
    if act == 'relu':
        return paddle.nn.functional.relu
    if act == 'silu':
        return paddle.nn.functional.silu
    if act == 'identity':
        return lambda x: x
    raise ValueError(f'unsupported activation: {act}')
