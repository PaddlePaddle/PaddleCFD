import math


def uniform(size, tensor):
    if tensor is None:
        return
    bound = 1.0 / math.sqrt(size) if size > 0 else 0.0
    tensor.set_value(tensor.uniform(min=-bound, max=bound))
