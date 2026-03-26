def to_ndim(x, ndim, pad_dim=-1):
    if pad_dim == -1:
        return x.reshape(*x.shape, *((1,) * (ndim - x.ndim)))
    elif pad_dim == 1:
        return x.reshape(x.shape[0], *((1,) * (ndim - x.ndim)), *x.shape[1:])
    else:
        raise NotImplementedError
