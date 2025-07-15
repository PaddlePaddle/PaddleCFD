import paddle
import numpy as np
import itertools


def resample(x, res_scale, axis, output_shape=None):
    """
    A module for generic n-dimentional interpolation (Fourier resampling).

    Parameters
    ----------
    x : paddle.Tensor
            input activation of size (batch_size, channels, d1, ..., dN)
    res_scale: int or tuple
            Scaling factor along each of the dimensions in 'axis' parameter. If res_scale is scaler, then isotropic 
            scaling is performed
    axis: axis or dimensions along which interpolation will be performed.
    output_shape : None or tuple[int]
    """
    if isinstance(res_scale, (float, int)):
        if axis is None:
            axis = list(range(2, x.ndim))
            res_scale = [res_scale] * len(axis)
        elif isinstance(axis, int):
            axis = [axis]
            res_scale = [res_scale]
        else:
            res_scale = [res_scale] * len(axis)
    else:
        assert len(res_scale) == len(axis
            ), 'leght of res_scale and axis are not same'
    old_size = tuple(x.shape)[-len(axis):]
    if output_shape is None:
        new_size = tuple([int(round(s * r)) for s, r in zip(old_size,
            res_scale)])
    else:
        new_size = output_shape
    if len(axis) == 1:
        return paddle.nn.functional.interpolate(x=x, size=new_size[0], mode
            ='linear', align_corners=True)
    if len(axis) == 2:
        return paddle.nn.functional.interpolate(x=x, size=new_size, mode=
            'bicubic', align_corners=True)
    X = paddle.fft.rfftn(x=x.astype(dtype='float32'), norm='forward', axes=axis
        )
    new_fft_size = list(new_size)
    new_fft_size[-1] = new_fft_size[-1] // 2 + 1
    new_fft_size_c = [min(i, j) for i, j in zip(new_fft_size, tuple(X.shape
        )[-len(axis):])]
    out_fft = paddle.zeros(shape=[tuple(x.shape)[0], tuple(x.shape)[1], *
        new_fft_size], dtype='complex64')
    mode_indexing = [((None, m // 2), (-m // 2, None)) for m in
        new_fft_size_c[:-1]] + [((None, new_fft_size_c[-1]),)]
    for i, boundaries in enumerate(itertools.product(*mode_indexing)):
        idx_tuple = [slice(None), slice(None)] + [slice(*b) for b in boundaries
            ]
        out_fft[idx_tuple] = X[idx_tuple]
    y = paddle.fft.irfftn(x=out_fft, s=new_size, norm='forward', axes=axis)
    return y


def iterative_resample(x, res_scale, axis):
    if isinstance(axis, list) and isinstance(res_scale, (float, int)):
        res_scale = [res_scale] * len(axis)
    if not isinstance(axis, list) and isinstance(res_scale, list):
        raise Exception('Axis is not a list but Scale factors are')
    if isinstance(axis, list) and isinstance(res_scale, list) and len(res_scale
        ) != len(axis):
        raise Exception('Axis and Scal factor are in different sizes')
    if isinstance(axis, list):
        for i in range(len(res_scale)):
            rs = res_scale[i]
            a = axis[i]
            x = resample(x, rs, a)
        return x
    old_res = tuple(x.shape)[axis]
    X = paddle.fft.rfft(x=x, axis=axis, norm='forward')
    newshape = list(tuple(x.shape))
    new_res = int(round(res_scale * newshape[axis]))
    newshape[axis] = new_res // 2 + 1
    Y = paddle.zeros(shape=newshape, dtype=X.dtype)
    modes = min(new_res, old_res)
    sl = [slice(None)] * x.ndim
    sl[axis] = slice(0, modes // 2 + 1)
    Y[tuple(sl)] = X[tuple(sl)]
    y = paddle.fft.irfft(x=Y, n=new_res, axis=axis, norm='forward')
    return y
