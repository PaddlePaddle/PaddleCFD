
import functools
import math
import types

import paddle
import paddle.nn.functional as F

############################## 相关utils函数，如下 ##############################
############################ PaConvert 自动生成的代码 ###########################

def _Tensor_min(self, *args, **kwargs):
    if "other" in kwargs:
        kwargs["y"] = kwargs.pop("other")
        ret = paddle.minimum(self, *args, **kwargs)
    elif len(args) == 1 and isinstance(args[0], paddle.Tensor):
        ret = paddle.minimum(self, *args, **kwargs)
    else:
        if "dim" in kwargs:
            kwargs["axis"] = kwargs.pop("dim")

        if "axis" in kwargs or len(args) >= 1:
            ret = paddle.min(self, *args, **kwargs), paddle.argmin(self, *args, **kwargs)
        else:
            ret = paddle.min(self, *args, **kwargs)

    return ret

setattr(paddle.Tensor, "_min", _Tensor_min)

def _Tensor_max(self, *args, **kwargs):
    if "other" in kwargs:
        kwargs["y"] = kwargs.pop("other")
        ret = paddle.maximum(self, *args, **kwargs)
    elif len(args) == 1 and isinstance(args[0], paddle.Tensor):
        ret = paddle.maximum(self, *args, **kwargs)
    else:
        if "dim" in kwargs:
            kwargs["axis"] = kwargs.pop("dim")

        if "axis" in kwargs or len(args) >= 1:
            ret = paddle.max(self, *args, **kwargs), paddle.argmax(self, *args, **kwargs)
        else:
            ret = paddle.max(self, *args, **kwargs)

    return ret

setattr(paddle.Tensor, "_max", _Tensor_max)

_PATCHED = False


class _TensorSizeProxy(int):
    """int-like tensor element count that also supports PyTorch-style size(dim)."""

    def __new__(cls, tensor):
        obj = int.__new__(cls, math.prod(tensor.shape))
        obj._shape = tuple(tensor.shape)
        return obj

    def __call__(self, dim=None):
        if dim is None:
            return tuple(self._shape)
        return self._shape[dim]


def _normalize_dim_kwargs(kwargs):
    if "dim" in kwargs and "axis" not in kwargs:
        kwargs["axis"] = kwargs.pop("dim")
    return kwargs


def _move_to_device(tensor, device):
    if device is None:
        return tensor
    return tensor.to(device)


def _shape_from_args(args, kwargs):
    device = kwargs.pop("device", None)
    if "size" in kwargs and "shape" not in kwargs:
        kwargs["shape"] = kwargs.pop("size")
    if "shape" in kwargs:
        shape = kwargs["shape"]
        rest = args
    elif len(args) > 1 and all(isinstance(arg, int) for arg in args):
        shape = list(args)
        rest = ()
    elif len(args) == 1:
        shape = args[0]
        rest = ()
    else:
        shape = args
        rest = ()
    if isinstance(shape, int):
        shape = [shape]
    kwargs["shape"] = shape
    return rest, kwargs, device


def _patch_paddle_api():
    global _PATCHED
    if _PATCHED:
        return
    _PATCHED = True

    if not hasattr(paddle, "long"):
        paddle.long = paddle.int64

    orig_to = paddle.Tensor.to
    orig_reshape = paddle.Tensor.reshape
    orig_transpose = paddle.Tensor.transpose
    orig_split = paddle.Tensor.split
    orig_chunk = paddle.Tensor.chunk
    orig_flatten = paddle.Tensor.flatten
    orig_mean = paddle.Tensor.mean
    orig_sum = paddle.Tensor.sum
    orig_std = paddle.Tensor.std
    orig_var = paddle.Tensor.var
    orig_min = paddle.Tensor.min
    orig_max = paddle.Tensor.max
    orig_norm = paddle.Tensor.norm
    orig_argmax = paddle.Tensor.argmax
    orig_topk = paddle.Tensor.topk
    orig_cumsum = paddle.Tensor.cumsum
    orig_unsqueeze = paddle.Tensor.unsqueeze
    orig_roll = paddle.Tensor.roll
    orig_index_select = paddle.Tensor.index_select
    orig_softmax = getattr(paddle.Tensor, "softmax", None)

    def tensor_to(self, *args, **kwargs):
        kwargs.pop("non_blocking", None)
        return orig_to(self, *args, **kwargs)

    def tensor_size(self):
        return _TensorSizeProxy(self)

    def tensor_reshape(self, *shape, **kwargs):
        if "shape" in kwargs:
            target_shape = kwargs.pop("shape")
        elif len(shape) == 1 and isinstance(shape[0], (list, tuple)):
            target_shape = shape[0]
        else:
            target_shape = list(shape)
        return orig_reshape(self, target_shape, **kwargs)

    def tensor_view(self, *shape):
        return tensor_reshape(self, *shape)

    def tensor_transpose(self, *args):
        if len(args) == 1 and isinstance(args[0], (list, tuple)):
            return orig_transpose(self, list(args[0]))
        if len(args) == 2 and all(isinstance(arg, int) for arg in args):
            perm = list(range(self.ndim))
            dim0, dim1 = args
            perm[dim0], perm[dim1] = perm[dim1], perm[dim0]
            return orig_transpose(self, perm)
        return orig_transpose(self, *args)

    def tensor_permute(self, *dims):
        if len(dims) == 1 and isinstance(dims[0], (list, tuple)):
            dims = dims[0]
        return orig_transpose(self, list(dims))

    def tensor_contiguous(self):
        return self

    def tensor_float(self):
        return self.astype(paddle.float32)

    def tensor_long(self):
        return self.astype(paddle.int64)

    def tensor_bool(self):
        return self.astype(paddle.bool)

    def tensor_clamp(self, min=None, max=None):
        return paddle.clip(self, min=min, max=max)

    def tensor_repeat(self, *repeat_times):
        if len(repeat_times) == 1 and isinstance(repeat_times[0], (list, tuple)):
            repeat_times = repeat_times[0]
        return paddle.tile(self, repeat_times=list(repeat_times))

    def _as_tensor_like(value, ref):
        if paddle.is_tensor(value):
            tensor = value
        else:
            tensor = paddle.to_tensor(value, dtype=ref.dtype)
        if tensor.dtype != ref.dtype:
            tensor = tensor.astype(ref.dtype)
        return tensor.to(ref.place)

    def tensor_add(self, other, *, alpha=1):
        other = _as_tensor_like(other, self)
        if alpha != 1:
            other = other * alpha
        return self + other

    def tensor_add_(self, other, *, alpha=1):
        self.set_value(tensor_add(self, other, alpha=alpha))
        return self

    def tensor_sub(self, other, *, alpha=1):
        other = _as_tensor_like(other, self)
        if alpha != 1:
            other = other * alpha
        return self - other

    def tensor_sub_(self, other, *, alpha=1):
        self.set_value(tensor_sub(self, other, alpha=alpha))
        return self

    def tensor_mul(self, other):
        other = _as_tensor_like(other, self)
        return self * other

    def tensor_mul_(self, other):
        self.set_value(tensor_mul(self, other))
        return self

    def tensor_div(self, other):
        other = _as_tensor_like(other, self)
        return self / other

    def tensor_div_(self, other):
        self.set_value(tensor_div(self, other))
        return self

    def tensor_exp_(self):
        self.set_value(paddle.exp(self))
        return self

    def tensor_copy_(self, other):
        self.set_value(_as_tensor_like(other, self))
        return self

    def tensor_zeros_(self):
        self.set_value(paddle.zeros(shape=self.shape, dtype=self.dtype).to(self.place))
        return self

    def tensor_new_zeros(self, *shape, dtype=None, size=None):
        if size is not None:
            shape = size
        elif len(shape) == 1 and isinstance(shape[0], (list, tuple)):
            shape = shape[0]
        return paddle.zeros(shape=shape, dtype=dtype or self.dtype).to(self.place)

    def tensor_new_ones(self, *shape, dtype=None, size=None):
        if size is not None:
            shape = size
        elif len(shape) == 1 and isinstance(shape[0], (list, tuple)):
            shape = shape[0]
        return paddle.ones(shape=shape, dtype=dtype or self.dtype).to(self.place)

    def tensor_split(self, split_size_or_sections, *args, **kwargs):
        _normalize_dim_kwargs(kwargs)
        axis = kwargs.pop("axis", args[0] if args else 0)
        if isinstance(split_size_or_sections, int):
            sections = self.shape[axis] // split_size_or_sections
            return paddle.split(self, sections, axis=axis)
        return orig_split(self, split_size_or_sections, axis=axis)

    def tensor_chunk(self, chunks, *args, **kwargs):
        _normalize_dim_kwargs(kwargs)
        axis = kwargs.pop("axis", args[0] if args else 0)
        return orig_chunk(self, chunks=chunks, axis=axis)

    def tensor_flatten(self, *args, **kwargs):
        if "start_dim" in kwargs and "start_axis" not in kwargs:
            kwargs["start_axis"] = kwargs.pop("start_dim")
        if "end_dim" in kwargs and "stop_axis" not in kwargs:
            kwargs["stop_axis"] = kwargs.pop("end_dim")
        return orig_flatten(self, *args, **kwargs)

    def _method_with_dim(orig_method):
        @functools.wraps(orig_method)
        def wrapper(self, *args, **kwargs):
            _normalize_dim_kwargs(kwargs)
            return orig_method(self, *args, **kwargs)

        return wrapper

    def tensor_roll(self, *args, **kwargs):
        if "dims" in kwargs and "axis" not in kwargs:
            kwargs["axis"] = kwargs.pop("dims")
        return orig_roll(self, *args, **kwargs)

    def tensor_index_select(self, *args, **kwargs):
        _normalize_dim_kwargs(kwargs)
        return orig_index_select(self, *args, **kwargs)

    def get_requires_grad(self):
        return not self.stop_gradient

    def set_requires_grad(self, value):
        self.stop_gradient = not value

    setattr(paddle.Tensor, "to", tensor_to)
    setattr(paddle.Tensor, "size", property(tensor_size))
    setattr(paddle.Tensor, "requires_grad", property(get_requires_grad, set_requires_grad))
    setattr(paddle.Tensor, "reshape", tensor_reshape)
    setattr(paddle.Tensor, "view", tensor_view)
    setattr(paddle.Tensor, "transpose", tensor_transpose)
    setattr(paddle.Tensor, "permute", tensor_permute)
    setattr(paddle.Tensor, "contiguous", tensor_contiguous)
    setattr(paddle.Tensor, "float", tensor_float)
    setattr(paddle.Tensor, "long", tensor_long)
    setattr(paddle.Tensor, "bool", tensor_bool)
    setattr(paddle.Tensor, "clamp", tensor_clamp)
    setattr(paddle.Tensor, "repeat", tensor_repeat)
    setattr(paddle.Tensor, "add", tensor_add)
    setattr(paddle.Tensor, "add_", tensor_add_)
    setattr(paddle.Tensor, "sub", tensor_sub)
    setattr(paddle.Tensor, "sub_", tensor_sub_)
    setattr(paddle.Tensor, "mul", tensor_mul)
    setattr(paddle.Tensor, "mul_", tensor_mul_)
    setattr(paddle.Tensor, "div", tensor_div)
    setattr(paddle.Tensor, "div_", tensor_div_)
    setattr(paddle.Tensor, "exp_", tensor_exp_)
    setattr(paddle.Tensor, "copy_", tensor_copy_)
    setattr(paddle.Tensor, "zeros_", tensor_zeros_)
    setattr(paddle.Tensor, "new_zeros", tensor_new_zeros)
    setattr(paddle.Tensor, "new_ones", tensor_new_ones)
    setattr(paddle.Tensor, "split", tensor_split)
    setattr(paddle.Tensor, "chunk", tensor_chunk)
    setattr(paddle.Tensor, "flatten", tensor_flatten)
    setattr(paddle.Tensor, "mean", _method_with_dim(orig_mean))
    setattr(paddle.Tensor, "sum", _method_with_dim(orig_sum))
    setattr(paddle.Tensor, "std", _method_with_dim(orig_std))
    setattr(paddle.Tensor, "var", _method_with_dim(orig_var))
    setattr(paddle.Tensor, "min", _method_with_dim(orig_min))
    setattr(paddle.Tensor, "max", _method_with_dim(orig_max))
    setattr(paddle.Tensor, "norm", _method_with_dim(orig_norm))
    setattr(paddle.Tensor, "argmax", _method_with_dim(orig_argmax))
    setattr(paddle.Tensor, "topk", _method_with_dim(orig_topk))
    setattr(paddle.Tensor, "cumsum", _method_with_dim(orig_cumsum))
    setattr(paddle.Tensor, "unsqueeze", _method_with_dim(orig_unsqueeze))
    setattr(paddle.Tensor, "roll", tensor_roll)
    setattr(paddle.Tensor, "index_select", tensor_index_select)
    if orig_softmax is not None:
        setattr(paddle.Tensor, "softmax", _method_with_dim(orig_softmax))

    def _wrap_dim_func(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            _normalize_dim_kwargs(kwargs)
            return func(*args, **kwargs)

        return wrapper

    def _wrap_shape_func(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            rest, kwargs, device = _shape_from_args(args, kwargs)
            tensor = func(*rest, **kwargs)
            return _move_to_device(tensor, device)

        return wrapper

    def _wrap_arange(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            device = kwargs.pop("device", None)
            tensor = func(*args, **kwargs)
            return _move_to_device(tensor, device)

        return wrapper

    def _wrap_full(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            device = kwargs.pop("device", None)
            if "size" in kwargs and "shape" not in kwargs:
                kwargs["shape"] = kwargs.pop("size")

            rest = list(args)
            if "shape" not in kwargs and rest:
                kwargs["shape"] = rest.pop(0)
            if "fill_value" not in kwargs and rest:
                kwargs["fill_value"] = rest.pop(0)
            if "dtype" not in kwargs and rest:
                kwargs["dtype"] = rest.pop(0)
            if "name" not in kwargs and rest:
                kwargs["name"] = rest.pop(0)
            if rest:
                raise TypeError(f"too many positional arguments for paddle.full: {rest}")
            if "shape" not in kwargs:
                raise TypeError("paddle.full() missing required argument: 'shape'")
            tensor = func(**kwargs)
            return _move_to_device(tensor, device)

        return wrapper

    def _wrap_to_tensor(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            device = kwargs.pop("device", None)
            if device is not None and "place" not in kwargs:
                kwargs["place"] = device
            return func(*args, **kwargs)

        return wrapper

    paddle.concat = _wrap_dim_func(paddle.concat)
    paddle.cat = paddle.concat
    paddle.stack = _wrap_dim_func(paddle.stack)
    paddle.sum = _wrap_dim_func(paddle.sum)
    paddle.mean = _wrap_dim_func(paddle.mean)
    paddle.gather = _wrap_dim_func(paddle.gather)
    paddle.split = _wrap_dim_func(paddle.split)
    paddle.flatten = _wrap_dim_func(paddle.flatten)
    paddle.zeros = _wrap_shape_func(paddle.zeros)
    paddle.ones = _wrap_shape_func(paddle.ones)
    paddle.full = _wrap_full(paddle.full)
    paddle.empty = _wrap_shape_func(paddle.empty)
    paddle.rand = _wrap_shape_func(paddle.rand)
    paddle.randn = _wrap_shape_func(paddle.randn)
    paddle.arange = _wrap_arange(paddle.arange)
    paddle.randperm = _wrap_arange(paddle.randperm)
    paddle.to_tensor = _wrap_to_tensor(paddle.to_tensor)

    orig_f_normalize = F.normalize
    orig_f_softmax = F.softmax
    F.normalize = _wrap_dim_func(orig_f_normalize)
    F.softmax = _wrap_dim_func(orig_f_softmax)
    if hasattr(paddle, "compat") and hasattr(paddle.compat, "nn"):
        paddle.compat.nn.functional.normalize = F.normalize
        paddle.compat.nn.functional.softmax = F.softmax

    def _apply_initializer(tensor, initializer):
        initializer(tensor)
        return tensor

    def init_zeros_(tensor):
        tensor.set_value(paddle.zeros(shape=tensor.shape, dtype=tensor.dtype).to(tensor.place))
        return tensor

    def init_constant_(tensor, val=0.0):
        tensor.set_value(
            paddle.full(shape=tensor.shape, fill_value=val, dtype=tensor.dtype).to(
                tensor.place
            )
        )
        return tensor

    def init_uniform_(tensor, a=0.0, b=1.0):
        return _apply_initializer(tensor, paddle.nn.initializer.Uniform(low=a, high=b))

    def init_normal_(tensor, mean=0.0, std=1.0):
        return _apply_initializer(tensor, paddle.nn.initializer.Normal(mean=mean, std=std))

    def init_trunc_normal_(tensor, mean=0.0, std=1.0, a=-2.0, b=2.0):
        return _apply_initializer(
            tensor, paddle.nn.initializer.TruncatedNormal(mean=mean, std=std, a=a, b=b)
        )

    def init_xavier_uniform_(tensor, gain=1.0):
        return _apply_initializer(tensor, paddle.nn.initializer.XavierUniform(gain=gain))

    paddle.nn.init = types.SimpleNamespace(
        zeros_=init_zeros_,
        constant_=init_constant_,
        uniform_=init_uniform_,
        normal_=init_normal_,
        trunc_normal_=init_trunc_normal_,
        xavier_uniform_=init_xavier_uniform_,
    )


_patch_paddle_api()

class PaddleFlag:
    cudnn_enabled = True
    cudnn_benchmark = False
    matmul_allow_tf32 = False
    cudnn_allow_tf32 = True
    cudnn_deterministic = False
############################## 相关utils函数，如上 ##############################
