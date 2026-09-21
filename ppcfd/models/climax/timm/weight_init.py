"""Weight initialization

Hacked together by / Copyright 2020 Ross Wightman
# Adapted from https://github.com/huggingface/pytorch-image-models (timm 1.0.24).

trunc_normal_ is a functional paddle implementation (paddle.uniform +
paddle.erfinv) that writes the result back into the original parameter,
preserving in-place semantics and the [-2*std, 2*std] truncation.
"""

import math
import warnings

import paddle


def _trunc_normal_(tensor, mean, std, a, b):
    def norm_cdf(x):
        return (1.0 + math.erf(x / math.sqrt(2.0))) / 2.0

    if mean < a - 2 * std or mean > b + 2 * std:
        warnings.warn(
            "mean is more than 2 std from [a, b] in nn.init.trunc_normal_. The distribution of values may be incorrect.",
            stacklevel=2,
        )
    l = norm_cdf((a - mean) / std)
    u = norm_cdf((b - mean) / std)

    # Uniformly sample in [2l-1, 2u-1], inverse-CDF transform to a truncated
    # standard normal, then rescale to the requested mean/std and clamp to
    # the proper range -- mirroring the torch in-place chain step by step.
    out = paddle.uniform(tensor.shape, min=2 * l - 1, max=2 * u - 1, dtype=tensor.dtype)
    out = paddle.erfinv(out)
    out = out * (std * math.sqrt(2.0)) + mean
    out = paddle.clip(out, min=a, max=b)
    tensor.set_value(paddle.cast(out, tensor.dtype))
    return tensor


def trunc_normal_(tensor, mean=0.0, std=1.0, a=-2.0, b=2.0):
    """Fills the input Tensor with values drawn from a truncated
    normal distribution. The values are effectively drawn from the
    normal distribution :math:`\\mathcal{N}(\\text{mean}, \\text{std}^2)`
    with values outside :math:`[a, b]` redrawn until they are within
    the bounds. The method used for generating the random values works
    best when :math:`a \\leq \\text{mean} \\leq b`.

    NOTE: this impl is similar to the PyTorch trunc_normal_, the bounds [a, b] are
    applied while sampling the normal with mean/std applied, therefore a, b args
    should be adjusted to match the range of mean, std args.

    Args:
        tensor: an n-dimensional `paddle.Tensor`
        mean: the mean of the normal distribution
        std: the standard deviation of the normal distribution
        a: the minimum cutoff value
        b: the maximum cutoff value
    """
    with paddle.no_grad():
        return _trunc_normal_(tensor, mean, std, a, b)
