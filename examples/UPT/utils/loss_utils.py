import paddle


def apply_reduction(tensor, reduction="mean"):
    if tensor.dtype == paddle.bool:
        tensor = tensor.float()
    if reduction == "mean":
        return tensor.mean()
    if reduction == "mean_per_sample":
        if tensor.ndim > 1:
            return tensor.flatten(start_axis=1).mean(axis=1)
        return tensor
    if reduction is None or reduction == "none":
        return tensor
    raise NotImplementedError
