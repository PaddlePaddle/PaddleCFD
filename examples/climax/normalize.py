import numpy as np
import paddle


class Normalize:
    """Normalize a tensor image with mean and standard deviation.

    Minimal replicate of ``torchvision.transforms.Normalize`` for paddle
    tensors. Given mean: ``(mean[1],...,mean[C])`` and std: ``(std[1],..,std[C])``
    for ``C`` channels, this transform will normalize each channel of the input
    ``paddle.Tensor`` i.e., ``output[channel] = (input[channel] - mean[channel]) / std[channel]``.
    Scalars broadcast over every element following numpy broadcasting rules.

    ``mean``/``std`` are kept as float32 numpy arrays (training scripts read
    ``.mean``/``.std`` back out to build the denormalization transform).

    Args:
        mean (sequence): Sequence of means for each channel.
        std (sequence): Sequence of standard deviations for each channel.
    """

    def __init__(self, mean, std):
        self.mean = np.asarray(mean, dtype="float32")
        self.std = np.asarray(std, dtype="float32")

    def __call__(self, t):
        mean = paddle.to_tensor(self.mean, dtype=t.dtype)
        std = paddle.to_tensor(self.std, dtype=t.dtype)
        if mean.ndim == 1 and t.ndim >= 3:
            # torchvision semantics: per-channel stats broadcast over the
            # channel dim (dim -3 of a (..., C, H, W) tensor). Plain numpy
            # broadcasting would wrongly align (C,) with the last dim (W,).
            shape = (1,) * (t.ndim - 3) + mean.shape + (1, 1)
            mean = mean.reshape(shape)
            std = std.reshape(shape)
        return (t - mean) / std
