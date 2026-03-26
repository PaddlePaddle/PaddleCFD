import paddle
from paddle.vision.transforms.functional import to_tensor

from .base.kd_transform import KDTransform


class KDColumnwiseNorm(KDTransform):
    def __init__(self, mode, inplace=True, eps=1e-6, **kwargs):
        super().__init__(**kwargs)
        assert mode in ["moment", "range"]
        self.mode = mode
        self.inplace = inplace
        self.eps = eps

    def __call__(self, x, _=None):
        if not paddle.is_tensor(x):
            x = to_tensor(x)
        _, h, w = x.shape
        if self.mode == "moment":
            to_sub = x.mean(dim=1).unsqueeze(1)
        elif self.mode == "range":
            to_sub = x.min(dim=1).values.unsqueeze(1)
        else:
            raise NotImplementedError
        if self.inplace:
            x -= to_sub
        else:
            x = x - to_sub
        if self.mode == "moment":
            to_div = x.std(dim=1).unsqueeze(1) + self.eps
        elif self.mode == "range":
            to_div = x.max(dim=1).values.unsqueeze(1)
        else:
            raise NotImplementedError
        if self.inplace:
            x /= to_div
        else:
            x = x / to_div
        return x
