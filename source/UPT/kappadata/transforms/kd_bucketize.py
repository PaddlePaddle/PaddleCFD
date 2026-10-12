import paddle
from paddle.vision.transforms.functional import to_tensor

from .base.kd_transform import KDTransform


class KDBucketize(KDTransform):
    def __init__(self, n_buckets, min_value=0., max_value=1.):
        super().__init__()
        assert n_buckets > 0
        self.n_buckets = n_buckets
        self.min_value = min_value
        self.max_value = max_value

    def __call__(self, x, ctx=None):
        if not paddle.is_tensor(x):
            x = to_tensor(x)
        # clamp to [min, max]
        x = paddle.clip(x, self.min_value, self.max_value)
        # normalize to [0, 1]
        x = (x - self.min_value) / self.max_value
        # bucketize
        x = (x * self.n_buckets).long()
        x[x == self.n_buckets] = self.n_buckets - 1
        return x
