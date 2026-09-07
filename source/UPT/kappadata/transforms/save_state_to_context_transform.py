import paddle
from paddle.vision.transforms.functional import to_tensor

from .base.kd_transform import KDTransform


class SaveStateToContextTransform(KDTransform):
    def __init__(self, state_name):
        super().__init__()
        self.state_name = state_name

    def __call__(self, x, ctx=None):
        if ctx is not None:
            if paddle.is_tensor(x):
                ctx[self.state_name] = x.clone()
            else:
                ctx[self.state_name] = to_tensor(x)
        return x
