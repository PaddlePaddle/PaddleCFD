import paddle
from paddle.vision.transforms.functional import to_tensor

from .base.kd_transform import KDTransform


class ImagePosEmbedGrid(KDTransform):
    def __call__(self, x, _=None):
        if not paddle.is_tensor(x):
            x = to_tensor(x)
        _, h, w = x.shape
        h_coords = paddle.linspace(-1., 1., h)
        w_coords = paddle.linspace(-1., 1., w)
        grid_h, grid_w = paddle.meshgrid(h_coords, w_coords, indexing="ij")

        return paddle.concat([x, grid_h.unsqueeze(0), grid_w.unsqueeze(0)])
