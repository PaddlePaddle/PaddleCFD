import paddle
from paddle.vision.transforms.functional import to_tensor

from kappadata.utils.pos_embed import get_2d_sincos_pos_embed
from .base.kd_transform import KDTransform


class ImagePosEmbedSincos(KDTransform):
    def __init__(self, dim=4):
        super().__init__()
        assert dim % 4 == 0
        self.dim = dim

    def __call__(self, x, _=None):
        if not paddle.is_tensor(x):
            x = to_tensor(x)
        _, h, w = x.shape
        pos_embed = get_2d_sincos_pos_embed(embed_dim=self.dim, h=h, w=w)
        return paddle.concat([x, pos_embed])
