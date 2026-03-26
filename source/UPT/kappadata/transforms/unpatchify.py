import einops

from .base.kd_transform import KDTransform


class Unpatchify(KDTransform):
    def __call__(self, x, ctx=None):
        return einops.rearrange(
            tensor=x,
            pattern="c seqlen_h seqlen_w patch_h patch_w -> c (seqlen_h patch_h) (seqlen_w patch_w)",
        )
