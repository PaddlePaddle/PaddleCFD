import einops

from .base.kd_transform import KDTransform


class UnpatchifyImage(KDTransform):
    def __call__(self, x, ctx=None):
        return einops.rearrange(
            tensor=x,
            pattern="c (lh lw) ph pw -> c (lh ph) (lw pw)",
            lh=ctx["patchify_lh"],
            lw=ctx["patchify_lw"],
        )
