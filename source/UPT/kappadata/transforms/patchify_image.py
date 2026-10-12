import einops
import paddle
from paddle.vision.transforms.functional import to_tensor

from kappadata.utils.param_checking import to_2tuple
from .base.kd_transform import KDTransform


class PatchifyImage(KDTransform):
    def __init__(self, patch_size):
        super().__init__()
        patch_size = to_2tuple(patch_size)
        self.patch_h, self.patch_w = patch_size
        assert isinstance(self.patch_h, int) and self.patch_h > 0
        assert isinstance(self.patch_w, int) and self.patch_w > 0

    def __call__(self, x, ctx=None):
        if not paddle.is_tensor(x):
            x = to_tensor(x)
        c, src_h, src_w = x.shape
        assert src_h % self.patch_h == 0 and src_w % self.patch_w == 0
        # how many patches are along height/width dimension
        lh = int(src_h / self.patch_h)
        lw = int(src_w / self.patch_w)
        if ctx is not None:
            ctx["patchify_lh"] = lh
            ctx["patchify_lw"] = lw
        # reshape to patches
        x = einops.rearrange(x, "c (lh ph) (lw pw) -> c (lh lw) ph pw", lh=lh, ph=self.patch_h, lw=lw, pw=self.patch_w)
        return x
