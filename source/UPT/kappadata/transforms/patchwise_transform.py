import einops
import paddle

from kappadata.factory import object_to_transform
from .base.kd_transform import KDTransform
from .patchify import Patchify
from .unpatchify import Unpatchify
from paddle.vision.transforms.functional import to_tensor


class PatchwiseTransform(KDTransform):
    def __init__(self, patch_size, transform, **kwargs):
        super().__init__(**kwargs)
        self.patchify = Patchify(patch_size)
        self.unpatchify = Unpatchify()
        self.transform = object_to_transform(transform)

    @property
    def is_deterministic(self):
        return self.transform.is_deterministic

    def set_rng(self, rng):
        return self.transform.set_rng(rng)

    @property
    def is_kd_transform(self):
        return self.transform.is_kd_transform

    def __call__(self, x, ctx=None):
        patches = self.patchify(x)
        ndim = (patches.ndim - 1) // 2
        if ndim == 2:
            _, seqlen_h, seqlen_w, _, _ = patches.shape
            patches = einops.rearrange(
                patches,
                "c seqlen_h seqlen_w patch_h patch_w -> c (seqlen_h seqlen_w) patch_h patch_w"
            )
            transformed_patches = []
            for i in range(patches.size(1)):
                transformed_patch = self.transform(patches[:, i], ctx=ctx)
                if not paddle.is_tensor(transformed_patch):
                    transformed_patch = to_tensor(transformed_patch)
                transformed_patches.append(transformed_patch)
            patches = paddle.stack(transformed_patches, dim=1)
            patches = einops.rearrange(
                patches,
                "c (seqlen_h seqlen_w) patch_h patch_w -> c seqlen_h seqlen_w patch_h patch_w",
                seqlen_h=seqlen_h,
                seqlen_w=seqlen_w,
            )
        else:
            raise NotImplementedError
        return self.unpatchify(patches)
