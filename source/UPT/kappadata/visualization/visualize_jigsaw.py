import numpy as np
import paddle
from kappadata.compat.image import to_tensor, resize, to_pil_image, InterpolationMode

from kappadata.transforms.patchify_image import PatchifyImage
from kappadata.transforms.unpatchify_image import UnpatchifyImage
from kappadata.utils.param_checking import to_2tuple


def visualize_jigsaw(img, size=300, patch_size=75, border=2, seed=0):
    if not paddle.is_tensor(img):
        img = to_tensor(img)
    assert img.ndim == 3
    img = resize(img, size=to_2tuple(size), interpolation=InterpolationMode.BILINEAR)
    ctx = {}
    patches = PatchifyImage(patch_size=to_2tuple(patch_size))(img, ctx=ctx)
    perm = paddle.to_tensor(np.random.default_rng(seed).permutation(patches.shape[1]), dtype=paddle.int64)
    patches = patches[:, perm]
    if border > 0:
        patches[:, :, :border] = 1.
        patches[:, :, -border:] = 1.
        patches[:, :, :, :border] = 1.
        patches[:, :, :, -border:] = 1.
    img = UnpatchifyImage()(patches, ctx=ctx)
    return to_pil_image(img), perm
