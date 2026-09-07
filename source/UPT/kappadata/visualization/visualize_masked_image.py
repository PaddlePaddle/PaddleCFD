import einops
import numpy as np
import paddle
from kappadata.compat.image import to_tensor, resize, to_pil_image, InterpolationMode

from kappadata.transforms.kd_random_resized_crop import KDRandomResizedCrop
from kappadata.transforms.patchify_image import PatchifyImage
from kappadata.transforms.unpatchify_image import UnpatchifyImage
from kappadata.utils.param_checking import to_2tuple


def visualize_masked_image(img, size=300, patch_size=75, mask=None, border=2, fill='gray', scale=None, seed=None):
    if not paddle.is_tensor(img):
        img = to_tensor(img)
    assert img.ndim == 3
    assert mask.ndim == 1

    size = to_2tuple(size)
    if scale is None:
        img = resize(img, size=size, interpolation=InterpolationMode.BILINEAR)
    else:
        rrc = KDRandomResizedCrop(scale=scale, size=size, interpolation=InterpolationMode.BILINEAR)
        if seed is not None:
            rrc.set_rng(np.random.default_rng(seed=seed))
        img = rrc(img)

    ctx = {}
    patches = PatchifyImage(patch_size=to_2tuple(patch_size))(img, ctx=ctx)
    assert patches.shape[1] == mask.shape[0]

    if fill == 'black':
        fill_value = 0.
    elif fill == 'gray':
        fill_value = 0.5
    elif fill == 'white':
        fill_value = 1.
    else:
        raise NotImplementedError
    background = paddle.full_like(patches, fill_value=fill_value)

    mask = einops.rearrange(mask, 'l -> 1 l 1 1')
    patches = patches * (1 - mask)
    background = background * mask
    patches = patches + background

    if border > 0:
        patches[:, :, :border] = 1.
        patches[:, :, -border:] = 1.
        patches[:, :, :, :border] = 1.
        patches[:, :, :, -border:] = 1.

    img = UnpatchifyImage()(patches, ctx=ctx)
    return to_pil_image(img)
