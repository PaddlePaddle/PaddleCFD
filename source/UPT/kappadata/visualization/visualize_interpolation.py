import einops
import paddle
from kappadata.compat.image import to_tensor, resize, to_pil_image, InterpolationMode, pad

from kappadata.utils.param_checking import to_2tuple, to_4tuple


def visualize_interpolation(img, size=300, border=2):
    if paddle.is_tensor(img):
        assert img.ndim == 3
        img = to_pil_image(img)

    imgs = [
        resize(img, size=to_2tuple(size), interpolation=interpolation)
        for interpolation in [
            InterpolationMode.NEAREST,
            InterpolationMode.LANCZOS,
            InterpolationMode.BILINEAR,
            InterpolationMode.BICUBIC,
            InterpolationMode.BOX,
            InterpolationMode.HAMMING,
        ]
    ]

    # make border
    imgs = [to_tensor(pad(img, padding=to_4tuple(border))) for img in imgs]

    # calculate diffs
    diffs = [img if i == 0 else (img - imgs[0]).abs() for i, img in enumerate(imgs)]

    # stack into grid
    imgs = paddle.stack(imgs)
    diffs = paddle.stack(diffs)
    img = einops.rearrange(imgs, "(gh gw) c h w -> c (gh h) (gw w)", gh=3, gw=2)
    diffs = einops.rearrange(diffs, "(gh gw) c h w -> c (gh h) (gw w)", gh=3, gw=2)

    # to image
    return to_pil_image(img), to_pil_image(diffs)
