import einops
import paddle
from kappadata.compat.image import to_tensor, resize, to_pil_image, InterpolationMode, gaussian_blur

from kappadata.transforms.patchify_image import PatchifyImage
from kappadata.transforms.unpatchify_image import UnpatchifyImage
from kappadata.utils.param_checking import to_2tuple


def visualize_mae_schematic(img, ids_restore, mask_ratio=0.75, size=300, patch_size=75, border=2):
    if not paddle.is_tensor(img):
        img = to_tensor(img)
    assert img.ndim == 3
    assert ids_restore.ndim == 1

    img = resize(img, size=to_2tuple(size), interpolation=InterpolationMode.BILINEAR)

    ctx = {}
    patches = PatchifyImage(patch_size=to_2tuple(patch_size))(img, ctx=ctx)
    assert patches.shape[1] == ids_restore.shape[0]
    if border > 0:
        patches[:, :, :border] = 1.
        patches[:, :, -border:] = 1.
        patches[:, :, :, :border] = 1.
        patches[:, :, :, -border:] = 1.

    mask = paddle.ones_like(ids_restore)
    mask[int(len(mask) * mask_ratio):] = 0
    mask = paddle.take_along_axis(mask, ids_restore, axis=0)
    visible = patches[:, (1 - mask).nonzero().flatten()]
    encoder_input_img = to_pil_image(UnpatchifyImage()(visible, ctx=dict(patchify_lh=visible.shape[1], patchify_lw=1)))

    encoded_tokens = paddle.ones_like(visible)
    encoded_tokens[0, :, border:-border, border:-border] = 153 / 255
    encoded_tokens[1, :, border:-border, border:-border] = 222 / 255
    encoded_tokens[2, :, border:-border, border:-border] = 1.
    encoded_tokens_img = to_pil_image(UnpatchifyImage()(encoded_tokens, ctx=dict(patchify_lh=encoded_tokens.shape[1], patchify_lw=1)))

    mask = einops.rearrange(mask, 'l -> 1 l 1 1')
    background = paddle.full_like(patches, fill_value=221 / 255)
    decoder_tokens = background * mask + encoded_tokens[:, :1] * (1 - mask)
    if border > 0:
        decoder_tokens[:, :, :border] = 1.
        decoder_tokens[:, :, -border:] = 1.
        decoder_tokens[:, :, :, :border] = 1.
        decoder_tokens[:, :, :, -border:] = 1.
    decoder_input_img = to_pil_image(UnpatchifyImage()(decoder_tokens, ctx=dict(patchify_lh=decoder_tokens.shape[1], patchify_lw=1)))

    decoded_tokens = paddle.ones_like(decoder_tokens)
    decoded_tokens[0, :, border:-border, border:-border] = 238 / 255
    decoded_tokens[1, :, border:-border, border:-border] = 136 / 255
    decoded_tokens[2, :, border:-border, border:-border] = 102 / 255
    decoded_tokens_img = to_pil_image(UnpatchifyImage()(decoded_tokens, ctx=dict(patchify_lh=decoded_tokens.shape[1], patchify_lw=1)))

    blurred_img = gaussian_blur(img, kernel_size=27, sigma=6.)
    blurred_patches = PatchifyImage(patch_size=to_2tuple(patch_size))(blurred_img)
    if border > 0:
        blurred_patches[:, :, :border] = 1.
        blurred_patches[:, :, -border:] = 1.
        blurred_patches[:, :, :, :border] = 1.
        blurred_patches[:, :, :, -border:] = 1.
    blurred_img = to_pil_image(UnpatchifyImage()(blurred_patches, ctx=dict(patchify_lh=blurred_patches.shape[1], patchify_lw=1)))
    flat_target = to_pil_image(UnpatchifyImage()(patches, ctx=dict(patchify_lh=patches.shape[1], patchify_lw=1)))

    return encoder_input_img, encoded_tokens_img, decoder_input_img, decoded_tokens_img, blurred_img, flat_target
