import numpy as np
import paddle
from enum import Enum
from PIL import Image, ImageFilter
from paddle.vision.transforms.functional import crop, hflip, normalize, pad, resize as paddle_resize, rotate as paddle_rotate, to_tensor


class InterpolationMode(str, Enum):
    NEAREST = 'nearest'
    BILINEAR = 'bilinear'
    BICUBIC = 'bicubic'
    BOX = 'box'
    HAMMING = 'hamming'
    LANCZOS = 'lanczos'


def resolve_interpolation(value):
    if isinstance(value, InterpolationMode):
        return value.value
    if isinstance(value, str):
        return value.lower()
    return value


def _to_numpy_image(img):
    if paddle.is_tensor(img):
        array = img.detach().cpu().numpy()
        if array.ndim == 3:
            array = np.transpose(array, (1, 2, 0))
        if array.dtype != np.uint8:
            array = np.clip(array * 255 if array.max() <= 1.0 else array, 0, 255).astype(np.uint8)
        if array.ndim == 3 and array.shape[2] == 1:
            array = array[:, :, 0]
        return array
    if isinstance(img, Image.Image):
        return np.array(img)
    return np.array(img)


def to_pil_image(img):
    if isinstance(img, Image.Image):
        return img
    return Image.fromarray(_to_numpy_image(img))


def get_image_size(img):
    if paddle.is_tensor(img):
        return int(img.shape[-1]), int(img.shape[-2])
    return to_pil_image(img).size


def get_image_num_channels(img):
    if paddle.is_tensor(img):
        return int(img.shape[-3])
    return len(to_pil_image(img).getbands())


def resize(img, size, interpolation=InterpolationMode.BILINEAR, **kwargs):
    return paddle_resize(img, size=size, interpolation=resolve_interpolation(interpolation), **kwargs)


def rotate(img, angle, interpolation=InterpolationMode.NEAREST, **kwargs):
    return paddle_rotate(img, angle=angle, interpolation=resolve_interpolation(interpolation), **kwargs)


def resized_crop(img, top, left, height, width, size, interpolation=InterpolationMode.BILINEAR):
    return resize(crop(img, top, left, height, width), size=size, interpolation=interpolation)


def gaussian_blur(img, kernel_size, sigma):
    if isinstance(kernel_size, int):
        kernel_size = (kernel_size, kernel_size)
    if isinstance(sigma, (int, float)):
        sigma = (float(sigma), float(sigma))
    radius = float(sum(sigma) / len(sigma))
    pil = to_pil_image(img)
    blurred = pil.filter(ImageFilter.GaussianBlur(radius=radius))
    if paddle.is_tensor(img):
        result = to_tensor(blurred)
        return result.astype(img.dtype) if result.dtype != img.dtype else result
    return blurred
