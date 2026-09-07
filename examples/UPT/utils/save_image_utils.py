import einops
import numpy as np
import paddle
from kappadata import get_denorm_transform, get_norm_transform
from kappadata.wrappers import XTransformWrapper
from matplotlib.pyplot import get_cmap
from paddle_utils import *
from PIL import Image

import numpy as np
import paddle

def paddle_to_pil(tensor):
    # 1. 将 Tensor 移动到 CPU 并转为 NumPy
    # 假设输入是 [C, H, W]，且在 0-1 之间
    img_array = tensor.numpy()
    
    # 2. 如果是 [C, H, W] 格式，需转为 PIL 需要的 [H, W, C]
    if len(img_array.shape) == 3:
        img_array = img_array.transpose((1, 2, 0))
    
    # 3. 如果数据是 0.0 - 1.0 之间，需转回 0-255 整数
    if img_array.max() <= 1.0:
        img_array = (img_array * 255).astype(np.uint8)
        
    return Image.fromarray(img_array)

def concat_images_square(images, scale, padding):
    columns = int(np.ceil(np.sqrt(len(images))))
    rows = int(np.ceil(len(images) / columns))
    w, h = images[0].size
    if scale != 1:
        images = [i.resize((w * scale, h * scale)) for i in images]
        w, h = images[0].size
    concated = Image.new(
        images[0].mode,
        (w * columns + padding * (columns + 1), h * rows + padding * (rows + 1)),
    )
    for i in range(len(images)):
        col = i % columns
        row = i // columns
        concated.paste(
            images[i], (w * col + padding * (col + 1), h * row + padding * (row + 1))
        )
    return concated


def concat_images_vertical(images, scale=1):
    scale_images(images, scale)
    w, h = images[0].size
    concated = Image.new(images[0].mode, (w, h * len(images)))
    for i in range(len(images)):
        concated.paste(images[i], (0, h * i))
    return concated


def concat_images_horizontal(images, scale=1):
    scale_images(images, scale)
    w, h = images[0].size
    concated = Image.new(images[0].mode, (w * len(images), h))
    for i in range(len(images)):
        concated.paste(images[i], (w * i, 0))
    return concated


def scale_images(images, scale):
    if scale != 1:
        return [i.resize((i.width * scale, i.height * scale)) for i in images]
    return images


def greyscale_to_viridis(tensor) -> Image:
    assert tensor.ndim == 3 and len(tensor) == 1
    tensor = tensor[0]
    cm = get_cmap("viridis")
    tensor = tensor.cpu().numpy()
    tensor = cm(tensor)
    tensor = np.uint8(tensor * 255)
    return Image.fromarray(tensor)


def rgba_to_rgb(image: Image) -> Image:
    background = Image.new("RGB", image.size, (255, 255, 255))
    background.paste(image, mask=image.split()[3])
    return background


def tensor_to_image(tensor, denormalize=None, scale_range_per_image=False):
    assert paddle.is_tensor(tensor) and tensor.ndim == 3
    if denormalize is not None:
        tensor = denormalize(tensor)
    if scale_range_per_image:
        tensor = tensor - tensor._min()
        tensor = tensor / tensor._max()
    if len(tensor) == 1:
        if tensor._min() < 0.0:
            tensor = tensor - tensor._min()
        if tensor._max() > 1.0:
            tensor = tensor / tensor._max()
        return greyscale_to_viridis(tensor)
    return paddle_to_pil(tensor)


def save_image_tensors(
    tensors,
    out_uri,
    denormalize=None,
    scale_range_per_image=False,
    scale=1.0,
    padding=2,
    transpose_xy=False,
):
    assert paddle.is_tensor(tensors) and tensors.ndim == 4
    if transpose_xy:
        tensors = einops.rearrange(tensors, "b c h w -> b c w h")
    images = [
        tensor_to_image(
            tensor, denormalize=denormalize, scale_range_per_image=scale_range_per_image
        )
        for tensor in tensors
    ]
    save_images(images, out_uri, scale, padding)


def save_images(images, out_uri, scale=1.0, padding=2):
    assert isinstance(images, list)
    w, h = images[0].size
    if w == h:
        concated = concat_images_square(images, scale, padding)
    elif h > w:
        concated = concat_images_horizontal(images, scale)
    else:
        concated = concat_images_vertical(images, scale)
    concated.save(out_uri)


def images_to_gif(image_uris, out_uri, duration=200):
    if len(image_uris) == 0:
        return
    imgs = (Image.open(f) for f in image_uris)
    img = next(imgs)
    img.save(
        fp=out_uri,
        format="GIF",
        append_images=imgs,
        save_all=True,
        duration=duration,
        loop=0,
    )


def get_norm_transform_from_datacontainer(data_container, dataset_key=None):
    ds, collator = data_container.get_dataset(key=dataset_key, mode="x")
    if collator is not None:
        raise NotImplementedError
    return get_norm_transform_from_dataset(ds)


def get_norm_transform_from_dataset(dataset):
    xtransform_wrapper = dataset.get_wrapper_of_type(XTransformWrapper)
    if xtransform_wrapper is None:
        return None
    return get_norm_transform(xtransform_wrapper.transform)


def get_denorm_from_datacontainer(data_container, dataset_key=None):
    ds, collator = data_container.get_dataset(key=dataset_key, mode="x")
    if collator is not None:
        raise NotImplementedError
    return get_denorm_from_dataset(ds)


def get_denorm_from_dataset(dataset):
    xtransform_wrapper = dataset.get_wrapper_of_type(XTransformWrapper)
    if xtransform_wrapper is None:
        return None
    return get_denorm_transform(xtransform_wrapper.transform)
