import einops
import numpy as np
import paddle
from PIL import Image
from paddle.vision.transforms.functional import to_tensor

from .constants import VIRIDIS_PALETTE_NP



def png_loader(path):
    with open(path, 'rb') as f:
        return to_tensor(Image.open(f))



def png_loader_viridis(path):
    img = (png_loader(path) * 255).squeeze(0).astype('int64').numpy()
    rgb = np.take(VIRIDIS_PALETTE_NP, img, axis=0)
    return einops.rearrange(paddle.to_tensor(rgb), 'h w c -> c h w')



def png_loader_with_info(path):
    with Image.open(path) as img:
        info = dict(img.info)
        data = np.array(img)
    if data.ndim == 2:
        data = data[None, ...]
    else:
        data = np.transpose(data, (2, 0, 1))
    return paddle.to_tensor(data, dtype='float32') / 255., info
