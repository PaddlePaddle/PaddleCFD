from PIL import Image

from .constants import VIRIDIS_PALETTE_LIST



def _to_numpy_uint8_image(img):
    if hasattr(img, 'detach'):
        img = img.detach().cpu().numpy()
    if img.ndim == 3 and img.shape[0] == 1:
        img = img[0]
    if img.dtype != 'uint8':
        max_value = float(img.max()) if img.size else 0.0
        if max_value <= 1.0:
            img = img * 255
        img = img.clip(0, 255).astype('uint8')
    return img



def png_writer_greyscale(img, fp, save_format=None):
    if img.ndim == 2:
        img = img.unsqueeze(0)
    assert img.ndim == 3 and img.shape[0] == 1
    Image.fromarray(_to_numpy_uint8_image(img), mode='L').save(fp, format=save_format)



def png_writer_viridis(img, fp, save_format=None):
    if img.ndim == 2:
        img = img.unsqueeze(0)
    assert img.ndim == 3 and img.shape[0] == 1
    img = Image.fromarray(_to_numpy_uint8_image(img), mode='L')
    img.putpalette(VIRIDIS_PALETTE_LIST)
    img.save(fp, format=save_format)
