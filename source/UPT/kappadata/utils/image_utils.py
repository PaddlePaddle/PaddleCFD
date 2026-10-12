import paddle


def get_dimensions(img):
    # the vision transform helpers have some backward compatibility issues with get_dimensions
    if paddle.is_tensor(img):
        c, h, w = img.shape
    else:
        w, h = img.size
        c = 1 if img.mode == "L" else 3
    return c, h, w
