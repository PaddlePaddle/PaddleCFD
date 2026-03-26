import paddle
from kappadata.compat.image import to_tensor, to_pil_image


def visualize_two_random_crop(img, transform):
    if not paddle.is_tensor(img):
        img = to_tensor(img)
    assert img.ndim == 3

    ctx = {}
    _ = transform(img, ctx=ctx)
    ctx = ctx["two_random_crop"]
    i0, j0, h0, w0 = ctx["i0"], ctx["j0"], ctx["h0"], ctx["w0"]
    i1, j1, h1, w1 = ctx["i1"], ctx["j1"], ctx["h1"], ctx["w1"]

    mask = paddle.zeros(*img.shape[1:], device=img.device)
    mask[i0:i0 + h0, j0:j0 + w0] = 1
    mask[i1:i1 + h1, j1:j1 + w1] = 1

    return to_pil_image(img * mask), ctx["out_of_tries"]
