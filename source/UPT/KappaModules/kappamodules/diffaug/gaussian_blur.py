import einops
import paddle


def gaussian_blur(x, sigma):
    if x.ndim == 2:
        mode = "hw"
        x = x.view(1, 1, *x.shape)
    elif x.ndim == 3:
        assert x.size(0) == 1
        mode = "1hw"
        x = x.unsqueeze(0)
    else:
        mode = None
    _, c, h, w = x.shape
    size = int(sigma * 3)
    f = paddle.arange(-size, size + 1, device=x.device).div(sigma).square().neg().exp2()
    f /= f.sum()
    padding = f.size(0) // 2
    f = einops.repeat(f, "f -> c 1 f", c=c)
    x = paddle.nn.functional.conv2d(
        input=x, weight=f.unsqueeze(2), padding=[0, padding], groups=c
    )
    x = paddle.nn.functional.conv2d(
        input=x, weight=f.unsqueeze(3), padding=[padding, 0], groups=c
    )
    if mode == "hw":
        x = x[0, 0]
    elif mode == "1hw":
        x = x[0]
    return x


class GaussianBlur(paddle.nn.Layer):
    def __init__(self, sigma):
        super().__init__()
        self.sigma = sigma

    def forward(self, x):
        return gaussian_blur(x, self.sigma)
