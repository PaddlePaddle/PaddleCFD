import os

import matplotlib.pyplot as plt
import torch
from torch import nn

from kappamodules.functional.pos_embed import interpolate_sincos


def dino(x, w, h):
    previous_dtype = x.dtype
    patch_pos_embed = x.float()
    w0 = w
    h0 = h
    M = 14
    kwargs = {}
    sx = float(w0 + 0.1) / M
    sy = float(h0 + 0.1) / M
    kwargs["scale_factor"] = (sx, sy)
    patch_pos_embed = nn.functional.interpolate(
        patch_pos_embed.reshape(1, M, M, 1).permute(0, 3, 1, 2),
        mode="bicubic",
        **kwargs,
    )
    assert (w0, h0) == patch_pos_embed.shape[-2:]
    patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, 1)
    return patch_pos_embed.to(previous_dtype)


def main():
    torch.manual_seed(0)
    x1 = torch.randn(1, 196, 1)
    x2 = x1.reshape(1, 14, 14, 1)

    y1 = dino(x1, 6, 6).reshape(1, 6, 6, 1).squeeze(-1).squeeze(0)
    y2 = interpolate_sincos(x2, seqlens=(6, 6)).squeeze(-1).squeeze(0)
    y3 = interpolate_sincos(x2, seqlens=(6, 6), interpolate_offset=0.1).squeeze(-1).squeeze(0)

    plt.imshow(y1)
    plt.show()
    plt.clf()
    plt.imshow(y2)
    plt.show()
    plt.clf()
    plt.imshow(y3)
    plt.show()

    print((y1 - y2).abs().max())
    print((y1 - y3).abs().max())


if __name__ == "__main__":
    if os.name == "nt":
        os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    main()
