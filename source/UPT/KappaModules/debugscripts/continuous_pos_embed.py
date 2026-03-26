import os

import matplotlib.pyplot as plt
import torch

from kappamodules.layers import ContinuousPosEmbed


@torch.no_grad()
def main():
    emb = ContinuousPosEmbed(dim=2, ndim=2, mode="learnable", max_value=(2, 2))
    emb.embed[0] = torch.tensor([0., 1.]).unsqueeze(1)
    emb.embed[1] = torch.tensor([0., 1.]).unsqueeze(1)

    xlin = torch.linspace(0, 1, 11)
    ylin = torch.linspace(0, 1, 11)
    data = []
    for x in xlin:
        cur = []
        for y in ylin:
            cur.append(emb(torch.tensor([x, y]).unsqueeze(0)))
        data.append(torch.concat(cur))
    data = torch.stack(data)

    fig, (ax0, ax1) = plt.subplots(1, 2)
    ax0.imshow(data[:, :, 0])
    ax1.imshow(data[:, :, 1])
    plt.show()


if __name__ == '__main__':
    if os.name == "nt":
        os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    main()
