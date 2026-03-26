import einops
from torch import nn


class BatchNorm1dViT(nn.BatchNorm1d):
    def forward(self, x):
        if x.ndim == 3:
            # transformer uses (batch_size, seqlen, dim) but BatchNorm1d expects (batch_size, dim, seqlen)
            x = einops.rearrange(x, "b l c -> b c l")
            x = super().forward(x)
            x = einops.rearrange(x, "b c l -> b l c")
        else:
            # support single token as well (e.g. if norm is applied after pooling)
            x = super().forward(x)
        return x
