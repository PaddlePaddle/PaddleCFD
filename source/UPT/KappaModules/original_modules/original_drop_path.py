import torch
from torch import nn


class OriginalDropPath(nn.Module):
    """ DropPath as in timm.layers.drop """

    def __init__(self, drop_prob: float = 0., scale_by_keep: bool = True, stochastic_drop_prob: bool = True):
        super().__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep
        self.stochastic_drop_prob = stochastic_drop_prob

    def forward(self, x):
        if self.drop_prob == 0. or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        if self.stochastic_drop_prob:
            # number of dropped values varies between batches
            random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
        else:
            # number of dropped values is constant between batches
            num_keep = int(len(x) * keep_prob)
            ids_keep = torch.randperm(len(x), device=x.device)[:num_keep]
            random_tensor = torch.zeros(len(x), device=x.device, dtype=x.dtype)
            random_tensor[ids_keep] = 1
            random_tensor = random_tensor.view(*shape)
        if keep_prob > 0.0 and self.scale_by_keep:
            random_tensor.div_(keep_prob)
        return x * random_tensor
