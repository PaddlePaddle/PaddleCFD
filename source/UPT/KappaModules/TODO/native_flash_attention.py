import einops
import torch.nn.functional as F
from torch import nn


class NativeFlashAttention(nn.Module):
    """ timm.models.vision_transformer.Attention but with scaled_dot_product_attention """

    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        assert hasattr(F, 'scaled_dot_product_attention')
        assert attn_drop == 0, "F.scaled_dot_product_attention dropout has no train/eval mode"
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        q, k, v = einops.rearrange(
            self.qkv(x),
            "bs seqlen (three num_heads head_dim) -> three bs num_heads seqlen head_dim",
            three=3,
            num_heads=self.num_heads,
            head_dim=self.head_dim,
        ).unbind(0)

        x = F.scaled_dot_product_attention(q, k, v)

        x = einops.rearrange(x, "bs num_heads seqlen head_dim -> bs seqlen (num_heads head_dim)")
        x = self.proj(x)
        x = self.proj_drop(x)
        return x
