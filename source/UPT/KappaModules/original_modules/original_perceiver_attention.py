import einops
import torch
from torch import nn


class OriginalPerceiverAttention(nn.Module):
    """
    adapted from https://github.com/mlfoundations/open_flamingo/blob/main/open_flamingo/src/helpers.py#L25
    changes:
    - swaped initialization order of to_kv and to_q (custom block first applies to_kv instead of to_q)
    - remove norms (should be handled by block)
    - changed "t n" to "tn" (original work has time + spatial dimension but reimplementation operates on 1d)
    """

    def __init__(self, dim, dim_head=64, heads=8):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        inner_dim = dim_head * heads

        # self.norm_media = nn.LayerNorm(dim)
        # self.norm_latents = nn.LayerNorm(dim)

        self.to_kv = nn.Linear(dim, inner_dim * 2, bias=False)
        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_out = nn.Linear(inner_dim, dim, bias=False)

    def forward(self, x, latents):
        # x = self.norm_media(x)
        # latents = self.norm_latents(latents)

        h = self.heads

        q = self.to_q(latents)
        kv_input = torch.cat((x, latents), dim=-2)
        k, v = self.to_kv(kv_input).chunk(2, dim=-1)
        q = einops.rearrange(q, "b tn (h d) -> b h tn d", h=h)
        k = einops.rearrange(k, "b tn (h d) -> b h tn d", h=h)
        v = einops.rearrange(v, "b tn (h d) -> b h tn d", h=h)
        q = q * self.scale

        # attention
        sim = torch.einsum("... i d, ... j d  -> ... i j", q, k)
        sim = sim - sim.amax(dim=-1, keepdim=True).detach()
        attn = sim.softmax(dim=-1)

        out = torch.einsum("... i j, ... j d -> ... i d", attn, v)
        out = einops.rearrange(out, "b h tn d -> b tn (h d)", h=h)
        return self.to_out(out)
