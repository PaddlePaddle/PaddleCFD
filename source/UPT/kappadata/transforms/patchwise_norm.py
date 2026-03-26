import einops

from .base.kd_transform import KDTransform


class PatchwiseNorm(KDTransform):
    def __call__(self, x, ctx=None):
        assert x.ndim == 4, "PatchwiseNorm expects (channels, sequence_length, patch_height, patch_width) input"
        c, _, ph, pw = x.shape
        x = einops.rearrange(x, "c l ph pw -> l (c ph pw)")
        mean = x.mean(dim=1, keepdim=True)
        var = x.var(dim=1, keepdim=True)
        x = (x - mean) / (var + 1e-6) ** 0.5
        x = einops.rearrange(x, "l (c ph pw) -> c l ph pw", c=c, ph=ph, pw=pw)
        return x
