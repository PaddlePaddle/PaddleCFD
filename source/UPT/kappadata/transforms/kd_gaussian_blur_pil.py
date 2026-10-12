import paddle
from PIL import ImageFilter
from kappadata.compat.image import to_pil_image

from .base.kd_stochastic_transform import KDStochasticTransform


class KDGaussianBlurPIL(KDStochasticTransform):
    def __init__(self, sigma, **kwargs):
        super().__init__(**kwargs)
        if isinstance(sigma, (int, float)):
            sigma = (float(sigma), float(sigma))
        assert isinstance(sigma, (tuple, list)) and len(sigma) == 2
        self.sigma_lb = float(sigma[0])
        self.sigma_ub = self.og_sigma_ub = float(sigma[1])
        self.ctx_key = f"{self.ctx_prefix}.sigma"

    def _scale_strength(self, factor):
        self.sigma_ub = self.sigma_lb + (self.og_sigma_ub - self.sigma_lb) * factor

    def __call__(self, x, ctx=None):
        if paddle.is_tensor(x):
            x = to_pil_image(x)
        sigma = self.get_params()
        if ctx is not None:
            ctx[self.ctx_key] = sigma
        return x.filter(ImageFilter.GaussianBlur(radius=sigma))

    def get_params(self):
        return self.rng.uniform(self.sigma_lb, self.sigma_ub)
