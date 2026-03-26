import paddle.vision.transforms.functional as F

from .base.kd_stochastic_transform import KDStochasticTransform


class KDGaussianBlurTV(KDStochasticTransform):
    def __init__(self, kernel_size, sigma, **kwargs):
        super().__init__(**kwargs)
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        assert isinstance(kernel_size, (tuple, list)) and len(kernel_size) == 2
        if isinstance(sigma, (int, float)):
            sigma = (float(sigma), float(sigma))
        assert isinstance(sigma, (tuple, list)) and len(sigma) == 2
        self.kernel_size = tuple(int(v) for v in kernel_size)
        self.sigma_lb = float(sigma[0])
        self.sigma_ub = self.og_sigma_ub = float(sigma[1])
        self.ctx_key = f"{self.ctx_prefix}.sigma"

    def _scale_strength(self, factor):
        self.sigma_ub = self.sigma_lb + (self.og_sigma_ub - self.sigma_lb) * factor

    def __call__(self, x, ctx=None):
        sigma = self.get_params()
        if ctx is not None:
            ctx[self.ctx_key] = sigma
        return F.gaussian_blur(x, self.kernel_size, [sigma, sigma])

    def get_params(self):
        return self.rng.uniform(self.sigma_lb, self.sigma_ub)
