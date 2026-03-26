import paddle
import paddle.vision.transforms.functional as F
from paddle.vision.transforms import RandomRotation
from kappadata.compat.image import InterpolationMode, resolve_interpolation

from .base.kd_stochastic_transform import KDStochasticTransform


class KDRandomRotation(KDStochasticTransform):
    def __init__(self, degrees, interpolation='nearest', expand=False, center=None, fill=0, **kwargs):
        super().__init__(**kwargs)
        self.rotation = RandomRotation(
            degrees=degrees,
            interpolation=resolve_interpolation(interpolation),
            expand=expand,
            center=center,
            fill=fill,
        )
        self.degree_lb = self.og_degree_lb = float(self.rotation.degrees[0])
        self.degree_ub = self.og_degree_ub = float(self.rotation.degrees[1])

    def _scale_strength(self, factor):
        assert self.og_degree_lb == self.og_degree_ub
        self.degree_lb = self.og_degree_lb * factor
        self.degree_ub = self.og_degree_ub * factor

    def __call__(self, x, ctx=None):
        fill = self.rotation.fill
        if paddle.is_tensor(x):
            if isinstance(fill, (int, float)):
                fill = [float(fill)] * F.get_image_num_channels(x)
            else:
                fill = [float(f) for f in fill]
        angle = self.rng.uniform(self.degree_lb, self.degree_ub)
        return F.rotate(
            img=x,
            angle=angle,
            interpolation=self.rotation.interpolation,
            expand=self.rotation.expand,
            center=self.rotation.center,
            fill=fill,
        )
