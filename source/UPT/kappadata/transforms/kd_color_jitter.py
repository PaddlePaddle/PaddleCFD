import paddle.vision.transforms.functional as F
from paddle.vision.transforms import ColorJitter

from .base.kd_stochastic_transform import KDStochasticTransform


class KDColorJitter(KDStochasticTransform):
    def __init__(self, brightness=0, contrast=0, saturation=0, hue=0, **kwargs):
        super().__init__(**kwargs)
        # ColorJitter preprocesses the parameters
        tv_colorjitter = ColorJitter(brightness=brightness, contrast=contrast, saturation=saturation, hue=hue)
        # store for scaling strength
        if tv_colorjitter.brightness is not None:
            self.brightness_lb = self.og_brightness_lb = tv_colorjitter.brightness[0]
            self.brightness_ub = self.og_brightness_ub = tv_colorjitter.brightness[1]
        else:
            self.brightness_lb = None
        if tv_colorjitter.contrast is not None:
            self.contrast_lb = self.og_contrast_lb = tv_colorjitter.contrast[0]
            self.contrast_ub = self.og_contrast_ub = tv_colorjitter.contrast[1]
        else:
            self.contrast_lb = None
        if tv_colorjitter.saturation is not None:
            self.saturation_lb = self.og_saturation_lb = tv_colorjitter.saturation[0]
            self.saturation_ub = self.og_saturation_ub = tv_colorjitter.saturation[1]
        else:
            self.saturation_lb = None
        if tv_colorjitter.hue is not None:
            self.hue_lb = self.og_hue_lb = tv_colorjitter.hue[0]
            self.hue_ub = self.og_hue_ub = tv_colorjitter.hue[1]
        else:
            self.hue_lb = None
        self.ctx_key_fn_idx = f"{self.ctx_prefix}.fn_idx"
        self.ctx_key_brightness = f"{self.ctx_prefix}.brightness"
        self.ctx_key_contrast = f"{self.ctx_prefix}.contrast"
        self.ctx_key_saturation = f"{self.ctx_prefix}.saturation"
        self.ctx_key_hue = f"{self.ctx_prefix}.hue"

    def _scale_strength(self, factor):
        # brightness/contrast/saturation are centered at 1. and should be >= 0
        if self.brightness_lb is not None:
            self.brightness_lb = max(0, 1 - (1 - self.og_brightness_lb) * factor)
            self.brightness_ub = 1. + (1. - self.og_brightness_ub) * factor
        if self.contrast_lb is not None:
            self.contrast_lb = max(0, 1 - (1 - self.og_contrast_lb) * factor)
            self.contrast_ub = 1. + (1. - self.og_contrast_ub) * factor
        if self.saturation_lb is not None:
            self.saturation_lb = max(0, 1 - (1 - self.og_saturation_lb) * factor)
            self.saturation_ub = 1. + (1. - self.og_saturation_ub) * factor
        # hue is centered at 0. and -0.5 <= hue <= 0.5
        if self.hue_lb is not None:
            self.hue_lb = max(-0.5, self.og_hue_lb * factor)
            self.hue_ub = max(0.5, self.og_hue_ub * factor)

    def __call__(self, x, ctx=None):
        fn_idx, brightness_factor, contrast_factor, saturation_factor, hue_factor = self.get_params()
        if ctx is not None:
            ctx[self.ctx_key_fn_idx] = fn_idx.tolist()
            ctx[self.ctx_key_brightness] = brightness_factor or -1
            ctx[self.ctx_key_contrast] = contrast_factor or -1
            ctx[self.ctx_key_saturation] = saturation_factor or -1
            ctx[self.ctx_key_hue] = hue_factor or -1
        for fn_id in fn_idx:
            if fn_id == 0 and brightness_factor is not None:
                x = F.adjust_brightness(x, brightness_factor)
            elif fn_id == 1 and contrast_factor is not None:
                x = F.adjust_contrast(x, contrast_factor)
            elif fn_id == 2 and saturation_factor is not None:
                x = F.adjust_saturation(x, saturation_factor)
            elif fn_id == 3 and hue_factor is not None:
                x = F.adjust_hue(x, hue_factor)
        return x

    def get_params(self):
        fn_idx = self.rng.permutation(4)

        b = None if self.brightness_lb is None else self.rng.uniform(self.brightness_lb, self.brightness_ub)
        c = None if self.contrast_lb is None else self.rng.uniform(self.contrast_lb, self.contrast_ub)
        s = None if self.saturation_lb is None else self.rng.uniform(self.saturation_lb, self.saturation_ub)
        h = None if self.hue_lb is None else self.rng.uniform(self.hue_lb, self.hue_ub)

        return fn_idx, b, c, s, h
