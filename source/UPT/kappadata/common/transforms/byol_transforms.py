from kappadata.transforms.base.kd_compose_transform import KDComposeTransform
from kappadata.transforms.kd_gaussian_blur_pil import KDGaussianBlurPIL
from kappadata.transforms.kd_random_color_jitter import KDRandomColorJitter
from kappadata.transforms.kd_random_gaussian_blur_pil import KDRandomGaussianBlurPIL
from kappadata.transforms.kd_random_grayscale import KDRandomGrayscale
from kappadata.transforms.kd_random_horizontal_flip import KDRandomHorizontalFlip
from kappadata.transforms.kd_random_resized_crop import KDRandomResizedCrop
from kappadata.transforms.kd_random_solarize import KDRandomSolarize
from kappadata.transforms.norm.kd_image_norm import KDImageNorm
from .norm import string_to_norm
from .norm.kd_image_net_norm import KDImageNetNorm


class BYOLTransform(KDComposeTransform):
    def __init__(
            self,
            size=224,
            interpolation="bicubic",
            min_scale=0.08,
            max_scale=1.0,
            flip_p=0.5,
            color_jitter_p=0.8,
            brightness=0.4,
            contrast=0.4,
            saturation=0.2,
            hue=0.1,
            gaussian_blur_p=0.1,
            sigma=(0.1, 2.0),
            grayscale_p=0.2,
            solarize_p=0.2,
            solarize_threshold=128,
            norm="imagenet",
    ):
        transforms = [KDRandomResizedCrop(size=size, interpolation=interpolation, scale=(min_scale, max_scale))]
        if flip_p > 0.:
            transforms.append(KDRandomHorizontalFlip(p=flip_p))
        if color_jitter_p > 0.:
            transforms.append(
                KDRandomColorJitter(
                    p=color_jitter_p,
                    brightness=brightness,
                    contrast=contrast,
                    saturation=saturation,
                    hue=hue,
                ),
            )
        if gaussian_blur_p > 0.:
            transforms.append(KDRandomGaussianBlurPIL(p=gaussian_blur_p, sigma=sigma))
        if grayscale_p > 0.:
            transforms.append(KDRandomGrayscale(p=grayscale_p))
        if solarize_p > 0.:
            transforms.append(KDRandomSolarize(p=solarize_p, threshold=solarize_threshold))
        if norm is not None:
            if isinstance(norm, str):
                transforms.append(string_to_norm(norm))
            else:
                assert len(norm) == 2 and len(norm[0]) == len(norm[1])
                mean, std = norm
                transforms.append(KDImageNorm(mean=mean, std=std))
        super().__init__(transforms=transforms)


class BYOLTransform0(KDComposeTransform):
    def __init__(self, size=224, min_scale=0.08, max_scale=1.0):
        super().__init__(transforms=[
            KDRandomResizedCrop(size=size, interpolation="bicubic", scale=(min_scale, max_scale)),
            KDRandomHorizontalFlip(),
            KDRandomColorJitter(p=0.8, brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1),
            KDGaussianBlurPIL(sigma=(0.1, 2.0)),
            KDRandomGrayscale(p=0.2),
            KDImageNetNorm(),
        ])


class BYOLTransform1(KDComposeTransform):
    def __init__(self, size=224, min_scale=0.08, max_scale=1.0):
        super().__init__(transforms=[
            KDRandomResizedCrop(size=size, interpolation="bicubic", scale=(min_scale, max_scale)),
            KDRandomHorizontalFlip(),
            KDRandomColorJitter(p=0.8, brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1),
            KDRandomGaussianBlurPIL(p=0.1, sigma=(0.1, 2.0)),
            KDRandomGrayscale(p=0.2),
            KDRandomSolarize(p=0.2, threshold=128),
            KDImageNetNorm(),
        ])
