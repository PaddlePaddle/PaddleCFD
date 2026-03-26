from functools import partial

from paddle.vision.transforms import Compose
from paddle.vision.transforms import Normalize

from kappadata.transforms import KDScheduledTransform, KDComposeTransform
from kappadata.transforms.norm.kd_norm_base import KDNormBase
from kappadata.wrappers.sample_wrappers import XTransformWrapper


def flatten_transform(transform):
    if transform is None:
        return []
    if isinstance(transform, KDScheduledTransform):
        return flatten_transform(transform.transform)
    if isinstance(transform, (Compose, KDComposeTransform)):
        result = []
        for t in transform.transforms:
            result += flatten_transform(t)
        return result
    return [transform]


def get_norm_transform(transform):
    transforms = flatten_transform(transform)
    norm_transforms = [transform for transform in transforms if isinstance(transform, (Normalize, KDNormBase))]
    if len(norm_transforms) == 0:
        return None
    assert len(norm_transforms) == 1
    return norm_transforms[0]


def get_denorm_transform(transform, inplace=False):
    norm_transform = get_norm_transform(transform)
    if norm_transform is None:
        return None
    if isinstance(norm_transform, KDNormBase):
        return partial(norm_transform.denormalize, inplace=inplace)
    return Compose([
        Normalize(mean=(0., 0., 0.), std=tuple(1 / s for s in norm_transform.std)),
        Normalize(mean=tuple(-m for m in norm_transform.mean), std=(1., 1., 1.)),
    ])


def get_x_transform(dataset):
    if dataset is None:
        return None
    wrappers = [wrapper for wrapper in dataset.all_wrappers if isinstance(wrapper, XTransformWrapper)]
    if len(wrappers) == 1:
        return wrappers[0].transform
    # try to extract it from datasets that implement the x_transform themselves
    if hasattr(dataset, "transform"):
        return dataset.transform
    if hasattr(dataset, "x_transform"):
        return dataset.x_transform
    return None
