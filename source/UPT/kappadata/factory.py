import inspect
from copy import deepcopy
from itertools import chain

import paddle.vision.transforms

_REGISTERED_TRANSFORMS = {}


def register_transform(name, cls):
    _REGISTERED_TRANSFORMS[name] = cls


def object_to_transform(obj):
    if obj is None:
        return None
    if not isinstance(obj, (list, dict)):
        return obj

    if isinstance(obj, list):
        transforms = [object_to_transform(transform) for transform in obj]
        from .transforms import KDComposeTransform
        return KDComposeTransform(transforms)

    assert 'kind' in obj and isinstance(obj['kind'], str)
    obj = deepcopy(obj)
    kind = obj.pop('kind')

    if kind in _REGISTERED_TRANSFORMS:
        return _REGISTERED_TRANSFORMS[kind](**obj)

    import kappadata.common.transforms
    import kappadata.transforms

    kd_pascal_ctor_list = inspect.getmembers(kappadata.transforms, inspect.isclass)
    kd_common_pascal_ctor_list = inspect.getmembers(kappadata.common.transforms, inspect.isclass)
    paddle_pascal_ctor_list = inspect.getmembers(paddle.vision.transforms, inspect.isclass)
    pascal_to_ctor = {
        name: ctor
        for name, ctor in chain(
            paddle_pascal_ctor_list,
            kd_pascal_ctor_list,
            kd_common_pascal_ctor_list,
        )
    }
    if kind[0].islower():
        kind = kind.replace('_', '')
        snake_to_pascal = {name.lower(): name for name in pascal_to_ctor.keys()}
        assert kind in snake_to_pascal, f"invalid kind '{kind}' (possibilities: {snake_to_pascal.keys()})"
        kind = snake_to_pascal[kind]
    ctor = pascal_to_ctor[kind]
    return ctor(**obj)
