import paddle

from kappautils.param_checking import to_2tuple



def xy_to_image(x, y, resolution, xmin=None, xmax=None, ymin=None, ymax=None, weights=None):
    coords = paddle.stack([x, y], axis=1)
    if xmin is not None:
        assert xmax is not None
        coords_min = paddle.to_tensor([xmin, ymin if ymin is not None else xmin], place=x.place)
    else:
        coords_min = None
    if ymax is not None:
        assert ymin is not None
        coords_max = paddle.to_tensor([xmax if xmax is not None else ymax, ymax], place=x.place)
    else:
        coords_max = None
    return coords_to_image(
        coords=coords,
        resolution=resolution,
        coords_min=coords_min,
        coords_max=coords_max,
        weights=weights,
    )



def coords_to_image(coords, resolution, coords_min=None, coords_max=None, weights=None):
    if not paddle.is_tensor(resolution):
        assert resolution is not None
        resolution = paddle.to_tensor(to_2tuple(resolution), place=coords.place, dtype='int64')
    else:
        resolution = resolution.astype('int64')
    assert resolution.ndim == 1 and resolution.numel() == 2
    if coords_min is not None:
        if not paddle.is_tensor(coords_min):
            coords_min = paddle.to_tensor(coords_min, place=coords.place, dtype=coords.dtype)
        else:
            coords_min = coords_min.astype(coords.dtype)
    else:
        coords_min = coords.min(axis=0)
    assert coords_min.ndim == 1 and coords_min.numel() == 2
    coords = coords - coords_min.unsqueeze(0)
    if coords_max is not None:
        if not paddle.is_tensor(coords_max):
            coords_max = paddle.to_tensor(coords_max, place=coords.place, dtype=coords.dtype)
        else:
            coords_max = coords_max.astype(coords.dtype)
    else:
        coords_max = coords.max(axis=0)
    assert coords_max.ndim == 1 and coords_max.numel() == 2
    coords_max = coords_max / (resolution.astype(coords.dtype) - 1)
    coords = coords / coords_max.unsqueeze(0)
    coords = paddle.round(coords).astype('int64')
    coords = coords[:, 0] * resolution[1] + coords[:, 1]
    minlength = int(resolution[0].item() * resolution[1].item())
    img = paddle.bincount(coords, weights=weights, minlength=minlength)
    return img.reshape([int(resolution[0]), int(resolution[1])])
