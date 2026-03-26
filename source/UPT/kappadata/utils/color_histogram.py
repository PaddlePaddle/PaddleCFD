import math

import paddle
from paddle.nn.functional import one_hot


def color_histogram(images, bins, density=False, batch_size=None):
    assert images.ndim == 4
    assert paddle.all(images >= 0.) and paddle.all(images <= 255.)
    assert 0 < bins <= 256
    assert 256 % bins == 0
    n_images, _, height, width = images.shape

    n_chunks = math.ceil(n_images / (batch_size or n_images))
    counts = []
    for chunk in paddle.chunk(images, n_chunks, axis=0):
        chunk = (chunk.flatten(start_axis=2) / (256 // bins)).astype(paddle.int64)
        chunk = one_hot(chunk, bins)
        counts.append(chunk.sum(axis=-2))
    counts = paddle.concat(counts, axis=0)

    if density:
        return counts / (height * width)
    return counts
