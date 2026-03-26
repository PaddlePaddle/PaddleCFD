import math
from dataclasses import dataclass

import paddle
from paddle.nn.functional import one_hot
from kappadata.compat.data import DataLoader
from kappadata.compat.image import get_image_size
from tqdm import tqdm


@dataclass
class HistogramData:
    values: list
    start: int
    end: int
    avg: float


def _noop_collate(batch):
    return batch


def _to_histogram(tensor, cutoff, bucket_size, batch_size):
    counts = []
    tensor_min = tensor.min()
    bins = int((((tensor.max() - tensor_min) / bucket_size) + 1).astype(paddle.int64).item())
    tensor = tensor - tensor_min
    n_chunks = max(1, math.ceil(len(tensor) / batch_size))
    for chunk in tqdm(paddle.chunk(tensor, n_chunks, axis=0)):
        chunk = one_hot((chunk // bucket_size).astype(paddle.int64), bins)
        counts.append(chunk.sum(axis=0))
    counts = paddle.stack(counts).astype(paddle.float32).mean(axis=0)
    counts /= counts.sum()

    cumsum = paddle.cumsum(counts, axis=0)
    start_idx = int((cumsum >= cutoff).nonzero().min().item())
    end_idx = int((cumsum <= (1 - cutoff)).nonzero().max().item())
    counts = counts[start_idx:end_idx]
    counts /= counts.sum()

    start = tensor_min + start_idx * bucket_size
    end = tensor_min + (end_idx - 1) * bucket_size
    return counts.cpu(), start.cpu(), end.cpu()


def visualize_dataset_imgsize(dataset, cutoff=0.05, bucket_size=1, batch_size=128, num_workers=10, device='cpu'):
    del device
    dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, collate_fn=_noop_collate)

    heights = []
    widths = []
    for batch in tqdm(dataloader):
        for img in batch:
            width, height = get_image_size(img)
            heights.append(height)
            widths.append(width)
    heights = paddle.to_tensor(heights)
    widths = paddle.to_tensor(widths)

    areas = heights * widths
    kwargs = dict(cutoff=cutoff, bucket_size=bucket_size, batch_size=batch_size)
    hist_heights, heights_min, heights_max = _to_histogram(heights, **kwargs)
    hist_widths, widths_min, widths_max = _to_histogram(widths, **kwargs)
    hist_areas, areas_min, areas_max = _to_histogram(areas, **kwargs)

    return (
        HistogramData(values=hist_heights, start=heights_min, end=heights_max, avg=heights.astype(paddle.float32).mean().cpu()),
        HistogramData(values=hist_widths, start=widths_min, end=widths_max, avg=widths.astype(paddle.float32).mean().cpu()),
        HistogramData(values=hist_areas, start=areas_min, end=areas_max, avg=areas.astype(paddle.float32).mean().cpu()),
    )
