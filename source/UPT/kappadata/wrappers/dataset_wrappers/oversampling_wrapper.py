import numpy as np
import paddle

from kappadata.datasets.kd_subset import KDSubset
from kappadata.utils.class_counts import get_class_counts
from kappadata.utils.getall_as_tensor import getall_as_tensor


class OversamplingWrapper(KDSubset):
    def __init__(self, dataset, mode='multiply'):
        self.mode = mode

        classes = getall_as_tensor(dataset)
        class_counts, _ = get_class_counts(classes, dataset.getdim_class())
        max_class_count = int(class_counts.max().item())
        indices = paddle.arange(len(dataset), dtype=paddle.int64)
        if self.mode == 'multiply':
            for i in range(len(class_counts)):
                if class_counts[i] == 0:
                    continue
                multiply_factor = int(np.floor(max_class_count / class_counts[i])) - 1
                if multiply_factor > 0:
                    all_indices = paddle.arange(len(dataset), dtype=paddle.int64)
                    sample_idxs = all_indices[classes == i]
                    indices = paddle.concat([indices, paddle.tile(sample_idxs, repeat_times=[multiply_factor])])
        elif self.mode == 'exact':
            indices = []
            for i in range(len(class_counts)):
                remaining_indices = max_class_count
                indices_for_cur_class = (classes == i).nonzero().flatten()
                while remaining_indices > 0:
                    perm = paddle.arange(len(indices_for_cur_class))[:remaining_indices]
                    indices.append(indices_for_cur_class[perm])
                    remaining_indices -= len(perm)
            indices = paddle.concat(indices)
        else:
            raise NotImplementedError(f"invalid oversampling mode '{self.mode}'")
        super().__init__(dataset=dataset, indices=indices.tolist() if paddle.is_tensor(indices) else indices)
