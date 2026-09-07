import numpy as np

from kappadata.datasets.kd_subset import KDSubset


class FewshotWrapper(KDSubset):
    def __init__(self, dataset, num_shots, seed=0):
        classes = np.array([dataset.getitem_class(i) for i in range(len(dataset))])
        rng = np.random.default_rng(seed=seed)
        num_classes = np.max(classes) + 1
        indices = []
        arange = np.arange(len(dataset))
        for i in range(num_classes):
            is_cur_class = classes == i
            cur_indices = arange[is_cur_class]
            perm = rng.permutation(len(cur_indices))[:num_shots]
            fewshot_indices = cur_indices[perm]
            indices += fewshot_indices.tolist()
        super().__init__(dataset=dataset, indices=indices)