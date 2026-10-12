from collections import defaultdict
import numpy as np

from kappadata.datasets.kd_subset import KDSubset

from kappadata.utils.global_rng import GlobalRng
from kappadata.utils.getall_as_tensor import getall_as_tensor

class IntraClassShuffleWrapper(KDSubset):
    def __init__(self, dataset, seed=None):
        num_classes = dataset.getdim_class()
        classes = getall_as_tensor(dataset)
        rng = GlobalRng if seed is None else np.random.default_rng(seed=seed)
        # create permutation per class
        cls_to_perm = []
        for i in range(num_classes):
            cls_to_perm.append(rng.permutation((classes == i).nonzero().squeeze(1)))
        # compose indices
        idx_in_cls = defaultdict(int)
        indices = []
        for cls in classes.tolist():
            indices.append(cls_to_perm[cls][idx_in_cls[cls]].item())
            idx_in_cls[cls] += 1
        super().__init__(dataset=dataset, indices=indices)
