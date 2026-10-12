from collections import defaultdict
import math

import numpy as np

from kappadata.datasets.kd_wrapper import KDWrapper
from kappadata.utils.global_rng import GlobalRng


class RandomSuperclassWrapper(KDWrapper):
    """
    randomly merge classes into superclasses
    Example input: dataset with 10 classes and classes_per_superclass=2
    Example output: dataset with 5 classes where each class consists of 2 input classes
    """

    def __init__(self, dataset, classes_per_superclass, superclass_splits=1, shuffle=True, seed=None):
        super().__init__(dataset=dataset)
        self.classes_per_superclass = classes_per_superclass
        self.superclass_splits = superclass_splits
        self.og_num_classes = math.ceil(dataset.getdim_class() / classes_per_superclass)
        if seed is not None:
            rng = np.random.default_rng(seed=seed)
        else:
            rng = GlobalRng()
        if shuffle:
            self.perm = rng.permutation(dataset.getdim_class())
        else:
            self.perm = np.arange(dataset.getdim_class())
        if superclass_splits > 1:
            classes = dataset.getall_class()
            if not isinstance(classes, np.ndarray):
                classes = np.array(classes)
            # shuffle classes to avoid patterns
            if shuffle:
                perm = rng.permutation(len(classes))
                classes = classes[perm]
            else:
                perm = np.arange(len(classes))
            # mapping for each sample to its split
            self.idx_within_class = []
            counter = defaultdict(int)
            for cls in classes:
                self.idx_within_class.append(counter[cls])
                counter[cls] += 1
            # invert shuffling
            inv_perm = np.argsort(perm)
            self.idx_within_class = np.array(self.idx_within_class)[inv_perm]
        else:
            self.idx_within_class = None

    def getshape_class(self):
        return self.og_num_classes * self.superclass_splits,

    def _map_cls(self, idx, cls):
        cls = self.perm[cls] // self.classes_per_superclass
        if self.idx_within_class is not None:
            cls += self.idx_within_class[idx] % self.superclass_splits * self.og_num_classes
        return cls

    def getitem_class(self, idx, ctx=None):
        cls = self.dataset.getitem_class(idx, ctx=ctx)
        return self._map_cls(idx=idx, cls=cls)

    def getall_class(self):
        classes = self.dataset.getall_class()
        return [self._map_cls(idx=idx, cls=cls) for idx, cls in enumerate(classes)]
