import math
from collections import defaultdict

import numpy as np
import paddle

from kappadata.datasets.kd_wrapper import KDWrapper
from kappadata.utils.getall_as_tensor import getall_as_list, getall
from kappadata.utils.global_rng import GlobalRng


class ClassGroupsWrapper(KDWrapper):
    def __init__(self, dataset, classes_per_group, shuffle=False, seed=0):
        super().__init__(dataset=dataset)
        self.classes_per_group = classes_per_group
        rng = GlobalRng() if seed is None else np.random.default_rng(seed=seed)
        classes = getall_as_list(dataset, item="class")

        # generate cls to clsgrp
        # Example: num_classes=10 classes_per_group=2 [0, 1, 2, 3, 4, 0, 1, 2, 3, 4] -> grp0=[0,5] grp1=[1,6] ...
        # Example: num_classes=8 classes_per_group=4 [0, 1, 0, 1, 0, 1, 0, 1] -> grp0=[0,2,4,6] grp1=[1,3,5,7]
        num_classes = dataset.getdim_class()
        num_clsgroups = math.ceil(num_classes / self.classes_per_group)
        self.cls_to_clsgroup = np.arange(num_clsgroups).repeat(self.classes_per_group)
        if shuffle:
            self.cls_to_clsgroup = rng.permuted(self.cls_to_clsgroup)

        self.idx_within_class = []
        counter = defaultdict(int)
        for cls in classes:
            self.idx_within_class.append(counter[cls])
            counter[cls] += 1

    def _map_cls(self, idx, cls):
        cls_group = self.cls_to_clsgroup[cls]
        idx_within_cls_group = self.idx_within_class[idx] % self.classes_per_group
        cls = cls_group * self.classes_per_group + idx_within_cls_group
        return cls

    def getitem_class(self, idx, ctx=None):
        cls = self.getitem_class_before_grouping(idx, ctx=ctx)
        return self._map_cls(idx=idx, cls=cls)

    def getall_class(self):
        classes = self.getall_class_before_grouping()
        return [self._map_cls(idx=idx, cls=cls) for idx, cls in enumerate(classes)]

    def getitem_class_before_grouping(self, idx, ctx=None):
        return self.dataset.getitem_class(idx, ctx=ctx)

    def getall_class_before_grouping(self):
        return getall(self.dataset, item="class")
