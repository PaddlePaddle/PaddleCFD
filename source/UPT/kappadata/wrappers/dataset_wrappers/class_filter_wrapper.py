import numpy as np

from kappadata.datasets.kd_subset import KDSubset


class ClassFilterWrapper(KDSubset):
    def __init__(
            self,
            dataset,
            valid_classes=None,
            invalid_classes=None,
            valid_class_names=None,
            invalid_class_names=None,
    ):
        # map names to numbers
        if valid_class_names is not None:
            assert valid_classes is None and invalid_classes is None and invalid_class_names is None
            class_names = np.array(dataset.class_names)
            valid_classes = np.squeeze(np.argwhere(np.isin(class_names, valid_class_names)), axis=1).tolist()
        if invalid_class_names is not None:
            assert valid_classes is None and invalid_classes is None and valid_class_names is None
            class_names = np.array(dataset.class_names)
            invalid_classes = np.squeeze(np.argwhere(np.isin(class_names, invalid_class_names)), axis=1).tolist()

        # check params and make unique
        self._check_params(valid_classes=valid_classes, invalid_classes=invalid_classes)
        self.valid_classes = set(valid_classes) if valid_classes is not None else None
        self.invalid_classes = set(invalid_classes) if invalid_classes is not None else None

        # use numpy for better performance
        # NOTE: np.isin requires list (not set)
        all_indices = np.arange(len(dataset), dtype=np.int64)
        classes = np.array([dataset.getitem_class(i) for i in all_indices])
        if self.valid_classes is not None:
            indices = all_indices[np.isin(classes, list(self.valid_classes))]
        else:
            indices = all_indices[~np.isin(classes, list(self.invalid_classes))]
        super().__init__(dataset=dataset, indices=indices)

    @staticmethod
    def _check_params(valid_classes, invalid_classes):
        assert (valid_classes is None) ^ (invalid_classes is None)
        if valid_classes is not None:
            assert invalid_classes is None
            assert isinstance(valid_classes, (list, tuple)) and all(isinstance(vc, int) for vc in valid_classes)
        else:
            assert valid_classes is None
            assert isinstance(invalid_classes, (list, tuple)) and all(isinstance(ic, int) for ic in invalid_classes)
