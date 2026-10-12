from kappadata.datasets.kd_subset import KDSubset
from kappadata.utils.getall_as_tensor import getall_as_tensor


class SortByClassWrapper(KDSubset):
    def __init__(self, dataset):
        num_classes = dataset.getdim_class()
        classes = getall_as_tensor(dataset)
        indices = []
        for i in range(num_classes):
            indices += (classes == i).nonzero().squeeze(1).tolist()
        super().__init__(dataset=dataset, indices=indices)
