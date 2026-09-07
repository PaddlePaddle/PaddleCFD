from kappadata.datasets.kd_subset import KDSubset
from kappadata.error_messages import too_little_samples_for_class
from kappadata.utils.class_counts import get_class_counts_and_indices


class ClasswiseSubsetWrapper(KDSubset):
    def __init__(
            self,
            dataset,
            start_index: int = None,
            end_index: int = None,
            start_percent: float = None,
            end_percent: float = None,
            check_enough_samples: bool = True,
    ):
        counts, all_indices = get_class_counts_and_indices(dataset)
        sub_indices = []

        if start_index is not None or end_index is not None:
            assert start_percent is None and end_percent is None
            # create indices from start/end index
            assert start_index is None or isinstance(start_index, int)
            assert end_index is None or isinstance(end_index, int)
            end_index = end_index or len(dataset)
            end_index = min(end_index, len(dataset))
            start_index = start_index or 0
            assert start_index <= end_index
            for i in range(dataset.getdim_class()):
                if check_enough_samples:
                    assert counts[i] >= end_index, too_little_samples_for_class(i, counts[i], end_index)
                if start_index >= counts[i]:
                    continue
                cur_end_index = min(end_index, counts[i])
                sub_indices += all_indices[i][start_index:cur_end_index].tolist()
        elif start_percent is not None or end_percent is not None:
            # create indices from start/end percent
            assert start_index is None and end_index is None
            assert start_percent is not None or end_percent is not None
            assert start_percent is None or (isinstance(start_percent, (float, int)) and 0. <= start_percent <= 1.)
            assert end_percent is None or (isinstance(end_percent, (float, int)) and 0. <= end_percent <= 1.)
            start_percent = start_percent or 0.
            end_percent = end_percent or 1.
            assert start_percent <= end_percent
            for i in range(dataset.getdim_class()):
                start_index = int(start_percent * counts[i])
                end_index = int(end_percent * counts[i])
                sub_indices += all_indices[i][start_index:end_index].tolist()
        else:
            raise NotImplementedError

        super().__init__(dataset=dataset, indices=sub_indices)
