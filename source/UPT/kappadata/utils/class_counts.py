import numpy as np
import paddle


def get_class_counts(classes, n_classes):
    if n_classes == 1:
        n_classes = 2

    if not paddle.is_tensor(classes):
        if isinstance(classes, np.ndarray):
            classes = paddle.to_tensor(classes).astype(paddle.int64)
        else:
            classes = paddle.to_tensor(classes, dtype=paddle.int64)
    unlabeled_count = int((classes == -1).sum().item())
    classes = classes[classes != -1]
    assert paddle.all(classes >= 0) and paddle.all(classes < n_classes)

    counts = paddle.zeros([n_classes], dtype=paddle.int64)
    unique_classes, unique_counts = paddle.unique(classes, return_counts=True)
    counts[unique_classes] = unique_counts
    return counts, unlabeled_count


def get_class_counts_from_dataset(dataset):
    classes = [dataset.getitem_class(i) for i in range(len(dataset))]
    return get_class_counts(classes=classes, n_classes=dataset.getdim_class())


def get_class_counts_and_indices(dataset):
    classes = np.array([dataset.getitem_class(i) for i in range(len(dataset))])
    counts, _ = get_class_counts(classes=classes, n_classes=dataset.getdim_class())
    indices = []
    for i in range(dataset.getdim_class()):
        indices.append((classes == i).nonzero()[0])
    return counts, indices
