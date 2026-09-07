import paddle

from kappadata.datasets.kd_wrapper import KDWrapper


class LabelSmoothingWrapper(KDWrapper):
    def __init__(self, dataset, smoothing):
        super().__init__(dataset=dataset)
        assert isinstance(smoothing, (int, float)) and 0. <= smoothing <= 1.
        self.smoothing = smoothing

    def getitem_class(self, idx, ctx=None):
        y = self.dataset.getitem_class(idx, ctx)
        if self.smoothing == 0:
            return y
        assert isinstance(y, int) or (paddle.is_tensor(y) and y.ndim == 0)
        n_classes = self.dataset.getdim_class()

        if y == -1:
            return paddle.full(shape=[n_classes], fill_value=-1., dtype=paddle.float32)

        if n_classes == 1:
            off_value = self.smoothing / 2
            return y - off_value if y > 0.5 else y + off_value

        off_value = self.smoothing / n_classes
        on_value = 1. - self.smoothing + off_value
        y_vector = paddle.full(shape=[n_classes], fill_value=off_value, dtype=paddle.float32)
        y_vector[y] = on_value
        return y_vector
