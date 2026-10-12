from kappadata.datasets.kd_wrapper import KDWrapper


class XRepeatWrapper(KDWrapper):
    def __init__(self, dataset, num_repeats, **kwargs):
        super().__init__(dataset=dataset, **kwargs)
        self.num_repeats = num_repeats

    def getitem_x(self, idx, ctx=None):
        item = self.dataset.getitem_x(idx, ctx=ctx)
        return [item.clone() for _ in range(self.num_repeats)]
