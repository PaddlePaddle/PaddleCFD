from kappadata.transforms.base.kd_transform import KDTransform


class KDNormBase(KDTransform):
    def __init__(self, inverse=False, inplace=True):
        super().__init__()
        self.inverse = inverse
        self.inplace = inplace

    def __call__(self, x, ctx=None):
        if self.inverse:
            return self.denormalize(x, inplace=self.inplace)
        return self.normalize(x, inplace=self.inplace)

    def normalize(self, x, inplace=True):
        raise NotImplementedError

    def denormalize(self, x, inplace=True):
        raise NotImplementedError
