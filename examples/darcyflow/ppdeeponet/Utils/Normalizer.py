import paddle


class UnitGaussianNormalizer:

    def __init__(self, x, eps=1e-08):
        super(UnitGaussianNormalizer, self).__init__()
        """Apply normaliztion to the first dimension of last axis of x
        Input:
            x: size(N, mesh_size, 1+d)
        Output:
            mean: size(mesh_szie, 1)
            std: size(mesh_size, 1)
        """
        self.mean = paddle.mean(x=x[..., 0:1], axis=0)
        self.std = paddle.std(x=x[..., 0:1], axis=0)
        self.eps = eps

    def encode(self, x):
        """
        Input:
            x: x: size(N, mesh_size, 1+d)
        """
        d = tuple(x.shape)[-1] - 1
        x_list = paddle.split(x=x, num_or_sections=[1, d], axis=-1)
        x = paddle.concat(x=[(x_list[0] - self.mean) / (self.std + self.eps), x_list[1]], axis=-1)
        return x

    def decode(self, x, sample_idx=None):
        """ """
        if sample_idx is not None:
            if len(tuple(self.mean.shape)) == len(tuple(sample_idx[0].shape)):
                std = self.std[sample_idx]
                mean = self.mean[sample_idx]
            if len(tuple(self.mean.shape)) > len(tuple(sample_idx[0].shape)):
                std = self.std[:, sample_idx]
                mean = self.mean[:, sample_idx]
        else:
            std = self.std
            mean = self.mean
        d = tuple(x.shape)[-1] - 1
        x_list = paddle.split(x=x, num_or_sections=[1, d], axis=-1)
        x = paddle.concat(x=[x_list[0] * (std + self.eps) + mean, x_list[1]], axis=-1)
        return x


class GaussianNormalizer:

    def __init__(self, x, eps=1e-08):
        super(GaussianNormalizer, self).__init__
        """Apply normaliztion to the first dimension of last axis of x
        Input:
            x: size(N, mesh_size, 1+d)
        Output:
            mean: size()
            std: size()
        """
        self.mean = paddle.mean(x=x[..., 0])
        self.std = paddle.std(x=x[..., 0])
        self.eps = eps

    def encode(self, x):
        """
        Input:
            x: x: size(N, mesh_size, 1+d)
        """
        d = tuple(x.shape)[-1] - 1
        x_list = paddle.split(x=x, num_or_sections=[1, d], axis=-1)
        x = paddle.concat(x=[(x_list[0] - self.mean) / (self.std + self.eps), x_list[1]], axis=-1)
        return x

    def decode(self, x):
        """
        Input:
            x: size(batch*n,?) or size(T*batch*n,?)
        """
        d = tuple(x.shape)[-1] - 1
        x_list = paddle.split(x=x, num_or_sections=[1, d], axis=-1)
        x = paddle.concat(x=[x_list[0] * (self.std + self.eps) + self.mean, x_list[1]], axis=-1)
        return x


class RangeNormalizer:

    def __init__(self, x, low=0.0, high=1.0):
        super(RangeNormalizer, self).__init__()
        """Apply normaliztion to the first dimension of last axis of x
        Input:
            x: size(N, mesh_size, 1+d)
        Output:
            a: size(mesh_size)
            b: size(mesh_size)
        """
        x_min = (paddle.min(x=x[..., 0:1], axis=0), paddle.argmin(x=x[..., 0:1], axis=0))[0].view(-1)
        x_max = (paddle.max(x=x[..., 0:1], axis=0), paddle.argmax(x=x[..., 0:1], axis=0))[0].view(-1)
        self.a = (high - low) / (x_max - x_min)
        self.b = low - self.a * x_min

    def encode(self, x):
        """
        Input:
            x: x: size(N, mesh_size, 1+d)
        """
        d = tuple(x.shape)[-1] - 1
        x_list = paddle.split(x=x, num_or_sections=[1, d], axis=-1)
        x0_size = tuple(x_list[0].shape)
        x0 = x_list[0].reshape(x0_size[0], -1)
        x0 = self.a * x0 + self.b
        x = paddle.concat(x=[x0.reshape(x0_size), x_list[1]], axis=-1)
        return x

    def decode(self, x):
        """
        Input:
            x: size(batch*n,?) or size(T*batch*n,?)
        """
        d = tuple(x.shape)[-1] - 1
        x_list = paddle.split(x=x, num_or_sections=[1, d], axis=-1)
        x0_size = tuple(x_list[0].shape)
        x0 = x_list[0].reshape(x0_size[0], -1)
        x0 = (x0 - self.b) / self.a
        x = paddle.concat(x=[x0.reshape(x0_size), x_list[1]], axis=-1)
        return x
