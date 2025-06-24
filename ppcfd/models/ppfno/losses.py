import sys
# sys.path.append('/home/chenkai26/PaddleScience-AeroShapeOpt/paddle_project')
# from .. import utils
import paddle


class LpLoss(object):

    def __init__(self, d=2, p=2, size_average=True, reduction=True):
        super(LpLoss, self).__init__()
        assert d > 0 and p > 0
        self.d = d
        self.p = p
        self.reduction = reduction
        self.size_average = size_average

    def abs(self, x, y):
        num_examples = tuple(x.shape)[0]
        h = 1.0 / (tuple(x.shape)[1] - 1.0)
        all_norms = h ** (self.d / self.p) * paddle.linalg.norm(x=x.view(
            num_examples, -1) - y.view(num_examples, -1), p=self.p, axis=1)
        if self.reduction:
            if self.size_average:
                return paddle.mean(x=all_norms)
            else:
                return paddle.sum(x=all_norms)
        return all_norms

    def rel(self, x, y):
        num_examples = tuple(x.shape)[0]
        diff_norms = paddle.linalg.norm(x=x.reshape((num_examples, -1)) - y.
            reshape((num_examples, -1)), p=self.p, axis=1)
        y_norms = paddle.linalg.norm(x=y.reshape((num_examples, -1)), p=self.
            p, axis=1)
        if self.reduction:
            if self.size_average:
                return paddle.mean(x=diff_norms / y_norms)
            else:
                return paddle.sum(x=diff_norms / y_norms)
        return diff_norms / y_norms

    def __call__(self, x, y):
        return self.rel(x, y)


class HsLoss(object):

    def __init__(self, d=2, p=2, k=1, a=None, group=False, size_average=
        True, reduction=True):
        super(HsLoss, self).__init__()
        assert d > 0 and p > 0
        self.d = d
        self.p = p
        self.k = k
        self.balanced = group
        self.reduction = reduction
        self.size_average = size_average
        if a == None:
            a = [1] * k
        self.a = a

    def rel(self, x, y):
        num_examples = tuple(x.shape)[0]
        diff_norms = paddle.linalg.norm(x=x.reshape(num_examples, -1) - y.
            reshape(num_examples, -1), p=self.p, axis=1)
        y_norms = paddle.linalg.norm(x=y.reshape(num_examples, -1), p=self.
            p, axis=1)
        if self.reduction:
            if self.size_average:
                return paddle.mean(x=diff_norms / y_norms)
            else:
                return paddle.sum(x=diff_norms / y_norms)
        return diff_norms / y_norms

    def __call__(self, x, y, a=None):
        nx = tuple(x.shape)[1]
        ny = tuple(x.shape)[2]
        k = self.k
        balanced = self.balanced
        a = self.a
        x = x.view(tuple(x.shape)[0], nx, ny, -1)
        y = y.view(tuple(y.shape)[0], nx, ny, -1)
        k_x = paddle.concat(x=(paddle.arange(start=0, end=nx // 2, step=1),
            paddle.arange(start=-nx // 2, end=0, step=1)), axis=0).reshape(nx,
            1).tile(repeat_times=[1, ny])
        k_y = paddle.concat(x=(paddle.arange(start=0, end=ny // 2, step=1),
            paddle.arange(start=-ny // 2, end=0, step=1)), axis=0).reshape(
            1, ny).tile(repeat_times=[nx, 1])
        k_x = paddle.abs(x=k_x).reshape(1, nx, ny, 1).to(x.place)
        k_y = paddle.abs(x=k_y).reshape(1, nx, ny, 1).to(x.place)
        x = paddle.fft.fftn(x=x, axes=[1, 2])
        y = paddle.fft.fftn(x=y, axes=[1, 2])
        if balanced == False:
            weight = 1
            if k >= 1:
                weight += a[0] ** 2 * (k_x ** 2 + k_y ** 2)
            if k >= 2:
                weight += a[1] ** 2 * (k_x ** 4 + 2 * k_x ** 2 * k_y ** 2 +
                    k_y ** 4)
            weight = paddle.sqrt(x=weight)
            loss = self.rel(x * weight, y * weight)
        else:
            loss = self.rel(x, y)
            if k >= 1:
                weight = a[0] * paddle.sqrt(x=k_x ** 2 + k_y ** 2)
                loss += self.rel(x * weight, y * weight)
            if k >= 2:
                weight = a[1] * paddle.sqrt(x=k_x ** 4 + 2 * k_x ** 2 * k_y **
                    2 + k_y ** 4)
                loss += self.rel(x * weight, y * weight)
            loss = loss / (k + 1)
        return loss
