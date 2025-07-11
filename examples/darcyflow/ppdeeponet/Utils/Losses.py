import paddle


class MyError(object):

    def __init__(self, d=2, p=2, size_average=True, reduction=True):
        super(MyError, self).__init__()
        """Relative/Absolute Lp Error
        Input:
            d: dimension of problem
            p: norm order
        """
        assert d > 0 and p > 0
        self.d = d
        self.p = p
        self.reduction = reduction
        self.size_average = size_average
        self.eps = 1e-08

    def LP_abs(self, y_pred: paddle.to_tensor, y_true: paddle.to_tensor):
        """Absolute Error
        Input:
            y_pred: size(n_batch, n_mesh, 1)
            y_true: size(n_batch, n_mesh, 1)
        """
        assert tuple(y_true.shape) == tuple(y_pred.shape)
        batch_size = tuple(y_true.shape)[0]
        h = 1.0 / (tuple(y_true.shape)[1] - 1.0)
        total_norm = h ** (self.d / self.p) * paddle.linalg.norm(
            x=y_true.reshape([batch_size, -1]) - y_pred.reshape([batch_size, -1]), p=self.p, axis=1
        )
        if self.reduction:
            if self.size_average:
                return paddle.mean(x=total_norm)
            else:
                return paddle.sum(x=total_norm)
        return total_norm

    def Lp_rel(self, y_pred: paddle.to_tensor, y_true: paddle.to_tensor):
        """Relative Error
        Input:
            y_pred: size(n_batch, n_mesh, 1)
            y_true: size(n_batch, n_mesh, 1)
        """
        assert tuple(y_true.shape) == tuple(y_pred.shape)
        batch_size = tuple(y_true.shape)[0]
        diff_norms = paddle.linalg.norm(
            x=y_true.reshape([batch_size, -1]) - y_pred.reshape([batch_size, -1]), p=self.p, axis=1
        )
        y_norms = paddle.linalg.norm(x=y_true.reshape([batch_size, -1]), p=self.p, axis=1) + self.eps
        if self.reduction:
            if self.size_average:
                return paddle.mean(x=diff_norms / y_norms)
            else:
                return paddle.sum(x=diff_norms / y_norms)
        return diff_norms / y_norms


class MyLoss(object):

    def __init__(self, size_average=True, reduction=True):
        super(MyLoss, self).__init__()
        self.reduction = reduction
        self.size_average = size_average
        self.eps = 1e-06

    def mse_org(self, y_pred: paddle.to_tensor, y_true: paddle.to_tensor):
        """The mse loss w/o relative
        Input:
            y_pred: size(n_batch, n_mesh, 1)
            y_true: size(n_batch, n_mesh, 1)
        """
        assert tuple(y_true.shape) == tuple(y_pred.shape)
        batch_size = tuple(y_pred.shape)[0]
        diff_norm = paddle.linalg.norm(
            x=y_true.reshape([batch_size, -1]) - y_pred.reshape([batch_size, -1]), p=2, axis=1
        )
        if self.reduction:
            if self.size_average:
                return paddle.mean(x=diff_norm)
            else:
                return paddle.sum(x=diff_norm)
        return diff_norm

    def mse_rel(self, y_pred: paddle.to_tensor, y_true: paddle.to_tensor):
        """The mse loss w relative
        Input:
           y_pred: size(n_batch, n_mesh, 1)
           y_true: size(n_batch, n_mesh, 1)
        """
        assert tuple(y_true.shape) == tuple(y_pred.shape)
        batch_size = tuple(y_pred.shape)[0]
        diff_norms = paddle.linalg.norm(
            x=y_true.reshape([batch_size, -1]) - y_pred.reshape(batch_size, -1), p=2, axis=1
        )
        y_norms = paddle.linalg.norm(x=y_true.reshape([batch_size, -1]), p=2, axis=1) + self.eps
        if self.reduction:
            if self.size_average:
                return paddle.mean(x=diff_norms / y_norms)
            else:
                return paddle.sum(x=diff_norms / y_norms)
        return diff_norms / y_norms
