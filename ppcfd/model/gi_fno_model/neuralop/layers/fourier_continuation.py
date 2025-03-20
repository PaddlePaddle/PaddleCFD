import paddle
import numpy as np
from numpy.polynomial.legendre import Legendre


class FCLegendre(paddle.nn.Layer):

    def __init__(self, n, d, dtype='float32'):
        super().__init__()
        self.dtype = dtype
        self.compute_extension_matrix(n, d)

    def compute_extension_matrix(self, n, d):
        self.n = n
        self.d = d
        a = 0.0
        h = 0.1
        total_points = 2 * n + d
        full_grid = a + h * np.arange(total_points, dtype=np.float64)
        fit_grid = np.concatenate((full_grid[0:self.n], full_grid[-self.n:]), 0
            )
        extension_grid = full_grid[self.n:-self.n]
        I = np.eye(2 * self.n, dtype=np.float64)
        polynomials = []
        for j in range(2 * self.n):
            polynomials.append(Legendre(I[j, :], domain=[full_grid[0],
                full_grid[-1]]))
        X = np.zeros((2 * self.n, 2 * self.n), dtype=np.float64)
        Q = np.zeros((self.d, 2 * self.n), dtype=np.float64)
        for j in range(2 * self.n):
            Q[:, j] = polynomials[j](extension_grid)
            X[:, j] = polynomials[j](fit_grid)
        ext_mat = np.matmul(Q, np.linalg.pinv(X, rcond=1e-31))
        self.register_buffer(name='ext_mat', tensor=paddle.to_tensor(data=
            ext_mat).to(dtype=self.dtype))
        self.register_buffer(name='ext_mat_T', tensor=self.ext_mat.T.clone())
        return self.ext_mat

    def extend_left_right(self, x):
        right_bnd = x[..., -self.n:]
        left_bnd = x[..., 0:self.n]
        y = paddle.concat(x=(right_bnd, left_bnd), axis=-1)
        ext = paddle.matmul(x=y, y=self.ext_mat_T)
        return paddle.concat(x=(x, ext), axis=-1)

    def extend_top_bottom(self, x):
        bottom_bnd = x[..., -self.n:, :]
        top_bnd = x[..., 0:self.n, :]
        y = paddle.concat(x=(bottom_bnd, top_bnd), axis=-2)
        ext = paddle.matmul(x=self.ext_mat, y=y)
        return paddle.concat(x=(x, ext), axis=-2)

    def extend2d(self, x):
        x = self.extend_left_right(x)
        x = self.extend_top_bottom(x)
        return x

    def forward(self, x):
        return self.extend2d(x)
