import math
from itertools import combinations_with_replacement

import numpy as np
import paddle


class RadialFun(object):

    def __init__(self):
        self.fun = {
            "linear": self.linear,
            "thin_plate_spline": self.thin_plate_spline,
            "cubic": self.cubic,
            "quintic": self.quintic,
            "multiquadric": self.multiquadric,
            "inverse_multiquadric": self.inverse_multiquadric,
            "inverse_quadratic": self.inverse_quadratic,
            "gaussian": self.gaussian,
        }
        self.min_degree = {"multiquadric": 0, "linear": 0, "thin_plate_spline": 1, "cubic": 1, "quintic": 2}

    def linear(self, r):
        return r

    def thin_plate_spline(self, r, min_eps=1e-07):
        r = paddle.clip(x=r, min=min_eps)
        return r

    def cubic(self, r):
        return r**3

    def quintic(self, r):
        return -(r**5)

    def multiquadric(self, r):
        return -paddle.sqrt(x=r**2 + 1)

    def inverse_multiquadric(self, r):
        return 1 / paddle.sqrt(x=r**2 + 1)

    def inverse_quadratic(self, r):
        return 1 / (r**2 + 1)

    def gaussian(self, r):
        return paddle.exp(x=-(r**2))


class RBFInterpolator(paddle.nn.Layer):

    def __init__(
        self,
        x_mesh: paddle.to_tensor,
        kernel: str = None,
        eps: float = None,
        degree: int = None,
        smoothing: float = 0.0,
        dtype="float32",
    ) -> None:
        super(RBFInterpolator, self).__init__()
        """Radial basis function interpolator in PaddlePaddle.
        Input:
            x_mesh: size(n_mesh, d)
            kernel: str, The kernel type.
            eps: float, shape parameter for the kernel function.
            degree: int, degree of the polynomial added to the interpolation function
            smoothing: float or (n,), tensor of smoothing parameters
            dtype: the datatype of tensors
        """
        assert x_mesh.ndim == 2
        scale_fun = {"linear", "thin_plate_spline", "cubic", "quintic"}
        self.x_mesh = x_mesh
        self.n_mesh, self.dx = tuple(x_mesh.shape)
        self.dtype = dtype
        self.Kernels = RadialFun()
        self.kernel_fun = self.Kernels.fun[kernel]
        if eps is None:
            if kernel in scale_fun:
                self.eps = 1.0
            else:
                raise ValueError("Require eps for the kernel.")
        else:
            self.eps = float(eps)
        if isinstance(smoothing, (int, float)):
            self.smoothing = paddle.full(shape=(self.n_mesh,), fill_value=smoothing, dtype=dtype).to(x_mesh.place)
        elif isinstance(smoothing, np.ndarray):
            smoothing = paddle.to_tensor(data=smoothing, dtype=dtype).to(x_mesh.place)
        elif isinstance(smoothing, paddle.to_tensor):
            smoothing = smoothing.to(x_mesh.place)
        else:
            raise ValueError("smoothing must be a scalar or a 1-dimensional tensor")
        min_degree = self.Kernels.min_degree.get(kernel, -1)
        if degree is None:
            degree = max(min_degree, 0)
        else:
            degree = int(degree)
            if degree < -1:
                raise ValueError("degree must be at least -1")
            elif degree < min_degree:
                raise ValueError(f"degree must be larger than {min_degree}")
        self.powers = self.monomial_powers(self.dx, degree).to(x_mesh.place)
        self.n_monos = tuple(self.powers.shape)[0]
        if self.n_monos > self.n_mesh:
            raise ValueError("The data is not compatible with the requested degree")
        self.lhs, self.shift, self.scale = self.build()
        self.x_eps_mesh = self.x_mesh * self.eps

    def forward(self, x: paddle.to_tensor, a_batch: paddle.to_tensor):
        """Returns the interpolated data at the given points `x`
        Input:
            x: size(n_batch, nx, d)
            a_batch: size(n_batch, n_mesh, 1)
        Return:
            a_pred: size(n_batch, nx)
        """
        assert x.ndim == 3 and a_batch.ndim == 3
        assert tuple(a_batch.shape)[1] == self.n_mesh
        self.coeff = self.solve(a_batch)
        x_eps = x * self.eps
        x_hat = (x - self.shift) / self.scale
        x_eps_mesh = self.x_eps_mesh.tile(repeat_times=(tuple(x.shape)[0], 1, 1))
        kv = self.kernel_matrix(x_eps, x_eps_mesh)
        pmat = self.polynomial_matrix(x_hat, self.powers)
        vec = paddle.concat(x=[kv, pmat], axis=-1)
        a_pred = paddle.matmul(x=vec, y=self.coeff)
        return a_pred

    def solve(self, a_batch: paddle.to_tensor):
        """Build then solve the RBF linear system
        Input:
            a_batch: size(n_batch, n_mesh, 1)
        Return:
            coeffs: size(n_batch, n_mesh+n_monos, 1)
        """
        assert a_batch.ndim == 3 and tuple(a_batch.shape)[1] == self.n_mesh
        lhs = self.lhs.tile(repeat_times=(tuple(a_batch.shape)[0], 1, 1))
        rhs = paddle.empty(shape=(tuple(a_batch.shape)[0], self.n_monos + self.n_mesh, 1), dtype=self.dtype)
        rhs[:, : self.n_mesh, :] = a_batch
        rhs[:, self.n_mesh :, :] = 0.0
        try:
            coeffs = paddle.linalg.solve(x=lhs, y=rhs)
        except RuntimeError:
            msg = "Singlar matrix."
            if self.n_monos > 0:
                pmat = self.polynomial_matrix((self.x_mesh - self.shift) / self.scale, self.powers)
                rank = paddle.linalg.matrix_rank(x=pmat)
                flag = rank < self.n_monos
                if sum(flag) > 0:
                    index = paddle.to_tensor(data=[i for i in range(1, tuple(a_batch.shape)[0] + 1)])
                    index = index[flag]
                    msg = f"Singular matrix. The matrix of monomials evaluated atthe data point coordinates ({index}) does not have full columnrank ({rank[flag]}/{self.n_monos})."
            raise ValueError(msg)
        return coeffs

    def build(self):
        """Build the linear equation: lhs * coeff = rhs"""
        mins = paddle.min(self.x_mesh, axis=0)
        maxs = paddle.max(self.x_mesh, axis=0)
        shift = (maxs + mins) / 2.0
        scale = (maxs - mins) / 2.0
        scale[scale == 0.0] = 1.0
        x_eps = self.x_mesh * self.eps
        x_hat = (self.x_mesh - shift) / scale
        lhs = paddle.empty(shape=(self.n_mesh + self.n_monos, self.n_mesh + self.n_monos), dtype=self.dtype)
        lhs[: self.n_mesh, : self.n_mesh] = self.kernel_matrix(x_eps, x_eps)
        lhs[: self.n_mesh, self.n_mesh :] = self.polynomial_matrix(x_hat, self.powers)
        lhs[self.n_mesh :, : self.n_mesh] = lhs[: self.n_mesh, self.n_mesh :].T
        lhs[self.n_mesh :, self.n_mesh :] = 0.0
        lhs[: self.n_mesh, : self.n_mesh] += paddle.diag(x=self.smoothing)
        return lhs, shift, scale

    def kernel_matrix(self, x_eps, x_eps_base):
        """Returns radial function values for all pairs of points in `x`"""
        return self.kernel_fun(paddle.cdist(x=x_eps, y=x_eps_base))

    def polynomial_matrix(self, x_hat, powers):
        """Evaluate monomials at `x` with given `powers`
        Input:
            x_hat: size(n_batch, nx, dx) or size(n_mesh, dx)
            powers: size(n_monos, dx)
        Out:
            out: size(n_batch nx, n_monos) or size(n_mesh, n_monos)
        """
        if x_hat.ndim == 3:
            x_ = paddle.repeat_interleave(x=x_hat, repeats=tuple(powers.shape)[0], axis=1)
            powers_ = powers.tile(repeat_times=[tuple(x_hat.shape)[1], 1]).astype("float32")
            out = paddle.prod(x=x_**powers_, axis=-1, keepdim=True).view(
                (x_hat.shape[0], x_hat.shape[1], powers.shape[0])
            )
        elif x_hat.ndim == 2:
            x_ = paddle.repeat_interleave(x=x_hat, repeats=tuple(powers.shape)[0], axis=0)
            powers_ = powers.tile(repeat_times=[tuple(x_hat.shape)[0], 1]).astype("float32")
            out = paddle.prod(x=x_**powers_, axis=1).view((x_hat.shape[0], powers.shape[0]))
        else:
            raise TypeError("x_has has a wrong type.")
        return out

    def monomial_powers(self, dx: int, degree: int):
        """Return the powers for each monomial in a polynomial.
        Input:
            dx: int, Number of variables in the polynomial.
            degree: int, Degree of the polynomial.
        Output:
            out: size(n_monos, dx), Array where each row contains the powers
                for each variable in a monomial.
        """
        n_monos = math.comb(degree + dx, dx)
        out = paddle.zeros(shape=(n_monos, dx), dtype="int32")
        count = 0
        for deg in range(degree + 1):
            for mono in combinations_with_replacement(range(dx), deg):
                for var in mono:
                    out[count, var] += 1
                count += 1
        return out
