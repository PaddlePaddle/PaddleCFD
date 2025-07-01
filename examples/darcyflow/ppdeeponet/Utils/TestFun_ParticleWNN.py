import math
import sys

import numpy as np
import paddle


class TestFun_ParticleWNN:
    """
    (1) Bump
    (2) Wendland
    (3) Cosin
    (4) Wendland with power k (k>=2)
    """

    def __init__(
        self,
        fun_type: str = "Cosin",
        dim: int = 1,
        n_mesh_or_grid: int = 9,
        grid_method: str = "mesh",
        dataType="float32",
    ):
        """
        Input:
            type: the test function type
            dim: the dimension of the problem
        """
        self._dim = dim
        self._eps = sys.float_info.epsilon
        self._n_mesh_or_grid = n_mesh_or_grid
        self._grid_method = grid_method
        self._dtype = dataType
        fun_dict = {
            "Cosin": self._Cosin,
            "Bump": self._Bump,
            "Wendland": self._Wendland,
            "Wendland_k": self._Wend_powerK,
        }
        if fun_type in fun_dict.keys():
            self.testFun = fun_dict[fun_type]
        else:
            raise NotImplementedError(f"No {fun_type} test function type.")

    def _dist(self, x: paddle.to_tensor) -> paddle.to_tensor:
        return paddle.linalg.norm(x, axis=-1, keepdim=True)

    def _grad(self, x: paddle.to_tensor, y: paddle.to_tensor) -> paddle.to_tensor:
        dy = paddle.grad(inputs=x, outputs=y, grad_outputs=paddle.ones_like(x=y), create_graph=True)[0]
        return dy

    def _Bump(self, x_mesh: paddle.to_tensor, dim: int = 1) -> paddle.to_tensor:
        r = 1.0 - paddle.nn.functional.relu(x=1.0 - self._dist(x_mesh))
        r_list = [r]
        for _ in range(3):
            r_list.append(r * r_list[-1])
        v = paddle.exp(x=1.0 - 1.0 / (1.0 - r_list[1] + self._eps))
        dv_dr_divide_by_r = v * -2.0 / ((1.0 - r_list[1]) ** 2 + self._eps)
        if dim == 1:
            dv = dv_dr_divide_by_r * r * paddle.sign(x=x_mesh)
        else:
            dv = dv_dr_divide_by_r * x_mesh
        return v.detach(), dv.detach()

    def _Wendland(self, x_mesh: paddle.to_tensor, dim: int = 1) -> paddle.to_tensor:
        l = math.floor(dim / 2) + 3  # noqa: E741
        r = 1.0 - paddle.nn.functional.relu(x=1.0 - self._dist(x_mesh))
        r_list = [r]
        for _ in range(1):
            r_list.append(r * r_list[-1])
        v = (1 - r) ** (l + 2) * ((l**2 + 4.0 * l + 3.0) * r_list[1] + (3.0 * l + 6.0) * r + 3.0) / 3.0
        dv_dr_divide_by_r = (
            (1 - r) ** (l + 1) * (-(l**3 + 8.0 * l**2 + 19.0 * l + 12) * r - (l**2 + 7.0 * l + 12)) / 3.0
        )
        if dim == 1:
            dv = dv_dr_divide_by_r * r * paddle.sign(x=x_mesh)
        else:
            dv = dv_dr_divide_by_r * x_mesh
        return v.detach(), dv.detach()

    def _Cosin(self, x_mesh: paddle.to_tensor, dim: int = 1) -> paddle.to_tensor:
        r = 1.0 - paddle.nn.functional.relu(x=1.0 - self._dist(x_mesh))
        v = (1.0 - paddle.cos(x=np.pi * (r + 1.0))) / paddle.to_tensor(np.pi)
        dv_dr_divide_by_r = paddle.sin(x=np.pi * (r + 1.0)) / (r + self._eps)
        if dim == 1:
            dv = dv_dr_divide_by_r * r * paddle.sign(x=x_mesh)
        else:
            dv = dv_dr_divide_by_r * x_mesh
        return v.detach(), dv.detach()

    def _Wend_powerK(self, x_mesh: paddle.to_tensor, dim: int = 1, k: int = 4) -> paddle.to_tensor:
        l = math.floor(dim / 2) + 3  # noqa: E741
        r = 1.0 - paddle.nn.functional.relu(x=1.0 - self._dist(x_mesh))
        r_list = [r]
        for _ in range(1):
            r_list.append(r * r_list[-1])
        v_wend = (1 - r) ** (l + 2) * ((l**2 + 4.0 * l + 3.0) * r_list[1] + (3.0 * l + 6.0) * r + 3.0) / 3.0
        dv_dr_divide_by_r_wend = (
            (1 - r) ** (l + 1) * (-(l**3 + 8.0 * l**2 + 19.0 * l + 12) * r - (l**2 + 7.0 * l + 12)) / 3.0
        )
        v = v_wend**k
        dv_dr_divide_by_r = k * v_wend ** (k - 1) * dv_dr_divide_by_r_wend
        if dim == 1:
            dv = dv_dr_divide_by_r * r * paddle.sign(x=x_mesh)
        else:
            dv = dv_dr_divide_by_r * x_mesh
        return v.detach(), dv.detach()

    def integral_grid(self, n_mesh_or_grid: int, method="mesh", dtype="float32"):
        """Meshgrid for calculating integrals from [-1.,1.]^2"""
        if method == "mesh":
            if self._dim == 1:
                grid_scaled = np.linspace(-1.0, 1.0, n_mesh_or_grid).reshape(-1, 1)
            elif self._dim == 2:
                x_mesh, y_mesh = np.meshgrid(
                    np.linspace(-1.0, 1.0, n_mesh_or_grid), np.linspace(-1.0, 1.0, n_mesh_or_grid)
                )
                grid = np.concatenate([x_mesh.reshape(-1, 1), y_mesh.reshape(-1, 1)], axis=1)
                index = np.where(np.linalg.norm(grid, axis=1, keepdims=True) < 1.0)[0]
                grid_scaled = grid[index, :]
            else:
                NotImplementedError(f"dim>{self._dim} is not available")
        else:
            raise NotImplementedError(f"No {method} method.")
        return paddle.to_tensor(data=grid_scaled, dtype=dtype)

    def get_testFun(self, grids: paddle.to_tensor = None) -> paddle.to_tensor:
        """
        Get the test function
        """
        if grids is None:
            grids = self.integral_grid(self._n_mesh_or_grid, self._grid_method, self._dtype)
        v, dv = self.testFun(grids, self._dim)
        return grids, v, dv
