import numpy as np
import paddle
from scipy.stats import qmc


class Point1D:

    def __init__(self, x_lb=0.0, x_ub=1.0, dataType="float32", random_seed=None):
        self.lb = x_lb
        self.ub = x_ub
        self.dtype = dataType
        np.random.seed(random_seed)
        self.lhs_x = qmc.LatinHypercube(1, seed=random_seed)

    def inner_point(self, num_sample: int = 100, method="uniform"):
        """ """
        if method == "mesh":
            X = np.linspace(self.lb, self.ub, num_sample).reshape(-1, 1)
        elif method == "uniform":
            X = np.random.uniform(self.lb, self.ub, num_sample).reshape(-1, 1)
        elif method == "hypercube":
            X = qmc.scale(self.lhs_x.random(num_sample), self.lb, self.ub)
        else:
            raise NotImplementedError
        return paddle.to_tensor(data=X, dtype=self.dtype)

    def boundary_point(self):
        """ """
        raise NotImplementedError

    def weight_centers(self):
        """ """
        raise NotImplementedError

    def integral_mesh(self):
        """ """
        raise NotImplementedError


class Point2D:

    def __init__(self, x_lb=[0.0, 0.0], x_ub=[1.0, 1.0], dataType="float32", random_seed=None):
        self.lb = x_lb
        self.ub = x_ub
        self.dtype = dataType
        np.random.seed(random_seed)
        self.lhs_1d = qmc.LatinHypercube(1, seed=random_seed)
        self.lhs_x = qmc.LatinHypercube(2, seed=random_seed)

    def inner_point(self, num_sample_or_mesh: int, method="uniform"):
        """Points inside the domain"""
        if method == "mesh":
            x_mesh = np.linspace(self.lb[0], self.ub[0], num_sample_or_mesh)
            y_mesh = np.linspace(self.lb[1], self.ub[1], num_sample_or_mesh)
            x_mesh, y_mesh = np.meshgrid(x_mesh, y_mesh)
            X = np.vstack((x_mesh.flatten(), y_mesh.flatten())).T
        elif method == "uniform":
            x_rand = np.random.uniform(self.lb[0], self.ub[0], num_sample_or_mesh)
            y_rand = np.random.uniform(self.lb[1], self.ub[1], num_sample_or_mesh)
            X = np.vstack((x_rand.flatten(), y_rand.flatten())).T
        elif method == "hypercube":
            X = qmc.scale(self.lhs_x.random(num_sample_or_mesh), np.array(self.lb), np.array(self.ub))
        else:
            raise NotImplementedError
        return paddle.to_tensor(data=X, dtype=self.dtype)

    def inner_point_sphere(self, num_sample: int, xc, radius, method: str = "muller"):
        """Points inside a sphere"""
        if method == "muller":
            x = np.random.normal(size=(num_sample, 2))
            r = np.power(np.random.random(size=(num_sample, 1)), 1 / 2)
            x = x * r / np.sqrt(np.sum(x**2, axis=1, keepdims=True))
        elif method == "dropped":
            x = np.random.normal(size=(num_sample, 2 + 2))
            x = x / np.sqrt(np.sum(x**2, axis=1, keepdims=True))
            x = x[..., 0:2]
        elif method == "mesh":
            x_mesh, y_mesh = np.meshgrid(np.linspace(-1.0, 1.0, num_sample), np.linspace(-1.0, 1.0, num_sample))
            grid = np.concatenate([x_mesh.reshape(-1, 1), y_mesh.reshape(-1, 1)], axis=1)
            index = np.where(np.linalg.norm(grid, axis=1, keepdims=True) < 1.0)[0]
            x = grid[index, :]
        else:
            raise NotImplementedError
        X = x * radius + np.array(xc)
        return paddle.to_tensor(data=X, dtype=self.dtype)

    def boundary_point(self, num_each_edge: int, method="uniform"):
        """Points on the boundary"""
        X_bd = []
        for d in range(2):
            if method == "mesh":
                x_mesh = np.linspace(self.lb[0], self.ub[0], num_each_edge)
                y_mesh = np.linspace(self.lb[1], self.ub[1], num_each_edge)
                X_lb = np.vstack((x_mesh, y_mesh)).T
                X_ub = np.vstack((x_mesh, y_mesh)).T
            elif method == "uniform":
                X_lb = np.random.uniform(self.lb, self.ub, (num_each_edge, 2))
                X_ub = np.random.uniform(self.lb, self.ub, (num_each_edge, 2))
            elif method == "hypercube":
                X_lb = qmc.scale(self.lhs_x.random(num_each_edge), np.array(self.lb), np.array(self.ub))
                X_ub = qmc.scale(self.lhs_x.random(num_each_edge), np.array(self.lb), np.array(self.ub))
            else:
                raise NotImplementedError
            X_lb[:, d] = self.lb[d]
            X_bd.append(paddle.to_tensor(data=X_lb, dtype=self.dtype))
            X_ub[:, d] = self.ub[d]
            X_bd.append(paddle.to_tensor(data=X_ub, dtype=self.dtype))
        return paddle.concat(x=X_bd, axis=0)

    def boundary_point_sphere(self, num_sample: int, xc, radius, method: str = "mesh"):
        """Points on the surface of a sphere"""
        if method == "muller":
            x = np.random.normal(size=(num_sample, 2))
            x = x / np.sqrt(np.sum(x**2, axis=1, keepdims=True))
        elif method == "hypercube":
            theta = [2.0 * np.pi] * self.lhs_1d.random(num_sample).flatten()
            x = np.vstack((np.cos(theta), np.sin(theta))).T
        elif method == "mesh":
            theta = np.linspace(0.0, 2.0 * np.pi, num_sample)
            x = np.vstack((np.cos(theta), np.sin(theta))).T
        else:
            raise NotImplementedError
        X = x * radius + np.array(xc)
        return paddle.to_tensor(data=X, dtype=self.dtype)

    def weight_centers(self, n_center: int, R_max: float = 0.0001, R_min: float = 0.0001):
        """Generate centers of compact support regions for test functions"""
        if R_max < R_min:
            raise ValueError("R_max should be larger than R_min.")
        elif 2.0 * R_max > np.min(np.array(self.ub) - np.array(self.lb)):
            raise ValueError("R_max is too large.")
        elif R_min < 0.0001 and self.dtype == "float32":
            raise ValueError("R_min<1e-4 when data_type is paddle.float32!")
        elif R_min < 1e-10 and self.dtype == "float64":
            raise ValueError("R_min<1e-10 when data_type is paddle.float64!")
        R = np.random.uniform(R_min, R_max, [n_center, 1])
        lb, ub = np.array(self.lb) + R, np.array(self.ub) - R
        xc = self.lhs_x.random(n_center) * (ub - lb) + lb
        return paddle.to_tensor(data=xc, dtype=self.dtype).view(-1, 1, 2), paddle.to_tensor(
            data=R, dtype=self.dtype
        ).view(-1, 1, 1)

    def integral_grid(self, n_mesh_or_grid: int = 9, method="mesh"):
        """Meshgrid for calculating integrals in [-1.,1.]^2"""
        if method == "mesh":
            x_mesh, y_mesh = np.meshgrid(
                np.linspace(-1.0, 1.0, n_mesh_or_grid), np.linspace(-1.0, 1.0, n_mesh_or_grid)
            )
            grid = np.concatenate([x_mesh.reshape(-1, 1), y_mesh.reshape(-1, 1)], axis=1)
            index = np.where(np.linalg.norm(grid, axis=1, keepdims=True) < 1.0)[0]
            grid_scaled = grid[index, :]
        else:
            raise NotImplementedError(f"No {method} method.")
        return paddle.to_tensor(data=grid_scaled, dtype=self.dtype)
