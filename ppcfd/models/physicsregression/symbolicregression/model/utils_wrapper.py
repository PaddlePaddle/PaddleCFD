import time
import traceback
from abc import ABC, abstractmethod
from functools import partial

import numpy as np
import paddle
import sklearn
import sympy as sp
from scipy.optimize import minimize


class TimedFun:
    def __init__(self, fun, verbose=False, stop_after=3):
        self.fun_in = fun
        self.started = False
        self.stop_after = stop_after
        self.best_fun_value = np.inf
        self.best_x = None
        self.loss_history = []
        self.verbose = verbose

    def fun(self, x, *args):
        if self.started is False:
            self.started = time.time()
        elif abs(time.time() - self.started) >= self.stop_after:
            self.loss_history.append(self.best_fun_value)
            raise ValueError("Time is over.")
        self.fun_value = self.fun_in(x, *args)
        self.loss_history.append(self.fun_value)
        if self.best_x is None:
            self.best_x = x
        elif self.fun_value < self.best_fun_value:
            self.best_fun_value = self.fun_value
            self.best_x = x
        self.x = x
        if np.isnan(self.fun_value):
            raise ValueError("invalid x!")
        return self.fun_value


class Scaler(ABC):
    """
    Base class for scalers
    """

    def __init__(self):
        pass

    @abstractmethod
    def fit(self, X):
        pass

    @abstractmethod
    def fit_transform(self, X):
        pass

    @abstractmethod
    def transform(self, X):
        pass

    @abstractmethod
    def get_params(self):
        pass

    def rescale_function(self, env, tree, a, b):
        prefix = tree.prefix().split(",")
        idx = 0
        while idx < len(prefix):
            if prefix[idx].startswith("x_"):
                k = int(prefix[idx][-1])
                if k >= len(a):
                    idx += 1
                    continue
                a_k, b_k = str(a[k]), str(b[k])
                prefix_to_add = ["add", b_k, "mul", a_k, prefix[idx]]
                prefix = (
                    prefix[:idx] + prefix_to_add + prefix[min(idx + 1, len(prefix)) :]
                )
                idx += len(prefix_to_add)
            else:
                idx += 1
                continue
        rescaled_tree = env.word_to_infix(prefix, is_float=False, str_array=False)
        return rescaled_tree


class StandardScaler(Scaler):
    def __init__(self):
        """
        transformation is:
        x' =  (x - mean)/std
        """
        self.scaler = sklearn.preprocessing.StandardScaler()

    def fit(self, X):
        self.scaler.fit(X)

    def fit_transform(self, X):
        scaled_X = self.scaler.fit_transform(X)
        return scaled_X

    def transform(self, X):
        m, s = self.scaler.mean_, np.sqrt(self.scaler.var_)
        return (X - m) / s

    def get_params(self):
        m, s = self.scaler.mean_, np.sqrt(self.scaler.var_)
        a, b = 1 / s, -m / s
        return a, b


class MinMaxScaler(Scaler):
    def __init__(self):
        """
        transformation is:
        x' =  2.*(x-xmin)/(xmax-xmin)-1.
        """
        self.scaler = sklearn.preprocessing.MinMaxScaler(feature_range=(-1, 1))

    def fit(self, X):
        self.scaler.fit(X)

    def fit_transform(self, X):
        scaled_X = self.scaler.fit_transform(X)
        return scaled_X

    def transform(self, X):
        val_min, val_max = self.scaler.data_min_, self.scaler.data_max_
        return 2 * (X - val_min) / (val_max - val_min) - 1.0

    def get_params(self):
        val_min, val_max = self.scaler.data_min_, self.scaler.data_max_
        a, b = 2.0 / (val_max - val_min), -1.0 - 2.0 * val_min / (val_max - val_min)
        return a, b


class BFGSRefinement:
    """
    Wrapper around scipy's BFGS solver
    """

    def __init__(self):
        """
        Args:
            func: a PyTorch function that maps dependent variabels and
                    parameters to function outputs for all data samples
                    `func(x, coeffs) -> y`
            x, y: problem data as PyTorch tensors. Shape of x is (d, n) and
                    shape of y is (n,)
        """
        super().__init__()

    def go(
        self,
        env,
        tree,
        coeffs0,
        X,
        y,
        downsample=-1,
        stop_after=10,
        safely_refine=False,
    ):
        generator = env.generator
        safely_refine = True
        if safely_refine:
            func = env.simplifier.safely_tree_to_paddle_module(
                tree, dtype=paddle.float64
            )
        else:
            func = env.simplifier.tree_to_paddle_module(tree, dtype=paddle.float64)
        self.X, self.y = X, y
        if downsample > 0:
            self.X = self.X[:downsample]
            self.y = self.y[:downsample]
        self.X = paddle.from_numpy(self.X).to(paddle.float64).requires_grad_(False)
        self.y = paddle.from_numpy(self.y).to(paddle.float64).requires_grad_(False)
        self.func = partial(func, self.X)

        def objective_paddle(coeffs):
            """
            Compute the non-linear least-squares objective value
                objective(coeffs) = (1/2) sum((y - func(coeffs)) ** 2)
            Returns a PyTorch tensor.
            """
            if not isinstance(coeffs, paddle.Tensor):
                coeffs = paddle.tensor(coeffs, dtype=paddle.float64, requires_grad=True)
            y_tilde = self.func(coeffs)
            if y_tilde is None:
                return None
            mse = (self.y - y_tilde).pow(2).mean().div(2)
            return mse

        def objective_numpy(coeffs):
            """
            Return the objective value as a float (for scipy).
            """
            return objective_paddle(coeffs).item()

        def gradient_numpy(coeffs):
            """
            使用 paddle.grad 实现
            """
            if not isinstance(coeffs, paddle.Tensor):
                coeffs = paddle.tensor(coeffs, dtype=paddle.float64, requires_grad=True)
            else:
                coeffs = coeffs.clone().detach().requires_grad_(True)
            output = objective_paddle(coeffs)
            grad_obj = paddle.grad(
                outputs=output, inputs=coeffs, create_graph=False, retain_graph=False
            )[0]
            return grad_obj.detach().numpy()

        objective_numpy_timed = TimedFun(objective_numpy, stop_after=stop_after)
        try:
            res = minimize(
                objective_numpy_timed.fun,
                coeffs0,
                method="BFGS",
                jac=gradient_numpy,
                options={"disp": False},
            )
        except ValueError as e:
            message = traceback.format_exc()
            return generator.wrap_equation_floats(tree, coeffs0), False
        best_constants = objective_numpy_timed.best_x
        return generator.wrap_equation_floats(tree, best_constants), res.success
