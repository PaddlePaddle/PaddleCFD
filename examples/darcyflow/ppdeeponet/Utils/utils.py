import numpy as np
import paddle


def np2tensor(x: np.array, dtype="float32"):
    """From numpy.array to paddle.to_tensor"""
    return paddle.to_tensor(data=x, dtype=dtype)


def detach2np(x: paddle.to_tensor):
    """Detach -> cpu -> numpy"""
    return x.detach().cpu().numpy()


def mesh1d(n, sub: int = 1, low=0.0, high=1.0):
    """ """
    assert low < high
    assert sub <= n
    mesh = np.linspace(low, high, n).reshape(-1, 1)
    return mesh[::sub, :]


def mesh2d(nx, ny, subx: int = 1, suby: int = 1, xlow=0.0, xhigh=1.0, ylow=0.0, yhigh=1.0):
    assert xlow < xhigh and ylow < yhigh
    assert subx <= nx and suby <= ny
    x_mesh = np.linspace(xlow, xhigh, nx)[::subx]
    y_mesh = np.linspace(ylow, yhigh, ny)[::suby]
    xy_mesh = np.meshgrid(x_mesh, y_mesh)
    mesh = np.vstack([xy_mesh[0].flatten(), xy_mesh[1].flatten()]).T
    return mesh
