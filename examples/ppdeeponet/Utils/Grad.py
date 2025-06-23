import paddle


def FDM_2d(u: paddle.to_tensor, dx_or_meshx, dy_or_meshy):
    """Compute 1st deritive of u with FDM
    Input:
        u: size(batch_size, my_size, mx_size, 1)
    Return:
        dudx: size(batch_size, my_size-2, mx_size-2, 1)
        dudy: size(batch_size, my_size-2, mx_size-2, 1)
    """
    if not isinstance(dx_or_meshx, float) or not isinstance(dy_or_meshy, float):
        deltax = dx_or_meshx[:, 1:-1, 2:, :] - dx_or_meshx[:, 1:-1, :-2, :]
        deltay = dy_or_meshy[:, 2:, 1:-1, :] - dy_or_meshy[:, :-2, 1:-1, :]
    else:
        deltax = 2.0 * dx_or_meshx
        deltay = 2.0 * dy_or_meshy
    dudx = (u[:, 1:-1, 2:, :] - u[:, 1:-1, :-2, :]) / deltax
    dudy = (u[:, 2:, 1:-1, :] - u[:, :-2, 1:-1, :]) / deltay
    return dudx, dudy


def du_FDM_2d(u: paddle.to_tensor, deltax: float, dim: int, order: int = 1, padding: str = "zeros"):
    """Compute 1st deritive of u with FDM (be careful with the size of u)
    @@@@: Require equal distance meshgrids!!!
    Input:
        u_mesh: size(batch_size, my_size, mx_size, 1)
        deltax: x(i+1) - x(i)
        dim: deritivate w.r.t x-axis or y-axis
        order: (x(i+1) - x(i-1))/deltax or ...
    Return:
        output: size(batch_size, my_size, mx_size, 1)
    """
    assert dim == 0 or dim == 1
    u = u.transpose(perm=[0, 3, 1, 2])
    if order == 1:
        ddx1D = paddle.to_tensor(data=[-0.5, 0.0, 0.5], dtype="float32").to(u.place)
    elif order == 3:
        ddx1D = paddle.to_tensor(
            data=[-1.0 / 60, 3.0 / 20, -3.0 / 4, 0.0, 3.0 / 4, -3.0 / 20, 1.0 / 60], dtype="float32"
        ).to(u.place)
    else:
        raise NotImplementedError(f"order={order} is not available")
    ddx3D = paddle.reshape(x=ddx1D, shape=[1, 1] + (1 - dim) * [1] + [-1] + dim * [1])
    if padding == "zeros":
        u = paddle.nn.functional.pad(
            x=u, pad=4 * [(tuple(ddx1D.shape)[0] - 1) // 2], mode="constant", value=0, pad_from_left_axis=False
        )
    elif padding == "copy":
        u = paddle.nn.functional.pad(
            x=u, pad=4 * [(tuple(ddx1D.shape)[0] - 1) // 2], mode="replicate", pad_from_left_axis=False
        )
    else:
        raise NotImplementedError(f"padding={padding} is not available")
    output = paddle.nn.functional.conv2d(x=u, weight=ddx3D, padding="valid")
    output = output / deltax
    if dim == 0:
        output = output[:, :, (tuple(ddx1D.shape)[0] - 1) // 2 : -(tuple(ddx1D.shape)[0] - 1) // 2, :]
    else:
        output = output[:, :, :, (tuple(ddx1D.shape)[0] - 1) // 2 : -(tuple(ddx1D.shape)[0] - 1) // 2]
    return output.transpose(perm=[0, 2, 3, 1])


def ddu_FDM_2d(u: paddle.to_tensor, deltax: float, dim: int, order: int = 1, padding: str = "zeros"):
    """Compute 2nd deritive of u with FDM (be careful with the size of u)
    @@@@: Require equal distance meshgrids!!!
    Input:
        u: size(batch_size, my_size, mx_size, 1)
        deltax: x(i+1) - x(i)
        dim: deritivate w.r.t x-axis or y-axis
        order: (x(i+1) - x(i-1))/deltax or ...
    Return:
        output: size(batch_size, my_size, mx_size, 1)
    """
    assert dim == 0 or dim == 1
    u = u.transpose(perm=[0, 3, 1, 2])
    if order == 1:
        ddx1D = paddle.to_tensor(data=[1.0, -2.0, 1.0], dtype="float32").to(u.place)
    elif order == 3:
        ddx1D = paddle.to_tensor(
            data=[1.0 / 90, -3.0 / 20, 3.0 / 2, -49.0 / 18, 3.0 / 2, -3.0 / 20, 1.0 / 90], dtype="float32"
        ).to(u.place)
    else:
        raise NotImplementedError(f"order={order} is not available")
    ddx3D = paddle.reshape(x=ddx1D, shape=[1, 1] + (1 - dim) * [1] + [-1] + dim * [1])
    if padding == "zeros":
        u = paddle.nn.functional.pad(
            x=u, pad=4 * [(tuple(ddx1D.shape)[0] - 1) // 2], mode="constant", value=0, pad_from_left_axis=False
        )
    elif padding == "copy":
        u = paddle.nn.functional.pad(
            x=u, pad=4 * [(tuple(ddx1D.shape)[0] - 1) // 2], mode="replicate", pad_from_left_axis=False
        )
    else:
        raise NotImplementedError(f"padding={padding} is not available")
    output = paddle.nn.functional.conv2d(x=u, weight=ddx3D, padding="valid")
    output = output / deltax**2
    if dim == 0:
        output = output[:, :, (tuple(ddx1D.shape)[0] - 1) // 2 : -(tuple(ddx1D.shape)[0] - 1) // 2, :]
    else:
        output = output[:, :, :, (tuple(ddx1D.shape)[0] - 1) // 2 : -(tuple(ddx1D.shape)[0] - 1) // 2]
    return output.transpose(perm=[0, 2, 3, 1])
