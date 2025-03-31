import paddle
from ..resample import resample


def test_resample():
    a = paddle.randn(shape=[10, 20, 40, 50])
    res_scale = [2, 3]
    axis = [-2, -1]
    b = resample(a, res_scale, axis)
    assert tuple(b.shape)[-1] == 3 * tuple(a.shape)[-1] and tuple(b.shape)[-2
        ] == 2 * tuple(a.shape)[-2]
    a = paddle.randn(shape=(10, 20, 40, 50, 60))
    res_scale = [0.5, 3, 4]
    axis = [-3, -2, -1]
    b = resample(a, res_scale, axis)
    assert tuple(b.shape)[-1] == 4 * tuple(a.shape)[-1] and tuple(b.shape)[-2
        ] == 3 * tuple(a.shape)[-2] and tuple(b.shape)[-3] == int(0.5 *
        tuple(a.shape)[-3])
