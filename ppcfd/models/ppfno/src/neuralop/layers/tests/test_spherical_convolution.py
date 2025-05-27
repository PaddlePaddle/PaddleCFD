import paddle
import pytest
from tltorch import FactorizedTensor

from ..spherical_convolution import SHT
from ..spherical_convolution import SphericalConv


@pytest.mark.parametrize(
    "factorization", ["ComplexDense", "ComplexCP", "ComplexTucker", "ComplexTT"]
)
@pytest.mark.parametrize("implementation", ["factorized", "reconstructed"])
def test_SphericalConv(factorization, implementation):
    """Test for SphericalConv (2D only)

    Compares Factorized and Dense convolution output
    Verifies that a dense conv and factorized conv with the same weight produce the same output

    Checks the output size

    Verifies that dynamically changing the number of Fourier modes doesn't break the conv
    """
    n_modes = 6, 6
    conv = SphericalConv(
        3,
        3,
        n_modes,
        n_layers=1,
        bias=False,
        implementation=implementation,
        factorization=factorization,
    )
    conv_dense = SphericalConv(
        3,
        3,
        n_modes,
        n_layers=1,
        bias=False,
        implementation="reconstructed",
        factorization=None,
    )
    conv_dense.weight[0] = FactorizedTensor.from_tensor(
        conv.weight[0].to_tensor(), rank=None, factorization="ComplexDense"
    )
    x = paddle.randn(shape=[2, 3, *(12, 12)])
    res_dense = conv_dense(x)
    res = conv(x)
    assert paddle.allclose(x=res_dense, y=res).item(), ""
    block = SphericalConv(3, 4, n_modes, n_layers=1, output_scaling_factor=0.5)
    x = paddle.randn(shape=[2, 3, *(12, 12)])
    res = block(x)
    assert list(tuple(res.shape)[2:]) == [12 // 2, 12 // 2]
    block = SphericalConv(3, 4, n_modes, n_layers=1, output_scaling_factor=2)
    x = paddle.randn(shape=[2, 3, *(12, 12)])
    res = block(x)
    assert tuple(res.shape)[1] == 4
    assert list(tuple(res.shape)[2:]) == [12 * 2, 12 * 2]
    block = SphericalConv(
        4,
        4,
        n_modes,
        n_layers=2,
        sht_grids=["equiangular", "legendre-gauss", "equiangular"],
    )
    x = paddle.randn(shape=[2, 4, *(12, 12)])
    res = block[0](x)
    res = block[1](res)
    assert tuple(res.shape)[2:] == tuple(x.shape)[2:]
    res = block[0].transform(x)
    res = block[1].transform(res)
    assert tuple(res.shape)[2:] == tuple(x.shape)[2:]


@pytest.mark.parametrize("grid", ["equiangular", "legendre-gauss"])
def test_sht(grid):
    nlat = 16
    nlon = 2 * nlat
    batch_size = 2
    if grid == "equiangular":
        mmax = nlat // 2
    else:
        mmax = nlat
    lmax = mmax
    norm = "ortho"
    dtype = "float32"
    sht_handle = SHT(dtype=dtype)
    coeffs = paddle.zeros(shape=[batch_size, lmax, mmax], dtype="complex64")
    coeffs[:, :lmax, :mmax] = paddle.randn(
        shape=[batch_size, lmax, mmax], dtype="complex64"
    )
    signal = sht_handle.isht(coeffs, s=(nlat, nlon), grid=grid, norm=norm).to("float32")
    coeffs = sht_handle.sht(signal, s=(lmax, mmax), grid=grid, norm=norm)
    rec = sht_handle.isht(coeffs, s=(nlat, nlon), grid=grid, norm=norm)
    assert paddle.allclose(rtol=0.0001, atol=0.0001, x=signal, y=rec).item(), ""
