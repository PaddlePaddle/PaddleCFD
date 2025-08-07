import paddle
from typing import Literal
import fused_segment_csr


def segment_csr(
    src: paddle.Tensor, indptr: paddle.Tensor, reduce: Literal["mean", "sum"]
):
    """segment_csr reduces all entries of a CSR-formatted
    matrix by summing or averaging over neighbors.

    Used to reduce features over neighborhoods
    in neuralop.layers.IntegralTransform

    Parameters
    ----------
    src : torch.Tensor
        tensor of features for each point
    indptr : torch.Tensor
        splits representing start and end indices
        of each neighborhood in src
    reduce : Literal['mean', 'sum']
        how to reduce a neighborhood. if mean,
        reduce by taking the average of all neighbors.
        Otherwise take the sum.
    """
    if reduce not in ["mean", "sum"]:
        raise ValueError("reduce must be one of 'mean', 'sum'")

    n_nbrs = indptr[1:] - indptr[:-1]
    output_shape = list(tuple(src.shape))
    output_shape[0] = tuple(indptr.shape)[0] - 1
    out = paddle.zeros(shape=output_shape)
    for i, start in enumerate(indptr[:-1]):
        if start == tuple(src.shape)[0]:
            break
        for j in range(n_nbrs[i]):
            out[i] += src[start + j]

    if reduce == "mean":
        out_result = paddle.empty_like(out)
        for i, start in enumerate(indptr[:-1]):
            if start == tuple(src.shape)[0]:
                break
            if n_nbrs[i] != 0:
                out_result[i] = out[i] / n_nbrs[i]
        return out_result
    return out


if __name__ == "__main__":
    sample_num = 10
    col_num = 20
    selected_num = 30

    def test_mean_1d():
        src = paddle.rand([sample_num], dtype="float32")
        src.stop_gradient = False
        idx_map = paddle.randint(0, sample_num, [selected_num], dtype="int64")
        selected_sample = src[idx_map]
        csr_index = paddle.to_tensor([0, 3, 4, 7, 9, 11, 15, 20, 20, 21], dtype="int64")

        out_ref = segment_csr(selected_sample, csr_index, "mean")

        src_clone = src.clone().detach_()
        src_clone.stop_gradient = False
        out = fused_segment_csr.select_segment_csr(
            src_clone, idx_map, csr_index, "mean"
        )
        assert out.shape == out_ref.shape
        assert paddle.allclose(out, out_ref)

        grad = paddle.rand(out.shape, out.dtype, out.place)
        out.backward(grad)
        out_ref.backward(grad)
        assert paddle.allclose(src_clone.grad, src.grad)

    def test_sum_1d():
        src = paddle.rand([sample_num], dtype="float32")
        src.stop_gradient = False
        idx_map = paddle.randint(0, sample_num, [selected_num], dtype="int64")
        selected_sample = src[idx_map]
        csr_index = paddle.to_tensor([0, 3, 4, 7, 9, 11, 15, 20, 20, 21], dtype="int64")

        out_ref = segment_csr(selected_sample, csr_index, "sum")

        src_clone = src.clone().detach_()
        src_clone.stop_gradient = False
        out = fused_segment_csr.select_segment_csr(src_clone, idx_map, csr_index, "sum")
        assert out.shape == out_ref.shape
        assert paddle.allclose(out, out_ref)

        grad = paddle.rand(out.shape, out.dtype, out.place)
        out.backward(grad)
        out_ref.backward(grad)
        assert paddle.allclose(src_clone.grad, src.grad)

    def test_mean_2d():
        src = paddle.rand([sample_num, col_num], dtype="float32")
        src.stop_gradient = False
        idx_map = paddle.randint(0, sample_num, [selected_num], dtype="int64")
        selected_sample = src[idx_map]
        csr_index = paddle.to_tensor([0, 3, 4, 7, 9, 11, 15, 20, 20, 21], dtype="int64")

        out_ref = segment_csr(selected_sample, csr_index, "mean")

        src_clone = src.clone().detach_()
        src_clone.stop_gradient = False
        out = fused_segment_csr.select_segment_csr(
            src_clone, idx_map, csr_index, "mean"
        )
        assert out.shape == out_ref.shape
        assert paddle.allclose(out, out_ref)

        grad = paddle.rand(out.shape, out.dtype, out.place)
        out.backward(grad)
        out_ref.backward(grad)
        assert paddle.allclose(src_clone.grad, src.grad)

    def test_sum_2d():
        src = paddle.rand([sample_num, col_num], dtype="float32")
        src.stop_gradient = False
        idx_map = paddle.randint(0, sample_num, [selected_num], dtype="int64")
        selected_sample = src[idx_map]
        csr_index = paddle.to_tensor([0, 3, 4, 7, 9, 11, 15, 20, 20, 21], dtype="int64")

        out_ref = segment_csr(selected_sample, csr_index, "sum")

        src_clone = src.clone().detach_()
        src_clone.stop_gradient = False
        out = fused_segment_csr.select_segment_csr(src_clone, idx_map, csr_index, "sum")
        assert out.shape == out_ref.shape
        assert paddle.allclose(out, out_ref)

        grad = paddle.rand(out.shape, out.dtype, out.place)
        out.backward(grad)
        out_ref.backward(grad)
        assert paddle.allclose(src_clone.grad, src.grad)

    test_mean_1d()
    test_sum_1d()
    test_mean_2d()
    test_sum_2d()
