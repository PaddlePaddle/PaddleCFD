import paddle
import fused_segment_csr._C as _C


class SelectSegmentCsrMean(paddle.autograd.PyLayer):
    @staticmethod
    def forward(ctx, *args, **kwargs):
        assert len(args) == 3
        assert len(kwargs) == 0
        src, idx_map, indptr = args
        ctx.src_shape = src.shape
        out= _C.select_segment_csr_mean(src, idx_map, indptr)
        ctx.save_for_backward(idx_map, indptr)
        return out

    @staticmethod
    def backward(ctx, *args):
        assert len(args) == 1
        grad_output = args[0]
        idx_map, indptr = ctx.saved_tensor()
        src_shape = ctx.src_shape
        return _C.select_segment_csr_mean_bwd(
            src_shape, grad_output, idx_map, indptr
        )


class SelectSegmentCsrSum(paddle.autograd.PyLayer):
    @staticmethod
    def forward(ctx, *args, **kwargs):
        assert len(args) == 3
        assert len(kwargs) == 0
        src, idx_map, indptr = args
        ctx.src_shape = src.shape
        ctx.save_for_backward(idx_map, indptr)
        return _C.select_segment_csr_sum(src, idx_map, indptr)

    @staticmethod
    def backward(ctx, *args):
        assert len(args) == 1
        grad_output = args[0]
        idx_map, indptr = ctx.saved_tensor()
        src_shape = ctx.src_shape
        return _C.select_segment_csr_sum_bwd(src_shape, grad_output, idx_map, indptr)


def select_segment_csr(src, idx_map, indptr, reduce="sum"):
    if reduce == "sum":
        return SelectSegmentCsrSum.apply(src, idx_map, indptr)
    elif reduce == "mean":
        return SelectSegmentCsrMean.apply(src, idx_map, indptr)
    else:
        raise ValueError(f"Unsupported reduce: {reduce}. Use 'sum' or 'mean'.")


__all__ = ["select_segment_csr"]
