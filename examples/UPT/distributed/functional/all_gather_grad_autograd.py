import paddle


class AllGatherGradAutograd(paddle.autograd.PyLayer):
    """
    Gathers tensors from all process and supports backward propagation
    for the gradients across processes.
    """

    @staticmethod
    def forward(ctx, x):
        output = [
            paddle.zeros_like(x) for _ in range(paddle.distributed.get_world_size())
        ]
        paddle.distributed.all_gather(tensor_list=output, tensor=x)
        return tuple(output)

    @staticmethod
    def backward(ctx, *grads):
        all_gradients = paddle.stack(grads)
        paddle.distributed.all_reduce(
            tensor=all_gradients, op=paddle.distributed.ReduceOp.SUM
        )
        grad_out = all_gradients[paddle.distributed.get_rank()]
        return grad_out
