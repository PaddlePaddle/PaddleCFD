import paddle


class AllGatherGradOverwrite:
    @staticmethod
    def apply(x):
        output = [
            paddle.zeros_like(x) for _ in range(paddle.distributed.get_world_size())
        ]
        paddle.distributed.all_gather(tensor_list=output, tensor=x)
        output[paddle.distributed.get_rank()] = x
        return output
