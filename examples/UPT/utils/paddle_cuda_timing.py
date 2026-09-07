import paddle


def cuda_start_event():
    start_event = paddle.cuda.Event(enable_timing=True)
    start_event.record()
    return start_event


def cuda_end_event(start_event):
    if paddle.distributed.is_available() and paddle.distributed.is_initialized():
        paddle.cuda.synchronize()
        paddle.distributed.barrier()
    end_event = paddle.cuda.Event(enable_timing=True)
    end_event.record()
    paddle.cuda.synchronize()
    return start_event.elapsed_time(end_event) / 1000
