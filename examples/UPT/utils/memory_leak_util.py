import gc

import paddle


def get_tensors_in_memory():
    all_objs = gc.get_objects()
    all_tensors = []
    cuda_tensors = []
    for obj in all_objs:
        try:
            if type(obj).__name__ != "_reduce_op" and paddle.is_tensor(obj):
                all_tensors.append(obj)
                # Paddle 3 EagerParamBase exposes ``place`` instead of ``device``.
                if obj.place.is_gpu_place():
                    cuda_tensors.append(obj)
        except ReferenceError:
            pass
    return all_tensors, cuda_tensors
