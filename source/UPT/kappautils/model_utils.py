import paddle
from contextlib import contextmanager

from .context_managers import temp_mode_change



def get_paramnames_with_no_gradient(model):
    return [name for name, param in model.named_parameters() if getattr(param, 'grad', None) is None and not param.stop_gradient]



def get_output_shape_of_model(model, forward_fn, **forward_kwargs):
    with temp_mode_change(model=model, mode=False):
        x = paddle.ones([1, *model.input_shape], dtype=paddle.float32)
        output = forward_fn(x, **forward_kwargs)
    return tuple(output.shape[1:])


@paddle.no_grad()
def copy_params(source_model, target_model):
    for target_param, source_param in zip(target_model.parameters(), source_model.parameters()):
        target_param.set_value(source_param)


@paddle.no_grad()
def update_ema(source_model, target_model, target_factor):
    source_factor = 1. - target_factor
    for target_param, source_param in zip(target_model.parameters(), source_model.parameters()):
        target_param.set_value(target_param * target_factor + source_param * source_factor)
    for target_buffer, source_buffer in zip(target_model.buffers(), source_model.buffers()):
        target_buffer.set_value(source_buffer)


def get_trainable_param_count(model):
    return sum(p.numel() for p in model.parameters() if not p.stop_gradient)


def get_frozen_param_count(model):
    return sum(p.numel() for p in model.parameters() if p.stop_gradient)


@paddle.no_grad()
def backup_all_buffers(model):
    buffers = {}
    for name, buffer in model.named_buffers():
        buffers[name] = buffer.clone()
    return buffers


@paddle.no_grad()
def restore_all_buffers(model, buffers):
    for name, buffer in model.named_buffers():
        buffer.set_value(buffers[name])
    return buffers


@contextmanager
def preserve_buffers(model):
    buffer_bkp = backup_all_buffers(model=model)
    yield
    restore_all_buffers(model=model, buffers=buffer_bkp)
