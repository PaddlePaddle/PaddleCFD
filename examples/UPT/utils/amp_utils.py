import logging

import paddle

FLOAT32_ALIASES = ["float32", 32]
FLOAT16_ALIASES = ["float16", 16]
BFLOAT16_ALIASES = ["bfloat16", "bf16"]
VALID_PRECISIONS = FLOAT32_ALIASES + FLOAT16_ALIASES + BFLOAT16_ALIASES


def get_supported_precision(desired_precision, device, backup_precision=None):
    assert desired_precision in VALID_PRECISIONS
    if backup_precision is not None:
        assert backup_precision in VALID_PRECISIONS
    if desired_precision in FLOAT32_ALIASES:
        return paddle.float32
    if desired_precision in FLOAT16_ALIASES:
        desired_precision = "float16"
    if desired_precision in BFLOAT16_ALIASES:
        desired_precision = "bfloat16"
    if desired_precision == "bfloat16":
        if is_bfloat16_compatible(device):
            return paddle.bfloat16
        else:
            if backup_precision is not None and backup_precision in FLOAT16_ALIASES:
                if is_float16_compatible(device):
                    logging.info("bfloat16 not supported -> using float16")
                    return paddle.float16
                else:
                    logging.info("bfloat16/float16 not supported -> using float32")
                    return paddle.float32
            logging.info("bfloat16 not supported -> using float32")
            return paddle.float32
    if desired_precision == "float16":
        if is_float16_compatible(device):
            return paddle.float16
        elif is_bfloat16_compatible(device):
            logging.info(f"float16 not supported -> using bfloat16")
            return paddle.bfloat16
    logging.info(f"float16/bfloat16 not supported -> using float32")
    return paddle.float32


def _is_compatible(device, dtype):
    try:
        if not isinstance(dtype, str):
            if dtype == paddle.bfloat16:
                dtype = "bfloat16"
            elif dtype == paddle.float16:
                dtype = "float16"
            else:
                dtype = "float32" # 默认兜底
        with paddle.amp.auto_cast(dtype=dtype):
            pass
    except RuntimeError:
        return False
    return True


def is_bfloat16_compatible(device):
    return _is_compatible(device, paddle.bfloat16)


def is_float16_compatible(device):
    return _is_compatible(device, paddle.float16)


class NoopContext:
    def __enter__(self):
        pass

    def __exit__(self, *args, **kwargs):
        pass


class NoopGradScaler:
    @staticmethod
    def scale(loss):
        return loss

    @staticmethod
    def unscale_(optimizer):
        pass

    @staticmethod
    def step(optimizer, *args, **kwargs):
        """Not Support auto convert *.step, please judge whether it is Pytorch API and convert by yourself"""
        optimizer.step()

    @staticmethod
    def update():
        pass


def get_grad_scaler_and_autocast_context(precision, device):
    if precision == paddle.float32:
        return NoopGradScaler(), NoopContext()
    if precision == paddle.bfloat16:
        return NoopGradScaler(), paddle.amp.auto_cast(enable=True, dtype="bfloat16")
    elif precision == paddle.float16:
        return paddle.amp.GradScaler(
            incr_every_n_steps=2000, init_loss_scaling=65536.0
        ), paddle.amp.auto_cast(enable=True, dtype="float16")
    raise NotImplementedError(f"Unsupported precision: {precision}")
