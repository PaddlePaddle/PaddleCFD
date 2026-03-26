import math


def linear(step, total_steps):
    """ linearly increasing from [0 to 1] """
    return step / max(1, total_steps - 1)


def cosine(step, total_steps):
    """ cosine schedule from [0 to 1] """
    progress = step / max(1, total_steps - 1)
    return 1 - (1 + math.cos(math.pi * progress)) / 2


def inverse_sqrt(step, abs_step):
    """ inverse square root schedule starting from 1 and going towards 0 """
    # https://github.com/facebookresearch/fairseq/blob/main/fairseq/optim/lr_scheduler/inverse_square_root_schedule.py
    decay_factor = max(1, abs_step - step) ** 0.5
    if step == abs_step:
        cur_step = abs_step + 1
    else:
        cur_step = abs_step
    return cur_step ** -0.5 * decay_factor


def polynomial(step, total_steps, power):
    """
    polynomial schedule from [0 to 1]
    https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.PolynomialLR.html
    """
    progress = step / max(1, total_steps - 1)
    return 1 - (1 - progress) ** power
