import logging
import random
from contextlib import ContextDecorator

import numpy as np
import paddle


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    paddle.seed(seed)
    # if paddle.cuda.is_available():
    #     paddle.cuda.manual_seed_all(seed)
    logging.info(f"set seed to {seed}")


def get_random_int(generator=None):
    return paddle.randint(low=0, high=999999999, shape=(1,)).item()


def get_random_states():
    return dict(
        torch_rng_state=paddle.get_rng_state(),
        np_rng_state=np.random.get_state(),
        py_rng_state=random.getstate(),
    )


def set_random_states(torch_rng_state, np_rng_state, py_rng_state):
    paddle.set_rng_state(state_list=torch_rng_state)
    np.random.set_state(np_rng_state)
    random.setstate(py_rng_state)


def unset_seed():
    import time

    t = 1000 * time.time()
    seed = int(t) % 2**32
    set_seed(seed)


def with_seed(seed):
    return WithSeedDecorator(seed)


class WithSeedDecorator(ContextDecorator):
    def __init__(self, seed):
        self.seed = seed

    def __enter__(self):
        set_seed(self.seed)

    def __exit__(self, exc_type, exc_val, exc_tb):
        unset_seed()
