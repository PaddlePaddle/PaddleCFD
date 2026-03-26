from pathlib import Path

import numpy as np

from kappadata.utils.logging import get_log_or_pass_function


def setup(log_fn, dst, seed):
    log = get_log_or_pass_function(log_fn)
    dst = Path(dst).expanduser()
    assert not dst.exists(), f"{dst.as_posix()} already exists"
    dst.mkdir(parents=True)
    rng = np.random.default_rng(seed=seed)
    return log, dst, rng
