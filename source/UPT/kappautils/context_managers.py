from contextlib import contextmanager


MODEL_MODES = {
    "train": True,
    "training": True,
    "eval": False,
    "evaluate": False,
}


@contextmanager
def temp_mode_change(model, mode):
    if mode in MODEL_MODES:
        mode = MODEL_MODES[mode]
    assert isinstance(mode, bool)

    prev_mode = model.training
    model.train(mode=mode)
    yield
    if prev_mode != model.training:
        model.train(mode=prev_mode)

