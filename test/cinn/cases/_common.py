"""Helpers shared by the case definitions.

A case module describes one PaddleCFD model with synthetic data. It must define:

    BATCH_SIZE: int                       samples per step, used for throughput
    build_model() -> paddle.nn.Layer      same arch/hyper-params as the example config
    build_batch() -> (inputs, labels)     flat dicts of paddle.Tensor
    compute_loss(model, inputs, labels)   scalar training loss, as in the example

and may define:

    forward(model, inputs)                inference call, defaults to model(inputs)
    prepare(model)                        one-off setup before to_static
    LEARNING_RATE: float                  Adam lr, defaults to 1e-3
    FULL_GRAPH: bool                      to_static full_graph, defaults to True (AST);
                                          False = SOT, for examples compiled that way
    EXAMPLE: str                          examples/ sub-directory the case mirrors
    NOTE: str                             what was simplified compared with the example

Import ppcfd models through this module (`C.ppcfd_module(...)`): the stock
ppcfd/__init__.py and ppcfd/models/__init__.py eagerly import every model family,
and physicsregression / prose_fd need torch-compat APIs (paddle.compat.nn) that
Paddle 3.2 lacks, so one broken family would take every case down with it.
"""
import importlib
import sys
import types
from pathlib import Path

import paddle
import paddle.nn.functional as F


REPO_ROOT = Path(__file__).resolve().parents[3]


def _stub_package(name, path):
    if name not in sys.modules:
        module = types.ModuleType(name)
        module.__path__ = [str(path)]
        sys.modules[name] = module


def _stub_module(name):
    """Empty module for an optional dependency that is imported but not used."""
    if name not in sys.modules:
        try:
            importlib.import_module(name)
        except ImportError:
            sys.modules[name] = types.ModuleType(name)


_stub_package("ppcfd", REPO_ROOT / "ppcfd")
_stub_package("ppcfd.models", REPO_ROOT / "ppcfd" / "models")


def ppcfd_module(name, optional_deps=()):
    """Import `ppcfd.models.<name>` without running the umbrella __init__ files.

    optional_deps are metric/plotting packages the sub-package imports at top
    level but the model never calls (e.g. xskillscore for ppdiffusion)."""
    for dep in optional_deps:
        _stub_module(dep)
    return importlib.import_module(f"ppcfd.models.{name}")


def example_dir(name):
    return REPO_ROOT / "examples" / name


def uniform(shape, low=0.0, high=1.0, requires_grad=False):
    t = paddle.uniform(shape, dtype="float32", min=low, max=high)
    t.stop_gradient = not requires_grad
    return t


def randn(shape, requires_grad=False):
    t = paddle.randn(shape, dtype="float32")
    t.stop_gradient = not requires_grad
    return t


def randint(shape, high, low=0):
    return paddle.randint(low, high, shape, dtype="int64")


def mse(a, b):
    return F.mse_loss(a, b)


def l1(a, b):
    return F.l1_loss(a, b)
