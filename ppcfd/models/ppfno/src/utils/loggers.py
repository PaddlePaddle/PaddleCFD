import matplotlib as mpl
import paddle

mpl.use("Agg")
import unittest
import warnings

from torchtyping import TensorType

try:
    import wandb
except ImportError:
    warnings.warn("wandb is not installed. wandb logger is not available.")
import time
from pathlib import Path
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from .visualization import fig_to_numpy


class Logger:
    def __init__(self):
        pass

    def log_scalar(self, tag: str, value: float, step: int):
        raise NotImplementedError

    def log_image(self, tag: str, img: paddle.Tensor, step: int):
        raise NotImplementedError

    def log_time(self, tag: str, duration: float, step: int):
        raise NotImplementedError

    def log_figure(self, tag: str, fig: mpl.figure.Figure, step: int):
        fig.set_tight_layout(True)
        fig.patch.set_facecolor("white")
        im = fig_to_numpy(fig)
        self.log_image(tag, im, step)


class WandBLogger(Logger):
    def __init__(
        self,
        project_name: str,
        run_name: str,
        log_dir: str,
        config: Optional[Dict] = None,
        entity: Optional[str] = None,
    ):
        super().__init__()
        wandb.init(project=project_name, name=run_name, dir=log_dir, entity=entity)
        if config is not None:
            wandb.config.update(config)

    def log_scalar(self, tag: str, value: float, step: int):
        wandb.log({tag: value}, step=step)

    def log_image(
        self, tag: str, img: Union[TensorType["H", "W", "C"], np.ndarray], step: int
    ):
        if isinstance(img, paddle.Tensor):
            img = img.numpy()
        if img.dtype != np.uint8:
            img = (img * 255).astype(np.uint8)
        if img.ndim == 3:
            assert tuple(img.shape)[-1] == 3
        wandb.log({tag: [wandb.Image(img)]}, step=step)

    def log_time(self, tag: str, duration: float, step: int):
        wandb.log({tag: duration}, step=step)


class Loggers(Logger):
    """
    A class that wraps multiple loggers.
    """

    def __init__(self, loggers: Union[Logger, list]):
        super().__init__()
        if isinstance(loggers, list):
            self.loggers = loggers
        else:
            self.loggers = [loggers]

    def log_scalar(self, tag: str, value: float, step: int):
        for logger in self.loggers:
            logger.log_scalar(tag, value, step)

    def log_image(self, tag: str, img: paddle.Tensor, step: int):
        for logger in self.loggers:
            logger.log_image(tag, img, step)

    def log_figure(self, tag: str, fig, step: int):
        for logger in self.loggers:
            logger.log_figure(tag, fig, step)

    def log_time(self, tag: str, duration: float, step: int):
        for logger in self.loggers:
            logger.log_time(tag, duration, step)


def init_logger(config: dict, prefix: str = "") -> Logger:
    loggers = []
    datetime = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
    log_dir = Path(config["log_dir"]) / datetime
    log_dir.mkdir(parents=True, exist_ok=True)
    for logger_type in config["logger_types"]:
        if logger_type == "wandb":
            loggers.append(
                WandBLogger(
                    config["project_name"],
                    config["run_name"],
                    entity=config["entity"],
                    log_dir=log_dir,
                    config=config,
                )
            )
        else:
            raise ValueError(f"Unknown logger type: {logger_type}")
    return Loggers(loggers)


class TestLoggers(unittest.TestCase):
    def setUp(self) -> None:
        X = np.arange(-5, 5, 0.25)
        Y = np.arange(-5, 5, 0.25)
        X, Y = np.meshgrid(X, Y)
        R = np.sqrt(X**2 + Y**2)
        Z = np.sin(R)
        self.test_data = X, Y, Z
        return super().setUp()

    def test_wandb(self):
        logger = WandBLogger("test", "test", "test")
        logger.log_scalar("test", 1.0, 0)
        logger.log_image("test", paddle.rand(shape=[64, 64, 3]), 0)
        logger.log_time("test", 1.0, 0)
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        ax.plot_surface(*self.test_data, cmap="viridis")
        logger.log_figure("test", fig, 0)

    def test_loggers(self):
        loggers = Loggers([WandBLogger("test", "test", "test")])
        loggers.log_scalar("test", 1.0, 0)
        loggers.log_image("test", paddle.rand(shape=[64, 64, 3]), 0)
        loggers.log_time("test", 1.0, 0)


if __name__ == "__main__":
    unittest.main()
