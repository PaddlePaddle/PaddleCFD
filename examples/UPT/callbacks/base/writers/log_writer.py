import logging
from collections import defaultdict
from contextlib import contextmanager

import paddle
import wandb
import yaml
from distributed.config import is_rank0
from providers.path_provider import PathProvider
from utils.update_counter import UpdateCounter


class LogWriter:
    def __init__(self, path_provider: PathProvider, update_counter: UpdateCounter):
        self.logger = logging.getLogger(type(self).__name__)
        self.path_provider = path_provider
        self.update_counter = update_counter
        self.log_entries = []
        self.log_cache = None
        self.is_wandb = wandb.run is not None
        self._postfix = None

    def finish(self):
        if len(self.log_entries) == 0 or not is_rank0():
            return
        entries_uri = self.path_provider.primitive_entries_uri
        self.logger.info(
            f"writing {len(self.log_entries)} log entries to {entries_uri}"
        )
        result = defaultdict(dict)
        for entry in self.log_entries:
            update = entry["update"]
            for key, value in entry.items():
                if key == "update":
                    continue
                result[key][update] = value
        with open(entries_uri, "w") as f:
            yaml.safe_dump(dict(result), f)

    def _log(self, key, value, logger=None, format_str=None):
        if self.log_cache is None:
            self.log_cache = dict(
                epoch=self.update_counter.epoch,
                update=self.update_counter.update,
                sample=self.update_counter.sample,
            )
        if self._postfix is not None:
            key = f"{key}/{self._postfix}"
        self.log_cache[key] = value
        if logger is not None:
            if format_str is not None:
                value = f"{value:{format_str}}"
            logger.info(f"{key}: {value}")

    def flush(self):
        if self.log_cache is None:
            return
        if self.is_wandb:
            wandb.log(self.log_cache)
        if len(self.log_entries) > 0:
            assert self.log_cache["update"] > self.log_entries[-1]["update"]
        self.log_entries.append(
            {
                k: v
                for k, v in self.log_cache.items()
                if not isinstance(v, wandb.Histogram)
            }
        )
        self.log_cache = None

    def add_scalar(self, key, value, logger=None, format_str=None):
        if paddle.is_tensor(value):
            value = value.item()
        self._log(key, value, logger=logger, format_str=format_str)

    def add_histogram(self, key, data):
        if self.is_wandb:
            self._log(key, wandb.Histogram(data))

    def add_previous_entry(self, entry):
        if self.is_wandb:
            wandb.log(entry)

    @contextmanager
    def with_postfix(self, postfix):
        prev_postfix = self._postfix
        if self._postfix is not None:
            self._postfix = f"{self._postfix}/{postfix}"
        else:
            self._postfix = postfix
        yield
        self._postfix = prev_postfix
