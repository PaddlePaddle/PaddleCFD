import logging
import os
from pathlib import Path
from typing import Optional

import kappaconfig as kc
from distributed.config import (get_local_rank, get_world_size, is_distributed,
                                is_managed)

from .wandb_config import WandbConfig


class StaticConfig:
    def __init__(self, uri: str, datasets_were_preloaded: bool = False):
        self._uri = Path(uri).expanduser()
        self._config = kc.DefaultResolver(template_path=".").resolve(
            kc.from_file_uri(self._uri)
        )
        self.datasets_were_preloaded = datasets_were_preloaded

    def __check_bool(self, key):
        value = self._config[key]
        assert isinstance(value, bool), f"{key} {value} is not a bool"
        return value

    @property
    def account_name(self) -> str:
        return self._config["account_name"]

    @property
    def output_path(self) -> Path:
        assert (
            "output_path" in self._config
        ), f"output_path is not in static_config.yaml"
        path = Path(self._config["output_path"]).expanduser()
        assert path.exists(), f"output_path '{path}' doesn't exist"
        return path

    @property
    def model_path(self) -> Optional[Path]:
        if "model_path" not in self._config:
            return None
        path = Path(self._config["model_path"]).expanduser()
        assert path.exists(), f"model_path '{path}' doesn't exist"
        return path

    def get_global_dataset_paths(self) -> dict:
        return self._config["global_dataset_paths"]

    def get_local_dataset_path(self) -> Path:
        if "local_dataset_path" not in self._config:
            return None
        path = Path(self._config["local_dataset_path"]).expanduser()
        path.mkdir(exist_ok=True)
        if os.name == "posix" and "local_dataset_path_group" in self._config:
            group = self._config["local_dataset_path_group"]
            os.system(f"chgrp -R {group} {path}")
            os.system(f"chmod g+rwxs {path}")
        if not self.datasets_were_preloaded and is_managed() and get_world_size() == 1:
            path = path / f"localrank{get_local_rank()}"
            path.mkdir(exist_ok=True)
        return path

    def get_data_source_modes(self) -> dict:
        if "data_source_modes" not in self._config:
            return {}
        data_source_modes = self._config["data_source_modes"]
        assert all(
            data_source_mode in ["global", "local"]
            for data_source_mode in data_source_modes.values()
        )
        return data_source_modes

    @property
    def temp_path(self):
        local_dataset_path = self.get_local_dataset_path()
        if local_dataset_path is None:
            return Path("temp")
        return local_dataset_path / f"REMOVE_ME"

    @property
    def mig_config(self):
        if "mig" not in self._config:
            return {}
        mig = self._config["mig"]
        assert isinstance(mig, dict), f"mig {mig} is not dict"
        for hostname, device_to_migid in mig.items():
            assert isinstance(
                hostname, str
            ), f"hostnames should be strings (got {hostname})"
            assert isinstance(
                device_to_migid, dict
            ), f"devices_to_migid should be dict (got {device_to_migid})"
            for device_idx, mig_id in device_to_migid.items():
                assert isinstance(
                    device_idx, int
                ), f"devices_to_migid keys should be int (got {device_idx})"
                assert isinstance(
                    mig_id, str
                ), f"devices_to_migid values should be str (got {mig_id})"
        return mig

    @property
    def default_wandb_mode(self) -> str:
        mode = self._config["default_wandb_mode"]
        assert (
            mode in WandbConfig.MODES
        ), f"default_wandb_mode '{mode}' not in {WandbConfig.MODES}"
        return mode

    @property
    def default_cudnn_benchmark(self) -> bool:
        return self.__check_bool("default_cudnn_benchmark")

    @property
    def default_cudnn_deterministic(self) -> bool:
        return self.__check_bool("default_cudnn_deterministic")

    @property
    def default_cuda_profiling(self) -> bool:
        return self.__check_bool("default_cuda_profiling")

    @property
    def default_sync_batchnorm(self) -> bool:
        return self.__check_bool("default_sync_batchnorm")

    @property
    def master_port(self) -> int:
        master_port = self._config["master_port"]
        assert isinstance(master_port, int), f"master_port {master_port} is not an int"
        return master_port

    def log(self, verbose=False):
        logging.info("------------------")
        logging.info("STATIC CONFIG")
        logging.info(f"account_name: {self.account_name}")
        logging.info(f"output_path: {self.output_path}")
        if verbose:
            logging.info(f"global_dataset_paths:")
            for key, dataset_path in self._config["global_dataset_paths"].items():
                logging.info(f"  {key}: {Path(dataset_path).expanduser()}")
        if "local_dataset_path" in self._config:
            logging.info(f"local_dataset_path: {self._config['local_dataset_path']}")
            if os.name == "posix":
                for line in (
                    os.popen(f"df -h {self._config['local_dataset_path']}")
                    .read()
                    .strip()
                    .split("\n")
                ):
                    logging.info(line)
        if "data_source_modes" in self._config:
            logging.info(f"data_source_modes:")
            for key, source_mode in self._config["data_source_modes"].items():
                logging.info(f"  {key}: {source_mode}")
        if verbose:
            logging.info(f"default_wandb_mode: {self.default_wandb_mode}")
            logging.info(f"default_cudnn_benchmark: {self.default_cudnn_benchmark}")
            logging.info(
                f"default_cudnn_deterministic: {self.default_cudnn_deterministic}"
            )
            logging.info(f"default_cuda_profiling: {self.default_cuda_profiling}")
            if is_distributed():
                logging.info(f"master_port: {self.master_port}")
                logging.info(f"default_sync_batchnorm: {self.default_sync_batchnorm}")
