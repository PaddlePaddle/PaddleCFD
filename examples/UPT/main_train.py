import sys
import multiprocessing
import os
from pathlib import Path
import importlib.util


def _setup_paths_and_aliases():
    cur_dir = Path(__file__).resolve().parent
    paddlecfd_root = cur_dir.parent.parent
    source_upt_dir = paddlecfd_root / "source" / "UPT"
    kappa_modules_dir = source_upt_dir / "KappaModules"

    # Ensure the restructured UPT layout in PaddleCFD is importable.
    for path in [cur_dir, paddlecfd_root, source_upt_dir, kappa_modules_dir]:
        p = str(path)
        if p not in sys.path:
            sys.path.insert(0, p)

    # Keep legacy "from models import ..." imports working after moving models.
    if "models" not in sys.modules:
        models_dir = paddlecfd_root / "ppcfd" / "models" / "UPT"
        spec = importlib.util.spec_from_file_location(
            "models",
            models_dir / "__init__.py",
            submodule_search_locations=[str(models_dir)],
        )
        models_module = importlib.util.module_from_spec(spec)
        sys.modules["models"] = models_module
        spec.loader.exec_module(models_module)


_setup_paths_and_aliases()

import paddle
from paddle_utils import *
# from utils.version_check import check_versions

# check_versions(verbose=False)
import logging

import kappaprofiler as kp
from configs.cli_args import parse_run_cli_args
from configs.static_config import StaticConfig
from distributed.config import (barrier, get_local_rank, get_rank,
                                get_world_size, is_managed)
from distributed.run import run_managed, run_single_or_multiprocess
from train_stage import train_stage
from utils.kappaconfig.util import get_stage_hp
from utils.logging_util import add_global_handlers, log_from_all_ranks
from utils.pytorch_cuda_timing import cuda_end_event, cuda_start_event


def main_single(device):
    cli_args = parse_run_cli_args()
    static_config = StaticConfig(
        uri="static_config.yaml",
        datasets_were_preloaded=cli_args.datasets_were_preloaded,
    )
    add_global_handlers(log_file_uri=None)
    with log_from_all_ranks():
        logging.info(
            f"initialized process rank={get_rank()} local_rank={get_local_rank()} pid={os.getpid()}"
        )
    barrier()
    logging.info(f"initialized {get_world_size()} processes")
    if cli_args.accelerator == "gpu":
        if cli_args.cudnn_benchmark or static_config.default_cudnn_benchmark:
            PaddleFlag.cudnn_benchmark = True
            assert (
                not static_config.default_cudnn_deterministic
            ), "cudnn_benchmark can make things non-deterministic"
        else:
            logging.warning(f"disabled cudnn benchmark")
            if static_config.default_cudnn_deterministic:
                PaddleFlag.cudnn_deterministic = True
                os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
                logging.warning(f"enabled cudnn deterministic")
    if cli_args.accelerator == "gpu":
        if cli_args.cuda_profiling or static_config.default_cuda_profiling:
            kp.setup_async(cuda_start_event, cuda_end_event)
            logging.info(f"initialized profiler to call sync cuda")
    else:
        kp.setup_async_as_sync()
    stage_hp = get_stage_hp(
        cli_args.hp,
        template_path="zztemplates",
        testrun=cli_args.testrun,
        minmodelrun=cli_args.minmodelrun,
        mindatarun=cli_args.mindatarun,
        mindurationrun=cli_args.mindurationrun,
    )
    train_stage(
        stage_hp=stage_hp, static_config=static_config, cli_args=cli_args, device=device
    )


def main():
    cli_args = parse_run_cli_args()
    static_config = StaticConfig(
        uri="static_config.yaml",
        datasets_were_preloaded=cli_args.datasets_were_preloaded,
    )
    add_global_handlers(log_file_uri=None)
    if is_managed():
        run_managed(
            accelerator=cli_args.accelerator,
            devices=cli_args.devices,
            main_single=main_single,
        )
    else:
        run_single_or_multiprocess(
            accelerator=cli_args.accelerator,
            devices=cli_args.devices,
            main_single=main_single,
            master_port=cli_args.master_port or static_config.master_port,
            mig_devices=static_config.mig_config,
        )


if __name__ == "__main__":
    # multiprocessing.set_start_method('spawn', force=True)
    main()
