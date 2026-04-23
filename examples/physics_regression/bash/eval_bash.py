import os
import sys
from pathlib import Path

import paddle

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import copy
import json
import warnings

import numpy as np

import ppcfd.models.physicsregression.symbolicregression as symbolicregression
from exp_utils import initialize_exp
from evaluate import Evaluator
from parsers import get_parser
from ppcfd.models.physicsregression.paddle_utils import device2str
from ppcfd.models.physicsregression.symbolicregression.envs import build_env
from ppcfd.models.physicsregression.symbolicregression.model import build_modules
from slurm import init_distributed_mode
from slurm import init_signal_handler
from trainer import Trainer


def init_eval(params):
    init_distributed_mode(params)
    logger = initialize_exp(params, write_dump_path=False)
    if params.is_slurm_job:
        init_signal_handler()
    paddle.device.set_device("cpu" if params.cpu else device2str(params.device))
    params.eval_only = True
    symbolicregression.utils.CUDA = not params.cpu
    if params.batch_size_eval is None:
        params.batch_size_eval = int(1.5 * params.batch_size)
    env = build_env(params)
    env.rng = np.random.RandomState()
    modules = build_modules(env, params)
    trainer = Trainer(modules, env, params)
    evaluator = Evaluator(trainer)
    return evaluator, logger


if __name__ == "__main__":
    parser = get_parser()
    parser.add_argument("--repeat_trials", type=int, default=0, help="repeat trials")
    parser.add_argument("--filename", type=str, default="", help="")
    parser.add_argument("--oraclename", type=str, default="", help="")
    parser.add_argument(
        "--current_eval_pos", type=int, default=0, help="where eval are currently?"
    )
    params = parser.parse_args()
    params.rescale = False
    params.generate_datapoints_distribution = "positive,single"
    params.max_trials = 1000
    params.sample_expr_num = -1
    params.max_number_bags = -1
    np.random.seed(2024 + params.repeat_trials)
    if params.expr_data != "feynman" and (
        params.expr_test_data_path is None or params.expr_test_data_path == ""
    ):
        raise FileNotFoundError(
            "Please follow README and place the testing dataset into `examples/physics_regression/data`."
        )
    evaluator, logger = init_eval(params)
    evaluator.set_env_copies(["test"])
    scores = evaluator.evaluate_oracle_mcts(
        "test",
        "functions",
        logger=logger,
        verbose=False,
        save_file=os.path.join(os.getcwd(), "eval_result", params.filename),
        datatype="test",
        oracle_name=params.oraclename,
        verbose_res=True,
    )
