import getpass
import json
import logging
import os
import random
import re
import subprocess
import sys

import numpy as np
import paddle
from omegaconf import OmegaConf

from ppcfd.models.prose_fd.runtime import get_amp_device_type
from ppcfd.models.prose_fd.runtime import get_runtime_device
from ppcfd.models.prose_fd.runtime import max_memory_allocated_mb
from ppcfd.models.prose_fd.runtime import set_runtime_device
from ppcfd.models.prose_fd.runtime import sync_tensor
from ppcfd.models.prose_fd.runtime import to_cuda

try:
    from examples.prose_fd.utils.logger import create_logger
except ImportError:
    from .logger import create_logger

DUMP_PATH = "checkpoint"


def set_seed(seed_value):
    """Set seed for reproducibility."""
    random.seed(seed_value)
    np.random.seed(seed_value)
    paddle.seed(seed_value)


def load_json(filename):
    with open(filename, "r") as f:
        config = json.load(f)
    return config


def zip_dic(lst):
    dico = {}
    for d in lst:
        for k in d:
            if k not in dico:
                dico[k] = []
            dico[k].append(d[k])
    for k in dico:
        if isinstance(dico[k][0], dict):
            dico[k] = zip_dic(dico[k])
    return dico


def initialize_exp(params, write_dump_path=True):
    """
    Initialize the experience:
    - dump parameters
    - create a logger
    """
    if write_dump_path:
        get_dump_path(params)
        if not os.path.exists(params.dump_path):
            os.makedirs(params.dump_path)
    OmegaConf.save(params, os.path.join(params.dump_path, "configs.yaml"))
    command = ["python", sys.argv[0]]
    for x in sys.argv[1:]:
        if x.startswith("--"):
            assert '"' not in x and "'" not in x
            command.append(x)
        else:
            assert "'" not in x
            if re.match("^[a-zA-Z0-9_]+$", x):
                command.append("%s" % x)
            else:
                command.append("'%s'" % x)
    command = " ".join(command)
    params.command = command + ' --exp_id "%s"' % params.exp_id
    assert len(params.exp_name.strip()) > 0
    if params.base_seed < 0:
        params.base_seed = np.random.randint(0, 1000000000)
    if params.test_seed < 0:
        params.test_seed = np.random.randint(0, 1000000000)
    logger = create_logger(
        os.path.join(params.dump_path, "train.log"),
        rank=getattr(params, "global_rank", 0),
    )
    logger.info("============ Initialized logger ============")
    logger.info("The experiment will be stored in %s\n" % params.dump_path)
    logger.info("Running command: %s" % command)
    logger.info("")
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    logging.getLogger("PIL").setLevel(logging.WARNING)
    return logger


def get_dump_path(params):
    """
    Create a directory to store the experiment.
    """
    if not params.dump_path:
        params.dump_path = DUMP_PATH
    sweep_path = os.path.join(params.dump_path, params.exp_name)
    if not os.path.exists(sweep_path):
        subprocess.Popen("mkdir -p %s" % sweep_path, shell=True).wait()
    if not params.exp_id:
        chars = "abcdefghijklmnopqrstuvwxyz0123456789"
        while True:
            exp_id = "".join(random.choice(chars) for _ in range(10))
            if not os.path.isdir(os.path.join(sweep_path, exp_id)):
                break
        params.exp_id = exp_id
    params.dump_path = os.path.join(sweep_path, params.exp_id)
    if not os.path.isdir(params.dump_path):
        subprocess.Popen("mkdir -p %s" % params.dump_path, shell=True).wait()
