import os
import platform
import shlex
import sys
import uuid
from argparse import ArgumentParser
from datetime import datetime
from pathlib import Path

import kappaconfig as kc
import yaml


def get_parser():
    parser = ArgumentParser()
    gpus_group = parser.add_mutually_exclusive_group()
    gpus_group.add_argument("--nodes", type=int)
    gpus_group.add_argument("--gpus", type=int)
    parser.add_argument("--time", type=str, required=True)
    parser.add_argument("--account", type=str)
    parser.add_argument("--qos", type=str)
    parser.add_argument(
        "--script", type=str, choices=["train", "run_folder"], default="train"
    )
    parser.add_argument("--preload", type=str)
    parser.add_argument("--resume_stage_id", type=str)
    parser.add_argument("--resume_checkpoint", type=str)
    return parser


def main(
    nodes, gpus, time, account, qos, script, preload, resume_stage_id, resume_checkpoint
):
    if nodes is None and gpus is None:
        print(f"no --nodes and no --gpus defined -> use 1 node")
        nodes = 1
    if nodes is not None:
        with open("template_sbatch_nodes.sh") as f:
            template = f.read()
    elif gpus is not None:
        with open("template_sbatch_gpus.sh") as f:
            template = f.read()
    else:
        raise NotImplementedError
    config = kc.DefaultResolver().resolve(kc.from_file_uri("sbatch_config.yaml"))
    chdir = Path(config["chdir"]).expanduser()
    assert chdir.exists(), f"chdir {chdir} doesn't exist"
    account = account or config["default_account"]
    qos = qos or config.get("default_qos")
    parser = get_parser()
    args_to_filter = []
    for action in parser._actions:
        if action.dest == "help":
            continue
        assert len(action.option_strings) == 1
        assert action.option_strings[0].startswith("--")
        args_to_filter.append(action.option_strings[0])
    train_args = []
    i = 0
    while i < len(sys.argv[1:]):
        arg = sys.argv[1 + i]
        if arg.startswith("--") and arg in args_to_filter:
            i += 2
        else:
            train_args.append(arg)
            i += 1
    cli_args_str = " ".join(map(shlex.quote, train_args))
    if preload is not None:
        assert "{preload}" in template
        config["preload"] = "true"
        config["preload_yaml"] = preload
        cli_args_str += " --datasets_were_preloaded"
    else:
        config["preload"] = "false"
        config["preload_yaml"] = "nothing"
    if script == "run_folder":
        cli_args_str += " --devices 0"
    if resume_stage_id is not None:
        assert script == "train"
        cli_args_str += f" --resume_stage_id {resume_stage_id}"
    if resume_checkpoint is not None:
        assert script == "train"
        cli_args_str += f" --resume_checkpoint {resume_checkpoint}"
    patched_template = template.format(
        time=time,
        nodes=nodes,
        gpus=gpus,
        account=account,
        qos=qos,
        script=script,
        cli_args=cli_args_str,
        **config,
    )
    print(patched_template)
    out_path = Path("submit")
    out_path.mkdir(exist_ok=True)
    fname = f"{datetime.now():%m.%d-%H.%M.%S}-{uuid.uuid4()}.sh"
    with open(out_path / fname, "w") as f:
        f.write(patched_template)
    if os.name != "nt":
        os.system(f"sbatch submit/{fname}")


if __name__ == "__main__":
    main(**vars(get_parser().parse_known_args()[0]))
