import logging
import os
import platform

import paddle
import psutil
import yaml

from .config import (get_local_rank, get_nodes, get_rank_from_env,
                     get_world_size_from_env, is_custom_managed_run,
                     is_managed, is_mpi_managed_run)


def run_managed(accelerator, devices, main_single):
    assert is_managed()
    if accelerator == "gpu":
        if (
            is_custom_managed_run()
            or is_mpi_managed_run()
            or len(os.environ["CUDA_VISIBLE_DEVICES"].split(",")) > 1
        ):
            os.environ["CUDA_VISIBLE_DEVICES"] = str(get_local_rank())
        # _check_single_device_visible()
    if devices is None:
        world_size = get_world_size_from_env()
        if world_size == 1:
            _run_managed_singleprocess(accelerator, main_single)
        else:
            _run_managed_multiprocess(accelerator, main_single)
    else:
        world_size, device_ids = _parse_devices(accelerator, devices)
        assert world_size == 1 and len(device_ids) == 1
        _log_device_info(accelerator, device_ids)
        _run_managed_singleprocess(accelerator, main_single)


def _run_managed_singleprocess(accelerator, main_single):
    logging.info(f"running single process slurm training")
    device = _accelerator_to_device(accelerator)
    main_single(device=device)


def _run_managed_multiprocess(accelerator, main_single):
    assert "MASTER_ADDR" in os.environ
    assert "MASTER_PORT" in os.environ
    world_size = get_world_size_from_env()
    rank = get_rank_from_env()
    logging.info(
        f"initializing rank={rank} local_rank={get_local_rank()} nodes={get_nodes()} hostname={platform.uname().node} master_addr={os.environ['MASTER_ADDR']} master_port={os.environ['MASTER_PORT']} (waiting for all {world_size} processes to connect)"
    )
    paddle.distributed.init_parallel_env()
    paddle.distributed.barrier()
    device = _accelerator_to_device(accelerator)
    main_single(device=device)



def run_single_or_multiprocess(
    accelerator, devices, main_single, master_port, mig_devices
):
    logging.info("------------------")
    assert devices is not None
    world_size, device_ids = _parse_devices(accelerator, devices, mig_devices)
    if world_size == 1:
        logging.info(f"running single process training")
        if accelerator == "gpu":
            os.environ["CUDA_VISIBLE_DEVICES"] = device_ids[0]
            # _check_single_device_visible()
        _log_device_info(accelerator, device_ids)
        device = _accelerator_to_device(accelerator)
        main_single(device=device)
    else:
        logging.info(
            f"running multi process training on {world_size} processes (devices={devices} host={platform.uname().node})"
        )
        master_port = _get_free_port(master_port)
        logging.info(f"master port: {master_port}")
        # spawn_args = (accelerator, device_ids, master_port, world_size, main_single)
        args = accelerator, device_ids, master_port, world_size, main_single
        if not isinstance(args, tuple):
            args_tuple = (args,)
        else:
            args_tuple = args
        spawn_args = (main_single,) + args_tuple
        # print('spawn_args的内容为:',spawn_args)
        paddle.distributed.spawn(func=_run_multiprocess, nprocs=world_size, args=spawn_args)
     


def _run_multiprocess(
    rank, accelerator, device_ids, master_port, world_size, main_single
):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(master_port)
    # if accelerator == "gpu":
    #     os.environ["CUDA_VISIBLE_DEVICES"] = device_ids[rank]
        # _check_single_device_visible()
    paddle.distributed.init_parallel_env()
    device = _accelerator_to_device(accelerator)
    main_single(device=device)

# def _run_multiprocess(
#     rank, *args
# ):
#     accelerator, device_ids, master_port, world_size, main_single = args
#     os.environ["MASTER_ADDR"] = "localhost"
#     os.environ["MASTER_PORT"] = str(master_port)
#     if accelerator == "gpu":
#         os.environ["CUDA_VISIBLE_DEVICES"] = device_ids[rank]
#         # _check_single_device_visible()
#     paddle.distributed.init_parallel_env()
#     device = _accelerator_to_device(accelerator)
#     main_single(device=device)



def get_backend(accelerator, device_ids=None):
    if accelerator == "cpu":
        return "gloo"
    if os.name == "nt":
        return "gloo"
    if device_ids is not None:
        for device_id in device_ids:
            try:
                int(device_id)
            except ValueError:
                return "gloo"
    return "nccl"


def _get_free_port(start_port):
    taken_ports = set()
    for connection in psutil.net_connections():
        if connection.laddr.ip == "127.0.0.1":
            taken_ports.add(connection.laddr.port)
        if len(connection.raddr) > 0 and connection.raddr.ip == "127.0.0.1":
            taken_ports.add(connection.raddr.port)
    for port in range(start_port, 65535):
        if port not in taken_ports:
            return port
    raise ValueError(f"all ports starting from {start_port} are taken")


def _parse_devices(accelerator, devices, mig_devices=None):
    try:
        device_ids = [int(devices)]
    except ValueError:
        device_ids = yaml.safe_load(f"[{devices}]")
        msg = f"invalid devices specification '{devices}' (specify multiple devices like this '0,1,2,3')"
        assert all(isinstance(d, int) for d in device_ids), msg
    device_ids = [str(device_id) for device_id in device_ids]
    if accelerator == "gpu" and mig_devices is not None:
        hostname = platform.uname().node
        if hostname in mig_devices:
            for i in range(len(device_ids)):
                device_id = int(device_ids[i])
                if device_id in mig_devices[hostname]:
                    mig_device_id = mig_devices[hostname][device_id]
                    device_ids[i] = mig_device_id
                    logging.info(f"device_id is MIG device with id {mig_device_id}")
    return len(device_ids), device_ids


# def _check_single_device_visible():
#     assert "CUDA_VISIBLE_DEVICES" in os.environ
#     visible_device_count = paddle.cuda.device_count()
#     assert visible_device_count <= 1, os.environ


def _log_device_info(accelerator, device_ids):
    if accelerator == "cpu":
        for i in range(len(device_ids)):
            logging.info(f"device {i}: cpu")
    elif accelerator == "gpu":
        all_devices = (
            os.popen("nvidia-smi --query-gpu=gpu_name --format=csv,noheader")
            .read()
            .strip()
            .split("\n")
        )
        for i, device_id in enumerate(device_ids):
            try:
                device_id = int(device_id)
                logging.info(f"device {i}: {all_devices[device_id]} (id={device_id})")
            except ValueError:
                logging.info(f"using MIG device")
    else:
        raise NotImplementedError


def _accelerator_to_device(accelerator):
    if accelerator == "cpu":
        return "cpu"
    elif accelerator == "gpu":
        return "cuda"
    raise NotImplementedError
