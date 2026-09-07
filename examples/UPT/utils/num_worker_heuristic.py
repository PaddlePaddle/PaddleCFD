import logging
import os

from distributed.config import is_slurm_run


def get_fair_cpu_count():
    total_cpu_count = get_total_cpu_count()
    if total_cpu_count == 0:
        return 0
    device_count = _get_device_count()
    if is_slurm_run():
        tasks_per_node = int(os.environ["SLURM_NTASKS_PER_NODE"])
        if "SLURM_CPUS_PER_TASK" in os.environ:
            cpus_per_task = int(os.environ["SLURM_CPUS_PER_TASK"])
        elif "SLURM_CPUS_ON_NODE" in os.environ:
            cpus_on_node = int(os.environ["SLURM_CPUS_ON_NODE"])
            cpus_per_task = cpus_on_node // tasks_per_node
        else:
            raise NotImplementedError
        assert device_count == tasks_per_node
        if total_cpu_count != cpus_per_task:
            logging.warning(
                f"total_cpu_count != cpus_per_task ({total_cpu_count} != {cpus_per_task})"
            )
        return cpus_per_task - 1
    return int(total_cpu_count / device_count)


def _get_device_count():
    nvidia_smi_lines = os.popen("nvidia-smi -L").read().strip().split("\n")
    devices_per_gpu = {}
    devices_counter = 0
    for i, line in enumerate(nvidia_smi_lines):
        if "MIG" in line:
            devices_counter += 1
        if (
            "GPU" in line
            and i == 0
            and len(nvidia_smi_lines) > 1
            and "MIG" in nvidia_smi_lines[i + 1]
        ):
            continue
        if "GPU" in line or i == len(nvidia_smi_lines) - 1:
            if devices_counter == 0:
                devices_counter = 1
            devices_per_gpu[len(devices_per_gpu)] = devices_counter
            devices_counter = 0
    devices_on_node = sum(devices_per_gpu.values())
    if devices_on_node == 0:
        devices_on_node = 1
    return devices_on_node


def get_total_cpu_count():
    if os.name == "nt":
        cpu_count = os.cpu_count()
        assert cpu_count is not None
        if cpu_count <= 16:
            return 0
    else:
        cpu_count = len(os.sched_getaffinity(0))
    return cpu_count
