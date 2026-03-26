import inspect
import math
from copy import deepcopy

import kappaschedules.schedules
from .schedules import (
    SequentialPercentSchedule,
    SequentialPercentScheduleConfig,
    SequentialStepSchedule,
    SequentialStepScheduleConfig,
    ConstantSchedule,
    CustomSchedule,
)
from .schedules.base import ScheduleBase


def object_to_schedule(obj, batch_size=None, updates_per_epoch=None, **kwargs) -> ScheduleBase:
    if obj is None:
        return None
    if isinstance(obj, (float, int)):
        return ConstantSchedule(value=obj)
    if isinstance(obj, list) and all(isinstance(obj[i], (float, int)) for i in range(len(obj))):
        return CustomSchedule(obj)
    if not isinstance(obj, (list, dict)):
        assert isinstance(obj, ScheduleBase)
        return obj
    obj = deepcopy(obj)

    # implicit sequential schedule
    if isinstance(obj, list):
        # check consistency between step/percent schedule
        step_counts = 0
        percent_counts = 0
        for schedule_config in obj:
            assert isinstance(schedule_config, dict)
            if "start_step" in schedule_config:
                step_counts += 1
                _check_mutually_exclusive_keys(
                    schedule_config=schedule_config,
                    forbidden_keys=["start_epoch", "start_update", "start_sample", "start_percent"]
                )
            elif "end_step" in schedule_config:
                step_counts += 1
                _check_mutually_exclusive_keys(
                    schedule_config=schedule_config,
                    forbidden_keys=["end_epoch", "end_update", "end_sample", "end_percent"]
                )
            if "start_epoch" in schedule_config:
                step_counts += 1
                _check_mutually_exclusive_keys(
                    schedule_config=schedule_config,
                    forbidden_keys=["start_step", "start_update", "start_sample", "start_percent"]
                )
                assert updates_per_epoch is not None
                schedule_config["start_step"] = schedule_config.pop("start_epoch") * updates_per_epoch
            elif "end_epoch" in schedule_config:
                step_counts += 1
                _check_mutually_exclusive_keys(
                    schedule_config=schedule_config,
                    forbidden_keys=["end_step", "end_update", "end_sample", "end_percent"]
                )
                assert updates_per_epoch is not None
                schedule_config["end_step"] = schedule_config.pop("end_epoch") * updates_per_epoch
            if "start_update" in schedule_config:
                step_counts += 1
                _check_mutually_exclusive_keys(
                    schedule_config=schedule_config,
                    forbidden_keys=["start_step", "start_epoch", "start_sample", "start_percent"]
                )
                schedule_config["start_step"] = schedule_config.pop("start_update")
            elif "end_update" in schedule_config:
                step_counts += 1
                _check_mutually_exclusive_keys(
                    schedule_config=schedule_config,
                    forbidden_keys=["end_step", "end_epoch", "end_sample", "end_percent"]
                )
                schedule_config["end_step"] = schedule_config.pop("end_update")
            if "start_epoch" in schedule_config:
                step_counts += 1
                _check_mutually_exclusive_keys(
                    schedule_config=schedule_config,
                    forbidden_keys=["start_step", "start_update", "start_sample", "start_percent"]
                )
                assert batch_size is not None
                schedule_config["start_step"] = math.ceil(schedule_config.pop("start_sample") / batch_size)
            elif "end_sample" in schedule_config:
                step_counts += 1
                _check_mutually_exclusive_keys(
                    schedule_config=schedule_config,
                    forbidden_keys=["end_step", "end_epoch", "end_update", "end_percent"]
                )
                assert batch_size is not None
                schedule_config["end_step"] = math.ceil(schedule_config.pop("end_sample") / batch_size)
            elif "start_percent" in schedule_config:
                percent_counts += 1
                _check_mutually_exclusive_keys(
                    schedule_config=schedule_config,
                    forbidden_keys=["start_step", "start_epoch", "start_update", "start_sample"]
                )
            elif "end_percent" in schedule_config:
                percent_counts += 1
                _check_mutually_exclusive_keys(
                    schedule_config=schedule_config,
                    forbidden_keys=["end_step", "end_epoch", "end_update", "end_sample"]
                )
        # if no start/end points are specified -> use step version
        if (step_counts == 0 and percent_counts == 0) or step_counts > 0:
            assert percent_counts == 0
            config_ctor = SequentialStepScheduleConfig
            ctor = SequentialStepSchedule
        elif percent_counts > 0:
            config_ctor = SequentialPercentScheduleConfig
            ctor = SequentialPercentSchedule
        else:
            raise NotImplementedError

        # sequential schedule
        schedule_configs = []
        for schedule_config in obj:
            assert "schedule" in schedule_config
            schedule = object_to_schedule(schedule_config["schedule"], **kwargs)
            cfg_kwargs = {k: v for k, v in schedule_config.items() if k != "schedule"}
            schedule_configs.append(config_ctor(schedule=schedule, **cfg_kwargs))
        return ctor(schedule_configs)

    # single schedules
    assert "kind" in obj and isinstance(obj["kind"], str)
    kind = obj.pop("kind")

    # replace kwargs with "_epochs" in them with steps based on updates_per_epoch
    for key in list(obj.keys()):
        if "_epochs" in key:
            new_key = key.replace("_epochs", "_steps")
            assert updates_per_epoch is not None, f"updates_per_epoch is required to convert {key} to {new_key}"
            value = obj.pop(key)
            new_value = value * updates_per_epoch
            obj[new_key] = new_value

    # replace kwargs with "_updates" in them with steps
    for key in list(obj.keys()):
        if "_updates" in key:
            new_key = key.replace("_updates", "_steps")
            obj[new_key] = obj.pop(key)

    # replace kwargs with "_samples" in them with steps based on batch_size
    for key in list(obj.keys()):
        if "_samples" in key:
            new_key = key.replace("_samples", "_steps")
            assert batch_size is not None, f"batch_size is required to convert {key} to {new_key}"
            value = obj.pop(key)
            new_value = math.ceil(value / batch_size)
            obj[new_key] = new_value

    # get all names and ctors of schedules
    pascal_ctor_list = inspect.getmembers(kappaschedules.schedules, inspect.isclass)
    pascal_to_ctor = {name: ctor for name, ctor in pascal_ctor_list}
    # allow snake_case (type name is in PascalCase)
    if kind[0].islower:
        kind = kind.replace("_", "")
        snake_to_pascal = {name.lower(): name for name in pascal_to_ctor.keys()}
        assert kind in snake_to_pascal.keys(), f"invalid kind '{kind}' (possibilities: {list(snake_to_pascal.keys())})"
        kind = snake_to_pascal[kind]
    ctor = pascal_to_ctor[kind]

    # create SequentialScheduleConfig objects
    if ctor == SequentialPercentSchedule:
        obj["schedule_configs"] = _obj_to_schedule_configs(obj["schedule_configs"], SequentialPercentScheduleConfig)
    elif ctor == SequentialStepSchedule:
        obj["schedule_configs"] = _obj_to_schedule_configs(obj["schedule_configs"], SequentialStepScheduleConfig)

    # remove min_value/max_value if schedule doesn't need it (e.g. ConstantSchedule)
    if "max_value" in kwargs and "max_value" not in _get_full_signature(ctor):
        kwargs = {k: v for k, v in kwargs.items() if k != "max_value"}

    return ctor(**obj, **kwargs)


def _get_full_signature(cls):
    signature = set(inspect.signature(cls.__init__).parameters.keys())
    if cls.__base__ != object:
        base_signature = _get_full_signature(cls.__base__)
        base_signature.update(signature)
        signature = base_signature
    return signature


def _check_mutually_exclusive_keys(schedule_config, forbidden_keys):
    for key in schedule_config.keys():
        assert key not in forbidden_keys, f"{key} is mutually exclusive to {forbidden_keys}"


def _obj_to_schedule_configs(obj, config_ctor):
    schedule_configs = []
    for schedule_config in obj:
        schedule = object_to_schedule(schedule_config.pop("schedule"))
        schedule_configs.append(config_ctor(schedule=schedule, **schedule_config))
    return schedule_configs
