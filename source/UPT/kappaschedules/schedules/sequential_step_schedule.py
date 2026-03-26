from dataclasses import dataclass
from typing import List

from .base import ScheduleBase


@dataclass
class SequentialStepScheduleConfig:
    schedule: ScheduleBase
    start_step: int = None
    end_step: int = None

    def __repr__(self):
        return str(self)

    def __str__(self):
        return f"{self.start_step} - {self.end_step} {self.schedule}"


class SequentialStepSchedule(ScheduleBase):
    def __init__(self, schedule_configs: List[SequentialStepScheduleConfig], **kwargs):
        super().__init__(**kwargs)
        assert len(schedule_configs) > 0
        self.schedule_configs = schedule_configs

        # if first schedule has no start_step -> set to 0 -> this ensures every schedule has a start_step
        if schedule_configs[0].start_step is None:
            schedule_configs[0].start_step = 0
        # propagate start/end
        for i in range(1, len(schedule_configs) - 1):
            # take start_step from previous schedule
            if schedule_configs[i].start_step is None:
                schedule_configs[i].start_step = schedule_configs[i - 1].end_step
            # take end_step from next schedule
            if schedule_configs[i].end_step is None:
                schedule_configs[i].end_step = schedule_configs[i + 1].start_step
        # edge case: last schedule propagate
        if len(schedule_configs) > 1:
            # propagate [-2].end_step forward
            if schedule_configs[-1].start_step is None:
                schedule_configs[-1].start_step = schedule_configs[-2].end_step
            # propagate [-1].start_step backward
            if schedule_configs[-2].end_step is None:
                schedule_configs[-2].end_step = schedule_configs[-1].start_step

        # check correctness of start/end
        if len(schedule_configs) == 1:
            # edge case: single schedule
            # always: 0 <= start
            # if end is not None: start <= end
            assert 0 <= schedule_configs[0].start_step
            if schedule_configs[0].end_step is not None:
                assert schedule_configs[0].start_step <= schedule_configs[0].end_step
        else:
            # check 0 <= cfg[i].start <= cfg[i].end
            # check cfg[i].end <= cfg[i+1].start
            for i in range(len(schedule_configs) - 1):
                assert schedule_configs[i].start_step is not None and schedule_configs[i].end_step is not None
                assert 0 <= schedule_configs[i].start_step <= schedule_configs[i].end_step
                assert schedule_configs[i].end_step <= schedule_configs[i + 1].start_step
            # last schedule is allowed to have no end_step
            if schedule_configs[-1].end_step is None:
                assert 0 <= schedule_configs[-1].start_step
            else:
                assert 0 <= schedule_configs[-1].start_step <= schedule_configs[-1].end_step

    def get_sequential_schedule_config(self, step: int) -> SequentialStepScheduleConfig:
        # step < config[0].start_step -> None
        # config[-1].end_step < step -> config[-1]
        for i in reversed(range(len(self.schedule_configs))):
            if self.schedule_configs[i].start_step <= step:
                return self.schedule_configs[i]
        return None

    def _get_value(self, step: int, total_steps: int, abs_step: int = None) -> float:
        config = self.get_sequential_schedule_config(step)
        if config is None:
            # adjust step/total_steps within SequentialSchedule to step/total_steps within schedule
            adj_step = self.schedule_configs[0].start_step
            end_step = self.schedule_configs[0].end_step or total_steps
            adj_total_steps = end_step - adj_step
            return self.schedule_configs[0].schedule.get_value(0, adj_total_steps, 0)
        # adjust step/total_steps within SequentialSchedule to step/total_steps within schedule
        adj_step = step - config.start_step
        end_step = config.end_step or total_steps
        adj_total_steps = end_step - config.start_step
        if adj_step >= adj_total_steps:
            # return last value of previous schedule
            return config.schedule.get_value(adj_total_steps - 1, adj_total_steps, end_step - 1)
        return config.schedule.get_value(adj_step, adj_total_steps, step)

    def __str__(self):
        return "\n".join([
            type(self).__name__,
            "\n".join(map(lambda item: f"  ({item[0]}): {item[1]}", enumerate(self.schedule_configs))),
            ")",
        ])
