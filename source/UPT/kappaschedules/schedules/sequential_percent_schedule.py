from dataclasses import dataclass
from typing import List

from .base import ScheduleBase


@dataclass
class SequentialPercentScheduleConfig:
    schedule: ScheduleBase
    start_percent: float = None
    end_percent: float = None

    def __repr__(self):
        return str(self)

    def __str__(self):
        return f"{self.start_percent * 100:.0f}% - {self.end_percent * 100:.0f}% {self.schedule}"


class SequentialPercentSchedule(ScheduleBase):
    def __init__(self, schedule_configs: List[SequentialPercentScheduleConfig], **kwargs):
        super().__init__(**kwargs)
        assert len(schedule_configs) > 0
        self.schedule_configs = schedule_configs

        # if first schedule has no start_percent -> set to 0 -> this ensures every schedule has a start_percent
        if schedule_configs[0].start_percent is None:
            schedule_configs[0].start_percent = 0.
        # propagate start/end
        for i in range(1, len(schedule_configs) - 1):
            # take start_percent from previous schedule
            if schedule_configs[i].start_percent is None:
                schedule_configs[i].start_percent = schedule_configs[i - 1].end_percent
            # take end_percent from next schedule
            if schedule_configs[i].end_percent is None:
                schedule_configs[i].end_percent = schedule_configs[i + 1].start_percent
        # edge case: last schedule propagate
        if len(schedule_configs) > 1:
            # propagate [-2].end_percent forward
            if schedule_configs[-1].start_percent is None:
                schedule_configs[-1].start_percent = schedule_configs[-2].end_percent
            # propagate [-1].start_percent backward
            if schedule_configs[-2].end_percent is None:
                schedule_configs[-2].end_percent = schedule_configs[-1].start_percent
        # set [-1].end_percent to 1.
        if schedule_configs[-1].end_percent is None:
            schedule_configs[-1].end_percent = 1.

        # check correctness of start/end
        if len(schedule_configs) == 1:
            # edge case: single schedule
            # always: 0. <= start <= 1.
            # if end is not None: start <= end <= 1.
            assert 0. <= schedule_configs[0].start_percent <= 1.
            if schedule_configs[0].end_percent is not None:
                assert schedule_configs[0].start_percent <= schedule_configs[0].end_percent <= 1.
        else:
            # check 0 <= cfg[i].start <= cfg[i].end <= 1.
            # check cfg[i].end <= cfg[i+1].start <= 1.
            for i in range(len(schedule_configs) - 1):
                assert schedule_configs[i].start_percent is not None and schedule_configs[i].end_percent is not None
                assert 0. <= schedule_configs[i].start_percent <= schedule_configs[i].end_percent <= 1.
                assert schedule_configs[i].end_percent <= schedule_configs[i + 1].start_percent <= 1.
            # last schedule is allowed to have no end_percent
            if schedule_configs[-1].end_percent is None:
                assert 0. <= schedule_configs[-1].start_percent <= 1.
            else:
                assert 0. <= schedule_configs[-1].start_percent <= schedule_configs[-1].end_percent <= 1.

    def get_sequential_schedule_config(self, step: int, total_steps: int) -> SequentialPercentScheduleConfig:
        total_steps_m1 = total_steps - 1
        # percent < config[0].start_percent -> None
        # config[-1].end_percent < percent -> config[-1]
        for i in reversed(range(len(self.schedule_configs))):
            if int(self.schedule_configs[i].start_percent * total_steps_m1) <= step:
                return self.schedule_configs[i]
        return None

    def _get_value(self, step: int, total_steps: int, abs_step: int = None) -> float:
        config = self.get_sequential_schedule_config(step, total_steps)
        total_steps_m1 = total_steps - 1
        if config is None:
            # adjust step/total_steps within SequentialSchedule to step/total_steps within schedule
            adj_step = int(total_steps_m1 * self.schedule_configs[0].start_percent)
            if self.schedule_configs[0].end_percent == 1.:
                end_step = total_steps
            else:
                end_step = int(total_steps_m1 * self.schedule_configs[0].end_percent)
            adj_total_steps = end_step - adj_step
            return self.schedule_configs[0].schedule.get_value(0, adj_total_steps, 0)
        # adjust step/total_steps within SequentialSchedule to step/total_steps within schedule
        start_step = int(total_steps_m1 * config.start_percent)
        adj_step = step - start_step
        end_step = total_steps if config.end_percent == 1. else int(total_steps_m1 * config.end_percent)
        adj_total_steps = end_step - start_step
        if adj_total_steps == 0:
            return config.schedule.get_value(0, 1, 0)
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
