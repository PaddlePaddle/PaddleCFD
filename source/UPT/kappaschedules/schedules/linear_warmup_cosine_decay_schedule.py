from .base import ScheduleBase
from .cosine_decreasing_schedule import CosineDecreasingSchedule
from .linear_increasing_schedule import LinearIncreasingSchedule
from .sequential_percent_schedule import SequentialPercentSchedule, SequentialPercentScheduleConfig
from .sequential_step_schedule import SequentialStepSchedule, SequentialStepScheduleConfig


class LinearWarmupCosineDecaySchedule(ScheduleBase):
    def __init__(
            self,
            warmup_steps=None,
            warmup_percent=None,
            start_value=0.,
            max_value=1.0,
            end_value=1e-6,
            **kwargs,
    ):
        super().__init__(**kwargs)
        assert (warmup_steps is None) ^ (warmup_percent is None), f"define one of warmup_steps or warmup_percent"
        self.warmup_steps = warmup_steps
        self.warmup_percent = warmup_percent
        if warmup_steps is not None:
            self.schedule = SequentialStepSchedule(
                schedule_configs=[
                    SequentialStepScheduleConfig(
                        schedule=LinearIncreasingSchedule(
                            exclude_first=start_value == 0,
                            exclude_last=True,
                            start_value=start_value,
                            max_value=max_value,
                        ),
                        end_step=warmup_steps,
                    ),
                    SequentialStepScheduleConfig(
                        schedule=CosineDecreasingSchedule(
                            exclude_first=False,
                            exclude_last=False,
                            max_value=max_value,
                            end_value=end_value,
                        ),
                    ),
                ],
            )
        if warmup_percent is not None:
            self.schedule = SequentialPercentSchedule(
                schedule_configs=[
                    SequentialPercentScheduleConfig(
                        schedule=LinearIncreasingSchedule(
                            exclude_first=start_value == 0,
                            exclude_last=True,
                            start_value=start_value,
                            max_value=max_value,
                        ),
                        end_percent=warmup_percent,
                    ),
                    SequentialPercentScheduleConfig(
                        schedule=CosineDecreasingSchedule(
                            exclude_first=False,
                            exclude_last=False,
                            max_value=max_value,
                            end_value=end_value,
                        ),
                    ),
                ],
            )

    def _get_value(self, step: int, total_steps: int, abs_step: int = None) -> float:
        return self.schedule.get_value(step=step, total_steps=total_steps, abs_step=abs_step)

    def __str__(self):
        if self.warmup_percent is not None:
            return f"{type(self).__name__}(warmup_percent={self.warmup_percent})"
        if self.warmup_steps is not None:
            return f"{type(self).__name__}(warmup_steps={self.warmup_steps})"
        raise RuntimeError
