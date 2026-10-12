from .base import ScheduleBase


class StepIntervalSchedule(ScheduleBase):
    def __init__(self, start_value, factor, interval, **kwargs):
        super().__init__(**kwargs)
        assert isinstance(interval, (int, float)) and 0. < interval < 1.
        self.start_value = start_value
        self.factor = factor
        self.interval = interval

    def __str__(self):
        return f"{type(self).__name__}(start_value={self.start_value}, factor={self.factor}, interval={self.interval})"

    def _get_value(self, step: int, total_steps: int, abs_step: int = None) -> float:
        progress = step / total_steps
        # round to 10th decimal place to avoid floating point precision errors
        step_idx = int(round(progress / self.interval, 10))
        return self.start_value * self.factor ** step_idx
