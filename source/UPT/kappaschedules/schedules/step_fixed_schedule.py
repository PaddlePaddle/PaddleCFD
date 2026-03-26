from .base import ScheduleBase


class StepFixedSchedule(ScheduleBase):
    def __init__(self, start_value: float, factor: float, steps: list, **kwargs):
        super().__init__(**kwargs)
        assert isinstance(steps, list) and len(steps) > 0
        self.steps = sorted(steps)
        assert all(0. < step < 1 for step in self.steps)
        self.start_value = start_value
        self.factor = factor

    def __str__(self):
        return f"{type(self).__name__}(start_value={self.start_value}, factor={self.factor}, steps={self.steps})"

    def _get_value(self, step: int, total_steps: int, abs_step: int = None) -> float:
        progress = step / total_steps
        # search for step
        for i in range(len(self.steps)):
            if self.steps[i] > progress:
                step_idx = i
                break
        else:
            step_idx = len(self.steps)
        # round to 10th decimal place to avoid floating point precision errors
        return self.start_value * self.factor ** step_idx
