from .base import DecreasingProgressSchedule
from .functional.schedules import cosine


class StepDecreasingSchedule(DecreasingProgressSchedule):
    def __init__(self, factor, interval, **kwargs):
        super().__init__(**kwargs)
        assert isinstance(interval, (int, float)) and 0. < interval < 1.
        self.factor = factor
        self.interval = interval

    def _get_progress(self, step: int, total_steps: int, abs_step: int):
        progress = step / total_steps
        # round to 10th decimal place to avoid floating point precision errors
        step_idx = int(round(progress / self.interval, 10))
        return 1 - self.factor ** step_idx
