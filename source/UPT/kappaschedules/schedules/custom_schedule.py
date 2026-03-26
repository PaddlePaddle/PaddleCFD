from .base import ScheduleBase


class CustomSchedule(ScheduleBase):
    def __init__(self, values, **kwargs):
        super().__init__(**kwargs)
        self.values = values

    def __str__(self):
        return f"{type(self).__name__}"

    def _get_value(self, step: int, total_steps: int, abs_step: int = None) -> float:
        return self.values[step]
