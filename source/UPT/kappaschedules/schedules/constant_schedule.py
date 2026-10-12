from .base import ScheduleBase


class ConstantSchedule(ScheduleBase):
    def __init__(self, value, **kwargs):
        super().__init__(**kwargs)
        self.value = value

    def __str__(self):
        return f"{type(self).__name__}(value={self.value})"

    def _get_value(self, step: int, total_steps: int, abs_step: int = None) -> float:
        return self.value
