from .schedule_base import ScheduleBase


class ProgressSchedule(ScheduleBase):
    def __init__(self, start_value, delta, exclude_first=False, exclude_last=False, **kwargs):
        super().__init__(**kwargs)
        self.start_value = start_value
        self.delta = delta
        self.exclude_first = exclude_first
        self.exclude_last = exclude_last

    def __str__(self):
        return (
            f"{type(self).__name__}("
            f"start={self.start_value}, "
            f"end={self.start_value + self.delta}, "
            f"excl_first={self.exclude_first}, "
            f"excl_last={self.exclude_last}"
            f")"
        )

    def _get_value(self, step: int, total_steps: int, abs_step: int = None) -> float:
        if self.exclude_last:
            total_steps += 1
        if self.exclude_first:
            step += 1
            total_steps += 1
        # get progress of schedule (going from 0 to 1)
        progress = self._get_progress(step, total_steps, abs_step)
        # adjust to "absolute value" (i.e. real learning rate)
        return self.start_value + progress * self.delta

    def _get_progress(self, step: int, total_steps: int, abs_step: int):
        raise NotImplementedError
