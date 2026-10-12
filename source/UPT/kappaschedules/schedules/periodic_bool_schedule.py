from .base import ScheduleBase


class PeriodicBoolSchedule(ScheduleBase):
    def __init__(
            self,
            initial_state,
            off_value=0.,
            on_value=1.,
            off_duration=1,
            on_duration=1,
            invert=False,
            **kwargs,
    ):
        super().__init__(**kwargs)
        self.initial_state = initial_state
        self.off_duration = off_duration
        self.on_duration = on_duration
        self.period_duration = off_duration + on_duration
        self.invert = invert
        self.off_value = off_value
        self.on_value = on_value

    def __str__(self):
        return (
            f"{type(self).__name__}"
            f"("
            f"intial_state={self.intial_state},"
            f"off_duration={self.off_duration},"
            f"on_duration={self.on_duration},"
            f"invert={self.invert}"
            f"off_value={self.off_value},"
            f"on_value={self.on_value},"
            f")"
        )

    def _get_value(self, step: int, total_steps: int, abs_step: int = None) -> float:
        bool_value = self._get_bool_value(step)
        if self.invert:
            if bool_value:
                return self.off_value
            else:
                return self.on_value
        else:
            if bool_value:
                return self.on_value
            else:
                return self.off_value

    def _get_bool_value(self, step: int) -> float:
        step_in_period = step % self.period_duration
        if self.initial_state:
            if step_in_period < self.on_duration:
                return True
            else:
                return False
        else:
            if step_in_period < self.off_duration:
                return False
            else:
                return True
