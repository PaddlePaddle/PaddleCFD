from .base import DecreasingProgressSchedule

from .functional.schedules import polynomial


class PolynomialDecreasingSchedule(DecreasingProgressSchedule):
    def __init__(self, *args, power=1., **kwargs):
        super().__init__(*args, **kwargs)
        self.power = power

    def _get_progress(self, step: int, total_steps: int, abs_step: int):
        return polynomial(step, total_steps, power=self.power)
