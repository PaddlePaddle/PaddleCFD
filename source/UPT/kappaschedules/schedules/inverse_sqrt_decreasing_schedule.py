from .base import DecreasingProgressSchedule
from .functional.schedules import inverse_sqrt


class InverseSqrtDecreasingSchedule(DecreasingProgressSchedule):
    def _get_progress(self, step: int, total_steps: int, abs_step: int):
        return 1 - inverse_sqrt(step, abs_step or step)
