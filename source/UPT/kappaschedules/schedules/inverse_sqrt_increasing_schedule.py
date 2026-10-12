from .base import IncreasingProgressSchedule
from .functional.schedules import inverse_sqrt


class InverseSqrtIncreasingSchedule(IncreasingProgressSchedule):
    def _get_progress(self, step: int, total_steps: int, abs_step: int):
        return 1 - inverse_sqrt(step, abs_step or step)
