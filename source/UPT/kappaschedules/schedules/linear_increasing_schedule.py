from .base import IncreasingProgressSchedule

from .functional.schedules import linear


class LinearIncreasingSchedule(IncreasingProgressSchedule):
    def _get_progress(self, step: int, total_steps: int, abs_step: int):
        return linear(step, total_steps)
