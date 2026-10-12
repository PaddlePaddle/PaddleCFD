from .base import DecreasingProgressSchedule

from .functional.schedules import linear


class LinearDecreasingSchedule(DecreasingProgressSchedule):
    def _get_progress(self, step: int, total_steps: int, abs_step: int):
        return linear(step, total_steps)
