from .base import IncreasingProgressSchedule
from .functional.schedules import cosine


class CosineIncreasingSchedule(IncreasingProgressSchedule):
    def _get_progress(self, step: int, total_steps: int, abs_step: int):
        return cosine(step, total_steps)
