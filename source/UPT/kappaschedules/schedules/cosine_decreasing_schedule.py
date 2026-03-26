from .base import DecreasingProgressSchedule
from .functional.schedules import cosine


class CosineDecreasingSchedule(DecreasingProgressSchedule):
    def _get_progress(self, step: int, total_steps: int, abs_step: int):
        return cosine(step, total_steps)
