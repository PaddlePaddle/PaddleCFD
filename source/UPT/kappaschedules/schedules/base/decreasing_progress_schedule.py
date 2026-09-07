from .progress_schedule import ProgressSchedule


class DecreasingProgressSchedule(ProgressSchedule):
    def __init__(self, max_value=1., end_value=0., **kwargs):
        delta = end_value - max_value
        assert delta <= 0.
        super().__init__(start_value=max_value, delta=delta, **kwargs)

    def _get_progress(self, step: int, total_steps: int, abs_step: int):
        raise NotImplementedError
