from .progress_schedule import ProgressSchedule


class IncreasingProgressSchedule(ProgressSchedule):
    def __init__(self, start_value=0., max_value=1., **kwargs):
        delta = max_value - start_value
        assert delta >= 0.
        super().__init__(start_value=start_value, delta=delta, **kwargs)

    def _get_progress(self, step: int, total_steps: int, abs_step: int):
        raise NotImplementedError
