class ScheduleBase:
    def __init__(self, recess_percent=None, recess_steps=None, overhang_percent=None, overhang_steps=None):
        # schedule is artificially prolonged at the start
        assert recess_percent is None or recess_steps is None, \
            f"recess_percent and recess_steps are mutually exclusive"
        # schedule is artificially prolonged at the end
        assert overhang_percent is None or overhang_steps is None, \
            f"overhang_percent and overhang_steps are mutually exclusive"
        self.recess_percent = recess_percent
        self.recess_steps = recess_steps
        self.overhang_percent = overhang_percent
        self.overhang_steps = overhang_steps
        # check that correct method is overwritten
        assert type(self).get_value == ScheduleBase.get_value

    def get_value(self, step: int, total_steps: int, abs_step: int = None) -> float:
        assert 0 <= step < total_steps, f"0 <= step < total_steps (step={step} total_steps={total_steps})"
        # recess
        if self.recess_percent is not None:
            offset = int(total_steps * self.recess_percent)
            step += offset
            total_steps += offset
        if self.recess_steps is not None:
            step += self.recess_steps
            total_steps += self.recess_steps
        # overhang
        if self.overhang_percent is not None:
            total_steps += int(total_steps * self.overhang_percent)
        if self.overhang_steps is not None:
            total_steps += self.overhang_steps
        # calculate value
        return self._get_value(step, total_steps, abs_step)

    def _get_value(self, step: int, total_steps: int, abs_step: int = None) -> float:
        raise NotImplementedError

    def __repr__(self):
        return str(self)

    def __str__(self):
        raise NotImplementedError
