from .linear_warmup_cosine_decay_schedule import LinearWarmupCosineDecaySchedule


class ColdstartLinearWarmupCosineDecaySchedule(LinearWarmupCosineDecaySchedule):
    def __init__(self, coldstart_steps=None, coldstart_percent=None, **kwargs):
        super().__init__(**kwargs)
        assert not (coldstart_steps is not None and coldstart_percent is not None), \
            "specify either coldstart_steps or coldstart_percent, not both"
        self.coldstart_steps = coldstart_steps
        self.coldstart_percent = coldstart_percent

    def _get_value(self, step: int, total_steps: int, abs_step: int = None) -> float:
        if self.coldstart_steps is not None:
            if step < self.coldstart_steps:
                return 0.
        if self.coldstart_percent is not None:
            coldstart_steps = self.coldstart_percent * total_steps
            if step < coldstart_steps:
                return 0.
        return self.schedule.get_value(step=step, total_steps=total_steps, abs_step=abs_step)

    def __str__(self):
        coldstart_str = ""
        if self.coldstart_percent is not None:
            coldstart_str = f"coldstart_percent={self.coldstart_percent},"
        if self.coldstart_steps is not None:
            coldstart_str = f"coldstart_steps={self.coldstart_steps},"
        if self.warmup_percent is not None:
            return f"{type(self).__name__}({coldstart_str}warmup_percent={self.warmup_percent})"
        if self.warmup_steps is not None:
            return f"{type(self).__name__}({coldstart_str}warmup_steps={self.warmup_steps})"
        raise RuntimeError
