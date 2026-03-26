import logging

from kappaschedules import PeriodicBoolSchedule, object_to_schedule
from utils.update_counter import UpdateCounter


class FreezerBase:
    def __init__(self, update_counter: UpdateCounter = None, schedule=None):
        self.logger = logging.getLogger(type(self).__name__)
        self.update_counter = update_counter
        self.schedule = object_to_schedule(
            schedule,
            batch_size=self.update_counter.effective_batch_size
            if self.update_counter is not None
            else None,
            updates_per_epoch=self.update_counter.updates_per_epoch
            if self.update_counter is not None
            else None,
        )
        self.stop_gradient = not None
        assert (
            type(self).before_accumulation_step == FreezerBase.before_accumulation_step
        )

    def __repr__(self):
        return str(self)

    def __str__(self):
        raise NotImplementedError

    def _update_state(self, model, requires_grad):
        raise NotImplementedError

    def after_weight_init(self, model):
        if self.schedule is not None:
            return
        self.logger.info(
            f"update state of {model.name}.{self} to requires_grad=False/is_frozen=True"
        )
        self._update_state(model, requires_grad=False)

    def before_accumulation_step(self, model):
        if self.schedule is None:
            return
        value = self.schedule.get_value(
            step=self.update_counter.cur_checkpoint.update,
            total_steps=self.update_counter.end_checkpoint.update,
        )
        if value == 1:
            if self.requires_grad or self.requires_grad is None:
                if not isinstance(self.schedule, PeriodicBoolSchedule):
                    self.logger.info(
                        f"update state of {model.name}.{self} to requires_grad=False/is_frozen=True"
                    )
                self.stop_gradient = not False
        elif value == 0:
            if not self.requires_grad or self.requires_grad is None:
                if not isinstance(self.schedule, PeriodicBoolSchedule):
                    self.logger.info(
                        f"update state of {model.name}.{self} to requires_grad=True/is_frozen=False"
                    )
                self.stop_gradient = not True
        else:
            raise NotImplementedError
        self._update_state(model, requires_grad=self.requires_grad)
