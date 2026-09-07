from collections import deque

import paddle
from callbacks.base.periodic_callback import PeriodicCallback


class SkipLossSpikesCallback(PeriodicCallback):
    def __init__(
        self,
        max_skipped_updates_in_a_row=100,
        queue_size=50,
        tolerance_factor=0.2,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.max_skipped_updates_in_a_row = max_skipped_updates_in_a_row
        self.queue_size = queue_size
        self.tolerance_factor = tolerance_factor
        self.queue = deque([], maxlen=queue_size)
        self.skipped_updates_in_a_row = 0
        self.accumulation_queue = []

    def state_dict(self):
        return dict(
            queue=list(self.queue),
            skipped_updates_in_a_row=self.skipped_updates_in_a_row,
        )

    def load_state_dict(self, state_dict):
        for item in state_dict["queue"]:
            self.queue.append(item)
        self.skipped_updates_in_a_row = state_dict["skipped_updates_in_a_row"]

    def after_every_backward(self, total_loss, **kwargs):
        self.accumulation_queue.append(total_loss.detach())

    def before_every_optim_step(self, model, **kwargs):
        if self.skipped_updates_in_a_row > self.max_skipped_updates_in_a_row:
            raise RuntimeError(
                f"skipped {self.max_skipped_updates_in_a_row} in a row -> kill"
            )
        if len(self.accumulation_queue) > 1:
            total_loss = paddle.stack(self.accumulation_queue).mean()
        else:
            total_loss = self.accumulation_queue[0]
        self.accumulation_queue.clear()
        if len(self.queue) == self.queue.maxlen:
            queue_avg = paddle.stack(list(self.queue)).mean()
            max_tolerable_loss = queue_avg * (1 + self.tolerance_factor)
            if total_loss > max_tolerable_loss:
                self.skipped_updates_in_a_row += 1
                self.logger.info(
                    f"{self.update_counter.cur_checkpoint} skipping batch due to high loss (total_loss={total_loss.item()} max_tolerable_loss={max_tolerable_loss.item()} tolerance_factor={self.tolerance_factor} skipped_in_a_row={self.skipped_updates_in_a_row})"
                )
                for p in model.parameters():
                    p.grad = None
            else:
                self.skipped_updates_in_a_row = 0
        else:
            pass
        self.queue.append(total_loss)
