import math

import torch
import torch.optim.lr_scheduler


class WarmupScheduler(torch.optim.lr_scheduler._LRScheduler):
    """A learning rate scheduler with a linear warmup."""

    def __init__(
        self, optimizer, warmup_steps=-1, schedule=(lambda steps: 1.0), last_epoch=-1
    ):
        self.warmup_steps = warmup_steps
        self.max_steps = -1
        self.schedule = schedule
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_steps:
            return [
                base_lr * self.last_epoch / self.warmup_steps
                for base_lr in self.base_lrs
            ]
        return [base_lr * self.schedule(self.last_epoch) for base_lr in self.base_lrs]


class CosineWarmupScheduler(WarmupScheduler):
    def __init__(self, optimizer, warmup_steps, total_steps, last_epoch=-1):
        self.total_steps = total_steps
        super().__init__(optimizer, warmup_steps, self._schedule, last_epoch)

    def _schedule(self, step):
        return 0.5 * (1.0 + torch.cos(math.pi * step / self.total_steps))

    def step(self, epoch=None):
        if epoch > self.total_steps:
            raise ValueError(
                f"Exhausted scheduler: the planned number of steps was {self.total_steps}"
            )
        super().step(epoch)


class LinearWarmupScheduler(WarmupScheduler):
    def __init__(self, optimizer, warmup_steps, total_steps, last_epoch=-1):
        self.total_steps = total_steps
        super().__init__(optimizer, warmup_steps, self._schedule, last_epoch)

    def _schedule(self, step):
        return 1.0 - step / self.total_steps

    def step(self, epoch=None):
        if epoch > self.total_steps:
            raise ValueError(
                f"Exhausted scheduler: the planned number of steps was {self.total_steps}"
            )
        super().step(epoch)


class NoamScheduler(WarmupScheduler):
    def __init__(self, optimizer, model_size, warmup_steps, last_epoch=-1):
        self.model_size = model_size
        super().__init__(optimizer, warmup_steps, self._schedule, last_epoch)

    def _schedule(self, step):
        return self.model_size ** -0.5 * step ** -0.5
