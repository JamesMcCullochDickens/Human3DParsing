from typing import List

import torch
import torch.optim.lr_scheduler as lr_scheduler


def get_k_step_lrs(optimizer: torch.optim.Optimizer, num_epochs: int,
                   interval: int, gamma_val: float) -> torch.optim.lr_scheduler:
    time_steps = list(range(interval, num_epochs, interval))
    return (torch.optim.lr_scheduler.
            MultiStepLR(optimizer=optimizer, milestones=time_steps, gamma=gamma_val))


def get_cos_annealing_lrs(optimizer: torch.optim.Optimizer, num_epochs: int,
                          min_learning_rate=0.0) -> torch.optim.lr_scheduler:
    lrs = (torch.optim.lr_scheduler.
           CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=min_learning_rate))
    return lrs


def get_evenly_distributed_lr(optimizer: torch.optim.Optimizer, starting_lr: float, num_epochs: int,
                              min_learning_rate: float = 0.0) -> torch.optim.lr_scheduler:
    ed_lr = EvenlyDistributedLR(optimizer, starting_lr, min_learning_rate, num_epochs)
    return ed_lr


class EvenlyDistributedLR(lr_scheduler.LRScheduler):
    def __init__(self, optimizer: torch.optim.Optimizer, starting_lr: float,
                 min_lr: float, num_epochs: int, last_epoch: int = -1) -> None:
        self.starting_lr = starting_lr
        self.min_lr = min_lr
        self.num_epochs = num_epochs
        self.step_size = (starting_lr - min_lr) / (num_epochs - 1)
        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> List[float]:
        return [max(self.min_lr, self.starting_lr - self.step_size * self.last_epoch)
                for _ in self.optimizer.param_groups]
