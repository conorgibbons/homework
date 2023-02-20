from typing import List

from torch.optim.lr_scheduler import _LRScheduler

import math


class CustomLRScheduler(_LRScheduler):
    """
    Testing.....
    """

    def __init__(self, optimizer, last_epoch=-1, **kwargs):
        """
        Create a new scheduler.

        Note to students: You can change the arguments to this constructor,
        if you need to add new parameters.

        """
        # ... Your Code Here ...

        self.start_factor = kwargs["start_factor"]
        self.end_factor = kwargs["end_factor"]
        self.total_iters = kwargs["total_iters"]
        super(CustomLRScheduler, self).__init__(optimizer, last_epoch)

    def get_lr(self) -> List[float]:
        """
        Test...
        """
        if self.last_epoch == 0:
            return [
                group["lr"] * self.start_factor for group in self.optimizer.param_groups
            ]

        if self.last_epoch > self.total_iters:
            return [group["lr"] for group in self.optimizer.param_groups]

        return [
            group["lr"]
            * (
                1.0
                + (self.end_factor - self.start_factor)
                / (
                    self.total_iters * self.start_factor
                    + (self.last_epoch - 1) * (self.end_factor - self.start_factor)
                )
            )
            for group in self.optimizer.param_groups
        ]

    def _get_closed_form_lr(self):
        """
        Test
        """
        return [
            base_lr
            * (
                self.start_factor
                + (self.end_factor - self.start_factor)
                * min(self.total_iters, self.last_epoch)
                / self.total_iters
            )
            for base_lr in self.base_lrs
        ]
