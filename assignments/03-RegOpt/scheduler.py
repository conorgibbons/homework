from typing import List

from torch.optim.lr_scheduler import _LRScheduler

import math


class CustomLRScheduler(_LRScheduler):
    """
    Decays the learning rate of each parameter group by gamma every
    step_size epochs. Notice that such decay can happen simultaneously with
    other changes to the learning rate from outside this scheduler. When
    last_epoch=-1, sets initial lr as lr.

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        step_size (int): Period of learning rate decay.
        gamma (float): Multiplicative factor of learning rate decay.
            Default: 0.1.
        last_epoch (int): The index of last epoch. Default: -1.
        verbose (bool): If ``True``, prints a message to stdout for
            each update. Default: ``False``.
    """

    def __init__(self, optimizer, last_epoch=-1, **kwargs):
        """
        Create a new scheduler.

        Note to students: You can change the arguments to this constructor,
        if you need to add new parameters.

        """
        # ... Your Code Here ...
        self.gamma = kwargs["gamma"]
        self.step_size = kwargs["step_size"]

        super(CustomLRScheduler, self).__init__(optimizer, last_epoch)

    def get_lr(self) -> List[float]:
        """
        Returns a list of learning rates for each parameter group in the optimizer. If the current epoch is
        zero or not a multiple of the step size, returns the current learning rate for each group. Otherwise,
        multiplies the current learning rate for each group by the gamma value.

        Returns:
        --------
        lrs : list of floats
            The learning rate for each parameter group in the optimizer.
        """
        lrs = []
        for group in self.optimizer.param_groups:
            if (self.last_epoch == 0) or (self.last_epoch % self.step_size != 0):
                lrs.append(group["lr"])
            else:
                lrs.append(group["lr"] * self.gamma)
        return lrs

    def _get_closed_form_lr(self):
        """
        Returns a list of learning rates for each parameter group in the optimizer, calculated using a closed-form
        formula. The learning rate for each group is calculated by multiplying the base learning rate for the group
        by the gamma value raised to the power of the current epoch divided by the step size.

        Returns:
        --------
        lrs : list of floats
            The learning rate for each parameter group in the optimizer.
        """
        lrs = []
        for base_lr in self.base_lrs:
            lr = base_lr * self.gamma ** (self.last_epoch // self.step_size)
            lrs.append(lr)
        return lrs
