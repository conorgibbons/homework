from typing import List

from torch.optim.lr_scheduler import _LRScheduler


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

        self.step_size = kwargs["step_size"]
        self.gamma = kwargs["gamma"]
        super(CustomLRScheduler, self).__init__(optimizer, last_epoch)

    def get_lr(self) -> List[float]:
        """
        Testing.....
        """
        # Note to students: You CANNOT change the arguments or return type of
        # this function (because it is called internally by Torch)

        # ... Your Code Here ...
        if not self._get_lr_called_within_step:
            warnigns.warn(
                "To get the last learning rate computer by the scheduler plesae use  `get_last_lr() `.",
                UserWarning,
            )

        if (self.last_epoch == 0) | (self.last_epoch % self.step_size != 0):
            return [group["lr"] for group in self.optimizer.param_groups]

        return [group["lr"] * self.gamma for group in self.optimizer.param_groups]

    def _get_closed_form_lr(self):
        """
        Testing.....
        """
        return [
            base_lr * self.gamma ** (self.last_epoch // self.step_size)
            for base_lr in self.base_lrs
        ]
