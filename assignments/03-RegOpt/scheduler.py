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

        self.gamma = kwargs["gamma"]
        self.milestones = Counter(kwargs["milestones"])
        super(CustomLRScheduler, self).__init__(optimizer, last_epoch)

    def get_lr(self) -> List[float]:
        """
        Test...
        """
        if self.last_epoch not in self.milestones:
            return [group['lr'] for group in self.optimizer.param_groups]
        return [group['lr'] * self.gamma ** self.milestones[self.last_epoch]
                for group in self.optimizer.param_groups]

    def _get_closed_form_lr(self):
        """
        Test
        """
        milestones = list(sorted(self.milestones.elements()))
        return [base_lr * self.gamma ** bisect_right(milestones, self.last_epoch)
                for base_lr in self.base_lrs]
