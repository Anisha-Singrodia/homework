from typing import List

from torch.optim.lr_scheduler import _LRScheduler
import math
import warnings


class CustomLRScheduler(_LRScheduler):
    """Custom LR Scheduler
    Args:
        _LRScheduler (_type_): Custom LR Scheduler
    """

    def __init__(self, optimizer, T_max, eta_min=0, last_epoch=-1, verbose=False):
        """
        Create a new scheduler.
        Note to students: You can change the arguments to this constructor,
        if you need to add new parameters.
        """
        self.T_max = T_max
        self.eta_min = eta_min
        super(CustomLRScheduler, self).__init__(optimizer, last_epoch)

        # ... Your Code Here ...
        # self.T_max = T_max
        # self.eta_min = eta_min

    #     if T_0 <= 0 or not isinstance(T_0, int):
    #         raise ValueError("Expected positive integer T_0, but got {}".format(T_0))
    #     if T_mult < 1 or not isinstance(T_mult, int):
    #         raise ValueError("Expected integer T_mult >= 1, but got {}".format(T_mult))
    #     self.T_0 = T_0
    #     self.T_i = T_0
    #     self.T_mult = T_mult
    #     self.eta_min = eta_min
    #     self.T_cur = last_epoch
    #     super(CustomLRScheduler, self).__init__(optimizer, last_epoch)

    def get_lr(self) -> List[float]:
        """Returns all schedulers
        Returns:
            List[float]: list of all the schedulers
        """

        if not self._get_lr_called_within_step:
            warnings.warn(
                "To get the last learning rate computed by the scheduler, "
                "please use `get_last_lr()`.",
                UserWarning,
            )

        if self.last_epoch == 0:
            return [group["lr"] for group in self.optimizer.param_groups]
        elif self._step_count == 1 and self.last_epoch > 0:
            return [
                self.eta_min
                + (base_lr - self.eta_min)
                * (1 + math.cos((self.last_epoch) * math.pi / self.T_max))
                / 2
                for base_lr, group in zip(self.base_lrs, self.optimizer.param_groups)
            ]
        elif (self.last_epoch - 1 - self.T_max) % (2 * self.T_max) == 0:
            return [
                group["lr"]
                + (base_lr - self.eta_min) * (1 - math.cos(math.pi / self.T_max)) / 2
                for base_lr, group in zip(self.base_lrs, self.optimizer.param_groups)
            ]
        return [
            (1 + math.cos(math.pi * self.last_epoch / self.T_max))
            / (1 + math.cos(math.pi * (self.last_epoch - 1) / self.T_max))
            * (group["lr"] - self.eta_min)
            + self.eta_min
            for group in self.optimizer.param_groups
        ]
