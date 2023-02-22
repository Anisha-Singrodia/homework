from typing import List

from torch.optim.lr_scheduler import _LRScheduler
import math
import warnings


class CustomLRScheduler(_LRScheduler):
    """Custom LR Scheduler
    Args:
        _LRScheduler (_type_): Custom LR Scheduler
    """

    def __init__(self, optimizer, T_0, T_mult, eta_min=0, last_epoch=-1, verbose=False):
        """
        Create a new scheduler.
        Note to students: You can change the arguments to this constructor,
        if you need to add new parameters.
        """
        # ... Your Code Here ...
        # self.T_max = T_max
        # self.eta_min = eta_min
        if T_0 <= 0 or not isinstance(T_0, int):
            raise ValueError("Expected positive integer T_0, but got {}".format(T_0))
        if T_mult < 1 or not isinstance(T_mult, int):
            raise ValueError("Expected integer T_mult >= 1, but got {}".format(T_mult))
        self.T_0 = T_0
        self.T_i = T_0
        self.T_mult = T_mult
        self.eta_min = eta_min
        self.T_cur = last_epoch
        super(CustomLRScheduler, self).__init__(optimizer, last_epoch)

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

        return [
            self.eta_min
            + (base_lr - self.eta_min)
            * (1 + math.cos(math.pi * self.T_cur / self.T_i))
            / 2
            for base_lr in self.base_lrs
        ]

    def step(self, epoch=None):
        """Step could be called after every batch update"""

        if epoch is None and self.last_epoch < 0:
            epoch = 0

        if epoch is None:
            epoch = self.last_epoch + 1
            self.T_cur = self.T_cur + 1
            if self.T_cur >= self.T_i:
                self.T_cur = self.T_cur - self.T_i
                self.T_i = self.T_i * self.T_mult
        else:
            if epoch < 0:
                raise ValueError(
                    "Expected non-negative epoch, but got {}".format(epoch)
                )
            if epoch >= self.T_0:
                if self.T_mult == 1:
                    self.T_cur = epoch % self.T_0
                else:
                    n = int(
                        math.log(
                            (epoch / self.T_0 * (self.T_mult - 1) + 1), self.T_mult
                        )
                    )
                    self.T_cur = epoch - self.T_0 * (self.T_mult**n - 1) / (
                        self.T_mult - 1
                    )
                    self.T_i = self.T_0 * self.T_mult ** (n)
            else:
                self.T_i = self.T_0
                self.T_cur = epoch
        self.last_epoch = math.floor(epoch)

        class _enable_get_lr_call:
            def __init__(self, o):
                self.o = o

            def __enter__(self):
                self.o._get_lr_called_within_step = True
                return self

            def __exit__(self, type, value, traceback):
                self.o._get_lr_called_within_step = False
                return self

        with _enable_get_lr_call(self):
            for i, data in enumerate(zip(self.optimizer.param_groups, self.get_lr())):
                param_group, lr = data
                param_group["lr"] = lr
                self.print_lr(self.verbose, i, lr, epoch)

        self._last_lr = [group["lr"] for group in self.optimizer.param_groups]
        # Note to students: You CANNOT change the arguments or return type of
        # this function (because it is called internally by Torch)

        # ... Your Code Here ...
        # Here's our dumb baseline implementation:
        # return [i for i in self.base_lrs]
        # if not self._get_lr_called_within_step:
        #     warnings.warn(
        #         "To get the last learning rate computed by the scheduler, please use `get_last_lr()`.",
        #         UserWarning,
        #     )

        # if self.last_epoch == 0:
        #     # print("in 1")
        #     return [group["lr"] for group in self.optimizer.param_groups]
        # elif self._step_count == 1 and self.last_epoch > 0:
        #     # print("in 2")
        #     return [
        #         self.eta_min
        #         + (base_lr - self.eta_min)
        #         * (1 + math.cos((self.last_epoch) * math.pi / self.T_max))
        #         / 2
        #         for base_lr, group in zip(self.base_lrs, self.optimizer.param_groups)
        #     ]
        # elif (self.last_epoch - 1 - self.T_max) % (2 * self.T_max) == 0:
        #     # print("in 3")
        #     return [
        #         group["lr"]
        #         + (base_lr - self.eta_min) * (1 - math.cos(math.pi / self.T_max)) / 2
        #         for base_lr, group in zip(self.base_lrs, self.optimizer.param_groups)
        #     ]
        # # print("in 4")
        # return [
        #     (1 + math.cos(math.pi * self.last_epoch / self.T_max))
        #     / (1 + math.cos(math.pi * (self.last_epoch - 1) / self.T_max))
        #     * (group["lr"] - self.eta_min)
        #     + self.eta_min
        #     for group in self.optimizer.param_groups
        # ]

    # def _get_closed_form_lr(self):
    #     return [
    #         self.eta_min
    #         + (base_lr - self.eta_min)
    #         * (1 + math.cos(math.pi * self.last_epoch / self.T_max))
    #         / 2
    #         for base_lr in self.base_lrs
    #     ]