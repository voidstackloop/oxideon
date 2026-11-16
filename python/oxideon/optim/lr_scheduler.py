"""Learning Rate Schedulers for Oxideon Optimizers.

Provides PyTorch-compatible schedules:
- `StepLR`
- `ExponentialLR`
- `CosineAnnealingLR`
- `CosineAnnealingWarmRestarts`
- `OneCycleLR`
- `LinearLR`
- `ReduceLROnPlateau`
"""

import math
from typing import Any, Callable, List, Optional, Sequence, Union


class _LRScheduler:
    """Base class for all learning rate schedulers."""

    def __init__(self, optimizer: Any, last_epoch: int = -1):
        self.optimizer = optimizer
        # Extract base learning rates (per param group when present).
        groups = getattr(optimizer, "param_groups", None)
        if groups is None:
            if hasattr(optimizer, "lr"):
                current = optimizer.lr
                lr_val = float(current() if callable(current) else current)
            else:
                lr_val = 1e-3
            groups = [{"params": getattr(optimizer, "params", []), "lr": lr_val}]
            try:
                optimizer.param_groups = groups
            except (AttributeError, TypeError):
                pass
        if isinstance(groups, (list, tuple)) and len(groups) > 0:
            for g in groups:
                if not isinstance(g, dict) or "lr" not in g:
                    raise ValueError("scheduler: optimizer param groups must be dicts with an 'lr' entry")
            self.base_lrs = [float(g["lr"]) for g in groups]
        elif hasattr(optimizer, "lr"):
            current = optimizer.lr
            self.base_lrs = [float(current() if callable(current) else current)]
        else:
            self.base_lrs = [1e-3]

        self.last_epoch = last_epoch
        self._last_lr = list(self.base_lrs)
        self.step()

    def get_lr(self) -> List[float]:
        raise NotImplementedError

    def get_last_lr(self) -> List[float]:
        """Return last computed learning rate by current scheduler."""
        return self._last_lr

    def _live_lrs(self) -> List[float]:
        """Current learning rates straight from the optimizer (torch parity for non-decay steps)."""
        groups = getattr(self.optimizer, "param_groups", None)
        if isinstance(groups, (list, tuple)) and len(groups) > 0:
            return [float(g["lr"]) for g in groups]
        if hasattr(self.optimizer, "lr"):
            current = self.optimizer.lr
            return [float(current() if callable(current) else current)]
        return list(self.base_lrs)

    def _apply_lrs(self, values: List[float]) -> List[float]:
        """Writes computed values back (per param group when present)."""
        groups = getattr(self.optimizer, "param_groups", None)
        if isinstance(groups, (list, tuple)) and len(groups) > 0:
            vals = list(values)
            if len(vals) == 1 and len(groups) > 1:
                vals = vals * len(groups)
            if len(vals) != len(groups):
                raise ValueError(f"scheduler produced {len(vals)} lr values for {len(groups)} param groups")
            for g, lr in zip(groups, vals):
                g["lr"] = float(lr)
            if hasattr(self.optimizer, "set_learning_rate") and callable(self.optimizer.set_learning_rate):
                try:
                    self.optimizer.set_learning_rate(float(vals[0]))
                except Exception:
                    pass
            return [float(lr) for lr in vals]
        for lr in values:
            if hasattr(self.optimizer, "set_learning_rate"):
                self.optimizer.set_learning_rate(lr)
            elif hasattr(self.optimizer, "lr"):
                try:
                    self.optimizer.lr = lr
                except AttributeError:
                    pass
        return [float(lr) for lr in values]

    def step(self, epoch: Optional[int] = None) -> None:
        """Step could be called after every batch or every epoch."""
        if epoch is None:
            self.last_epoch += 1
            values = self.get_lr()
        else:
            self.last_epoch = epoch
            values = self.get_lr()

        self._last_lr = self._apply_lrs(values)


class StepLR(_LRScheduler):
    """Decays the learning rate of each parameter group by gamma every step_size epochs.

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        step_size (int): Period of learning rate decay.
        gamma (float): Multiplicative factor of learning rate decay. Default: 0.1.
        last_epoch (int): The index of last epoch. Default: -1.
    """

    def __init__(self, optimizer: Any, step_size: int, gamma: float = 0.1, last_epoch: int = -1):
        if isinstance(step_size, bool) or not isinstance(step_size, int) or step_size <= 0:
            raise ValueError(f"StepLR step_size must be a positive int, got {step_size!r}")
        self.step_size = step_size
        self.gamma = gamma
        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> List[float]:
        if self.last_epoch == 0 or self.last_epoch % self.step_size != 0:
            return self._live_lrs()
        return [
            base_lr * (self.gamma ** (self.last_epoch // self.step_size))
            for base_lr in self.base_lrs
        ]


class ExponentialLR(_LRScheduler):
    """Decays the learning rate of each parameter group by gamma every epoch.

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        gamma (float): Multiplicative factor of learning rate decay.
        last_epoch (int): The index of last epoch. Default: -1.
    """

    def __init__(self, optimizer: Any, gamma: float, last_epoch: int = -1):
        self.gamma = gamma
        self._constructing = True
        super().__init__(optimizer, last_epoch)

    def _current_lr(self) -> float:
        if hasattr(self.optimizer, "lr"):
            current = self.optimizer.lr
            return float(current() if callable(current) else current)
        return self.base_lrs[0] * (self.gamma ** max(0, self.last_epoch - 1))

    def get_lr(self) -> List[float]:
        # torch parity: the construction step lands on base_lr, afterwards each
        # step compounds the *live* lr (external edits are picked up).
        if self._constructing and self.last_epoch == 0:
            self._constructing = False
            return list(self.base_lrs)
        self._constructing = False
        return [self._current_lr() * self.gamma for _ in self.base_lrs]


class CosineAnnealingLR(_LRScheduler):
    """Set the learning rate using a cosine annealing schedule.

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        T_max (int): Maximum number of iterations.
        eta_min (float): Minimum learning rate. Default: 0.0.
        last_epoch (int): The index of last epoch. Default: -1.
    """

    def __init__(self, optimizer: Any, T_max: int, eta_min: float = 0.0, last_epoch: int = -1):
        if T_max <= 0:
            raise ValueError(f"CosineAnnealingLR: T_max must be positive, got {T_max}")
        self.T_max = T_max
        self.eta_min = eta_min
        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> List[float]:
        # torch parity: no clamp at T_max; the schedule oscillates past it.
        if self.last_epoch == 0:
            return self.base_lrs
        return [
            self.eta_min
            + (base_lr - self.eta_min)
            * (1.0 + math.cos(math.pi * self.last_epoch / self.T_max))
            / 2.0
            for base_lr in self.base_lrs
        ]


class CosineAnnealingWarmRestarts(_LRScheduler):
    """Set the learning rate using cosine annealing schedule with periodic restarts.

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        T_0 (int): Number of iterations for the first restart.
        T_mult (int, optional): A factor increases T_i after a restart. Default: 1.
        eta_min (float, optional): Minimum learning rate. Default: 0.0.
        last_epoch (int, optional): The index of last epoch. Default: -1.
    """

    def __init__(self, optimizer: Any, T_0: int, T_mult: int = 1, eta_min: float = 0.0, last_epoch: int = -1):
        if T_0 <= 0 or not isinstance(T_0, int):
            raise ValueError(f"Expected positive integer T_0, but got {T_0}")
        if T_mult < 1 or not isinstance(T_mult, int):
            raise ValueError(f"Expected integer T_mult >= 1, but got {T_mult}")
        self.T_0 = T_0
        self.T_mult = T_mult
        self.eta_min = eta_min
        # torch parity: T_cur tracks last_epoch so that the base-class
        # __init__ step() lands on T_cur=0 (base_lr), not T_cur=1.
        self.T_cur = last_epoch
        self.T_i = self.T_0
        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> List[float]:
        cos_val = math.cos(math.pi * self.T_cur / self.T_i)
        return [
            self.eta_min + (base_lr - self.eta_min) * (1.0 + cos_val) / 2.0
            for base_lr in self.base_lrs
        ]

    def step(self, epoch: Optional[int] = None) -> None:
        if epoch is None:
            self.last_epoch += 1
            self.T_cur += 1
            if self.T_cur >= self.T_i:
                self.T_cur = self.T_cur % self.T_i
                self.T_i = self.T_i * self.T_mult
        else:
            if epoch < 0:
                raise ValueError(f"Expected non-negative epoch, but got {epoch}")
            if epoch >= self.T_0:
                if self.T_mult == 1:
                    self.T_cur = epoch % self.T_0
                    self.T_i = self.T_0
                else:
                    n = int(math.log(epoch / self.T_0 * (self.T_mult - 1) + 1, self.T_mult))
                    self.T_cur = epoch - self.T_0 * (self.T_mult**n - 1) / (self.T_mult - 1)
                    self.T_i = self.T_0 * self.T_mult**n
            else:
                self.T_i = self.T_0
                self.T_cur = epoch
            self.last_epoch = math.floor(epoch)

        values = self.get_lr()
        self._last_lr = self._apply_lrs(values)


class LinearLR(_LRScheduler):
    """Decays or warms up the learning rate linearly until total_iters is reached.

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        start_factor (float): The number we multiply learning rate in the first epoch. Default: 1.0 / 3.
        end_factor (float): The number we multiply learning rate at the end of linear cycle. Default: 1.0.
        total_iters (int): The number of iterations that multiplicative factor reaches to 1. Default: 5.
        last_epoch (int): The index of the last epoch. Default: -1.
    """

    def __init__(
        self,
        optimizer: Any,
        start_factor: float = 1.0 / 3,
        end_factor: float = 1.0,
        total_iters: int = 5,
        last_epoch: int = -1,
    ):
        if start_factor > 1.0 or start_factor <= 0:
            raise ValueError("Starting multiplicative factor expected to be greater than 0 and less or equal to 1.")
        if end_factor > 1.0 or end_factor < 0:
            raise ValueError("Ending multiplicative factor expected to be between 0 and 1.")
        if total_iters <= 0:
            raise ValueError(f"LinearLR: total_iters must be positive, got {total_iters}")
        self.start_factor = start_factor
        self.end_factor = end_factor
        self.total_iters = total_iters
        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> List[float]:
        if self.last_epoch >= self.total_iters:
            factor = self.end_factor
        else:
            factor = self.start_factor + (self.end_factor - self.start_factor) * (
                self.last_epoch / self.total_iters
            )
        return [base_lr * factor for base_lr in self.base_lrs]


class OneCycleLR(_LRScheduler):
    """Sets the learning rate according to the 1cycle learning rate policy (Smith & Topin, 2019).

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        max_lr (float or list): Upper learning rate boundary in the cycle.
        total_steps (int): The total number of steps in the cycle.
        pct_start (float): The percentage of the cycle spent increasing the learning rate. Default: 0.3.
        div_factor (float): Determines the initial learning rate via initial_lr = max_lr/div_factor. Default: 25.
        final_div_factor (float): Determines the minimum learning rate via min_lr = initial_lr/final_div_factor. Default: 1e4.
    """

    def __init__(
        self,
        optimizer: Any,
        max_lr: float,
        total_steps: int,
        pct_start: float = 0.3,
        div_factor: float = 25.0,
        final_div_factor: float = 1e4,
        last_epoch: int = -1,
    ):
        if total_steps <= 0 or not isinstance(total_steps, int):
            raise ValueError(f"Expected positive integer total_steps, but got {total_steps}")
        if pct_start < 0 or pct_start > 1:
            raise ValueError(f"Expected pct_start between 0 and 1, but got {pct_start}")
        if div_factor <= 0:
            raise ValueError(f"OneCycleLR: div_factor must be positive, got {div_factor}")
        if final_div_factor <= 0:
            raise ValueError(f"OneCycleLR: final_div_factor must be positive, got {final_div_factor}")
        self.max_lr = max_lr
        self.total_steps = total_steps
        self.pct_start = pct_start
        self.div_factor = div_factor
        self.final_div_factor = final_div_factor
        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> List[float]:
        # torch parity: linear ramp peaking after `up = pct*total - 1` steps,
        # cosine tail over `down = total - up - 1` steps, ValueError past total.
        step = self.last_epoch
        if step > self.total_steps:
            raise ValueError(
                f"Tried to step {step} times. The specified number of total steps is {self.total_steps}"
            )
        initial_lr = self.max_lr / self.div_factor
        final_lr = initial_lr / self.final_div_factor
        up = max(1.0, self.total_steps * self.pct_start - 1.0)
        down = max(1.0, self.total_steps - up - 1.0)

        if step <= up:
            # Cosine ramp (torch default anneal_strategy='cos'); linear only
            # matches at the midpoint, quarter points discriminate.
            lr = initial_lr + (self.max_lr - initial_lr) * (1.0 - math.cos(math.pi * step / up)) / 2.0
        else:
            pct = (step - up) / down
            cos_val = math.cos(math.pi * pct)
            lr = final_lr + (self.max_lr - final_lr) * (1.0 + cos_val) / 2.0
        return [lr]


class ReduceLROnPlateau:
    """Reduce learning rate when a metric has stopped improving.

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        mode (str): One of `min`, `max`. In `min` mode, lr will be reduced when the quantity monitored has stopped decreasing.
        factor (float): Factor by which the learning rate will be reduced. new_lr = lr * factor. Default: 0.1.
        patience (int): Number of epochs with no improvement after which learning rate will be reduced. Default: 10.
        threshold (float): Threshold for measuring the new optimum. Default: 1e-4.
        threshold_mode (str): One of `rel`, `abs`. In `rel` mode (torch default),
            the threshold is relative to the best value. Default: `rel`.
        min_lr (float): A scalar lower bound on the learning rate. Default: 0.0.
    """

    def __init__(
        self,
        optimizer: Any,
        mode: str = "min",
        factor: float = 0.1,
        patience: int = 10,
        threshold: float = 1e-4,
        threshold_mode: str = "rel",
        min_lr: float = 0.0,
    ):
        if factor >= 1.0 or factor <= 0.0:
            raise ValueError(f"Factor should be in (0, 1), got {factor}")
        if mode not in ("min", "max"):
            raise ValueError(f"Mode should be 'min' or 'max', got {mode}")
        if threshold_mode not in ("rel", "abs"):
            raise ValueError(f"threshold_mode should be 'rel' or 'abs', got {threshold_mode}")

        self.optimizer = optimizer
        self.mode = mode
        self.factor = factor
        self.patience = patience
        self.threshold = threshold
        self.threshold_mode = threshold_mode
        self.min_lr = min_lr

        self.best = math.inf if mode == "min" else -math.inf
        self.num_bad_epochs = 0
        self.last_epoch = 0

    def _is_better(self, current: float) -> bool:
        if self.threshold_mode == "rel":
            if self.mode == "min":
                return current < self.best * (1.0 - self.threshold)
            return current > self.best * (1.0 + self.threshold)
        if self.mode == "min":
            return current < self.best - self.threshold
        return current > self.best + self.threshold

    def step(self, metrics: float) -> None:
        current = float(metrics)
        self.last_epoch += 1

        is_better = self._is_better(current)

        if is_better:
            self.best = current
            self.num_bad_epochs = 0
        else:
            self.num_bad_epochs += 1

        if self.num_bad_epochs > self.patience:
            self._reduce_lr()
            self.num_bad_epochs = 0

    def _reduce_lr(self) -> None:
        groups = getattr(self.optimizer, "param_groups", None)
        if isinstance(groups, (list, tuple)) and len(groups) > 0:
            for g in groups:
                g["lr"] = max(float(g["lr"]) * self.factor, self.min_lr)
            return
        if hasattr(self.optimizer, "lr"):
            cur = self.optimizer.lr
            cur_lr = float(cur() if callable(cur) else cur)
            new_lr = max(cur_lr * self.factor, self.min_lr)
            if hasattr(self.optimizer, "set_learning_rate"):
                self.optimizer.set_learning_rate(new_lr)
            else:
                try:
                    self.optimizer.lr = new_lr
                except AttributeError:
                    pass


class WarmupCosineScheduler(_LRScheduler):
    """Linear warmup followed by cosine annealing decay (standard LLM scheduler)."""

    def __init__(
        self,
        optimizer: Any,
        warmup_steps: int,
        total_steps: int,
        min_lr: float = 0.0,
        last_epoch: int = -1,
    ):
        if warmup_steps < 0 or total_steps <= warmup_steps:
            raise ValueError(
                f"WarmupCosineScheduler: need 0 <= warmup_steps < total_steps, got {warmup_steps}, {total_steps}"
            )
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr = float(min_lr)
        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> List[float]:
        step = self.last_epoch
        if step < self.warmup_steps:
            pct = float(step) / float(self.warmup_steps) if self.warmup_steps > 0 else 1.0
            return [base_lr * pct for base_lr in self.base_lrs]
        else:
            progress = float(step - self.warmup_steps) / float(self.total_steps - self.warmup_steps)
            progress = min(max(progress, 0.0), 1.0)
            cos_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
            return [
                self.min_lr + (base_lr - self.min_lr) * cos_decay
                for base_lr in self.base_lrs
            ]


class PolynomialLR(_LRScheduler):
    """Decays the learning rate using a polynomial function up to total_iters."""

    def __init__(
        self,
        optimizer: Any,
        total_iters: int = 5,
        power: float = 1.0,
        last_epoch: int = -1,
    ):
        if total_iters <= 0:
            raise ValueError(f"PolynomialLR: total_iters must be positive, got {total_iters}")
        self.total_iters = total_iters
        self.power = float(power)
        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> List[float]:
        if self.last_epoch >= self.total_iters:
            factor = 0.0
        else:
            factor = (1.0 - float(self.last_epoch) / float(self.total_iters)) ** self.power
        return [base_lr * factor for base_lr in self.base_lrs]


class CyclicLR(_LRScheduler):
    """Sets the learning rate according to the cyclical learning rate (CLR) policy."""

    def __init__(
        self,
        optimizer: Any,
        base_lr: float,
        max_lr: float,
        step_size_up: int = 2000,
        last_epoch: int = -1,
    ):
        if step_size_up <= 0:
            raise ValueError(f"CyclicLR: step_size_up must be positive, got {step_size_up}")
        self.clr_base_lr = float(base_lr)
        self.clr_max_lr = float(max_lr)
        self.step_size_up = step_size_up
        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> List[float]:
        cycle = math.floor(1.0 + float(self.last_epoch) / (2.0 * self.step_size_up))
        x = abs(float(self.last_epoch) / float(self.step_size_up) - 2.0 * cycle + 1.0)
        lr = self.clr_base_lr + (self.clr_max_lr - self.clr_base_lr) * max(0.0, 1.0 - x)
        return [lr for _ in self.base_lrs]


class ChainedScheduler:
    """Chains a list of learning rate schedulers."""

    def __init__(self, schedulers: Sequence[Any]):
        self.schedulers = list(schedulers)
        if len(self.schedulers) < 1:
            raise ValueError("ChainedScheduler expects at least one scheduler to be chained, but got no scheduler.")

    def step(self):
        for s in self.schedulers:
            s.step()

    def get_last_lr(self) -> List[float]:
        return self.schedulers[-1].get_last_lr() if self.schedulers else []


class MultiStepLR(_LRScheduler):
    """Decays the learning rate of each parameter group by gamma once the number of epoch reaches one of the milestones.

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        milestones (list): List of epoch indices. Must be increasing.
        gamma (float): Multiplicative factor of learning rate decay. Default: 0.1.
        last_epoch (int): The index of last epoch. Default: -1.
    """
    def __init__(self, optimizer: Any, milestones: Sequence[int], gamma: float = 0.1, last_epoch: int = -1):
        self.milestones = sorted(list(milestones))
        self.gamma = gamma
        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> List[float]:
        import bisect
        if self.last_epoch not in self.milestones:
            return self._live_lrs()
        count = bisect.bisect_right(self.milestones, self.last_epoch)
        return [base_lr * (self.gamma ** count) for base_lr in self.base_lrs]


class ConstantLR(_LRScheduler):
    """Decays the learning rate of each parameter group by a small constant factor until the number of epochs reaches total_iters.

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        factor (float): The number we multiply learning rate until the milestone. Default: 1.0 / 3.
        total_iters (int): The number of steps that the scheduler decays the learning rate. Default: 5.
        last_epoch (int): The index of the last epoch. Default: -1.
    """
    def __init__(self, optimizer: Any, factor: float = 1.0 / 3.0, total_iters: int = 5, last_epoch: int = -1):
        if factor > 1.0 or factor < 0:
            raise ValueError("Constant multiplicative factor expected to be between 0 and 1.")
        self.factor = float(factor)
        self.total_iters = total_iters
        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> List[float]:
        if self.last_epoch < self.total_iters:
            return [base_lr * self.factor for base_lr in self.base_lrs]
        return [base_lr for base_lr in self.base_lrs]


class LambdaLR(_LRScheduler):
    """Sets the learning rate of each parameter group to the initial lr times a given function.

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        lr_lambda (function or list): A function which computes a multiplicative factor given an integer parameter epoch, or a list of such functions.
        last_epoch (int): The index of last epoch. Default: -1.
    """
    def __init__(self, optimizer: Any, lr_lambda: Union[Callable[[int], float], Sequence[Callable[[int], float]]], last_epoch: int = -1):
        n_groups = len(getattr(optimizer, "param_groups", [None]))
        if not isinstance(lr_lambda, (list, tuple)):
            self.lr_lambdas = [lr_lambda] * n_groups
        else:
            self.lr_lambdas = list(lr_lambda)
            if len(self.lr_lambdas) != n_groups:
                raise ValueError(f"Expected {n_groups} lr_lambdas, but got {len(self.lr_lambdas)}")
        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> List[float]:
        return [
            base_lr * lmbda(self.last_epoch)
            for lmbda, base_lr in zip(self.lr_lambdas, self.base_lrs)
        ]


class MultiplicativeLR(_LRScheduler):
    """Multiply the learning rate of each parameter group by the factor given in the specified function."""

    def __init__(self, optimizer: Any, lr_lambda: Union[Callable[[int], float], Sequence[Callable[[int], float]]], last_epoch: int = -1):
        n_groups = len(getattr(optimizer, "param_groups", [None]))
        if not isinstance(lr_lambda, (list, tuple)):
            self.lr_lambdas = [lr_lambda] * n_groups
        else:
            self.lr_lambdas = list(lr_lambda)
            if len(self.lr_lambdas) != n_groups:
                raise ValueError(f"Expected {n_groups} lr_lambdas, but got {len(self.lr_lambdas)}")
        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> List[float]:
        if self.last_epoch > 0:
            return [
                group_lr * lmbda(self.last_epoch)
                for lmbda, group_lr in zip(self.lr_lambdas, self._last_lr)
            ]
        return self._live_lrs()


class SequentialLR(_LRScheduler):
    """Receives the list of schedulers that is to be called sequentially in an epoch manner.

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        schedulers (list): List of chained schedulers.
        milestones (list): List of integers that reflects milestone points.
        last_epoch (int): The index of last epoch. Default: -1.
    """
    def __init__(self, optimizer: Any, schedulers: Sequence[Any], milestones: Sequence[int], last_epoch: int = -1):
        self.schedulers = list(schedulers)
        self.milestones = list(milestones)
        if len(self.schedulers) < 1:
            raise ValueError("SequentialLR expects at least one scheduler, but got no scheduler.")
        if len(self.milestones) != len(self.schedulers) - 1:
            raise ValueError(
                "SequentialLR expects number of schedulers provided to be one more "
                f"than the number of milestone points, but got {len(self.schedulers)} schedulers "
                f"and {len(self.milestones)} milestones"
            )
        self._last_lr = self.schedulers[0].get_last_lr()
        super().__init__(optimizer, last_epoch)

    def step(self, epoch: Optional[int] = None) -> None:
        if epoch is None:
            self.last_epoch += 1
        else:
            self.last_epoch = epoch

        idx = 0
        for m in self.milestones:
            if self.last_epoch >= m:
                idx += 1
            else:
                break
        idx = min(idx, len(self.schedulers) - 1)
        self.schedulers[idx].step()
        self._last_lr = self.schedulers[idx].get_last_lr()

    def get_lr(self) -> List[float]:
        return self._last_lr


__all__ = [
    "StepLR",
    "MultiStepLR",
    "ConstantLR",
    "LinearLR",
    "ExponentialLR",
    "CosineAnnealingLR",
    "CosineAnnealingWarmRestarts",
    "OneCycleLR",
    "ReduceLROnPlateau",
    "WarmupCosineScheduler",
    "PolynomialLR",
    "CyclicLR",
    "ChainedScheduler",
    "LambdaLR",
    "MultiplicativeLR",
    "SequentialLR",
]
