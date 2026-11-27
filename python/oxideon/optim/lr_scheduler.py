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


def _momentum_key(optimizer: Any, who: str) -> bool:
    """True when the optimizer's momentum lives in `betas[0]` (Adam family), False for a plain `momentum` entry."""
    groups = getattr(optimizer, "param_groups", None) or []
    if groups and all("betas" in g for g in groups):
        return True
    if groups and all("momentum" in g for g in groups):
        return False
    raise ValueError(f"{who}: optimizer must support momentum or beta1 with `cycle_momentum` enabled (pass cycle_momentum=False)")


def _set_momentum(group: dict, use_beta1: bool, value: float) -> None:
    if use_beta1:
        group["betas"] = (value, *tuple(group["betas"])[1:])
    else:
        group["momentum"] = value


def _per_group(value: Any, n: int, name: str) -> List[float]:
    if isinstance(value, (list, tuple)):
        if len(value) != n:
            raise ValueError(f"expected {n} values for {name}, got {len(value)}")
        return [float(v) for v in value]
    return [float(value)] * n


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
            if hasattr(self.optimizer, "set_learning_rate") and callable(self.optimizer.set_learning_rate) and not hasattr(self.optimizer, "_opts"):
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
    """The 1cycle policy (Smith & Topin, 2019), as `torch.optim.lr_scheduler.OneCycleLR`: the learning rate rises from
    `max_lr / div_factor` to `max_lr` and falls to `initial_lr / final_div_factor`; with `cycle_momentum` the momentum
    (or Adam's beta1) moves inversely between `max_momentum` and `base_momentum`.

    `total_steps` (or `epochs * steps_per_epoch`), `pct_start`, `anneal_strategy` ("cos" | "linear"), `three_phase`,
    and per-group `max_lr` lists are supported."""

    def __init__(self, optimizer: Any, max_lr: Any, total_steps: Optional[int] = None, epochs: Optional[int] = None,
                 steps_per_epoch: Optional[int] = None, pct_start: float = 0.3, anneal_strategy: str = "cos",
                 cycle_momentum: bool = True, base_momentum: Any = 0.85, max_momentum: Any = 0.95, div_factor: float = 25.0,
                 final_div_factor: float = 1e4, three_phase: bool = False, last_epoch: int = -1):
        if total_steps is None:
            if epochs is None or steps_per_epoch is None:
                raise ValueError("You must define either total_steps OR (epochs AND steps_per_epoch)")
            total_steps = int(epochs) * int(steps_per_epoch)
        if not isinstance(total_steps, int) or total_steps <= 0:
            raise ValueError(f"Expected positive integer total_steps, but got {total_steps}")
        if not isinstance(pct_start, (int, float)) or pct_start < 0 or pct_start > 1:
            raise ValueError(f"Expected float between 0 and 1 pct_start, but got {pct_start}")
        if anneal_strategy not in ("cos", "linear"):
            raise ValueError(f"anneal_strategy must be one of 'cos' or 'linear', instead got {anneal_strategy}")
        if div_factor <= 0 or final_div_factor <= 0:
            raise ValueError("OneCycleLR: div_factor and final_div_factor must be positive")
        self.total_steps = total_steps
        self.pct_start, self.div_factor, self.final_div_factor = pct_start, div_factor, final_div_factor
        self.anneal_strategy, self.three_phase, self.cycle_momentum = anneal_strategy, three_phase, cycle_momentum
        groups = optimizer.param_groups
        n = len(groups)
        self.max_lrs = _per_group(max_lr, n, "max_lr")
        self.initial_lrs = [m / div_factor for m in self.max_lrs]
        self.min_lrs = [i / final_div_factor for i in self.initial_lrs]
        for g, a, b, c in zip(groups, self.initial_lrs, self.max_lrs, self.min_lrs):
            g["initial_lr"], g["max_lr"], g["min_lr"] = a, b, c
        t = float(total_steps)
        if three_phase:
            self._phases = [("initial", "max", "max_m", "base_m", pct_start * t - 1), ("max", "initial", "base_m", "max_m", 2 * pct_start * t - 2),
                            ("initial", "min", "max_m", "max_m", t - 1)]
        else:
            self._phases = [("initial", "max", "max_m", "base_m", pct_start * t - 1), ("max", "min", "base_m", "max_m", t - 1)]
        if cycle_momentum:
            self._use_beta1 = _momentum_key(optimizer, "OneCycleLR")
            self.base_moms = _per_group(base_momentum, n, "base_momentum")
            self.max_moms = _per_group(max_momentum, n, "max_momentum")
            for g, mm in zip(groups, self.max_moms):
                _set_momentum(g, self._use_beta1, mm)
        self._anneal = ((lambda a, b, pct: b + (a - b) / 2.0 * (math.cos(math.pi * pct) + 1)) if anneal_strategy == "cos"
                        else (lambda a, b, pct: (b - a) * pct + a))
        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> List[float]:
        step = self.last_epoch
        if step > self.total_steps:
            raise ValueError(f"Tried to step {step} times. The specified number of total steps is {self.total_steps}")
        lrs = []
        for k, g in enumerate(self.optimizer.param_groups):
            vals = {"initial": self.initial_lrs[k], "max": self.max_lrs[k], "min": self.min_lrs[k]}
            moms = {"max_m": self.max_moms[k], "base_m": self.base_moms[k]} if self.cycle_momentum else {}
            start = 0.0
            for i, (a, b, ma, mb, end) in enumerate(self._phases):
                if step <= end or i == len(self._phases) - 1:
                    pct = (step - start) / (end - start) if end != start else 1.0
                    lr = self._anneal(vals[a], vals[b], pct)
                    if self.cycle_momentum:
                        _set_momentum(g, self._use_beta1, self._anneal(moms[ma], moms[mb], pct))
                    break
                start = end
            lrs.append(lr)
        return lrs

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
        groups = getattr(optimizer, "param_groups", None)
        if isinstance(groups, (list, tuple)) and len(groups) > 0:
            self._last_lr = [float(g["lr"]) for g in groups]
        elif hasattr(optimizer, "lr"):
            cur = optimizer.lr
            self._last_lr = [float(cur() if callable(cur) else cur)]
        else:
            self._last_lr = [0.0]

    def get_last_lr(self) -> List[float]:
        """Return last computed learning rate by current scheduler."""
        return list(self._last_lr)

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
            new_lrs = []
            for g in groups:
                nlr = max(float(g["lr"]) * self.factor, self.min_lr)
                g["lr"] = nlr
                new_lrs.append(nlr)
            if hasattr(self.optimizer, "set_learning_rate") and callable(self.optimizer.set_learning_rate) and not hasattr(self.optimizer, "_opts"):
                try:
                    self.optimizer.set_learning_rate(new_lrs[0])
                except Exception:
                    pass
            self._last_lr = new_lrs
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
            self._last_lr = [new_lr]


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
    """Cyclical learning rate policy (Smith, 2017), as `torch.optim.lr_scheduler.CyclicLR`: `mode` is "triangular",
    "triangular2" or "exp_range" (with `gamma`), or pass `scale_fn` and `scale_mode` ("cycle" | "iterations"). With
    `cycle_momentum` the momentum (or Adam's beta1) cycles inversely between `max_momentum` and `base_momentum`."""

    def __init__(self, optimizer: Any, base_lr: Any, max_lr: Any, step_size_up: int = 2000, step_size_down: Optional[int] = None,
                 mode: str = "triangular", gamma: float = 1.0, scale_fn: Optional[Callable[[float], float]] = None,
                 scale_mode: str = "cycle", cycle_momentum: bool = True, base_momentum: Any = 0.8, max_momentum: Any = 0.9,
                 last_epoch: int = -1):
        if step_size_up <= 0 or (step_size_down is not None and step_size_down <= 0):
            raise ValueError(f"CyclicLR: step_size_up and step_size_down must be positive, got {step_size_up} and {step_size_down}")
        if mode not in ("triangular", "triangular2", "exp_range") and scale_fn is None:
            raise ValueError("mode is invalid and scale_fn is None")
        groups = optimizer.param_groups
        n = len(groups)
        self.clr_base_lrs = _per_group(base_lr, n, "base_lr")
        self.max_lrs = _per_group(max_lr, n, "max_lr")
        for g, b in zip(groups, self.clr_base_lrs):
            g["lr"] = b
        down = float(step_size_up if step_size_down is None else step_size_down)
        self.total_size = float(step_size_up) + down
        self.step_ratio = float(step_size_up) / self.total_size
        self.mode, self.gamma = mode, gamma
        if scale_fn is None:
            self._scale_fn = {"triangular": lambda x: 1.0, "triangular2": lambda x: 1 / (2.0 ** (x - 1)), "exp_range": lambda x: gamma ** x}[mode]
            self.scale_mode = "iterations" if mode == "exp_range" else "cycle"
        else:
            self._scale_fn, self.scale_mode = scale_fn, scale_mode
        self.cycle_momentum = cycle_momentum
        if cycle_momentum:
            self._use_beta1 = _momentum_key(optimizer, "CyclicLR")
            self.base_moms = _per_group(base_momentum, n, "base_momentum")
            self.max_moms = _per_group(max_momentum, n, "max_momentum")
            for g, mm in zip(groups, self.max_moms):
                _set_momentum(g, self._use_beta1, mm)
        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> List[float]:
        cycle = math.floor(1 + self.last_epoch / self.total_size)
        x = 1.0 + self.last_epoch / self.total_size - cycle
        scale = x / self.step_ratio if x <= self.step_ratio else (x - 1) / (self.step_ratio - 1)
        factor = self._scale_fn(cycle if self.scale_mode == "cycle" else self.last_epoch)
        lrs = []
        for k, g in enumerate(self.optimizer.param_groups):
            lrs.append(self.clr_base_lrs[k] + (self.max_lrs[k] - self.clr_base_lrs[k]) * scale * factor)
            if self.cycle_momentum:
                _set_momentum(g, self._use_beta1, self.max_moms[k] - (self.max_moms[k] - self.base_moms[k]) * scale * factor)
        return lrs

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
