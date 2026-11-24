"""Standard and advanced optimizers for Oxideon.

Implements the unified Optimizer base class alongside industrial-grade implementations of
RMSprop, Adagrad, Adadelta, Adamax, NAdam, RAdam, ASGD, and L-BFGS.
"""

from collections import defaultdict
import math
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Union
import numpy as np

from .. import Tensor, from_numpy


def _copy_optim_state(value: Any) -> Any:
    """Deep-copies optimizer state values (NumPy buffers, nested containers)."""
    if isinstance(value, np.ndarray):
        return value.copy()
    if isinstance(value, Tensor):
        return value.detach() * 1.0
    if isinstance(value, dict):
        return {k: _copy_optim_state(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_copy_optim_state(v) for v in value]
    if isinstance(value, tuple):
        return tuple(_copy_optim_state(v) for v in value)
    return value


def _zeros(p: Tensor) -> Tensor:
    """A zero state tensor with p's shape and dtype (Rust storage)."""
    return p.detach() * 0.0


def _zeros_shape(shape: Sequence[int], p: Tensor) -> Tensor:
    """A zero state tensor of `shape` in p's dtype."""
    z = from_numpy(np.zeros(tuple(shape), dtype=np.float32))
    return z if p.dtype == "f32" else z.to_dtype(p.dtype)


def _kernel(opt: "Optimizer", kind: str, p: Tensor, state: Dict[str, Any], names: Sequence[str], hyper: Dict[str, Any]) -> None:
    """Runs the Rust update (`optim_kernels.rs`) for one parameter: the state tensors `names` go in, the updated
    parameter and states come back."""
    from ..oxideon import _optim_step
    bufs = []
    for n in names:
        v = state[n]
        if isinstance(v, np.ndarray):  # state from an older (NumPy-based) checkpoint
            v = from_numpy(np.ascontiguousarray(v))
        bufs.append(v)
    new_p, new_bufs = _optim_step(kind, p.detach(), p.grad, bufs, {k: float(v) for k, v in hyper.items()})
    p.assign(new_p)
    for n, v in zip(names, new_bufs):
        state[n] = v


class Optimizer:
    """Base class for all Oxideon optimizers."""

    def __init__(self, params: Union[Iterable[Tensor], Sequence[Dict[str, Any]]], defaults: Dict[str, Any]):
        self.defaults = defaults
        self.state = defaultdict(dict)
        self.param_groups: List[Dict[str, Any]] = []

        if isinstance(params, Tensor):
            raise TypeError(
                "params argument given to the optimizer should be an iterable of Tensors or dicts, "
                "but got Tensor"
            )
        param_groups = list(params)
        if len(param_groups) == 0:
            raise ValueError("optimizer got an empty parameter list")
        if not isinstance(param_groups[0], dict):
            param_groups = [{"params": param_groups}]

        for param_group in param_groups:
            self.add_param_group(param_group)

    def __repr__(self) -> str:
        format_string = self.__class__.__name__ + " ("
        for i, group in enumerate(self.param_groups):
            format_string += f"\nParameter Group {i}\n"
            for key in sorted(group.keys()):
                if key != "params":
                    format_string += f"    {key}: {group[key]}\n"
        format_string += ")"
        return format_string

    def state_dict(self) -> Dict[str, Any]:
        """Returns the state of the optimizer as a dict (copies, not aliases)."""
        return {
            "state": {k: _copy_optim_state(dict(v)) for k, v in self.state.items()},
            "param_groups": [{**g, "params": list(g["params"])} for g in self.param_groups],
        }

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Loads the optimizer state (copies, not aliases).

        Like torch, hyper-parameters and per-param state are mapped onto the
        CURRENT params by order; the checkpoint's stored param objects are
        never adopted.
        """
        saved_groups = state_dict["param_groups"]
        if len(saved_groups) != len(self.param_groups):
            raise ValueError(
                f"loaded state dict has {len(saved_groups)} parameter groups, "
                f"but the optimizer has {len(self.param_groups)}"
            )
        for live_g, saved_g in zip(self.param_groups, saved_groups):
            if len(saved_g["params"]) != len(live_g["params"]):
                raise ValueError(
                    f"loaded state dict group has a different number of parameters "
                    f"({len(saved_g['params'])} vs {len(live_g['params'])})"
                )
        for live_g, saved_g in zip(self.param_groups, saved_groups):
            for k, v in saved_g.items():
                if k != "params":
                    live_g[k] = _copy_optim_state(v)
        live_params = [p for g in self.param_groups for p in g["params"]]
        saved_params = [p for g in saved_groups for p in g["params"]]
        saved_state = state_dict["state"]
        new_state: Dict[Any, Any] = defaultdict(dict)
        for live_p, saved_p in zip(live_params, saved_params):
            key = id(saved_p) if isinstance(saved_p, Tensor) else saved_p
            if key in saved_state:
                v = saved_state[key]
                new_state[id(live_p)] = _copy_optim_state(dict(v)) if isinstance(v, dict) else _copy_optim_state(v)
        self.state = new_state

    def _prune_state(self) -> None:
        """Drops state entries for params no longer in any group.

        State is keyed by ``id(param)``; ids can be reused after garbage
        collection, so entries for dead params must not linger where a new
        same-id tensor would inherit them.
        """
        live = {id(p) for group in self.param_groups for p in group["params"]}
        for key in [k for k in self.state if isinstance(k, int) and k not in live]:
            del self.state[key]

    def zero_grad(self, set_to_none: bool = False) -> None:
        """Sets the gradients of all optimized Tensors to zero."""
        for group in self.param_groups:
            for p in group["params"]:
                if hasattr(p, "grad") and p.grad is not None:
                    if set_to_none:
                        p.grad = None
                    elif hasattr(p.grad, "zero_"):
                        p.grad.zero_()
                    else:
                        p.grad = None

    def add_param_group(self, param_group: Dict[str, Any]) -> None:
        """Add a param group to the Optimizer's param_groups."""
        if not isinstance(param_group, dict):
            raise TypeError("param_group must be a dict")

        params = param_group["params"]
        if isinstance(params, Tensor):
            param_group["params"] = [params]
        elif isinstance(params, set):
            raise TypeError("optimizer parameters need to be organized in ordered collections, but a set was found")
        else:
            param_group["params"] = list(params)

        for p in param_group["params"]:
            if not isinstance(p, Tensor):
                raise TypeError(f"optimizer can only optimize Tensors, but one of the params is {type(p).__name__}")

        for name, default in self.defaults.items():
            param_group.setdefault(name, default)

        self.param_groups.append(param_group)
        self._prune_state()

    def step(self, closure: Optional[Callable[[], float]] = None) -> Optional[float]:
        """Performs a single optimization step."""
        raise NotImplementedError


class RMSprop(Optimizer):
    """Implements RMSprop algorithm."""

    def __init__(
        self,
        params: Union[Iterable[Tensor], Sequence[Dict[str, Any]]],
        lr: float = 1e-2,
        alpha: float = 0.99,
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        momentum: float = 0.0,
        centered: bool = False,
    ):
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= momentum:
            raise ValueError(f"Invalid momentum value: {momentum}")
        if not 0.0 <= weight_decay:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        if not 0.0 <= alpha:
            raise ValueError(f"Invalid alpha value: {alpha}")
        defaults = dict(lr=lr, alpha=alpha, eps=eps, weight_decay=weight_decay, momentum=momentum, centered=centered)
        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable[[], float]] = None) -> Optional[float]:
        loss = closure() if closure is not None else None
        for group in self.param_groups:
            hyper = dict(lr=group["lr"], alpha=group["alpha"], eps=group["eps"], weight_decay=group["weight_decay"],
                         momentum=group["momentum"], centered=float(bool(group["centered"])))
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[id(p)]
                if len(state) == 0:
                    state["square_avg"] = _zeros(p)
                    state["momentum_buffer"] = _zeros(p)
                    state["grad_avg"] = _zeros(p)
                _kernel(self, "rmsprop", p, state, ("square_avg", "momentum_buffer", "grad_avg"), hyper)
        return loss


class Adagrad(Optimizer):
    """Implements Adagrad algorithm."""

    def __init__(
        self,
        params: Union[Iterable[Tensor], Sequence[Dict[str, Any]]],
        lr: float = 1e-2,
        lr_decay: float = 0.0,
        weight_decay: float = 0.0,
        initial_accumulator_value: float = 0.0,
        eps: float = 1e-10,
    ):
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= lr_decay:
            raise ValueError(f"Invalid lr_decay value: {lr_decay}")
        if not 0.0 <= weight_decay:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        if not 0.0 <= initial_accumulator_value:
            raise ValueError(f"Invalid initial_accumulator_value value: {initial_accumulator_value}")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps}")
        defaults = dict(
            lr=lr,
            lr_decay=lr_decay,
            weight_decay=weight_decay,
            initial_accumulator_value=initial_accumulator_value,
            eps=eps,
        )
        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable[[], float]] = None) -> Optional[float]:
        loss = closure() if closure is not None else None
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[id(p)]
                if len(state) == 0:
                    state["step"] = 0
                    state["sum"] = _zeros(p) + float(group["initial_accumulator_value"])
                state["step"] += 1
                clr = group["lr"] / (1.0 + (state["step"] - 1) * group["lr_decay"])
                _kernel(self, "adagrad", p, state, ("sum",), dict(clr=clr, eps=group["eps"], weight_decay=group["weight_decay"]))
        return loss


class Adadelta(Optimizer):
    """Implements Adadelta algorithm."""

    def __init__(
        self,
        params: Union[Iterable[Tensor], Sequence[Dict[str, Any]]],
        lr: float = 1.0,
        rho: float = 0.9,
        eps: float = 1e-6,
        weight_decay: float = 0.0,
    ):
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= rho <= 1.0:
            raise ValueError(f"Invalid rho value: {rho}")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= weight_decay:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        defaults = dict(lr=lr, rho=rho, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable[[], float]] = None) -> Optional[float]:
        loss = closure() if closure is not None else None
        for group in self.param_groups:
            hyper = dict(lr=group["lr"], rho=group["rho"], eps=group["eps"], weight_decay=group["weight_decay"])
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[id(p)]
                if len(state) == 0:
                    state["square_avg"] = _zeros(p)
                    state["acc_delta"] = _zeros(p)
                _kernel(self, "adadelta", p, state, ("square_avg", "acc_delta"), hyper)
        return loss


class Adamax(Optimizer):
    """Implements Adamax algorithm (a variant of Adam based on the infinity norm)."""

    def __init__(
        self,
        params: Union[Iterable[Tensor], Sequence[Dict[str, Any]]],
        lr: float = 2e-3,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
    ):
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        if not 0.0 <= weight_decay:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable[[], float]] = None) -> Optional[float]:
        loss = closure() if closure is not None else None
        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[id(p)]
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = _zeros(p)
                    state["exp_inf"] = _zeros(p)
                state["step"] += 1
                _kernel(self, "adamax", p, state, ("exp_avg", "exp_inf"),
                        dict(lr=group["lr"], beta1=beta1, beta2=beta2, eps=group["eps"], weight_decay=group["weight_decay"], step=state["step"]))
        return loss


class NAdam(Optimizer):
    """Implements NAdam algorithm (Nesterov-accelerated Adaptive Moment Estimation)."""

    def __init__(
        self,
        params: Union[Iterable[Tensor], Sequence[Dict[str, Any]]],
        lr: float = 2e-3,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        momentum_decay: float = 0.004,
    ):
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        if not 0.0 <= weight_decay:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        if not 0.0 <= momentum_decay:
            raise ValueError(f"Invalid momentum_decay value: {momentum_decay}")
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, momentum_decay=momentum_decay)
        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable[[], float]] = None) -> Optional[float]:
        loss = closure() if closure is not None else None
        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            momentum_decay = group["momentum_decay"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[id(p)]
                if len(state) == 0:
                    state["step"] = 0
                    state["m"] = _zeros(p)
                    state["v"] = _zeros(p)
                    state["mu_product"] = 1.0
                state["step"] += 1
                step = state["step"]
                mu = beta1 * (1.0 - 0.5 * 0.96 ** (step * momentum_decay))
                mu_next = beta1 * (1.0 - 0.5 * 0.96 ** ((step + 1) * momentum_decay))
                state["mu_product"] *= mu
                _kernel(self, "nadam", p, state, ("m", "v"),
                        dict(lr=group["lr"], beta1=beta1, beta2=beta2, eps=group["eps"], weight_decay=group["weight_decay"],
                             step=step, mu=mu, mu_next=mu_next, mu_product=state["mu_product"]))
        return loss


class RAdam(Optimizer):
    """Implements Rectified Adam (RAdam) algorithm."""

    def __init__(
        self,
        params: Union[Iterable[Tensor], Sequence[Dict[str, Any]]],
        lr: float = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
    ):
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        if not 0.0 <= weight_decay:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable[[], float]] = None) -> Optional[float]:
        loss = closure() if closure is not None else None
        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[id(p)]
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = _zeros(p)
                    state["exp_avg_sq"] = _zeros(p)
                state["step"] += 1
                _kernel(self, "radam", p, state, ("exp_avg", "exp_avg_sq"),
                        dict(lr=group["lr"], beta1=beta1, beta2=beta2, eps=group["eps"], weight_decay=group["weight_decay"], step=state["step"]))
        return loss


class ASGD(Optimizer):
    """Implements Averaged Stochastic Gradient Descent (ASGD)."""

    def __init__(
        self,
        params: Union[Iterable[Tensor], Sequence[Dict[str, Any]]],
        lr: float = 1e-2,
        lambd: float = 1e-4,
        alpha: float = 0.75,
        t0: float = 1e6,
        weight_decay: float = 0.0,
    ):
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= lambd:
            raise ValueError(f"Invalid lambd value: {lambd}")
        if not 0.0 <= alpha:
            raise ValueError(f"Invalid alpha value: {alpha}")
        if not 0.0 <= weight_decay:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        defaults = dict(lr=lr, lambd=lambd, alpha=alpha, t0=t0, weight_decay=weight_decay)
        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable[[], float]] = None) -> Optional[float]:
        loss = closure() if closure is not None else None
        for group in self.param_groups:
            lr, lambd, alpha, t0 = group["lr"], group["lambd"], group["alpha"], group["t0"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[id(p)]
                if len(state) == 0:
                    state["step"] = 0
                    state["eta"] = lr
                    state["mu"] = 1.0
                    state["ax"] = p.detach() * 1.0
                state["step"] += 1
                _kernel(self, "asgd", p, state, ("ax",),
                        dict(eta=state["eta"], mu=state["mu"], lambd=lambd, weight_decay=group["weight_decay"]))
                state["eta"] = lr / ((1 + lambd * lr * state["step"]) ** alpha)
                state["mu"] = 1 / max(1, state["step"] - t0)
        return loss


class LBFGS(Optimizer):
    """Implements L-BFGS algorithm with two-loop recursion (fixed step size; no line search)."""

    def __init__(
        self,
        params: Union[Iterable[Tensor], Sequence[Dict[str, Any]]],
        lr: float = 1.0,
        max_iter: int = 20,
        max_eval: Optional[int] = None,
        tolerance_grad: float = 1e-7,
        tolerance_change: float = 1e-9,
        history_size: int = 100,
        line_search_fn: Optional[str] = None,
    ):
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if isinstance(max_iter, bool) or not isinstance(max_iter, int) or max_iter < 1:
            raise ValueError(f"LBFGS: max_iter must be a positive integer, got {max_iter!r}")
        if isinstance(history_size, bool) or not isinstance(history_size, int) or history_size < 1:
            raise ValueError(f"LBFGS: history_size must be a positive integer, got {history_size!r}")
        if line_search_fn is not None:
            raise ValueError(f"LBFGS: line_search_fn={line_search_fn!r} is not supported (fixed step size; no line search)")
        if max_eval is None:
            max_eval = max_iter * 5 // 4
        defaults = dict(
            lr=lr,
            max_iter=max_iter,
            max_eval=max_eval,
            tolerance_grad=tolerance_grad,
            tolerance_change=tolerance_change,
            history_size=history_size,
            line_search_fn=line_search_fn,
        )
        super().__init__(params, defaults)
        if len(self.param_groups) != 1:
            raise ValueError("LBFGS doesn't support per-parameter options (parameter groups)")

    # The flattened parameters, gradients and history vectors are oxideon tensors, so every vector operation of the
    # two-loop recursion (dot products, axpy updates, norms) runs in Rust.
    def _params(self) -> List[Tensor]:
        return [p for group in self.param_groups for p in group["params"]]

    def _gather_flat_grad(self) -> Tensor:
        from ..oxideon import cat
        views = [(p.grad.detach() if p.grad is not None else p.detach() * 0.0).reshape([-1]) for p in self._params()]
        return cat(views, 0)

    def _set_flat_params(self, flat_params: Tensor) -> None:
        offset = 0
        for p in self._params():
            numel = int(p.numel())
            p.assign(flat_params.narrow(0, offset, numel).reshape(list(p.shape)).to_dtype(p.dtype))
            offset += numel

    def _gather_flat_params(self) -> Tensor:
        from ..oxideon import cat
        return cat([p.detach().reshape([-1]) for p in self._params()], 0)

    @staticmethod
    def _dot(a: Tensor, b: Tensor) -> float:
        return float((a * b).sum().item())

    @staticmethod
    def _max_abs(a: Tensor) -> float:
        return float(a.abs().max().item()) if a.numel() else 0.0

    def step(self, closure: Callable[[], float]) -> float:
        if closure is None:
            raise RuntimeError("L-BFGS requires a closure that clears gradients and re-evaluates the loss")

        group = self.param_groups[0]
        lr = group["lr"]
        max_iter = group["max_iter"]
        history_size = group["history_size"]
        tolerance_grad = group["tolerance_grad"]
        tolerance_change = group["tolerance_change"]

        # Global (flat) L-BFGS history: string key so _prune_state never drops it.
        state = self.state["lbfgs"]
        if len(state) == 0:
            state["s_history"] = []
            state["y_history"] = []
            state["rho_history"] = []
        # histories saved by the NumPy-based version
        for key in ("s_history", "y_history"):
            state[key] = [from_numpy(np.ascontiguousarray(v)) if isinstance(v, np.ndarray) else v for v in state[key]]

        s_history = state["s_history"]
        y_history = state["y_history"]
        rho_history = state["rho_history"]

        loss = closure()
        flat_grad = self._gather_flat_grad()
        if self._max_abs(flat_grad) <= tolerance_grad:
            return loss

        for _ in range(max_iter):
            # Two-loop recursion to compute direction d = -H * g
            q = flat_grad * 1.0
            k = len(s_history)
            alphas = [0.0] * k
            for i in reversed(range(k)):
                alpha = rho_history[i] * self._dot(s_history[i], q)
                alphas[i] = alpha
                q = q - y_history[i] * alpha
            if k > 0:
                gamma = self._dot(s_history[-1], y_history[-1]) / (self._dot(y_history[-1], y_history[-1]) + 1e-12)
                r = q * gamma
            else:
                r = q
            for i in range(k):
                beta = rho_history[i] * self._dot(y_history[i], r)
                r = r + s_history[i] * (alphas[i] - beta)
            direction = r * -1.0

            prev_flat_params = self._gather_flat_params()
            new_flat_params = prev_flat_params + direction * lr
            self._set_flat_params(new_flat_params)

            new_loss = closure()
            new_flat_grad = self._gather_flat_grad()

            s_k = new_flat_params - prev_flat_params
            y_k = new_flat_grad - flat_grad
            sy = self._dot(s_k, y_k)
            if sy > 1e-10:
                if len(s_history) >= history_size:
                    s_history.pop(0)
                    y_history.pop(0)
                    rho_history.pop(0)
                s_history.append(s_k)
                y_history.append(y_k)
                rho_history.append(1.0 / sy)

            if abs(new_loss - loss) < tolerance_change:
                return new_loss
            loss, flat_grad = new_loss, new_flat_grad
            if self._max_abs(flat_grad) <= tolerance_grad:
                return loss

        return loss


class Lion(Optimizer):
    """Implements Lion algorithm (EvoLved Sign Momentum, Chen et al. 2023)."""

    def __init__(
        self,
        params: Union[Iterable[Tensor], Sequence[Dict[str, Any]]],
        lr: float = 1e-4,
        betas: Tuple[float, float] = (0.9, 0.99),
        weight_decay: float = 0.0,
    ):
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        if not 0.0 <= weight_decay:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")

        defaults = dict(lr=lr, betas=betas, weight_decay=weight_decay)
        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable[[], float]] = None) -> Optional[float]:
        loss = closure() if closure is not None else None
        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            hyper = dict(lr=group["lr"], beta1=beta1, beta2=beta2, weight_decay=group["weight_decay"])
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[id(p)]
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = _zeros(p)
                state["step"] += 1
                _kernel(self, "lion", p, state, ("exp_avg",), hyper)
        return loss


class Lamb(Optimizer):
    """Implements LAMB algorithm (You et al. 2020)."""

    def __init__(
        self,
        params: Union[Iterable[Tensor], Sequence[Dict[str, Any]]],
        lr: float = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-6,
        weight_decay: float = 0.0,
    ):
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        if not 0.0 <= weight_decay:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")

        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable[[], float]] = None) -> Optional[float]:
        loss = closure() if closure is not None else None
        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[id(p)]
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = _zeros(p)
                    state["exp_avg_sq"] = _zeros(p)
                state["step"] += 1
                _kernel(self, "lamb", p, state, ("exp_avg", "exp_avg_sq"),
                        dict(lr=group["lr"], beta1=beta1, beta2=beta2, eps=group["eps"], weight_decay=group["weight_decay"], step=state["step"]))
        return loss


class Adafactor(Optimizer):
    """Implements Adafactor algorithm (Shazeer & Stern 2018)."""

    def __init__(
        self,
        params: Union[Iterable[Tensor], Sequence[Dict[str, Any]]],
        lr: Optional[float] = None,
        eps: Tuple[float, float] = (1e-30, 1e-3),
        clip_threshold: float = 1.0,
        decay_rate: float = -0.8,
        beta1: Optional[float] = None,
        weight_decay: float = 0.0,
        scale_parameter: bool = True,
        relative_step: bool = True,
        warmup_init: bool = False,
    ):
        if lr is not None and lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")

        actual_lr = 1e-2 if lr is None else lr
        defaults = dict(
            lr=actual_lr,
            eps=eps,
            clip_threshold=clip_threshold,
            decay_rate=decay_rate,
            beta1=beta1,
            weight_decay=weight_decay,
            scale_parameter=scale_parameter,
            relative_step=relative_step,
            warmup_init=warmup_init,
        )
        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable[[], float]] = None) -> Optional[float]:
        loss = closure() if closure is not None else None
        for group in self.param_groups:
            eps1, eps2 = group["eps"]
            beta1 = group["beta1"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[id(p)]
                shape = tuple(p.shape)
                if len(state) == 0:
                    state["step"] = 0
                    if len(shape) > 1:
                        state["exp_avg_sq_row"] = _zeros_shape(shape[:-1], p)
                        state["exp_avg_sq_col"] = _zeros_shape(shape[:-2] + (shape[-1],), p)
                    else:
                        state["exp_avg_sq"] = _zeros(p)
                    if beta1 is not None:
                        state["exp_avg"] = _zeros(p)
                state["step"] += 1
                names = ("exp_avg_sq_row", "exp_avg_sq_col") if len(shape) > 1 else ("exp_avg_sq",)
                if beta1 is not None:
                    names = names + ("exp_avg",)
                _kernel(self, "adafactor", p, state, names,
                        dict(lr=group["lr"], beta2=1.0 - state["step"] ** group["decay_rate"], eps1=eps1, eps2=eps2,
                             clip_threshold=group["clip_threshold"], beta1=-1.0 if beta1 is None else beta1,
                             weight_decay=group["weight_decay"]))
        return loss


class Rprop(Optimizer):
    """Implements the resilient backpropagation algorithm."""
    def __init__(
        self,
        params,
        lr: float = 1e-2,
        etas: Tuple[float, float] = (0.5, 1.2),
        step_sizes: Tuple[float, float] = (1e-6, 50.0),
    ):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not (0.0 < etas[0] < 1.0 < etas[1]):
            raise ValueError(f"Invalid eta values: {etas[0]}, {etas[1]}")
        defaults = dict(lr=lr, etas=etas, step_sizes=step_sizes)
        super().__init__(params, defaults)

    def step(self, closure=None):
        loss = closure() if closure is not None else None
        for group in self.param_groups:
            eta_minus, eta_plus = group["etas"]
            step_size_min, step_size_max = group["step_sizes"]
            hyper = dict(eta_minus=eta_minus, eta_plus=eta_plus, step_size_min=step_size_min, step_size_max=step_size_max)
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[id(p)]  # keyed by id like every optimizer, so state_dict round-trips keep it
                if len(state) == 0:
                    state["step"] = 0
                    state["prev_grad"] = _zeros(p)
                    state["step_size"] = _zeros(p) + float(group["lr"])
                _kernel(self, "rprop", p, state, ("prev_grad", "step_size"), hyper)
                state["step"] += 1
        return loss


class SparseAdam(Optimizer):
    """Adam for parameters whose gradients touch few rows (embedding tables): each step updates the moments and the
    values only for the rows whose gradient is non-zero; untouched rows keep their moments and values (a "lazy" Adam,
    as `torch.optim.SparseAdam` does with sparse gradients). A row is a slice along dimension 0 (an element for 1-D
    parameters)."""

    def __init__(self, params: Union[Iterable[Tensor], Sequence[Dict[str, Any]]], lr: float = 1e-3,
                 betas: Tuple[float, float] = (0.9, 0.999), eps: float = 1e-8, maximize: bool = False):
        if not 0.0 < lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 < eps:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        super().__init__(params, dict(lr=lr, betas=betas, eps=eps, maximize=maximize))

    def step(self, closure: Optional[Callable[[], float]] = None) -> Optional[float]:
        from ..oxideon import _sparse_adam_step
        loss = closure() if closure is not None else None
        for group in self.param_groups:
            lr, (beta1, beta2), eps = group["lr"], group["betas"], group["eps"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[id(p)]
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = p.detach() * 0.0
                    state["exp_avg_sq"] = p.detach() * 0.0
                state["step"] += 1
                new_p, state["exp_avg"], state["exp_avg_sq"] = _sparse_adam_step(
                    p.detach(), p.grad, state["exp_avg"], state["exp_avg_sq"], state["step"], lr, beta1, beta2, eps,
                    bool(group["maximize"]))
                p.assign(new_p)
        return loss
