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
    if isinstance(value, dict):
        return {k: _copy_optim_state(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_copy_optim_state(v) for v in value]
    if isinstance(value, tuple):
        return tuple(_copy_optim_state(v) for v in value)
    return value


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
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            alpha = group["alpha"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]
            momentum = group["momentum"]
            centered = group["centered"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad.to_numpy()
                state = self.state[id(p)]

                if len(state) == 0:
                    state["square_avg"] = np.zeros_like(grad)
                    if momentum > 0:
                        state["momentum_buffer"] = np.zeros_like(grad)
                    if centered:
                        state["grad_avg"] = np.zeros_like(grad)

                square_avg = state["square_avg"]

                if weight_decay != 0:
                    grad = grad + weight_decay * p.to_numpy()

                square_avg[:] = alpha * square_avg + (1 - alpha) * (grad ** 2)

                if centered:
                    grad_avg = state["grad_avg"]
                    grad_avg[:] = alpha * grad_avg + (1 - alpha) * grad
                    avg = np.sqrt(square_avg - (grad_avg ** 2) + eps)
                else:
                    avg = np.sqrt(square_avg + eps)

                if momentum > 0:
                    buf = state["momentum_buffer"]
                    buf[:] = momentum * buf + grad / avg
                    update = buf
                else:
                    update = grad / avg

                p_data = p.to_numpy() - lr * update
                p.assign(from_numpy(np.ascontiguousarray(p_data, dtype=p_data.dtype)))

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
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            lr_decay = group["lr_decay"]
            weight_decay = group["weight_decay"]
            eps = group["eps"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad.to_numpy()
                state = self.state[id(p)]

                if len(state) == 0:
                    state["step"] = 0
                    state["sum"] = np.full_like(grad, group["initial_accumulator_value"])

                state["step"] += 1
                state_sum = state["sum"]

                if weight_decay != 0:
                    grad = grad + weight_decay * p.to_numpy()

                clr = lr / (1.0 + (state["step"] - 1) * lr_decay)
                state_sum[:] += grad ** 2
                std = np.sqrt(state_sum) + eps
                p_data = p.to_numpy() - clr * (grad / std)
                p.assign(from_numpy(np.ascontiguousarray(p_data, dtype=p_data.dtype)))

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
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            rho = group["rho"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad.to_numpy()
                state = self.state[id(p)]

                if len(state) == 0:
                    state["square_avg"] = np.zeros_like(grad)
                    state["acc_delta"] = np.zeros_like(grad)

                square_avg = state["square_avg"]
                acc_delta = state["acc_delta"]

                if weight_decay != 0:
                    grad = grad + weight_decay * p.to_numpy()

                square_avg[:] = rho * square_avg + (1 - rho) * (grad ** 2)
                std = np.sqrt(square_avg + eps)
                delta = (np.sqrt(acc_delta + eps) / std) * grad
                acc_delta[:] = rho * acc_delta + (1 - rho) * (delta ** 2)

                p_data = p.to_numpy() - lr * delta
                p.assign(from_numpy(np.ascontiguousarray(p_data, dtype=p_data.dtype)))

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
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad.to_numpy()
                state = self.state[id(p)]

                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = np.zeros_like(grad)
                    state["exp_inf"] = np.zeros_like(grad)

                state["step"] += 1
                exp_avg = state["exp_avg"]
                exp_inf = state["exp_inf"]

                if weight_decay != 0:
                    grad = grad + weight_decay * p.to_numpy()

                exp_avg[:] = beta1 * exp_avg + (1 - beta1) * grad
                norm_buf = np.maximum(beta2 * exp_inf, np.abs(grad))
                exp_inf[:] = norm_buf

                bias_correction = 1 - beta1 ** state["step"]
                clr = lr / bias_correction

                p_data = p.to_numpy() - clr * (exp_avg / (exp_inf + eps))
                p.assign(from_numpy(np.ascontiguousarray(p_data, dtype=p_data.dtype)))

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
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]
            momentum_decay = group["momentum_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad.to_numpy()
                state = self.state[id(p)]

                if len(state) == 0:
                    state["step"] = 0
                    state["m"] = np.zeros_like(grad)
                    state["v"] = np.zeros_like(grad)
                    state["mu_product"] = 1.0

                state["step"] += 1
                step = state["step"]
                m = state["m"]
                v = state["v"]

                if weight_decay != 0:
                    grad = grad + weight_decay * p.to_numpy()

                m[:] = beta1 * m + (1 - beta1) * grad
                v[:] = beta2 * v + (1 - beta2) * (grad ** 2)

                mu = beta1 * (1.0 - 0.5 * 0.96 ** (step * momentum_decay))
                mu_next = beta1 * (1.0 - 0.5 * 0.96 ** ((step + 1) * momentum_decay))
                state["mu_product"] *= mu
                prod = state["mu_product"]
                denom = np.sqrt(v / (1 - beta2 ** step)) + eps
                m_hat = mu_next * m / (1 - prod * mu_next) + (1 - mu) * grad / (1 - prod)

                p_data = p.to_numpy() - lr * m_hat / denom
                p.assign(from_numpy(np.ascontiguousarray(p_data, dtype=p_data.dtype)))

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
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]
            rho_inf = 2 / (1 - beta2) - 1

            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad.to_numpy()
                state = self.state[id(p)]

                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = np.zeros_like(grad)
                    state["exp_avg_sq"] = np.zeros_like(grad)

                state["step"] += 1
                step = state["step"]
                exp_avg = state["exp_avg"]
                exp_avg_sq = state["exp_avg_sq"]

                if weight_decay != 0:
                    grad = grad + weight_decay * p.to_numpy()

                exp_avg[:] = beta1 * exp_avg + (1 - beta1) * grad
                exp_avg_sq[:] = beta2 * exp_avg_sq + (1 - beta2) * (grad ** 2)

                beta1_t = beta1 ** step
                beta2_t = beta2 ** step
                rho_t = rho_inf - 2 * step * beta2_t / (1 - beta2_t)

                if rho_t > 4:
                    r_t = math.sqrt(
                        ((rho_t - 4) * (rho_t - 2) * rho_inf) / ((rho_inf - 4) * (rho_inf - 2) * rho_t)
                    )
                    v_hat = np.sqrt(exp_avg_sq / (1 - beta2_t)) + eps
                    step_size = lr * r_t / (1 - beta1_t)
                    update = step_size * exp_avg / v_hat
                else:
                    step_size = lr / (1 - beta1_t)
                    update = step_size * exp_avg

                p_data = p.to_numpy() - update
                p.assign(from_numpy(np.ascontiguousarray(p_data, dtype=p_data.dtype)))

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
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            lambd = group["lambd"]
            alpha = group["alpha"]
            t0 = group["t0"]
            weight_decay = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad.to_numpy()
                state = self.state[id(p)]

                if len(state) == 0:
                    state["step"] = 0
                    state["eta"] = lr
                    state["mu"] = 1.0
                    state["ax"] = p.to_numpy().copy()

                state["step"] += 1
                step = state["step"]
                ax = state["ax"]

                if weight_decay != 0:
                    grad = grad + weight_decay * p.to_numpy()

                p_data = p.to_numpy() - state["eta"] * grad
                p.assign(from_numpy(np.ascontiguousarray(p_data, dtype=p_data.dtype)))

                if state["mu"] != 1:
                    ax[:] += (p.to_numpy() - ax) * state["mu"]
                else:
                    ax[:] = p.to_numpy()

                state["eta"] = lr / ((1 + lambd * lr * step) ** alpha)
                state["mu"] = 1 / max(1, step - t0)

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

    def _gather_flat_grad(self) -> np.ndarray:
        views = []
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    view = np.zeros_like(p.to_numpy())
                else:
                    view = p.grad.to_numpy()
                views.append(view.reshape(-1))
        return np.concatenate(views)

    def _set_flat_params(self, flat_params: np.ndarray) -> None:
        offset = 0
        for group in self.param_groups:
            for p in group["params"]:
                numel = p.to_numpy().size
                arr = flat_params[offset : offset + numel].reshape(p.to_numpy().shape)
                p.assign(from_numpy(np.ascontiguousarray(arr, dtype=p.to_numpy().dtype)))
                offset += numel

    def _gather_flat_params(self) -> np.ndarray:
        views = []
        for group in self.param_groups:
            for p in group["params"]:
                views.append(p.to_numpy().reshape(-1))
        return np.concatenate(views)

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

        s_history = state["s_history"]
        y_history = state["y_history"]
        rho_history = state["rho_history"]

        loss = closure()
        flat_grad = self._gather_flat_grad()
        gnorm = float(np.linalg.norm(flat_grad, ord=np.inf))

        if gnorm <= tolerance_grad:
            return loss

        for _ in range(max_iter):
            # Two-loop recursion to compute direction d = -H * g
            q = flat_grad.copy()
            k = len(s_history)
            alphas = [0.0] * k

            for i in reversed(range(k)):
                s = s_history[i]
                y = y_history[i]
                rho = rho_history[i]
                alpha = rho * float(np.dot(s, q))
                alphas[i] = alpha
                q -= alpha * y

            if k > 0:
                gamma = float(np.dot(s_history[-1], y_history[-1])) / float(np.dot(y_history[-1], y_history[-1]) + 1e-12)
                r = gamma * q
            else:
                r = q.copy()

            for i in range(k):
                s = s_history[i]
                y = y_history[i]
                rho = rho_history[i]
                beta = rho * float(np.dot(y, r))
                r += s * (alphas[i] - beta)

            direction = -r

            # Update parameter
            prev_flat_params = self._gather_flat_params()
            new_flat_params = prev_flat_params + lr * direction
            self._set_flat_params(new_flat_params)

            # Re-evaluate
            new_loss = closure()
            new_flat_grad = self._gather_flat_grad()

            # Update histories
            s_k = new_flat_params - prev_flat_params
            y_k = new_flat_grad - flat_grad
            sy = float(np.dot(s_k, y_k))

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
            if float(np.linalg.norm(flat_grad, ord=np.inf)) <= tolerance_grad:
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
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            weight_decay = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad.to_numpy()
                state = self.state[id(p)]

                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = np.zeros_like(grad)

                state["step"] += 1
                exp_avg = state["exp_avg"]

                p_data = p.to_numpy().copy()
                if weight_decay != 0.0:
                    p_data -= lr * weight_decay * p_data

                update = np.sign(beta1 * exp_avg + (1.0 - beta1) * grad)
                p_data -= lr * update
                exp_avg[:] = beta2 * exp_avg + (1.0 - beta2) * grad

                p.assign(from_numpy(np.ascontiguousarray(p_data, dtype=p_data.dtype)))

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
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad.to_numpy()
                state = self.state[id(p)]

                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = np.zeros_like(grad)
                    state["exp_avg_sq"] = np.zeros_like(grad)

                state["step"] += 1
                step = state["step"]
                exp_avg = state["exp_avg"]
                exp_avg_sq = state["exp_avg_sq"]

                p_data = p.to_numpy().copy()
                exp_avg[:] = beta1 * exp_avg + (1.0 - beta1) * grad
                exp_avg_sq[:] = beta2 * exp_avg_sq + (1.0 - beta2) * (grad ** 2)

                bias_correction1 = 1.0 - beta1 ** step
                bias_correction2 = 1.0 - beta2 ** step

                m_hat = exp_avg / bias_correction1
                v_hat = exp_avg_sq / bias_correction2
                adam_step = m_hat / (np.sqrt(v_hat) + eps)

                if weight_decay > 0.0:
                    adam_step += weight_decay * p_data

                w_norm = float(np.linalg.norm(p_data))
                r_norm = float(np.linalg.norm(adam_step))
                trust_ratio = (w_norm / r_norm) if (w_norm > 0.0 and r_norm > 0.0) else 1.0

                p_data -= lr * trust_ratio * adam_step
                p.assign(from_numpy(np.ascontiguousarray(p_data, dtype=p_data.dtype)))

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
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            eps1, eps2 = group["eps"]
            clip_threshold = group["clip_threshold"]
            decay_rate = group["decay_rate"]
            beta1 = group["beta1"]
            weight_decay = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad.to_numpy()
                state = self.state[id(p)]

                if len(state) == 0:
                    state["step"] = 0
                    if grad.ndim > 1:
                        state["exp_avg_sq_row"] = np.zeros(grad.shape[:-1], dtype=np.float32)
                        state["exp_avg_sq_col"] = np.zeros(grad.shape[:-2] + (grad.shape[-1],), dtype=np.float32)
                    else:
                        state["exp_avg_sq"] = np.zeros_like(grad)
                    if beta1 is not None:
                        state["exp_avg"] = np.zeros_like(grad)

                state["step"] += 1
                step = state["step"]
                p_data = p.to_numpy().copy()

                beta2 = 1.0 - (step ** decay_rate)
                grad_sq = grad * grad + eps1

                if grad.ndim > 1:
                    exp_avg_sq_row = state["exp_avg_sq_row"]
                    exp_avg_sq_col = state["exp_avg_sq_col"]
                    exp_avg_sq_row[:] = beta2 * exp_avg_sq_row + (1.0 - beta2) * np.mean(grad_sq, axis=-1)
                    exp_avg_sq_col[:] = beta2 * exp_avg_sq_col + (1.0 - beta2) * np.mean(grad_sq, axis=-2)
                    r_factor = exp_avg_sq_row / np.maximum(np.mean(exp_avg_sq_row, axis=-1, keepdims=True), 1e-12)
                    v = np.expand_dims(r_factor, -1) * np.expand_dims(exp_avg_sq_col, -2)
                else:
                    exp_avg_sq = state["exp_avg_sq"]
                    exp_avg_sq[:] = beta2 * exp_avg_sq + (1.0 - beta2) * grad_sq
                    v = exp_avg_sq

                u = grad / (np.sqrt(v) + eps2)
                u_norm = float(np.linalg.norm(u))
                if clip_threshold > 0.0 and u_norm > clip_threshold:
                    u = u * (clip_threshold / u_norm)

                if beta1 is not None:
                    exp_avg = state["exp_avg"]
                    exp_avg[:] = beta1 * exp_avg + (1.0 - beta1) * u
                    update = exp_avg
                else:
                    update = u

                if weight_decay > 0.0:
                    p_data -= lr * weight_decay * p_data

                p_data -= lr * update
                p.assign(from_numpy(np.ascontiguousarray(p_data, dtype=p_data.dtype)))

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
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            eta_minus, eta_plus = group["etas"]
            step_size_min, step_size_max = group["step_sizes"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad.to_numpy()
                state = self.state[p]

                if len(state) == 0:
                    state["step"] = 0
                    state["prev_grad"] = np.zeros_like(grad)
                    state["step_size"] = np.full_like(grad, group["lr"])

                prev_grad = state["prev_grad"]
                step_size = state["step_size"]

                sign = grad * prev_grad
                step_size = np.where(
                    sign > 0,
                    np.minimum(step_size * eta_plus, step_size_max),
                    np.where(sign < 0, np.maximum(step_size * eta_minus, step_size_min), step_size),
                )
                grad = np.where(sign < 0, 0.0, grad)
                delta = -np.sign(grad) * step_size

                p_data = p.to_numpy() + delta
                p.assign(from_numpy(np.ascontiguousarray(p_data, dtype=p_data.dtype)))

                state["prev_grad"] = grad
                state["step_size"] = step_size
                state["step"] += 1

        return loss


__all__ = [
    "Optimizer",
    "RMSprop",
    "Adagrad",
    "Adadelta",
    "Adamax",
    "NAdam",
    "RAdam",
    "ASGD",
    "LBFGS",
    "Lion",
    "Lamb",
    "Adafactor",
    "Rprop",
]
