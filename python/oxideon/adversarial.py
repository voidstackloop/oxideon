"""Adversarial robustness, attack synthesis, and adversarial training for Oxideon.

Enables stress-testing model vulnerability and hardening against adversarial perturbations:
  - fgsm_attack: Fast Gradient Sign Method (Goodfellow et al., 2014)
  - pgd_attack: Projected Gradient Descent multi-step attack (Madry et al., 2018)
  - adversarial_training_step: Joint optimization on perturbed samples
"""

from typing import Optional, Callable, Any, Tuple
import numpy as np

from .oxideon import Tensor as _Tensor, from_numpy as _from_numpy, cross_entropy
from .nn import Module


def _zero_model_grads(model: Any) -> None:
    params_fn = getattr(model, "parameters", None)
    if callable(params_fn):
        for p in params_fn():
            if hasattr(p, "zero_grad"):
                p.zero_grad()
            elif hasattr(p, "grad"):
                try:
                    p.grad = None
                except AttributeError:
                    pass


def fgsm_attack(
    model: Module,
    x: _Tensor,
    y: _Tensor,
    epsilon: float = 0.03,
    loss_fn: Optional[Callable[[_Tensor, _Tensor], _Tensor]] = None,
    clip_min: Optional[float] = None,
    clip_max: Optional[float] = None,
    targeted: bool = False,
) -> _Tensor:
    """Fast Gradient Sign Method (FGSM) single-step adversarial attack."""
    if float(epsilon) < 0.0:
        raise ValueError(f"epsilon must be non-negative, got {epsilon}")
    if loss_fn is None:
        loss_fn = cross_entropy

    x_adv = x.clone().detach().requires_grad_(True)
    out = model(x_adv)
    loss = loss_fn(out, y)
    loss.backward()

    grad = x_adv.grad() if callable(x_adv.grad) else x_adv.grad
    if grad is None:
        raise RuntimeError("Gradients could not be computed for input x")

    _zero_model_grads(model)

    direction = -1.0 if targeted else 1.0
    grad_sign = np.sign(grad.to_numpy())
    perturbed = x.to_numpy() + direction * epsilon * grad_sign
    if clip_min is not None or clip_max is not None:
        perturbed = np.clip(perturbed, clip_min, clip_max)
    return _from_numpy(np.ascontiguousarray(perturbed, dtype=np.float32))


def pgd_attack(
    model: Module,
    x: _Tensor,
    y: _Tensor,
    epsilon: float = 0.03,
    alpha: float = 0.01,
    num_steps: int = 10,
    random_start: bool = True,
    loss_fn: Optional[Callable[[_Tensor, _Tensor], _Tensor]] = None,
    clip_min: Optional[float] = None,
    clip_max: Optional[float] = None,
    targeted: bool = False,
    norm: str = "linf",
    seed: Optional[int] = None,
) -> _Tensor:
    """Projected Gradient Descent (PGD) multi-step iterative adversarial attack."""
    if float(epsilon) < 0.0:
        raise ValueError(f"epsilon must be non-negative, got {epsilon}")
    if float(alpha) < 0.0:
        raise ValueError(f"alpha must be non-negative, got {alpha}")
    if num_steps < 1:
        raise ValueError(f"num_steps must be >= 1, got {num_steps}")
    norm_l = str(norm).lower()
    if norm_l not in ("linf", "inf", "l2", "2"):
        raise ValueError(f"norm must be 'linf' or 'l2', got '{norm}'")
    if loss_fn is None:
        loss_fn = cross_entropy

    rng = np.random.default_rng(seed)
    x_orig = x.to_numpy()
    if random_start and epsilon > 0:
        if norm_l in ("l2", "2"):
            raw = rng.standard_normal(x_orig.shape).astype(np.float32)
            flat = raw.reshape(len(raw), -1) if raw.ndim > 1 else raw.reshape(1, -1)
            norms = np.maximum(np.linalg.norm(flat, axis=1, keepdims=True), 1e-12)
            radii = rng.uniform(0.0, epsilon, size=(flat.shape[0], 1)).astype(np.float32)
            delta = (flat / norms * radii).reshape(x_orig.shape)
            x_curr = x_orig + delta
        else:
            delta = rng.uniform(-epsilon, epsilon, size=x_orig.shape).astype(np.float32)
            x_curr = np.clip(x_orig + delta, x_orig - epsilon, x_orig + epsilon)
    else:
        x_curr = x_orig.copy()
    if clip_min is not None or clip_max is not None:
        x_curr = np.clip(x_curr, clip_min, clip_max)

    direction = -1.0 if targeted else 1.0

    for _ in range(num_steps):
        x_tensor = _from_numpy(np.ascontiguousarray(x_curr, dtype=np.float32)).requires_grad_(True)
        out = model(x_tensor)
        loss = loss_fn(out, y)
        loss.backward()

        grad = x_tensor.grad() if callable(x_tensor.grad) else x_tensor.grad
        _zero_model_grads(model)
        if grad is None:
            break

        g_np = grad.to_numpy()
        if norm_l in ("l2", "2"):
            flat_g = g_np.reshape(len(g_np), -1) if g_np.ndim > 1 else g_np.reshape(1, -1)
            g_norms = np.maximum(np.linalg.norm(flat_g, axis=1, keepdims=True), 1e-12)
            step_vec = (flat_g / g_norms).reshape(g_np.shape)
            x_curr = x_curr + direction * alpha * step_vec
            diff = x_curr - x_orig
            flat_d = diff.reshape(len(diff), -1) if diff.ndim > 1 else diff.reshape(1, -1)
            d_norms = np.linalg.norm(flat_d, axis=1, keepdims=True)
            scale = np.minimum(1.0, epsilon / np.maximum(d_norms, 1e-12))
            x_curr = x_orig + (flat_d * scale).reshape(x_orig.shape)
        else:
            grad_sign = np.sign(g_np)
            x_curr = x_curr + direction * alpha * grad_sign
            x_curr = np.clip(x_curr, x_orig - epsilon, x_orig + epsilon)

        if clip_min is not None or clip_max is not None:
            x_curr = np.clip(x_curr, clip_min, clip_max)

    return _from_numpy(np.ascontiguousarray(x_curr, dtype=np.float32))


def adversarial_training_step(
    model: Module,
    optimizer: Any,
    x: _Tensor,
    y: _Tensor,
    epsilon: float = 0.03,
    alpha_adv: float = 0.5,
) -> float:
    """Performs single adversarial training step combining clean and adversarial loss."""
    if float(epsilon) < 0.0:
        raise ValueError(f"epsilon must be non-negative, got {epsilon}")
    if not (0.0 <= float(alpha_adv) <= 1.0):
        raise ValueError(f"alpha_adv must be in [0, 1], got {alpha_adv}")

    # Generate adversarial examples
    x_adv = fgsm_attack(model, x, y, epsilon=epsilon)
    optimizer.zero_grad()

    # Clean loss
    out_clean = model(x)
    loss_clean = cross_entropy(out_clean, y)

    # Adversarial loss
    out_adv = model(x_adv)
    loss_adv = cross_entropy(out_adv, y)

    total_loss = (1.0 - alpha_adv) * loss_clean + alpha_adv * loss_adv
    total_loss.backward()
    optimizer.step()

    return float(total_loss.to_numpy().reshape(-1)[0])


__all__ = [
    "fgsm_attack",
    "pgd_attack",
    "adversarial_training_step",
]
