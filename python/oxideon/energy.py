"""
Energy-Based Models (EBMs) and Langevin Dynamics for Oxideon.
Includes exact-autograd SGLD sampling, Contrastive Divergence, and Free Energy scoring for OOD detection.
"""
from typing import Callable, Optional, Tuple
import numpy as np
from .oxideon import Tensor, from_numpy as _from_numpy
from .nn import Module


class LangevinSampler:
    """
    Stochastic Gradient Langevin Dynamics (SGLD) sampler for unnormalized energy models.
    Generates samples x ~ p(x) proportional to exp(-E(x)) using exact reverse-mode autograd
    (falling back to coordinate-wise central differences for non-differentiable black-box functions).
    """
    def __init__(
        self,
        step_size: float = 0.01,
        num_steps: int = 50,
        noise_scale: float = 0.005,
        clip_grad: float = 1.0,
    ):
        if num_steps < 1:
            raise ValueError(f"num_steps must be >= 1, got {num_steps}")
        if step_size <= 0:
            raise ValueError(f"step_size must be > 0, got {step_size}")
        self.step_size = step_size
        self.num_steps = num_steps
        self.noise_scale = noise_scale
        self.clip_grad = clip_grad

    def _compute_grad(self, energy_fn: Callable[[Tensor], Tensor], x_curr: np.ndarray) -> np.ndarray:
        x_t = _from_numpy(x_curr.astype(np.float32)).requires_grad_(True)
        e_out = energy_fn(x_t)
        e_sum = e_out.sum() if e_out.numel() > 1 else e_out
        if getattr(e_sum, "requires_grad", False):
            e_sum.backward()
            if x_t.grad is not None:
                return x_t.grad.to_numpy().astype(np.float32)

        # Fallback: coordinate-wise central finite difference
        eps = 1e-4
        flat = x_curr.reshape(-1).astype(np.float32)
        grad_flat = np.zeros_like(flat)
        for i in range(len(flat)):
            xp = flat.copy()
            xm = flat.copy()
            xp[i] += eps
            xm[i] -= eps
            ep = float(np.sum(energy_fn(_from_numpy(xp.reshape(x_curr.shape))).to_numpy()))
            em = float(np.sum(energy_fn(_from_numpy(xm.reshape(x_curr.shape))).to_numpy()))
            grad_flat[i] = (ep - em) / (2.0 * eps)
        return grad_flat.reshape(x_curr.shape)

    def sample(self, energy_fn: Callable[[Tensor], Tensor], init_x: Tensor) -> Tensor:
        x_curr = init_x.to_numpy().astype(np.float32).copy()

        for _ in range(self.num_steps):
            grad = self._compute_grad(energy_fn, x_curr)
            if self.clip_grad > 0:
                norm = float(np.linalg.norm(grad))
                if norm > self.clip_grad:
                    grad = grad * (self.clip_grad / norm)

            noise = (
                np.random.randn(*x_curr.shape).astype(np.float32) * self.noise_scale
                if self.noise_scale > 0.0
                else 0.0
            )
            x_curr = x_curr - 0.5 * self.step_size * grad + noise

        return _from_numpy(x_curr.astype(np.float32))


class ContrastiveDivergenceLoss:
    """
    Differentiable Contrastive Divergence loss for training Energy-Based Models:
    L = E(x_pos) - E(x_neg) + alpha * (E(x_pos)^2 + E(x_neg)^2)
    """
    def __init__(self, reg_weight: float = 0.1):
        self.reg_weight = float(reg_weight)

    def __call__(self, pos_energy: Tensor, neg_energy: Tensor) -> Tensor:
        if pos_energy.numel() == 0 or neg_energy.numel() == 0:
            raise ValueError("ContrastiveDivergenceLoss requires non-empty energy samples")

        loss = pos_energy.mean() - neg_energy.mean()
        reg = ((pos_energy * pos_energy).mean() + (neg_energy * neg_energy).mean()) * self.reg_weight
        return loss + reg


class FreeEnergyScorer:
    """
    Computes differentiable Helmholtz free energy E(x; T) = -T * logsumexp(logits / T).
    Lower energy corresponds to higher in-distribution density; higher energy flags OOD samples.
    """
    def __init__(self, temperature: float = 1.0):
        if temperature <= 0.0:
            raise ValueError(f"temperature must be > 0, got {temperature}")
        self.temperature = float(temperature)

    def compute_energy(self, logits: Tensor) -> Tensor:
        if len(logits.shape) <= 1:
            if len(logits.shape) == 0 or logits.shape[-1] == 0:
                return Tensor.zeros([0])
            scaled = logits * (1.0 / self.temperature)
            m = float(np.max(scaled.to_numpy()))
            lse = ((scaled - m).exp().sum() + 1e-12).log() + m
            return (lse * (-self.temperature)).reshape([1])
        if logits.shape[-1] == 0:
            return Tensor.zeros(list(logits.shape[:-1]))
        scaled = logits * (1.0 / self.temperature)
        m = float(np.max(scaled.to_numpy()))
        lse = ((scaled - m).exp().sum(dim=-1) + 1e-12).log() + m
        return lse * (-self.temperature)
