"""
Evidential Deep Learning and Uncertainty Decomposition for Oxideon.
Includes differentiable Dirichlet prior loss (Sensoy et al., NeurIPS 2018),
information-theoretic decomposition into aleatoric and epistemic uncertainty,
Brier scoring, and Expected Calibration Error (ECE).
"""
from typing import Tuple
import math
import numpy as np
from .oxideon import Tensor, from_numpy as _from_numpy


def _digamma_tensor(x: Tensor) -> Tensor:
    """Differentiable asymptotic digamma approximation psi(x) approx ln(x) - 1/(2x) - 1/(12x^2) for x >= 1."""
    x_safe = x.clamp(1e-6, 1e12)
    inv_x = x_safe.pow(-1.0)
    return x_safe.log() - inv_x * 0.5 - (inv_x * inv_x) * (1.0 / 12.0)


def _lgamma_tensor(x: Tensor) -> Tensor:
    """Differentiable Stirling log-gamma approximation for x >= 1."""
    x_safe = x.clamp(1e-6, 1e12)
    half_ln_2pi = 0.5 * math.log(2.0 * math.pi)
    return (x_safe - 0.5) * x_safe.log() - x_safe + half_ln_2pi + x_safe.pow(-1.0) * (1.0 / 12.0)


class DirichletLoss:
    """
    Differentiable Evidential Deep Learning classification loss (Sensoy et al., NeurIPS 2018).
    Places a Dirichlet prior over class probabilities:
      alpha_k = relu(logits_k) + 1
      S = sum_k alpha_k
      L_data = sum_k y_k * (psi(S) - psi(alpha_k))
    """
    def __init__(self, kl_weight: float = 0.01):
        if kl_weight < 0.0:
            raise ValueError(f"kl_weight cannot be negative, got {kl_weight}")
        self.kl_weight = float(kl_weight)

    def __call__(self, logits: Tensor, targets_onehot: Tensor) -> Tensor:
        if list(logits.shape) != list(targets_onehot.shape):
            raise ValueError(f"Shape mismatch: logits {logits.shape} vs targets {targets_onehot.shape}")
        if logits.shape[-1] == 0:
            raise ValueError("Logits must have at least 1 class")

        alpha = logits.relu() + 1.0
        s_sum = alpha.sum(dim=-1, keepdim=True)

        l_data = (targets_onehot * (_digamma_tensor(s_sum) - _digamma_tensor(alpha))).sum(dim=-1)

        # Regularizer towards flat Dirichlet prior Dir(1)
        one_minus_y = (targets_onehot * -1.0) + 1.0
        alpha_tilde = targets_onehot + one_minus_y * alpha
        s_tilde = alpha_tilde.sum(dim=-1, keepdim=True)
        k_classes = alpha.shape[-1]

        ln_gamma_s = _lgamma_tensor(s_tilde).squeeze(-1)
        ln_gamma_alpha = _lgamma_tensor(alpha_tilde).sum(dim=-1)
        ln_gamma_k = math.lgamma(k_classes)
        kl_first = (ln_gamma_s - ln_gamma_alpha) - ln_gamma_k
        kl_second = ((alpha_tilde - 1.0) * (_digamma_tensor(alpha_tilde) - _digamma_tensor(s_tilde))).sum(dim=-1)
        kl = (kl_first + kl_second).relu()

        return (l_data + kl * self.kl_weight).mean()


def decompose_uncertainty(ensemble_probs: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
    """
    Decomposes ensemble predictions into total entropy, expected data uncertainty (aleatoric),
    and epistemic mutual information (knowledge uncertainty).
    """
    probs = ensemble_probs.to_numpy()
    if probs.ndim == 2:
        probs = np.expand_dims(probs, axis=1)
    elif probs.ndim != 3:
        raise ValueError(f"ensemble_probs must be 2D (M, K) or 3D (M, B, K), got shape {probs.shape}")
    m_members, _, _ = probs.shape
    if m_members == 0:
        raise ValueError("ensemble_probs must contain at least 1 ensemble member (M >= 1)")

    mean_p = np.mean(probs, axis=0)
    total_entropy = -np.sum(mean_p * np.log(mean_p + 1e-9), axis=-1)

    individual_entropies = -np.sum(probs * np.log(probs + 1e-9), axis=-1)
    aleatoric = np.mean(individual_entropies, axis=0)
    epistemic = np.maximum(0.0, total_entropy - aleatoric)

    return (
        _from_numpy(total_entropy.astype(np.float32)),
        _from_numpy(aleatoric.astype(np.float32)),
        _from_numpy(epistemic.astype(np.float32)),
    )


def brier_score(probs: Tensor, targets_onehot: Tensor) -> float:
    """Computes standard multi-class Brier score: (1 / N) * sum_i ||p_i - y_i||_2^2."""
    p = probs.to_numpy()
    y = targets_onehot.to_numpy()
    if p.shape != y.shape:
        raise ValueError(f"Shape mismatch: probs {p.shape} vs targets {y.shape}")
    if p.size == 0:
        raise ValueError("probs and targets cannot be empty")
    score = np.mean(np.sum((p - y) ** 2, axis=-1))
    return float(score)


def expected_calibration_error(probs: Tensor, targets: Tensor, n_bins: int = 10) -> float:
    """
    Computes Expected Calibration Error (ECE) over `n_bins` confidence intervals.
    Accepts either class indices `(N,)` or one-hot targets `(N, K)`.
    """
    if n_bins < 1:
        raise ValueError("n_bins must be >= 1")
    p = np.asarray(probs.to_numpy(), dtype=np.float64)
    t = np.asarray(targets.to_numpy())
    if p.ndim != 2 or p.shape[0] == 0:
        return 0.0
    if t.ndim == 2:
        labels = np.argmax(t, axis=-1)
    else:
        labels = t.reshape(-1).astype(np.int64)

    confidences = np.max(p, axis=-1)
    predictions = np.argmax(p, axis=-1)
    accuracies = (predictions == labels).astype(np.float64)

    bin_boundaries = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(confidences)
    for i in range(n_bins):
        lo, hi = bin_boundaries[i], bin_boundaries[i + 1]
        if i == 0:
            mask = (confidences >= lo) & (confidences <= hi)
        else:
            mask = (confidences > lo) & (confidences <= hi)
        count = int(np.sum(mask))
        if count > 0:
            avg_conf = float(np.mean(confidences[mask]))
            avg_acc = float(np.mean(accuracies[mask]))
            ece += (count / n) * abs(avg_acc - avg_conf)
    return float(ece)
