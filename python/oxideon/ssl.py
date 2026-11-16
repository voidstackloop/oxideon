"""Semi-Supervised Learning (SSL) and Self-Supervised Representation Learning for Oxideon.

Implements state-of-the-art pseudo-labeling, consistency regularization, and contrastive methods:
  - PseudoLabelLoss: Confidence-thresholded pseudo-label cross-entropy (FixMatch / Lee, 2013)
  - ConsistencyLoss: Unsupervised prediction invariance under perturbation
  - MeanTeacher: Exponential Moving Average (EMA) teacher model coordinator (Tarvainen et al., 2017)
  - SimCLRLoss: Normalized Temperature-scaled Cross Entropy (NT-Xent) contrastive loss (Chen et al., 2020)
"""

from typing import Tuple, Optional, Any, List, Dict
import numpy as np

from .oxideon import (
    Tensor as _Tensor,
    from_numpy as _from_numpy,
    cross_entropy,
    cat,
)
from .functional import softmax, log_softmax
from .nn import Module, Parameter


class PseudoLabelLoss(Module):
    """FixMatch-style confidence-thresholded pseudo-label loss on unlabeled predictions."""

    def __init__(self, threshold: float = 0.95):
        super().__init__()
        if not (0.0 <= threshold <= 1.0):
            raise ValueError(f"threshold must be between 0 and 1, got {threshold}")
        self.threshold = threshold

    def forward(self, unlabeled_logits: _Tensor) -> Tuple[_Tensor, float]:
        """Calculates loss on samples where max predicted probability exceeds threshold.

        Returns: (loss_tensor, mask_ratio).
        """
        arr = unlabeled_logits.to_numpy()
        is_1d = arr.ndim == 1
        logits_in = unlabeled_logits.unsqueeze(0) if is_1d else unlabeled_logits

        probs = softmax(logits_in, dim=-1).to_numpy()
        max_probs = np.max(probs, axis=-1)
        pseudo_labels = np.argmax(probs, axis=-1)

        mask = max_probs >= self.threshold
        mask_ratio = float(np.mean(mask)) if len(mask) > 0 else 0.0

        if not np.any(mask):
            return _from_numpy(np.array([0.0], dtype=np.float32)), 0.0

        valid_indices = np.where(mask)[0]
        if getattr(unlabeled_logits, "requires_grad", False):
            selected_rows = [logits_in[int(idx)].unsqueeze(0) for idx in valid_indices]
            masked_logits = cat(selected_rows, dim=0) if len(selected_rows) > 1 else selected_rows[0]
        else:
            masked_logits = _from_numpy(np.ascontiguousarray(logits_in.to_numpy()[mask]))

        masked_labels = _from_numpy(np.ascontiguousarray(pseudo_labels[mask], dtype=np.int64))

        loss = cross_entropy(masked_logits, masked_labels)
        return loss, mask_ratio


class ConsistencyLoss(Module):
    """Consistency regularization loss enforcing prediction invariance under perturbation."""

    def __init__(self, metric: str = "mse"):
        super().__init__()
        if metric not in ("mse", "kl"):
            raise ValueError(f"metric must be 'mse' or 'kl', got '{metric}'")
        self.metric = metric

    def forward(self, pred1: _Tensor, pred2: _Tensor) -> _Tensor:
        if pred1.shape != pred2.shape:
            raise ValueError(f"Shape mismatch: pred1 shape {pred1.shape} != pred2 shape {pred2.shape}")
        if self.metric == "mse":
            p1 = softmax(pred1, dim=-1)
            p2 = softmax(pred2, dim=-1)
            diff = p1 - p2
            return (diff * diff).mean()
        else:
            # Differentiable KL divergence
            p1_det = softmax(pred1.detach(), dim=-1)
            log_p1 = log_softmax(pred1.detach(), dim=-1)
            log_p2 = log_softmax(pred2, dim=-1)
            kl = (p1_det * (log_p1 - log_p2)).sum(dim=-1).mean()
            return kl


class MeanTeacher:
    """Maintains Exponential Moving Average (EMA) teacher model parameters."""

    def __init__(self, student: Module, alpha: float = 0.99):
        if not (0.0 <= alpha <= 1.0):
            raise ValueError(f"alpha must be in [0, 1], got {alpha}")
        self.student = student
        self.alpha = alpha
        self.teacher_params: Dict[str, np.ndarray] = {
            name: p.to_numpy().copy() for name, p in student.named_parameters()
        }

    def update(self, student: Module):
        """Updates teacher parameters: theta_t = alpha * theta_t + (1 - alpha) * theta_s."""
        s_params = dict(student.named_parameters())
        for name, s_param in s_params.items():
            if name in self.teacher_params:
                t_arr = self.teacher_params[name]
                s_arr = s_param.to_numpy()
                self.teacher_params[name] = self.alpha * t_arr + (1.0 - self.alpha) * s_arr

    def apply_to(self, model: Module):
        """Applies EMA teacher parameters to the provided model."""
        for name, arr in self.teacher_params.items():
            parts = name.split(".")
            curr = model
            for p in parts[:-1]:
                curr = getattr(curr, p)
            setattr(curr, parts[-1], Parameter(_from_numpy(np.ascontiguousarray(arr, dtype=np.float32))))


class SimCLRLoss(Module):
    """Normalized Temperature-scaled Cross Entropy (NT-Xent) contrastive loss for SimCLR."""

    def __init__(self, temperature: float = 0.07):
        super().__init__()
        if temperature <= 0.0:
            raise ValueError(f"temperature must be > 0, got {temperature}")
        self.temperature = temperature

    def forward(self, z_i: _Tensor, z_j: _Tensor) -> _Tensor:
        """Takes normalized latent representations [N, D] for two augmented views."""
        zi = z_i.unsqueeze(0) if z_i.ndim == 1 else z_i
        zj = z_j.unsqueeze(0) if z_j.ndim == 1 else z_j

        if zi.shape != zj.shape:
            raise ValueError(f"Shape mismatch: z_i {zi.shape} vs z_j {zj.shape}")

        n = zi.shape[0]
        if n == 0:
            return _from_numpy(np.array(0.0, dtype=np.float32))

        # Differentiable normalization
        zi_norm = zi / ((zi * zi).sum(dim=-1, keepdim=True).sqrt() + 1e-12)
        zj_norm = zj / ((zj * zj).sum(dim=-1, keepdim=True).sqrt() + 1e-12)

        representations = cat([zi_norm, zj_norm], dim=0)  # [2N, D]
        sim = representations.matmul(representations.t()) * (1.0 / max(self.temperature, 1e-4))

        diag_mask = _from_numpy(np.eye(2 * n, dtype=np.float32) * 1e9)
        sim_masked = sim - diag_mask

        labels = np.concatenate([np.arange(n, 2 * n), np.arange(0, n)])
        labels_tensor = _from_numpy(labels.astype(np.int64))

        return cross_entropy(sim_masked, labels_tensor)


__all__ = [
    "PseudoLabelLoss",
    "ConsistencyLoss",
    "MeanTeacher",
    "SimCLRLoss",
]
