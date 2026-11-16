"""Advanced Machine Learning Loss Functions.

Implements:
24. DiceLoss: Multi-class Sørensen-Dice loss for semantic segmentation
25. AdaptiveFocalLoss: Modulated focal loss down-weighting easy examples (Lin et al., 2017)
26. LabelSmoothingCrossEntropy: Regularized cross-entropy with uniform smoothing
27. TripletMarginWithDistanceLoss: Generalized metric learning triplet loss
28. RankNetLoss: Pairwise ranking loss for information retrieval (Burges et al.)
29. InfoNCELoss: Self-supervised contrastive representation loss (Oord et al.)
"""

from typing import Optional, Callable
import numpy as np

from .oxideon import Tensor as _Tensor, from_numpy as _from_numpy
from .nn import Module


class DiceLoss(Module):
    """Multi-class Sørensen-Dice Loss for image segmentation."""

    def __init__(self, smooth: float = 1e-6):
        super().__init__()
        self.smooth = float(smooth)

    def forward(self, pred_probs: _Tensor, targets_onehot: _Tensor) -> _Tensor:
        """Args: pred_probs: [B, C, ...], targets_onehot: [B, C, ...]"""
        c = pred_probs.shape[1]
        p_flat = pred_probs.transpose(0, 1).reshape([c, -1])
        y_flat = targets_onehot.transpose(0, 1).reshape([c, -1])

        intersection = (p_flat * y_flat).sum(dim=1)
        cardinality = (p_flat + y_flat).sum(dim=1)

        dice_score = (intersection * 2.0 + self.smooth) / (cardinality + self.smooth)
        return (1.0 - dice_score.mean()).reshape([1])


class AdaptiveFocalLoss(Module):
    """Focal Loss for addressing class imbalance (Lin et al., ICCV 2017)."""

    def __init__(self, gamma: float = 2.0, alpha: Optional[float] = 0.25):
        super().__init__()
        if gamma < 0.0:
            raise ValueError(f"gamma must be >= 0, got {gamma}")
        self.gamma = float(gamma)
        self.alpha = alpha

    def forward(self, logits: _Tensor, targets: _Tensor) -> _Tensor:
        """Args: logits: [B, C], targets: [B] (integer class indices)"""
        z_np = logits.to_numpy()
        y = targets.to_numpy().astype(np.int64).flatten()
        if len(z_np) != len(y):
            raise ValueError(f"Length mismatch: logits has {len(z_np)}, targets has {len(y)}")
        if len(y) > 0 and (np.any(y < 0) or np.any(y >= z_np.shape[-1])):
            raise ValueError(f"Target class index out of bounds for {z_np.shape[-1]} classes")
        if len(y) == 0:
            return _from_numpy(np.array([0.0], dtype=np.float32))

        b, c = logits.shape
        probs = logits.softmax(-1)
        one_hot = np.zeros((b, c), dtype=np.float32)
        one_hot[np.arange(b), y] = 1.0
        p_t = (probs * _from_numpy(one_hot)).sum(dim=-1).clamp(1e-7, 1.0 - 1e-7)

        focal_weight = (1.0 - p_t).pow(self.gamma)
        if self.alpha is not None:
            focal_weight = focal_weight * float(self.alpha)

        loss = -(focal_weight * p_t.log()).mean()
        return loss.reshape([1])


class LabelSmoothingCrossEntropy(Module):
    """Cross-Entropy loss with uniform label smoothing."""

    def __init__(self, smoothing: float = 0.1):
        super().__init__()
        if not (0.0 <= smoothing <= 1.0):
            raise ValueError(f"smoothing must be between 0.0 and 1.0, got {smoothing}")
        self.smoothing = float(smoothing)

    def forward(self, logits: _Tensor, targets: _Tensor) -> _Tensor:
        """Args: logits: [B, C], targets: [B] (class indices)"""
        z_np = logits.to_numpy()
        y = targets.to_numpy().astype(np.int64).flatten()
        b, c = z_np.shape
        if len(y) != b:
            raise ValueError(f"Length mismatch: logits has {b}, targets has {len(y)}")
        if len(y) > 0 and (np.any(y < 0) or np.any(y >= c)):
            raise ValueError(f"Target class index out of bounds for {c} classes")
        if b == 0:
            return _from_numpy(np.array([0.0], dtype=np.float32))

        probs = logits.softmax(-1).clamp(1e-9, 1.0)
        log_probs = probs.log()

        one_hot = np.zeros((b, c), dtype=np.float32)
        one_hot[np.arange(b), y] = 1.0
        nll = -(log_probs * _from_numpy(one_hot)).sum(dim=-1)
        smooth_loss = -log_probs.mean(dim=-1)

        loss = (nll * (1.0 - self.smoothing) + smooth_loss * self.smoothing).mean()
        return loss.reshape([1])


class TripletMarginWithDistanceLoss(Module):
    """Triplet loss with customizable distance metric."""

    def __init__(self, margin: float = 1.0, distance_fn: Optional[Callable[[np.ndarray, np.ndarray], np.ndarray]] = None):
        super().__init__()
        self.margin = float(margin)
        self.distance_fn = distance_fn

    def forward(self, anchor: _Tensor, positive: _Tensor, negative: _Tensor) -> _Tensor:
        if self.distance_fn is None:
            diff_pos = anchor - positive
            diff_neg = anchor - negative
            d_pos = ((diff_pos * diff_pos).sum(dim=-1) + 1e-12).sqrt()
            d_neg = ((diff_neg * diff_neg).sum(dim=-1) + 1e-12).sqrt()
            return (d_pos - d_neg + self.margin).relu().mean().reshape([1])

        a = anchor.to_numpy()
        p = positive.to_numpy()
        n = negative.to_numpy()
        d_pos = self.distance_fn(a, p)
        d_neg = self.distance_fn(a, n)
        loss = np.mean(np.maximum(0.0, d_pos - d_neg + self.margin))
        return _from_numpy(np.array([loss], dtype=np.float32))


class RankNetLoss(Module):
    """Pairwise ranking loss for learning-to-rank systems."""

    def __init__(self, sigma: float = 1.0):
        super().__init__()
        if sigma <= 0.0:
            raise ValueError(f"sigma must be positive (> 0), got {sigma}")
        self.sigma = float(sigma)

    def forward(self, score_i: _Tensor, score_j: _Tensor, target_ij: _Tensor) -> _Tensor:
        """Args: score_i, score_j: [B], target_ij: [B] in {0, 1} indicating if i > j."""
        s_i = score_i.reshape([-1])
        s_j = score_j.reshape([-1])
        y_ij = target_ij.reshape([-1])
        if not (s_i.shape[0] == s_j.shape[0] == y_ij.shape[0]):
            raise ValueError(f"Length mismatch: score_i {s_i.shape[0]}, score_j {s_j.shape[0]}, target_ij {y_ij.shape[0]}")

        s_diff = (s_i - s_j) * self.sigma
        softplus_neg = ((-s_diff).clamp(-30.0, 30.0).exp() + 1.0).log()
        loss = ((1.0 - y_ij) * s_diff + softplus_neg).mean()
        return loss.reshape([1])


class InfoNCELoss(Module):
    """Contrastive self-supervised InfoNCE representation loss."""

    def __init__(self, temperature: float = 0.07):
        super().__init__()
        if temperature <= 0.0:
            raise ValueError(f"temperature must be positive (> 0), got {temperature}")
        self.temperature = float(temperature)

    def forward(self, query: _Tensor, positive_key: _Tensor, negative_keys: Optional[_Tensor] = None) -> _Tensor:
        """Args:
            query: [B, D]
            positive_key: [B, D]
            negative_keys: Optional [B, M, D]
        """
        if list(query.shape) != list(positive_key.shape):
            raise ValueError(f"Shape mismatch: query {query.shape} vs positive_key {positive_key.shape}")

        q_norm = query / ((query * query).sum(dim=-1, keepdim=True).sqrt() + 1e-9)
        k_pos_norm = positive_key / ((positive_key * positive_key).sum(dim=-1, keepdim=True).sqrt() + 1e-9)

        if negative_keys is not None:
            k_neg_norm = negative_keys / ((negative_keys * negative_keys).sum(dim=-1, keepdim=True).sqrt() + 1e-9)
            pos_sim = (q_norm * k_pos_norm).sum(dim=-1, keepdim=True) * (1.0 / self.temperature)
            neg_sim = q_norm.unsqueeze(1).matmul(k_neg_norm.transpose(1, 2)).squeeze(1) * (1.0 / self.temperature)
            all_sim = _Tensor.cat([pos_sim, neg_sim], dim=-1)
            probs = all_sim.softmax(-1)
            b, m_plus_1 = probs.shape
            sel = np.zeros((b, m_plus_1), dtype=np.float32)
            sel[:, 0] = 1.0
            pos_probs = (probs * _from_numpy(sel)).sum(dim=-1)
            return (-(pos_probs + 1e-9).log().mean()).reshape([1])
        else:
            b = q_norm.shape[0]
            all_sim = q_norm.matmul(k_pos_norm.transpose(0, 1)) * (1.0 / self.temperature)
            probs = all_sim.softmax(-1)
            eye = _from_numpy(np.eye(b, dtype=np.float32))
            pos_probs = (probs * eye).sum(dim=-1)
            return (-(pos_probs + 1e-9).log().mean()).reshape([1])


__all__ = [
    "DiceLoss",
    "AdaptiveFocalLoss",
    "LabelSmoothingCrossEntropy",
    "TripletMarginWithDistanceLoss",
    "RankNetLoss",
    "InfoNCELoss",
]
