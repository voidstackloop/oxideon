"""Algorithmic Fairness, Bias Auditing, and Mitigation for Oxideon.

Enables auditing model predictions against sensitive attributes (e.g. demographic group):
  - demographic_parity_difference: Maximum gap in positive prediction rate across groups
  - equal_opportunity_difference: Maximum gap in True Positive Rate (TPR) across groups
  - equalized_odds_difference: Maximum gap in TPR and FPR across groups
  - disparate_impact_ratio: Ratio of positive prediction rates (EEOC 80% rule)
  - generalized_entropy_index: Individual/group inequality index (Speicher et al., 2018)
  - reweighting_weights: Instance weights mitigating demographic disparity during training
"""

from typing import Tuple, Optional, Any, Dict, List
import numpy as np

from .oxideon import Tensor as _Tensor, from_numpy as _from_numpy


def _to_1d_array(x: Any, is_pred: bool = False) -> np.ndarray:
    arr = x.to_numpy() if isinstance(x, _Tensor) else np.asarray(x)
    if arr.size == 0:
        raise ValueError("Fairness inputs cannot be empty")
    if is_pred and arr.ndim == 2:
        if arr.shape[1] == 2:
            # Binary probability / logit table [N, 2]: use positive class (col 1) or argmax
            if np.any(arr < 0) or not np.allclose(np.sum(arr, axis=1), 1.0, atol=1e-2):
                return np.argmax(arr, axis=1).astype(np.float32)
            return arr[:, 1].astype(np.float32)
        elif arr.shape[1] > 2:
            return np.argmax(arr, axis=1).astype(np.float32)
    return arr.reshape(-1)


def demographic_parity_difference(y_pred: Any, sensitive_features: Any) -> float:
    """Computes the difference between the largest and smallest group selection rates."""
    pred = (_to_1d_array(y_pred, is_pred=True) >= 0.5).astype(np.float32)
    group = _to_1d_array(sensitive_features)
    if len(pred) != len(group):
        raise ValueError(f"Length mismatch: y_pred has {len(pred)}, sensitive_features has {len(group)}")

    unique_groups = np.unique(group)
    rates = []
    for g in unique_groups:
        mask = group == g
        if np.any(mask):
            rates.append(np.mean(pred[mask]))

    if not rates:
        return 0.0
    return float(np.max(rates) - np.min(rates))


def disparate_impact_ratio(y_pred: Any, sensitive_features: Any) -> float:
    """Computes selection rate ratio min(rate) / max(rate) (EEOC four-fifths rule)."""
    pred = (_to_1d_array(y_pred, is_pred=True) >= 0.5).astype(np.float32)
    group = _to_1d_array(sensitive_features)
    if len(pred) != len(group):
        raise ValueError(f"Length mismatch: y_pred has {len(pred)}, sensitive_features has {len(group)}")

    unique_groups = np.unique(group)
    rates = []
    for g in unique_groups:
        mask = group == g
        if np.any(mask):
            rates.append(np.mean(pred[mask]))

    if not rates or np.max(rates) == 0:
        return 1.0
    return float(np.min(rates) / np.max(rates))


def equal_opportunity_difference(
    y_true: Any, y_pred: Any, sensitive_features: Any
) -> float:
    """Computes the maximum difference in True Positive Rate (TPR) across protected groups."""
    true = (_to_1d_array(y_true) >= 0.5).astype(bool)
    pred = (_to_1d_array(y_pred, is_pred=True) >= 0.5).astype(bool)
    group = _to_1d_array(sensitive_features)
    if not (len(true) == len(pred) == len(group)):
        raise ValueError(
            f"Length mismatch: y_true has {len(true)}, y_pred has {len(pred)}, sensitive_features has {len(group)}"
        )

    unique_groups = np.unique(group)
    tprs = []
    for g in unique_groups:
        pos = (group == g) & true
        if np.any(pos):
            tprs.append(float(np.mean(pred[pos])))

    if not tprs:
        return 0.0
    return float(np.max(tprs) - np.min(tprs))


def equalized_odds_difference(
    y_true: Any, y_pred: Any, sensitive_features: Any
) -> float:
    """Computes the maximum difference in TPR and FPR across protected groups."""
    true = (_to_1d_array(y_true) >= 0.5).astype(bool)
    pred = (_to_1d_array(y_pred, is_pred=True) >= 0.5).astype(bool)
    group = _to_1d_array(sensitive_features)
    if not (len(true) == len(pred) == len(group)):
        raise ValueError(f"Length mismatch: y_true has {len(true)}, y_pred has {len(pred)}, sensitive_features has {len(group)}")

    unique_groups = np.unique(group)
    tprs, fprs = [], []

    for g in unique_groups:
        mask_g = group == g
        # TPR = TP / (TP + FN) = pred[pos] / len(pos)
        pos = mask_g & true
        if np.any(pos):
            tprs.append(np.mean(pred[pos]))
        # FPR = FP / (FP + TN) = pred[neg] / len(neg)
        neg = mask_g & (~true)
        if np.any(neg):
            fprs.append(np.mean(pred[neg]))

    tpr_diff = (np.max(tprs) - np.min(tprs)) if tprs else 0.0
    fpr_diff = (np.max(fprs) - np.min(fprs)) if fprs else 0.0
    return float(max(tpr_diff, fpr_diff))


def generalized_entropy_index(
    y_true: Any, y_pred: Any, alpha: float = 2.0
) -> float:
    """Computes the Generalized Entropy Index (Speicher et al., 2018) over individual benefits.

    Benefit is defined as b_i = y_pred_i - y_true_i + 1 >= 0.
    """
    true = (_to_1d_array(y_true) >= 0.5).astype(np.float64)
    pred = (_to_1d_array(y_pred, is_pred=True) >= 0.5).astype(np.float64)
    if len(true) != len(pred):
        raise ValueError(f"Length mismatch: y_true has {len(true)}, y_pred has {len(pred)}")

    b = pred - true + 1.0
    mu = float(np.mean(b))
    if mu <= 1e-12:
        return 0.0
    ratio = np.maximum(b / mu, 1e-12)
    a = float(alpha)
    if abs(a - 1.0) < 1e-8:
        # Theil index limit (alpha -> 1)
        return float(np.mean(ratio * np.log(ratio)))
    if abs(a) < 1e-8:
        # Mean log deviation limit (alpha -> 0)
        return float(-np.mean(np.log(ratio)))
    return float(np.mean(ratio ** a - 1.0) / (a * (a - 1.0)))


def reweighting_weights(y_true: Any, sensitive_features: Any) -> _Tensor:
    """Computes sample weights W(x) = (P(Y) * P(A)) / P(Y, A) to balance representation."""
    y = (_to_1d_array(y_true) >= 0.5).astype(np.int64)
    a = _to_1d_array(sensitive_features)
    if len(y) != len(a):
        raise ValueError(f"Length mismatch: y_true has {len(y)}, sensitive_features has {len(a)}")
    n = len(y)

    weights = np.ones(n, dtype=np.float32)
    unique_y = np.unique(y)
    unique_a = np.unique(a)

    for y_val in unique_y:
        p_y = np.mean(y == y_val)
        for a_val in unique_a:
            p_a = np.mean(a == a_val)
            mask_ya = (y == y_val) & (a == a_val)
            p_ya = np.mean(mask_ya)
            if p_ya > 0:
                weights[mask_ya] = (p_y * p_a) / p_ya

    return _from_numpy(np.ascontiguousarray(weights, dtype=np.float32))


__all__ = [
    "demographic_parity_difference",
    "disparate_impact_ratio",
    "equal_opportunity_difference",
    "equalized_odds_difference",
    "generalized_entropy_index",
    "reweighting_weights",
]
