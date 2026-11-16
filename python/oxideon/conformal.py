"""Conformal Prediction and Distribution-Free Uncertainty Guarantees for Oxideon.

Provides mathematically guaranteed finite-sample coverage P(Y in C(X)) >= 1 - alpha:
  - ConformalClassifier: Constructs valid prediction sets for multi-class classification (LAC & APS)
  - ConformalRegressor: Constructs valid prediction intervals [y_hat - q, y_hat + q]
  - coverage_score: Evaluates empirical coverage rate across test ground truth
"""

from typing import List, Tuple, Optional, Any, Union
import math
import numpy as np

from .oxideon import Tensor as _Tensor, from_numpy as _from_numpy
from .functional import softmax


class ConformalClassifier:
    """Split Conformal Prediction for multi-class classification.

    Supports both:
      - ``method='lac'``: Least Ambiguous Set-Valued Classifiers (Sadinle et al., 2019), score = 1 - P(y_true)
      - ``method='aps'``: Adaptive Prediction Sets (Romano et al., 2020), cumulative sorted probability mass
    """

    def __init__(self, alpha: float = 0.1, method: str = "lac"):
        if not (0.0 < float(alpha) < 1.0):
            raise ValueError(f"alpha must be strictly in (0, 1), got {alpha}")
        method_l = method.lower()
        if method_l not in ("lac", "aps"):
            raise ValueError(f"method must be 'lac' or 'aps', got '{method}'")
        self.alpha = float(alpha)
        self.method = method_l
        self.q_hat: Optional[float] = None

    def calibrate(self, val_probs_or_logits: Any, val_targets: Any) -> "ConformalClassifier":
        """Computes the conformal non-conformity quantile on validation calibration data."""
        p = val_probs_or_logits.to_numpy() if isinstance(val_probs_or_logits, _Tensor) else np.asarray(val_probs_or_logits, dtype=np.float32)
        y = val_targets.to_numpy() if isinstance(val_targets, _Tensor) else np.asarray(val_targets, dtype=np.int64).reshape(-1)

        if p.ndim != 2 or len(p) == 0:
            raise ValueError(f"val_probs_or_logits must be non-empty 2D [N, C], got shape {p.shape}")
        if len(p) != len(y):
            raise ValueError(f"Length mismatch: val_probs_or_logits has {len(p)} samples, val_targets has {len(y)}")
        if np.any(y < 0) or np.any(y >= p.shape[1]):
            raise ValueError(f"val_targets class indices must be in [0, {p.shape[1]})")

        # Normalize probabilities
        if np.any(p < 0) or not np.allclose(np.sum(p, axis=1), 1.0, atol=1e-2):
            exp_p = np.exp(p - np.max(p, axis=1, keepdims=True))
            p = exp_p / np.sum(exp_p, axis=1, keepdims=True)

        n = len(y)
        if self.method == "aps":
            scores = np.zeros(n, dtype=np.float32)
            for i in range(n):
                order = np.argsort(p[i])[::-1]
                cumsum = np.cumsum(p[i][order])
                rank = int(np.where(order == y[i])[0][0])
                scores[i] = cumsum[rank]
        else:
            # Non-conformity score: 1 - P(y_true)
            true_probs = p[np.arange(n), y]
            scores = 1.0 - true_probs

        # Conformal quantile level: ceil((n + 1) * (1 - alpha)) / n
        level = min(1.0, math.ceil((n + 1) * (1.0 - self.alpha)) / n)
        self.q_hat = float(np.quantile(scores, level, method="higher"))
        return self

    def predict_sets(self, test_probs_or_logits: Any) -> List[List[int]]:
        """Returns prediction set of class indices per sample with 1 - alpha coverage guarantee."""
        if self.q_hat is None:
            raise RuntimeError("Classifier must be calibrated first.")

        p = test_probs_or_logits.to_numpy() if isinstance(test_probs_or_logits, _Tensor) else np.asarray(test_probs_or_logits, dtype=np.float32)
        if p.ndim == 1:
            p = np.expand_dims(p, 0)
        elif p.ndim != 2:
            raise ValueError(f"test_probs_or_logits must be 1D [C] or 2D [N, C], got shape {p.shape}")

        if np.any(p < 0) or not np.allclose(np.sum(p, axis=1), 1.0, atol=1e-2):
            exp_p = np.exp(p - np.max(p, axis=1, keepdims=True))
            p = exp_p / np.sum(exp_p, axis=1, keepdims=True)

        sets = []
        for row in p:
            if self.method == "aps":
                order = np.argsort(row)[::-1]
                cumsum = np.cumsum(row[order])
                k = int(np.searchsorted(cumsum, self.q_hat, side="left")) + 1
                k = min(max(k, 1), len(row))
                included = sorted(order[:k].tolist())
                sets.append(included)
            else:
                # Classes where 1 - P(c) <= q_hat (i.e. P(c) >= 1 - q_hat)
                included = np.where(1.0 - row <= self.q_hat)[0].tolist()
                if not included:
                    included = [int(np.argmax(row))]
                sets.append(included)

        return sets


class ConformalRegressor:
    """Split Conformal Prediction for continuous regression."""

    def __init__(self, alpha: float = 0.1):
        if not (0.0 < float(alpha) < 1.0):
            raise ValueError(f"alpha must be strictly in (0, 1), got {alpha}")
        self.alpha = float(alpha)
        self.q_hat: Optional[float] = None

    def calibrate(self, val_preds: Any, val_targets: Any) -> "ConformalRegressor":
        y_hat = val_preds.to_numpy() if isinstance(val_preds, _Tensor) else np.asarray(val_preds, dtype=np.float32).reshape(-1)
        y = val_targets.to_numpy() if isinstance(val_targets, _Tensor) else np.asarray(val_targets, dtype=np.float32).reshape(-1)

        if len(y_hat) != len(y) or len(y) == 0:
            raise ValueError(f"Length mismatch or empty calibration data: preds={len(y_hat)}, targets={len(y)}")

        residuals = np.abs(y - y_hat)
        n = len(y)
        level = min(1.0, math.ceil((n + 1) * (1.0 - self.alpha)) / n)
        self.q_hat = float(np.quantile(residuals, level, method="higher"))
        return self

    def predict_intervals(self, test_preds: Any) -> Tuple[np.ndarray, np.ndarray]:
        """Returns lower and upper bounds (y_lower, y_upper) with 1 - alpha coverage."""
        if self.q_hat is None:
            raise RuntimeError("Regressor must be calibrated first.")

        y_hat = test_preds.to_numpy() if isinstance(test_preds, _Tensor) else np.asarray(test_preds, dtype=np.float32).reshape(-1)
        lower = y_hat - self.q_hat
        upper = y_hat + self.q_hat
        return lower, upper


def coverage_score(
    prediction_sets_or_intervals: Any, targets: Any
) -> float:
    """Computes empirical fraction of targets covered by prediction sets/intervals."""
    y = targets.to_numpy() if isinstance(targets, _Tensor) else np.asarray(targets).reshape(-1)
    if len(y) == 0:
        raise ValueError("targets cannot be empty")

    if isinstance(prediction_sets_or_intervals, list):
        if len(prediction_sets_or_intervals) != len(y):
            raise ValueError(f"Length mismatch: {len(prediction_sets_or_intervals)} prediction sets vs {len(y)} targets")
        covered = [y[i] in prediction_sets_or_intervals[i] for i in range(len(y))]
        return float(np.mean(covered))
    elif isinstance(prediction_sets_or_intervals, tuple) and len(prediction_sets_or_intervals) == 2:
        low, high = np.asarray(prediction_sets_or_intervals[0]).reshape(-1), np.asarray(prediction_sets_or_intervals[1]).reshape(-1)
        if len(low) != len(y) or len(high) != len(y):
            raise ValueError(f"Length mismatch in intervals vs targets ({len(low)} vs {len(y)})")
        covered = (y >= low) & (y <= high)
        return float(np.mean(covered))
    else:
        raise ValueError("Invalid format for prediction_sets_or_intervals")


__all__ = [
    "ConformalClassifier",
    "ConformalRegressor",
    "coverage_score",
]
