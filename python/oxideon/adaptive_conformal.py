"""Adaptive Conformal Prediction and Conformalized Quantile Regression.

Provides finite-sample distribution-free uncertainty guarantees P(Y in C(X)) >= 1 - alpha:
- AdaptivePredictionSets (APS): Cumulative probability non-conformity sets (Romano et al., 2020)
- ConformalizedQuantileRegression (CQR): Valid prediction intervals via quantile bounds (Romano et al., 2019)
- MondrianConformalGroup: Group-conditional exact coverage across subgroups
- empirical_coverage: Evaluates empirical coverage of prediction sets on ground-truth labels
"""

from typing import Dict, List, Optional, Tuple, Any, Union, Sequence
import math
import numpy as np

from .oxideon import Tensor as _Tensor, from_numpy as _from_numpy


class AdaptivePredictionSets:
    """Adaptive Prediction Sets (APS) for classification with exact coverage guarantees."""

    def __init__(self, alpha: float = 0.1):
        if not (0.0 < float(alpha) < 1.0):
            raise ValueError(f"alpha must be in (0, 1), got {alpha}")
        self.alpha = float(alpha)
        self.q_hat: Optional[float] = None

    def calibrate(self, val_probs: Union[_Tensor, np.ndarray], val_labels: Union[_Tensor, np.ndarray]) -> "AdaptivePredictionSets":
        """Computes the conformal non-conformity quantile on validation calibration data."""
        p = val_probs.to_numpy() if isinstance(val_probs, _Tensor) else np.asarray(val_probs, dtype=np.float32)
        y_raw = val_labels.to_numpy() if isinstance(val_labels, _Tensor) else np.asarray(val_labels)
        y = y_raw.astype(np.int64).reshape(-1)

        n = len(y)
        if n == 0:
            raise ValueError("Calibration set cannot be empty.")
        scores = []
        for i in range(n):
            prob_i = p[i]
            target_class = int(y[i])
            sorted_idx = np.argsort(-prob_i)
            matches = np.where(sorted_idx == target_class)[0]
            if len(matches) == 0:
                raise ValueError(f"Target class {target_class} out of bounds for num_classes={len(prob_i)}")
            rank = int(matches[0])
            cum_prob = float(np.sum(prob_i[sorted_idx[:rank + 1]]))
            scores.append(cum_prob)

        scores_arr = np.array(scores, dtype=np.float32)
        level = min(1.0, math_ceil_quantile(n, 1.0 - self.alpha))
        self.q_hat = float(np.quantile(scores_arr, level, method="higher"))
        return self

    def predict(self, test_probs: Union[_Tensor, np.ndarray]) -> List[List[int]]:
        """Constructs prediction sets for test instances guaranteeing 1 - alpha coverage."""
        if self.q_hat is None:
            raise RuntimeError("Model must be calibrated before calling predict().")

        p = test_probs.to_numpy() if isinstance(test_probs, _Tensor) else np.asarray(test_probs, dtype=np.float32)
        prediction_sets: List[List[int]] = []

        for prob_i in p:
            sorted_idx = np.argsort(-prob_i)
            cum_prob = 0.0
            p_set = []
            for class_idx in sorted_idx:
                p_set.append(int(class_idx))
                cum_prob += float(prob_i[class_idx])
                if cum_prob >= self.q_hat - 1e-7:
                    break
            prediction_sets.append(p_set)

        return prediction_sets


class ConformalizedQuantileRegression:
    """Conformalized Quantile Regression (CQR) for regression intervals."""

    def __init__(self, alpha: float = 0.1):
        if not (0.0 < float(alpha) < 1.0):
            raise ValueError(f"alpha must be in (0, 1), got {alpha}")
        self.alpha = float(alpha)
        self.q_hat: Optional[float] = None

    def calibrate(
        self,
        lower_bounds: Union[_Tensor, np.ndarray],
        upper_bounds: Union[_Tensor, np.ndarray],
        targets: Union[_Tensor, np.ndarray],
    ) -> "ConformalizedQuantileRegression":
        """Computes non-conformity residuals E_i = max(q_low - y, y - q_high)."""
        low = lower_bounds.to_numpy().flatten() if isinstance(lower_bounds, _Tensor) else np.asarray(lower_bounds).flatten()
        high = upper_bounds.to_numpy().flatten() if isinstance(upper_bounds, _Tensor) else np.asarray(upper_bounds).flatten()
        y = targets.to_numpy().flatten() if isinstance(targets, _Tensor) else np.asarray(targets).flatten()

        n = len(y)
        if n == 0:
            raise ValueError("Calibration set cannot be empty.")
        residuals = np.maximum(low - y, y - high)
        level = min(1.0, math_ceil_quantile(n, 1.0 - self.alpha))
        self.q_hat = float(np.quantile(residuals, level, method="higher"))
        return self

    def predict(
        self,
        lower_bounds: Union[_Tensor, np.ndarray],
        upper_bounds: Union[_Tensor, np.ndarray],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Constructs guaranteed [lower - q_hat, upper + q_hat] confidence intervals."""
        if self.q_hat is None:
            raise RuntimeError("CQR model must be calibrated before predict().")

        low = lower_bounds.to_numpy() if isinstance(lower_bounds, _Tensor) else np.asarray(lower_bounds)
        high = upper_bounds.to_numpy() if isinstance(upper_bounds, _Tensor) else np.asarray(upper_bounds)

        calibrated_low = low - self.q_hat
        calibrated_high = high + self.q_hat
        return calibrated_low, calibrated_high


class MondrianConformalGroup:
    """Group-conditional conformal prediction ensuring equal coverage across subgroups."""

    def __init__(self, alpha: float = 0.1):
        if not (0.0 < float(alpha) < 1.0):
            raise ValueError(f"alpha must be in (0, 1), got {alpha}")
        self.alpha = float(alpha)
        self.group_quantiles: Dict[Any, float] = {}

    def calibrate(
        self,
        scores: Union[_Tensor, np.ndarray],
        groups: Union[_Tensor, np.ndarray],
    ) -> "MondrianConformalGroup":
        """Calibrates independent quantiles per group."""
        s_np = scores.to_numpy().flatten() if isinstance(scores, _Tensor) else np.asarray(scores).flatten()
        g_np = groups.to_numpy().flatten() if isinstance(groups, _Tensor) else np.asarray(groups).flatten()
        if len(s_np) != len(g_np):
            raise ValueError(f"Length mismatch: scores has {len(s_np)}, groups has {len(g_np)}")

        unique_groups = np.unique(g_np)
        for g in unique_groups:
            mask = g_np == g
            g_scores = s_np[mask]
            n_g = len(g_scores)
            level = min(1.0, math_ceil_quantile(n_g, 1.0 - self.alpha))
            q_val = float(np.quantile(g_scores, level, method="higher"))
            key = int(g) if isinstance(g, (np.integer, float, np.floating)) and float(g).is_integer() else g.item() if hasattr(g, "item") else g
            self.group_quantiles[key] = q_val
            self.group_quantiles[g] = q_val
        return self

    def get_threshold(self, group: Any) -> float:
        """Returns conformal threshold for group, falling back to global mean."""
        if not self.group_quantiles:
            raise RuntimeError("MondrianConformalGroup has not been calibrated yet.")
        return self.group_quantiles.get(group, float(np.mean(list(self.group_quantiles.values()))))


def math_ceil_quantile(n: int, level: float) -> float:
    """Calculates conformal quantile rank: ceil((n + 1) * level) / n."""
    return math.ceil((n + 1) * level) / float(max(n, 1))


def empirical_coverage(prediction_sets: Sequence[Sequence[int]], true_labels: Union[_Tensor, np.ndarray, Sequence[int]]) -> float:
    """Computes the fraction of test instances whose true label is contained in its prediction set."""
    y = true_labels.to_numpy().flatten() if isinstance(true_labels, _Tensor) else np.asarray(true_labels).flatten()
    if len(prediction_sets) != len(y):
        raise ValueError(f"Length mismatch: {len(prediction_sets)} prediction sets vs {len(y)} labels")
    if len(y) == 0:
        return 0.0
    hits = sum(1 for p_set, lbl in zip(prediction_sets, y) if int(lbl) in p_set)
    return float(hits) / float(len(y))


__all__ = [
    "AdaptivePredictionSets",
    "ConformalizedQuantileRegression",
    "MondrianConformalGroup",
    "math_ceil_quantile",
    "empirical_coverage",
]
