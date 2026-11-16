"""Causal Inference, Treatment Effect Estimation, and Observational Matching for Oxideon.

Enables unconfounded causal effect estimation, propensity score matching, and Double ML:
  - average_treatment_effect: Computes sample ATE and Inverse Probability Weighting (IPW)
  - doubly_robust_ate: Augmented Inverse Probability Weighting (AIPW) doubly robust ATE
  - PropensityScoreMatcher: Nearest-neighbor propensity score matching with caliper threshold
  - DoubleMachineLearning: Robinson transformation / Neyman orthogonal score estimator (Chernozhukov et al., 2018)
"""

from typing import Tuple, Optional, Any, List, Dict
import numpy as np

from .oxideon import Tensor as _Tensor, from_numpy as _from_numpy


def _to_numpy_1d(x: Any) -> np.ndarray:
    if isinstance(x, _Tensor):
        return x.to_numpy().reshape(-1)
    return np.asarray(x, dtype=np.float32).reshape(-1)


def average_treatment_effect(
    outcomes: Any,
    treatment: Any,
    propensity_scores: Optional[Any] = None,
) -> float:
    """Calculates Average Treatment Effect (ATE) via difference-in-means or IPW estimator."""
    y = _to_numpy_1d(outcomes)
    t = _to_numpy_1d(treatment).astype(bool)
    if len(y) != len(t) or len(y) == 0:
        raise ValueError(f"Length mismatch or empty inputs: outcomes={len(y)}, treatment={len(t)}")
    if not np.any(t) or not np.any(~t):
        raise ValueError("Both treated and control units must be present to estimate ATE")

    if propensity_scores is None:
        # Difference in means
        ate = np.mean(y[t]) - np.mean(y[~t])
        return float(ate)

    # Inverse Probability Weighting (IPW)
    ps = _to_numpy_1d(propensity_scores)
    if len(ps) != len(y):
        raise ValueError(f"Length mismatch: propensity_scores={len(ps)} vs outcomes={len(y)}")
    e = np.clip(ps, 1e-4, 1.0 - 1e-4)
    w1 = t / e
    w0 = (1.0 - t) / (1.0 - e)

    mu1 = np.sum(w1 * y) / np.sum(w1)
    mu0 = np.sum(w0 * y) / np.sum(w0)
    return float(mu1 - mu0)


def doubly_robust_ate(
    outcomes: Any,
    treatment: Any,
    propensity_scores: Any,
    mu1_preds: Any,
    mu0_preds: Any,
) -> float:
    """Augmented Inverse Probability Weighting (AIPW) Doubly Robust ATE estimator.

    Consistent if either the propensity model e(X) or the outcome regressions
    (mu_1(X), mu_0(X)) are correctly specified.
    """
    y = _to_numpy_1d(outcomes)
    t = _to_numpy_1d(treatment).astype(np.float32)
    e = np.clip(_to_numpy_1d(propensity_scores), 1e-4, 1.0 - 1e-4)
    mu1 = _to_numpy_1d(mu1_preds)
    mu0 = _to_numpy_1d(mu0_preds)

    if not (len(y) == len(t) == len(e) == len(mu1) == len(mu0)) or len(y) == 0:
        raise ValueError("All inputs to doubly_robust_ate must be non-empty and have matching lengths")

    psi_1 = mu1 + (t / e) * (y - mu1)
    psi_0 = mu0 + ((1.0 - t) / (1.0 - e)) * (y - mu0)
    return float(np.mean(psi_1 - psi_0))


class PropensityScoreMatcher:
    """Matches treated units to control units based on propensity scores within caliper."""

    def __init__(self, caliper: float = 0.05):
        if float(caliper) <= 0.0:
            raise ValueError(f"caliper must be > 0, got {caliper}")
        self.caliper = float(caliper)
        self.matched_treated_indices: List[int] = []
        self.matched_control_indices: List[int] = []

    def match(
        self, propensity_scores: Any, treatment: Any
    ) -> Tuple[np.ndarray, np.ndarray]:
        scores = _to_numpy_1d(propensity_scores)
        t = _to_numpy_1d(treatment).astype(bool)
        if len(scores) != len(t) or len(scores) == 0:
            raise ValueError(f"Length mismatch or empty inputs: propensity_scores={len(scores)}, treatment={len(t)}")

        treated_idx = np.where(t)[0]
        control_idx = np.where(~t)[0]

        available_control = set(control_idx)

        matched_t, matched_c = [], []

        for ti in treated_idx:
            t_score = scores[ti]
            # Find closest available control
            valid_c = [ci for ci in available_control if abs(scores[ci] - t_score) <= self.caliper]
            if valid_c:
                best_c = min(valid_c, key=lambda ci: abs(scores[ci] - t_score))
                matched_t.append(ti)
                matched_c.append(best_c)
                available_control.remove(best_c)

        self.matched_treated_indices = matched_t
        self.matched_control_indices = matched_c
        return np.array(matched_t, dtype=np.int64), np.array(matched_c, dtype=np.int64)


class DoubleMachineLearning:
    """Partially Linear Double Machine Learning (DML) causal estimator."""

    def __init__(self):
        self.theta_: Optional[float] = None

    def fit(self, y: Any, t: Any, x: Any) -> float:
        """Estimates treatment effect theta in Y = theta * T + g(X) + eps."""
        y_arr = _to_numpy_1d(y)
        t_arr = _to_numpy_1d(t)
        x_mat = x.to_numpy() if isinstance(x, _Tensor) else np.asarray(x, dtype=np.float32)
        if x_mat.ndim == 1:
            x_mat = x_mat[:, None]
        if not (len(y_arr) == len(t_arr) == len(x_mat)) or len(y_arr) < 2:
            raise ValueError(
                f"Length mismatch or insufficient samples (need >= 2): y={len(y_arr)}, t={len(t_arr)}, x={len(x_mat)}"
            )

        # Prepend unpenalized intercept column so non-zero baseline outcomes/treatments are partialled out
        ones = np.ones((len(x_mat), 1), dtype=np.float64)
        x_aug = np.hstack([ones, x_mat.astype(np.float64)])
        reg = 1e-4 * np.eye(x_aug.shape[1], dtype=np.float64)
        reg[0, 0] = 0.0

        beta_y = np.linalg.solve(x_aug.T @ x_aug + reg, x_aug.T @ y_arr.astype(np.float64))
        beta_t = np.linalg.solve(x_aug.T @ x_aug + reg, x_aug.T @ t_arr.astype(np.float64))

        y_res = y_arr.astype(np.float64) - x_aug @ beta_y
        t_res = t_arr.astype(np.float64) - x_aug @ beta_t

        # Second stage: regress Y residual on T residual
        theta = float(np.sum(t_res * y_res) / max(np.sum(t_res**2), 1e-12))
        self.theta_ = theta
        return theta


__all__ = [
    "average_treatment_effect",
    "doubly_robust_ate",
    "PropensityScoreMatcher",
    "DoubleMachineLearning",
]
