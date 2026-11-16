"""
Deep Survival Analysis, Time-to-Event Modeling, and Censored Data for Oxideon.
Includes differentiable Cox Proportional Hazards partial log-likelihood with exact Breslow tie handling,
Harrell's Concordance Index (C-Index), Kaplan-Meier survival curves, and Nelson-Aalen cumulative hazard.
"""
from typing import Tuple
import numpy as np
from .oxideon import Tensor, from_numpy as _from_numpy


class CoxProportionalHazardsLoss:
    """
    Differentiable negative partial log-likelihood of the Cox proportional hazards model
    with exact Breslow tie handling:
      L = - (1 / N_events) * sum_{i: E_i = 1} [ h_i - log(sum_{j: T_j >= T_i} exp(h_j)) ]
    """
    def __call__(self, log_hazards: Tensor, durations: Tensor, events: Tensor) -> Tensor:
        n = log_hazards.numel()
        if durations.numel() != n or events.numel() != n:
            raise ValueError(
                f"Length mismatch: log_hazards has {n}, durations has {durations.numel()}, events has {events.numel()}"
            )
        t_np = durations.to_numpy().flatten()
        e_np = events.to_numpy().flatten()
        if np.any(t_np < 0):
            raise ValueError("durations must be non-negative")
        if n == 0:
            return _from_numpy(np.array(0.0, dtype=np.float32))

        h_flat = log_hazards.reshape([n])
        # Risk set matrix R[i, j] = 1 if T_j >= T_i else 0 (handles tied durations identically)
        risk_mask = (t_np[None, :] >= t_np[:, None]).astype(np.float32)  # [N, N]
        h_max = float(np.max(h_flat.to_numpy()))
        exp_h = (h_flat - h_max).exp()  # [N]

        # Risk set sum of exp(h_j - h_max) for each subject i
        risk_mask_t = _from_numpy(risk_mask)
        risk_sums = risk_mask_t.matmul(exp_h.reshape([n, 1])).reshape([n])
        log_risk = (risk_sums + 1e-12).log() + h_max

        events_t = _from_numpy(e_np.astype(np.float32))
        total_events = max(float(np.sum(e_np)), 1.0)
        neg_ll = ((log_risk - h_flat) * events_t).sum() * (1.0 / total_events)
        return neg_ll


def concordance_index(risk_scores: Tensor, durations: Tensor, events: Tensor) -> float:
    """
    Computes Harrell's C-index (concordance index) for survival evaluation.
    Measures rank correlation between predicted risk and actual failure times.
    """
    r = risk_scores.to_numpy().flatten()
    t = durations.to_numpy().flatten()
    e = events.to_numpy().flatten()
    if not (len(r) == len(t) == len(e)):
        raise ValueError(f"Length mismatch: risk_scores has {len(r)}, durations has {len(t)}, events has {len(e)}")
    if np.any(t < 0):
        raise ValueError("durations must be non-negative")

    t_diff = t[None, :] - t[:, None]
    permissible_mask = (e[:, None] == 1) & (t_diff > 0)
    permissible = float(np.sum(permissible_mask))

    if permissible == 0.0:
        return 0.5

    r_diff = r[:, None] - r[None, :]
    concordant = float(
        np.sum(permissible_mask & (r_diff > 0)) + 0.5 * np.sum(permissible_mask & (r_diff == 0))
    )
    return float(concordant / permissible)


class KaplanMeierEstimator:
    """Non-parametric empirical survival curve estimator."""
    def fit(self, durations: Tensor, events: Tensor) -> Tuple[np.ndarray, np.ndarray]:
        t = durations.to_numpy().flatten()
        e = events.to_numpy().flatten()
        if len(t) != len(e):
            raise ValueError(f"Length mismatch: durations has {len(t)}, events has {len(e)}")
        if np.any(t < 0):
            raise ValueError("durations must be non-negative")

        order = np.argsort(t)
        t_sorted = t[order]
        e_sorted = e[order]

        unique_times = np.unique(t_sorted)
        survival_probs = []
        s_curr = 1.0
        n_at_risk = len(t)

        for ut in unique_times:
            mask = t_sorted == ut
            d_i = np.sum(e_sorted[mask])
            c_i = np.sum(1.0 - e_sorted[mask])

            if n_at_risk > 0:
                s_curr *= 1.0 - d_i / n_at_risk
            survival_probs.append(s_curr)
            n_at_risk -= d_i + c_i

        return unique_times, np.array(survival_probs, dtype=np.float32)


class NelsonAalenEstimator:
    """Non-parametric Nelson-Aalen cumulative hazard estimator: H(t) = sum_{t_i <= t} d_i / n_i."""
    def fit(self, durations: Tensor, events: Tensor) -> Tuple[np.ndarray, np.ndarray]:
        t = durations.to_numpy().flatten()
        e = events.to_numpy().flatten()
        if len(t) != len(e):
            raise ValueError(f"Length mismatch: durations has {len(t)}, events has {len(e)}")
        if np.any(t < 0):
            raise ValueError("durations must be non-negative")

        order = np.argsort(t)
        t_sorted = t[order]
        e_sorted = e[order]

        unique_times = np.unique(t_sorted)
        cum_hazard = []
        h_curr = 0.0
        n_at_risk = float(len(t))

        for ut in unique_times:
            mask = t_sorted == ut
            d_i = float(np.sum(e_sorted[mask]))
            total_i = float(np.sum(mask))
            if n_at_risk > 0.0:
                h_curr += d_i / n_at_risk
            cum_hazard.append(h_curr)
            n_at_risk -= total_i

        return unique_times, np.array(cum_hazard, dtype=np.float32)
