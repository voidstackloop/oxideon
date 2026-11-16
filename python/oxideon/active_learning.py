"""Active Learning and Uncertainty-based Acquisition Strategies for Oxideon.

Enables query strategies for selecting the most informative unlabeled samples:
  - entropy_sampling: Shannon entropy query strategy
  - margin_sampling: Top-2 probability margin query strategy
  - least_confidence_sampling: Inverse top-1 confidence query strategy
  - bald_sampling: Bayesian Active Learning by Disagreement on MC dropout passes
  - core_set_kcenter_greedy: K-Center Greedy core-set diversity selection (Sener & Savarese, 2018)
"""

from typing import Any, Tuple, Optional, List
import numpy as np

from .oxideon import Tensor as _Tensor


def _to_probs(probs_or_logits: Any) -> np.ndarray:
    p = probs_or_logits.to_numpy() if isinstance(probs_or_logits, _Tensor) else np.asarray(probs_or_logits, dtype=np.float32)
    if p.size == 0:
        raise ValueError("probs_or_logits cannot be empty")
    if np.any(p < 0) or not np.allclose(np.sum(p, axis=-1), 1.0, atol=1e-2):
        exp_p = np.exp(p - np.max(p, axis=-1, keepdims=True))
        p = exp_p / np.sum(exp_p, axis=-1, keepdims=True)
    return p


def entropy_sampling(probs_or_logits: Any, n_samples: int) -> np.ndarray:
    """Selects top `n_samples` with highest predictive Shannon entropy."""
    if n_samples < 1:
        raise ValueError(f"n_samples must be >= 1, got {n_samples}")
    probs = _to_probs(probs_or_logits)
    entropy = -np.sum(probs * np.log(np.maximum(probs, 1e-12)), axis=-1)
    # Highest entropy first
    indices = np.argsort(entropy)[::-1][:n_samples]
    return indices


def margin_sampling(probs_or_logits: Any, n_samples: int) -> np.ndarray:
    """Selects top `n_samples` with smallest margin between top-1 and top-2 class probabilities."""
    if n_samples < 1:
        raise ValueError(f"n_samples must be >= 1, got {n_samples}")
    probs = _to_probs(probs_or_logits)
    is_1d = probs.ndim == 1
    if is_1d:
        probs = np.expand_dims(probs, 0)
    if probs.shape[-1] > 2:
        top2 = np.sort(np.partition(probs, -2, axis=-1)[:, -2:], axis=-1)
        margins = top2[:, 1] - top2[:, 0]
    elif probs.shape[-1] == 2:
        margins = np.abs(probs[:, 0] - probs[:, 1])
    else:
        margins = probs.flatten()
    # Smallest margins first
    indices = np.argsort(margins)[:n_samples]
    return indices


def least_confidence_sampling(probs_or_logits: Any, n_samples: int) -> np.ndarray:
    """Selects top `n_samples` with lowest top-1 predicted probability."""
    if n_samples < 1:
        raise ValueError(f"n_samples must be >= 1, got {n_samples}")
    probs = _to_probs(probs_or_logits)
    max_prob = np.max(probs, axis=-1)
    # Smallest max probability first
    indices = np.argsort(max_prob)[:n_samples]
    return indices


def bald_sampling(mc_probs: Any, n_samples: int) -> np.ndarray:
    """Bayesian Active Learning by Disagreement (BALD) for MC dropout runs [S, N, C].

    Computes mutual information: I(y; W | x) = H(y | x) - E_W[H(y | x, W)].
    """
    if n_samples < 1:
        raise ValueError(f"n_samples must be >= 1, got {n_samples}")
    p = mc_probs.to_numpy() if isinstance(mc_probs, _Tensor) else np.asarray(mc_probs, dtype=np.float32)
    if p.ndim != 3:
        raise ValueError(f"mc_probs must be 3D of shape [num_mc_samples, num_instances, num_classes], got shape {p.shape}")
    if p.size == 0:
        raise ValueError("mc_probs cannot be empty")
    if np.any(p < 0) or not np.allclose(np.sum(p, axis=-1), 1.0, atol=1e-2):
        exp_p = np.exp(p - np.max(p, axis=-1, keepdims=True))
        p = exp_p / np.sum(exp_p, axis=-1, keepdims=True)
    # Expected predictions across samples [N, C]
    expected_p = np.mean(p, axis=0)
    total_entropy = -np.sum(expected_p * np.log(np.maximum(expected_p, 1e-12)), axis=-1)

    # Expected entropy of predictions [N]
    sample_entropies = -np.sum(p * np.log(np.maximum(p, 1e-12)), axis=-1)
    expected_sample_entropy = np.mean(sample_entropies, axis=0)

    mutual_info = total_entropy - expected_sample_entropy
    # Highest mutual information first
    indices = np.argsort(mutual_info)[::-1][:n_samples]
    return indices


def core_set_kcenter_greedy(
    embeddings: Any,
    labeled_indices: Any,
    n_samples: int,
) -> np.ndarray:
    """K-Center Greedy core-set selection (Sener & Savarese, 2018).

    Iteratively selects the unlabeled sample whose minimum Euclidean distance
    to the currently labeled set is maximized.
    """
    if n_samples < 1:
        raise ValueError(f"n_samples must be >= 1, got {n_samples}")
    emb = embeddings.to_numpy() if isinstance(embeddings, _Tensor) else np.asarray(embeddings, dtype=np.float32)
    if emb.ndim == 1:
        emb = emb[:, None]
    elif emb.ndim != 2 or len(emb) == 0:
        raise ValueError(f"embeddings must be non-empty 2D [N, D], got shape {emb.shape}")

    n_total = len(emb)
    lab = list(np.asarray(labeled_indices, dtype=np.int64).reshape(-1))
    if any(i < 0 or i >= n_total for i in lab):
        raise ValueError(f"labeled_indices out of bounds [0, {n_total})")

    selected: List[int] = []
    if len(lab) == 0:
        first = int(np.argmax(np.sum(emb ** 2, axis=1)))
        selected.append(first)
        lab.append(first)

    # Compute min squared distance from all points to current labeled set
    lab_emb = emb[lab]
    diff = emb[:, None, :] - lab_emb[None, :, :]
    min_dists = np.min(np.sum(diff ** 2, axis=-1), axis=1)
    min_dists[lab] = -1.0

    while len(selected) < n_samples and len(selected) < n_total:
        best_idx = int(np.argmax(min_dists))
        if min_dists[best_idx] < 0.0:
            break
        selected.append(best_idx)
        min_dists[best_idx] = -1.0
        new_dists = np.sum((emb - emb[best_idx : best_idx + 1]) ** 2, axis=1)
        min_dists = np.minimum(min_dists, new_dists)
        min_dists[lab] = -1.0
        for s in selected:
            min_dists[s] = -1.0

    return np.array(selected, dtype=np.int64)


__all__ = [
    "entropy_sampling",
    "margin_sampling",
    "least_confidence_sampling",
    "bald_sampling",
    "core_set_kcenter_greedy",
]
