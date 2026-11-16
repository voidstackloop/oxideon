"""Anomaly Detection and Out-of-Distribution (OOD) Scoring for Oxideon.

Enables detecting out-of-distribution samples, anomalies, and safety violations:
  - EnergyScore: Energy-based OOD scoring from output logits (Liu et al., 2020)
  - MahalanobisDistance: Class-conditional Mahalanobis distance scoring (Lee et al., 2018)
  - IsolationForest: Tree-based ensemble anomaly isolation detector
"""

from typing import Tuple, Optional, Any, List
import math
import numpy as np

from .oxideon import Tensor as _Tensor, from_numpy as _from_numpy


class EnergyScore:
    """Computes free energy score E(x; T) = -T * log(sum(exp(f(x) / T))) from model logits."""

    def __init__(self, temperature: float = 1.0):
        if float(temperature) <= 0.0:
            raise ValueError(f"temperature must be > 0, got {temperature}")
        self.temperature = float(temperature)

    def score(self, logits: Any) -> _Tensor:
        """Returns scalar energy values per sample. Lower energy = in-distribution, higher energy = OOD."""
        arr = logits.to_numpy() if isinstance(logits, _Tensor) else np.asarray(logits, dtype=np.float32)
        if arr.size == 0:
            raise ValueError("logits cannot be empty")
        if isinstance(logits, _Tensor) and logits.requires_grad:
            scaled = logits / self.temperature
            max_np = np.max(arr / self.temperature, axis=-1, keepdims=True).astype(np.float32)
            max_t = _from_numpy(np.ascontiguousarray(max_np))
            max_sq = _from_numpy(np.ascontiguousarray(max_np.squeeze(-1)))
            lse = max_sq + (scaled - max_t).exp().sum(dim=-1).log()
            return lse * (-self.temperature)
        scaled = arr / self.temperature
        # LogSumExp
        max_l = np.max(scaled, axis=-1, keepdims=True)
        lse = max_l.squeeze(-1) + np.log(np.sum(np.exp(scaled - max_l), axis=-1))
        # Free energy is negative LogSumExp scaled by T
        energy = -self.temperature * lse
        return _from_numpy(np.ascontiguousarray(energy, dtype=np.float32))

    def __call__(self, logits: Any) -> _Tensor:
        return self.score(logits)


class MahalanobisDistance:
    """Computes Mahalanobis distance D_M(x) = sqrt((x - mu)^T Sigma^{-1} (x - mu))."""

    def __init__(self, mean: Any, cov: Optional[Any] = None, inv_cov: Optional[Any] = None):
        self.mean = mean.to_numpy() if isinstance(mean, _Tensor) else np.asarray(mean, dtype=np.float32)
        self.mean = self.mean.reshape(1, -1)
        d = self.mean.shape[1]
        if d == 0:
            raise ValueError("mean vector cannot be empty")

        if inv_cov is not None:
            ic = inv_cov.to_numpy() if isinstance(inv_cov, _Tensor) else np.asarray(inv_cov, dtype=np.float32)
            if ic.ndim != 2 or ic.shape != (d, d):
                raise ValueError(f"inv_cov shape {ic.shape} does not match mean dimension ({d}, {d})")
            self.inv_cov = ic
        elif cov is not None:
            c = cov.to_numpy() if isinstance(cov, _Tensor) else np.asarray(cov, dtype=np.float32)
            if c.ndim != 2 or c.shape != (d, d):
                raise ValueError(f"cov shape {c.shape} does not match mean dimension ({d}, {d})")
            reg = 1e-5 * np.eye(c.shape[0], dtype=np.float32)
            self.inv_cov = np.linalg.pinv(c + reg)
        else:
            raise ValueError("Must provide either cov or inv_cov")

    def score(self, x: Any) -> _Tensor:
        """Computes distance vector for input embeddings [N, D] or [D]."""
        arr = x.to_numpy() if isinstance(x, _Tensor) else np.asarray(x, dtype=np.float32)
        single = arr.ndim == 1
        if single:
            arr = arr.reshape(1, -1)
        elif arr.ndim != 2:
            raise ValueError(f"Input x must be 1D [D] or 2D [N, D], got shape {arr.shape}")
        if arr.shape[1] != self.mean.shape[1]:
            raise ValueError(f"Feature dimension mismatch: expected {self.mean.shape[1]}, got {arr.shape[1]}")

        if isinstance(x, _Tensor) and x.requires_grad:
            x_2d = x.reshape([1, -1]) if single else x
            mean_t = _from_numpy(np.ascontiguousarray(self.mean, dtype=np.float32))
            ic_t = _from_numpy(np.ascontiguousarray(self.inv_cov, dtype=np.float32))
            diff_t = x_2d - mean_t
            dist_sq_t = (diff_t.matmul(ic_t) * diff_t).sum(dim=-1).relu()
            return (dist_sq_t + 1e-12).sqrt()

        diff = arr - self.mean  # [N, D]
        dist_sq = np.sum((diff @ self.inv_cov) * diff, axis=-1)
        dist = np.sqrt(np.maximum(0.0, dist_sq))
        return _from_numpy(np.ascontiguousarray(dist, dtype=np.float32))

    def __call__(self, x: Any) -> _Tensor:
        return self.score(x)


class _IsolationTree:
    """Single randomized recursive isolation partition tree."""

    def __init__(self, depth: int, max_depth: int, rng: Optional[np.random.RandomState] = None):
        self.depth = depth
        self.max_depth = max_depth
        self.rng = rng if rng is not None else np.random.RandomState()
        self.split_feat: Optional[int] = None
        self.split_val: Optional[float] = None
        self.size: int = 0
        self.left: Optional["_IsolationTree"] = None
        self.right: Optional["_IsolationTree"] = None

    def fit(self, x: np.ndarray):
        self.size = len(x)
        if self.depth >= self.max_depth or self.size <= 1:
            return

        n_samples, n_features = x.shape
        self.split_feat = int(self.rng.randint(0, n_features))
        min_v = float(np.min(x[:, self.split_feat]))
        max_v = float(np.max(x[:, self.split_feat]))

        if min_v == max_v:
            return

        self.split_val = float(self.rng.uniform(min_v, max_v))
        left_mask = x[:, self.split_feat] < self.split_val

        self.left = _IsolationTree(self.depth + 1, self.max_depth, rng=self.rng)
        self.left.fit(x[left_mask])

        self.right = _IsolationTree(self.depth + 1, self.max_depth, rng=self.rng)
        self.right.fit(x[~left_mask])

    def path_length(self, x: np.ndarray) -> np.ndarray:
        if self.left is None or self.right is None:
            # Average path length of unsuccessful search
            if self.size <= 1:
                return np.full(len(x), float(self.depth), dtype=np.float32)
            c = 2.0 * (math.log(self.size - 1) + 0.5772156649) - (2.0 * (self.size - 1) / self.size)
            return np.full(len(x), float(self.depth + c), dtype=np.float32)

        paths = np.zeros(len(x), dtype=np.float32)
        left_mask = x[:, self.split_feat] < self.split_val
        if np.any(left_mask):
            paths[left_mask] = self.left.path_length(x[left_mask])
        if np.any(~left_mask):
            paths[~left_mask] = self.right.path_length(x[~left_mask])
        return paths


class IsolationForest:
    """Ensemble of isolation trees for anomaly and outlier detection."""

    def __init__(self, n_estimators: int = 50, max_samples: int = 128, seed: Optional[int] = None):
        if n_estimators < 1:
            raise ValueError(f"n_estimators must be >= 1, got {n_estimators}")
        if max_samples < 1:
            raise ValueError(f"max_samples must be >= 1, got {max_samples}")
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.seed = seed
        self.trees: List[_IsolationTree] = []
        self._n_features: Optional[int] = None

    def fit(self, x: Any) -> "IsolationForest":
        arr = x.to_numpy() if isinstance(x, _Tensor) else np.asarray(x, dtype=np.float32)
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)
        elif arr.ndim != 2:
            raise ValueError(f"Input x must be 1D or 2D, got shape {arr.shape}")
        n = len(arr)
        if n == 0 or arr.shape[1] == 0:
            raise ValueError("Training data x cannot be empty")
        self._n_features = arr.shape[1]
        subsample_size = min(n, self.max_samples)
        self._subsample_size = subsample_size
        max_depth = int(np.ceil(np.log2(max(subsample_size, 2))))

        rng = np.random.RandomState(self.seed)
        self.trees = []
        for _ in range(self.n_estimators):
            idx = rng.choice(n, subsample_size, replace=False)
            tree = _IsolationTree(depth=0, max_depth=max_depth, rng=rng)
            tree.fit(arr[idx])
            self.trees.append(tree)

        return self

    def score(self, x: Any) -> np.ndarray:
        """Computes anomaly score in range [0, 1]. Higher score close to 1 = anomaly."""
        if not hasattr(self, "_subsample_size") or not self.trees:
            raise ValueError("IsolationForest has not been fitted yet.")
        arr = x.to_numpy() if isinstance(x, _Tensor) else np.asarray(x, dtype=np.float32)
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)
        elif arr.ndim != 2:
            raise ValueError(f"Input x must be 1D or 2D, got shape {arr.shape}")
        if self._n_features is not None and arr.shape[1] != self._n_features:
            raise ValueError(f"Feature dimension mismatch: expected {self._n_features}, got {arr.shape[1]}")
        subsample_size = self._subsample_size
        c = 2.0 * (math.log(max(subsample_size - 1, 1)) + 0.5772156649) - (2.0 * (subsample_size - 1) / max(subsample_size, 1))

        paths = np.stack([t.path_length(arr) for t in self.trees], axis=0)  # [n_trees, N]
        mean_path = np.mean(paths, axis=0)
        # s(x, n) = 2^(-E(h(x)) / c)
        anomaly_scores = 2.0 ** (-mean_path / max(c, 1e-12))
        return anomaly_scores

    def predict(self, x: Any, threshold: float = 0.5) -> np.ndarray:
        """Returns binary anomaly indicators (1 = anomaly, 0 = normal) given a score threshold."""
        scores = self.score(x)
        return (scores >= float(threshold)).astype(np.int64)


__all__ = [
    "EnergyScore",
    "MahalanobisDistance",
    "IsolationForest",
]
