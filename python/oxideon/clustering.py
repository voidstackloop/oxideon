"""Unsupervised representation learning, clustering, and dimensionality reduction for Oxideon.

Enables feature discovery, compression, and cluster analysis on Tensors:
  - pairwise_distances: High-throughput distance matrix computation (Euclidean, Cosine, Dot)
  - KMeans: Fast k-means clustering with inertia tracking and k-means++ initialization
  - PCA: Principal Component Analysis via covariance matrix spectral decomposition
"""

from typing import Optional, Union, Any, Tuple
import numpy as np

from .oxideon import Tensor as _Tensor, from_numpy as _from_numpy


def pairwise_distances(
    x: Any, y: Optional[Any] = None, metric: str = "euclidean"
) -> _Tensor:
    """Computes pairwise distance matrix between rows of X and Y."""
    has_grad = (isinstance(x, _Tensor) and x.requires_grad) or (isinstance(y, _Tensor) and y.requires_grad)
    if has_grad:
        x_t = x if isinstance(x, _Tensor) else _from_numpy(np.ascontiguousarray(x, dtype=np.float32))
        y_t = x_t if y is None else (y if isinstance(y, _Tensor) else _from_numpy(np.ascontiguousarray(y, dtype=np.float32)))
        if x_t.ndim == 1:
            x_t = x_t.reshape([1, -1])
        if y_t.ndim == 1:
            y_t = y_t.reshape([1, -1])
        if metric == "euclidean":
            x_norm = (x_t * x_t).sum(dim=1, keepdim=True)
            y_norm = (y_t * y_t).sum(dim=1, keepdim=True).transpose(0, 1)
            dist_sq = (x_norm + y_norm - x_t.matmul(y_t.transpose(0, 1)) * 2.0).relu()
            return (dist_sq + 1e-12).sqrt()
        elif metric == "cosine":
            x_n = ((x_t * x_t).sum(dim=1, keepdim=True) + 1e-12).sqrt()
            y_n = ((y_t * y_t).sum(dim=1, keepdim=True) + 1e-12).sqrt()
            sim = (x_t / x_n).matmul((y_t / y_n).transpose(0, 1)).clamp(min=-1.0, max=1.0)
            return 1.0 - sim
        elif metric == "dot":
            return x_t.matmul(y_t.transpose(0, 1))
        else:
            raise ValueError(f"Unknown metric '{metric}' (choose 'euclidean', 'cosine', 'dot')")

    x_np = x.to_numpy() if isinstance(x, _Tensor) else np.asarray(x, dtype=np.float32)
    y_np = x_np if y is None else (y.to_numpy() if isinstance(y, _Tensor) else np.asarray(y, dtype=np.float32))

    if x_np.ndim == 1:
        x_np = np.expand_dims(x_np, 0)
    if y_np.ndim == 1:
        y_np = np.expand_dims(y_np, 0)

    if metric == "euclidean":
        x_norm = np.sum(x_np**2, axis=1, keepdims=True)
        y_norm = np.sum(y_np**2, axis=1, keepdims=True)
        dist_sq = np.maximum(0.0, x_norm + y_norm.T - 2.0 * np.dot(x_np, y_np.T))
        dist = np.sqrt(dist_sq)
    elif metric == "cosine":
        x_norm = np.linalg.norm(x_np, axis=1, keepdims=True)
        y_norm = np.linalg.norm(y_np, axis=1, keepdims=True)
        x_norm = np.maximum(x_norm, 1e-12)
        y_norm = np.maximum(y_norm, 1e-12)
        sim = np.dot(x_np / x_norm, (y_np / y_norm).T)
        dist = 1.0 - np.clip(sim, -1.0, 1.0)
    elif metric == "dot":
        dist = np.dot(x_np, y_np.T)
    else:
        raise ValueError(f"Unknown metric '{metric}' (choose 'euclidean', 'cosine', 'dot')")

    return _from_numpy(np.ascontiguousarray(dist, dtype=np.float32))


class KMeans:
    """K-Means vector quantization and clustering algorithm."""

    def __init__(
        self,
        n_clusters: int = 8,
        max_iter: int = 300,
        tol: float = 1e-4,
        random_state: Optional[int] = None,
    ):
        if n_clusters < 1:
            raise ValueError(f"n_clusters must be >= 1, got {n_clusters}")
        if max_iter < 1:
            raise ValueError(f"max_iter must be >= 1, got {max_iter}")
        if tol < 0.0:
            raise ValueError(f"tol must be >= 0, got {tol}")
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state

        self.cluster_centers_ = None
        self.inertia_ = None
        self.labels_ = None

    def fit(self, x: Any) -> "KMeans":
        """Fits k-means clusters onto dataset X."""
        x_np = x.to_numpy() if isinstance(x, _Tensor) else np.asarray(x, dtype=np.float32)
        if x_np.ndim == 1:
            x_np = np.expand_dims(x_np, 0)
        n_samples, n_features = x_np.shape

        if n_samples < self.n_clusters:
            raise ValueError(f"n_samples ({n_samples}) must be >= n_clusters ({self.n_clusters})")

        rng = np.random.RandomState(self.random_state)

        # Precompute x_sq once for fast GEMM-based distance evaluation: ||x - c||^2 = ||x||^2 + ||c||^2 - 2 x.c
        x_sq = np.sum(x_np ** 2, axis=1, keepdims=True)

        # k-means++ initialization
        centers = [x_np[rng.choice(n_samples)]]
        for _ in range(1, self.n_clusters):
            curr_c = np.array(centers, dtype=np.float32)
            c_sq = np.sum(curr_c ** 2, axis=1, keepdims=True).T
            dists_sq = np.maximum(0.0, x_sq + c_sq - 2.0 * np.dot(x_np, curr_c.T))
            dists = np.min(dists_sq, axis=1)
            total_d = float(np.sum(dists))
            if total_d > 1e-12:
                probs = dists / total_d
                next_center = x_np[rng.choice(n_samples, p=probs)]
            else:
                next_center = x_np[rng.choice(n_samples)]
            centers.append(next_center)
        centers = np.array(centers, dtype=np.float32)

        # Iterative expectation-maximization
        for it in range(self.max_iter):
            c_sq = np.sum(centers ** 2, axis=1, keepdims=True).T
            dists = np.maximum(0.0, x_sq + c_sq - 2.0 * np.dot(x_np, centers.T))
            labels = np.argmin(dists, axis=1)

            new_centers = np.zeros_like(centers)
            for k in range(self.n_clusters):
                mask = labels == k
                if np.any(mask):
                    new_centers[k] = np.mean(x_np[mask], axis=0)
                else:
                    new_centers[k] = x_np[rng.choice(n_samples)]

            shift = np.linalg.norm(new_centers - centers)
            centers = new_centers
            if shift < self.tol:
                break

        self.cluster_centers_ = _from_numpy(np.ascontiguousarray(centers, dtype=np.float32))
        self.labels_ = labels
        c_sq = np.sum(centers ** 2, axis=1, keepdims=True).T
        dists = np.maximum(0.0, x_sq + c_sq - 2.0 * np.dot(x_np, centers.T))
        self.inertia_ = float(np.sum(np.min(dists, axis=1)))
        return self

    def predict(self, x: Any) -> np.ndarray:
        """Assigns samples in X to nearest cluster center."""
        if self.cluster_centers_ is None:
            raise RuntimeError("KMeans instance is not fitted yet.")
        x_np = x.to_numpy() if isinstance(x, _Tensor) else np.asarray(x, dtype=np.float32)
        if x_np.ndim == 1:
            x_np = np.expand_dims(x_np, 0)
        centers = self.cluster_centers_.to_numpy()
        x_sq = np.sum(x_np ** 2, axis=1, keepdims=True)
        c_sq = np.sum(centers ** 2, axis=1, keepdims=True).T
        dists = np.maximum(0.0, x_sq + c_sq - 2.0 * np.dot(x_np, centers.T))
        return np.argmin(dists, axis=1)

    def fit_predict(self, x: Any) -> np.ndarray:
        self.fit(x)
        return self.labels_


class PCA:
    """Principal Component Analysis (PCA) dimensionality reduction."""

    def __init__(self, n_components: int = 2):
        if n_components < 1:
            raise ValueError(f"n_components must be >= 1, got {n_components}")
        self.n_components = n_components
        self.components_ = None
        self.explained_variance_ = None
        self.explained_variance_ratio_ = None
        self.mean_ = None

    def fit(self, x: Any) -> "PCA":
        """Fits PCA components on data matrix X."""
        x_np = x.to_numpy() if isinstance(x, _Tensor) else np.asarray(x, dtype=np.float32)
        if x_np.ndim == 1:
            x_np = np.expand_dims(x_np, 0)
        n_samples, n_features = x_np.shape

        if self.n_components > n_features:
            raise ValueError(f"n_components ({self.n_components}) must be <= n_features ({n_features})")

        mean = np.mean(x_np, axis=0, keepdims=True)
        self.mean_ = _from_numpy(np.ascontiguousarray(mean, dtype=np.float32))

        centered = x_np - mean
        cov = (centered.T @ centered) / max(n_samples - 1, 1)

        eigvals, eigvecs = np.linalg.eigh(cov)
        idx = np.argsort(eigvals)[::-1]
        sorted_eigvals = eigvals[idx]
        sorted_eigvecs = eigvecs[:, idx]

        total_variance = np.sum(np.maximum(0.0, sorted_eigvals))
        components = sorted_eigvecs[:, : self.n_components].T
        variances = np.maximum(0.0, sorted_eigvals[: self.n_components])

        self.components_ = _from_numpy(np.ascontiguousarray(components, dtype=np.float32))
        self.explained_variance_ = variances
        self.explained_variance_ratio_ = variances / max(total_variance, 1e-12)
        return self

    def transform(self, x: Any) -> _Tensor:
        """Projects data X onto principal components."""
        if self.components_ is None or self.mean_ is None:
            raise RuntimeError("PCA instance is not fitted yet.")
        if isinstance(x, _Tensor) and x.requires_grad:
            x_t = x.reshape([1, -1]) if x.ndim == 1 else x
            return (x_t - self.mean_).matmul(self.components_.transpose(0, 1))
        x_np = x.to_numpy() if isinstance(x, _Tensor) else np.asarray(x, dtype=np.float32)
        if x_np.ndim == 1:
            x_np = np.expand_dims(x_np, 0)
        mean_np = self.mean_.to_numpy()
        components_np = self.components_.to_numpy()

        centered = x_np - mean_np
        proj = centered @ components_np.T
        return _from_numpy(np.ascontiguousarray(proj, dtype=np.float32))

    def fit_transform(self, x: Any) -> _Tensor:
        self.fit(x)
        return self.transform(x)

    def inverse_transform(self, x_proj: Any) -> _Tensor:
        """Reconstructs original feature space from projected components."""
        if self.components_ is None or self.mean_ is None:
            raise RuntimeError("PCA instance is not fitted yet.")
        if isinstance(x_proj, _Tensor) and x_proj.requires_grad:
            proj_t = x_proj.reshape([1, -1]) if x_proj.ndim == 1 else x_proj
            return proj_t.matmul(self.components_) + self.mean_
        proj_np = x_proj.to_numpy() if isinstance(x_proj, _Tensor) else np.asarray(x_proj, dtype=np.float32)
        if proj_np.ndim == 1:
            proj_np = np.expand_dims(proj_np, 0)
        mean_np = self.mean_.to_numpy()
        components_np = self.components_.to_numpy()

        reconstructed = proj_np @ components_np + mean_np
        return _from_numpy(np.ascontiguousarray(reconstructed, dtype=np.float32))


__all__ = [
    "pairwise_distances",
    "KMeans",
    "PCA",
]
