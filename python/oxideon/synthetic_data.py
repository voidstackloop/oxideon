"""Generative Tabular and Latent Synthetic Data Generators for Oxideon.

Enables synthesizing high-fidelity synthetic datasets while preserving statistical correlations:
  - GaussianCopulaSynthesizer: Generates tabular data preserving marginals and Gaussian copula correlations
  - CTGANGenerator: Conditional generator for complex tabular data (Xu et al., 2019)
  - DifferentialPrivacySynthesizer: Differentially private noisy marginal synthesizer
  - make_synthetic_classification: Deterministic benchmark dataset generation
"""

from typing import Tuple, Optional, Any, List, Dict
import numpy as np

from .oxideon import Tensor as _Tensor, from_numpy as _from_numpy
from .nn import Module, Linear, Sequential, ReLU, Parameter


def make_synthetic_classification(
    n_samples: int = 100,
    n_features: int = 4,
    n_classes: int = 2,
    seed: Optional[int] = None,
) -> Tuple[_Tensor, _Tensor]:
    """Generates synthetic multi-class feature matrix X and label vector y."""
    if n_samples < n_classes:
        raise ValueError(f"n_samples ({n_samples}) must be >= n_classes ({n_classes})")
    if n_features < 1 or n_classes < 1:
        raise ValueError("n_features and n_classes must be >= 1")

    rng = np.random.RandomState(seed) if seed is not None else np.random
    samples_per_class = n_samples // n_classes
    remainder = n_samples % n_classes

    features_list = []
    labels_list = []

    # Assign distinct cluster center for each class
    centers = rng.uniform(-3.0, 3.0, size=(n_classes, n_features))

    for c in range(n_classes):
        count = samples_per_class + (1 if c < remainder else 0)
        c_features = rng.randn(count, n_features).astype(np.float32) + centers[c].astype(np.float32)
        features_list.append(c_features)
        labels_list.append(np.full(count, c, dtype=np.int64))

    X = np.concatenate(features_list, axis=0)
    y = np.concatenate(labels_list, axis=0)

    # Shuffle
    perm = rng.permutation(n_samples)
    X = X[perm]
    y = y[perm]

    return _from_numpy(np.ascontiguousarray(X)), _from_numpy(np.ascontiguousarray(y))


class GaussianCopulaSynthesizer:
    """Gaussian Copula tabular data generator preserving feature covariance."""

    def __init__(self):
        self.means: Optional[np.ndarray] = None
        self.stds: Optional[np.ndarray] = None
        self.corr_matrix: Optional[np.ndarray] = None
        self.n_features: int = 0

    def fit(self, x: Any) -> "GaussianCopulaSynthesizer":
        arr = x.to_numpy() if isinstance(x, _Tensor) else np.asarray(x, dtype=np.float32)
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)
        if len(arr) < 2:
            raise ValueError(f"GaussianCopulaSynthesizer requires at least 2 samples, got {len(arr)}")
        self.n_features = arr.shape[1]
        self.means = np.mean(arr, axis=0)
        self.stds = np.std(arr, axis=0) + 1e-8

        # Standardize
        z = (arr - self.means) / self.stds
        # Correlation matrix
        self.corr_matrix = np.corrcoef(z, rowvar=False)
        if self.corr_matrix.ndim == 0:
            self.corr_matrix = self.corr_matrix.reshape(1, 1)
        # Ensure positive semidefinite
        self.corr_matrix = np.nan_to_num(self.corr_matrix, nan=0.0)
        np.fill_diagonal(self.corr_matrix, 1.0)
        eigvals, eigvecs = np.linalg.eigh(self.corr_matrix)
        self.corr_matrix = eigvecs @ np.diag(np.maximum(eigvals, 1e-5)) @ eigvecs.T

        return self

    def sample(self, n_samples: int, seed: Optional[int] = None) -> _Tensor:
        """Generates synthetic tabular samples following the learned copula."""
        if n_samples < 0:
            raise ValueError(f"n_samples must be >= 0, got {n_samples}")
        if self.corr_matrix is None or self.means is None or self.stds is None:
            raise RuntimeError("Synthesizer is not fitted yet.")
        if n_samples == 0:
            return _from_numpy(np.empty((0, self.n_features), dtype=np.float32))

        rng = np.random.RandomState(seed) if seed is not None else np.random
        # Draw correlated standard normals
        z = rng.multivariate_normal(
            mean=np.zeros(self.n_features), cov=self.corr_matrix, size=n_samples
        ).astype(np.float32)

        # Invert standardization
        synth = z * self.stds + self.means
        return _from_numpy(np.ascontiguousarray(synth, dtype=np.float32))


class CTGANGenerator(Module):
    """Conditional Tabular GAN generator network."""

    def __init__(self, latent_dim: int = 16, output_dim: int = 10, hidden_dim: int = 32):
        super().__init__()
        if latent_dim <= 0 or output_dim <= 0 or hidden_dim <= 0:
            raise ValueError(f"latent_dim, output_dim, and hidden_dim must be positive, got {latent_dim}, {output_dim}, {hidden_dim}")
        self.latent_dim = latent_dim
        self.net = Sequential(
            Linear(latent_dim, hidden_dim),
            ReLU(),
            Linear(hidden_dim, hidden_dim),
            ReLU(),
            Linear(hidden_dim, output_dim),
        )

    def forward(self, noise: _Tensor) -> _Tensor:
        return self.net(noise)

    def sample(self, n_samples: int, seed: Optional[int] = None) -> _Tensor:
        if n_samples < 0:
            raise ValueError(f"n_samples must be >= 0, got {n_samples}")
        rng = np.random.RandomState(seed) if seed is not None else np.random
        noise = _from_numpy(rng.randn(n_samples, self.latent_dim).astype(np.float32))
        return self.forward(noise)


class DifferentialPrivacySynthesizer:
    """Synthesizes 1D/2D empirical distributions under epsilon-Laplace privacy budget."""

    def __init__(self, epsilon: float = 1.0, num_bins: int = 10):
        if float(epsilon) <= 0.0:
            raise ValueError(f"epsilon must be positive (> 0), got {epsilon}")
        if int(num_bins) < 1:
            raise ValueError(f"num_bins must be >= 1, got {num_bins}")
        self.epsilon = float(epsilon)
        self.num_bins = int(num_bins)
        self.hist: Optional[np.ndarray] = None
        self.bin_edges: Optional[np.ndarray] = None

    def fit(self, x: Any, seed: Optional[int] = None) -> "DifferentialPrivacySynthesizer":
        arr = x.to_numpy() if isinstance(x, _Tensor) else np.asarray(x, dtype=np.float32).reshape(-1)
        hist, edges = np.histogram(arr, bins=self.num_bins)

        rng = np.random.RandomState(seed) if seed is not None else np.random
        # Add Laplace noise with scale = 1.0 / epsilon
        scale = 1.0 / max(self.epsilon, 1e-4)
        noise = rng.laplace(0.0, scale, size=hist.shape)
        noisy_hist = np.maximum(0.0, hist.astype(np.float32) + noise)

        total = np.sum(noisy_hist)
        if total > 0:
            self.hist = (noisy_hist / total)
        else:
            self.hist = np.full(self.num_bins, 1.0 / self.num_bins, dtype=np.float32)
        self.bin_edges = edges
        return self

    def sample(self, n_samples: int, seed: Optional[int] = None) -> _Tensor:
        if self.hist is None or self.bin_edges is None:
            raise RuntimeError("Synthesizer is not fitted yet.")

        p = self.hist
        total_p = np.sum(p)
        if total_p <= 0 or not np.isfinite(total_p):
            p = np.full(self.num_bins, 1.0 / self.num_bins, dtype=np.float32)
        else:
            p = p / total_p

        rng = np.random.RandomState(seed) if seed is not None else np.random
        bin_indices = rng.choice(self.num_bins, size=n_samples, p=p)
        lows = self.bin_edges[bin_indices]
        highs = self.bin_edges[bin_indices + 1]

        # Uniform interpolation within chosen bins
        samples = rng.uniform(lows, highs).astype(np.float32)
        return _from_numpy(np.ascontiguousarray(samples))


__all__ = [
    "GaussianCopulaSynthesizer",
    "CTGANGenerator",
    "DifferentialPrivacySynthesizer",
    "make_synthetic_classification",
]
