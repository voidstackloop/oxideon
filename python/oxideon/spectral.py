"""Spectral Convolutions, Chebyshev Graph Filters, and Density of States for Oxideon.

Enables spectral graph learning and eigenvalue spectrum analysis:
  - chebyshev_polynomials: Computes recursive orthogonal Chebyshev basis matrices
  - ChebyshevConv: Fast spectral graph convolution layer (Defferrard et al., 2016)
  - spectral_density_of_states: Computes empirical spectral density and condition numbers
"""

from typing import Tuple, List, Optional, Any
import numpy as np

from .oxideon import Tensor as _Tensor, from_numpy as _from_numpy
from .nn import Module, Parameter


def chebyshev_polynomials(laplacian: Any, K: int = 3) -> List[_Tensor]:
    """Computes K Chebyshev polynomial basis matrices T_0..T_{K-1}."""
    if K < 1:
        raise ValueError("K must be >= 1")
    l = laplacian.to_numpy() if isinstance(laplacian, _Tensor) else np.asarray(laplacian, dtype=np.float32)
    if l.ndim != 2 or l.shape[0] != l.shape[1]:
        raise ValueError(f"laplacian must be a square 2D matrix, got shape {l.shape}")
    n = l.shape[0]

    # Scaled Laplacian: \tilde{L} = (2 / lambda_max) * L - I
    eigvals = np.linalg.eigvalsh(l)
    lambda_max = float(np.max(eigvals)) if len(eigvals) > 0 else 2.0
    l_tilde = (2.0 / max(lambda_max, 1e-4)) * l - np.eye(n, dtype=np.float32)

    t_list = []
    t0 = np.eye(n, dtype=np.float32)
    t_list.append(_from_numpy(t0))

    if K > 1:
        t1 = l_tilde.copy()
        t_list.append(_from_numpy(t1))

    for k in range(2, K):
        t_prev = t_list[-1].to_numpy()
        t_prev2 = t_list[-2].to_numpy()
        tk = 2.0 * (l_tilde @ t_prev) - t_prev2
        t_list.append(_from_numpy(np.ascontiguousarray(tk, dtype=np.float32)))

    return t_list


class ChebyshevConv(Module):
    """Chebyshev Spectral Graph Convolutional Layer (Defferrard et al., 2016)."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        K: int = 3,
        bias: bool = True,
        seed: Optional[int] = None,
    ):
        super().__init__()
        if in_features < 1:
            raise ValueError(f"in_features must be >= 1, got {in_features}")
        if out_features < 1:
            raise ValueError(f"out_features must be >= 1, got {out_features}")
        if K < 1:
            raise ValueError(f"K must be >= 1, got {K}")
        self.in_features = in_features
        self.out_features = out_features
        self.K = K

        scale = np.sqrt(2.0 / (K * in_features))
        rng = np.random.RandomState(seed) if seed is not None else np.random
        w = rng.uniform(-scale, scale, (K, in_features, out_features)).astype(np.float32)
        self.weight = Parameter(_from_numpy(w))

        if bias:
            b = np.zeros(out_features, dtype=np.float32)
            self.bias = Parameter(_from_numpy(b))
        else:
            self.bias = None

    def forward(self, x: Any, chebyshev_bases: List[_Tensor]) -> _Tensor:
        """Convolves node features X with K Chebyshev bases: out = sum_{k=0}^{K-1} T_k X W_k."""
        if len(chebyshev_bases) < self.K:
            raise ValueError(
                f"chebyshev_bases has length {len(chebyshev_bases)}, but ChebyshevConv expects K={self.K}"
            )

        x_t = x if isinstance(x, _Tensor) else _from_numpy(np.asarray(x, dtype=np.float32))
        if x_t.ndim == 1:
            x_t = x_t.unsqueeze(0)

        out_tensor = None
        for k in range(self.K):
            t_k = chebyshev_bases[k] if isinstance(chebyshev_bases[k], _Tensor) else _from_numpy(np.asarray(chebyshev_bases[k], dtype=np.float32))
            filtered = t_k.matmul(x_t)
            w_k = self.weight[k]
            term = filtered.matmul(w_k)
            out_tensor = term if out_tensor is None else out_tensor + term

        if self.bias is not None:
            out_tensor = out_tensor + self.bias
        return out_tensor


def spectral_density_of_states(matrix: Any) -> Tuple[np.ndarray, float]:
    """Computes empirical eigenvalues and condition number of symmetric/Gram matrix."""
    arr = matrix.to_numpy() if isinstance(matrix, _Tensor) else np.asarray(matrix, dtype=np.float32)
    if arr.size == 0:
        return np.zeros(0, dtype=np.float32), 1.0
    if arr.ndim != 2 or arr.shape[0] != arr.shape[1]:
        raise ValueError(f"matrix must be a square 2D matrix, got shape {arr.shape}")
    eigvals = np.linalg.eigvalsh(arr)
    min_eig = float(np.min(np.abs(eigvals)))
    max_eig = float(np.max(np.abs(eigvals)))
    cond = max_eig / max(min_eig, 1e-12)
    return eigvals, cond


__all__ = [
    "chebyshev_polynomials",
    "ChebyshevConv",
    "spectral_density_of_states",
]
