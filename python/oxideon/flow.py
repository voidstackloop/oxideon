"""Normalizing Flows and Invertible Neural Networks (RealNVP) for Oxideon.

Enables exact likelihood computation, density estimation, and generative sampling:
  - AffineCouplingLayer: Split-scale-shift coupling layer with exact Jacobian log-determinant
  - InvertibleLinear: Square invertible transformation with determinant tracking
  - RealNVP: Deep normalizing flow architecture with forward density and inverse sampling
"""

from typing import Tuple, Optional, Any, List
import math
import numpy as np

from .oxideon import Tensor as _Tensor, from_numpy as _from_numpy
from .nn import Module, Linear, Sequential, ReLU, Parameter, _check_repeats


class InvertibleLinear(Module):
    """Square invertible linear transformation y = x @ W^T with exact log|det(W)|."""

    def __init__(self, dim: int):
        super().__init__()
        if dim < 1:
            raise ValueError(f"dim must be >= 1, got {dim}")
        self.dim = dim
        q, _ = np.linalg.qr(np.random.randn(dim, dim).astype(np.float32))
        if np.linalg.det(q) < 0:
            q[:, 0] = -q[:, 0]
        self.weight = Parameter(_from_numpy(np.ascontiguousarray(q, dtype=np.float32)))

    def forward(self, x: Any) -> Tuple[_Tensor, _Tensor]:
        arr = x.to_numpy() if isinstance(x, _Tensor) else np.asarray(x, dtype=np.float32)
        is_1d = arr.ndim == 1
        if is_1d:
            arr = np.expand_dims(arr, 0)
        elif arr.ndim != 2:
            raise ValueError(f"InvertibleLinear expects 1D or 2D input, got shape {arr.shape}")
        if arr.shape[1] != self.dim:
            raise ValueError(f"Feature dimension mismatch: expected {self.dim}, got {arr.shape[1]}")

        w = self.weight.to_numpy()
        _, logabsdet = np.linalg.slogdet(w)

        if (isinstance(x, _Tensor) and x.requires_grad) or self.weight.requires_grad:
            x_t = x if isinstance(x, _Tensor) else _from_numpy(np.ascontiguousarray(arr, dtype=np.float32))
            if is_1d:
                x_t = x_t.reshape([1, self.dim])
            y_t = x_t.matmul(self.weight.transpose(0, 1))
            if is_1d:
                y_t = y_t.reshape([self.dim])
            # Gradient of log|det(W)| w.r.t. W is (W^{-1})^T; attach first-order surrogate
            w_inv_t = _from_numpy(np.ascontiguousarray(np.linalg.inv(w).T, dtype=np.float32))
            surrogate = (self.weight * w_inv_t).sum()
            base_ld = float(logabsdet) - float(surrogate.to_numpy().item())
            ld_single = surrogate + base_ld
            if is_1d:
                log_det_t = ld_single
            else:
                log_det_t = _Tensor.ones([arr.shape[0]]) * ld_single
            return y_t, log_det_t

        y = arr @ w.T
        log_det = np.full(arr.shape[0], float(logabsdet), dtype=np.float32)
        if is_1d:
            y = y.squeeze(0)
            log_det = log_det.squeeze(0)
        return (
            _from_numpy(np.ascontiguousarray(y, dtype=np.float32)),
            _from_numpy(np.ascontiguousarray(log_det, dtype=np.float32)),
        )

    def inverse(self, y: Any) -> _Tensor:
        arr = y.to_numpy() if isinstance(y, _Tensor) else np.asarray(y, dtype=np.float32)
        is_1d = arr.ndim == 1
        if is_1d:
            arr = np.expand_dims(arr, 0)
        elif arr.ndim != 2:
            raise ValueError(f"InvertibleLinear expects 1D or 2D input, got shape {arr.shape}")
        if arr.shape[1] != self.dim:
            raise ValueError(f"Feature dimension mismatch: expected {self.dim}, got {arr.shape[1]}")

        w_inv = np.linalg.inv(self.weight.to_numpy())
        if isinstance(y, _Tensor) and y.requires_grad:
            y_t = y.reshape([1, self.dim]) if is_1d else y
            w_inv_t = _from_numpy(np.ascontiguousarray(w_inv.T, dtype=np.float32))
            x_t = y_t.matmul(w_inv_t)
            return x_t.reshape([self.dim]) if is_1d else x_t
        x = arr @ w_inv.T
        if is_1d:
            x = x.squeeze(0)
        return _from_numpy(np.ascontiguousarray(x, dtype=np.float32))


class AffineCouplingLayer(Module):
    """RealNVP-style affine coupling layer (Dinh et al., 2017)."""

    def __init__(self, dim: int, hidden_dim: int = 64, mask_type: str = "even"):
        super().__init__()
        if dim < 2:
            raise ValueError(f"dim must be >= 2 for AffineCouplingLayer, got {dim}")
        if hidden_dim < 1:
            raise ValueError(f"hidden_dim must be >= 1, got {hidden_dim}")
        if mask_type not in ("even", "odd"):
            raise ValueError(f"mask_type must be 'even' or 'odd', got '{mask_type}'")
        self.dim = dim
        self.split_dim = dim // 2
        self.rest_dim = dim - self.split_dim
        self.mask_type = mask_type

        self.net = Sequential(
            Linear(self.split_dim, hidden_dim),
            ReLU(),
            Linear(hidden_dim, 2 * self.rest_dim),
        )

    def forward(self, x: Any) -> Tuple[_Tensor, _Tensor]:
        """Maps x -> y and computes log-determinant of Jacobian while preserving Tensor autograd."""
        arr = x.to_numpy() if isinstance(x, _Tensor) else np.asarray(x, dtype=np.float32)
        is_1d = arr.ndim == 1
        if is_1d:
            arr = np.expand_dims(arr, 0)
        elif arr.ndim != 2:
            raise ValueError(f"AffineCouplingLayer expects 1D or 2D input, got shape {arr.shape}")
        if arr.shape[1] != self.dim:
            raise ValueError(f"Feature dimension mismatch: expected {self.dim}, got {arr.shape[1]}")

        x_t = x if isinstance(x, _Tensor) else _from_numpy(np.ascontiguousarray(arr, dtype=np.float32))
        if is_1d:
            x_t = x_t.reshape([1, self.dim])

        if self.mask_type == "even":
            x1 = x_t.narrow(1, 0, self.split_dim)
            x2 = x_t.narrow(1, self.split_dim, self.rest_dim)
        else:
            x2 = x_t.narrow(1, 0, self.rest_dim)
            x1 = x_t.narrow(1, self.rest_dim, self.split_dim)

        st = self.net(x1)
        s = st.narrow(1, 0, self.rest_dim).tanh()
        t = st.narrow(1, self.rest_dim, self.rest_dim)

        y1 = x1
        y2 = x2 * s.exp() + t

        if self.mask_type == "even":
            y = _Tensor.cat([y1, y2], dim=1)
        else:
            y = _Tensor.cat([y2, y1], dim=1)

        log_det = s.sum(dim=1)
        if is_1d:
            y = y.reshape([self.dim])
            log_det = log_det.reshape([])

        return y, log_det

    def inverse(self, y: Any) -> _Tensor:
        """Inverts transformation y -> x."""
        arr = y.to_numpy() if isinstance(y, _Tensor) else np.asarray(y, dtype=np.float32)
        is_1d = arr.ndim == 1
        if is_1d:
            arr = np.expand_dims(arr, 0)
        elif arr.ndim != 2:
            raise ValueError(f"AffineCouplingLayer expects 1D or 2D input, got shape {arr.shape}")
        if arr.shape[1] != self.dim:
            raise ValueError(f"Feature dimension mismatch: expected {self.dim}, got {arr.shape[1]}")

        y_t = y if isinstance(y, _Tensor) else _from_numpy(np.ascontiguousarray(arr, dtype=np.float32))
        if is_1d:
            y_t = y_t.reshape([1, self.dim])

        if self.mask_type == "even":
            y1 = y_t.narrow(1, 0, self.split_dim)
            y2 = y_t.narrow(1, self.split_dim, self.rest_dim)
        else:
            y2 = y_t.narrow(1, 0, self.rest_dim)
            y1 = y_t.narrow(1, self.rest_dim, self.split_dim)

        st = self.net(y1)
        s = st.narrow(1, 0, self.rest_dim).tanh()
        t = st.narrow(1, self.rest_dim, self.rest_dim)

        x1 = y1
        x2 = (y2 - t) * (-s).exp()

        if self.mask_type == "even":
            x = _Tensor.cat([x1, x2], dim=1)
        else:
            x = _Tensor.cat([x2, x1], dim=1)

        if is_1d:
            x = x.reshape([self.dim])

        return x


class RealNVP(Module):
    """Deep invertible Normalizing Flow architecture composed of Affine Coupling layers."""

    def __init__(self, in_features: int, num_layers: int = 4, hidden_dim: int = 64):
        super().__init__()
        if in_features < 2:
            raise ValueError(f"in_features must be >= 2 for RealNVP, got {in_features}")
        num_layers = _check_repeats("num_layers", num_layers)
        if hidden_dim < 1:
            raise ValueError(f"hidden_dim must be >= 1, got {hidden_dim}")
        self.in_features = in_features
        self.layers = []
        for i in range(num_layers):
            mask = "even" if i % 2 == 0 else "odd"
            layer = AffineCouplingLayer(in_features, hidden_dim=hidden_dim, mask_type=mask)
            setattr(self, f"coupling_{i}", layer)
            self.layers.append(layer)

    def forward(self, x: Any) -> Tuple[_Tensor, _Tensor]:
        """Transforms data space x to latent space z, tracking total log-det."""
        arr = x.to_numpy() if isinstance(x, _Tensor) else np.asarray(x, dtype=np.float32)
        is_1d = arr.ndim == 1
        if is_1d:
            arr = np.expand_dims(arr, 0)
        elif arr.ndim != 2:
            raise ValueError(f"RealNVP expects 1D or 2D input, got shape {arr.shape}")
        if arr.shape[1] != self.in_features:
            raise ValueError(f"Feature dimension mismatch: expected {self.in_features}, got {arr.shape[1]}")

        z = x if isinstance(x, _Tensor) else _from_numpy(np.ascontiguousarray(arr, dtype=np.float32))
        if is_1d:
            z = z.reshape([1, self.in_features])
        total_log_det = _Tensor.zeros([arr.shape[0]])
        for layer in self.layers:
            z, log_det = layer(z)
            total_log_det = total_log_det + log_det

        if is_1d:
            z = z.reshape([self.in_features])
            total_log_det = total_log_det.reshape([])
        return z, total_log_det

    def inverse(self, z: Any) -> _Tensor:
        """Maps latent space z back to data space x."""
        x = z if isinstance(z, _Tensor) else _from_numpy(np.ascontiguousarray(z, dtype=np.float32))
        for layer in reversed(self.layers):
            x = layer.inverse(x)
        return x

    def log_prob(self, x: Any) -> _Tensor:
        """Computes exact log p(x) under standard Gaussian prior p(z)."""
        arr = x.to_numpy() if isinstance(x, _Tensor) else np.asarray(x, dtype=np.float32)
        is_1d = arr.ndim == 1
        x_in = x if isinstance(x, _Tensor) else _from_numpy(np.ascontiguousarray(arr, dtype=np.float32))
        if is_1d:
            x_in = x_in.reshape([1, self.in_features])
        z, log_det = self.forward(x_in)
        dim = self.in_features
        log_pz = ((z * z).sum(dim=1) + (dim * math.log(2.0 * math.pi))) * (-0.5)
        log_px = log_pz + log_det
        if is_1d:
            log_px = log_px.reshape([])
        return log_px

    def sample(self, num_samples: int, seed: Optional[int] = None) -> _Tensor:
        """Generates synthetic samples by mapping Gaussian noise through inverse flow."""
        if num_samples < 1:
            raise ValueError(f"num_samples must be >= 1, got {num_samples}")
        rng = np.random.default_rng(seed)
        z_noise = rng.standard_normal((num_samples, self.in_features)).astype(np.float32)
        return self.inverse(_from_numpy(z_noise))


__all__ = [
    "InvertibleLinear",
    "AffineCouplingLayer",
    "RealNVP",
]
