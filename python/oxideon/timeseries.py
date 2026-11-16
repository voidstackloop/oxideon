"""Time-Series analysis, decomposition, feature engineering, and neural forecasting for Oxideon.

Provides classical and modern deep learning tools for temporal data:
  - seasonal_decompose: Classical trend-seasonal-residual decomposition
  - lag_features: Autoregressive lag matrix generator
  - rolling_window: Sliding sequence window generator for temporal prediction
  - exponential_smoothing: Single exponential smoothing filter
  - PatchTST: Modern Patch Time-Series Transformer for long-horizon forecasting
"""

from typing import List, Tuple, Optional, Any, Union, Iterator
import math
import numpy as np

from .oxideon import Tensor as _Tensor, from_numpy as _from_numpy
from .nn import Module, Linear, Sequential, ReLU, LayerNorm, Parameter


class DecomposeResult:
    """Holds trend, seasonal, and residual components of time series decomposition."""

    def __init__(self, trend: np.ndarray, seasonal: np.ndarray, resid: np.ndarray):
        self.trend = _from_numpy(np.ascontiguousarray(trend, dtype=np.float32))
        self.seasonal = _from_numpy(np.ascontiguousarray(seasonal, dtype=np.float32))
        self.resid = _from_numpy(np.ascontiguousarray(resid, dtype=np.float32))

    def __iter__(self) -> Iterator[_Tensor]:
        yield self.trend
        yield self.seasonal
        yield self.resid

    def __repr__(self) -> str:
        return f"DecomposeResult(length={len(self.trend)})"


def seasonal_decompose(
    series: Any, period: int, model: str = "additive"
) -> DecomposeResult:
    """Decomposes a 1D time series into trend, seasonal, and residual components."""
    if period < 2:
        raise ValueError(f"period must be >= 2, got {period}")
    arr = series.to_numpy() if isinstance(series, _Tensor) else np.asarray(series, dtype=np.float32).reshape(-1)
    n = len(arr)
    if n < 2 * period:
        raise ValueError(f"Series length {n} must be at least twice period {period}")

    # 1. Moving average trend via fast 1D convolution
    half = period // 2
    w_size = period if period % 2 == 1 else period
    trend = np.full(n, np.nan, dtype=np.float32)
    kernel = np.ones(w_size, dtype=np.float32) / float(w_size)
    ma = np.convolve(arr, kernel, mode="valid")
    trend[half : half + len(ma)] = ma

    # Detrended
    if model == "additive":
        detrended = arr - trend
    elif model == "multiplicative":
        detrended = arr / np.maximum(trend, 1e-12)
    else:
        raise ValueError(f"Unknown model '{model}' (use 'additive' or 'multiplicative')")

    # 2. Seasonal averages per period index
    seasonal_cycle = np.zeros(period, dtype=np.float32)
    for p in range(period):
        indices = np.arange(p, n, period)
        valid = detrended[indices]
        valid = valid[~np.isnan(valid)]
        seasonal_cycle[p] = np.mean(valid) if len(valid) > 0 else (0.0 if model == "additive" else 1.0)

    # Center seasonal cycle
    if model == "additive":
        seasonal_cycle -= np.mean(seasonal_cycle)
    else:
        seasonal_cycle /= max(np.mean(seasonal_cycle), 1e-12)

    seasonal = np.tile(seasonal_cycle, int(np.ceil(n / period)))[:n]

    # 3. Residual
    if model == "additive":
        resid = arr - np.nan_to_num(trend, nan=arr) - seasonal
    else:
        resid = arr / (np.maximum(np.nan_to_num(trend, nan=arr), 1e-12) * np.maximum(seasonal, 1e-12))

    return DecomposeResult(trend, seasonal, resid)


def lag_features(series: Any, lags: List[int]) -> _Tensor:
    """Generates matrix of autoregressive lagged values [T - max_lag, len(lags)]."""
    if len(lags) == 0:
        raise ValueError("lags cannot be empty")
    if any(l < 0 for l in lags):
        raise ValueError(f"lags must be non-negative, got {lags}")
    arr = series.to_numpy() if isinstance(series, _Tensor) else np.asarray(series, dtype=np.float32).reshape(-1)
    max_lag = max(lags)
    n = len(arr)
    if n <= max_lag:
        raise ValueError(f"Series length {n} must be greater than max lag {max_lag}")

    cols = []
    for l in lags:
        cols.append(arr[max_lag - l : n - l])

    mat = np.column_stack(cols)
    return _from_numpy(np.ascontiguousarray(mat, dtype=np.float32))


def rolling_window(
    series: Any, window_size: int = 10, horizon: int = 1, step: int = 1
) -> Tuple[_Tensor, _Tensor]:
    """Generates sliding window sequences (X, y) for sequential forecasting."""
    if window_size < 1 or horizon < 1 or step < 1:
        raise ValueError(f"window_size, horizon, and step must be >= 1 (got window_size={window_size}, horizon={horizon}, step={step})")
    arr = series.to_numpy() if isinstance(series, _Tensor) else np.asarray(series, dtype=np.float32).reshape(-1)
    n = len(arr)
    total_len = window_size + horizon
    if n < total_len:
        raise ValueError(f"Series length {n} is smaller than window_size + horizon {total_len}")

    # Optimized 0-copy sliding window strides
    windows = np.lib.stride_tricks.sliding_window_view(arr, window_shape=total_len)[::step]
    x_arr = np.ascontiguousarray(windows[:, :window_size])
    y_arr = np.ascontiguousarray(windows[:, window_size:total_len])

    return (
        _from_numpy(x_arr),
        _from_numpy(y_arr),
    )


def exponential_smoothing(series: Any, alpha: float = 0.3) -> _Tensor:
    """Applies Holt exponential smoothing filter S_t = alpha * Y_t + (1 - alpha) * S_{t-1}."""
    if not (0.0 <= float(alpha) <= 1.0):
        raise ValueError(f"alpha must be in [0, 1], got {alpha}")

    if getattr(series, "requires_grad", False):
        if len(series) == 0:
            return series
        outputs = [series[0].unsqueeze(0)]
        curr = series[0]
        from .oxideon import cat
        for t in range(1, len(series)):
            curr = curr * (1.0 - alpha) + series[t] * alpha
            outputs.append(curr.unsqueeze(0))
        return cat(outputs, dim=0)

    arr = series.to_numpy() if isinstance(series, _Tensor) else np.asarray(series, dtype=np.float32).reshape(-1)
    out = np.zeros_like(arr)
    if len(arr) == 0:
        return _from_numpy(out)

    out[0] = arr[0]
    for t in range(1, len(arr)):
        out[t] = alpha * arr[t] + (1.0 - alpha) * out[t - 1]

    return _from_numpy(np.ascontiguousarray(out, dtype=np.float32))


def autocorrelation(series: Any, max_lag: int = 20) -> _Tensor:
    """Computes sample autocorrelation function (ACF) up to max_lag."""
    arr = series.to_numpy() if isinstance(series, _Tensor) else np.asarray(series, dtype=np.float32).reshape(-1)
    n = len(arr)
    mean = float(np.mean(arr))
    var = float(np.var(arr)) + 1e-12
    acf = np.zeros(max_lag + 1, dtype=np.float32)
    for lag in range(max_lag + 1):
        if lag < n:
            acf[lag] = float(np.mean((arr[: n - lag] - mean) * (arr[lag:] - mean))) / var
    return _from_numpy(acf)


class PatchTST(Module):
    """Patch Time Series Transformer for multi-step temporal forecasting."""

    def __init__(
        self,
        seq_len: int = 32,
        pred_len: int = 8,
        patch_len: int = 8,
        stride: int = 4,
        d_model: int = 32,
    ):
        super().__init__()
        if seq_len < 1 or pred_len < 1 or patch_len < 1 or stride < 1 or d_model < 1:
            raise ValueError(
                f"seq_len, pred_len, patch_len, stride, and d_model must be >= 1 "
                f"(got seq_len={seq_len}, pred_len={pred_len}, patch_len={patch_len}, stride={stride}, d_model={d_model})"
            )
        if patch_len > seq_len:
            raise ValueError(f"patch_len ({patch_len}) cannot exceed seq_len ({seq_len})")
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.patch_len = patch_len
        self.stride = stride
        self.num_patches = (seq_len - patch_len) // stride + 1

        self.patch_proj = Linear(patch_len, d_model)
        self.hidden = Sequential(
            Linear(d_model * self.num_patches, d_model * 2),
            ReLU(),
            Linear(d_model * 2, pred_len),
        )

    def forward(self, x: Any) -> _Tensor:
        """Forward pass taking [B, seq_len] or [seq_len] and predicting [B, pred_len] or [pred_len]."""
        t = x if isinstance(x, _Tensor) else _from_numpy(np.asarray(x, dtype=np.float32))
        single = t.ndim == 1
        if single:
            t = t.unsqueeze(0)
        if t.shape[1] != self.seq_len:
            raise ValueError(f"Input sequence length ({t.shape[1]}) must match PatchTST seq_len ({self.seq_len})")
        b = t.shape[0]

        patches = [t[:, i * self.stride : i * self.stride + self.patch_len] for i in range(self.num_patches)]
        from .oxideon import cat
        flat_patches = cat(patches, dim=1)
        proj = self.patch_proj(flat_patches.reshape(b * self.num_patches, self.patch_len))
        pred = self.hidden(proj.reshape(b, -1))
        if single:
            return pred.squeeze(0)
        return pred


__all__ = [
    "DecomposeResult",
    "seasonal_decompose",
    "lag_features",
    "rolling_window",
    "exponential_smoothing",
    "autocorrelation",
    "PatchTST",
]
