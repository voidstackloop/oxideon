"""Digital signal processing, analytical windowing, and spectral transforms for Oxideon.

Implements standard analytical window functions (Hann, Hamming, Blackman, Bartlett, Kaiser,
Gaussian, Exponential, Chebyshev), Short-Time Fourier Transform (STFT/ISTFT), Hilbert transform,
cross-correlation, convolution, and polyphase resampling.
"""

import math
from typing import Any, Optional, Sequence, Tuple, Union, List, Dict
import numpy as np

from .oxideon import Tensor as _Tensor, from_numpy as _from_numpy
from .primitives import tensor, float32, zeros
from .nn import _check_non_negative


class windows:
    """Standard analytical window functions for spectral analysis and filter design."""

    @staticmethod
    def hann(M: int, sym: bool = True, *, dtype: Any = float32, device: str = "cpu") -> _Tensor:
        M = _check_non_negative("window_length", M)
        if M == 0:
            return tensor([], dtype=dtype, device=device)
        if M == 1:
            return tensor([1.0], dtype=dtype, device=device)
        n = M if not sym else M - 1
        i = np.arange(M, dtype=np.float64)
        w = 0.5 - 0.5 * np.cos(2.0 * np.pi * i / n)
        return tensor(w, dtype=dtype, device=device)

    @staticmethod
    def hamming(M: int, sym: bool = True, alpha: float = 0.54, beta: float = 0.46, *, dtype: Any = float32, device: str = "cpu") -> _Tensor:
        M = _check_non_negative("window_length", M)
        if M == 0:
            return tensor([], dtype=dtype, device=device)
        if M == 1:
            return tensor([1.0], dtype=dtype, device=device)
        n = M if not sym else M - 1
        i = np.arange(M, dtype=np.float64)
        w = alpha - beta * np.cos(2.0 * np.pi * i / n)
        return tensor(w, dtype=dtype, device=device)

    @staticmethod
    def blackman(M: int, sym: bool = True, *, dtype: Any = float32, device: str = "cpu") -> _Tensor:
        M = _check_non_negative("window_length", M)
        if M == 0:
            return tensor([], dtype=dtype, device=device)
        if M == 1:
            return tensor([1.0], dtype=dtype, device=device)
        n = M if not sym else M - 1
        i = np.arange(M, dtype=np.float64)
        w = 0.42 - 0.5 * np.cos(2.0 * np.pi * i / n) + 0.08 * np.cos(4.0 * np.pi * i / n)
        return tensor(w, dtype=dtype, device=device)

    @staticmethod
    def bartlett(M: int, sym: bool = True, *, dtype: Any = float32, device: str = "cpu") -> _Tensor:
        M = _check_non_negative("window_length", M)
        if M == 0:
            return tensor([], dtype=dtype, device=device)
        if M == 1:
            return tensor([1.0], dtype=dtype, device=device)
        n = M if not sym else M - 1
        i = np.arange(M, dtype=np.float64)
        w = np.where(i <= n / 2, 2.0 * i / n, 2.0 - 2.0 * i / n)
        return tensor(w, dtype=dtype, device=device)

    @staticmethod
    def kaiser(M: int, beta: float = 12.0, sym: bool = True, *, dtype: Any = float32, device: str = "cpu") -> _Tensor:
        M = _check_non_negative("window_length", M)
        if M == 0:
            return tensor([], dtype=dtype, device=device)
        if M == 1:
            return tensor([1.0], dtype=dtype, device=device)
        n = M if not sym else M - 1
        denom = float(np.i0(beta))
        i = np.arange(M, dtype=np.float64)
        vals = beta * np.sqrt(np.maximum(0.0, 1.0 - ((2.0 * i / n) - 1.0) ** 2))
        w = np.i0(vals) / denom
        return tensor(w, dtype=dtype, device=device)

    @staticmethod
    def gaussian(M: int, std: float = 1.0, sym: bool = True, *, dtype: Any = float32, device: str = "cpu") -> _Tensor:
        M = _check_non_negative("window_length", M)
        if std <= 0.0:
            raise ValueError(f"std must be positive (> 0), got {std}")
        if M == 0:
            return tensor([], dtype=dtype, device=device)
        if M == 1:
            return tensor([1.0], dtype=dtype, device=device)
        n = M if not sym else M - 1
        i = np.arange(M, dtype=np.float64)
        center = n / 2.0
        w = np.exp(-0.5 * ((i - center) / std) ** 2)
        return tensor(w, dtype=dtype, device=device)

    @staticmethod
    def exponential(M: int, center: Optional[float] = None, tau: float = 1.0, sym: bool = True, *, dtype: Any = float32, device: str = "cpu") -> _Tensor:
        M = _check_non_negative("window_length", M)
        if M == 0:
            return tensor([], dtype=dtype, device=device)
        if M == 1:
            return tensor([1.0], dtype=dtype, device=device)
        n = M if not sym else M - 1
        i = np.arange(M, dtype=np.float64)
        c = center if center is not None else n / 2.0
        w = np.exp(-np.abs(i - c) / max(tau, 1e-12))
        return tensor(w, dtype=dtype, device=device)

    @staticmethod
    def chebyshev(M: int, at: float = 100.0, sym: bool = True, *, dtype: Any = float32, device: str = "cpu") -> _Tensor:
        M = _check_non_negative("window_length", M)
        if M == 0:
            return tensor([], dtype=dtype, device=device)
        if M == 1:
            return tensor([1.0], dtype=dtype, device=device)
        n = M if not sym else M - 1
        # beta = cosh(1 / n * arccosh(10^(at / 20)))
        atten = 10.0 ** (abs(at) / 20.0)
        gamma = math.acosh(atten) / n
        beta = math.cosh(gamma)

        def cheby_poly(k: int, x: float) -> float:
            if abs(x) <= 1.0:
                return math.cos(k * math.acos(x))
            elif x > 1.0:
                return math.cosh(k * math.acosh(x))
            else:
                return ((-1) ** k) * math.cosh(k * math.acosh(-x))

        # Evaluate DFT coefficients
        k_vals = np.arange(M)
        W = np.array([cheby_poly(n, beta * math.cos(math.pi * k / M)) for k in k_vals], dtype=np.complex128)
        # IDFT to get time domain window
        w = np.fft.ifft(W).real
        w = np.roll(w, (M - 1) // 2)
        w = w / np.max(np.abs(w))
        return _from_numpy(np.ascontiguousarray(w, dtype=np.float32))


# Top-level window helpers matching standard PyTorch conventions
def hann_window(window_length: int, periodic: bool = True, *, dtype: Any = float32, device: str = "cpu") -> _Tensor:
    return windows.hann(window_length, sym=not periodic, dtype=dtype, device=device)


def hamming_window(window_length: int, periodic: bool = True, alpha: float = 0.54, beta: float = 0.46, *, dtype: Any = float32, device: str = "cpu") -> _Tensor:
    return windows.hamming(window_length, sym=not periodic, alpha=alpha, beta=beta, dtype=dtype, device=device)


def blackman_window(window_length: int, periodic: bool = True, *, dtype: Any = float32, device: str = "cpu") -> _Tensor:
    return windows.blackman(window_length, sym=not periodic, dtype=dtype, device=device)


def bartlett_window(window_length: int, periodic: bool = True, *, dtype: Any = float32, device: str = "cpu") -> _Tensor:
    return windows.bartlett(window_length, sym=not periodic, dtype=dtype, device=device)


def kaiser_window(window_length: int, periodic: bool = True, beta: float = 12.0, *, dtype: Any = float32, device: str = "cpu") -> _Tensor:
    return windows.kaiser(window_length, beta=beta, sym=not periodic, dtype=dtype, device=device)


def gaussian_window(window_length: int, std: float = 1.0, periodic: bool = True, *, dtype: Any = float32, device: str = "cpu") -> _Tensor:
    return windows.gaussian(window_length, std=std, sym=not periodic, dtype=dtype, device=device)


def exponential_window(window_length: int, center: Optional[float] = None, tau: float = 1.0, periodic: bool = True, *, dtype: Any = float32, device: str = "cpu") -> _Tensor:
    return windows.exponential(window_length, center=center, tau=tau, sym=not periodic, dtype=dtype, device=device)


def chebyshev_window(window_length: int, at: float = 100.0, periodic: bool = True, *, dtype: Any = float32, device: str = "cpu") -> _Tensor:
    return windows.chebyshev(window_length, at=at, sym=not periodic, dtype=dtype, device=device)


chebwin = chebyshev_window


def stft(
    input: _Tensor,
    n_fft: int,
    hop_length: Optional[int] = None,
    win_length: Optional[int] = None,
    window: Optional[_Tensor] = None,
    center: bool = True,
    pad_mode: str = "reflect",
    normalized: bool = False,
    onesided: Optional[bool] = None,
    return_complex: Optional[bool] = None,
) -> _Tensor:
    """Short-time Fourier transform (STFT)."""
    if hop_length is None:
        hop_length = n_fft // 4
    if hop_length < 1:
        raise ValueError(f"hop_length must be >= 1, got {hop_length}")
    if win_length is None:
        win_length = n_fft

    arr = input.to_numpy()
    orig_1d = (arr.ndim == 1)
    if orig_1d:
        arr = arr[np.newaxis, :]

    batch_size, length = arr.shape
    if window is not None:
        win = window.to_numpy().flatten()
    else:
        win = np.hanning(win_length).astype(np.float32)

    pad_amount = n_fft // 2 if center else 0
    if pad_amount > 0:
        arr = np.pad(arr, ((0, 0), (pad_amount, pad_amount)), mode="reflect" if pad_mode == "reflect" else "constant")

    length = arr.shape[1]
    frames = []
    start = 0
    while start + win_length <= length:
        frame = arr[:, start : start + win_length] * win
        spec = np.fft.rfft(frame, n=n_fft, axis=-1)
        frames.append(spec)
        start += hop_length

    if not frames:
        frames.append(np.fft.rfft(arr[:, :min(length, n_fft)], n=n_fft, axis=-1))

    # [batch, freq, time]
    stacked = np.stack(frames, axis=-1)
    if normalized:
        stacked = stacked / math.sqrt(n_fft)

    # Return [batch, freq, time, 2] for real/imag
    real_imag = np.stack([np.real(stacked), np.imag(stacked)], axis=-1)
    if orig_1d:
        real_imag = real_imag[0]
    return tensor(real_imag, dtype=float32, device=input.device)


def istft(
    input: _Tensor,
    n_fft: int,
    hop_length: Optional[int] = None,
    win_length: Optional[int] = None,
    window: Optional[_Tensor] = None,
    center: bool = True,
    normalized: bool = False,
    onesided: Optional[bool] = None,
    length: Optional[int] = None,
    return_complex: bool = False,
) -> _Tensor:
    """Inverse Short-time Fourier transform (ISTFT) via overlap-add synthesis."""
    if hop_length is None:
        hop_length = n_fft // 4
    if hop_length < 1:
        raise ValueError(f"hop_length must be >= 1, got {hop_length}")
    if win_length is None:
        win_length = n_fft

    arr = input.to_numpy()
    orig_ndim = arr.ndim
    # Check if last dim is 2 (real, imag)
    if arr.shape[-1] == 2:
        complex_spec = arr[..., 0] + 1j * arr[..., 1]
    else:
        complex_spec = arr.astype(np.complex128)

    if complex_spec.ndim == 2:
        # [freq, time] -> add batch dim [1, freq, time]
        complex_spec = complex_spec[np.newaxis, ...]
        is_1d = True
    else:
        is_1d = False

    batch_size, n_freqs, n_frames = complex_spec.shape

    if normalized:
        complex_spec = complex_spec * math.sqrt(n_fft)

    if window is not None:
        win = window.to_numpy().flatten().astype(np.float64)
    else:
        win = np.hanning(win_length).astype(np.float64)

    expected_len = (n_frames - 1) * hop_length + win_length
    out_sig = np.zeros((batch_size, expected_len), dtype=np.float64)
    win_sum = np.zeros(expected_len, dtype=np.float64)
    win_sq = win ** 2

    for t in range(n_frames):
        spec_frame = complex_spec[:, :, t]
        time_frame = np.fft.irfft(spec_frame, n=n_fft, axis=-1)[:, :win_length]
        start = t * hop_length
        end = start + win_length
        out_sig[:, start:end] += time_frame * win
        win_sum[start:end] += win_sq

    # Nonzero overlap-add normalization
    win_sum = np.maximum(win_sum, 1e-11)
    out_sig = out_sig / win_sum

    if center:
        pad = n_fft // 2
        out_sig = out_sig[:, pad:-pad] if out_sig.shape[1] > 2 * pad else out_sig

    if length is not None:
        if out_sig.shape[1] > length:
            out_sig = out_sig[:, :length]
        elif out_sig.shape[1] < length:
            out_sig = np.pad(out_sig, ((0, 0), (0, length - out_sig.shape[1])))

    if is_1d:
        out_sig = out_sig[0]

    return _from_numpy(np.ascontiguousarray(out_sig, dtype=np.float32))


def hilbert(input: _Tensor, n: Optional[int] = None, axis: int = -1) -> _Tensor:
    """Computes the analytic signal x_a(t) = x(t) + j * H(x)(t) via the Hilbert transform."""
    arr = input.to_numpy()
    N = n if n is not None else arr.shape[axis]
    X = np.fft.fft(arr, n=N, axis=axis)

    h = np.zeros(N, dtype=np.float32)
    if N % 2 == 0:
        h[0] = 1.0
        h[N // 2] = 1.0
        h[1 : N // 2] = 2.0
    else:
        h[0] = 1.0
        h[1 : (N + 1) // 2] = 2.0

    # Reshape h for broadcasting along axis
    shape = [1] * arr.ndim
    shape[axis] = N
    h = h.reshape(shape)

    analytic = np.fft.ifft(X * h, axis=axis)
    real_imag = np.stack([np.real(analytic), np.imag(analytic)], axis=-1)
    return _from_numpy(np.ascontiguousarray(real_imag, dtype=np.float32))


def correlate(input: _Tensor, other: _Tensor, mode: str = "valid") -> _Tensor:
    """1D cross-correlation along the last dimension."""
    if mode not in ("valid", "same", "full"):
        raise ValueError(f"mode must be 'valid', 'same', or 'full', got '{mode}'")

    if (getattr(input, "requires_grad", False) or getattr(other, "requires_grad", False)) and input.ndim == 1 and other.ndim == 1:
        L = input.shape[0]
        K = other.shape[0]
        if mode == "valid":
            out_len = max(L - K + 1, 0)
            outputs = [(input[i : i + K] * other).sum().unsqueeze(0) for i in range(out_len)]
            return from_numpy(np.empty(0, dtype=np.float32)) if not outputs else from_numpy(np.zeros([0])) if out_len == 0 else _differentiable_cat(outputs)
        elif mode == "same":
            pad_left = (K - 1) // 2
            pad_right = K - 1 - pad_left
            padded = _pad_1d(input, pad_left, pad_right)
            outputs = [(padded[i : i + K] * other).sum().unsqueeze(0) for i in range(L)]
            return _differentiable_cat(outputs)
        else: # full
            padded = _pad_1d(input, K - 1, K - 1)
            outputs = [(padded[i : i + K] * other).sum().unsqueeze(0) for i in range(L + K - 1)]
            return _differentiable_cat(outputs)

    a = input.to_numpy()
    b = other.to_numpy()
    if a.ndim == 1 and b.ndim == 1:
        res = np.correlate(a, b, mode=mode)
    else:
        # Batched correlation along last axis
        a_flat = a.reshape(-1, a.shape[-1])
        b_flat = b.reshape(-1, b.shape[-1]) if b.ndim > 1 else np.tile(b, (len(a_flat), 1))
        if len(a_flat) != len(b_flat):
            raise ValueError(f"Batch dimension mismatch: {a.shape} vs {other.to_numpy().shape}")
        res_list = [np.correlate(a_flat[i], b_flat[i], mode=mode) for i in range(len(a_flat))]
        res = np.array(res_list).reshape(*a.shape[:-1], -1)
    return _from_numpy(np.ascontiguousarray(res, dtype=np.float32))


def _pad_1d(t: _Tensor, pad_left: int, pad_right: int) -> _Tensor:
    parts = []
    if pad_left > 0:
        parts.append(zeros([pad_left], dtype=t.dtype, device=t.device))
    parts.append(t)
    if pad_right > 0:
        parts.append(zeros([pad_right], dtype=t.dtype, device=t.device))
    from .oxideon import cat
    return cat(parts, dim=0)


def _differentiable_cat(tensors: List[_Tensor]) -> _Tensor:
    from .oxideon import cat
    return cat(tensors, dim=0)


def convolve(input: _Tensor, other: _Tensor, mode: str = "full") -> _Tensor:
    """1D convolution along the last dimension."""
    if mode not in ("valid", "same", "full"):
        raise ValueError(f"mode must be 'valid', 'same', or 'full', got '{mode}'")

    if (getattr(input, "requires_grad", False) or getattr(other, "requires_grad", False)) and input.ndim == 1 and other.ndim == 1:
        L = input.shape[0]
        K = other.shape[0]
        rev_b = _from_numpy(other.to_numpy()[::-1].copy().astype(np.float32))
        if other.requires_grad:
            rev_parts = [other[K - 1 - i].unsqueeze(0) for i in range(K)]
            rev_b = _differentiable_cat(rev_parts)

        if mode == "valid":
            out_len = max(L - K + 1, 0)
            outputs = [(input[i : i + K] * rev_b).sum().unsqueeze(0) for i in range(out_len)]
            return from_numpy(np.empty(0, dtype=np.float32)) if not outputs else _differentiable_cat(outputs)
        elif mode == "same":
            pad_left = (K - 1) // 2
            pad_right = K - 1 - pad_left
            padded = _pad_1d(input, pad_left, pad_right)
            outputs = [(padded[i : i + K] * rev_b).sum().unsqueeze(0) for i in range(L)]
            return _differentiable_cat(outputs)
        else: # full
            padded = _pad_1d(input, K - 1, K - 1)
            outputs = [(padded[i : i + K] * rev_b).sum().unsqueeze(0) for i in range(L + K - 1)]
            return _differentiable_cat(outputs)

    a = input.to_numpy()
    b = other.to_numpy()
    if a.ndim == 1 and b.ndim == 1:
        res = np.convolve(a, b, mode=mode)
    else:
        a_flat = a.reshape(-1, a.shape[-1])
        b_flat = b.reshape(-1, b.shape[-1]) if b.ndim > 1 else np.tile(b, (len(a_flat), 1))
        if len(a_flat) != len(b_flat):
            raise ValueError(f"Batch dimension mismatch: {a.shape} vs {other.to_numpy().shape}")
        res_list = [np.convolve(a_flat[i], b_flat[i], mode=mode) for i in range(len(a_flat))]
        res = np.array(res_list).reshape(*a.shape[:-1], -1)
    return _from_numpy(np.ascontiguousarray(res, dtype=np.float32))


def find_peaks(
    signal: Any,
    height: Optional[Union[float, Tuple[float, float]]] = None,
    distance: Optional[int] = None,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Finds peaks (local maxima) in a 1D signal with optional height and distance constraints."""
    arr = signal.to_numpy() if isinstance(signal, _Tensor) else np.asarray(signal, dtype=np.float32)
    arr = arr.flatten()
    if len(arr) < 3:
        return np.empty(0, dtype=np.int64), {"peak_heights": np.empty(0, dtype=np.float32)}

    # Local maxima: arr[i] > arr[i-1] and arr[i] >= arr[i+1]
    peaks = np.where((arr[1:-1] > arr[:-2]) & (arr[1:-1] >= arr[2:]))[0] + 1

    if height is not None:
        if isinstance(height, (tuple, list)):
            h_min, h_max = height
            mask = (arr[peaks] >= h_min) & (arr[peaks] <= h_max)
        else:
            mask = arr[peaks] >= float(height)
        peaks = peaks[mask]

    if distance is not None and len(peaks) > 1:
        dist = int(distance)
        keep = [peaks[0]]
        for p in peaks[1:]:
            if p - keep[-1] >= dist:
                keep.append(p)
        peaks = np.array(keep, dtype=np.int64)

    props = {"peak_heights": arr[peaks]}
    return peaks, props


def resample_poly(input: _Tensor, up: int, down: int, axis: int = -1) -> _Tensor:
    """Resamples a signal along an axis by a rational factor up / down using polyphase filtering."""
    arr = input.to_numpy()
    up = int(up)
    down = int(down)
    if up <= 0 or down <= 0:
        raise ValueError("up and down resampling factors must be positive integers.")
    if up == down:
        return input

    # Rational resampling via Fourier domain interpolation / decimation
    orig_len = arr.shape[axis]
    if orig_len == 0:
        new_shape = list(arr.shape)
        new_shape[axis] = 0
        return _from_numpy(np.empty(new_shape, dtype=np.float32))
    target_len = int(round(orig_len * up / down))

    # FFT along target axis
    X = np.fft.rfft(arr, axis=axis)
    # Resample in frequency domain
    new_shape = list(arr.shape)
    new_shape[axis] = target_len

    # Scaled irfft with target length
    resampled = np.fft.irfft(X, n=target_len, axis=axis) * (float(target_len) / float(orig_len))
    return _from_numpy(np.ascontiguousarray(resampled, dtype=np.float32))


import sys as _sys
_sys.modules["oxideon.signal"] = _sys.modules[__name__]
_sys.modules["oxideon.signal.windows"] = windows

__all__ = [
    "windows",
    "hann_window",
    "hamming_window",
    "blackman_window",
    "bartlett_window",
    "kaiser_window",
    "gaussian_window",
    "exponential_window",
    "chebyshev_window",
    "chebwin",
    "stft",
    "istft",
    "hilbert",
    "correlate",
    "convolve",
    "resample_poly",
    "find_peaks",
]
