"""Signal Processing, Time-Series Warping, and Fidelity Metrics.

Implements:
47. MelFilterbank: Triangular audio filterbank matrix for linear-to-Mel spectral projection
48. FastDTW: Dynamic Time Warping with Sakoe-Chiba constraint band
49. ExponentialMovingAverage: Shadow parameter tracking for inference stabilization
50. Autocorrelation1D: Normalized 1D autocorrelation with pitch period detection
51. PeakSignalToNoiseRatio: PSNR metric in decibels for signal/image reconstruction
52. StructuralSimilarityIndex: SSIM metric with luminance, contrast, and structure comparison
"""

from typing import Optional, Tuple, List, Dict, Union, Any
import math
import numpy as np

from .oxideon import Tensor as _Tensor, from_numpy as _from_numpy
from .nn import Parameter


class MelFilterbank:
    """Mel-scale triangular filterbank matrix generator."""

    @staticmethod
    def hz_to_mel(freq_hz: Union[float, np.ndarray], htk: bool = False) -> Union[float, np.ndarray]:
        if htk:
            return 2595.0 * np.log10(1.0 + freq_hz / 700.0)
        return 1127.0 * np.log(1.0 + freq_hz / 700.0)

    @staticmethod
    def mel_to_hz(mel: Union[float, np.ndarray], htk: bool = False) -> Union[float, np.ndarray]:
        if htk:
            return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)
        return 700.0 * (np.exp(mel / 1127.0) - 1.0)

    @staticmethod
    def create(
        sr: int = 16000,
        n_fft: int = 512,
        n_mels: int = 40,
        f_min: float = 0.0,
        f_max: Optional[float] = None,
        htk: bool = False,
    ) -> _Tensor:
        """Constructs filterbank matrix of shape [n_mels, n_fft // 2 + 1]."""
        if f_max is None:
            f_max = float(sr // 2)

        min_mel = MelFilterbank.hz_to_mel(f_min, htk=htk)
        max_mel = MelFilterbank.hz_to_mel(f_max, htk=htk)

        mel_points = np.linspace(min_mel, max_mel, n_mels + 2)
        hz_points = MelFilterbank.mel_to_hz(mel_points, htk=htk)

        num_bins = n_fft // 2 + 1
        fft_freqs = np.linspace(0.0, float(sr) / 2.0, num_bins)

        fb = np.zeros((n_mels, num_bins), dtype=np.float32)
        for m in range(1, n_mels + 1):
            f_left = hz_points[m - 1]
            f_center = hz_points[m]
            f_right = hz_points[m + 1]

            up_slope = (fft_freqs - f_left) / max(f_center - f_left, 1e-8)
            down_slope = (f_right - fft_freqs) / max(f_right - f_center, 1e-8)
            row = np.maximum(0.0, np.minimum(up_slope, down_slope)).astype(np.float32)
            # Ensure narrow filters between adjacent FFT bins still activate closest bin
            if np.sum(row) == 0.0 and num_bins > 0:
                closest_bin = int(np.argmin(np.abs(fft_freqs - f_center)))
                row[closest_bin] = 1.0
            fb[m - 1] = row

        return _from_numpy(fb)


class FastDTW:
    """Dynamic Time Warping with Sakoe-Chiba constraint band for temporal alignment."""

    @staticmethod
    def distance(
        s1: Union[_Tensor, np.ndarray],
        s2: Union[_Tensor, np.ndarray],
        radius: Optional[int] = None,
    ) -> float:
        """Computes alignment distance between two 1D or 2D sequences."""
        a = s1.to_numpy() if isinstance(s1, _Tensor) else np.array(s1, dtype=np.float64)
        b = s2.to_numpy() if isinstance(s2, _Tensor) else np.array(s2, dtype=np.float64)

        if a.ndim == 1:
            a = a[:, np.newaxis]
        if b.ndim == 1:
            b = b[:, np.newaxis]

        n, m = len(a), len(b)
        if radius is None:
            eff_radius = max(n, m)
        else:
            eff_radius = max(int(radius), abs(n - m))

        prev_cost = np.full(m + 1, np.inf, dtype=np.float64)
        prev_cost[0] = 0.0
        curr_cost = np.full(m + 1, np.inf, dtype=np.float64)

        for i in range(1, n + 1):
            curr_cost.fill(np.inf)
            j_start = max(1, i - eff_radius)
            j_end = min(m + 1, i + eff_radius + 1)
            ai_1 = a[i - 1]
            for j in range(j_start, j_end):
                d = float(np.linalg.norm(ai_1 - b[j - 1]))
                curr_cost[j] = d + min(
                    prev_cost[j],
                    curr_cost[j - 1],
                    prev_cost[j - 1],
                )
            prev_cost, curr_cost = curr_cost, prev_cost

        return float(prev_cost[m])


class ExponentialMovingAverage:
    """Polyak-Ruppert Exponential Moving Average (EMA) for model parameters."""

    def __init__(self, params: List[Parameter], decay: float = 0.999, warm_up: int = 100):
        self.params = params
        self.decay = decay
        self.warm_up = warm_up
        self.step_count = 0
        self.shadow_params: Dict[int, np.ndarray] = {
            id(p): p.to_numpy().copy() for p in params
        }
        self.backup_params: Dict[int, np.ndarray] = {}

    def update(self) -> None:
        self.step_count += 1
        effective_decay = min(self.decay, (1.0 + self.step_count) / (10.0 + self.step_count)) if self.step_count < self.warm_up else self.decay

        for p in self.params:
            pid = id(p)
            w = p.to_numpy()
            self.shadow_params[pid] = effective_decay * self.shadow_params[pid] + (1.0 - effective_decay) * w

    def apply_shadow(self) -> None:
        for p in self.params:
            pid = id(p)
            self.backup_params[pid] = p.to_numpy().copy()
            t = _from_numpy(self.shadow_params[pid].copy())
            if hasattr(p, "assign"):
                p.assign(t)
            elif hasattr(p, "copy_"):
                p.copy_(t)
            elif hasattr(p, "set_data"):
                p.set_data(t)

    def restore(self) -> None:
        for p in self.params:
            pid = id(p)
            if pid in self.backup_params:
                t = _from_numpy(self.backup_params[pid].copy())
                if hasattr(p, "assign"):
                    p.assign(t)
                elif hasattr(p, "copy_"):
                    p.copy_(t)
                elif hasattr(p, "set_data"):
                    p.set_data(t)
        self.backup_params.clear()


class Autocorrelation1D:
    """Normalized 1D Autocorrelation and Pitch/Periodicity Detection."""

    @staticmethod
    def correlate(signal: Union[_Tensor, np.ndarray], max_lag: Optional[int] = None) -> _Tensor:
        x = signal.to_numpy() if isinstance(signal, _Tensor) else np.array(signal, dtype=np.float64)
        x_centered = x - np.mean(x)
        n = len(x)
        if max_lag is None:
            max_lag = n
        else:
            max_lag = min(max_lag, n)

        var = np.sum(x_centered ** 2)
        if var < 1e-12:
            return _from_numpy(np.zeros(max_lag, dtype=np.float32))

        corr = np.correlate(x_centered, x_centered, mode="full")
        lags = corr[n - 1 : n - 1 + max_lag] / var
        return _from_numpy(lags.astype(np.float32))

    @staticmethod
    def detect_pitch_period(signal: Union[_Tensor, np.ndarray], min_lag: int = 10, max_lag: int = 200) -> int:
        if min_lag >= max_lag:
            raise ValueError(f"min_lag ({min_lag}) must be less than max_lag ({max_lag})")
        corr = Autocorrelation1D.correlate(signal, max_lag=max_lag + 1).to_numpy()
        if len(corr) <= min_lag:
            raise ValueError(f"signal length ({len(corr)}) must be greater than min_lag ({min_lag})")
        search_window = corr[min_lag : max_lag + 1]
        best_offset = int(np.argmax(search_window))
        return min_lag + best_offset


class PeakSignalToNoiseRatio:
    """Peak Signal-to-Noise Ratio (PSNR) calculation in decibels (dB)."""

    @staticmethod
    def compute(
        y_true: Union[_Tensor, np.ndarray],
        y_pred: Union[_Tensor, np.ndarray],
        data_range: float = 1.0,
    ) -> float:
        yt = y_true.to_numpy() if isinstance(y_true, _Tensor) else np.array(y_true, dtype=np.float64)
        yp = y_pred.to_numpy() if isinstance(y_pred, _Tensor) else np.array(y_pred, dtype=np.float64)

        mse = float(np.mean((yt - yp) ** 2))
        if mse < 1e-15:
            return 999.0
        psnr = 10.0 * math.log10((data_range ** 2) / mse)
        return float(psnr)


class StructuralSimilarityIndex:
    """Structural Similarity Index Measure (SSIM) (Wang et al., 2004)."""

    @staticmethod
    def compute(
        x: Union[_Tensor, np.ndarray],
        y: Union[_Tensor, np.ndarray],
        data_range: float = 1.0,
        k1: float = 0.01,
        k2: float = 0.03,
    ) -> float:
        img1 = x.to_numpy() if isinstance(x, _Tensor) else np.array(x, dtype=np.float64)
        img2 = y.to_numpy() if isinstance(y, _Tensor) else np.array(y, dtype=np.float64)

        c1 = (k1 * data_range) ** 2
        c2 = (k2 * data_range) ** 2

        mu1 = float(np.mean(img1))
        mu2 = float(np.mean(img2))

        sigma1_sq = float(np.var(img1))
        sigma2_sq = float(np.var(img2))
        sigma12 = float(np.mean((img1 - mu1) * (img2 - mu2)))

        numerator = (2.0 * mu1 * mu2 + c1) * (2.0 * sigma12 + c2)
        denominator = (mu1 ** 2 + mu2 ** 2 + c1) * (sigma1_sq + sigma2_sq + c2)

        ssim = numerator / (denominator + 1e-12)
        return float(np.clip(ssim, -1.0, 1.0))
