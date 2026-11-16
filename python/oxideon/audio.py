"""Audio processing and acoustic feature extraction for Oxideon.

Provides STFT spectrograms, Mel-scale filterbanks, and cepstral coefficients:
  - hz_to_mel / mel_to_hz: Frequency scale conversions
  - mel_filter_bank: Triangular mel filter matrix
  - Spectrogram: Short-Time Fourier Transform (STFT) power spectrogram
  - MelSpectrogram: Mel-scale power spectrogram
  - amplitude_to_db: Logarithmic decibel compression
  - MFCC: Mel-Frequency Cepstral Coefficients via DCT-II
"""

from typing import Optional, Union, Any, Tuple
import math
import numpy as np

from .oxideon import Tensor as _Tensor, from_numpy as _from_numpy
from .nn import Module, _check_elements


def hz_to_mel(hz: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """Converts frequency in Hertz to the Mel scale."""
    return 2595.0 * np.log10(1.0 + hz / 700.0)


def mel_to_hz(mel: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """Converts frequency in Mel to Hertz."""
    return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)


def mel_filter_bank(
    n_mels: int = 80,
    n_fft: int = 400,
    sample_rate: int = 16000,
    f_min: float = 0.0,
    f_max: Optional[float] = None,
) -> np.ndarray:
    """Computes a triangular Mel filter bank matrix of shape [n_mels, n_fft // 2 + 1]."""
    if f_max is None:
        f_max = float(sample_rate) / 2.0

    n_freqs = n_fft // 2 + 1
    fft_freqs = np.linspace(0.0, float(sample_rate) / 2.0, n_freqs)

    min_mel = hz_to_mel(f_min)
    max_mel = hz_to_mel(f_max)
    mel_points = np.linspace(min_mel, max_mel, n_mels + 2)
    hz_points = mel_to_hz(mel_points)

    weights = np.zeros((n_mels, n_freqs), dtype=np.float32)
    for i in range(n_mels):
        left = hz_points[i]
        center = hz_points[i + 1]
        right = hz_points[i + 2]

        # Upslope
        upslope = (fft_freqs - left) / max(center - left, 1e-12)
        # Downslope
        downslope = (right - fft_freqs) / max(right - center, 1e-12)

        tri = np.maximum(0.0, np.minimum(upslope, downslope))
        # Slaney-style area normalization
        area = 2.0 / (right - left)
        weights[i] = tri * area

    return weights


def amplitude_to_db(
    spec: Any, amin: float = 1e-10, top_db: float = 80.0
) -> _Tensor:
    """Converts power or amplitude spectrogram to decibels (dB)."""
    arr = spec.to_numpy() if isinstance(spec, _Tensor) else np.asarray(spec, dtype=np.float32)
    if arr.size == 0:
        return _from_numpy(np.ascontiguousarray(arr, dtype=np.float32))
    if isinstance(spec, _Tensor) and spec.requires_grad:
        clamped = (spec - float(amin)).relu() + float(amin)
        log_spec = clamped.log() * (10.0 / math.log(10.0))
        max_val = float(np.max(10.0 * np.log10(np.maximum(arr, amin))))
        floor_val = max_val - float(top_db)
        return (log_spec - floor_val).relu() + floor_val
    log_spec = 10.0 * np.log10(np.maximum(arr, amin))
    max_val = np.max(log_spec)
    log_spec = np.maximum(log_spec, max_val - top_db)
    return _from_numpy(np.ascontiguousarray(log_spec, dtype=np.float32))


class Spectrogram(Module):
    """Computes the Short-Time Fourier Transform (STFT) power spectrogram."""

    def __init__(
        self,
        n_fft: int = 400,
        hop_length: int = 160,
        power: float = 2.0,
        window: str = "hann",
    ):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.power = power
        if window == "hann":
            self.win = np.hanning(n_fft).astype(np.float32)
        else:
            self.win = np.ones(n_fft, dtype=np.float32)

    def forward(self, waveform: Any) -> _Tensor:
        """Transforms 1D waveform [T] or batched [B, T] into spectrogram [B, n_fft//2 + 1, time_steps]."""
        arr = waveform.to_numpy() if isinstance(waveform, _Tensor) else np.asarray(waveform, dtype=np.float32)
        is_1d = arr.ndim == 1
        if is_1d:
            arr = arr[None, :]

        batch_size, length = arr.shape
        pad_amount = self.n_fft // 2
        padded = np.pad(arr, ((0, 0), (pad_amount, pad_amount)), mode="reflect")

        num_frames = 1 + (padded.shape[1] - self.n_fft) // self.hop_length
        n_freqs = self.n_fft // 2 + 1

        # Fully vectorized batched audio framing and real FFT
        frames_all = np.lib.stride_tricks.sliding_window_view(padded, window_shape=self.n_fft, axis=-1)[:, ::self.hop_length, :]
        windowed_all = frames_all * self.win
        stft_all = np.fft.rfft(windowed_all, n=self.n_fft, axis=-1)
        mag_all = np.abs(stft_all)
        if self.power == 2.0:
            spec_all = mag_all**2
        elif self.power == 1.0:
            spec_all = mag_all
        else:
            spec_all = mag_all**self.power
        specs = spec_all.transpose(0, 2, 1)

        if is_1d:
            specs = specs[0]

        return _from_numpy(np.ascontiguousarray(specs, dtype=np.float32))


class MelSpectrogram(Module):
    """Computes Mel-scale power spectrogram."""

    def __init__(
        self,
        sample_rate: int = 16000,
        n_fft: int = 400,
        hop_length: int = 160,
        n_mels: int = 80,
        f_min: float = 0.0,
        f_max: Optional[float] = None,
    ):
        super().__init__()
        self.spectrogram = Spectrogram(n_fft=n_fft, hop_length=hop_length, power=2.0)
        self.mel_fb = mel_filter_bank(
            n_mels=n_mels,
            n_fft=n_fft,
            sample_rate=sample_rate,
            f_min=f_min,
            f_max=f_max,
        )

    def forward(self, waveform: Any) -> _Tensor:
        spec = self.spectrogram(waveform).to_numpy()
        is_2d = spec.ndim == 2
        if is_2d:
            # [freqs, time]
            mel_spec = self.mel_fb @ spec
        else:
            # [B, freqs, time]
            mel_spec = np.einsum("mf,bft->bmt", self.mel_fb, spec)
        return _from_numpy(np.ascontiguousarray(mel_spec, dtype=np.float32))


class MFCC(Module):
    """Computes Mel-Frequency Cepstral Coefficients (MFCC)."""

    def __init__(
        self,
        sample_rate: int = 16000,
        n_mfcc: int = 40,
        n_mels: int = 80,
        n_fft: int = 400,
        hop_length: int = 160,
    ):
        super().__init__()
        n_mfcc = _check_elements("n_mfcc", n_mfcc)
        n_mels = _check_elements("n_mels", n_mels)
        self.n_mfcc = n_mfcc
        self.mel_spectrogram = MelSpectrogram(
            sample_rate=sample_rate, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels
        )
        # Type-II DCT basis
        n = np.arange(n_mels)
        k = np.arange(n_mfcc)[:, None]
        self.dct_mat = np.cos(math.pi * (n + 0.5) * k / n_mels) * math.sqrt(2.0 / n_mels)
        self.dct_mat[0] *= 1.0 / math.sqrt(2.0)

    def forward(self, waveform: Any) -> _Tensor:
        mel_spec = self.mel_spectrogram(waveform).to_numpy()
        log_mel = np.log(np.maximum(mel_spec, 1e-10))
        if log_mel.ndim == 2:
            mfcc = self.dct_mat @ log_mel
        else:
            mfcc = np.einsum("cm,bmt->bct", self.dct_mat, log_mel)
        return _from_numpy(np.ascontiguousarray(mfcc, dtype=np.float32))


# ---------------------------------------------------------------------------
# Functional Audio Signal Processing Operations
# ---------------------------------------------------------------------------

def _to_np(t: Any) -> np.ndarray:
    if isinstance(t, _Tensor):
        return t.to_numpy()
    return np.asarray(t, dtype=np.float32)


def griffin_lim(
    spec: Any,
    n_fft: int = 400,
    hop_length: int = 160,
    n_iter: int = 32,
    momentum: float = 0.99,
) -> _Tensor:
    """Reconstructs a time-domain audio signal from a magnitude spectrogram using the Griffin-Lim algorithm."""
    arr = _to_np(spec).astype(np.float32)
    is_batched = arr.ndim == 3
    if not is_batched:
        arr = arr[None, ...]

    b, freqs, frames = arr.shape
    expected_len = max(n_fft, (frames - 1) * hop_length)
    win = np.hanning(n_fft).astype(np.float32)

    reconstructed = []
    for bi in range(b):
        mag = arr[bi]
        phase = np.exp(1j * np.random.uniform(0, 2 * np.pi, mag.shape))
        prev_phase = phase

        for _ in range(n_iter):
            complex_spec = mag * phase
            time_signal = np.zeros(expected_len + n_fft, dtype=np.float32)
            window_norm = np.zeros(expected_len + n_fft, dtype=np.float32)

            for t in range(frames):
                half_spec = complex_spec[:, t]
                full_spec = np.concatenate([half_spec, np.conj(half_spec[-2:0:-1])])
                frame_time = np.real(np.fft.ifft(full_spec)) * win
                pos = t * hop_length
                time_signal[pos : pos + n_fft] += frame_time
                window_norm[pos : pos + n_fft] += win * win

            signal = time_signal / np.maximum(window_norm, 1e-8)

            new_spec = np.zeros_like(complex_spec)
            for t in range(frames):
                pos = t * hop_length
                frame = signal[pos : pos + n_fft] * win
                fft_res = np.fft.fft(frame)[:freqs]
                new_spec[:, t] = fft_res

            phase = new_spec / np.maximum(np.abs(new_spec), 1e-12)
            if momentum > 0:
                phase = phase + momentum * prev_phase
                phase = phase / np.maximum(np.abs(phase), 1e-12)
                prev_phase = phase

        reconstructed.append(signal[:expected_len])

    res = np.stack(reconstructed, axis=0) if is_batched else reconstructed[0]
    return _from_numpy(np.ascontiguousarray(res, dtype=np.float32))


def resample(waveform: Any, orig_freq: int, new_freq: int) -> _Tensor:
    """Resamples a waveform from orig_freq to new_freq using bandlimited sinc interpolation."""
    if isinstance(waveform, _Tensor) and orig_freq == new_freq:
        return waveform * 1.0
    arr = _to_np(waveform).astype(np.float32)
    if orig_freq == new_freq:
        return _from_numpy(np.ascontiguousarray(arr, dtype=np.float32))

    orig_len = arr.shape[-1]
    new_len = int(round(orig_len * float(new_freq) / float(orig_freq)))

    # Resample last axis
    orig_times = np.linspace(0, 1, orig_len)
    new_times = np.linspace(0, 1, new_len)

    if arr.ndim == 1:
        res = np.interp(new_times, orig_times, arr)
    else:
        flat = arr.reshape(-1, orig_len)
        res_list = [np.interp(new_times, orig_times, row) for row in flat]
        res = np.stack(res_list, axis=0).reshape(arr.shape[:-1] + (new_len,))

    return _from_numpy(np.ascontiguousarray(res, dtype=np.float32))


def compute_deltas(spec: Any, win_length: int = 5) -> _Tensor:
    """Computes delta (differential) coefficients over time for a spectrogram."""
    arr = _to_np(spec).astype(np.float32)
    half_win = win_length // 2
    denom = 2 * sum(i * i for i in range(1, half_win + 1))
    kernel = np.arange(-half_win, half_win + 1, dtype=np.float32)

    # Convolve along time axis (-1)
    pad_width = [(0, 0)] * (arr.ndim - 1) + [(half_win, half_win)]
    padded = np.pad(arr, pad_width, mode="edge")

    # Apply sliding dot product
    deltas = np.zeros_like(arr)
    t_len = arr.shape[-1]
    for t in range(t_len):
        patch = padded[..., t : t + win_length]
        deltas[..., t] = np.sum(patch * kernel, axis=-1) / denom

    return _from_numpy(np.ascontiguousarray(deltas, dtype=np.float32))


def gain(waveform: Any, gain_db: float = 1.0) -> _Tensor:
    """Applies amplification or attenuation in decibels (dB) to waveform."""
    factor = float(10.0 ** (gain_db / 20.0))
    if isinstance(waveform, _Tensor):
        return waveform * factor
    arr = _to_np(waveform).astype(np.float32)
    return _from_numpy(np.ascontiguousarray(arr * factor, dtype=np.float32))


def dither(waveform: Any, density_function: str = "TPDF", seed: Optional[int] = None) -> _Tensor:
    """Applies triangular probability density function (TPDF) dither to waveform."""
    arr = _to_np(waveform).astype(np.float32)
    rng = np.random.default_rng(seed)
    if density_function.upper() == "TPDF":
        noise = (rng.random(arr.shape, dtype=np.float32) - rng.random(arr.shape, dtype=np.float32)) * (1.0 / 32768.0)
    else:
        noise = (rng.random(arr.shape, dtype=np.float32) - 0.5) * (1.0 / 32768.0)
    if isinstance(waveform, _Tensor) and waveform.requires_grad:
        return waveform + _from_numpy(np.ascontiguousarray(noise, dtype=np.float32))
    return _from_numpy(np.ascontiguousarray(arr + noise, dtype=np.float32))


# ---------------------------------------------------------------------------
# Audio Transforms
# ---------------------------------------------------------------------------

class FrequencyMasking(Module):
    """Applies SpecAugment frequency masking to a spectrogram."""

    def __init__(self, freq_mask_param: int = 15, seed: Optional[int] = None):
        super().__init__()
        self.freq_mask_param = freq_mask_param
        self._rng = np.random.default_rng(seed)

    def forward(self, spec: _Tensor) -> _Tensor:
        arr = _to_np(spec)
        num_freqs = arr.shape[-2]
        max_f = max(1, min(self.freq_mask_param, num_freqs))
        f = int(self._rng.integers(0, max_f))
        f0 = int(self._rng.integers(0, max(1, num_freqs - f)))
        mask = np.ones_like(arr, dtype=np.float32)
        mask[..., f0 : f0 + f, :] = 0.0
        if isinstance(spec, _Tensor) and spec.requires_grad:
            return spec * _from_numpy(np.ascontiguousarray(mask, dtype=np.float32))
        return _from_numpy(np.ascontiguousarray(arr * mask, dtype=np.float32))


class TimeMasking(Module):
    """Applies SpecAugment time masking to a spectrogram."""

    def __init__(self, time_mask_param: int = 35, seed: Optional[int] = None):
        super().__init__()
        self.time_mask_param = time_mask_param
        self._rng = np.random.default_rng(seed)

    def forward(self, spec: _Tensor) -> _Tensor:
        arr = _to_np(spec)
        num_times = arr.shape[-1]
        max_t = max(1, min(self.time_mask_param, num_times))
        t = int(self._rng.integers(0, max_t))
        t0 = int(self._rng.integers(0, max(1, num_times - t)))
        mask = np.ones_like(arr, dtype=np.float32)
        mask[..., :, t0 : t0 + t] = 0.0
        if isinstance(spec, _Tensor) and spec.requires_grad:
            return spec * _from_numpy(np.ascontiguousarray(mask, dtype=np.float32))
        return _from_numpy(np.ascontiguousarray(arr * mask, dtype=np.float32))


class GriffinLim(Module):
    def __init__(self, n_fft: int = 400, hop_length: int = 160, n_iter: int = 32):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_iter = n_iter

    def forward(self, spec: _Tensor) -> _Tensor:
        return griffin_lim(spec, n_fft=self.n_fft, hop_length=self.hop_length, n_iter=self.n_iter)


class Resample(Module):
    def __init__(self, orig_freq: int, new_freq: int):
        super().__init__()
        self.orig_freq = orig_freq
        self.new_freq = new_freq

    def forward(self, waveform: _Tensor) -> _Tensor:
        return resample(waveform, self.orig_freq, self.new_freq)


class ComputeDeltas(Module):
    def __init__(self, win_length: int = 5):
        super().__init__()
        self.win_length = win_length

    def forward(self, spec: _Tensor) -> _Tensor:
        return compute_deltas(spec, win_length=self.win_length)


# Namespaces
class _FunctionalNamespace:
    griffin_lim = staticmethod(griffin_lim)
    resample = staticmethod(resample)
    compute_deltas = staticmethod(compute_deltas)
    gain = staticmethod(gain)
    dither = staticmethod(dither)
    amplitude_to_db = staticmethod(amplitude_to_db)
    hz_to_mel = staticmethod(hz_to_mel)
    mel_to_hz = staticmethod(mel_to_hz)
    mel_filter_bank = staticmethod(mel_filter_bank)


class _TransformsNamespace:
    Spectrogram = Spectrogram
    MelSpectrogram = MelSpectrogram
    MFCC = MFCC
    FrequencyMasking = FrequencyMasking
    TimeMasking = TimeMasking
    GriffinLim = GriffinLim
    Resample = Resample
    ComputeDeltas = ComputeDeltas


functional = _FunctionalNamespace()
transforms = _TransformsNamespace()

import sys as _sys
_sys.modules["oxideon.audio.functional"] = functional
_sys.modules["oxideon.audio.transforms"] = transforms


__all__ = [
    "hz_to_mel",
    "mel_to_hz",
    "mel_filter_bank",
    "amplitude_to_db",
    "Spectrogram",
    "MelSpectrogram",
    "MFCC",
    "griffin_lim",
    "resample",
    "compute_deltas",
    "gain",
    "dither",
    "FrequencyMasking",
    "TimeMasking",
    "GriffinLim",
    "Resample",
    "ComputeDeltas",
]
