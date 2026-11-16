"""
Digital Signal Processing (DSP) and Audio Waveform Synthesis for Oxideon.
Includes tone generators, frequency chirps, biquad IIR filters, and ADSR envelopes.
"""
import math
from typing import Optional
import numpy as np
from .oxideon import Tensor, from_numpy as _from_numpy


class SineGenerator:
    """Generates continuous pure sinusoidal waveforms."""
    def __init__(self, sample_rate: int = 16000):
        self.sample_rate = sample_rate

    def generate(self, freq_hz: float, duration_s: float, phase: float = 0.0) -> Tensor:
        num_samples = int(self.sample_rate * duration_s)
        t = np.arange(num_samples) / float(self.sample_rate)
        waveform = np.sin(2.0 * np.pi * freq_hz * t + phase).astype(np.float32)
        return _from_numpy(waveform)


class ChirpGenerator:
    """Generates frequency-swept chirps (linear or exponential frequency ramp)."""
    def __init__(self, sample_rate: int = 16000):
        self.sample_rate = sample_rate

    def generate(self, f0: float, f1: float, duration_s: float, method: str = "linear") -> Tensor:
        num_samples = int(self.sample_rate * duration_s)
        t = np.arange(num_samples) / float(self.sample_rate)
        if method == "linear":
            beta = (f1 - f0) / duration_s
            phase = 2.0 * np.pi * (f0 * t + 0.5 * beta * (t ** 2))
        elif method == "exponential":
            if f0 <= 0 or f1 <= 0:
                raise ValueError("f0 and f1 must be positive for exponential chirp")
            if f0 == f1:
                phase = 2.0 * np.pi * f0 * t
            else:
                w0 = 2.0 * np.pi * f0
                w1 = 2.0 * np.pi * f1
                phase = (w0 * duration_s / np.log(w1 / w0)) * ((w1 / w0) ** (t / duration_s) - 1.0)
        else:
            raise ValueError(f"Unknown chirp method: {method}")

        waveform = np.sin(phase).astype(np.float32)
        return _from_numpy(waveform)


class WhiteNoiseGenerator:
    """Generates Gaussian or uniform white noise."""
    def __init__(self, sample_rate: int = 16000, seed: Optional[int] = None):
        self.sample_rate = sample_rate
        self.rng = np.random.default_rng(seed)

    def generate(self, duration_s: float, std: float = 0.2) -> Tensor:
        num_samples = int(self.sample_rate * duration_s)
        noise = self.rng.normal(0.0, std, size=num_samples).astype(np.float32)
        return _from_numpy(noise)


class BiquadFilter:
    """
    Standard digital audio biquad IIR filter (Robert Bristow-Johnson Audio EQ Cookbook).
    Supports 'lowpass', 'highpass', 'bandpass', and 'notch'.
    """
    def __init__(self, filter_type: str = "lowpass", cutoff_hz: float = 1000.0, q: float = 0.7071, sample_rate: int = 16000):
        self.filter_type = filter_type.lower()
        self.cutoff_hz = float(cutoff_hz)
        self.q = float(q)
        self.sample_rate = int(sample_rate)
        self._compute_coefficients()

    def _compute_coefficients(self):
        w0 = 2.0 * np.pi * self.cutoff_hz / self.sample_rate
        cos_w0 = np.cos(w0)
        sin_w0 = np.sin(w0)
        alpha = sin_w0 / (2.0 * self.q)

        if self.filter_type == "lowpass":
            b0 = (1.0 - cos_w0) / 2.0
            b1 = 1.0 - cos_w0
            b2 = (1.0 - cos_w0) / 2.0
            a0 = 1.0 + alpha
            a1 = -2.0 * cos_w0
            a2 = 1.0 - alpha
        elif self.filter_type == "highpass":
            b0 = (1.0 + cos_w0) / 2.0
            b1 = -(1.0 + cos_w0)
            b2 = (1.0 + cos_w0) / 2.0
            a0 = 1.0 + alpha
            a1 = -2.0 * cos_w0
            a2 = 1.0 - alpha
        elif self.filter_type == "bandpass":
            b0 = alpha
            b1 = 0.0
            b2 = -alpha
            a0 = 1.0 + alpha
            a1 = -2.0 * cos_w0
            a2 = 1.0 - alpha
        elif self.filter_type == "notch":
            b0 = 1.0
            b1 = -2.0 * cos_w0
            b2 = 1.0
            a0 = 1.0 + alpha
            a1 = -2.0 * cos_w0
            a2 = 1.0 - alpha
        else:
            raise ValueError(f"Unsupported filter type: {self.filter_type}")

        self.b0 = b0 / a0
        self.b1 = b1 / a0
        self.b2 = b2 / a0
        self.a1 = a1 / a0
        self.a2 = a2 / a0

    def apply(self, signal: Tensor) -> Tensor:
        x = signal.to_numpy()
        y = np.zeros_like(x)

        if x.ndim == 1:
            d1, d2 = 0.0, 0.0
            for n in range(len(x)):
                xn = x[n]
                yn = self.b0 * xn + d1
                d1 = self.b1 * xn - self.a1 * yn + d2
                d2 = self.b2 * xn - self.a2 * yn
                y[n] = yn
        else:
            orig_shape = x.shape
            x_2d = x.reshape(-1, orig_shape[-1])
            y_2d = np.zeros_like(x_2d)
            for ch in range(x_2d.shape[0]):
                d1, d2 = 0.0, 0.0
                for n in range(x_2d.shape[1]):
                    xn = x_2d[ch, n]
                    yn = self.b0 * xn + d1
                    d1 = self.b1 * xn - self.a1 * yn + d2
                    d2 = self.b2 * xn - self.a2 * yn
                    y_2d[ch, n] = yn
            y = y_2d.reshape(orig_shape)

        return _from_numpy(y.astype(np.float32))


class EnvelopeADSR:
    """Attack, Decay, Sustain, Release (ADSR) amplitude envelope generator."""
    def __init__(self, attack_s: float = 0.05, decay_s: float = 0.05, sustain_level: float = 0.7, release_s: float = 0.1, sample_rate: int = 16000):
        self.attack_s = attack_s
        self.decay_s = decay_s
        self.sustain_level = sustain_level
        self.release_s = release_s
        self.sample_rate = sample_rate

    def generate(self, total_duration_s: float) -> Tensor:
        n_total = int(self.sample_rate * total_duration_s)
        if n_total <= 0:
            return _from_numpy(np.zeros(0, dtype=np.float32))
        n_att = int(self.sample_rate * self.attack_s)
        n_dec = int(self.sample_rate * self.decay_s)
        n_rel = int(self.sample_rate * self.release_s)

        fixed_sum = n_att + n_dec + n_rel
        if fixed_sum > n_total and fixed_sum > 0:
            scale = float(n_total) / float(fixed_sum)
            n_att = int(round(n_att * scale))
            n_dec = int(round(n_dec * scale))
            n_rel = max(1, n_total - n_att - n_dec)
            n_sus = 0
        else:
            n_sus = max(0, n_total - fixed_sum)

        att = np.linspace(0.0, 1.0, n_att, endpoint=False) if n_att > 0 else np.zeros(0)
        dec = np.linspace(1.0, self.sustain_level, n_dec, endpoint=False) if n_dec > 0 else np.zeros(0)
        sus = np.full(n_sus, self.sustain_level) if n_sus > 0 else np.zeros(0)
        rel = np.linspace(self.sustain_level, 0.0, n_rel) if n_rel > 0 else np.zeros(0)

        env = np.concatenate([att, dec, sus, rel])
        if len(env) < n_total:
            env = np.pad(env, (0, n_total - len(env)))
        elif len(env) > n_total:
            env = env[:n_total]

        return _from_numpy(env.astype(np.float32))


__all__ = [
    "SineGenerator",
    "ChirpGenerator",
    "WhiteNoiseGenerator",
    "BiquadFilter",
    "EnvelopeADSR",
]
