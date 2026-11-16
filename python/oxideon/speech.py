"""Speech front-end: VAD, diarization, enhancement, masks, fingerprints.

- ``frame_energies_db`` / ``energy_vad``: RMS energy voice activity detection
  with hangover bridging and minimum-duration pruning.
- ``SpeakerDiarizer``: windowed embeddings with greedy average-linkage
  clustering (caller-provided embedder).
- ``spectral_subtraction`` / ``wiener_filter``: classical single-channel
  enhancement reusing ``signal.stft`` / ``istft``.
- ``ideal_ratio_mask`` / ``apply_mask``: oracle time-frequency masking.
- ``audio_fingerprint`` / ``fingerprint_matches``: Shazam-style constellation
  fingerprints and offset-consistent match counts.
"""

from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import zlib
import math as _math
import re as _re

import numpy as np

from .oxideon import Tensor as _Tensor, from_numpy as _from_numpy
from .signal import istft as _istft, stft as _stft


def _mono(waveform: Any, name: str = "waveform") -> np.ndarray:
    arr = np.asarray(
        waveform.to_numpy() if isinstance(waveform, _Tensor) else waveform,
        dtype=np.float64,
    ).flatten()
    if arr.shape[0] < 1:
        raise ValueError(f"{name} must hold at least one sample")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must be finite")
    return arr


def frame_energies_db(
    waveform: Any, sample_rate: int, frame_ms: float = 30.0, hop_ms: float = 10.0
) -> _Tensor:
    """Per-frame RMS energy in dBFS (silence floors at -240 dB)."""
    if sample_rate < 1 or frame_ms <= 0 or hop_ms <= 0:
        raise ValueError("sample_rate/frame_ms/hop_ms must be positive")
    wav = _mono(waveform)
    frame = max(int(sample_rate * frame_ms / 1000.0), 1)
    hop = max(int(sample_rate * hop_ms / 1000.0), 1)
    if wav.shape[0] < frame:
        raise ValueError("waveform shorter than one frame")
    n_frames = 1 + (wav.shape[0] - frame) // hop
    idx = np.arange(frame)[None, :] + hop * np.arange(n_frames)[:, None]
    rms = np.sqrt(np.mean(wav[idx] ** 2, axis=1))
    db = 20.0 * np.log10(np.maximum(rms, 1e-12))
    return _from_numpy(np.ascontiguousarray(db.astype(np.float32)))


def energy_vad(
    waveform: Any,
    sample_rate: int,
    frame_ms: float = 30.0,
    hop_ms: float = 10.0,
    threshold_db: float = -40.0,
    hangover: int = 5,
    min_duration_ms: float = 100.0,
) -> List[Tuple[float, float]]:
    """Speech segments in seconds from an RMS energy gate with hangover."""
    if hangover < 0 or min_duration_ms < 0:
        raise ValueError("hangover and min_duration_ms must be non-negative")
    db = frame_energies_db(waveform, sample_rate, frame_ms, hop_ms).to_numpy()
    active = list(db > threshold_db)
    filled = active[:]
    i = 0
    n = len(filled)
    while i < n:
        if not filled[i]:
            j = i
            while j < n and not filled[j]:
                j += 1
            if i > 0 and j < n and j - i <= hangover:
                for k in range(i, j):
                    filled[k] = True
            i = j
        else:
            i += 1
    hop_s = max(int(sample_rate * hop_ms / 1000.0), 1) / sample_rate
    frame_s = max(int(sample_rate * frame_ms / 1000.0), 1) / sample_rate
    total = len(_mono(waveform)) / sample_rate
    segments = []
    i = 0
    while i < n:
        if not filled[i]:
            i += 1
            continue
        j = i
        while j + 1 < n and filled[j + 1]:
            j += 1
        start = i * hop_s
        end = min(j * hop_s + frame_s, total)
        if (end - start) * 1000.0 >= min_duration_ms:
            segments.append((start, end))
        i = j + 1
    return segments


class SpeakerDiarizer:
    """Windowed-embedding diarization with greedy average-linkage clustering."""

    def __init__(
        self,
        embed: Callable[[np.ndarray], Any],
        window_s: float = 1.5,
        hop_s: float = 0.75,
        threshold: float = 0.3,
    ):
        if not callable(embed):
            raise TypeError("embed must be callable")
        if window_s <= 0 or hop_s <= 0:
            raise ValueError("window_s and hop_s must be positive")
        if not 0.0 <= threshold <= 2.0:
            raise ValueError("threshold must sit within [0, 2] (cosine distance)")
        self.embed = embed
        self.window_s = window_s
        self.hop_s = hop_s
        self.threshold = threshold

    def diarize(self, waveform: Any, sample_rate: int) -> List[Tuple[float, float, int]]:
        """Greedy clustering of window embeddings into speaker segments."""
        if sample_rate < 1:
            raise ValueError("sample_rate must be positive")
        wav = _mono(waveform)
        win = int(self.window_s * sample_rate)
        hop = int(self.hop_s * sample_rate)
        if win < 1 or hop < 1 or wav.shape[0] < win:
            raise ValueError("waveform shorter than one diarization window")
        starts = list(range(0, wav.shape[0] - win + 1, hop))
        vecs = []
        for s in starts:
            vec = np.asarray(self.embed(wav[s : s + win]), dtype=np.float64).flatten()
            if vec.shape[0] < 1 or not np.all(np.isfinite(vec)):
                raise ValueError("embed must return a non-empty finite vector")
            norm = float(np.linalg.norm(vec))
            vecs.append(vec / norm if norm > 1e-12 else vec)
        mat = np.stack(vecs)
        dist = 1.0 - mat @ mat.T
        clusters: List[List[int]] = [[i] for i in range(len(starts))]
        while len(clusters) > 1:
            best = None
            for a in range(len(clusters)):
                for b in range(a + 1, len(clusters)):
                    sub = dist[np.ix_(clusters[a], clusters[b])]
                    score = float(sub.mean())
                    if best is None or score < best[0]:
                        best = (score, a, b)
            if best is None or best[0] >= self.threshold:
                break
            _, a, b = best
            clusters[a] = sorted(clusters[a] + clusters[b])
            del clusters[b]
        label_of = {}
        for label, members in enumerate(sorted(clusters, key=min)):
            for m in members:
                label_of[m] = label
        segments = []
        i = 0
        order = sorted(range(len(starts)))
        while i < len(order):
            j = i
            while (
                j + 1 < len(order)
                and label_of[order[j + 1]] == label_of[order[i]]
                and starts[order[j + 1]] <= starts[order[j]] + win
            ):
                j += 1
            segments.append(
                (starts[order[i]] / sample_rate, (starts[order[j]] + win) / sample_rate, label_of[order[i]])
            )
            i = j + 1
        return segments


def _tf_magnitude(noisy: np.ndarray, n_fft: int, hop: int) -> Tuple[np.ndarray, np.ndarray, int]:
    # Centered so overlap-add has full window coverage (avoids edge blowup).
    spec = _stft(_from_numpy(np.ascontiguousarray(noisy.astype(np.float32))), n_fft, hop_length=hop, center=True).to_numpy()
    mag = np.sqrt(spec[..., 0] ** 2 + spec[..., 1] ** 2)
    phase = np.arctan2(spec[..., 1], spec[..., 0])
    return mag, phase, spec.shape[1]


def _tf_synth(mag: np.ndarray, phase: np.ndarray, n_fft: int, hop: int, length: int) -> _Tensor:
    real_imag = np.stack([mag * np.cos(phase), mag * np.sin(phase)], axis=-1).astype(np.float32)
    return _istft(_from_numpy(np.ascontiguousarray(real_imag)), n_fft, hop_length=hop, center=True, length=length)


def spectral_subtraction(
    noisy: Any,
    n_fft: int = 512,
    hop_length: Optional[int] = None,
    noise_frames: int = 10,
    alpha: float = 1.0,
    floor: float = 0.01,
) -> _Tensor:
    """Magnitude spectral subtraction with a noise floor; phase is kept."""
    if n_fft < 8 or (hop_length is not None and hop_length < 1):
        raise ValueError("n_fft >= 8 and hop_length >= 1 are required")
    if noise_frames < 1:
        raise ValueError("noise_frames must be >= 1")
    if alpha < 0 or not 0.0 <= floor <= 1.0:
        raise ValueError("need alpha >= 0 and floor within [0, 1]")
    wav = _mono(noisy)
    hop = n_fft // 4 if hop_length is None else hop_length
    mag, phase, _ = _tf_magnitude(wav, n_fft, hop)
    if mag.shape[1] < noise_frames:
        raise ValueError("fewer frames than noise_frames")
    noise = mag[:, :noise_frames].mean(axis=1, keepdims=True)
    clean = np.maximum(mag - alpha * noise, floor * mag)
    return _tf_synth(clean, phase, n_fft, hop, wav.shape[0])


def wiener_filter(
    noisy: Any,
    n_fft: int = 512,
    hop_length: Optional[int] = None,
    noise_frames: int = 10,
    iterations: int = 2,
    floor: float = 0.01,
) -> _Tensor:
    """Iterative Wiener gain refinement from a leading-noise estimate."""
    if n_fft < 8 or (hop_length is not None and hop_length < 1):
        raise ValueError("n_fft >= 8 and hop_length >= 1 are required")
    if noise_frames < 1 or iterations < 1:
        raise ValueError("noise_frames and iterations must be >= 1")
    if not 0.0 <= floor <= 1.0:
        raise ValueError("floor must sit within [0, 1]")
    wav = _mono(noisy)
    hop = n_fft // 4 if hop_length is None else hop_length
    mag, phase, _ = _tf_magnitude(wav, n_fft, hop)
    if mag.shape[1] < noise_frames:
        raise ValueError("fewer frames than noise_frames")
    noise_pow = (mag[:, :noise_frames] ** 2).mean(axis=1, keepdims=True)
    clean_pow = np.maximum(mag**2 - noise_pow, 1e-12)
    for _ in range(iterations):
        gain = clean_pow / (clean_pow + noise_pow)
        gain = np.maximum(gain, floor)
        clean_pow = (gain * mag) ** 2
    out_mag = np.sqrt(np.maximum(clean_pow, 0.0))
    return _tf_synth(out_mag, phase, n_fft, hop, wav.shape[0])


def _complex_spec(spec: Any, name: str) -> np.ndarray:
    if isinstance(spec, _Tensor):
        arr = spec.to_numpy()
        if arr.ndim == 3 and arr.shape[-1] == 2:
            return arr[..., 0] + 1j * arr[..., 1]
        return np.asarray(arr, dtype=np.complex128)
    arr = np.asarray(spec)
    if arr.ndim == 3 and arr.shape[-1] == 2 and not np.iscomplexobj(arr):
        return arr[..., 0] + 1j * arr[..., 1]
    return np.asarray(arr, dtype=np.complex128)


def ideal_ratio_mask(mix_spec: Any, target_spec: Any, beta: float = 0.5) -> np.ndarray:
    """Oracle mask (|S|^2 / (|S|^2 + |N|^2))^beta with N = mix - target."""
    if beta < 0:
        raise ValueError("beta must be non-negative")
    mix = _complex_spec(mix_spec, "mix_spec")
    tgt = _complex_spec(target_spec, "target_spec")
    if mix.shape != tgt.shape:
        raise ValueError(f"target shape {tgt.shape} != mix shape {mix.shape}")
    ps, pn = np.abs(tgt) ** 2, np.abs(mix - tgt) ** 2
    return (ps / np.maximum(ps + pn, 1e-12)) ** beta


def apply_mask(mix_spec: Any, mask: Any) -> Any:
    """Applies a real time-frequency mask to a complex spectrogram."""
    if isinstance(mix_spec, _Tensor) or isinstance(mask, _Tensor):
        s_t = mix_spec if isinstance(mix_spec, _Tensor) else _from_numpy(np.asarray(mix_spec, dtype=np.float32))
        m_t = mask if isinstance(mask, _Tensor) else _from_numpy(np.asarray(mask, dtype=np.float32))
        if s_t.ndim == 3 and s_t.shape[-1] == 2 and m_t.ndim == 2:
            return s_t * m_t.unsqueeze(-1)
        return s_t * m_t

    mix = _complex_spec(mix_spec, "mix_spec")
    gain = np.asarray(mask, dtype=np.float64)
    if gain.shape != mix.shape:
        raise ValueError(f"mask shape {gain.shape} != mix shape {mix.shape}")
    if (gain < 0).any():
        raise ValueError("mask must be non-negative")
    return mix * gain


def mel_filterbank(
    sr: int,
    n_fft: int,
    n_mels: int = 80,
    f_min: float = 0.0,
    f_max: Optional[float] = None,
) -> _Tensor:
    """Constructs a triangular Mel filterbank matrix of shape [n_mels, n_fft // 2 + 1]."""
    if f_max is None:
        f_max = float(sr // 2)
    n_freqs = n_fft // 2 + 1
    m_min = 2595.0 * _math.log10(1.0 + f_min / 700.0)
    m_max = 2595.0 * _math.log10(1.0 + f_max / 700.0)
    m_pts = np.linspace(m_min, m_max, n_mels + 2)
    f_pts = 700.0 * (10.0 ** (m_pts / 2595.0) - 1.0)
    bins = np.floor((n_fft + 1) * f_pts / sr).astype(int)
    fb = np.zeros((n_mels, n_freqs), dtype=np.float32)
    for m in range(1, n_mels + 1):
        f_m_minus = bins[m - 1]
        f_m = bins[m]
        f_m_plus = bins[m + 1]
        for k in range(f_m_minus, f_m):
            if f_m > f_m_minus and k < n_freqs:
                fb[m - 1, k] = (k - f_m_minus) / (f_m - f_m_minus)
        for k in range(f_m, f_m_plus):
            if f_m_plus > f_m and k < n_freqs:
                fb[m - 1, k] = (f_m_plus - k) / (f_m_plus - f_m)
    return _from_numpy(fb)


def compute_mel_spectrogram(
    waveform: Any,
    sr: int,
    n_fft: int = 512,
    hop_length: Optional[int] = None,
    n_mels: int = 80,
) -> _Tensor:
    """Computes Mel-scale power/magnitude spectrogram from 1D audio waveform."""
    t = waveform if isinstance(waveform, _Tensor) else _from_numpy(np.asarray(waveform, dtype=np.float32))
    hop = n_fft // 4 if hop_length is None else hop_length
    spec = _stft(t, n_fft, hop_length=hop, center=True)
    spec_arr = spec.to_numpy()
    mag = np.sqrt(spec_arr[..., 0] ** 2 + spec_arr[..., 1] ** 2)
    fb = mel_filterbank(sr, n_fft, n_mels=n_mels).to_numpy()
    mel_spec = np.dot(fb, mag)
    return _from_numpy(np.ascontiguousarray(mel_spec, dtype=np.float32))


def audio_fingerprint(
    waveform: Any,
    sample_rate: int,
    n_fft: int = 1024,
    hop_length: Optional[int] = None,
    top_k: int = 5,
    fanout: int = 5,
    max_dt: int = 100,
) -> List[Tuple[int, int]]:
    """Shazam-style constellation fingerprints: [(hash, anchor_frame)]."""
    if sample_rate < 1 or n_fft < 16:
        raise ValueError("sample_rate >= 1 and n_fft >= 16 are required")
    if hop_length is not None and hop_length < 1:
        raise ValueError("hop_length must be >= 1")
    if top_k < 1 or fanout < 1 or max_dt < 1:
        raise ValueError("top_k, fanout and max_dt must be >= 1")
    wav = _mono(waveform)
    hop = n_fft // 4 if hop_length is None else hop_length
    mag, _, _ = _tf_magnitude(wav, n_fft, hop)
    floor = np.median(mag) * 2.0 + 1e-9
    peaks = []
    for t in range(mag.shape[1]):
        order = np.argsort(-mag[:, t], kind="stable")[:top_k]
        for f in order:
            if mag[f, t] > floor:
                peaks.append((t, int(f)))
    peaks.sort()
    prints = []
    for i, (t1, f1) in enumerate(peaks):
        for t2, f2 in peaks[i + 1 : i + 1 + fanout]:
            dt = t2 - t1
            if dt < 1 or dt > max_dt:
                continue
            code = zlib.crc32(f"{f1},{f2},{dt}".encode("ascii")) & 0xFFFFFFFF
            prints.append((code, t1))
    return prints


def fingerprint_matches(
    fp_a: Sequence[Tuple[int, int]], fp_b: Sequence[Tuple[int, int]]
) -> Tuple[int, int]:
    """Best (offset, count) of offset-consistent shared hashes; (0, 0) if none."""
    times_b: Dict[int, List[int]] = {}
    for code, t in fp_b:
        times_b.setdefault(code, []).append(t)
    offsets: Dict[int, int] = {}
    for code, ta in fp_a:
        for tb in times_b.get(code, ()):
            offsets[ta - tb] = offsets.get(ta - tb, 0) + 1
    if not offsets:
        return (0, 0)
    best = max(sorted(offsets), key=lambda o: (offsets[o], -abs(o)))
    return (best, offsets[best])


def _to_np(t: Any) -> np.ndarray:
    if isinstance(t, _Tensor):
        return t.to_numpy()
    return np.asarray(t)


def verification_scores(enroll: Any, test: Any) -> _Tensor:
    """Cosine similarity for paired enrollment/test embeddings [N, D]."""
    a = _to_np(enroll).astype(np.float64)
    b = _to_np(test).astype(np.float64)
    if a.ndim != 2 or b.shape != a.shape:
        raise ValueError("enroll/test must be matching [N, D]")
    if len(a) == 0:
        raise ValueError("need at least one trial")
    if not np.all(np.isfinite(a)) or not np.all(np.isfinite(b)):
        raise ValueError("embeddings must be finite")
    na = np.linalg.norm(a, axis=1)
    nb = np.linalg.norm(b, axis=1)
    if np.any(na == 0) or np.any(nb == 0):
        raise ValueError("embeddings must have non-zero norm")
    s = np.sum(a * b, axis=1) / na / nb
    return _from_numpy(np.ascontiguousarray(s.astype(np.float32)))


def _far_frr(scores: np.ndarray, labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    order = np.argsort(-scores, kind="stable")
    s, y = scores[order], labels[order]
    n_pos = int(y.sum())
    n_neg = len(y) - n_pos
    if n_pos == 0 or n_neg == 0:
        raise ValueError("labels must contain both classes")
    tp = np.cumsum(y)
    fp = np.cumsum(1 - y)
    # Operating points after each distinct score, plus reject-all/all-accept ends.
    fpr = np.concatenate([[0.0], fp / n_neg])
    fnr = np.concatenate([[1.0], 1.0 - tp / n_pos])
    thr = np.concatenate([[np.inf], s])
    return fpr, fnr, thr


def equal_error_rate(scores: Any, labels: Any) -> Tuple[float, float]:
    """EER (interpolated FAR=FRR crossing) and threshold at nearest point."""
    s = _to_np(scores).astype(np.float64).reshape(-1)
    y = _to_np(labels).astype(np.int64).reshape(-1)
    if s.shape != y.shape or len(s) == 0:
        raise ValueError("scores/labels must be non-empty equal-length vectors")
    if not np.all(np.isfinite(s)) or not set(np.unique(y)).issubset({0, 1}):
        raise ValueError("scores must be finite and labels binary")
    fpr, fnr, thr = _far_frr(s, y)
    diff = fpr - fnr
    eer = float(np.min(np.maximum(fpr, fnr)))
    best_t = float(thr[int(np.argmin(np.maximum(fpr, fnr)))])
    for i in range(len(diff) - 1):
        if diff[i] == 0:
            return float(fpr[i]), float(thr[i])
        if diff[i] * diff[i + 1] < 0:
            t = diff[i] / (diff[i] - diff[i + 1])
            eer = float(fpr[i] + t * (fpr[i + 1] - fpr[i]))
            best_t = float(thr[i])
            break
    return eer, best_t


def min_detection_cost(
    scores: Any,
    labels: Any,
    p_target: float = 0.01,
    c_miss: float = 1.0,
    c_fa: float = 1.0,
) -> float:
    """Minimum NIST detection cost, normalized by the naive-system cost."""
    if not 0.0 < float(p_target) < 1.0:
        raise ValueError("p_target must be in (0, 1)")
    if not float(c_miss) > 0 or not float(c_fa) > 0:
        raise ValueError("costs must be positive")
    s = _to_np(scores).astype(np.float64).reshape(-1)
    y = _to_np(labels).astype(np.int64).reshape(-1)
    if s.shape != y.shape or len(s) == 0:
        raise ValueError("scores/labels must be non-empty equal-length vectors")
    if not np.all(np.isfinite(s)) or not set(np.unique(y)).issubset({0, 1}):
        raise ValueError("scores must be finite and labels binary")
    fpr, fnr, _ = _far_frr(s, y)
    p = float(p_target)
    dcf = float(c_miss) * p * fnr + float(c_fa) * (1 - p) * fpr
    naive = min(float(c_miss) * p, float(c_fa) * (1 - p))
    return float(dcf.min() / naive)


def det_curve(scores: Any, labels: Any) -> Dict[str, List[float]]:
    """DET curve points: fpr/fnr/thresholds from accept-all to reject-all."""
    s = _to_np(scores).astype(np.float64).reshape(-1)
    y = _to_np(labels).astype(np.int64).reshape(-1)
    if s.shape != y.shape or len(s) == 0:
        raise ValueError("scores/labels must be non-empty equal-length vectors")
    if not np.all(np.isfinite(s)) or not set(np.unique(y)).issubset({0, 1}):
        raise ValueError("scores must be finite and labels binary")
    fpr, fnr, thr = _far_frr(s, y)
    return {"fpr": fpr.tolist(), "fnr": fnr.tolist(), "thresholds": thr.tolist()}


class NGramLM:
    """Word n-gram LM with Jelinek-Mercer interpolation + uniform floor.

    P(w|h) = sum_m lambda[m] * P_ML(w|h_m) + lambda[0] / (V + 1), where h_m is
    the order-m context. Unseen words share the floor mass (open vocabulary).

    Args:
        n: Maximum order (>= 1). lambdas: Length-(n+1) weights for
            [floor, unigram, ..., n-gram]; default uniform.
    """

    def __init__(self, n: int = 3, lambdas: Optional[Sequence[float]] = None) -> None:
        if int(n) < 1:
            raise ValueError("n must be >= 1")
        self.n = int(n)
        if lambdas is None:
            self.lambdas = [1.0 / (self.n + 1)] * (self.n + 1)
        else:
            lam = [float(v) for v in lambdas]
            if len(lam) != self.n + 1 or any(v < 0 for v in lam) or sum(lam) <= 0:
                raise ValueError(f"lambdas must be {self.n + 1} non-negative weights")
            tot = sum(lam)
            self.lambdas = [v / tot for v in lam]
        self.counts: List[Dict[Tuple[str, ...], int]] = [{} for _ in range(self.n + 1)]
        self.vocab: set = set()
        self.n_tokens = 0

    def fit(self, corpus: Sequence[Sequence[str]]) -> "NGramLM":
        """Counts n-grams over tokenized sentences (chainable)."""
        if not corpus:
            raise ValueError("corpus must be non-empty")
        counts: List[Dict[Tuple[str, ...], int]] = [{} for _ in range(self.n + 1)]
        vocab: set = set()
        n_tokens = 0
        for sent in corpus:
            toks = [str(w) for w in sent]
            if not toks:
                continue
            vocab.update(toks)
            n_tokens += len(toks)
            for m in range(1, self.n + 1):
                for i in range(len(toks) - m + 1):
                    key = tuple(toks[i : i + m])
                    counts[m][key] = counts[m].get(key, 0) + 1
        if n_tokens == 0:
            raise ValueError("corpus holds no tokens")
        self.counts, self.vocab, self.n_tokens = counts, vocab, n_tokens
        return self

    @property
    def fitted(self) -> bool:
        return self.n_tokens > 0

    def _ml(self, context: Tuple[str, ...], word: str) -> float:
        if not context:
            c = self.counts[1].get((word,), 0)
            return c / self.n_tokens if self.n_tokens else 0.0
        ctx = self.counts[len(context)].get(context, 0)
        if ctx == 0:
            return 0.0
        return self.counts[len(context) + 1].get(context + (word,), 0) / ctx

    def prob(self, word: str, context: Sequence[str] = ()) -> float:
        """P(word | context) with interpolation (context clipped to n-1)."""
        if not self.fitted:
            raise ValueError("language model is not fitted")
        ctx = tuple(str(w) for w in context)[-(self.n - 1) :] if self.n > 1 else ()
        floor = 1.0 / (len(self.vocab) + 1)
        p = self.lambdas[0] * floor
        for m in range(1, self.n + 1):
            sub = ctx[-(m - 1) :] if m > 1 else ()
            p += self.lambdas[m] * self._ml(sub, str(word))
        return max(p, 1e-12)

    def sentence_logprob(self, words: Sequence[str]) -> float:
        """Sum of log P(w_i | w_<i) over the sentence (natural log)."""
        toks = [str(w) for w in words]
        if not toks:
            raise ValueError("sentence must be non-empty")
        return sum(_math.log(self.prob(w, toks[:i])) for i, w in enumerate(toks))

    def perplexity(self, words: Sequence[str]) -> float:
        """exp(-mean logprob) over the word sequence."""
        toks = [str(w) for w in words]
        if not toks:
            raise ValueError("sequence must be non-empty")
        return float(_math.exp(-self.sentence_logprob(toks) / len(toks)))


def beam_rescore(
    prefixes: Sequence[Tuple[Sequence[str], float]],
    lm: NGramLM,
    lm_weight: float = 1.0,
) -> List[Tuple[Tuple[str, ...], float, float, float]]:
    """Shallow fusion: reranks (words, acoustic_score) by am + w * lm_score.

    Returns [(words, combined, acoustic, lm)] sorted by decreasing combined.
    """
    if not prefixes:
        raise ValueError("prefixes must be non-empty")
    if not isinstance(lm, NGramLM):
        raise ValueError("lm must be an NGramLM instance")
    if not lm.fitted:
        raise ValueError("language model is not fitted")
    if not np.isfinite(lm_weight):
        raise ValueError("lm_weight must be finite")
    out = []
    for words, am in prefixes:
        toks = tuple(str(w) for w in words)
        if not toks:
            raise ValueError("prefixes must hold non-empty word sequences")
        lms = lm.sentence_logprob(toks)
        out.append((toks, float(am) + float(lm_weight) * lms, float(am), lms))
    return sorted(out, key=lambda r: r[1], reverse=True)


def ctc_forced_align(
    log_probs: Any, targets: Sequence[int], blank: int = 0
) -> Tuple[List[int], List[Tuple[int, int]]]:
    """Viterbi CTC forced alignment of emissions to a target sequence.

    Args:
        log_probs: [T, C] log-probabilities. targets: [U] non-blank ids.
        blank: Blank id. Requires T >= 2 * U + 1.

    Returns:
        (frame_labels [T] over the blank-interleaved alphabet, segments [(s, e)]
        inclusive frame spans per target unit).
    """
    lp = _to_np(log_probs).astype(np.float64)
    tgt = [int(v) for v in targets]
    if lp.ndim != 2 or lp.shape[0] == 0 or lp.shape[1] == 0:
        raise ValueError("log_probs must be non-empty [T, C]")
    if not np.all(np.isfinite(lp)):
        raise ValueError("log_probs must be finite")
    if not tgt:
        raise ValueError("targets must be non-empty")
    if not 0 <= int(blank) < lp.shape[1]:
        raise ValueError("blank must be in [0, C)")
    if any(v == int(blank) or not 0 <= v < lp.shape[1] for v in tgt):
        raise ValueError("targets must be non-blank ids in [0, C)")
    ext = [int(blank)]
    for v in tgt:
        ext += [v, int(blank)]
    n_frames, n_states = lp.shape[0], len(ext)
    if n_frames < n_states:
        raise ValueError(f"need T >= 2U+1 = {n_states}, got T = {n_frames}")
    neg = float("-inf")
    dp = np.full((n_frames, n_states), neg)
    bp = np.zeros((n_frames, n_states), dtype=np.int64)
    dp[0, 0] = lp[0, ext[0]]
    if n_states > 1:
        dp[0, 1] = lp[0, ext[1]]
    for t in range(1, n_frames):
        for s in range(n_states):
            cands = [(dp[t - 1, s], s)]
            if s - 1 >= 0:
                cands.append((dp[t - 1, s - 1], s - 1))
            if s - 2 >= 0 and ext[s] != int(blank) and ext[s] != ext[s - 2]:
                cands.append((dp[t - 1, s - 2], s - 2))
            best_v, best_s = max(cands, key=lambda c: c[0])
            dp[t, s] = best_v + lp[t, ext[s]]
            bp[t, s] = best_s
    end = n_states - 1 if dp[-1, n_states - 1] >= dp[-1, n_states - 2] else n_states - 2
    path = [0] * n_frames
    path[-1] = end
    for t in range(n_frames - 2, -1, -1):
        path[t] = int(bp[t + 1, path[t + 1]])
    labels = [ext[s] for s in path]
    segments = []
    for u in range(len(tgt)):
        frames = [t for t, s in enumerate(path) if s == 2 * u + 1]
        segments.append((min(frames), max(frames)))
    return labels, segments


_ONES = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight",
         "nine", "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen",
         "sixteen", "seventeen", "eighteen", "nineteen"]
_TENS = ["", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"]
_MONTHS = ["january", "february", "march", "april", "may", "june", "july", "august",
           "september", "october", "november", "december"]
_ABBR = {"dr": "doctor", "mr": "mister", "mrs": "missus", "ms": "miss",
         "vs": "versus", "eg": "for example", "ie": "that is"}


def _cardinal(n: int) -> str:
    if n < 0:
        return "minus " + _cardinal(-n)
    if n < 20:
        return _ONES[n]
    if n < 100:
        return _TENS[n // 10] + ("" if n % 10 == 0 else " " + _ONES[n % 10])
    if n < 1000:
        rest = n % 100
        return _ONES[n // 100] + " hundred" + ("" if rest == 0 else " " + _cardinal(rest))
    for scale, name in ((10**9, "billion"), (10**6, "million"), (10**3, "thousand")):
        if n >= scale:
            hi, rest = divmod(n, scale)
            return _cardinal(hi) + " " + name + ("" if rest == 0 else " " + _cardinal(rest))
    raise AssertionError("unreachable")


def _ordinal_word(n: int) -> str:
    if n <= 0:
        raise ValueError("ordinal needs a positive int")
    special = {1: "first", 2: "second", 3: "third", 5: "fifth", 8: "eighth",
               9: "ninth", 12: "twelfth"}
    if n in special:
        return special[n]
    if n < 20:
        return _ONES[n] + "th"
    if n < 100 and n % 10 == 0:
        return _TENS[n // 10][:-1] + "ieth"
    words = _cardinal(n).split(" ")
    tail = words[-1]
    if tail in ("hundred", "thousand", "million", "billion"):
        words[-1] = tail + "th"
    elif tail in _ONES:
        idx = _ONES.index(tail)
        words[-1] = special.get(idx, tail + "th")
    elif tail in _TENS:
        words[-1] = tail[:-1] + "ieth"
    else:
        words[-1] = tail + "th"
    return " ".join(words)


def _digit_words(digits: str) -> str:
    return " ".join(_ONES[int(d)] for d in digits)


def _year_words(y: int) -> str:
    if y < 1000 or y > 2999:
        return _cardinal(y)
    if y % 100 == 0:
        return _cardinal(y // 100) + " hundred" if y not in (1000, 2000) else _cardinal(y // 1000) + " thousand"
    if 2000 <= y <= 2009:
        return "two thousand " + _ONES[y % 10]
    hi, lo = y // 100, y % 100
    if lo < 10:
        return _cardinal(hi) + " oh " + _ONES[lo]
    return _cardinal(hi) + " " + _cardinal(lo)


def _money_words(amount: str, unit: str, subunit: str) -> str:
    if "." in amount:
        major, minor = amount.split(".")
        minor = (minor + "00")[:2]
    else:
        major, minor = amount, "00"
    parts = []
    if int(major) > 0:
        parts.append(_cardinal(int(major)) + " " + unit + ("s" if int(major) != 1 else ""))
    if int(minor) > 0:
        parts.append(_cardinal(int(minor)) + " " + subunit + ("s" if int(minor) != 1 else ""))
    return " and ".join(parts) if parts else "zero " + unit + "s"


def _decimal_words(num: str) -> str:
    head, tail = num.split(".")
    return _cardinal(int(head)) + " point " + _digit_words(tail)


def normalize_text(text: str) -> str:
    """Rule-based TTS text normalization (lowercase English, no lexicon).

    Expands currency ($/EUR/GBP), percents, times, MM/DD/YYYY dates, years,
    ordinals, digit-hyphen runs (phone-like), decimals, comma/cardinal numbers
    (7+ bare digits read digit-wise), fractions, abbreviations and symbols.
    """
    if not isinstance(text, str):
        raise ValueError("text must be a string")
    s = text.lower()

    def _money(m: "_re.Match") -> str:
        sym, amt = m.group(1), m.group(2).replace(",", "")
        units = {"$": ("dollar", "cent"), "\u20ac": ("euro", "cent"), "\u00a3": ("pound", "penny")}
        unit, sub = units[sym]
        out = _money_words(amt, unit, "cent" if sym != "\u00a3" else "penny")
        return out.replace("pennys", "pennies").replace("pennyes", "pennies")

    s = _re.sub(r"([$€£])(\d[\d,]*(?:\.\d{1,2})?)", _money, s)
    s = _re.sub(r"(\d+(?:\.\d+)?)%", lambda m: (_decimal_words(m.group(1)) if "." in m.group(1) else _cardinal(int(m.group(1)))) + " percent", s)
    s = _re.sub(r"\b(\d{1,2}):(\d{2})\s*(am|pm)?",
                lambda m: _cardinal(int(m.group(1))) + " " + (_digit_words(m.group(2)) if m.group(2).startswith("0") else _cardinal(int(m.group(2)))) + ("" if not m.group(3) else " " + " ".join(m.group(3))), s)
    s = _re.sub(r"\b(\d{1,2})/(\d{1,2})/(\d{4})",
                lambda m: _MONTHS[int(m.group(1)) - 1] + " " + _ordinal_word(int(m.group(2))) + " " + _year_words(int(m.group(3))) if 1 <= int(m.group(1)) <= 12 else m.group(0), s)
    s = _re.sub(r"\b(\d+)/(\d+)\b",
                lambda m: "one half" if (int(m.group(1)), int(m.group(2))) == (1, 2)
                else _cardinal(int(m.group(1))) + " " + _ordinal_word(int(m.group(2))) + ("s" if int(m.group(1)) != 1 else ""), s)
    s = _re.sub(r"\b(\d+)-(\d+(?:-\d+)*)\b", lambda m: _digit_words(m.group(0).replace("-", "")), s)
    s = _re.sub(r"-(\d)", r"minus \1", s)
    s = _re.sub(r"\b(1\d{3}|2\d{3})\b", lambda m: _year_words(int(m.group(1))), s)
    s = _re.sub(r"\b(\d+)(st|nd|rd|th)\b", lambda m: _ordinal_word(int(m.group(1))), s)
    s = s.replace("e.g.", "for example").replace("i.e.", "that is")
    for abbr, full in sorted(_ABBR.items(), key=lambda kv: -len(kv[0])):
        s = _re.sub(r"\b" + abbr + r"\.", full, s)
    s = _re.sub(r"\bvs\b", "versus", s)
    s = s.replace("&", " and ").replace("+", " plus ").replace("=", " equals ").replace("@", " at ")
    s = _re.sub(r"\d+\*\d+", " times ", s).replace("*", " star ").replace("#", " number ")
    s = _re.sub(r"(\d[\d,]*)\.(\d+)", lambda m: _decimal_words(m.group(1).replace(",", "") + "." + m.group(2)), s)

    def _num(m: "_re.Match") -> str:
        tok = m.group(0)
        if "," in tok:
            return _cardinal(int(tok.replace(",", "")))
        if len(tok) >= 7:
            return _digit_words(tok)
        return _cardinal(int(tok))

    s = _re.sub(r"\b\d[\d,]*\b", _num, s)
    s = s.replace("-", " ")
    s = _re.sub(r"[^a-z0-9\s]", "", s)
    return _re.sub(r"\s+", " ", s).strip()

__all__ = [
    "frame_energies_db",
    "energy_vad",
    "SpeakerDiarizer",
    "spectral_subtraction",
    "wiener_filter",
    "ideal_ratio_mask",
    "apply_mask",
    "mel_filterbank",
    "compute_mel_spectrogram",
    "audio_fingerprint",
    "fingerprint_matches",
    "verification_scores",
    "equal_error_rate",
    "min_detection_cost",
    "det_curve",
    "NGramLM",
    "beam_rescore",
    "ctc_forced_align",
    "normalize_text",
]
