"""Video analytics: background subtraction, shot detection, interpolation, stabilization.

- ``MOGBackgroundSubtractor``: Stauffer-Grimson mixture-of-Gaussians foreground
  segmentation (grayscale, deterministic).
- ``detect_shot_boundaries``: histogram shot cuts with single-frame flash veto.
- ``interpolate_frames``: block-matching motion-compensated mid-frame (t=0.5).
- ``estimate_translation``: FFT phase-correlation shift (parabolic subpixel).
- ``stabilize_frames``: translation-only stabilization with valid-region crop.
"""

from typing import Any, Dict, List, Optional, Sequence, Tuple
import numpy as np

from .oxideon import Tensor as _Tensor, from_numpy as _from_numpy


def _to_np(t: Any) -> np.ndarray:
    if isinstance(t, _Tensor):
        return t.to_numpy()
    return np.asarray(t)


def _as_gray(frame: np.ndarray, name: str) -> np.ndarray:
    f = np.asarray(frame, dtype=np.float64)
    if f.ndim == 3 and f.shape[2] == 3:
        return f.mean(axis=2)
    if f.ndim == 2:
        return f
    raise ValueError(f"{name} must be [H, W] or [H, W, 3], got shape {f.shape}")


class MOGBackgroundSubtractor:
    """Mixture-of-Gaussians background subtraction (grayscale, K modes/px).

    Args:
        n_modes: Gaussians per pixel (>= 1). learning_rate: Weight adaptation
            rate in (0, 1). bg_threshold: Cumulative-weight cutoff selecting
            background modes. init_var: Initial variance (pixel-scale units).
            match_dist: Match gate in standard deviations.
    """

    def __init__(
        self,
        n_modes: int = 3,
        learning_rate: float = 0.01,
        bg_threshold: float = 0.7,
        init_var: float = 900.0,
        match_dist: float = 2.5,
    ) -> None:
        if int(n_modes) < 1:
            raise ValueError("n_modes must be >= 1")
        if not 0.0 < float(learning_rate) < 1.0:
            raise ValueError("learning_rate must be in (0, 1)")
        if not 0.0 < float(bg_threshold) < 1.0:
            raise ValueError("bg_threshold must be in (0, 1)")
        if not np.isfinite(init_var) or init_var <= 0:
            raise ValueError("init_var must be finite and positive")
        if not np.isfinite(match_dist) or match_dist <= 0:
            raise ValueError("match_dist must be finite and positive")
        self.n_modes = int(n_modes)
        self.lr = float(learning_rate)
        self.bg_threshold = float(bg_threshold)
        self.init_var = float(init_var)
        self.match_dist = float(match_dist)
        self._w: Optional[np.ndarray] = None
        self._mu: Optional[np.ndarray] = None
        self._var: Optional[np.ndarray] = None

    @property
    def initialized(self) -> bool:
        return self._w is not None

    def _init_state(self, frame: np.ndarray) -> None:
        h, w = frame.shape
        k = self.n_modes
        self._w = np.full((h, w, k), 1.0 / k)
        self._mu = np.broadcast_to(frame[:, :, None], (h, w, k)).copy()
        self._var = np.full((h, w, k), self.init_var)

    def apply(self, frame: Any) -> _Tensor:
        """Updates the model and returns the foreground mask [H, W] uint8."""
        f = _as_gray(_to_np(frame), "frame")
        if not np.all(np.isfinite(f)):
            raise ValueError("frame must be finite")
        if self._w is None:
            self._init_state(f)
        assert self._w is not None and self._mu is not None and self._var is not None
        if f.shape != self._w.shape[:2]:
            raise ValueError(f"frame shape {f.shape} disagrees with model {self._w.shape[:2]}")
        w, mu, var = self._w, self._mu, self._var
        x = f[:, :, None]
        sigma = np.sqrt(np.maximum(var, 1e-12))
        match = np.abs(x - mu) < self.match_dist * sigma
        has = match.any(axis=2)
        first = np.argmax(match, axis=2)
        # Weight update: matched mode toward 1, others decay.
        w *= 1.0 - self.lr
        w[has, first[has]] += self.lr
        # Mean/variance update for the matched mode (rho-weighted).
        rho = self.lr * np.exp(-0.5 * ((x - mu) / sigma) ** 2) / (np.sqrt(2 * np.pi) * sigma + 1e-12)
        upd = np.zeros_like(w)
        upd[has, first[has]] = rho[has, first[has]]
        mu += upd * (x - mu)
        var += upd * ((x - mu) ** 2 - var)
        var[:] = np.clip(var, 4.0, 1e8)
        # Unmatched pixels: replace the lowest-weight mode.
        if not bool(has.all()):
            yy, xx = np.nonzero(~has)
            victims = np.argmin(w[yy, xx], axis=1)
            w[yy, xx, victims] = self.lr
            mu[yy, xx, victims] = f[yy, xx]
            var[yy, xx, victims] = self.init_var
            w[yy, xx] /= w[yy, xx].sum(axis=1, keepdims=True)
        # Foreground test against background modes (ranked by w/sigma).
        rank = np.argsort(-w / np.sqrt(np.maximum(var, 1e-12)), axis=2, kind="stable")
        ordered = np.take_along_axis(match, rank, axis=2)
        cumw = np.cumsum(np.take_along_axis(w, rank, axis=2), axis=2)
        n_bg = np.sum(cumw <= self.bg_threshold, axis=2) + 1
        n_bg = np.clip(n_bg, 1, self.n_modes)
        bg_hit = np.zeros_like(has)
        for b in range(self.n_modes):
            bg_hit |= ordered[:, :, b] & (n_bg > b)
        return _from_numpy(np.ascontiguousarray((~bg_hit).astype(np.uint8)))

    def get_background(self) -> _Tensor:
        """Mean of the highest-weight mode per pixel [H, W] float32."""
        if self._w is None or self._mu is None or self._var is None:
            raise ValueError("subtractor not initialized; call apply() first")
        best = np.argmax(self._w / np.sqrt(np.maximum(self._var, 1e-12)), axis=2)
        bg = self._mu[np.arange(self._w.shape[0])[:, None], np.arange(self._w.shape[1]), best]
        return _from_numpy(np.ascontiguousarray(bg.astype(np.float32)))


def _frame_hist(frame: np.ndarray, bins: int = 32) -> np.ndarray:
    if frame.ndim == 3:
        parts = [np.histogram(frame[:, :, c], bins=bins, range=(0, 256))[0] for c in range(frame.shape[2])]
        h = np.concatenate(parts).astype(np.float64)
    else:
        h = np.histogram(frame, bins=bins, range=(0, 256))[0].astype(np.float64)
    s = h.sum()
    return h / s if s > 0 else h


def detect_shot_boundaries(
    frames: Sequence[Any], thresh: float = 0.5, bins: int = 32
) -> Dict[str, List[int]]:
    """Histogram shot-cut detection with single-frame flash veto.

    Distance is histogram-intersection distance in [0, 1]. A cut at t marks a
    boundary between frames t-1 and t. A frame flanked by two large distances
    that are mutually similar is reported as a flash, not a cut.

    Returns:
        {"cuts": [...], "flashes": [...]} frame indices.
    """
    if not np.isfinite(thresh) or not 0.0 < float(thresh) < 1.0:
        raise ValueError("thresh must be in (0, 1)")
    if int(bins) < 2:
        raise ValueError("bins must be >= 2")
    arr = [np.asarray(_to_np(f), dtype=np.float64) for f in frames]
    if not arr:
        raise ValueError("frames must be non-empty")
    shapes = {a.shape for a in arr}
    if len(shapes) != 1:
        raise ValueError("all frames must share one shape")
    if any(a.ndim not in (2, 3) for a in arr):
        raise ValueError("frames must be [H, W] or [H, W, 3]")
    hists = [_frame_hist(a, int(bins)) for a in arr]
    dists = [0.0] * len(arr)
    for t in range(1, len(arr)):
        dists[t] = 1.0 - float(np.minimum(hists[t - 1], hists[t]).sum())
    cuts, flashes = [], []
    skip_next = False  # frame after a flash still differs from it: not a cut
    for t in range(1, len(arr)):
        if skip_next:
            skip_next = False
            continue
        if dists[t] <= thresh:
            continue
        nxt = dists[t + 1] if t + 1 < len(arr) else 0.0
        if nxt > thresh and t + 1 < len(arr):
            skip = 1.0 - float(np.minimum(hists[t - 1], hists[t + 1]).sum())
            if skip < thresh / 2.0:
                flashes.append(t)
                skip_next = True
                continue
        cuts.append(t)
    return {"cuts": cuts, "flashes": flashes}


def _bilinear_sample(img: np.ndarray, xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
    h, w = img.shape
    x0 = np.clip(np.floor(xs).astype(np.int64), 0, w - 1)
    y0 = np.clip(np.floor(ys).astype(np.int64), 0, h - 1)
    x1 = np.clip(x0 + 1, 0, w - 1)
    y1 = np.clip(y0 + 1, 0, h - 1)
    fx = np.clip(xs - np.floor(xs), 0.0, 1.0)
    fy = np.clip(ys - np.floor(ys), 0.0, 1.0)
    return (
        img[y0, x0] * (1 - fx) * (1 - fy)
        + img[y0, x1] * fx * (1 - fy)
        + img[y1, x0] * (1 - fx) * fy
        + img[y1, x1] * fx * fy
    )


def interpolate_frames(prev: Any, next: Any, block: int = 16, search: int = 7) -> _Tensor:
    """Motion-compensated mid-frame (t=0.5) via block matching (SAD).

    Motion is estimated on luma and applied per channel with bilinear
    half-pel warps from both inputs. Images are edge-padded to whole blocks.
    """
    a = np.asarray(_to_np(prev), dtype=np.float64)
    b = np.asarray(_to_np(next), dtype=np.float64)
    if a.shape != b.shape:
        raise ValueError(f"frame shapes disagree: {a.shape} vs {b.shape}")
    if a.ndim not in (2, 3):
        raise ValueError("frames must be [H, W] or [H, W, C]")
    if not np.all(np.isfinite(a)) or not np.all(np.isfinite(b)):
        raise ValueError("frames must be finite")
    if int(block) < 1 or int(search) < 0:
        raise ValueError("need block >= 1 and search >= 0")
    block, search = int(block), int(search)
    h, w = a.shape[:2]
    single = a.ndim == 2
    ac = a if not single else a[:, :, None]
    bc = b if not single else b[:, :, None]
    la = ac.mean(axis=2)
    lb = bc.mean(axis=2)
    ph = (block - h % block) % block
    pw = (block - w % block) % block
    pa = np.pad(la, ((0, ph), (0, pw)), mode="edge")
    pb = np.pad(lb, ((0, ph), (0, pw)), mode="edge")
    hh, ww = pa.shape
    out = np.zeros_like(np.pad(ac, ((0, ph), (0, pw), (0, 0)), mode="edge"))
    ys, xs = np.mgrid[0:block, 0:block]
    for by in range(0, hh, block):
        for bx in range(0, ww, block):
            ref = pa[by : by + block, bx : bx + block]
            best, mv = np.inf, (0, 0)
            for dy in range(-search, search + 1):
                for dx in range(-search, search + 1):
                    yy, xx = by + dy, bx + dx
                    if yy < 0 or xx < 0 or yy + block > hh or xx + block > ww:
                        continue
                    sad = float(np.abs(ref - pb[yy : yy + block, xx : xx + block]).sum())
                    if sad < best:
                        best, mv = sad, (dy, dx)
            wy = (ys + by - mv[0] / 2.0).ravel()
            wx = (xs + bx - mv[1] / 2.0).ravel()
            zy = (ys + by + mv[0] / 2.0).ravel()
            zx = (xs + bx + mv[1] / 2.0).ravel()
            for c in range(ac.shape[2]):
                wa = _bilinear_sample(np.pad(ac[:, :, c], ((0, ph), (0, pw)), mode="edge"), wx, wy)
                wb = _bilinear_sample(np.pad(bc[:, :, c], ((0, ph), (0, pw)), mode="edge"), zx, zy)
                out[by : by + block, bx : bx + block, c] = ((wa + wb) / 2.0).reshape(block, block)
    out = out[:h, :w]
    if single:
        out = out[:, :, 0]
    return _from_numpy(np.ascontiguousarray(out.astype(np.float32)))


def estimate_translation(ref: Any, mov: Any) -> Tuple[float, float]:
    """Phase-correlation shift (dy, dx) with mov ~= roll(ref, (dy, dx)).

    Integer peak from FFT phase correlation, refined with a parabolic fit.
    """
    a = _as_gray(_to_np(ref), "ref")
    b = _as_gray(_to_np(mov), "mov")
    if a.shape != b.shape:
        raise ValueError(f"frame shapes disagree: {a.shape} vs {b.shape}")
    if min(a.shape) < 2:
        raise ValueError("frames must be at least 2x2")
    if not np.all(np.isfinite(a)) or not np.all(np.isfinite(b)):
        raise ValueError("frames must be finite")
    f1 = np.fft.fft2(a - a.mean())
    f2 = np.fft.fft2(b - b.mean())
    cross = f1 * np.conj(f2)
    cross = cross / np.maximum(np.abs(cross), 1e-12)
    resp = np.fft.ifft2(cross).real
    h, w = resp.shape
    iy, ix = np.unravel_index(int(np.argmax(resp)), resp.shape)
    # ifft peak sits at -s for mov = roll(ref, s); negate to report s.
    dy = -float(iy if iy <= h // 2 else iy - h)
    dx = -float(ix if ix <= w // 2 else ix - w)
    # Parabolic subpixel refinement (wrapped neighbors).
    cy = resp[iy, ix]
    for axis, idx, size, acc in (("y", iy, h, "dy"), ("x", ix, w, "dx")):
        l = resp[(idx - 1) % size, ix] if axis == "y" else resp[iy, (idx - 1) % size]
        r = resp[(idx + 1) % size, ix] if axis == "y" else resp[iy, (idx + 1) % size]
        denom = l - 2 * cy + r
        off = 0.5 * (l - r) / denom if abs(denom) > 1e-12 else 0.0
        off = float(np.clip(off, -0.5, 0.5))
        if acc == "dy":
            dy -= off  # peak-relative offset shares the negation
        else:
            dx -= off
    return dy, dx


def stabilize_frames(frames: Sequence[Any], radius: int = 5) -> Tuple[_Tensor, _Tensor]:
    """Translation-only stabilization: smooths the camera path, integer warps.

    Args:
        frames: [T] frames of shared [H, W] or [H, W, C] shape.
        radius: Moving-average half-window over the motion path.

    Returns:
        (stabilized [T, H', W'(, C)] float32, corrections [T, 2] float (dy, dx)).
        Borders cropped by the maximum applied shift (no wrap artifacts).
    """
    if int(radius) < 0:
        raise ValueError("radius must be >= 0")
    arr = [np.asarray(_to_np(f), dtype=np.float64) for f in frames]
    if not arr:
        raise ValueError("frames must be non-empty")
    if len({a.shape for a in arr}) != 1:
        raise ValueError("all frames must share one shape")
    if arr[0].ndim not in (2, 3):
        raise ValueError("frames must be [H, W] or [H, W, C]")
    if any(not np.all(np.isfinite(a)) for a in arr):
        raise ValueError("frames must be finite")
    t = len(arr)
    pos = np.zeros((t, 2))
    for i in range(1, t):
        dy, dx = estimate_translation(arr[i - 1], arr[i])
        pos[i] = pos[i - 1] + [dy, dx]
    radius = int(radius)
    smooth = np.stack([
        pos[max(0, i - radius) : min(t, i + radius + 1)].mean(axis=0) for i in range(t)
    ])
    corr = smooth - pos
    applied = np.round(corr).astype(np.int64)
    warped = [np.roll(a, shift=(int(applied[i, 0]), int(applied[i, 1])), axis=(0, 1)) for i, a in enumerate(arr)]
    cy = int(np.abs(applied[:, 0]).max()) if t else 0
    cx = int(np.abs(applied[:, 1]).max()) if t else 0
    h, w = arr[0].shape[:2]
    if 2 * cy >= h or 2 * cx >= w:
        raise ValueError("stabilization crop exceeds frame size")
    crop = [fr[cy : h - cy if cy else h, cx : w - cx if cx else w] for fr in warped]
    return (
        _from_numpy(np.ascontiguousarray(np.stack(crop).astype(np.float32))),
        _from_numpy(np.ascontiguousarray(corr.astype(np.float32))),
    )


def lucas_kanade_optical_flow(
    frame1: Any, frame2: Any, window_size: int = 5
) -> Tuple[_Tensor, _Tensor]:
    """Calculates dense Lucas-Kanade optical flow motion field (u, v)."""
    f1 = _as_gray(_to_np(frame1), "frame1").astype(np.float32)
    f2 = _as_gray(_to_np(frame2), "frame2").astype(np.float32)
    if f1.shape != f2.shape:
        raise ValueError(f"Frame shapes mismatch: {f1.shape} vs {f2.shape}")

    h, w = f1.shape
    Ix = (np.roll(f1, -1, axis=1) - np.roll(f1, 1, axis=1)) * 0.5
    Iy = (np.roll(f1, -1, axis=0) - np.roll(f1, 1, axis=0)) * 0.5
    It = f2 - f1

    pad = window_size // 2

    def box(x):
        xp = np.pad(x, pad, mode="edge")
        out = np.zeros_like(x)
        for dy in range(window_size):
            for dx in range(window_size):
                out += xp[dy : dy + h, dx : dx + w]
        return out

    A11 = box(Ix * Ix)
    A12 = box(Ix * Iy)
    A22 = box(Iy * Iy)
    b1 = -box(Ix * It)
    b2 = -box(Iy * It)

    det = A11 * A22 - A12 * A12 + 1e-6
    u = (A22 * b1 - A12 * b2) / det
    v = (A11 * b2 - A12 * b1) / det

    return _from_numpy(np.ascontiguousarray(u, dtype=np.float32)), _from_numpy(
        np.ascontiguousarray(v, dtype=np.float32)
    )


def temporal_filter(
    video: Any, kernel: Sequence[float] = (0.25, 0.5, 0.25)
) -> _Tensor:
    """Applies a 1D temporal smoothing filter across the time axis (axis 0) of a video sequence."""
    arr = video.to_numpy() if isinstance(video, _Tensor) else np.asarray(video, dtype=np.float32)
    k = np.asarray(kernel, dtype=np.float32)
    k = k / np.sum(k)
    t = arr.shape[0]
    k_len = len(k)
    pad = k_len // 2

    pad_width = [(pad, pad)] + [(0, 0)] * (arr.ndim - 1)
    padded = np.pad(arr, pad_width, mode="edge")

    out = np.zeros_like(arr)
    for i in range(k_len):
        out += k[i] * padded[i : i + t]

    return _from_numpy(np.ascontiguousarray(out, dtype=np.float32))


__all__ = [
    "MOGBackgroundSubtractor",
    "detect_shot_boundaries",
    "interpolate_frames",
    "estimate_translation",
    "stabilize_frames",
    "lucas_kanade_optical_flow",
    "temporal_filter",
]
