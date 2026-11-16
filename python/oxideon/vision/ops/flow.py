"""Pyramidal Lucas-Kanade optical flow (Bouguet) for Oxideon Vision."""

from typing import Any, List, Tuple
import numpy as np

from ...oxideon import Tensor as _Tensor, from_numpy as _from_numpy


def _to_np(t: Any) -> np.ndarray:
    if isinstance(t, _Tensor):
        return t.to_numpy()
    return np.asarray(t, dtype=np.float32)


def _downsample(img: np.ndarray) -> np.ndarray:
    h, w = img.shape
    h2, w2 = max(h // 2, 1), max(w // 2, 1)
    return img[: h2 * 2, : w2 * 2].reshape(h2, 2, w2, 2).mean(axis=(1, 3))


def _gradients(img: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    gx = np.zeros_like(img)
    gy = np.zeros_like(img)
    gx[:, 1:-1] = (img[:, 2:] - img[:, :-2]) / 2.0
    gy[1:-1, :] = (img[2:, :] - img[:-2, :]) / 2.0
    return gx, gy


def _bilinear(img: np.ndarray, xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
    h, w = img.shape
    x0 = np.floor(xs).astype(np.int64)
    y0 = np.floor(ys).astype(np.int64)
    x1 = np.clip(x0 + 1, 0, w - 1)
    y1 = np.clip(y0 + 1, 0, h - 1)
    x0 = np.clip(x0, 0, w - 1)
    y0 = np.clip(y0, 0, h - 1)
    fx = np.clip(xs - np.floor(xs), 0.0, 1.0)
    fy = np.clip(ys - np.floor(ys), 0.0, 1.0)
    return (
        img[y0, x0] * (1 - fx) * (1 - fy)
        + img[y0, x1] * fx * (1 - fy)
        + img[y1, x0] * (1 - fx) * fy
        + img[y1, x1] * fx * fy
    )


def lucas_kanade(
    prev: Any,
    next: Any,
    points: Any,
    window: int = 15,
    levels: int = 3,
    iterations: int = 10,
) -> Tuple[_Tensor, _Tensor]:
    """Tracks points from `prev` to `next` with pyramidal Lucas-Kanade flow.

    Args:
        prev / next: Grayscale images, 2-D arrays/Tensors of identical shape.
        points: [N, 2] (x, y) coordinates in `prev`.
        window: Odd integration window side (>= 3).
        levels: Pyramid depth (>= 1); level 0 is full resolution.
        iterations: Gauss-Newton iterations per level per point.

    Returns:
        (new_points [N, 2] float32, status [N] int64) where status is 1 when
        the point stayed in bounds with a solvable normal equation, else 0
        (position left at its last valid estimate).
    """
    a = np.asarray(_to_np(prev), dtype=np.float64)
    b = np.asarray(_to_np(next), dtype=np.float64)
    p = np.asarray(_to_np(points), dtype=np.float64)
    if a.ndim != 2 or b.ndim != 2:
        raise ValueError("images must be 2-D grayscale")
    if a.shape != b.shape:
        raise ValueError(f"image shapes disagree: {a.shape} vs {b.shape}")
    if min(a.shape) < 4:
        raise ValueError("images must be at least 4x4")
    if not np.all(np.isfinite(a)) or not np.all(np.isfinite(b)):
        raise ValueError("images must be finite")
    if p.ndim != 2 or p.shape[1] != 2:
        raise ValueError(f"points must be [N, 2], got shape {p.shape}")
    if len(p) == 0:
        raise ValueError("points must be non-empty")
    if not np.all(np.isfinite(p)):
        raise ValueError("points must be finite")
    if int(window) < 3 or int(window) % 2 == 0:
        raise ValueError("window must be an odd int >= 3")
    if int(levels) < 1 or int(iterations) < 1:
        raise ValueError("levels and iterations must be >= 1")
    window, levels, iterations = int(window), int(levels), int(iterations)

    pyr_a: List[np.ndarray] = [a]
    pyr_b: List[np.ndarray] = [b]
    for _ in range(1, levels):
        if min(pyr_a[-1].shape) < 4:
            break
        pyr_a.append(_downsample(pyr_a[-1]))
        pyr_b.append(_downsample(pyr_b[-1]))
    levels = len(pyr_a)
    grads = [_gradients(img) for img in pyr_a]

    half = window // 2
    off = np.arange(-half, half + 1, dtype=np.float64)
    oy, ox = np.meshgrid(off, off, indexing="ij")
    oxf, oyf = ox.ravel(), oy.ravel()

    flow = np.zeros_like(p)
    status = np.ones(len(p), dtype=np.int64)
    for li in range(levels - 1, -1, -1):
        scale = 2.0**li
        ia, ib = pyr_a[li], pyr_b[li]
        ga, gb = grads[li]
        h, w = ia.shape
        # NOTE: `flow` is kept in full-resolution pixels, so no rescaling
        # between levels is needed (level guess = flow / scale below).
        for i in range(len(p)):
            if status[i] == 0:
                continue
            px = p[i] / scale
            gx = px[0] + flow[i, 0] / scale
            gy = px[1] + flow[i, 1] / scale
            sx = px[0] + oxf  # window over prev: fixed across iterations
            sy = px[1] + oyf
            if sx.min() < 0 or sy.min() < 0 or sx.max() > w - 1 or sy.max() > h - 1:
                status[i] = 0
                continue
            ix = _bilinear(ga, sx, sy)
            iy = _bilinear(gb, sx, sy)
            ia_win = _bilinear(ia, sx, sy)
            ok = True
            for _ in range(iterations):
                wx = gx + oxf
                wy = gy + oyf
                if wx.min() < 0 or wy.min() < 0 or wx.max() > w - 1 or wy.max() > h - 1:
                    ok = False
                    break
                it = _bilinear(ib, wx, wy) - ia_win
                gxx, gxy, gyy = np.dot(ix, ix), np.dot(ix, iy), np.dot(iy, iy)
                det = gxx * gyy - gxy * gxy
                if not np.isfinite(det) or det <= 1e-12:
                    ok = False
                    break
                bx = np.dot(ix, it)
                by = np.dot(iy, it)
                dx = (gyy * bx - gxy * by) / det
                dy = (gxx * by - gxy * bx) / det
                if not np.all(np.isfinite((dx, dy))):
                    ok = False
                    break
                gx -= dx
                gy -= dy
                if dx * dx + dy * dy < 1e-10:
                    break
            if not ok:
                status[i] = 0
                continue
            flow[i, 0] = (gx - px[0]) * scale
            flow[i, 1] = (gy - px[1]) * scale
    new_points = p + flow
    return (
        _from_numpy(np.ascontiguousarray(new_points.astype(np.float32))),
        _from_numpy(np.ascontiguousarray(status)),
    )
