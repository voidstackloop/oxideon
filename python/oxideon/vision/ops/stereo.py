"""Dense stereo block matching (SAD, winner-take-all) for Oxideon Vision."""

from typing import Any
import numpy as np

from ...oxideon import Tensor as _Tensor, from_numpy as _from_numpy


def _to_np(t: Any) -> np.ndarray:
    if isinstance(t, _Tensor):
        return t.to_numpy()
    return np.asarray(t, dtype=np.float32)


def block_match_disparity(
    left: Any,
    right: Any,
    max_disparity: int = 64,
    window: int = 9,
) -> _Tensor:
    """Rectified stereo disparity via SAD block matching (reference impl).

    For each pixel, searches d in [0, max_disparity) minimizing the sum of
    absolute differences over a `window` x `window` patch between
    ``left[y, x]`` and ``right[y, x - d]``. Ties pick the smallest d.

    Args:
        left / right: Grayscale images, 2-D arrays/Tensors of identical shape.
        max_disparity: Number of disparity hypotheses (>= 1, < width).
        window: Odd aggregation window side (>= 1).

    Returns:
        int32 Tensor [H, W] of disparities; pixels whose window or full
        disparity range would leave the image are marked -1 (invalid). If the
        image is too narrow for the range, every pixel is -1.
    """
    a = np.asarray(_to_np(left), dtype=np.float64)
    b = np.asarray(_to_np(right), dtype=np.float64)
    if a.ndim != 2 or b.ndim != 2:
        raise ValueError("images must be 2-D grayscale")
    if a.shape != b.shape:
        raise ValueError(f"image shapes disagree: {a.shape} vs {b.shape}")
    if not np.all(np.isfinite(a)) or not np.all(np.isfinite(b)):
        raise ValueError("images must be finite")
    h, w = a.shape
    if int(window) < 1 or int(window) % 2 == 0:
        raise ValueError("window must be an odd int >= 1")
    if int(max_disparity) < 1 or int(max_disparity) >= w:
        raise ValueError(f"max_disparity must satisfy 1 <= D < width ({w})")
    window, max_disparity = int(window), int(max_disparity)
    half = window // 2

    out = np.full((h, w), -1, dtype=np.int32)
    rows = slice(half, h - half if half else h)
    cols = slice(half + max_disparity - 1, w - half if half else w)
    if rows.stop is not None and rows.stop <= rows.start:
        return _from_numpy(np.ascontiguousarray(out))
    if cols.stop is not None and cols.stop <= cols.start:
        return _from_numpy(np.ascontiguousarray(out))
    sad_vol = np.empty((max_disparity, h, w), dtype=np.float64)
    for d in range(max_disparity):
        rb = np.zeros_like(a)
        rb[:, d:] = b[:, : w - d]
        sad_vol[d] = _box_filter(np.abs(a - rb), half)
    # Zero-fill above cannot leak into the valid region: every window tap
    # around a valid pixel reads right columns >= 0 (x - half - d >= 0).
    region = sad_vol[:, rows, cols]
    out[rows, cols] = np.argmin(region, axis=0).astype(np.int32)
    return _from_numpy(np.ascontiguousarray(out))


def _box_filter(img: np.ndarray, half: int) -> np.ndarray:
    if half == 0:
        return img.copy()
    padded = np.pad(img, half, mode="constant")
    cs = np.zeros((padded.shape[0] + 1, padded.shape[1] + 1), dtype=np.float64)
    cs[1:, 1:] = np.cumsum(np.cumsum(padded, axis=0), axis=1)
    k = 2 * half + 1
    return cs[k:, k:] - cs[:-k, k:] - cs[k:, :-k] + cs[:-k, :-k]
