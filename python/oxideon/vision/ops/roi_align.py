"""Region of Interest (RoI) pooling and bilinear alignment operators for Oxideon Vision."""

from typing import Any, Tuple, Union
import numpy as np
import operator

from ...oxideon import Tensor as _Tensor, from_numpy as _from_numpy


def _to_np(t: Any) -> np.ndarray:
    if isinstance(t, _Tensor):
        return t.to_numpy()
    return np.asarray(t, dtype=np.float32)


def _check_roi(
    feat: np.ndarray,
    b_np: np.ndarray,
    output_size: Union[int, Tuple[int, int]],
    spatial_scale: float,
) -> Tuple[int, int]:
    if feat.ndim != 4:
        raise ValueError(f"input must be [N, C, H, W], got shape {feat.shape}")
    if b_np.ndim != 2 or b_np.shape[1] not in (4, 5):
        raise ValueError(f"boxes must be [K, 4] or [K, 5], got shape {b_np.shape}")
    if len(b_np) and not np.all(np.isfinite(b_np)):
        raise ValueError("boxes must be finite")
    if isinstance(output_size, (tuple, list)):
        dims = tuple(output_size)
    else:
        dims = (output_size, output_size)
    if len(dims) != 2:
        raise ValueError(f"output_size must be an int or (h, w) pair, got {output_size!r}")
    try:
        out_h, out_w = (operator.index(v) for v in dims)
    except TypeError:
        raise ValueError(f"output_size dimensions must be ints, got {output_size!r}") from None
    if out_h < 1 or out_w < 1:
        raise ValueError(f"output_size dimensions must be positive, got {(out_h, out_w)}")
    if not isinstance(spatial_scale, (int, float, np.integer, np.floating)) or not np.isfinite(
        spatial_scale
    ):
        raise ValueError(f"spatial_scale must be a finite number, got {spatial_scale!r}")
    return out_h, out_w


def _check_batch_index(b_i: int, n: int, roi_idx: int) -> None:
    if not 0 <= b_i < n:
        raise ValueError(f"box {roi_idx} batch_index {b_i} out of range for batch size {n}")


def roi_align(
    input: _Tensor,
    boxes: _Tensor,
    output_size: Union[int, Tuple[int, int]],
    spatial_scale: float = 1.0,
    sampling_ratio: int = -1,
    aligned: bool = True,
) -> _Tensor:
    """Performs Region of Interest (RoI) Align operator with bilinear interpolation.

    Args:
        input: Feature map tensor of shape [N, C, H, W].
        boxes: Tensor of shape [K, 4] or [K, 5] (batch_index, x1, y1, x2, y2).
        output_size: Target (height, width) or int for square output.
        spatial_scale: Multiplicative scale mapping box coordinates to feature map resolution.
        sampling_ratio: Sampling points per bin (-1 defaults to ceil(bin_size)).
        aligned: Pixel shift alignment (0.5 pixel coordinate shift if True).
    """
    feat = _to_np(input).astype(np.float32)
    b_np = _to_np(boxes).astype(np.float32)
    out_h, out_w = _check_roi(feat, b_np, output_size, spatial_scale)

    if isinstance(input, _Tensor) and hasattr(input, "roi_align"):
        if b_np.shape[1] == 4:
            zeros = np.zeros((b_np.shape[0], 1), dtype=b_np.dtype)
            b5_np = np.concatenate([zeros, b_np], axis=1)
        else:
            b5_np = b_np.copy()
        if aligned and b5_np.shape[0] > 0:
            b5_np[:, 1:5] = b5_np[:, 1:5] - 0.5 / float(spatial_scale)
        rois_t = _from_numpy(np.ascontiguousarray(b5_np, dtype=np.float64)).to_dtype(input.dtype).to(input.device)
        sr = int(sampling_ratio) if sampling_ratio > 0 else 2
        return input.roi_align(rois_t, [out_h, out_w], float(spatial_scale), sr)

    n, c, h, w = feat.shape
    num_rois = b_np.shape[0]
    out = np.zeros((num_rois, c, out_h, out_w), dtype=np.float32)

    offset = 0.5 if aligned else 0.0

    for roi_idx in range(num_rois):
        box = b_np[roi_idx]
        if box.shape[0] == 5:
            b_i = int(box[0])
            x1, y1, x2, y2 = box[1:5]
        else:
            b_i = 0
            x1, y1, x2, y2 = box[:4]
        _check_batch_index(b_i, n, roi_idx)

        roi_start_w = x1 * spatial_scale - offset
        roi_start_h = y1 * spatial_scale - offset
        roi_end_w = x2 * spatial_scale - offset
        roi_end_h = y2 * spatial_scale - offset

        roi_width = max(roi_end_w - roi_start_w, 1.0 if not aligned else 0.0)
        roi_height = max(roi_end_h - roi_start_h, 1.0 if not aligned else 0.0)

        bin_size_h = roi_height / out_h
        bin_size_w = roi_width / out_w

        grid_h = sampling_ratio if sampling_ratio > 0 else int(np.ceil(bin_size_h))
        grid_w = sampling_ratio if sampling_ratio > 0 else int(np.ceil(bin_size_w))
        grid_h = max(grid_h, 1)
        grid_w = max(grid_w, 1)

        count = max(grid_h * grid_w, 1)

        # Bilinear sampling for each output bin
        for ph in range(out_h):
            for pw in range(out_w):
                val = np.zeros(c, dtype=np.float32)
                for iy in range(grid_h):
                    yy = roi_start_h + ph * bin_size_h + (iy + 0.5) * bin_size_h / grid_h
                    if yy < -1.0 or yy > h:
                        continue
                    yy = min(max(yy, 0.0), h - 1.0001)
                    y_low = int(yy)
                    y_high = min(y_low + 1, h - 1)
                    ly = yy - y_low
                    hy = 1.0 - ly

                    for ix in range(grid_w):
                        xx = roi_start_w + pw * bin_size_w + (ix + 0.5) * bin_size_w / grid_w
                        if xx < -1.0 or xx > w:
                            continue
                        xx = min(max(xx, 0.0), w - 1.0001)
                        x_low = int(xx)
                        x_high = min(x_low + 1, w - 1)
                        lx = xx - x_low
                        hx = 1.0 - lx

                        w1 = hy * hx
                        w2 = hy * lx
                        w3 = ly * hx
                        w4 = ly * lx

                        sample = (
                            w1 * feat[b_i, :, y_low, x_low]
                            + w2 * feat[b_i, :, y_low, x_high]
                            + w3 * feat[b_i, :, y_high, x_low]
                            + w4 * feat[b_i, :, y_high, x_high]
                        )
                        val += sample

                out[roi_idx, :, ph, pw] = val / count

    return _from_numpy(np.ascontiguousarray(out, dtype=np.float32))


def roi_pool(
    input: _Tensor,
    boxes: _Tensor,
    output_size: Union[int, Tuple[int, int]],
    spatial_scale: float = 1.0,
) -> _Tensor:
    """Performs standard max RoI Pooling operator."""
    feat = _to_np(input).astype(np.float32)
    b_np = _to_np(boxes).astype(np.float32)
    out_h, out_w = _check_roi(feat, b_np, output_size, spatial_scale)

    n, c, h, w = feat.shape
    num_rois = b_np.shape[0]
    out = np.zeros((num_rois, c, out_h, out_w), dtype=np.float32)

    for roi_idx in range(num_rois):
        box = b_np[roi_idx]
        if box.shape[0] == 5:
            b_i = int(box[0])
            x1, y1, x2, y2 = box[1:5]
        else:
            b_i = 0
            x1, y1, x2, y2 = box[:4]
        _check_batch_index(b_i, n, roi_idx)

        roi_start_w = int(np.round(x1 * spatial_scale))
        roi_start_h = int(np.round(y1 * spatial_scale))
        roi_end_w = int(np.round(x2 * spatial_scale))
        roi_end_h = int(np.round(y2 * spatial_scale))

        roi_width = max(roi_end_w - roi_start_w + 1, 1)
        roi_height = max(roi_end_h - roi_start_h + 1, 1)

        bin_size_h = roi_height / float(out_h)
        bin_size_w = roi_width / float(out_w)

        for ph in range(out_h):
            for pw in range(out_w):
                hstart = int(np.floor(ph * bin_size_h)) + roi_start_h
                wstart = int(np.floor(pw * bin_size_w)) + roi_start_w
                hend = int(np.ceil((ph + 1) * bin_size_h)) + roi_start_h
                wend = int(np.ceil((pw + 1) * bin_size_w)) + roi_start_w

                hstart = min(max(hstart, 0), h)
                hend = min(max(hend, 0), h)
                wstart = min(max(wstart, 0), w)
                wend = min(max(wend, 0), w)

                if hend > hstart and wend > wstart:
                    patch = feat[b_i, :, hstart:hend, wstart:wend]
                    out[roi_idx, :, ph, pw] = np.max(patch, axis=(1, 2))
                else:
                    out[roi_idx, :, ph, pw] = 0.0

    return _from_numpy(np.ascontiguousarray(out, dtype=np.float32))


__all__ = [
    "roi_align",
    "roi_pool",
]
