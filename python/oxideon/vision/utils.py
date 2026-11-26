"""Visualization and image-grid utilities for Oxideon (`oxideon.vision.utils`).

Provides zero-dependency implementations of:
- `image_grid`: Tiles `[B, C, H, W]` image batches into a `[C, H_grid, W_grid]` tensor
- `save_image`: Saves a tensor or batch grid directly to PNG (path or binary file object)
- `draw_bounding_boxes`: Renders `[N, 4]` `(xmin, ymin, xmax, ymax)` boxes and text labels
- `draw_segmentation_masks`: Alpha-blends `[M, H, W]` boolean masks onto `uint8` images
- `draw_keypoints`: Renders `[N, K, 2|3]` keypoints and skeleton `connectivity` edges
"""

import math
import os
from pathlib import Path
import struct
from typing import Any, BinaryIO, Dict, List, Optional, Sequence, Tuple, Union
import zlib

import numpy as np

from ..oxideon import Tensor as _Tensor, from_numpy as _from_numpy, stack as _stack


_DEFAULT_PALETTE: List[Tuple[int, int, int]] = [
    (255, 56, 56),
    (44, 153, 255),
    (50, 205, 50),
    (255, 180, 0),
    (186, 85, 211),
    (0, 206, 209),
    (255, 105, 180),
    (255, 140, 0),
    (124, 252, 0),
    (100, 149, 237),
]

_NAMED_COLORS: Dict[str, Tuple[int, int, int]] = {
    "red": (255, 0, 0),
    "green": (0, 200, 0),
    "blue": (0, 0, 255),
    "yellow": (255, 255, 0),
    "cyan": (0, 255, 255),
    "magenta": (255, 0, 255),
    "white": (255, 255, 255),
    "black": (0, 0, 0),
    "orange": (255, 165, 0),
    "purple": (128, 0, 128),
}

_GLYPHS_3X5 = {
    "0": [0b111, 0b101, 0b101, 0b101, 0b111],
    "1": [0b010, 0b110, 0b010, 0b010, 0b111],
    "2": [0b111, 0b001, 0b111, 0b100, 0b111],
    "3": [0b111, 0b001, 0b111, 0b001, 0b111],
    "4": [0b101, 0b101, 0b111, 0b001, 0b001],
    "5": [0b111, 0b100, 0b111, 0b001, 0b111],
    "6": [0b111, 0b100, 0b111, 0b101, 0b111],
    "7": [0b111, 0b001, 0b010, 0b010, 0b010],
    "8": [0b111, 0b101, 0b111, 0b101, 0b111],
    "9": [0b111, 0b101, 0b111, 0b001, 0b111],
    "A": [0b010, 0b101, 0b111, 0b101, 0b101],
    "B": [0b110, 0b101, 0b110, 0b101, 0b110],
    "C": [0b111, 0b100, 0b100, 0b100, 0b111],
    "D": [0b110, 0b101, 0b101, 0b101, 0b110],
    "E": [0b111, 0b100, 0b110, 0b100, 0b111],
    "F": [0b111, 0b100, 0b110, 0b100, 0b100],
    "G": [0b111, 0b100, 0b101, 0b101, 0b111],
    "H": [0b101, 0b101, 0b111, 0b101, 0b101],
    "I": [0b111, 0b010, 0b010, 0b010, 0b111],
    "J": [0b001, 0b001, 0b001, 0b101, 0b111],
    "K": [0b101, 0b110, 0b100, 0b110, 0b101],
    "L": [0b100, 0b100, 0b100, 0b100, 0b111],
    "M": [0b101, 0b111, 0b101, 0b101, 0b101],
    "N": [0b101, 0b111, 0b111, 0b101, 0b101],
    "O": [0b111, 0b101, 0b101, 0b101, 0b111],
    "P": [0b111, 0b101, 0b111, 0b100, 0b100],
    "Q": [0b111, 0b101, 0b101, 0b111, 0b001],
    "R": [0b110, 0b101, 0b110, 0b101, 0b101],
    "S": [0b111, 0b100, 0b111, 0b001, 0b111],
    "T": [0b111, 0b010, 0b010, 0b010, 0b010],
    "U": [0b101, 0b101, 0b101, 0b101, 0b111],
    "V": [0b101, 0b101, 0b101, 0b101, 0b010],
    "W": [0b101, 0b101, 0b101, 0b111, 0b101],
    "X": [0b101, 0b101, 0b010, 0b101, 0b101],
    "Y": [0b101, 0b101, 0b010, 0b010, 0b010],
    "Z": [0b111, 0b001, 0b010, 0b100, 0b111],
    ".": [0b000, 0b000, 0b000, 0b000, 0b010],
    ":": [0b000, 0b010, 0b000, 0b010, 0b000],
    "-": [0b000, 0b000, 0b111, 0b000, 0b000],
    "_": [0b000, 0b000, 0b000, 0b000, 0b111],
    "%": [0b101, 0b001, 0b010, 0b100, 0b101],
    " ": [0b000, 0b000, 0b000, 0b000, 0b000],
}


def _parse_color(c: Any) -> Tuple[int, int, int]:
    if isinstance(c, str):
        s = c.strip().lower()
        if s in _NAMED_COLORS:
            return _NAMED_COLORS[s]
        if s.startswith("#") and len(s) == 7:
            return (int(s[1:3], 16), int(s[3:5], 16), int(s[5:7], 16))
        raise ValueError(f"Unknown color string {c!r}")
    if isinstance(c, (tuple, list)) and len(c) == 3:
        return (int(c[0]), int(c[1]), int(c[2]))
    raise TypeError(f"Color must be a name, '#RRGGBB', or 3-tuple of ints; got {c!r}")


def _resolve_colors(colors: Any, count: int) -> List[Tuple[int, int, int]]:
    if colors is None:
        return [_DEFAULT_PALETTE[i % len(_DEFAULT_PALETTE)] for i in range(count)]
    if isinstance(colors, str) or (isinstance(colors, tuple) and len(colors) == 3 and isinstance(colors[0], int)):
        rgb = _parse_color(colors)
        return [rgb] * count
    color_list = [_parse_color(c) for c in colors]
    if len(color_list) < count:
        raise ValueError(f"Expected at least {count} colors, got {len(color_list)}")
    return color_list[:count]


def _encode_png_bytes(hwc: np.ndarray) -> bytes:
    """Encodes an `uint8` `[H, W, C]` array (`C` in `1..=4`) into valid compressed PNG bytes."""
    if hwc.ndim != 3 or hwc.dtype != np.uint8:
        raise ValueError(f"_encode_png_bytes expected uint8 [H, W, C] array, got {hwc.dtype} {hwc.shape}")
    h, w, c = hwc.shape
    color_type_map = {1: 0, 2: 4, 3: 2, 4: 6}
    if c not in color_type_map:
        raise ValueError(f"PNG supports 1, 2, 3, or 4 channels; got {c}")
    if h == 0 or w == 0:
        raise ValueError(f"PNG dimensions must be positive, got {w}x{h}")

    scanlines = np.concatenate(
        [np.zeros((h, 1), dtype=np.uint8), hwc.reshape(h, w * c)],
        axis=1,
    )
    compressed = zlib.compress(scanlines.tobytes(), level=6)

    def _chunk(tag: bytes, payload: bytes) -> bytes:
        crc = zlib.crc32(tag + payload) & 0xFFFFFFFF
        return struct.pack(">I", len(payload)) + tag + payload + struct.pack(">I", crc)

    ihdr = struct.pack(">IIBBBBB", w, h, 8, color_type_map[c], 0, 0, 0)
    return b"\x89PNG\r\n\x1a\n" + _chunk(b"IHDR", ihdr) + _chunk(b"IDAT", compressed) + _chunk(b"IEND", b"")


# ---------------------------------------------------------------------------
# `image_grid` & `save_image`
# ---------------------------------------------------------------------------

def image_grid(
    tensor: Union[_Tensor, List[_Tensor]],
    nrow: int = 8,
    padding: int = 2,
    normalize: bool = False,
    value_range: Optional[Tuple[float, float]] = None,
    scale_each: bool = False,
    pad_value: float = 0.0,
) -> _Tensor:
    """Arranges a batch of `[B, C, H, W]` (or list of `[C, H, W]`) images into a single `[C, H_grid, W_grid]` tensor."""
    if nrow < 1:
        raise ValueError(f"nrow must be >= 1, got {nrow}")
    if padding < 0:
        raise ValueError(f"padding must be >= 0, got {padding}")

    if isinstance(tensor, list):
        if not tensor:
            raise ValueError("image_grid received an empty list of tensors")
        batch = _stack(tensor, dim=0)
    elif isinstance(tensor, _Tensor):
        batch = tensor
    else:
        raise TypeError(f"tensor or list of tensors expected, got {type(tensor).__name__}")

    arr = batch.detach().to_numpy().astype(np.float32, copy=True)

    if arr.ndim == 2:
        arr = arr[None, None, :, :]
    elif arr.ndim == 3:
        if arr.shape[0] == 1:
            arr = np.repeat(arr, 3, axis=0)
        arr = arr[None, :, :, :]
    elif arr.ndim != 4:
        raise ValueError(f"image_grid expected 2D, 3D, or 4D tensor, got shape {arr.shape}")

    if arr.shape[1] == 1:
        arr = np.repeat(arr, 3, axis=1)

    if normalize:
        def _norm_ip(img: np.ndarray, low: float, high: float) -> None:
            np.clip(img, low, high, out=img)
            denom = max(high - low, 1e-5)
            img -= low
            img /= denom

        def _norm_range(t: np.ndarray, vrange: Optional[Tuple[float, float]]) -> None:
            if vrange is not None:
                _norm_ip(t, float(vrange[0]), float(vrange[1]))
            else:
                _norm_ip(t, float(t.min()), float(t.max()))

        if scale_each:
            for i in range(arr.shape[0]):
                _norm_range(arr[i], value_range)
        else:
            _norm_range(arr, value_range)

    nmaps, channels, h, w = arr.shape
    if nmaps == 1:
        return _from_numpy(np.ascontiguousarray(arr[0]))

    xmaps = min(nrow, nmaps)
    ymaps = int(math.ceil(float(nmaps) / xmaps))
    cell_h, cell_w = h + padding, w + padding
    grid = np.full(
        (channels, cell_h * ymaps + padding, cell_w * xmaps + padding),
        float(pad_value),
        dtype=np.float32,
    )
    k = 0
    for y in range(ymaps):
        for x in range(xmaps):
            if k >= nmaps:
                break
            y0 = y * cell_h + padding
            x0 = x * cell_w + padding
            grid[:, y0 : y0 + h, x0 : x0 + w] = arr[k]
            k += 1

    return _from_numpy(np.ascontiguousarray(grid))


def save_image(
    tensor: Union[_Tensor, List[_Tensor]],
    fp: Union[str, os.PathLike, BinaryIO],
    format: Optional[str] = None,
    **kwargs: Any,
) -> None:
    """Saves an image tensor or tiled batch grid directly into a PNG file or binary stream."""
    if format is not None and format.lower() != "png":
        raise ValueError(f"Oxideon zero-dependency save_image supports PNG format, got {format!r}")

    if isinstance(tensor, _Tensor) and tensor.dtype == "u8" and len(tensor.shape) == 3 and not kwargs:
        chw = tensor.detach().to_numpy()
    else:
        grid = image_grid(tensor, **kwargs)
        g_np = grid.detach().to_numpy()
        if isinstance(tensor, _Tensor) and tensor.dtype == "u8" and not kwargs.get("normalize", False):
            chw = np.clip(g_np, 0, 255).astype(np.uint8)
        else:
            chw = np.clip(g_np * 255.0 + 0.5, 0, 255).astype(np.uint8)

    hwc = np.ascontiguousarray(np.transpose(chw, (1, 2, 0)))
    png_bytes = _encode_png_bytes(hwc)

    if hasattr(fp, "write") and callable(fp.write):
        fp.write(png_bytes)  # type: ignore[union-attr]
    else:
        p = Path(fp).expanduser().resolve()  # type: ignore[arg-type]
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "wb") as f:
            f.write(png_bytes)


# ---------------------------------------------------------------------------
# Drawing Utilities (`draw_bounding_boxes`, `draw_segmentation_masks`, `draw_keypoints`)
# ---------------------------------------------------------------------------

def _validate_uint8_chw(image: _Tensor, fn_name: str) -> np.ndarray:
    if not isinstance(image, _Tensor):
        raise TypeError(f"{fn_name}: expected Tensor, got {type(image).__name__}")
    if image.dtype != "u8":
        raise ValueError(f"{fn_name}: expected uint8 Tensor, got {image.dtype}")
    if len(image.shape) != 3 or image.shape[0] not in (1, 3):
        raise ValueError(f"{fn_name}: expected [1, H, W] or [3, H, W] image, got {tuple(image.shape)}")
    arr = image.detach().to_numpy().copy()
    if arr.shape[0] == 1:
        arr = np.repeat(arr, 3, axis=0)
    return arr


def _draw_text_3x5(canvas_chw: np.ndarray, x: int, y: int, text: str, fg: Tuple[int, int, int], bg: Tuple[int, int, int]) -> None:
    _, h, w = canvas_chw.shape
    txt = text.upper()
    box_w = len(txt) * 4 + 1
    box_h = 7
    y0 = max(0, min(y, max(h - box_h, 0)))
    x0 = max(0, min(x, max(w - 1, 0)))
    y1 = min(h, y0 + box_h)
    x1 = min(w, x0 + box_w)
    for c_idx in range(3):
        canvas_chw[c_idx, y0:y1, x0:x1] = bg[c_idx]

    cx = x0 + 1
    cy = y0 + 1
    for ch in txt:
        rows = _GLYPHS_3X5.get(ch, _GLYPHS_3X5[" "])
        for ry, bits in enumerate(rows):
            py = cy + ry
            if py >= h:
                break
            for rx in range(3):
                px = cx + rx
                if px >= w:
                    break
                if (bits >> (2 - rx)) & 1:
                    canvas_chw[0, py, px] = fg[0]
                    canvas_chw[1, py, px] = fg[1]
                    canvas_chw[2, py, px] = fg[2]
        cx += 4


def draw_bounding_boxes(
    image: _Tensor,
    boxes: _Tensor,
    labels: Optional[Sequence[str]] = None,
    colors: Optional[Any] = None,
    fill: bool = False,
    width: int = 1,
) -> _Tensor:
    """Draws `(xmin, ymin, xmax, ymax)` bounding boxes and optional text labels on a `uint8` `[C, H, W]` image."""
    if width < 1:
        raise ValueError(f"width must be >= 1, got {width}")
    canvas = _validate_uint8_chw(image, "draw_bounding_boxes")
    _, h, w = canvas.shape

    b_np = boxes.detach().to_numpy() if isinstance(boxes, _Tensor) else np.asarray(boxes)
    if b_np.size == 0:
        return _from_numpy(np.ascontiguousarray(canvas))
    if b_np.ndim != 2 or b_np.shape[1] != 4:
        raise ValueError(f"boxes must have shape [N, 4], got {b_np.shape}")

    n = b_np.shape[0]
    if labels is not None and len(labels) != n:
        raise ValueError(f"Number of labels ({len(labels)}) must match number of boxes ({n})")
    rgb_list = _resolve_colors(colors, n)

    for i in range(n):
        x1 = int(np.clip(round(float(b_np[i, 0])), 0, max(w - 1, 0)))
        y1 = int(np.clip(round(float(b_np[i, 1])), 0, max(h - 1, 0)))
        x2 = int(np.clip(round(float(b_np[i, 2])), 0, max(w - 1, 0)))
        y2 = int(np.clip(round(float(b_np[i, 3])), 0, max(h - 1, 0)))
        if x2 < x1 or y2 < y1:
            raise ValueError(f"Invalid box (xmin <= xmax and ymin <= ymax required): {b_np[i].tolist()}")

        col = rgb_list[i]
        if fill:
            for c_idx in range(3):
                region = canvas[c_idx, y1 : y2 + 1, x1 : x2 + 1].astype(np.float32)
                canvas[c_idx, y1 : y2 + 1, x1 : x2 + 1] = np.clip(0.8 * region + 0.2 * col[c_idx], 0, 255).astype(np.uint8)

        for t in range(width):
            yt_top = min(y1 + t, h - 1)
            yt_bot = max(y2 - t, 0)
            xl = min(x1 + t, w - 1)
            xr = max(x2 - t, 0)
            for c_idx in range(3):
                canvas[c_idx, yt_top, x1 : x2 + 1] = col[c_idx]
                canvas[c_idx, yt_bot, x1 : x2 + 1] = col[c_idx]
                canvas[c_idx, y1 : y2 + 1, xl] = col[c_idx]
                canvas[c_idx, y1 : y2 + 1, xr] = col[c_idx]

        if labels is not None and labels[i]:
            lum = 0.299 * col[0] + 0.587 * col[1] + 0.114 * col[2]
            fg = (0, 0, 0) if lum > 140 else (255, 255, 255)
            _draw_text_3x5(canvas, x1, max(y1 - 7, 0), str(labels[i]), fg=fg, bg=col)

    return _from_numpy(np.ascontiguousarray(canvas))


def draw_segmentation_masks(
    image: _Tensor,
    masks: _Tensor,
    alpha: float = 0.8,
    colors: Optional[Any] = None,
) -> _Tensor:
    """Alpha-blends boolean segmentation masks onto a `uint8` `[C, H, W]` image."""
    if not (0.0 <= float(alpha) <= 1.0):
        raise ValueError(f"alpha must be in [0, 1], got {alpha}")
    canvas = _validate_uint8_chw(image, "draw_segmentation_masks").astype(np.float32)
    _, h, w = canvas.shape

    m_np = masks.detach().to_numpy() if isinstance(masks, _Tensor) else np.asarray(masks)
    if m_np.dtype != np.bool_:
        raise ValueError(f"masks must be a boolean tensor, got {m_np.dtype}")
    if m_np.ndim == 2:
        m_np = m_np[None, :, :]
    if m_np.ndim != 3 or m_np.shape[1:] != (h, w):
        raise ValueError(f"masks shape {m_np.shape} does not match image spatial dimensions {(h, w)}")

    num_masks = m_np.shape[0]
    rgb_list = _resolve_colors(colors, num_masks)

    for i in range(num_masks):
        mask = m_np[i]
        if not np.any(mask):
            continue
        col = rgb_list[i]
        for c_idx in range(3):
            canvas[c_idx, mask] = (1.0 - alpha) * canvas[c_idx, mask] + alpha * float(col[c_idx])

    return _from_numpy(np.ascontiguousarray(np.clip(canvas, 0, 255).round().astype(np.uint8)))


def _draw_line(canvas_chw: np.ndarray, x0: int, y0: int, x1: int, y1: int, col: Tuple[int, int, int], width: int) -> None:
    _, h, w = canvas_chw.shape
    steps = max(abs(x1 - x0), abs(y1 - y0), 1)
    xs = np.linspace(x0, x1, steps + 1)
    ys = np.linspace(y0, y1, steps + 1)
    half = max(width - 1, 0) // 2
    for xf, yf in zip(xs, ys):
        cx, cy = int(round(xf)), int(round(yf))
        y_lo, y_hi = max(cy - half, 0), min(cy + half + 1, h)
        x_lo, x_hi = max(cx - half, 0), min(cx + half + 1, w)
        for c_idx in range(3):
            canvas_chw[c_idx, y_lo:y_hi, x_lo:x_hi] = col[c_idx]


def draw_keypoints(
    image: _Tensor,
    keypoints: _Tensor,
    connectivity: Optional[Sequence[Tuple[int, int]]] = None,
    colors: Optional[Any] = None,
    radius: int = 2,
    width: int = 1,
) -> _Tensor:
    """Draws `[num_instances, K, 2]` or `[num_instances, K, 3]` keypoints and optional skeleton `connectivity`
    onto a `uint8` `[C, H, W]` image.
    """
    if radius < 0 or width < 1:
        raise ValueError("radius must be >= 0 and width must be >= 1")
    canvas = _validate_uint8_chw(image, "draw_keypoints")
    _, h, w = canvas.shape

    kp_np = keypoints.detach().to_numpy() if isinstance(keypoints, _Tensor) else np.asarray(keypoints)
    if kp_np.ndim != 3 or kp_np.shape[2] not in (2, 3):
        raise ValueError(f"keypoints must have shape [N, K, 2] or [N, K, 3], got {kp_np.shape}")

    pt_color = _parse_color(colors) if colors is not None else (255, 0, 0)
    has_vis = kp_np.shape[2] == 3

    yy, xx = np.ogrid[-radius : radius + 1, -radius : radius + 1]
    disk = (xx * xx + yy * yy) <= (radius * radius)

    for inst in range(kp_np.shape[0]):
        pts = kp_np[inst]
        if connectivity is not None:
            for j0, j1 in connectivity:
                if has_vis and (pts[j0, 2] <= 0 or pts[j1, 2] <= 0):
                    continue
                _draw_line(
                    canvas,
                    int(round(float(pts[j0, 0]))),
                    int(round(float(pts[j0, 1]))),
                    int(round(float(pts[j1, 0]))),
                    int(round(float(pts[j1, 1]))),
                    pt_color,
                    width=width,
                )

        for k in range(pts.shape[0]):
            if has_vis and pts[k, 2] <= 0:
                continue
            cx = int(round(float(pts[k, 0])))
            cy = int(round(float(pts[k, 1])))
            for dy in range(-radius, radius + 1):
                py = cy + dy
                if py < 0 or py >= h:
                    continue
                for dx in range(-radius, radius + 1):
                    px = cx + dx
                    if px < 0 or px >= w:
                        continue
                    if disk[dy + radius, dx + radius]:
                        canvas[0, py, px] = pt_color[0]
                        canvas[1, py, px] = pt_color[1]
                        canvas[2, py, px] = pt_color[2]

    return _from_numpy(np.ascontiguousarray(canvas))


__all__ = [
    "image_grid",
    "save_image",
    "draw_bounding_boxes",
    "draw_segmentation_masks",
    "draw_keypoints",
]
