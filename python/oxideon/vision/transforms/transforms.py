"""Image and tensor transforms for Oxideon Computer Vision."""

from typing import Any, Callable, List, Optional, Sequence, Tuple, Union
import math
import operator
import random
import numpy as np

from ...oxideon import Tensor as _Tensor, from_numpy as _from_numpy
from ... import functional as _F


def _as_tensor(img: Any) -> _Tensor:
    if isinstance(img, _Tensor):
        return img
    return _from_numpy(np.ascontiguousarray(img))


def _to_np(img: Any) -> np.ndarray:
    if isinstance(img, _Tensor):
        return img.to_numpy()
    if hasattr(img, "numpy"):
        return img.numpy()
    return np.asarray(img, dtype=np.float32)


class Compose:
    """Composes several transforms together in sequence."""

    def __init__(self, transforms: Sequence[Callable[[Any], Any]]):
        self.transforms = list(transforms)

    def __call__(self, img: Any) -> Any:
        for t in self.transforms:
            img = t(img)
        return img

    def __repr__(self) -> str:
        format_string = self.__class__.__name__ + "("
        for t in self.transforms:
            format_string += f"\n    {t}"
        format_string += "\n)"
        return format_string


class ToTensor:
    """Converts a PIL Image or numpy.ndarray (H x W x C) in [0, 255] to a Tensor (C x H x W) in [0.0, 1.0]."""

    def __call__(self, pic: Any) -> _Tensor:
        arr = _to_np(pic)
        if arr.ndim == 2:
            arr = arr[None, :, :]  # grayscale (H, W) -> (1, H, W), unambiguous
        elif arr.ndim == 3 and not (
            arr.shape[0] in (1, 3, 4) and arr.shape[2] not in (1, 3, 4)
        ):
            # 3-D input is HWC per the contract: transpose to CHW unless the
            # layout is unambiguously CHW already (leading channel dim only).
            arr = np.transpose(arr, (2, 0, 1))

        if arr.dtype == np.uint8 or np.max(arr) > 1.01:
            arr = arr.astype(np.float32) / 255.0
        else:
            arr = arr.astype(np.float32)

        return _from_numpy(np.ascontiguousarray(arr, dtype=np.float32))


class Normalize:
    """Normalizes a tensor image with mean and standard deviation: (x - mean) / std."""

    def __init__(self, mean: Sequence[float], std: Sequence[float]):
        mean_a = np.array(mean, dtype=np.float32).reshape(-1, 1, 1)
        std_a = np.array(std, dtype=np.float32).reshape(-1, 1, 1)
        if mean_a.size == 0 or not np.all(np.isfinite(mean_a)):
            raise ValueError("mean must be non-empty and finite")
        if std_a.size == 0 or not np.all(np.isfinite(std_a)) or np.any(std_a <= 0):
            raise ValueError("std must be non-empty, finite and positive")
        self.mean = mean_a
        self.std = std_a

    def __call__(self, tensor: _Tensor) -> _Tensor:
        t = _as_tensor(tensor)
        mean_t = _from_numpy(np.ascontiguousarray(self.mean)).to_dtype(t.dtype).to(t.device)
        std_t = _from_numpy(np.ascontiguousarray(self.std)).to_dtype(t.dtype).to(t.device)
        return (t - mean_t) / std_t


class Resize:
    """Resizes the input image/tensor to the given size (H, W) using bilinear interpolation."""

    def __init__(self, size: Union[int, Tuple[int, int]]):
        self.size = _check_hw_size(size, "Resize")

    def __call__(self, img: _Tensor) -> _Tensor:
        t = _as_tensor(img)
        if t.ndim not in (3, 4):
            raise ValueError(f"Resize expects [C, H, W] or [B, C, H, W], got shape {t.shape}")
        is_batched = t.ndim == 4
        t4 = t if is_batched else t.unsqueeze(0)
        out = _F.interpolate(t4, size=self.size, mode="bilinear", align_corners=True)
        return out if is_batched else out.squeeze(0)


def _check_hw_size(size: Union[int, Sequence[int]], name: str) -> Tuple[int, int]:
    if isinstance(size, (tuple, list)):
        raw = tuple(size)
        if len(raw) != 2:
            raise ValueError(f"{name} size must be an int or (h, w) pair, got {size!r}")
    else:
        raw = (size, size)
    try:
        dims = (operator.index(raw[0]), operator.index(raw[1]))
    except TypeError:
        raise ValueError(f"{name} size dimensions must be ints, got {size!r}") from None
    if dims[0] < 1 or dims[1] < 1:
        raise ValueError(f"{name} size dimensions must be positive, got {dims}")
    return dims


class CenterCrop:
    """Crops the center of the image to the given size."""

    def __init__(self, size: Union[int, Tuple[int, int]]):
        self.size = _check_hw_size(size, "CenterCrop")

    def __call__(self, img: _Tensor) -> _Tensor:
        t = _as_tensor(img)
        th, tw = self.size
        h, w = t.shape[-2], t.shape[-1]
        if th > h or tw > w:
            raise ValueError(f"crop size {(th, tw)} larger than input {(h, w)}")
        i = max(0, int(round((h - th) / 2.0)))
        j = max(0, int(round((w - tw) / 2.0)))
        return t.narrow(-2, i, th).narrow(-1, j, tw)


class RandomCrop:
    """Crops the given image at a random location."""

    def __init__(self, size: Union[int, Tuple[int, int]], padding: Optional[int] = None):
        self.size = _check_hw_size(size, "RandomCrop")
        self.padding = padding

    def __call__(self, img: _Tensor) -> _Tensor:
        t = _as_tensor(img)
        if self.padding is not None and self.padding > 0:
            p = int(self.padding)
            t = _F.pad(t, (p, p, p, p), mode="constant", value=0.0)

        th, tw = self.size
        h, w = t.shape[-2], t.shape[-1]
        if th > h or tw > w:
            raise ValueError(f"crop size {(th, tw)} larger than input {(h, w)}")
        i = random.randint(0, max(0, h - th))
        j = random.randint(0, max(0, w - tw))
        return t.narrow(-2, i, th).narrow(-1, j, tw)


class RandomHorizontalFlip:
    """Randomly flips the image horizontally with probability p."""

    def __init__(self, p: float = 0.5):
        self.p = p

    def __call__(self, img: _Tensor) -> _Tensor:
        t = _as_tensor(img)
        if random.random() < self.p:
            w = t.shape[-1]
            idx = _from_numpy(np.arange(w - 1, -1, -1, dtype=np.int64)).to(t.device)
            return t.index_select(t.ndim - 1, idx)
        return t


class RandomVerticalFlip:
    """Randomly flips the image vertically with probability p."""

    def __init__(self, p: float = 0.5):
        self.p = p

    def __call__(self, img: _Tensor) -> _Tensor:
        t = _as_tensor(img)
        if random.random() < self.p:
            h = t.shape[-2]
            idx = _from_numpy(np.arange(h - 1, -1, -1, dtype=np.int64)).to(t.device)
            return t.index_select(t.ndim - 2, idx)
        return t


def _cast_like(out: np.ndarray, dtype: np.dtype) -> np.ndarray:
    if np.issubdtype(dtype, np.integer):
        info = np.iinfo(dtype)
        return np.clip(np.round(out), info.min, info.max).astype(dtype)
    return out.astype(dtype, copy=False)


def _rotate_bilinear(arr: np.ndarray, angle_deg: float) -> np.ndarray:
    """Counter-clockwise rotation about the image center (same shape, 0 outside)."""
    h, w = arr.shape[-2], arr.shape[-1]
    theta = math.radians(angle_deg)
    cos_t, sin_t = math.cos(theta), math.sin(theta)
    cx, cy = (w - 1) / 2.0, (h - 1) / 2.0
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
    dx, dy = xx - cx, yy - cy
    xs = cx + cos_t * dx - sin_t * dy
    ys = cy + sin_t * dx + cos_t * dy
    x0 = np.floor(xs).astype(np.int64)
    y0 = np.floor(ys).astype(np.int64)
    fx = np.clip(xs - x0, 0.0, 1.0)
    fy = np.clip(ys - y0, 0.0, 1.0)
    x0c = np.clip(x0, 0, w - 1)
    x1c = np.clip(x0 + 1, 0, w - 1)
    y0c = np.clip(y0, 0, h - 1)
    y1c = np.clip(y0 + 1, 0, h - 1)
    in0x = (x0 >= 0) & (x0 < w)
    in1x = (x0 + 1 >= 0) & (x0 + 1 < w)
    in0y = (y0 >= 0) & (y0 < h)
    in1y = (y0 + 1 >= 0) & (y0 + 1 < h)
    w00 = (1.0 - fx) * (1.0 - fy) * in0x * in0y
    w01 = fx * (1.0 - fy) * in1x * in0y
    w10 = (1.0 - fx) * fy * in0x * in1y
    w11 = fx * fy * in1x * in1y
    flat = arr.astype(np.float64).reshape(-1, h, w)
    out = (
        flat[:, y0c, x0c] * w00
        + flat[:, y0c, x1c] * w01
        + flat[:, y1c, x0c] * w10
        + flat[:, y1c, x1c] * w11
    )
    return out.reshape(arr.shape)


class RandomRotation:
    """Rotates the image by a random angle in `degrees` (counter-clockwise).

    Output keeps the input shape; pixels mapping outside the image are 0.
    """

    def __init__(self, degrees: Union[float, Tuple[float, float]]):
        if isinstance(degrees, (int, float)):
            self.degrees = (-float(degrees), float(degrees))
        else:
            self.degrees = degrees

    def __call__(self, img: _Tensor) -> _Tensor:
        angle_deg = random.uniform(self.degrees[0], self.degrees[1])
        t = _as_tensor(img)
        if angle_deg % 360.0 == 0.0:
            return t
        if t.ndim in (3, 4) and str(t.dtype) in ("f32", "f64", "bf16", "f16"):
            rad = math.radians(angle_deg)
            c_a = math.cos(rad)
            s_a = math.sin(rad)
            is_batched = t.ndim == 4
            t4 = t if is_batched else t.unsqueeze(0)
            n = t4.shape[0]
            theta_np = np.tile(np.array([[c_a, -s_a, 0.0], [s_a, c_a, 0.0]], dtype=np.float64), (n, 1, 1))
            theta = _from_numpy(theta_np).to_dtype(t4.dtype).to(t4.device)
            grid = _F.affine_grid(theta, t4.shape, align_corners=True)
            out = _F.grid_sample(t4, grid, mode="bilinear", padding_mode="zeros", align_corners=True)
            return out if is_batched else out.squeeze(0)
        arr = _to_np(t)
        out = _cast_like(_rotate_bilinear(arr, angle_deg), arr.dtype)
        return _from_numpy(np.ascontiguousarray(out))


def _rgb_to_hsv(rgb: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Vectorized RGB->HSV matching `colorsys` conventions (float [..., 3])."""
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    mx = np.maximum(np.maximum(r, g), b)
    mn = np.minimum(np.minimum(r, g), b)
    diff = mx - mn
    s = np.where(mx == 0, 0.0, diff / np.maximum(mx, 1e-12))
    h = np.zeros_like(mx)
    nz = diff != 0
    is_r = nz & (mx == r)
    is_g = nz & ~is_r & (mx == g)
    is_b = nz & ~is_r & ~is_g
    h[is_r] = (((g[is_r] - b[is_r]) / diff[is_r]) % 6.0) / 6.0
    h[is_g] = ((b[is_g] - r[is_g]) / diff[is_g] + 2.0) / 6.0
    h[is_b] = ((r[is_b] - g[is_b]) / diff[is_b] + 4.0) / 6.0
    return h % 1.0, s, mx


def _hsv_to_rgb(h: np.ndarray, s: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Vectorized HSV->RGB matching `colorsys` conventions."""
    hh = (h * 6.0) % 6.0
    i = np.floor(hh).astype(np.int64)
    f = hh - i
    p = v * (1.0 - s)
    q = v * (1.0 - s * f)
    t = v * (1.0 - s * (1.0 - f))
    out = np.empty(h.shape + (3,), dtype=np.float64)
    sectors = ((v, t, p), (q, v, p), (p, v, t), (p, q, v), (t, p, v), (v, p, q))
    for k, (cr, cg, cb) in enumerate(sectors):
        m = i == k
        out[m, 0], out[m, 1], out[m, 2] = cr[m], cg[m], cb[m]
    gray = s == 0
    out[gray] = v[gray, None]
    return out


def _shift_hue(arr: np.ndarray, delta: float) -> np.ndarray:
    rgb = np.moveaxis(arr.astype(np.float64), -3, -1)
    h, s, v = _rgb_to_hsv(rgb)
    out = _hsv_to_rgb((h + delta) % 1.0, s, v)
    return np.moveaxis(out, -1, -3).astype(np.float32)


class ColorJitter:
    """Randomly changes brightness, contrast, saturation, and hue of an image.

    Expects float images in [0, 1]. Brightness,
    contrast, and saturation are non-negative magnitudes, hue is in [0, 0.5].
    """

    def __init__(
        self,
        brightness: float = 0.0,
        contrast: float = 0.0,
        saturation: float = 0.0,
        hue: float = 0.0,
    ):
        for name, value in (
            ("brightness", brightness),
            ("contrast", contrast),
            ("saturation", saturation),
        ):
            if value < 0:
                raise ValueError(f"{name} must be non-negative, got {value}")
        if not 0 <= hue <= 0.5:
            raise ValueError(f"hue must be in [0, 0.5], got {hue}")
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue

    def __call__(self, img: _Tensor) -> _Tensor:
        arr = _to_np(img).astype(np.float32)
        if self.brightness > 0:
            factor = 1.0 + random.uniform(-self.brightness, self.brightness)
            arr = arr * factor
        if self.contrast > 0:
            factor = 1.0 + random.uniform(-self.contrast, self.contrast)
            mean = np.mean(arr, axis=(-2, -1), keepdims=True)
            arr = (arr - mean) * factor + mean
        if self.saturation > 0:
            if arr.ndim < 3 or arr.shape[-3] != 3:
                raise ValueError(f"saturation jitter requires 3 channels, got shape {arr.shape}")
            factor = 1.0 + random.uniform(-self.saturation, self.saturation)
            w = np.array([0.2989, 0.5870, 0.1140], dtype=np.float32).reshape(
                (1,) * (arr.ndim - 3) + (3, 1, 1)
            )
            luma = np.sum(arr * w, axis=-3, keepdims=True)
            arr = arr * factor + luma * (1.0 - factor)
        if self.hue > 0:
            if arr.ndim < 3 or arr.shape[-3] != 3:
                raise ValueError(f"hue jitter requires 3 channels, got shape {arr.shape}")
            arr = _shift_hue(arr, random.uniform(-self.hue, self.hue))
        return _from_numpy(np.ascontiguousarray(np.clip(arr, 0.0, 1.0), dtype=np.float32))


def _check_pad(padding: Union[int, Sequence[int]]) -> Tuple[int, int]:
    if isinstance(padding, (tuple, list)):
        raw = tuple(padding)
        if len(raw) != 2:
            raise ValueError(f"Pad padding must be an int or (h, w) pair, got {padding!r}")
    else:
        raw = (padding, padding)
    try:
        dims = (operator.index(raw[0]), operator.index(raw[1]))
    except TypeError:
        raise ValueError(f"Pad padding dimensions must be ints, got {padding!r}") from None
    if dims[0] < 0 or dims[1] < 0:
        raise ValueError(f"Pad padding must be non-negative, got {dims}")
    return dims


class Pad:
    """Pads the given image on all sides with specified padding."""

    def __init__(self, padding: Union[int, Sequence[int]], fill: float = 0.0):
        self.padding = _check_pad(padding)
        self.fill = fill

    def __call__(self, img: _Tensor) -> _Tensor:
        t = _as_tensor(img)
        pad_h, pad_w = self.padding
        return _F.pad(t, (pad_w, pad_w, pad_h, pad_h), mode="constant", value=self.fill)


class RandomErasing:
    """Randomly selects a rectangle region in a tensor image and erases its pixels."""

    def __init__(
        self,
        p: float = 0.5,
        scale: Tuple[float, float] = (0.02, 0.33),
        ratio: Tuple[float, float] = (0.3, 3.3),
        value: float = 0.0,
    ):
        self.p = p
        self.scale = scale
        self.ratio = ratio
        self.value = value

    def __call__(self, img: _Tensor) -> _Tensor:
        t = _as_tensor(img)
        if random.random() > self.p:
            return t

        h, w = t.shape[-2], t.shape[-1]
        area = h * w

        for _ in range(10):
            target_area = random.uniform(self.scale[0], self.scale[1]) * area
            aspect_ratio = random.uniform(self.ratio[0], self.ratio[1])

            eh = int(round(math.sqrt(target_area * aspect_ratio)))
            ew = int(round(math.sqrt(target_area / aspect_ratio)))

            if eh < h and ew < w:
                i = random.randint(0, h - eh)
                j = random.randint(0, w - ew)
                mask_np = np.ones((1,) * (t.ndim - 2) + (h, w), dtype=np.float64)
                mask_np[..., i : i + eh, j : j + ew] = 0.0
                mask = _from_numpy(mask_np).to_dtype(t.dtype).to(t.device)
                if self.value == 0.0:
                    return t * mask
                return t * mask + (1.0 - mask) * float(self.value)

        return t


__all__ = [
    "Compose",
    "ToTensor",
    "Normalize",
    "Resize",
    "CenterCrop",
    "RandomCrop",
    "RandomHorizontalFlip",
    "RandomVerticalFlip",
    "RandomRotation",
    "ColorJitter",
    "Pad",
    "RandomErasing",
]
