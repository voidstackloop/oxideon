"""Data pipeline, encoding, training workflow, artifact caching, and checkpoint rotation utilities.

Provides Oxideon-native implementations of:
- `one_hot_encode`: Multi-dimensional class-index to one-hot indicator encoding
- `axis_normalize`: Zero-safe Lp normalization along any axis
- `pad_sequences`: Pre/post sequence padding and pre/post truncation
- `sliding_window_dataset`: Strided/sampled sliding-window dataset & loader over sequences
- `split_dataset`: Fractional or integer dataset/tensor splitter
- `image_folder_dataset` & `text_folder_dataset`: Subdirectory dataset & loader builders
- `ProgressBar`: Stateful / running-average CLI training progress bar
- `compute_class_weight` & `compute_sample_weight`: Imbalanced class weighting helpers
- `pack_batch` & `unpack_batch`: `(x, y, sample_weight)` batch tuple helpers
- `fetch_artifact` & `verify_checksum`: Cached URL downloader with SHA-256/MD5 check & safe archive extraction
- `SerializationScope`, `serialization_scope`, `register_serializable`, `get_registered_objects`,
  `get_registered_name`, `get_registered_object`, `serialize_config`, `deserialize_config`
- `CheckpointVault`: Rotating checkpoint directory manager with `max_to_keep` pruning
- `ExponentialMovingAverage`: Shadow-weight EMA tracker with `average_parameters()` context manager
"""

from contextlib import contextmanager
import hashlib
import json
import math
import os
from pathlib import Path
import random
import struct
import sys
import tarfile
import time
import urllib.request
import zipfile
import zlib
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np

from .data import DataLoader, Dataset, Subset, TensorDataset
from .oxideon import Tensor as _Tensor, from_numpy as _from_numpy
from .primitives import no_grad as _no_grad
from .serialization import load as _ox_load, save as _ox_save, save_file as _ox_save_file


_DTYPE_MAP = {
    "float32": np.float32,
    "f32": np.float32,
    "float": np.float32,
    "float64": np.float64,
    "f64": np.float64,
    "double": np.float64,
    "int32": np.int32,
    "i32": np.int32,
    "int64": np.int64,
    "i64": np.int64,
    "long": np.int64,
    "uint8": np.uint8,
    "u8": np.uint8,
    "bool": np.bool_,
}


def _resolve_np_dtype(dtype: Any) -> np.dtype:
    if isinstance(dtype, str) and dtype in _DTYPE_MAP:
        return np.dtype(_DTYPE_MAP[dtype])
    return np.dtype(dtype)


# ---------------------------------------------------------------------------
# `one_hot_encode`, `axis_normalize`, `pad_sequences`
# ---------------------------------------------------------------------------

def one_hot_encode(
    y: Any,
    num_classes: Optional[int] = None,
    dtype: Any = "float32",
) -> _Tensor:
    """Converts integer class indices of arbitrary shape into a one-hot indicator `Tensor`."""
    arr = y.detach().to_numpy() if isinstance(y, _Tensor) else np.asarray(y)
    arr = np.asarray(arr, dtype=np.int64)
    input_shape = arr.shape
    if input_shape and input_shape[-1] == 1 and len(input_shape) > 1:
        input_shape = tuple(input_shape[:-1])

    flat = arr.reshape(-1)
    if flat.size > 0 and np.any(flat < 0):
        raise ValueError("one_hot_encode: class indices must be non-negative")

    if num_classes is None:
        num_classes = int(np.max(flat)) + 1 if flat.size > 0 else 0
    num_classes = int(num_classes)
    if num_classes < 0 or (flat.size > 0 and int(np.max(flat)) >= num_classes):
        raise ValueError(
            f"one_hot_encode: num_classes={num_classes} is too small for max class index {int(np.max(flat))}"
        )

    np_dt = _resolve_np_dtype(dtype)
    encoded = np.zeros((flat.size, num_classes), dtype=np_dt)
    if flat.size > 0:
        encoded[np.arange(flat.size), flat] = 1
    output_shape = input_shape + (num_classes,)
    return _from_numpy(np.ascontiguousarray(encoded.reshape(output_shape)))


def axis_normalize(
    x: Any,
    axis: int = -1,
    order: int = 2,
) -> _Tensor:
    """Normalizes a tensor or array along `axis` using the `L_order` norm (zero-norm slices stay zero)."""
    if order <= 0:
        raise ValueError(f"axis_normalize: order must be > 0, got {order}")
    if isinstance(x, _Tensor) and x.dtype in ("f32", "f64") and getattr(x, "requires_grad", False):
        ndim = len(x.shape)
        ax = axis if axis >= 0 else axis + ndim
        from .oxideon import where
        sum_p = x.abs().powf(float(order)).sum(ax, keepdim=True)
        is_zero = sum_p == 0
        safe_sum = where(is_zero, sum_p.detach() * 0.0 + 1.0, sum_p)
        norms = safe_sum.powf(1.0 / float(order))
        return where(is_zero.expand(list(x.shape)), x * 0.0, x / norms)

    arr = x.detach().to_numpy() if isinstance(x, _Tensor) else np.asarray(x, dtype=np.float32)
    if not np.issubdtype(arr.dtype, np.floating):
        arr = arr.astype(np.float32)
    norms_np = np.atleast_1d(np.linalg.norm(arr, ord=order, axis=axis))
    norms_np[norms_np == 0] = 1.0
    if arr.ndim == 1:
        out = arr / norms_np
    else:
        out = arr / np.expand_dims(norms_np, axis=axis)
    return _from_numpy(np.ascontiguousarray(out))


def pad_sequences(
    sequences: Sequence[Any],
    maxlen: Optional[int] = None,
    dtype: Any = "int32",
    padding: str = "pre",
    truncating: str = "pre",
    value: Union[int, float] = 0.0,
) -> _Tensor:
    """Pads and/or truncates variable-length sequences (`padding`/`truncating` in `{'pre', 'post'}`)."""
    if padding not in ("pre", "post"):
        raise ValueError(f'padding must be "pre" or "post", got {padding!r}')
    if truncating not in ("pre", "post"):
        raise ValueError(f'truncating must be "pre" or "post", got {truncating!r}')
    if maxlen is not None and int(maxlen) < 0:
        raise ValueError(f"maxlen must be >= 0, got {maxlen}")

    num_samples = len(sequences)
    np_dt = _resolve_np_dtype(dtype)
    if num_samples == 0:
        ml = 0 if maxlen is None else int(maxlen)
        return _from_numpy(np.full((0, ml), value, dtype=np_dt))

    np_seqs: List[np.ndarray] = []
    lengths: List[int] = []
    sample_shape: Tuple[int, ...] = ()
    found_non_empty = False

    for s in sequences:
        a = s.detach().to_numpy() if isinstance(s, _Tensor) else np.asarray(s)
        np_seqs.append(a)
        lengths.append(int(a.shape[0]) if a.ndim >= 1 else 0)
        if not found_non_empty and a.size > 0 and a.ndim >= 1:
            sample_shape = tuple(a.shape[1:])
            found_non_empty = True

    target_len = max(lengths) if maxlen is None else int(maxlen)
    out = np.full((num_samples, target_len) + sample_shape, value, dtype=np_dt)

    for idx, a in enumerate(np_seqs):
        if a.size == 0 or target_len == 0:
            continue
        trunc = a[-target_len:] if truncating == "pre" else a[:target_len]
        trunc = np.asarray(trunc, dtype=np_dt)
        if trunc.shape[1:] != sample_shape:
            raise ValueError(
                f"Shape of sample {trunc.shape[1:]} of sequence at position {idx} "
                f"differs from expected shape {sample_shape}"
            )
        if padding == "post":
            out[idx, : len(trunc)] = trunc
        else:
            out[idx, -len(trunc) :] = trunc

    return _from_numpy(np.ascontiguousarray(out))


# ---------------------------------------------------------------------------
# `pack_batch` / `unpack_batch`
# ---------------------------------------------------------------------------

def unpack_batch(data: Any) -> Tuple[Any, Any, Any]:
    """Unpacks a training batch into `(inputs, targets, sample_weight)`."""
    if isinstance(data, list):
        data = tuple(data)
    if not isinstance(data, tuple):
        return data, None, None
    if len(data) == 1:
        return data[0], None, None
    if len(data) == 2:
        return data[0], data[1], None
    if len(data) == 3:
        return data[0], data[1], data[2]
    raise ValueError(
        f"Batch must be `x`, `(x,)`, `(x, y)`, or `(x, y, sample_weight)`, found {len(data)} elements"
    )


def pack_batch(x: Any, y: Any = None, sample_weight: Any = None) -> Any:
    """Packs `(inputs, targets, sample_weight)` into minimal batch form."""
    if y is None:
        return x if not isinstance(x, (tuple, list)) else (x,)
    if sample_weight is None:
        return (x, y)
    return (x, y, sample_weight)


# ---------------------------------------------------------------------------
# `compute_class_weight` & `compute_sample_weight`
# ---------------------------------------------------------------------------

def compute_class_weight(
    class_weight: Union[None, str, Dict[Any, float]],
    *,
    classes: Sequence[Any],
    y: Any,
) -> np.ndarray:
    """Computes per-class weights (`'balanced'`, `'uniform'`/`None`, or explicit `dict`)."""
    cls_arr = np.asarray(list(classes))
    if cls_arr.size == 0:
        raise ValueError("classes cannot be empty")
    y_arr = y.detach().to_numpy().reshape(-1) if isinstance(y, _Tensor) else np.asarray(y).reshape(-1)

    if class_weight is None or (isinstance(class_weight, str) and class_weight == "uniform"):
        return np.ones(len(cls_arr), dtype=np.float64)

    if isinstance(class_weight, str):
        if class_weight != "balanced":
            raise ValueError(f"class_weight must be 'balanced', 'uniform', None, or a dict; got {class_weight!r}")
        n_samples = len(y_arr)
        n_classes = len(cls_arr)
        counts = np.array([np.sum(y_arr == c) for c in cls_arr], dtype=np.float64)
        if np.any(counts == 0):
            missing = cls_arr[counts == 0].tolist()
            raise ValueError(f"Classes {missing} are missing from y")
        return n_samples / (n_classes * counts)

    if isinstance(class_weight, dict):
        weights = np.ones(len(cls_arr), dtype=np.float64)
        for idx, c in enumerate(cls_arr.tolist()):
            if c not in class_weight:
                raise ValueError(f"Class {c!r} not found in class_weight dict")
            weights[idx] = float(class_weight[c])
        return weights

    raise TypeError(f"Unsupported class_weight type: {type(class_weight).__name__}")


def compute_sample_weight(
    class_weight: Union[None, str, Dict[Any, float]],
    y: Any,
) -> _Tensor:
    """Computes per-sample weights for `y` according to `class_weight` (`'balanced'`, `None`, or `dict`)."""
    y_arr = y.detach().to_numpy() if isinstance(y, _Tensor) else np.asarray(y)
    flat = y_arr.reshape(-1)
    if flat.size == 0:
        return _from_numpy(np.empty(y_arr.shape, dtype=np.float32))
    classes = np.unique(flat)
    cw = compute_class_weight(class_weight, classes=classes.tolist(), y=flat)
    cls_to_w = {c: w for c, w in zip(classes.tolist(), cw.tolist())}
    sample_w = np.array([cls_to_w[val] for val in flat.tolist()], dtype=np.float32).reshape(y_arr.shape)
    return _from_numpy(np.ascontiguousarray(sample_w))


# ---------------------------------------------------------------------------
# `sliding_window_dataset` & `split_dataset`
# ---------------------------------------------------------------------------

class _SlidingWindowDataset(Dataset):
    def __init__(
        self,
        data_t: _Tensor,
        targets_t: Optional[_Tensor],
        positions: Sequence[int],
        sequence_length: int,
        sampling_rate: int,
    ) -> None:
        self._data_np = data_t.detach().to_numpy()
        self._targets_np = targets_t.detach().to_numpy() if targets_t is not None else None
        self._positions = list(positions)
        self._seq_len = sequence_length
        self._rate = sampling_rate

    def __len__(self) -> int:
        return len(self._positions)

    def __getitem__(self, idx: int) -> Any:
        start = self._positions[idx]
        end = start + self._seq_len * self._rate
        win = _from_numpy(np.ascontiguousarray(self._data_np[start : end : self._rate]))
        if self._targets_np is None:
            return win
        tgt = _from_numpy(np.ascontiguousarray(np.asarray(self._targets_np[start])))
        return win, tgt


def sliding_window_dataset(
    data: Any,
    targets: Optional[Any],
    sequence_length: int,
    sequence_stride: int = 1,
    sampling_rate: int = 1,
    batch_size: Optional[int] = 128,
    shuffle: bool = False,
    seed: Optional[int] = None,
    start_index: Optional[int] = None,
    end_index: Optional[int] = None,
) -> Union[DataLoader, Dataset]:
    """Creates a sliding-window time-series `DataLoader` (or `Dataset` when `batch_size=None`)."""
    sequence_length = int(sequence_length)
    sequence_stride = int(sequence_stride)
    sampling_rate = int(sampling_rate)
    if sequence_length <= 0:
        raise ValueError(f"sequence_length must be > 0, got {sequence_length}")
    if sequence_stride <= 0:
        raise ValueError(f"sequence_stride must be > 0, got {sequence_stride}")
    if sampling_rate <= 0:
        raise ValueError(f"sampling_rate must be > 0, got {sampling_rate}")

    data_np = data.detach().to_numpy() if isinstance(data, _Tensor) else np.asarray(data)
    n_total = int(data_np.shape[0])
    s_idx = 0 if start_index is None else int(start_index)
    e_idx = n_total if end_index is None else int(end_index)

    if s_idx < 0:
        raise ValueError(f"start_index must be >= 0, got {s_idx}")
    if e_idx > n_total:
        raise ValueError(f"end_index ({e_idx}) cannot exceed len(data) ({n_total})")
    if e_idx <= s_idx:
        raise ValueError(f"end_index ({e_idx}) must be greater than start_index ({s_idx})")

    targets_t = None
    if targets is not None:
        t_np = targets.detach().to_numpy() if isinstance(targets, _Tensor) else np.asarray(targets)
        if len(t_np) < e_idx - (sequence_length - 1) * sampling_rate:
            raise ValueError("targets array is shorter than the number of valid window start indices")
        targets_t = _from_numpy(np.ascontiguousarray(t_np))

    span = (sequence_length - 1) * sampling_rate + 1
    last_start = e_idx - span
    if last_start < s_idx:
        raise ValueError(
            f"Slice [{s_idx}:{e_idx}] is shorter than window span {span} "
            f"(sequence_length={sequence_length}, sampling_rate={sampling_rate})"
        )

    positions = list(range(s_idx, last_start + 1, sequence_stride))
    if shuffle:
        rng = random.Random(seed)
        rng.shuffle(positions)

    ds = _SlidingWindowDataset(
        _from_numpy(np.ascontiguousarray(data_np)),
        targets_t,
        positions,
        sequence_length=sequence_length,
        sampling_rate=sampling_rate,
    )
    if batch_size is None:
        return ds
    return DataLoader(ds, batch_size=int(batch_size), shuffle=False)


def split_dataset(
    dataset: Any,
    left_size: Optional[Union[int, float]] = None,
    right_size: Optional[Union[int, float]] = None,
    shuffle: bool = False,
    seed: Optional[int] = None,
) -> Tuple[Subset, Subset]:
    """Splits a `Dataset`, `Tensor`, NumPy array, or tuple of tensors into `(left_subset, right_subset)`."""
    if left_size is None and right_size is None:
        raise ValueError("At least one of `left_size` or `right_size` must be specified.")

    if isinstance(dataset, Dataset):
        ds: Dataset = dataset
    elif isinstance(dataset, _Tensor):
        ds = TensorDataset(dataset)
    elif isinstance(dataset, np.ndarray):
        ds = TensorDataset(_from_numpy(np.ascontiguousarray(dataset)))
    elif isinstance(dataset, (tuple, list)) and dataset and all(isinstance(t, (_Tensor, np.ndarray)) for t in dataset):
        tensors = [t if isinstance(t, _Tensor) else _from_numpy(np.ascontiguousarray(t)) for t in dataset]
        ds = TensorDataset(*tensors)
    else:
        class _SeqDataset(Dataset):
            def __init__(self, items: Sequence[Any]) -> None:
                self._items = list(items)
            def __len__(self) -> int:
                return len(self._items)
            def __getitem__(self, i: int) -> Any:
                return self._items[i]
        ds = _SeqDataset(list(dataset))

    total = len(ds)
    if total == 0:
        raise ValueError("Cannot split an empty dataset")

    def _to_count(val: Optional[Union[int, float]], name: str) -> Optional[int]:
        if val is None:
            return None
        if isinstance(val, float):
            if not (0.0 < val < 1.0):
                raise ValueError(f"{name} float fraction must be in (0, 1), got {val}")
            return int(round(total * val))
        if isinstance(val, int) and not isinstance(val, bool):
            if not (0 <= val <= total):
                raise ValueError(f"{name} integer count must be in [0, {total}], got {val}")
            return val
        raise TypeError(f"Invalid type for {name}: {type(val).__name__}")

    n_left = _to_count(left_size, "left_size")
    n_right = _to_count(right_size, "right_size")
    if n_left is None:
        assert n_right is not None
        n_left = total - n_right
    elif n_right is None:
        n_right = total - n_left

    if n_left + n_right > total:
        raise ValueError(f"left_size ({n_left}) + right_size ({n_right}) exceeds dataset size ({total})")

    indices = list(range(total))
    if shuffle:
        rng = random.Random(seed)
        rng.shuffle(indices)

    return Subset(ds, indices[:n_left]), Subset(ds, indices[n_left : n_left + n_right])


# ---------------------------------------------------------------------------
# `image_folder_dataset` & `text_folder_dataset`
# ---------------------------------------------------------------------------

def _paeth_predictor(a: int, b: int, c: int) -> int:
    p = a + b - c
    pa = abs(p - a)
    pb = abs(p - b)
    pc = abs(p - c)
    if pa <= pb and pa <= pc:
        return a
    if pb <= pc:
        return b
    return c


def _read_png_hwc(path: str) -> np.ndarray:
    with open(path, "rb") as f:
        data = f.read()
    if len(data) < 8 or data[:8] != b"\x89PNG\r\n\x1a\n":
        raise ValueError(f"Invalid PNG signature in '{path}'")

    pos = 8
    width = height = bit_depth = color_type = interlace = 0
    plte: Optional[bytes] = None
    trns: Optional[bytes] = None
    idat_chunks: List[bytes] = []

    while pos + 8 <= len(data):
        length = struct.unpack(">I", data[pos : pos + 4])[0]
        ctype = data[pos + 4 : pos + 8]
        chunk_data = data[pos + 8 : pos + 8 + length]
        pos += 12 + length
        if ctype == b"IHDR":
            width, height, bit_depth, color_type, _, _, interlace = struct.unpack(">IIBBBBB", chunk_data[:13])
        elif ctype == b"PLTE":
            plte = chunk_data
        elif ctype == b"tRNS":
            trns = chunk_data
        elif ctype == b"IDAT":
            idat_chunks.append(chunk_data)
        elif ctype == b"IEND":
            break

    if interlace != 0:
        raise ValueError("Interlaced PNG requires non-interlaced input")
    if bit_depth != 8:
        raise ValueError(f"Only 8-bit PNG supported in pure-Python loader, got {bit_depth}-bit")

    channels_map = {0: 1, 2: 3, 3: 1, 4: 2, 6: 4}
    if color_type not in channels_map:
        raise ValueError(f"Unsupported PNG color_type {color_type}")
    raw_c = channels_map[color_type]
    raw = zlib.decompress(b"".join(idat_chunks))
    stride = width * raw_c
    if len(raw) < height * (1 + stride):
        raise ValueError("Truncated PNG IDAT payload")

    recon = np.zeros((height, stride), dtype=np.uint8)
    bpp = raw_c
    offset = 0
    for r in range(height):
        ftype = raw[offset]
        scan = np.frombuffer(raw, dtype=np.uint8, count=stride, offset=offset + 1).copy()
        offset += 1 + stride
        if ftype == 0:
            recon[r] = scan
        elif ftype == 1:
            for i in range(bpp, stride):
                scan[i] = (int(scan[i]) + int(scan[i - bpp])) & 0xFF
            recon[r] = scan
        elif ftype == 2:
            if r > 0:
                scan = ((scan.astype(np.uint16) + recon[r - 1].astype(np.uint16)) & 0xFF).astype(np.uint8)
            recon[r] = scan
        elif ftype == 3:
            for i in range(stride):
                left = int(scan[i - bpp]) if i >= bpp else 0
                up = int(recon[r - 1, i]) if r > 0 else 0
                scan[i] = (int(scan[i]) + ((left + up) >> 1)) & 0xFF
            recon[r] = scan
        elif ftype == 4:
            for i in range(stride):
                a = int(scan[i - bpp]) if i >= bpp else 0
                b = int(recon[r - 1, i]) if r > 0 else 0
                c = int(recon[r - 1, i - bpp]) if (r > 0 and i >= bpp) else 0
                scan[i] = (int(scan[i]) + _paeth_predictor(a, b, c)) & 0xFF
            recon[r] = scan
        else:
            raise ValueError(f"Unknown PNG filter type {ftype}")

    if color_type == 3:
        if plte is None:
            raise ValueError("Palette PNG missing PLTE chunk")
        palette = np.frombuffer(plte, dtype=np.uint8).reshape(-1, 3)
        idx = recon.reshape(height, width)
        rgb = palette[idx]
        if trns is not None:
            alpha_lut = np.full(len(palette), 255, dtype=np.uint8)
            alpha_lut[: len(trns)] = np.frombuffer(trns, dtype=np.uint8)
            alpha = alpha_lut[idx][..., None]
            return np.concatenate([rgb, alpha], axis=-1)
        return rgb

    return recon.reshape(height, width, raw_c)


def _convert_color_mode(hwc: np.ndarray, color_mode: str) -> np.ndarray:
    c = hwc.shape[2]
    if color_mode == "grayscale":
        if c == 1:
            return hwc
        rgb = hwc[..., :3].astype(np.float32)
        gray = (0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]).round().clip(0, 255).astype(np.uint8)
        return gray[..., None]
    if color_mode == "rgb":
        if c == 3:
            return hwc
        if c == 4:
            return hwc[..., :3]
        return np.repeat(hwc[..., :1], 3, axis=2)
    if color_mode == "rgba":
        if c == 4:
            return hwc
        alpha = np.full(hwc.shape[:2] + (1,), 255, dtype=np.uint8)
        if c == 3:
            return np.concatenate([hwc, alpha], axis=2)
        if c == 1:
            return np.concatenate([np.repeat(hwc, 3, axis=2), alpha], axis=2)
        return np.concatenate([np.repeat(hwc[..., :1], 3, axis=2), hwc[..., 1:2]], axis=2)
    raise ValueError(f"Unsupported color_mode {color_mode!r}")


def _resize_bilinear_hwc(hwc: np.ndarray, target_hw: Tuple[int, int]) -> np.ndarray:
    th, tw = int(target_hw[0]), int(target_hw[1])
    sh, sw, _ = hwc.shape
    if (sh, sw) == (th, tw):
        return hwc.astype(np.float32)
    ys = np.linspace(0.0, max(sh - 1, 0), th, dtype=np.float32)
    xs = np.linspace(0.0, max(sw - 1, 0), tw, dtype=np.float32)
    y0 = np.floor(ys).astype(np.int64)
    x0 = np.floor(xs).astype(np.int64)
    y1 = np.minimum(y0 + 1, sh - 1)
    x1 = np.minimum(x0 + 1, sw - 1)
    wy = (ys - y0)[:, None, None]
    wx = (xs - x0)[None, :, None]

    src = hwc.astype(np.float32)
    top = src[y0][:, x0] * (1.0 - wx) + src[y0][:, x1] * wx
    bot = src[y1][:, x0] * (1.0 - wx) + src[y1][:, x1] * wx
    return top * (1.0 - wy) + bot * wy


def _discover_directory_samples(
    directory: Union[str, os.PathLike],
    extensions: Tuple[str, ...],
    labels: Union[Optional[str], Sequence[int]],
    class_names: Optional[Sequence[str]],
) -> Tuple[List[str], Optional[List[int]], List[str]]:
    root = Path(directory)
    if not root.is_dir():
        raise FileNotFoundError(f"Directory not found: '{directory}'")

    if labels is None:
        paths = sorted(
            str(p) for p in root.rglob("*") if p.is_file() and p.suffix.lower() in extensions
        )
        return paths, None, []

    subdirs = sorted(d.name for d in root.iterdir() if d.is_dir())
    if class_names is not None:
        missing = set(class_names) - set(subdirs)
        if missing:
            raise ValueError(f"class_names {sorted(missing)} not found as subdirectories in '{directory}'")
        ordered_classes = list(class_names)
    else:
        ordered_classes = subdirs

    file_paths: List[str] = []
    inferred_labels: List[int] = []
    for cls_idx, cls_name in enumerate(ordered_classes):
        cls_dir = root / cls_name
        for p in sorted(cls_dir.rglob("*")):
            if p.is_file() and p.suffix.lower() in extensions:
                file_paths.append(str(p))
                inferred_labels.append(cls_idx)

    if labels == "inferred":
        return file_paths, inferred_labels, ordered_classes
    if isinstance(labels, (list, tuple)):
        if len(labels) != len(file_paths):
            raise ValueError(f"Expected {len(file_paths)} labels, got {len(labels)}")
        return file_paths, [int(x) for x in labels], ordered_classes
    raise ValueError(f"labels must be 'inferred', None, or a list/tuple of ints; got {labels!r}")


def _apply_validation_split(
    file_paths: List[str],
    labels: Optional[List[int]],
    validation_split: Optional[float],
    subset: Optional[str],
    shuffle: bool,
    seed: Optional[int],
) -> List[Tuple[List[str], Optional[List[int]]]]:
    n = len(file_paths)
    order = list(range(n))
    if shuffle:
        rng = random.Random(seed if seed is not None else 1337)
        rng.shuffle(order)

    paths_shuf = [file_paths[i] for i in order]
    labels_shuf = [labels[i] for i in order] if labels is not None else None

    if validation_split is None:
        if subset is not None:
            raise ValueError("`subset` requires `validation_split` to be set.")
        return [(paths_shuf, labels_shuf)]

    if not (0.0 < float(validation_split) < 1.0):
        raise ValueError(f"validation_split must be in (0, 1), got {validation_split}")
    if subset not in ("training", "validation", "both"):
        raise ValueError(f"subset must be 'training', 'validation', or 'both', got {subset!r}")

    n_val = int(n * float(validation_split))
    n_train = n - n_val
    train_pair = (paths_shuf[:n_train], labels_shuf[:n_train] if labels_shuf is not None else None)
    val_pair = (paths_shuf[n_train:], labels_shuf[n_train:] if labels_shuf is not None else None)
    if subset == "training":
        return [train_pair]
    if subset == "validation":
        return [val_pair]
    return [train_pair, val_pair]


class _FolderImageDataset(Dataset):
    def __init__(
        self,
        file_paths: List[str],
        labels: Optional[List[int]],
        class_names: List[str],
        label_mode: Optional[str],
        color_mode: str,
        image_size: Tuple[int, int],
        channels_first: bool,
    ) -> None:
        self.file_paths = file_paths
        self.labels = labels
        self.class_names = class_names
        self.label_mode = label_mode
        self.color_mode = color_mode
        self.image_size = image_size
        self.channels_first = channels_first

    def __len__(self) -> int:
        return len(self.file_paths)

    def __getitem__(self, idx: int) -> Any:
        hwc = _read_png_hwc(self.file_paths[idx])
        hwc = _convert_color_mode(hwc, self.color_mode)
        resized = _resize_bilinear_hwc(hwc, self.image_size)
        arr = np.transpose(resized, (2, 0, 1)) if self.channels_first else resized
        img_t = _from_numpy(np.ascontiguousarray(arr, dtype=np.float32))
        if self.labels is None or self.label_mode is None:
            return img_t

        lbl = self.labels[idx]
        if self.label_mode == "int":
            lbl_t = _from_numpy(np.asarray(lbl, dtype=np.int64))
        elif self.label_mode == "binary":
            lbl_t = _from_numpy(np.asarray([float(lbl)], dtype=np.float32))
        elif self.label_mode == "categorical":
            one = np.zeros(max(len(self.class_names), lbl + 1), dtype=np.float32)
            one[lbl] = 1.0
            lbl_t = _from_numpy(one)
        else:
            raise ValueError(f"Unsupported label_mode {self.label_mode!r}")
        return img_t, lbl_t


def image_folder_dataset(
    directory: Union[str, os.PathLike],
    labels: Union[Optional[str], Sequence[int]] = "inferred",
    label_mode: Optional[str] = "int",
    class_names: Optional[Sequence[str]] = None,
    color_mode: str = "rgb",
    batch_size: Optional[int] = 32,
    image_size: Tuple[int, int] = (256, 256),
    shuffle: bool = True,
    seed: Optional[int] = None,
    validation_split: Optional[float] = None,
    subset: Optional[str] = None,
    channels_first: bool = True,
) -> Any:
    """Builds an image `DataLoader` (or `Dataset` when `batch_size=None`) from class subdirectories."""
    if label_mode not in ("int", "categorical", "binary", None):
        raise ValueError(f"Invalid label_mode {label_mode!r}")
    if color_mode not in ("grayscale", "rgb", "rgba"):
        raise ValueError(f"Invalid color_mode {color_mode!r}")
    if labels is None:
        label_mode = None

    paths, y_list, cls_names = _discover_directory_samples(
        directory, (".png",), labels=labels, class_names=class_names
    )
    if not paths:
        raise ValueError(f"No PNG images found in '{directory}'")
    if label_mode == "binary" and len(cls_names) != 2:
        raise ValueError(f"label_mode='binary' requires exactly 2 classes, found {len(cls_names)}")

    splits = _apply_validation_split(paths, y_list, validation_split, subset, shuffle, seed)
    outputs = []
    for sp_paths, sp_labels in splits:
        ds = _FolderImageDataset(
            sp_paths,
            sp_labels,
            cls_names,
            label_mode=label_mode,
            color_mode=color_mode,
            image_size=(int(image_size[0]), int(image_size[1])),
            channels_first=channels_first,
        )
        if batch_size is None:
            outputs.append(ds)
        else:
            loader = DataLoader(ds, batch_size=int(batch_size), shuffle=False)
            loader.class_names = cls_names  # type: ignore[attr-defined]
            loader.file_paths = sp_paths  # type: ignore[attr-defined]
            outputs.append(loader)

    return outputs if len(outputs) > 1 else outputs[0]


class _FolderTextDataset(Dataset):
    def __init__(
        self,
        file_paths: List[str],
        labels: Optional[List[int]],
        class_names: List[str],
        label_mode: Optional[str],
        max_length: Optional[int],
    ) -> None:
        self.file_paths = file_paths
        self.labels = labels
        self.class_names = class_names
        self.label_mode = label_mode
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.file_paths)

    def __getitem__(self, idx: int) -> Any:
        with open(self.file_paths[idx], "r", encoding="utf-8", errors="replace") as f:
            txt = f.read()
        if self.max_length is not None:
            txt = txt[: int(self.max_length)]
        if self.labels is None or self.label_mode is None:
            return txt
        lbl = self.labels[idx]
        if self.label_mode == "int":
            lbl_t = _from_numpy(np.asarray(lbl, dtype=np.int64))
        elif self.label_mode == "binary":
            lbl_t = _from_numpy(np.asarray([float(lbl)], dtype=np.float32))
        elif self.label_mode == "categorical":
            one = np.zeros(max(len(self.class_names), lbl + 1), dtype=np.float32)
            one[lbl] = 1.0
            lbl_t = _from_numpy(one)
        else:
            raise ValueError(f"Unsupported label_mode {self.label_mode!r}")
        return txt, lbl_t


def text_folder_dataset(
    directory: Union[str, os.PathLike],
    labels: Union[Optional[str], Sequence[int]] = "inferred",
    label_mode: Optional[str] = "int",
    class_names: Optional[Sequence[str]] = None,
    batch_size: Optional[int] = 32,
    max_length: Optional[int] = None,
    shuffle: bool = True,
    seed: Optional[int] = None,
    validation_split: Optional[float] = None,
    subset: Optional[str] = None,
) -> Any:
    """Builds a text `DataLoader` (or `Dataset` when `batch_size=None`) from `.txt` files in class subdirectories."""
    if label_mode not in ("int", "categorical", "binary", None):
        raise ValueError(f"Invalid label_mode {label_mode!r}")
    if labels is None:
        label_mode = None

    paths, y_list, cls_names = _discover_directory_samples(
        directory, (".txt",), labels=labels, class_names=class_names
    )
    if not paths:
        raise ValueError(f"No .txt files found in '{directory}'")
    if label_mode == "binary" and len(cls_names) != 2:
        raise ValueError(f"label_mode='binary' requires exactly 2 classes, found {len(cls_names)}")

    splits = _apply_validation_split(paths, y_list, validation_split, subset, shuffle, seed)
    outputs = []
    for sp_paths, sp_labels in splits:
        ds = _FolderTextDataset(sp_paths, sp_labels, cls_names, label_mode=label_mode, max_length=max_length)
        if batch_size is None:
            outputs.append(ds)
        else:
            loader = DataLoader(ds, batch_size=int(batch_size), shuffle=False)
            loader.class_names = cls_names  # type: ignore[attr-defined]
            loader.file_paths = sp_paths  # type: ignore[attr-defined]
            outputs.append(loader)
    return outputs if len(outputs) > 1 else outputs[0]


# ---------------------------------------------------------------------------
# `ProgressBar`
# ---------------------------------------------------------------------------

class ProgressBar:
    """Stateful and running-average CLI progress bar for training and evaluation loops."""

    def __init__(
        self,
        target: Optional[int],
        width: int = 30,
        verbose: int = 1,
        interval: float = 0.05,
        stateful_metrics: Optional[Iterable[str]] = None,
        unit_name: str = "step",
        stream: Any = None,
    ) -> None:
        self.target = target
        self.width = width
        self.verbose = verbose
        self.interval = interval
        self.stateful_metrics = set(stateful_metrics or ())
        self.unit_name = unit_name
        self._stream = sys.stdout if stream is None else stream

        self._seen_so_far = 0
        self._values: Dict[str, Tuple[float, float]] = {}
        self._values_order: List[str] = []
        self._start = time.perf_counter()
        self._last_update = 0.0
        self._total_width = 0

    def update(
        self,
        current: int,
        values: Optional[Sequence[Tuple[str, Any]]] = None,
        finalize: Optional[bool] = None,
    ) -> None:
        if finalize is None:
            finalize = self.target is not None and current >= self.target

        values = values or []
        step_delta = max(current - self._seen_so_far, 1)
        for k, v in values:
            if k not in self._values_order:
                self._values_order.append(k)
            scalar = float(v.item() if hasattr(v, "item") else v)
            if k not in self.stateful_metrics:
                prev_sum, prev_cnt = self._values.get(k, (0.0, 0.0))
                self._values[k] = (prev_sum + scalar * step_delta, prev_cnt + step_delta)
            else:
                self._values[k] = (scalar, 1.0)

        self._seen_so_far = current
        now = time.perf_counter()
        if self.verbose == 0:
            return
        if not finalize and (now - self._last_update) < self.interval:
            return
        self._last_update = now

        elapsed = max(now - self._start, 1e-9)
        if self.verbose == 1:
            bar = self._format_bar(current, elapsed, finalize)
            info = self._format_metrics()
            line = f"{bar}{info}"
            pad = max(self._total_width - len(line), 0)
            self._total_width = len(line)
            end_ch = "\n" if finalize else "\r"
            self._stream.write(line + (" " * pad) + end_ch)
            self._stream.flush()
        elif self.verbose == 2 and finalize:
            info = self._format_metrics()
            self._stream.write(f"{elapsed:.2f}s{info}\n")
            self._stream.flush()

    def add(self, n: int, values: Optional[Sequence[Tuple[str, Any]]] = None) -> None:
        self.update(self._seen_so_far + n, values=values)

    def _format_bar(self, current: int, elapsed: float, finalize: bool) -> str:
        if self.target is not None and self.target > 0:
            numdigits = int(math.log10(self.target)) + 1
            prefix = f"{current:{numdigits}d}/{self.target} ["
            prog = min(float(current) / float(self.target), 1.0)
            prog_width = int(self.width * prog)
            if prog_width > 0:
                bar = "=" * (prog_width - 1) + (">" if current < self.target else "=")
            else:
                bar = ""
            bar += "." * (self.width - prog_width)
            if finalize or current >= self.target:
                time_str = f"{elapsed:.2f}s"
            elif current > 0:
                eta = elapsed * (self.target - current) / current
                time_str = f"ETA: {eta:.1f}s"
            else:
                time_str = "ETA: ?"
            return f"{prefix}{bar}] - {time_str}"
        return f"{current} {self.unit_name}s - {elapsed:.2f}s"

    def _format_metrics(self) -> str:
        parts = []
        for k in self._values_order:
            total, cnt = self._values[k]
            avg = total / max(cnt, 1.0)
            if abs(avg) > 1e-3 or avg == 0.0:
                parts.append(f"{k}: {avg:.4f}")
            else:
                parts.append(f"{k}: {avg:.4e}")
        return (" - " + " - ".join(parts)) if parts else ""


# ---------------------------------------------------------------------------
# `fetch_artifact` & `verify_checksum`
# ---------------------------------------------------------------------------

def verify_checksum(fpath: Union[str, os.PathLike], expected_hash: str, algorithm: str = "auto", chunk_size: int = 65536) -> bool:
    """Verifies a file against an expected SHA-256 or MD5 hex digest."""
    expected = expected_hash.strip().lower()
    if algorithm in ("sha256", "SHA256") or (algorithm == "auto" and len(expected) == 64):
        hasher = hashlib.sha256()
    else:
        hasher = hashlib.md5()
    with open(fpath, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            hasher.update(chunk)
    return hasher.hexdigest().lower() == expected


def _safe_extract_archive(archive_path: str, extract_dir: str) -> None:
    target_root = Path(extract_dir).resolve()
    target_root.mkdir(parents=True, exist_ok=True)

    if zipfile.is_zipfile(archive_path):
        with zipfile.ZipFile(archive_path, "r") as zf:
            for member in zf.infolist():
                dest = (target_root / member.filename).resolve()
                if target_root not in dest.parents and dest != target_root:
                    raise RuntimeError(f"Unsafe zip member path blocked: {member.filename!r}")
            zf.extractall(str(target_root))
        return

    if tarfile.is_tarfile(archive_path):
        with tarfile.open(archive_path, "r:*") as tf:
            for member in tf.getmembers():
                dest = (target_root / member.name).resolve()
                if target_root not in dest.parents and dest != target_root:
                    raise RuntimeError(f"Unsafe tar member path blocked: {member.name!r}")
            tf.extractall(str(target_root))
        return

    raise ValueError(f"Unsupported archive format for extraction: '{archive_path}'")


def fetch_artifact(
    fname: Optional[str] = None,
    origin: Optional[str] = None,
    cache_dir: Optional[Union[str, os.PathLike]] = None,
    cache_subdir: str = "datasets",
    file_hash: Optional[str] = None,
    hash_algorithm: str = "auto",
    extract: bool = False,
    force_download: bool = False,
) -> str:
    """Downloads an artifact from `origin` into a local cache directory with optional checksum verification
    and path-safe archive extraction.
    """
    if origin is None:
        raise ValueError('Please specify the "origin" URL.')
    if cache_dir is None:
        cache_dir = os.environ.get("OXIDEON_HOME", os.path.expanduser("~/.cache/oxideon"))
    dest_dir = Path(cache_dir).expanduser().resolve() / cache_subdir
    dest_dir.mkdir(parents=True, exist_ok=True)

    if fname is None:
        fname = os.path.basename(origin.split("?")[0])
        if not fname:
            raise ValueError(f"Cannot infer filename from origin {origin!r}; pass `fname` explicitly.")

    fpath = dest_dir / fname
    need_download = force_download or not fpath.exists()
    if not need_download and file_hash is not None:
        if not verify_checksum(fpath, file_hash, algorithm=hash_algorithm):
            need_download = True

    if need_download:
        tmp_path = str(fpath) + ".part"
        req = urllib.request.Request(origin, headers={"User-Agent": "Oxideon/1.0"})
        with urllib.request.urlopen(req) as resp, open(tmp_path, "wb") as out_f:
            while True:
                chunk = resp.read(65536)
                if not chunk:
                    break
                out_f.write(chunk)
        if file_hash is not None and not verify_checksum(tmp_path, file_hash, algorithm=hash_algorithm):
            os.remove(tmp_path)
            raise ValueError(f"Downloaded artifact failed checksum verification: expected {file_hash}")
        if fpath.exists():
            os.remove(str(fpath))
        os.rename(tmp_path, str(fpath))

    if extract:
        extract_dir = dest_dir / f"{fpath.stem}_extracted"
        _safe_extract_archive(str(fpath), str(extract_dir))
        return str(extract_dir)
    return str(fpath)


# ---------------------------------------------------------------------------
# Serializable Component Registry (`SerializationScope`, `register_serializable`)
# ---------------------------------------------------------------------------

_GLOBAL_REGISTERED_OBJECTS: Dict[str, Any] = {}
_REGISTERED_OBJECT_NAMES: Dict[Any, str] = {}
_SCOPED_REGISTRY_STACK: List[Dict[str, Any]] = []


def get_registered_objects() -> Dict[str, Any]:
    """Returns the global dictionary of registered serializable classes and functions."""
    return _GLOBAL_REGISTERED_OBJECTS


class SerializationScope:
    """Context manager temporarily registering custom classes/functions for config deserialization."""

    def __init__(self, *args: Dict[str, Any], **kwargs: Any) -> None:
        merged: Dict[str, Any] = {}
        for d in args:
            if not isinstance(d, dict):
                raise TypeError("Positional arguments to SerializationScope must be dicts")
            merged.update(d)
        merged.update(kwargs)
        self.objects = merged

    def __enter__(self) -> "SerializationScope":
        _SCOPED_REGISTRY_STACK.append(self.objects)
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        _SCOPED_REGISTRY_STACK.pop()


serialization_scope = SerializationScope


def register_serializable(package: str = "Custom", name: Optional[str] = None) -> Callable[[Any], Any]:
    """Decorator registering a custom class or function in Oxideon's config serialization registry."""
    def decorator(arg: Any) -> Any:
        cls_name = name if name is not None else arg.__name__
        registered_name = f"{package}>{cls_name}"
        _GLOBAL_REGISTERED_OBJECTS[registered_name] = arg
        _GLOBAL_REGISTERED_OBJECTS.setdefault(cls_name, arg)
        _REGISTERED_OBJECT_NAMES[arg] = registered_name
        return arg
    return decorator


def get_registered_name(obj: Any) -> str:
    """Returns the registered name of `obj` (or `obj.__name__` if unregistered)."""
    if obj in _REGISTERED_OBJECT_NAMES:
        return _REGISTERED_OBJECT_NAMES[obj]
    return getattr(obj, "__name__", type(obj).__name__)


def get_registered_object(name: str, custom_objects: Optional[Dict[str, Any]] = None) -> Any:
    """Resolves a registered class or function by `name` across active `SerializationScope`s and the global registry."""
    if custom_objects and name in custom_objects:
        return custom_objects[name]
    for scope in reversed(_SCOPED_REGISTRY_STACK):
        if name in scope:
            return scope[name]
    return _GLOBAL_REGISTERED_OBJECTS.get(name)


def serialize_config(instance: Any) -> Any:
    """Serializes an object implementing `get_config()` (or a callable/primitive) into a JSON-compatible dictionary."""
    if instance is None or isinstance(instance, (bool, int, float, str)):
        return instance
    if isinstance(instance, (list, tuple)):
        return [serialize_config(x) for x in instance]
    if isinstance(instance, dict):
        return {k: serialize_config(v) for k, v in instance.items()}
    if hasattr(instance, "get_config") and callable(instance.get_config):
        cls = type(instance)
        return {
            "class_name": cls.__name__,
            "registered_name": _REGISTERED_OBJECT_NAMES.get(cls),
            "config": serialize_config(instance.get_config()),
        }
    if callable(instance):
        return {
            "class_name": "function",
            "config": get_registered_name(instance),
        }
    raise TypeError(f"Cannot serialize object {instance!r} of type {type(instance).__name__}")


def deserialize_config(
    identifier: Any,
    custom_objects: Optional[Dict[str, Any]] = None,
) -> Any:
    """Reconstructs an instance or function from `serialize_config` output."""
    if identifier is None or isinstance(identifier, (bool, int, float)):
        return identifier
    if isinstance(identifier, str):
        found = get_registered_object(identifier, custom_objects=custom_objects)
        if found is not None:
            return found
        raise ValueError(f"Unknown serialized identifier: {identifier!r}")
    if isinstance(identifier, list):
        return [deserialize_config(x, custom_objects=custom_objects) for x in identifier]
    if isinstance(identifier, dict):
        if "class_name" not in identifier or "config" not in identifier:
            return {k: deserialize_config(v, custom_objects=custom_objects) for k, v in identifier.items()}
        cls_name = identifier["class_name"]
        cfg = identifier["config"]
        if cls_name == "function":
            fn = get_registered_object(cfg, custom_objects=custom_objects)
            if fn is None:
                raise ValueError(f"Unknown serialized function: {cfg!r}")
            return fn
        reg_name = identifier.get("registered_name")
        cls = (get_registered_object(reg_name, custom_objects=custom_objects) if reg_name else None) or get_registered_object(
            cls_name, custom_objects=custom_objects
        )
        if cls is None:
            import oxideon.nn as _nn
            cls = getattr(_nn, cls_name, None)
        if cls is None:
            raise ValueError(f"Unknown class_name {cls_name!r} (registered_name={reg_name!r})")
        deserialized_cfg = deserialize_config(cfg, custom_objects=custom_objects)
        if hasattr(cls, "from_config") and callable(cls.from_config):
            return cls.from_config(deserialized_cfg)
        return cls(**deserialized_cfg)
    return identifier


# ---------------------------------------------------------------------------
# `CheckpointVault` & `ExponentialMovingAverage`
# ---------------------------------------------------------------------------

class CheckpointVault:
    """Manages numbered checkpoints in a directory, automatically pruning older files beyond `max_to_keep`."""

    def __init__(
        self,
        directory: Union[str, os.PathLike],
        max_to_keep: Optional[int] = 5,
        checkpoint_name: str = "ckpt",
        extension: str = ".safetensors",
    ) -> None:
        if max_to_keep is not None and int(max_to_keep) < 1:
            raise ValueError(f"max_to_keep must be >= 1 or None, got {max_to_keep}")
        if not extension.startswith("."):
            extension = "." + extension
        self.directory = Path(directory).expanduser().resolve()
        self.directory.mkdir(parents=True, exist_ok=True)
        self.max_to_keep = int(max_to_keep) if max_to_keep is not None else None
        self.checkpoint_name = checkpoint_name
        self.extension = extension
        self._manifest_path = self.directory / "checkpoint_vault.json"
        self._checkpoints: List[str] = []
        self._step_counter = 0
        self._load_manifest()

    def _load_manifest(self) -> None:
        if self._manifest_path.exists():
            try:
                with open(self._manifest_path, "r", encoding="utf-8") as f:
                    meta = json.load(f)
                self._checkpoints = [p for p in meta.get("checkpoints", []) if os.path.exists(p)]
                self._step_counter = int(meta.get("last_step", len(self._checkpoints)))
            except Exception:
                self._checkpoints = []
                self._step_counter = 0

    def _write_manifest(self) -> None:
        payload = {
            "latest_checkpoint": self.latest_checkpoint,
            "checkpoints": list(self._checkpoints),
            "last_step": self._step_counter,
        }
        with open(self._manifest_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

    @property
    def checkpoints(self) -> List[str]:
        return list(self._checkpoints)

    @property
    def latest_checkpoint(self) -> Optional[str]:
        return self._checkpoints[-1] if self._checkpoints else None

    def save(self, state: Any, step: Optional[int] = None) -> str:
        """Saves `state` (a `Module`, tensor `state_dict`, or checkpoint dict) and prunes old checkpoints."""
        if step is None:
            self._step_counter += 1
            step_num = self._step_counter
        else:
            step_num = int(step)
            self._step_counter = max(self._step_counter, step_num)

        ext = self.extension
        if ext == ".safetensors" and isinstance(state, dict) and any(not isinstance(v, _Tensor) for v in state.values()):
            ext = ".pt"

        ckpt_path = str(self.directory / f"{self.checkpoint_name}-{step_num}{ext}")
        if ext == ".safetensors":
            _ox_save_file(state, ckpt_path)
        else:
            _ox_save(state, ckpt_path)

        if ckpt_path in self._checkpoints:
            self._checkpoints.remove(ckpt_path)
        self._checkpoints.append(ckpt_path)

        if self.max_to_keep is not None:
            while len(self._checkpoints) > self.max_to_keep:
                oldest = self._checkpoints.pop(0)
                if os.path.exists(oldest):
                    os.remove(oldest)

        self._write_manifest()
        return ckpt_path

    def restore_latest(self, target: Optional[Any] = None, strict: bool = True) -> Any:
        """Restores the latest checkpoint into `target` (if given) or returns the loaded state."""
        latest = self.latest_checkpoint
        if latest is None:
            return None
        return _ox_load(latest, model=target, strict=strict)


class ShadowWeightEMA:
    """Maintains shadow exponential moving averages of parameters with an `average_parameters()` context manager."""

    def __init__(
        self,
        parameters: Iterable[_Tensor],
        decay: float = 0.999,
        use_num_updates: bool = True,
    ) -> None:
        if not (0.0 <= float(decay) <= 1.0):
            raise ValueError(f"decay must be in [0, 1], got {decay}")
        self.decay = float(decay)
        self.use_num_updates = bool(use_num_updates)
        self.num_updates = 0
        self._params: List[_Tensor] = list(parameters)
        with _no_grad():
            self.shadow_params: List[_Tensor] = [p.detach().clone() for p in self._params]
        self._backup: Optional[List[_Tensor]] = None

    def update(self, parameters: Optional[Iterable[_Tensor]] = None) -> None:
        """Updates shadow parameters after an optimizer step."""
        self.num_updates += 1
        d = self.decay
        if self.use_num_updates:
            d = min(d, (1.0 + self.num_updates) / (10.0 + self.num_updates))
        src = list(parameters) if parameters is not None else self._params
        with _no_grad():
            for i, (s, p) in enumerate(zip(self.shadow_params, src)):
                self.shadow_params[i] = s * d + p.detach() * (1.0 - d)

    def copy_to(self, parameters: Optional[Iterable[_Tensor]] = None) -> None:
        """Copies shadow averages directly into `parameters` (or the tracked parameters)."""
        dst = list(parameters) if parameters is not None else self._params
        with _no_grad():
            for p, s in zip(dst, self.shadow_params):
                p.copy_(s)

    def store(self, parameters: Optional[Iterable[_Tensor]] = None) -> None:
        """Backs up live parameter values so `restore()` can reinstate them."""
        src = list(parameters) if parameters is not None else self._params
        with _no_grad():
            self._backup = [p.detach().clone() for p in src]

    def restore(self, parameters: Optional[Iterable[_Tensor]] = None) -> None:
        """Restores backed-up training weights after evaluation."""
        if self._backup is None:
            raise RuntimeError("ShadowWeightEMA.restore() called without a prior store()/apply()")
        dst = list(parameters) if parameters is not None else self._params
        with _no_grad():
            for p, b in zip(dst, self._backup):
                p.copy_(b)
        self._backup = None

    def apply(self) -> None:
        """Backs up live parameters and copies shadow averages into them."""
        if self._backup is not None:
            raise RuntimeError("Shadow parameters are already applied; call restore() first")
        self.store()
        self.copy_to()

    @contextmanager
    def average_parameters(self, parameters: Optional[Iterable[_Tensor]] = None):
        """Context manager temporarily swapping EMA shadow weights in for evaluation."""
        self.store(parameters)
        self.copy_to(parameters)
        try:
            yield
        finally:
            self.restore(parameters)

    def state_dict(self) -> Dict[str, Any]:
        return {
            "decay": self.decay,
            "num_updates": self.num_updates,
            "use_num_updates": self.use_num_updates,
            "shadow_params": [s.detach().clone() for s in self.shadow_params],
        }

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        self.decay = float(state_dict["decay"])
        self.num_updates = int(state_dict["num_updates"])
        self.use_num_updates = bool(state_dict.get("use_num_updates", True))
        with _no_grad():
            self.shadow_params = [s.detach().clone() for s in state_dict["shadow_params"]]


__all__ = [
    "one_hot_encode",
    "axis_normalize",
    "pad_sequences",
    "pack_batch",
    "unpack_batch",
    "compute_class_weight",
    "compute_sample_weight",
    "sliding_window_dataset",
    "split_dataset",
    "image_folder_dataset",
    "text_folder_dataset",
    "ProgressBar",
    "fetch_artifact",
    "verify_checksum",
    "SerializationScope",
    "serialization_scope",
    "register_serializable",
    "get_registered_objects",
    "get_registered_name",
    "get_registered_object",
    "serialize_config",
    "deserialize_config",
    "CheckpointVault",
    "ShadowWeightEMA",
]
