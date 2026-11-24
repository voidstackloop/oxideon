"""Comprehensive Fast Fourier Transform (FFT) Suite for Oxideon.

Provides multi-dimensional discrete Fourier transforms matching the PyTorch `torch.fft`
specification, including forward, inverse, real-to-complex, complex-to-real, Hermitian,
frequency calculations, and spectral shift operations.
"""

from typing import Any, List, Optional, Sequence, Tuple, Union
import math
import numpy as np
from ._contig import contig as _contig

from .oxideon import Tensor as _Tensor, from_numpy as _from_numpy


def _to_np(x: Any) -> np.ndarray:
    if isinstance(x, _Tensor):
        return x.to_numpy()
    return np.asarray(x, dtype=np.float32)


def _to_complex_np(x: Any) -> np.ndarray:
    if hasattr(x, "numpy") and callable(x.numpy):
        arr = x.numpy()
        if np.iscomplexobj(arr):
            return arr
    arr = _to_np(x)
    if arr.ndim > 0 and arr.shape[-1] == 2:
        return arr[..., 0] + 1j * arr[..., 1]
    return arr.astype(np.complex64)


def _norm_scale(n: int, norm: Optional[str], inverse: bool) -> float:
    norm_s = "backward" if norm is None else str(norm)
    if norm_s == "ortho":
        return 1.0 / math.sqrt(float(n))
    if inverse:
        return 1.0 / float(n) if norm_s == "backward" else 1.0
    else:
        return 1.0 / float(n) if norm_s == "forward" else 1.0


def view_as_real(input: Any) -> Any:
    """Returns a view of input as a real tensor where the last dimension has size 2 (real, imag)."""
    if hasattr(input, "is_complex") and input.is_complex():
        from .primitives import view_as_real as _var
        return _var(input)
    arr = _to_np(input)
    if arr.ndim > 0 and arr.shape[-1] == 2:
        return input
    if np.iscomplexobj(arr):
        real_arr = np.stack([arr.real, arr.imag], axis=-1)
        return _from_numpy(_contig(real_arr, dtype=np.float32))
    stacked = np.stack([arr, np.zeros_like(arr)], axis=-1)
    return _from_numpy(_contig(stacked, dtype=np.float32))


def view_as_complex(input: Any) -> Any:
    """Returns a complex tensor representation from a real tensor with last dimension 2."""
    if hasattr(input, "is_complex") and input.is_complex():
        return input
    from .primitives import view_as_complex as _vac
    if isinstance(input, _Tensor):
        return _vac(input)
    arr = _to_np(input)
    assert arr.shape[-1] == 2, f"Tensor must have last dimension 2, got {arr.shape}"
    return input


def _as_complex_pair(t: _Tensor) -> Tuple[_Tensor, bool]:
    if t.ndim >= 1 and t.shape[-1] == 2:
        return t, True
    zeros = _Tensor.zeros(list(t.shape), dtype=t.dtype, device=t.device)
    return _Tensor.stack([t, zeros], dim=-1), False


def _resize_along_axis(t: _Tensor, axis: int, n: int) -> _Tensor:
    cur = t.shape[axis]
    if n < cur:
        return t.narrow(axis, 0, n)
    if n > cur:
        pad_shape = list(t.shape)
        pad_shape[axis] = n - cur
        return _Tensor.cat([t, _Tensor.zeros(pad_shape, dtype=t.dtype, device=t.device)], dim=axis)
    return t


def _apply_fft_norm(out: _Tensor, n_fft: int, norm: Optional[str], inverse: bool) -> _Tensor:
    norm_s = "backward" if norm is None else str(norm)
    if norm_s == "ortho":
        scale = math.sqrt(float(n_fft)) if inverse else 1.0 / math.sqrt(float(n_fft))
        return out * scale
    if norm_s == "forward":
        scale = float(n_fft) if inverse else 1.0 / float(n_fft)
        return out * scale
    return out


def _fft_1d_rust(
    input: _Tensor,
    n: Optional[int],
    dim: int,
    norm: Optional[str],
    inverse: bool,
) -> _Tensor:
    pair, _ = _as_complex_pair(input)
    logical_ndim = pair.ndim - 1
    axis = dim if dim >= 0 else logical_ndim + dim
    if axis != logical_ndim - 1:
        pair = pair.transpose(axis, logical_ndim - 1)
    if n is not None:
        pair = _resize_along_axis(pair, logical_ndim - 1, int(n))
    n_fft = pair.shape[-2]
    out = pair.ifft() if inverse else pair.fft()
    out = _apply_fft_norm(out, n_fft, norm, inverse)
    if axis != logical_ndim - 1:
        out = out.transpose(axis, logical_ndim - 1)
    return out


def fft(
    input: Any,
    n: Optional[int] = None,
    dim: int = -1,
    norm: Optional[str] = None,
) -> Any:
    """Computes the 1D discrete Fourier transform of input, returning [..., 2] real-imag representation or ComplexTensor."""
    if isinstance(input, _Tensor):
        return _fft_1d_rust(input, n=n, dim=dim, norm=norm, inverse=False)
    arr = _to_complex_np(input)
    res = np.fft.fft(arr, n=n, axis=dim, norm=norm)
    if hasattr(input, "is_complex") and input.is_complex():
        from .primitives import complex as _make_complex
        r = _from_numpy(_contig(res.real, dtype=np.float32))
        i = _from_numpy(_contig(res.imag, dtype=np.float32))
        dev = getattr(input, "device", None)
        if dev is not None:
            r = r.to(dev)
            i = i.to(dev)
        return _make_complex(r, i)
    stacked = np.stack([res.real, res.imag], axis=-1)
    return _from_numpy(_contig(stacked, dtype=np.float32))


def ifft(
    input: Any,
    n: Optional[int] = None,
    dim: int = -1,
    norm: Optional[str] = None,
) -> Any:
    """Computes the 1D inverse discrete Fourier transform of input."""
    if isinstance(input, _Tensor):
        return _fft_1d_rust(input, n=n, dim=dim, norm=norm, inverse=True)
    arr = _to_complex_np(input)
    res = np.fft.ifft(arr, n=n, axis=dim, norm=norm)
    if hasattr(input, "is_complex") and input.is_complex():
        from .primitives import complex as _make_complex
        r = _from_numpy(_contig(res.real, dtype=np.float32))
        i = _from_numpy(_contig(res.imag, dtype=np.float32))
        dev = getattr(input, "device", None)
        if dev is not None:
            r = r.to(dev)
            i = i.to(dev)
        return _make_complex(r, i)
    stacked = np.stack([res.real, res.imag], axis=-1)
    return _from_numpy(_contig(stacked, dtype=np.float32))


def rfft(
    input: _Tensor,
    n: Optional[int] = None,
    dim: int = -1,
    norm: Optional[str] = None,
) -> _Tensor:
    """Computes the 1D discrete Fourier transform of real input, returning one-sided spectrum [..., 2]."""
    if isinstance(input, _Tensor) and input.requires_grad and (dim == -1 or dim == input.ndim - 1):
        in_len = input.shape[-1]
        n_fft = in_len if n is None else int(n)
        x = _resize_along_axis(input, input.ndim - 1, n_fft)
        full = x.rfft()
        out = full.narrow(full.ndim - 2, 0, n_fft // 2 + 1)
        scale = _norm_scale(n_fft, norm, inverse=False)
        if scale != 1.0:
            out = out * scale
        return out

    arr = _to_np(input)
    res = np.fft.rfft(arr, n=n, axis=dim, norm=norm)
    stacked = np.stack([res.real, res.imag], axis=-1)
    return _from_numpy(_contig(stacked, dtype=np.float32))


def irfft(
    input: _Tensor,
    n: Optional[int] = None,
    dim: int = -1,
    norm: Optional[str] = None,
) -> _Tensor:
    """Computes the inverse of rfft, returning a purely real-valued tensor."""
    if isinstance(input, _Tensor) and input.requires_grad and input.ndim >= 2 and input.shape[-1] == 2 and (dim == -1 or dim == input.ndim - 2):
        k_freqs = input.shape[-2]
        n_fft = 2 * (k_freqs - 1) if n is None else int(n)
        # Multiplicity weights for one-sided Hermitian spectrum
        mult = np.ones((k_freqs, 1), dtype=np.float32) * 2.0
        mult[0, 0] = 1.0
        if n_fft % 2 == 0 and k_freqs == n_fft // 2 + 1:
            mult[-1, 0] = 1.0
        k_idx = np.arange(k_freqs, dtype=np.float32)[:, None]
        t_idx = np.arange(n_fft, dtype=np.float32)[None, :]
        angles = 2.0 * math.pi * (k_idx @ t_idx) / float(n_fft)
        scale = _norm_scale(n_fft, norm, inverse=True)
        cos_mat = _from_numpy(_contig(mult * np.cos(angles) * scale, dtype=np.float32))
        sin_mat = _from_numpy(_contig(-mult * np.sin(angles) * scale, dtype=np.float32))

        orig_prefix = list(input.shape[:-2])
        flat = input.reshape([-1, k_freqs, 2])
        re = flat.narrow(2, 0, 1).reshape([-1, k_freqs])
        im = flat.narrow(2, 1, 1).reshape([-1, k_freqs])
        out = re.matmul(cos_mat) + im.matmul(sin_mat)
        return out.reshape(orig_prefix + [n_fft])

    arr = _to_complex_np(input)
    res = np.fft.irfft(arr, n=n, axis=dim, norm=norm)
    return _from_numpy(_contig(res, dtype=np.float32))


def fft2(
    input: _Tensor,
    s: Optional[Sequence[int]] = None,
    dim: Tuple[int, int] = (-2, -1),
    norm: Optional[str] = None,
) -> _Tensor:
    """Computes the 2D discrete Fourier transform of input."""
    if isinstance(input, _Tensor):
        pair, _ = _as_complex_pair(input)
        logical_ndim = pair.ndim - 1
        d0 = dim[0] if dim[0] >= 0 else logical_ndim + dim[0]
        d1 = dim[1] if dim[1] >= 0 else logical_ndim + dim[1]
        if s is None and (d0, d1) == (logical_ndim - 2, logical_ndim - 1) and norm in (None, "backward"):
            return pair.fft2d()
        s0, s1 = (s[0], s[1]) if s is not None else (None, None)
        out = _fft_1d_rust(pair, n=s1, dim=d1, norm=norm, inverse=False)
        return _fft_1d_rust(out, n=s0, dim=d0, norm=norm, inverse=False)
    arr = _to_complex_np(input)
    res = np.fft.fft2(arr, s=s, axes=dim, norm=norm)
    stacked = np.stack([res.real, res.imag], axis=-1)
    return _from_numpy(_contig(stacked, dtype=np.float32))


def ifft2(
    input: _Tensor,
    s: Optional[Sequence[int]] = None,
    dim: Tuple[int, int] = (-2, -1),
    norm: Optional[str] = None,
) -> _Tensor:
    """Computes the 2D inverse discrete Fourier transform of input."""
    if isinstance(input, _Tensor):
        pair, _ = _as_complex_pair(input)
        logical_ndim = pair.ndim - 1
        d0 = dim[0] if dim[0] >= 0 else logical_ndim + dim[0]
        d1 = dim[1] if dim[1] >= 0 else logical_ndim + dim[1]
        if s is None and (d0, d1) == (logical_ndim - 2, logical_ndim - 1) and norm in (None, "backward"):
            return pair.ifft2d()
        s0, s1 = (s[0], s[1]) if s is not None else (None, None)
        out = _fft_1d_rust(pair, n=s1, dim=d1, norm=norm, inverse=True)
        return _fft_1d_rust(out, n=s0, dim=d0, norm=norm, inverse=True)
    arr = _to_complex_np(input)
    res = np.fft.ifft2(arr, s=s, axes=dim, norm=norm)
    stacked = np.stack([res.real, res.imag], axis=-1)
    return _from_numpy(_contig(stacked, dtype=np.float32))


def rfft2(
    input: _Tensor,
    s: Optional[Sequence[int]] = None,
    dim: Tuple[int, int] = (-2, -1),
    norm: Optional[str] = None,
) -> _Tensor:
    """Computes the 2D discrete Fourier transform of real input."""
    if isinstance(input, _Tensor) and input.requires_grad and tuple(dim) in ((-2, -1), (input.ndim - 2, input.ndim - 1)):
        s0, s1 = (s[0], s[1]) if s is not None else (None, None)
        x = rfft(input, n=s1, dim=-1, norm=norm)
        return _fft_1d_rust(x, n=s0, dim=-2, norm=norm, inverse=False)
    arr = _to_np(input)
    res = np.fft.rfft2(arr, s=s, axes=dim, norm=norm)
    stacked = np.stack([res.real, res.imag], axis=-1)
    return _from_numpy(_contig(stacked, dtype=np.float32))


def irfft2(
    input: _Tensor,
    s: Optional[Sequence[int]] = None,
    dim: Tuple[int, int] = (-2, -1),
    norm: Optional[str] = None,
) -> _Tensor:
    """Computes the inverse of rfft2, returning a purely real-valued tensor."""
    arr = _to_complex_np(input)
    res = np.fft.irfft2(arr, s=s, axes=dim, norm=norm)
    return _from_numpy(_contig(res, dtype=np.float32))


def fftn(
    input: _Tensor,
    s: Optional[Sequence[int]] = None,
    dim: Optional[Sequence[int]] = None,
    norm: Optional[str] = None,
) -> _Tensor:
    """Computes the N-dimensional discrete Fourier transform of input."""
    arr = _to_complex_np(input)
    res = np.fft.fftn(arr, s=s, axes=dim, norm=norm)
    stacked = np.stack([res.real, res.imag], axis=-1)
    return _from_numpy(_contig(stacked, dtype=np.float32))


def ifftn(
    input: _Tensor,
    s: Optional[Sequence[int]] = None,
    dim: Optional[Sequence[int]] = None,
    norm: Optional[str] = None,
) -> _Tensor:
    """Computes the N-dimensional inverse discrete Fourier transform of input."""
    arr = _to_complex_np(input)
    res = np.fft.ifftn(arr, s=s, axes=dim, norm=norm)
    stacked = np.stack([res.real, res.imag], axis=-1)
    return _from_numpy(_contig(stacked, dtype=np.float32))


def rfftn(
    input: _Tensor,
    s: Optional[Sequence[int]] = None,
    dim: Optional[Sequence[int]] = None,
    norm: Optional[str] = None,
) -> _Tensor:
    """Computes the N-dimensional discrete Fourier transform of real input."""
    arr = _to_np(input)
    res = np.fft.rfftn(arr, s=s, axes=dim, norm=norm)
    stacked = np.stack([res.real, res.imag], axis=-1)
    return _from_numpy(_contig(stacked, dtype=np.float32))


def irfftn(
    input: _Tensor,
    s: Optional[Sequence[int]] = None,
    dim: Optional[Sequence[int]] = None,
    norm: Optional[str] = None,
) -> _Tensor:
    """Computes the inverse of rfftn, returning a purely real-valued tensor."""
    arr = _to_complex_np(input)
    res = np.fft.irfftn(arr, s=s, axes=dim, norm=norm)
    return _from_numpy(_contig(res, dtype=np.float32))


def hfft(
    input: _Tensor,
    n: Optional[int] = None,
    dim: int = -1,
    norm: Optional[str] = None,
) -> _Tensor:
    """Computes the 1D discrete Fourier transform of an Hermitian symmetric input."""
    arr = _to_complex_np(input)
    res = np.fft.hfft(arr, n=n, axis=dim, norm=norm)
    return _from_numpy(_contig(res, dtype=np.float32))


def ihfft(
    input: _Tensor,
    n: Optional[int] = None,
    dim: int = -1,
    norm: Optional[str] = None,
) -> _Tensor:
    """Computes the inverse of hfft."""
    arr = _to_np(input)
    res = np.fft.ihfft(arr, n=n, axis=dim, norm=norm)
    stacked = np.stack([res.real, res.imag], axis=-1)
    return _from_numpy(_contig(stacked, dtype=np.float32))


def fftfreq(n: int, d: float = 1.0, dtype: Optional[Any] = None, device: Optional[Any] = None) -> _Tensor:
    """Computes the discrete Fourier transform sample frequencies."""
    arr = np.fft.fftfreq(n, d=d)
    return _from_numpy(_contig(arr, dtype=np.float32))


def rfftfreq(n: int, d: float = 1.0, dtype: Optional[Any] = None, device: Optional[Any] = None) -> _Tensor:
    """Computes the sample frequencies for rfft."""
    arr = np.fft.rfftfreq(n, d=d)
    return _from_numpy(_contig(arr, dtype=np.float32))


def fftshift(input: _Tensor, dim: Optional[Union[int, Sequence[int]]] = None) -> _Tensor:
    """Shifts the zero-frequency component to the center of the spectrum."""
    arr = _to_np(input)
    res = np.fft.fftshift(arr, axes=dim)
    return _from_numpy(_contig(res, dtype=arr.dtype))


def ifftshift(input: _Tensor, dim: Optional[Union[int, Sequence[int]]] = None) -> _Tensor:
    """Inverse of fftshift."""
    arr = _to_np(input)
    res = np.fft.ifftshift(arr, axes=dim)
    return _from_numpy(_contig(res, dtype=arr.dtype))


__all__ = [
    "fft",
    "ifft",
    "rfft",
    "irfft",
    "fft2",
    "ifft2",
    "rfft2",
    "irfft2",
    "fftn",
    "ifftn",
    "rfftn",
    "irfftn",
    "hfft",
    "ihfft",
    "fftfreq",
    "rfftfreq",
    "fftshift",
    "ifftshift",
    "view_as_real",
    "view_as_complex",
]


# ---------------------------------------------------------------------------------------------------------------------
# Multi-dimensional Hermitian FFTs (complex values are stored as a trailing (real, imag) dimension, as elsewhere here)
# ---------------------------------------------------------------------------------------------------------------------

_SWAP_NORM = {None: "forward", "backward": "forward", "forward": "backward", "ortho": "ortho"}


def _check_norm(norm: Optional[str]) -> None:
    if norm not in _SWAP_NORM:
        raise ValueError(f"Invalid normalization mode: {norm!r}")


def _cplx(x: _Tensor) -> _Tensor:
    """A real tensor as complex (real, imag) pairs."""
    from .oxideon import stack
    return stack([x, x * 0.0], -1)


def _conj(c: _Tensor) -> _Tensor:
    return c * _from_numpy(np.array([1.0, -1.0], dtype=np.float32)).to_dtype(c.dtype)


def _fft_axis(c: _Tensor, axis: int, inverse: bool) -> _Tensor:
    """The (Rust) 1-D complex FFT of `c` ((real, imag) pairs in the last dimension) along complex axis `axis`,
    unnormalized in both directions."""
    nd = len(c.shape) - 1
    perm = [i for i in range(nd) if i != axis] + [axis, nd]
    inv = [0] * (nd + 1)
    for i, p in enumerate(perm):
        inv[p] = i
    moved = c.permute(perm).contiguous()
    out = moved.ifft() * float(c.shape[axis]) if inverse else moved.fft()
    return out.permute(inv).contiguous()


def _fit(c: _Tensor, axis: int, size: int) -> _Tensor:
    """Crops or zero-pads complex axis `axis` of `c` to `size`."""
    from .oxideon import cat
    n = int(c.shape[axis])
    if n == size:
        return c
    if n > size:
        return c.narrow(axis, 0, size).contiguous()
    shape = list(c.shape)
    shape[axis] = size - n
    return cat([c, _from_numpy(np.zeros(tuple(shape), dtype=np.float32)).to_dtype(c.dtype)], axis)


def _dims(nd: int, s: Optional[Sequence[int]], dim: Optional[Sequence[int]]) -> List[int]:
    if dim is not None:
        return [d % nd for d in (dim if isinstance(dim, (list, tuple)) else [dim])]
    if s is not None:
        return list(range(nd - len(s), nd))
    return list(range(nd))


def _scale(norm: Optional[str], n: int, forward_direction: bool) -> float:
    """The normalization factor: "backward" (the default) leaves forward transforms unscaled and divides inverse ones
    by n, "forward" does the opposite, "ortho" divides both by sqrt(n)."""
    if norm == "ortho":
        return 1.0 / math.sqrt(n)
    backward = norm in (None, "backward")
    return 1.0 if backward == forward_direction else 1.0 / n


def hfftn(input: _Tensor, s: Optional[Sequence[int]] = None, dim: Optional[Sequence[int]] = None, norm: Optional[str] = None) -> _Tensor:
    """The N-dimensional FFT of a Hermitian-symmetric signal given by its one-sided half (the last dimension in `dim`
    holds n // 2 + 1 values); the result is real. The inverse of `ihfftn`. Computed with the Rust FFT."""
    _check_norm(norm)
    from .oxideon import cat
    x = input if isinstance(input, _Tensor) else _from_numpy(_contig(_to_np(input)))
    if not (len(x.shape) >= 1 and x.shape[-1] == 2):
        x = _cplx(x)
    x = x.double()
    nd = len(x.shape) - 1
    dims = _dims(nd, s, dim)
    last = dims[-1]
    sizes = [int(v) for v in s] if s is not None else [int(x.shape[d]) for d in dims[:-1]] + [2 * (int(x.shape[last]) - 1)]
    if sizes[-1] < 1:
        raise ValueError(f"Invalid number of data points ({sizes[-1]}) specified")
    c = _conj(x)
    for d, n in zip(dims[:-1], sizes[:-1]):
        c = _fft_axis(_fit(c, d, n), d, inverse=True)
    n_last = sizes[-1]
    m = n_last // 2 + 1
    c = _fit(c, last, m)
    if n_last - m > 0:
        idx = _from_numpy(np.arange(n_last - m, 0, -1, dtype=np.int64))
        c = cat([c, _conj(c.index_select(last, idx))], last)
    y = _fft_axis(c, last, inverse=True)
    real = y.narrow(len(y.shape) - 1, 0, 1).reshape(list(y.shape[:-1]))
    total = int(np.prod(sizes))
    out = real * _scale(norm, total, forward_direction=True)  # hfftn is a forward transform
    return out.float() if input.dtype != "f64" else out


def hfft2(input: _Tensor, s: Optional[Sequence[int]] = None, dim: Sequence[int] = (-2, -1), norm: Optional[str] = None) -> _Tensor:
    """`hfftn` over two dimensions (the last two by default)."""
    return hfftn(input, s=s, dim=dim, norm=norm)


def ihfftn(input: _Tensor, s: Optional[Sequence[int]] = None, dim: Optional[Sequence[int]] = None, norm: Optional[str] = None) -> _Tensor:
    """The N-dimensional inverse FFT of a real signal, returning only the non-redundant half of the last dimension in
    `dim` (as (real, imag) pairs). The inverse of `hfftn`. Computed with the Rust FFT."""
    _check_norm(norm)
    x = input if isinstance(input, _Tensor) else _from_numpy(_contig(_to_np(input)))
    c = _cplx(x.double())
    nd = len(c.shape) - 1
    dims = _dims(nd, s, dim)
    sizes = [int(v) for v in s] if s is not None else [int(c.shape[d]) for d in dims]
    for d, n in zip(dims, sizes):
        c = _fft_axis(_fit(c, d, n), d, inverse=False)
    last = dims[-1]
    c = c.narrow(last, 0, sizes[-1] // 2 + 1).contiguous()
    total = int(np.prod(sizes))
    out = _conj(c) * _scale(norm, total, forward_direction=False)  # ihfftn is an inverse transform
    return out.float() if x.dtype != "f64" else out


def ihfft2(input: _Tensor, s: Optional[Sequence[int]] = None, dim: Sequence[int] = (-2, -1), norm: Optional[str] = None) -> _Tensor:
    """`ihfftn` over two dimensions (the last two by default)."""
    return ihfftn(input, s=s, dim=dim, norm=norm)


__all__ += ["hfftn", "hfft2", "ihfftn", "ihfft2"]
