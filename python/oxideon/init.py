"""Parameter initialization schemes for Oxideon (nn.init).

Provides standard weight initialization routines:
  - uniform_, normal_, constant_, zeros_, ones_, eye_, trunc_normal_
  - xavier_uniform_, xavier_normal_ (Glorot initialization)
  - kaiming_uniform_, kaiming_normal_ (He initialization)
  - orthogonal_
"""

from typing import Tuple, Optional
import math
import numpy as np

from .oxideon import Tensor as _Tensor, from_numpy as _from_numpy


def _calculate_fan_in_and_fan_out(tensor: _Tensor) -> Tuple[int, int]:
    shape = tensor.shape
    dimensions = len(shape)
    if dimensions < 2:
        raise ValueError(
            "Fan in and fan out can not be computed for tensor with fewer than 2 dimensions"
        )

    num_input_fmaps = shape[1]
    num_output_fmaps = shape[0]
    receptive_field_size = 1
    if dimensions > 2:
        for s in shape[2:]:
            receptive_field_size *= s
    fan_in = num_input_fmaps * receptive_field_size
    fan_out = num_output_fmaps * receptive_field_size
    return fan_in, fan_out


def calculate_gain(nonlinearity: str, param: Optional[float] = None) -> float:
    """Returns the recommended gain value for the given nonlinearity function."""
    linear_fns = ["linear", "conv1d", "conv2d", "conv3d"]
    if nonlinearity in linear_fns or nonlinearity == "sigmoid":
        return 1.0
    elif nonlinearity == "tanh":
        return 5.0 / 3.0
    elif nonlinearity == "relu":
        return math.sqrt(2.0)
    elif nonlinearity == "leaky_relu":
        negative_slope = 0.01 if param is None else float(param)
        return math.sqrt(2.0 / (1.0 + negative_slope ** 2))
    elif nonlinearity == "selu":
        return 3.0 / 4.0
    else:
        raise ValueError(f"calculate_gain: unsupported nonlinearity {nonlinearity!r}")


def _assign_inplace(tensor: _Tensor, arr: np.ndarray) -> _Tensor:
    new_t = _from_numpy(np.ascontiguousarray(arr, dtype=np.float32))
    if hasattr(tensor, "assign"):
        tensor.assign(new_t)
        return tensor
    return new_t


def uniform_(tensor: _Tensor, a: float = 0.0, b: float = 1.0, seed: Optional[int] = None) -> _Tensor:
    """Fills the input Tensor with values drawn from uniform distribution U(a, b)."""
    rng = np.random.default_rng(seed) if seed is not None else np.random
    arr = rng.uniform(a, b, size=tensor.shape).astype(np.float32)
    return _assign_inplace(tensor, arr)


def normal_(tensor: _Tensor, mean: float = 0.0, std: float = 1.0, seed: Optional[int] = None) -> _Tensor:
    """Fills the input Tensor with values drawn from normal distribution N(mean, std^2)."""
    rng = np.random.default_rng(seed) if seed is not None else np.random
    arr = rng.normal(mean, std, size=tensor.shape).astype(np.float32)
    return _assign_inplace(tensor, arr)


def trunc_normal_(
    tensor: _Tensor,
    mean: float = 0.0,
    std: float = 1.0,
    a: float = -2.0,
    b: float = 2.0,
    seed: Optional[int] = None,
) -> _Tensor:
    """Fills the input Tensor with values drawn from a truncated normal distribution."""
    if a >= b:
        raise ValueError(f"trunc_normal_ requires a < b, got a={a}, b={b}")
    rng = np.random.default_rng(seed) if seed is not None else np.random
    numel = int(np.prod(tensor.shape)) if len(tensor.shape) > 0 else 1
    samples = rng.normal(mean, std, size=numel).astype(np.float32)
    for _ in range(10):
        mask = (samples < a) | (samples > b)
        if not np.any(mask):
            break
        samples[mask] = rng.normal(mean, std, size=int(np.sum(mask))).astype(np.float32)
    samples = np.clip(samples, a, b).reshape(tensor.shape).astype(np.float32)
    return _assign_inplace(tensor, samples)


def constant_(tensor: _Tensor, val: float) -> _Tensor:
    """Fills the input Tensor with the constant value val."""
    arr = np.full(tensor.shape, val, dtype=np.float32)
    return _assign_inplace(tensor, arr)


def zeros_(tensor: _Tensor) -> _Tensor:
    """Fills the input Tensor with the scalar value 0."""
    return constant_(tensor, 0.0)


def ones_(tensor: _Tensor) -> _Tensor:
    """Fills the input Tensor with the scalar value 1."""
    return constant_(tensor, 1.0)


def eye_(tensor: _Tensor) -> _Tensor:
    """Fills the 2D input Tensor with the identity matrix."""
    shape = tensor.shape
    if len(shape) != 2:
        raise ValueError("eye_ only supports 2D tensors")
    arr = np.eye(shape[0], shape[1], dtype=np.float32)
    return _assign_inplace(tensor, arr)


def xavier_uniform_(tensor: _Tensor, gain: float = 1.0, seed: Optional[int] = None) -> _Tensor:
    """Fills the input Tensor with values according to the method described in
    'Understanding the difficulty of training deep feedforward neural networks' (Glorot & Bengio).
    """
    fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor)
    std = gain * math.sqrt(2.0 / float(fan_in + fan_out))
    a = math.sqrt(3.0) * std
    return uniform_(tensor, -a, a, seed=seed)


def xavier_normal_(tensor: _Tensor, gain: float = 1.0, seed: Optional[int] = None) -> _Tensor:
    """Fills the input Tensor with values according to Xavier normal initialization."""
    fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor)
    std = gain * math.sqrt(2.0 / float(fan_in + fan_out))
    return normal_(tensor, 0.0, std, seed=seed)


def kaiming_uniform_(
    tensor: _Tensor,
    a: float = 0.0,
    mode: str = "fan_in",
    nonlinearity: str = "leaky_relu",
    seed: Optional[int] = None,
) -> _Tensor:
    """Fills the input Tensor with values according to He / Kaiming uniform initialization."""
    fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor)
    if mode not in ("fan_in", "fan_out"):
        raise ValueError(f"mode must be 'fan_in' or 'fan_out', got {mode!r}")
    fan = fan_in if mode == "fan_in" else fan_out
    gain = calculate_gain(nonlinearity, a)
    std = gain / math.sqrt(fan)
    bound = math.sqrt(3.0) * std
    return uniform_(tensor, -bound, bound, seed=seed)


def kaiming_normal_(
    tensor: _Tensor,
    a: float = 0.0,
    mode: str = "fan_in",
    nonlinearity: str = "leaky_relu",
    seed: Optional[int] = None,
) -> _Tensor:
    """Fills the input Tensor with values according to He / Kaiming normal initialization."""
    fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor)
    if mode not in ("fan_in", "fan_out"):
        raise ValueError(f"mode must be 'fan_in' or 'fan_out', got {mode!r}")
    fan = fan_in if mode == "fan_in" else fan_out
    gain = calculate_gain(nonlinearity, a)
    std = gain / math.sqrt(fan)
    return normal_(tensor, 0.0, std, seed=seed)


def orthogonal_(tensor: _Tensor, gain: float = 1.0, seed: Optional[int] = None) -> _Tensor:
    """Fills the input Tensor with a (semi) orthogonal matrix."""
    shape = tensor.shape
    if len(shape) < 2:
        raise ValueError("Only tensors with 2 or more dimensions are supported")

    rows = shape[0]
    cols = 1
    for s in shape[1:]:
        cols *= s

    rng = np.random.default_rng(seed) if seed is not None else np.random
    flattened = rng.normal(0.0, 1.0, (rows, cols)).astype(np.float32)
    if rows < cols:
        flattened = flattened.T

    # QR decomposition
    q, r = np.linalg.qr(flattened)
    d = np.diag(r)
    ph = np.sign(d)
    q *= ph

    if rows < cols:
        q = q.T

    arr = (gain * q).reshape(shape).astype(np.float32)
    return _assign_inplace(tensor, arr)


__all__ = [
    "calculate_gain",
    "uniform_",
    "normal_",
    "trunc_normal_",
    "constant_",
    "zeros_",
    "ones_",
    "eye_",
    "xavier_uniform_",
    "xavier_normal_",
    "kaiming_uniform_",
    "kaiming_normal_",
    "orthogonal_",
]
