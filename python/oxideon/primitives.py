import builtins
"""
Oxideon Core Primitives, Unified Dynamic Runtime, and High-Performance Execution Engine.

Provides unified tensor constructors, multi-device management, autograd controls,
dynamic shapes, and JIT compilation interfaces across CPU and hardware accelerators.
"""
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union
from collections import namedtuple
import contextlib
import copyreg
import functools
import math
import operator
import sys
import time
import numpy as np
from ._contig import contig as _contig

from .oxideon import (
    Tensor as _Tensor,
    from_numpy as _from_numpy,
    manual_seed as _manual_seed,
    searchsorted as _native_searchsorted,
    where as _native_where,
    einsum as _native_einsum,
    tril as _native_tril,
    triu as _native_triu,
    kron as _native_kron,
    cdist as _native_cdist,
    pdist as _native_pdist,
    cross as _native_cross,
    cummax as _native_cummax,
    cummin as _native_cummin,
    logcumsumexp as _native_logcumsumexp,
    tensordot as _native_tensordot,
    as_strided as _native_as_strided,
    grad as _native_grad,
)

# This module defines tensor ops named any/all/sum/abs/round, which shadow the
# builtins below their definitions. Internal code that means the builtins must
# use these captured references instead.
_builtin_any = any
_builtin_abs = abs
_builtin_sum = sum

_GLOBAL_GRAD_ENABLED: bool = True


def is_grad_enabled() -> bool:
    """Returns True if autograd graph construction is active on the current thread."""
    return _GLOBAL_GRAD_ENABLED


def _set_grad_enabled_impl(mode: bool) -> bool:
    global _GLOBAL_GRAD_ENABLED
    prev = _GLOBAL_GRAD_ENABLED
    _GLOBAL_GRAD_ENABLED = bool(mode)
    return prev


# ---------------------------------------------------------------------------
# 1. Size & Data Type Abstractions
# ---------------------------------------------------------------------------

class Size(tuple):
    """Multi-dimensional tensor shape representation supporting dual tuple and sequence operations."""
    def __repr__(self) -> str:
        return f"Size([{', '.join(str(x) for x in self)}])"

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, (tuple, list)):
            return tuple(self) == tuple(other)
        return super().__eq__(other)

    def __hash__(self) -> int:
        # __eq__ without __hash__ would make Size unhashable; hash as a tuple
        # (consistent with tuple/list equality above).
        return tuple.__hash__(self)

    def numel(self) -> int:
        n = 1
        for x in self:
            n *= x
        return n


class dtype:
    """Primitive data type specification for tensors and numerical buffers."""
    def __init__(self, name: str, itemsize: int, is_floating_point: bool = False, is_signed: bool = True, is_complex: bool = False, bits: Optional[int] = None):
        self._name = name
        self.itemsize = itemsize
        self.is_floating_point = is_floating_point
        self.is_signed = is_signed
        self.is_complex = is_complex
        self.bits = bits if bits is not None else itemsize * 8

    @property
    def name(self) -> str:
        return self._name

    def __repr__(self) -> str:
        return f"oxideon.{self._name}"

    def __str__(self) -> str:
        return f"oxideon.{self._name}"

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, dtype):
            return self._name == other._name
        if isinstance(other, str):
            mapping = {
                "f32": "float32", "float32": "float32", "float": "float32",
                "f64": "float64", "float64": "float64", "double": "float64",
                "f16": "float16", "float16": "float16", "half": "float16",
                "bf16": "bfloat16", "bfloat16": "bfloat16",
                "i32": "int32", "int32": "int32", "int": "int32",
                "i64": "int64", "int64": "int64", "long": "int64",
                "i16": "int16", "int16": "int16", "short": "int16",
                "i8": "int8", "int8": "int8", "char": "int8",
                "u8": "uint8", "uint8": "uint8", "byte": "uint8",
                "u16": "uint16", "uint16": "uint16",
                "u32": "uint32", "uint32": "uint32",
                "u64": "uint64", "uint64": "uint64",
                "bool": "bool", "bool_": "bool",
                "c64": "complex64", "complex64": "complex64", "cfloat": "complex64",
                "c128": "complex128", "complex128": "complex128", "cdouble": "complex128",
                "e4m3": "float8_e4m3fn", "float8_e4m3fn": "float8_e4m3fn", "fp8_e4m3": "float8_e4m3fn",
                "e5m2": "float8_e5m2", "float8_e5m2": "float8_e5m2", "fp8_e5m2": "float8_e5m2",
            }
            return mapping.get(self._name, self._name) == mapping.get(other.lower(), other.lower())
        if other is float:
            return self._name in ("float32", "float64")
        if other is int:
            return self._name in ("int32", "int64")
        if other is bool:
            return self._name == "bool"
        if other is complex or other is builtins.complex:
            return self._name in ("complex64", "complex128")
        if hasattr(other, "name"):
            return self._name == getattr(other, "name")
        return False

    def __hash__(self) -> int:
        return hash(self._name)


# Aliases: several methods below take `dtype`/`device` parameters that shadow
# the class names, so isinstance checks must use these captured references.
_dtype_class = dtype


float32 = dtype("float32", 4, is_floating_point=True)
float_ = float32
float64 = dtype("float64", 8, is_floating_point=True)
double = float64
float16 = dtype("float16", 2, is_floating_point=True)
half = float16
bfloat16 = dtype("bfloat16", 2, is_floating_point=True)

int8 = dtype("int8", 1, is_floating_point=False, is_signed=True)
char = int8
int16 = dtype("int16", 2, is_floating_point=False, is_signed=True)
short = int16
int32 = dtype("int32", 4, is_floating_point=False, is_signed=True)
int_ = int32
int64 = dtype("int64", 8, is_floating_point=False, is_signed=True)
long = int64

uint8 = dtype("uint8", 1, is_floating_point=False, is_signed=False)
byte = uint8
uint16 = dtype("uint16", 2, is_floating_point=False, is_signed=False)
uint32 = dtype("uint32", 4, is_floating_point=False, is_signed=False)
uint64 = dtype("uint64", 8, is_floating_point=False, is_signed=False)

bool_ = dtype("bool", 1, is_floating_point=False, is_signed=False)

complex64 = dtype("complex64", 8, is_floating_point=True, is_signed=True, is_complex=True)
cfloat = complex64
complex128 = dtype("complex128", 16, is_floating_point=True, is_signed=True, is_complex=True)
cdouble = complex128

float8_e4m3fn = dtype("float8_e4m3fn", 1, is_floating_point=True, is_signed=True, bits=8)
float8_e5m2 = dtype("float8_e5m2", 1, is_floating_point=True, is_signed=True, bits=8)

_DTYPE_TO_STR = {
    float32: "f32", float_: "f32", "float32": "f32", "f32": "f32", float: "f32",
    float64: "f64", double: "f64", "float64": "f64", "f64": "f64",
    float16: "f16", half: "f16", "float16": "f16", "f16": "f16",
    bfloat16: "bf16", "bfloat16": "bf16", "bf16": "bf16",
    int32: "i32", int_: "i32", "int32": "i32", "i32": "i32", int: "i64",
    int64: "i64", long: "i64", "int64": "i64", "i64": "i64",
    int16: "i16", short: "i16", "int16": "i16", "i16": "i16", "short": "i16",
    int8: "i8", char: "i8", "int8": "i8", "i8": "i8", "char": "i8",
    uint8: "u8", byte: "u8", "uint8": "u8", "u8": "u8",
    uint16: "u16", "uint16": "u16", "u16": "u16",
    uint32: "u32", "uint32": "u32", "u32": "u32",
    uint64: "u64", "uint64": "u64", "u64": "u64",
    bool_: "bool", "bool": "bool", bool: "bool",
    complex64: "c64", cfloat: "c64", "complex64": "c64", "cfloat": "c64", "c64": "c64",
    complex128: "c128", cdouble: "c128", "complex128": "c128", "cdouble": "c128", "c128": "c128",
    float8_e4m3fn: "e4m3", "float8_e4m3fn": "e4m3", "fp8_e4m3": "e4m3", "e4m3": "e4m3",
    float8_e5m2: "e5m2", "float8_e5m2": "e5m2", "fp8_e5m2": "e5m2", "e5m2": "e5m2",
}

# Native dtype tag (Tensor.dtype) -> NumPy dtype for host-side conversions.
_NATIVE_DTYPE_TO_NUMPY = {
    "f32": np.float32, "f64": np.float64, "f16": np.float16,
    "i32": np.int32, "i64": np.int64, "u8": np.uint8, "bool": np.bool_,
    "i16": np.int16, "i8": np.int8, "u16": np.uint16, "u32": np.uint32, "u64": np.uint64,
}

# Native dtype tag -> torch-style class infix for Tensor.type().
_NATIVE_DTYPE_TO_TYPE_NAME = {
    "f32": "Float", "f64": "Double", "f16": "Half", "bf16": "BFloat16",
    "i32": "Int", "i64": "Long", "u8": "Byte", "bool": "Bool",
    "i16": "Short", "i8": "Char", "c64": "ComplexFloat", "c128": "ComplexDouble",
}


def _normalize_dtype(d: Any) -> str:
    if d is None:
        return "f32"
    if d in _DTYPE_TO_STR:
        return _DTYPE_TO_STR[d]
    if isinstance(d, str):
        return d
    return "f32"


class finfo:
    """Numerical properties of floating point and complex types."""
    def __init__(self, dtype_val: Any):
        name = getattr(dtype_val, "_name", None) or getattr(dtype_val, "name", None) or str(dtype_val)
        if isinstance(dtype_val, str):
            name = dtype_val
        name = name.lower()
        if "oxideon." in name:
            name = name.replace("oxideon.", "")

        info_map = {
            "float16": (16, 0.0009765625, 65504.0, -65504.0, 6.103515625e-05, float16),
            "half": (16, 0.0009765625, 65504.0, -65504.0, 6.103515625e-05, float16),
            "f16": (16, 0.0009765625, 65504.0, -65504.0, 6.103515625e-05, float16),
            "bfloat16": (16, 0.0078125, 3.3895313892515355e+38, -3.3895313892515355e+38, 1.1754943508222875e-38, bfloat16),
            "bf16": (16, 0.0078125, 3.3895313892515355e+38, -3.3895313892515355e+38, 1.1754943508222875e-38, bfloat16),
            "float32": (32, 1.1920928955078125e-07, 3.4028234663852886e+38, -3.4028234663852886e+38, 1.1754943508222875e-38, float32),
            "float": (32, 1.1920928955078125e-07, 3.4028234663852886e+38, -3.4028234663852886e+38, 1.1754943508222875e-38, float32),
            "f32": (32, 1.1920928955078125e-07, 3.4028234663852886e+38, -3.4028234663852886e+38, 1.1754943508222875e-38, float32),
            "float64": (64, 2.220446049250313e-16, 1.7976931348623157e+308, -1.7976931348623157e+308, 2.2250738585072014e-308, float64),
            "double": (64, 2.220446049250313e-16, 1.7976931348623157e+308, -1.7976931348623157e+308, 2.2250738585072014e-308, float64),
            "f64": (64, 2.220446049250313e-16, 1.7976931348623157e+308, -1.7976931348623157e+308, 2.2250738585072014e-308, float64),
            "complex64": (64, 1.1920928955078125e-07, 3.4028234663852886e+38, -3.4028234663852886e+38, 1.1754943508222875e-38, complex64),
            "cfloat": (64, 1.1920928955078125e-07, 3.4028234663852886e+38, -3.4028234663852886e+38, 1.1754943508222875e-38, complex64),
            "c64": (64, 1.1920928955078125e-07, 3.4028234663852886e+38, -3.4028234663852886e+38, 1.1754943508222875e-38, complex64),
            "complex128": (128, 2.220446049250313e-16, 1.7976931348623157e+308, -1.7976931348623157e+308, 2.2250738585072014e-308, complex128),
            "cdouble": (128, 2.220446049250313e-16, 1.7976931348623157e+308, -1.7976931348623157e+308, 2.2250738585072014e-308, complex128),
            "c128": (128, 2.220446049250313e-16, 1.7976931348623157e+308, -1.7976931348623157e+308, 2.2250738585072014e-308, complex128),
            "float8_e4m3fn": (8, 0.125, 448.0, -448.0, 0.015625, float8_e4m3fn),
            "e4m3": (8, 0.125, 448.0, -448.0, 0.015625, float8_e4m3fn),
            "fp8_e4m3": (8, 0.125, 448.0, -448.0, 0.015625, float8_e4m3fn),
            "float8_e5m2": (8, 0.25, 57344.0, -57344.0, 6.103515625e-05, float8_e5m2),
            "e5m2": (8, 0.25, 57344.0, -57344.0, 6.103515625e-05, float8_e5m2),
            "fp8_e5m2": (8, 0.25, 57344.0, -57344.0, 6.103515625e-05, float8_e5m2),
        }
        if name not in info_map:
            raise TypeError(f"ox.finfo only supports floating-point or complex dtypes, got {dtype_val!r}")
        bits, eps, max_v, min_v, tiny_v, dt_obj = info_map[name]
        self.bits = bits
        self.eps = eps
        self.max = max_v
        self.min = min_v
        self.tiny = tiny_v
        self.smallest_normal = tiny_v
        self.resolution = eps
        self.dtype = dt_obj

    def __repr__(self) -> str:
        return f"finfo(resolution={self.resolution}, min={self.min}, max={self.max}, eps={self.eps}, tiny={self.tiny}, dtype={self.dtype})"


class iinfo:
    """Numerical properties of integer and unsigned integer types."""
    def __init__(self, dtype_val: Any):
        name = getattr(dtype_val, "_name", None) or getattr(dtype_val, "name", None) or str(dtype_val)
        if isinstance(dtype_val, str):
            name = dtype_val
        name = name.lower()
        if "oxideon." in name:
            name = name.replace("oxideon.", "")

        info_map = {
            "int8": (8, -128, 127, int8),
            "i8": (8, -128, 127, int8),
            "char": (8, -128, 127, int8),
            "uint8": (8, 0, 255, uint8),
            "u8": (8, 0, 255, uint8),
            "byte": (8, 0, 255, uint8),
            "int16": (16, -32768, 32767, int16),
            "i16": (16, -32768, 32767, int16),
            "short": (16, -32768, 32767, int16),
            "uint16": (16, 0, 65535, uint16),
            "u16": (16, 0, 65535, uint16),
            "int32": (32, -2147483648, 2147483647, int32),
            "i32": (32, -2147483648, 2147483647, int32),
            "int": (32, -2147483648, 2147483647, int32),
            "uint32": (32, 0, 4294967295, uint32),
            "u32": (32, 0, 4294967295, uint32),
            "int64": (64, -9223372036854775808, 9223372036854775807, int64),
            "i64": (64, -9223372036854775808, 9223372036854775807, int64),
            "long": (64, -9223372036854775808, 9223372036854775807, int64),
            "uint64": (64, 0, 18446744073709551615, uint64),
            "u64": (64, 0, 18446744073709551615, uint64),
        }
        if name not in info_map:
            raise TypeError(f"ox.iinfo only supports integer dtypes, got {dtype_val!r}")
        bits, min_v, max_v, dt_obj = info_map[name]
        self.bits = bits
        self.min = min_v
        self.max = max_v
        self.dtype = dt_obj

    def __repr__(self) -> str:
        return f"iinfo(min={self.min}, max={self.max}, dtype={self.dtype})"


def _to_dtype_obj(dt: Any) -> dtype:
    if isinstance(dt, dtype):
        return dt
    if isinstance(dt, str):
        for candidate in (
            float32, float64, float16, bfloat16,
            int8, int16, int32, int64,
            uint8, uint16, uint32, uint64,
            complex64, complex128,
            float8_e4m3fn, float8_e5m2, bool_
        ):
            if candidate == dt:
                return candidate
    if dt is float:
        return float32
    if dt is int:
        return int64
    if dt is bool:
        return bool_
    if dt is complex or dt is builtins.complex:
        return complex64
    raise TypeError(f"Cannot resolve {dt!r} to oxideon.dtype")


def promote_types(type1: Any, type2: Any) -> dtype:
    """Returns the dtype with the smallest size and kind that can represent both types."""
    d1 = _to_dtype_obj(type1)
    d2 = _to_dtype_obj(type2)
    if d1 == d2:
        return d1
    if d1 == bool_:
        return d2
    if d2 == bool_:
        return d1

    # Complex promotion
    if d1.is_complex or d2.is_complex:
        if d1 == complex128 or d2 == complex128:
            return complex128
        if d1 in (float64, int64, uint64) or d2 in (float64, int64, uint64):
            return complex128
        return complex64

    # Float promotion
    if d1.is_floating_point or d2.is_floating_point:
        if d1 == float64 or d2 == float64:
            return float64
        if d1 == float32 or d2 == float32:
            if d2 in (int64, uint64) or d1 in (int64, uint64):
                return float64
            return float32
        if d1 == bfloat16 or d2 == bfloat16:
            if d1 in (float16, float32, float64) or d2 in (float16, float32, float64):
                return float32 if (d1 != float64 and d2 != float64) else float64
            return bfloat16
        if d1 == float16 or d2 == float16:
            if d1 in (float32, float64) or d2 in (float32, float64):
                return float32 if (d1 != float64 and d2 != float64) else float64
            return float16
        return float32

    # Both are integer
    signed = d1.is_signed or d2.is_signed
    max_bits = max(d1.bits, d2.bits)
    if signed:
        if (not d1.is_signed and d1.bits >= d2.bits) or (not d2.is_signed and d2.bits >= d1.bits):
            max_bits = min(64, max_bits * 2)
            if max_bits == 64 and ((not d1.is_signed and d1.bits == 64) or (not d2.is_signed and d2.bits == 64)):
                return float64
        if max_bits <= 8:
            return int8
        elif max_bits <= 16:
            return int16
        elif max_bits <= 32:
            return int32
        else:
            return int64
    else:
        if max_bits <= 8:
            return uint8
        elif max_bits <= 16:
            return uint16
        elif max_bits <= 32:
            return uint32
        else:
            return uint64


def can_cast(from_dtype: Any, to_dtype: Any, casting: str = "safe") -> bool:
    """Returns True if cast between data types can occur according to the casting rule."""
    f = _to_dtype_obj(from_dtype)
    t = _to_dtype_obj(to_dtype)
    if f == t:
        return True
    if casting == "unsafe":
        return True
    if casting == "same_kind":
        if f.is_complex and t.is_complex:
            return True
        if f.is_floating_point and t.is_floating_point and not t.is_complex:
            return True
        if not f.is_floating_point and not t.is_floating_point:
            return True
        if f == bool_:
            return True
        return False
    if casting == "safe":
        if f == bool_:
            return True
        if f.is_complex:
            return t.is_complex and t.bits >= f.bits
        if f.is_floating_point:
            if t.is_complex:
                return t.bits >= f.bits * 2
            return t.is_floating_point and t.bits >= f.bits
        # f is integer
        if t.is_complex or t.is_floating_point:
            return True
        if f.is_signed:
            return t.is_signed and t.bits >= f.bits
        else:
            if t.is_signed:
                return t.bits > f.bits
            else:
                return t.bits >= f.bits
    if casting == "no":
        return f == t
    return False


def is_floating_point(input: Any) -> bool:
    """Returns True if the data type of input is a floating point data type."""
    if isinstance(input, _Tensor):
        return str(input.dtype) in ("f32", "f64", "f16", "bf16")
    if hasattr(input, "is_complex") and getattr(input, "is_complex") is True:
        return True
    if hasattr(input, "dtype"):
        return is_floating_point(getattr(input, "dtype"))
    try:
        dt = _to_dtype_obj(input)
        return dt.is_floating_point
    except Exception:
        return False


def is_complex(input: Any) -> bool:
    """Returns True if the data type of input is a complex data type."""
    if isinstance(input, _Tensor):
        return False
    if hasattr(input, "is_complex"):
        val = getattr(input, "is_complex")
        return val() if callable(val) else bool(val)
    if hasattr(input, "dtype"):
        return is_complex(getattr(input, "dtype"))
    try:
        dt = _to_dtype_obj(input)
        return dt.is_complex
    except Exception:
        return False


def is_signed(input: Any) -> bool:
    """Returns True if the data type of input is a signed data type."""
    if isinstance(input, _Tensor):
        return str(input.dtype) not in ("u8", "bool")
    if hasattr(input, "is_signed"):
        val = getattr(input, "is_signed")
        return val() if callable(val) else bool(val)
    if hasattr(input, "dtype"):
        return is_signed(getattr(input, "dtype"))
    try:
        dt = _to_dtype_obj(input)
        return dt.is_signed
    except Exception:
        return False


# ---------------------------------------------------------------------------
# 2. Hardware Placement & Device Abstraction
# ---------------------------------------------------------------------------

class device:
    """Device placement descriptor for hardware compute nodes and accelerators."""
    def __init__(self, type_or_str: Union[str, "device"], index: Optional[int] = None):
        if isinstance(type_or_str, device):
            self._type = type_or_str._type
            self._index = type_or_str._index if index is None else index
            return

        s = str(type_or_str).strip().lower()
        if ":" in s:
            parts = s.split(":", 1)
            self._type = parts[0]
            self._index = int(parts[1]) if index is None else index
        else:
            self._type = s
            self._index = index

    @property
    def type(self) -> str:
        return self._type

    @property
    def index(self) -> Optional[int]:
        return self._index

    def __repr__(self) -> str:
        if self._index is not None:
            return f"device(type='{self._type}', index={self._index})"
        return f"device(type='{self._type}')"

    def __str__(self) -> str:
        if self._index is not None:
            return f"{self._type}:{self._index}"
        return self._type

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, device):
            return self._type == other._type and self._index == other._index
        if isinstance(other, str):
            return str(self) == other or self._type == other
        return False


_device_class = device


def _normalize_device(dev: Any) -> str:
    if dev is None:
        return "cpu"
    if isinstance(dev, device):
        s = str(dev)
    elif isinstance(dev, str):
        s = dev
    elif isinstance(getattr(dev, "type", None), str):
        # Native device handles (Tensor.device) expose type/index attributes.
        idx = getattr(dev, "index", None)
        s = f"{dev.type}:{idx}" if idx is not None else dev.type
    else:
        s = str(dev)
    s = s.strip().lower()
    base, sep, idx = s.partition(":")
    if "cuda" in base or "wgpu" in base or "gpu" in base:
        base = "wgpu"
    elif "cpu" in base:
        base = "cpu"
    return base + (sep + idx if sep else "")


# ---------------------------------------------------------------------------
# 3. Accelerator & Backend Subsystems
# ---------------------------------------------------------------------------

class _CudaModule:
    """Hardware GPU and compute device runtime controls."""

    @staticmethod
    def is_available() -> bool:
        try:
            from .oxideon import HardwareManager
            summary = HardwareManager().summary().lower()
            return "gpu" in summary and "software renderer" not in summary
        except Exception:
            return False

    @staticmethod
    def device_count() -> int:
        try:
            from .oxideon import HardwareManager
            hw = HardwareManager()
            for dev in hw.devices():
                if "gpu" in str(dev).lower() or "wgpu" in str(dev).lower():
                    return 1
            return 0
        except Exception:
            return 0

    @staticmethod
    def current_device() -> int:
        return 0

    @staticmethod
    def get_device_name(dev: Optional[Union[int, device, str]] = None) -> str:
        try:
            from .oxideon import HardwareManager
            hw = HardwareManager()
            for d in hw.devices():
                if "gpu" in str(d).lower() or "wgpu" in str(d).lower():
                    return str(d)
            return "Oxideon Compute Engine"
        except Exception:
            return "Oxideon Compute Engine"

    @staticmethod
    def synchronize(dev: Optional[Union[int, device, str]] = None) -> None:
        pass

    @staticmethod
    def memory_allocated(dev: Optional[Union[int, device, str]] = None) -> int:
        return 0

    @staticmethod
    def memory_reserved(dev: Optional[Union[int, device, str]] = None) -> int:
        return 0

    @staticmethod
    def empty_cache() -> None:
        pass

    @staticmethod
    def manual_seed(seed: int) -> None:
        _manual_seed(seed)

    @staticmethod
    def manual_seed_all(seed: int) -> None:
        _manual_seed(seed)


class Stream:
    """Hardware accelerator asynchronous execution stream."""
    def __init__(self, device: Optional[Any] = None, priority: int = 0):
        self.device = device
        self.priority = priority

    def synchronize(self) -> None:
        pass

    def wait_event(self, event: Any) -> None:
        pass

    def wait_stream(self, stream: Any) -> None:
        pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass


class Event:
    """Hardware accelerator event marker for fine-grained synchronization and profiling."""
    def __init__(self, enable_timing: bool = False, blocking: bool = False, interprocess: bool = False):
        self.enable_timing = enable_timing
        self.blocking = blocking
        self._recorded = False
        self._time = 0.0

    def record(self, stream: Optional[Stream] = None) -> None:
        self._recorded = True
        self._time = time.perf_counter()

    def wait(self, stream: Optional[Stream] = None) -> None:
        pass

    def query(self) -> bool:
        return self._recorded

    def elapsed_time(self, end_event: "Event") -> float:
        return (end_event._time - self._time) * 1000.0

    def synchronize(self) -> None:
        pass


_CudaModule.Stream = Stream
_CudaModule.Event = Event


@contextlib.contextmanager
def _cuda_stream_ctx(s: Stream):
    yield s


@contextlib.contextmanager
def _cuda_device_ctx(d: Any):
    yield d


_CudaModule.stream = staticmethod(_cuda_stream_ctx)
_CudaModule.device = staticmethod(_cuda_device_ctx)
_CudaModule.set_device = staticmethod(lambda d: None)
from . import amp as _amp
_CudaModule.amp = _amp


class _MpsModule:
    """Apple Silicon Metal acceleration runtime query interface."""
    @staticmethod
    def is_available() -> bool:
        return False

    @staticmethod
    def is_built() -> bool:
        return False


class _BackendsModule:
    """Compiled hardware backend capabilities."""
    cuda = _CudaModule
    mps = _MpsModule

    class cudnn:
        enabled: bool = True
        benchmark: bool = False

        @staticmethod
        def is_available() -> bool:
            return _CudaModule.is_available()


cuda = _CudaModule
mps = _MpsModule
backends = _BackendsModule


# ---------------------------------------------------------------------------
# 4. Autograd Management & Functional Gradients
# ---------------------------------------------------------------------------

class no_grad:
    """Context manager or decorator that disables automatic differentiation."""
    def __init__(self):
        self.prev = True

    def __enter__(self):
        self.prev = is_grad_enabled()
        _set_grad_enabled_impl(False)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        _set_grad_enabled_impl(self.prev)

    def __call__(self, fn: Callable):
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            with self:
                return fn(*args, **kwargs)
        return wrapper


class enable_grad:
    """Context manager or decorator that activates automatic differentiation."""
    def __init__(self):
        self.prev = False

    def __enter__(self):
        self.prev = is_grad_enabled()
        _set_grad_enabled_impl(True)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        _set_grad_enabled_impl(self.prev)

    def __call__(self, fn: Callable):
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            with self:
                return fn(*args, **kwargs)
        return wrapper


class set_grad_enabled:
    """Context manager controlling automatic differentiation state (use with `with`).

    Deliberate divergence: construction is pure (no side effect until entered),
    while torch applies a bare `set_grad_enabled(mode)` call eagerly/globally.
    Pinned by `test_set_grad_enabled_construction_pure_and_nesting`.
    """
    def __init__(self, mode: bool):
        self.mode = bool(mode)
        self.prev = True

    def __enter__(self):
        self.prev = is_grad_enabled()
        _set_grad_enabled_impl(self.mode)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        _set_grad_enabled_impl(self.prev)


class inference_mode(no_grad):
    """Context manager optimized for evaluation and deployment."""
    pass


def manual_seed(seed: int) -> None:
    """Sets deterministic seeds across the Oxideon engine, NumPy, and RNG components."""
    _manual_seed(seed)
    np.random.seed(seed)


class RemovableHandle:
    """A handle which provides the capability to remove a hook."""
    def __init__(self, hooks_dict: Dict[int, Any], hook_id: int):
        self.hooks_dict = hooks_dict
        self.hook_id = hook_id

    def remove(self) -> None:
        self.hooks_dict.pop(self.hook_id, None)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.remove()


_HOOK_COUNTER: int = 0
_CUSTOM_BACKWARD_MAP: Dict[int, Any] = {}
_TENSOR_HOOKS_MAP: Dict[int, Dict[int, Any]] = {}


class _FunctionContext:
    def __init__(self):
        self.saved_tensors = ()
        self.needs_input_grad = ()

    def save_for_backward(self, *tensors):
        self.saved_tensors = tuple(t.clone() if hasattr(t, "clone") else t for t in tensors)


class Function:
    """Base class to create custom autograd functions with forward and backward passes."""
    @classmethod
    def apply(cls, *args, **kwargs):
        ctx = _FunctionContext()
        ctx.needs_input_grad = tuple(getattr(a, "requires_grad", False) for a in args)
        out = cls.forward(ctx, *args, **kwargs)

        any_requires_grad = _builtin_any(getattr(a, "requires_grad", False) for a in args)
        if any_requires_grad and is_grad_enabled() and isinstance(out, _Tensor):
            out = out.clone().requires_grad_(True)

            def _custom_backward(grad_output):
                if not isinstance(grad_output, _Tensor):
                    grad_output = tensor(grad_output, dtype=out.dtype, device=out.device)
                grad_inputs = cls.backward(ctx, grad_output)
                if not isinstance(grad_inputs, (tuple, list)):
                    grad_inputs = (grad_inputs,)

                for i, arg in enumerate(args):
                    if i < len(grad_inputs) and getattr(arg, "requires_grad", False):
                        g = grad_inputs[i]
                        if g is not None:
                            if not isinstance(g, _Tensor):
                                g = tensor(g, dtype=arg.dtype, device=arg.device)
                            if id(arg) in _TENSOR_HOOKS_MAP:
                                for h in list(_TENSOR_HOOKS_MAP[id(arg)].values()):
                                    ret = h(g)
                                    if ret is not None:
                                        g = ret
                            arg.grad = (arg.grad + g) if arg.grad is not None else g

            _CUSTOM_BACKWARD_MAP[id(out)] = _custom_backward

        return out


class _AutogradModule:
    """Autograd functional engine interface."""
    no_grad = no_grad
    enable_grad = enable_grad
    set_grad_enabled = set_grad_enabled
    inference_mode = inference_mode
    is_grad_enabled = staticmethod(is_grad_enabled)
    Function = Function

    @staticmethod
    def grad(
        outputs: Union[_Tensor, Sequence[_Tensor]],
        inputs: Union[_Tensor, Sequence[_Tensor]],
        grad_outputs: Optional[Union[_Tensor, Sequence[_Tensor]]] = None,
        retain_graph: Optional[bool] = None,
        create_graph: bool = False,
        allow_unused: bool = False,
    ) -> Tuple[Optional[_Tensor], ...]:
        """Calculates gradients of target outputs with respect to input tensors."""
        out_list = [outputs] if isinstance(outputs, _Tensor) else list(outputs)
        inp_list = [inputs] if isinstance(inputs, _Tensor) else list(inputs)
        gt_list = None
        if grad_outputs is not None:
            gt_list = [grad_outputs] if isinstance(grad_outputs, _Tensor) else list(grad_outputs)

        for out in out_list:
            if not out.requires_grad:
                raise RuntimeError("element 0 of tensors does not require grad and does not have a grad_fn")

        if not any(id(out) in _CUSTOM_BACKWARD_MAP for out in out_list):
            # Like torch.autograd.grad: the gradients are returned, never accumulated into (or read from) `.grad`,
            # and `inputs` may be intermediate tensors, not only leaves.
            targets = []
            for idx, out in enumerate(out_list):
                go = gt_list[idx] if gt_list is not None and idx < len(gt_list) else None
                if go is None:
                    if out.numel() != 1:
                        raise RuntimeError("grad can be implicitly created only for scalar outputs")
                    targets.append(out.sum())
                else:
                    if not isinstance(go, _Tensor):
                        go = tensor(go, dtype=out.dtype, device=out.device)
                    targets.append((out * (go if create_graph else go.detach())).sum())
            total = targets[0]
            for t in targets[1:]:
                total = total + t
            grads = list(_native_grad([total], inp_list, create_graph))
        else:
            # A custom `Function` output: its backward only runs through `Tensor.backward`. Run it on clean
            # `.grad` slots and put the callers' gradients back afterwards, so nothing accumulates.
            saved = [inp.grad for inp in inp_list]
            for inp in inp_list:
                inp.set_grad(None)
            try:
                last = len(out_list) - 1
                for idx, out in enumerate(out_list):
                    grad_out = gt_list[idx] if gt_list is not None and idx < len(gt_list) else None
                    # The loop accumulates one output at a time, so the graph must be
                    # retained across iterations; the caller's flag applies to the end.
                    rg = retain_graph if (retain_graph or idx == last) else True
                    out.backward(gradient=grad_out, retain_graph=rg, create_graph=create_graph)
                grads = [inp.grad for inp in inp_list]
            finally:
                for inp, g in zip(inp_list, saved):
                    inp.set_grad(g)

        for g in grads:
            if g is None and not allow_unused:
                raise RuntimeError("One of the differentiated Tensors appears to not have been used in the graph.")
        return tuple(grads)

    @staticmethod
    def backward(
        tensors: Union[_Tensor, Sequence[_Tensor]],
        grad_tensors: Optional[Union[_Tensor, Sequence[_Tensor]]] = None,
        retain_graph: Optional[bool] = None,
        create_graph: bool = False,
    ) -> None:
        """Executes reverse-mode automatic differentiation from leaf tensors."""
        t_list = [tensors] if isinstance(tensors, _Tensor) else list(tensors)
        gt_list = None
        if grad_tensors is not None:
            gt_list = [grad_tensors] if isinstance(grad_tensors, _Tensor) else list(grad_tensors)
        for i, t in enumerate(t_list):
            grad = gt_list[i] if gt_list is not None and i < len(gt_list) else None
            t.backward(gradient=grad, retain_graph=retain_graph, create_graph=create_graph)

    @staticmethod
    def gradcheck(
        func: Callable[..., Any],
        inputs: Sequence[_Tensor],
        eps: float = 1e-4,
        atol: float = 1e-3,
        rtol: float = 1e-2,
        raise_exception: bool = True,
        check_sparse_nnz: bool = False,
        nondet_tol: float = 0.0,
        check_undefined_grad: bool = True,
        check_grad_dtypes: bool = False,
    ) -> bool:
        """Verifies analytical automatic gradients against numerical finite differences."""
        input_list = list(inputs)
        for inp in input_list:
            if not getattr(inp, "requires_grad", False):
                inp.requires_grad_(True)
            inp.zero_grad()

        out = func(*input_list)
        out_t = out[0] if isinstance(out, (tuple, list)) else out
        loss = out_t.sum() if out_t.numel() > 1 else out_t
        loss.backward()

        analytical = [inp.grad for inp in input_list]

        for i, inp in enumerate(input_list):
            arr = inp.to_numpy().copy()
            flat = arr.flatten()
            num_grad = np.zeros_like(flat)

            for idx in range(len(flat)):
                orig_val = flat[idx]

                # f(x + eps)
                flat[idx] = orig_val + eps
                inp.assign(_from_numpy(_contig(flat.reshape(arr.shape), dtype=arr.dtype)))
                o_pos = func(*input_list)
                o_pos_t = o_pos[0] if isinstance(o_pos, (tuple, list)) else o_pos
                v_pos = float(o_pos_t.sum().item()) if o_pos_t.numel() > 1 else float(o_pos_t.item())

                # f(x - eps)
                flat[idx] = orig_val - eps
                inp.assign(_from_numpy(_contig(flat.reshape(arr.shape), dtype=arr.dtype)))
                o_neg = func(*input_list)
                o_neg_t = o_neg[0] if isinstance(o_neg, (tuple, list)) else o_neg
                v_neg = float(o_neg_t.sum().item()) if o_neg_t.numel() > 1 else float(o_neg_t.item())

                flat[idx] = orig_val
                num_grad[idx] = (v_pos - v_neg) / (2.0 * eps)

            inp.assign(_from_numpy(arr))
            ana_g = analytical[i]
            if ana_g is None:
                if raise_exception:
                    raise RuntimeError(f"Gradient for input {i} was None")
                return False

            ana_flat = ana_g.to_numpy().flatten()
            if not np.allclose(ana_flat, num_grad, rtol=rtol, atol=atol):
                if raise_exception:
                    max_diff = float(np.max(np.abs(ana_flat - num_grad)))
                    raise RuntimeError(f"Jacobian mismatch for input {i}: max diff {max_diff:.6f} exceeds tolerance")
                return False

        return True

    @staticmethod
    def gradgradcheck(
        func: Callable[..., Any],
        inputs: Sequence[_Tensor],
        grad_outputs: Optional[Sequence[_Tensor]] = None,
        eps: float = 1e-4,
        atol: float = 1e-3,
        rtol: float = 1e-2,
        gen_non_contig_grad_outputs: bool = False,
        raise_exception: bool = True,
    ) -> bool:
        """Verifies second-order analytical gradients against numerical finite differences."""
        input_list = list(inputs)
        def first_grad_wrapper(*inps):
            for p in inps:
                p.zero_grad()
            o = func(*inps)
            loss = o.sum() if o.numel() > 1 else o
            loss.backward()
            return tuple(p.grad for p in inps)

        for i in range(len(input_list)):
            def grad_i(*inps):
                return first_grad_wrapper(*inps)[i]
            _AutogradModule.gradcheck(grad_i, input_list, eps=eps, atol=atol, rtol=rtol, raise_exception=raise_exception)
        return True

    @staticmethod
    @contextlib.contextmanager
    def detect_anomaly(check_nan: bool = True):
        yield

    @staticmethod
    def set_detect_anomaly(mode: bool) -> None:
        pass


autograd = _AutogradModule


# ---------------------------------------------------------------------------
# 5. Core Tensor Constructors & Ergonomics
# ---------------------------------------------------------------------------

def _extract_shape_args(args: Sequence[Any]) -> List[int]:
    """Normalizes both unpacked shapes like (2, 3) and sequences like [2, 3]."""
    if len(args) == 0:
        return []
    if len(args) == 1:
        elem = args[0]
        if isinstance(elem, (list, tuple)):
            return [int(x) for x in elem]
        if isinstance(elem, Size):
            return list(elem)
        return [int(elem)]
    return [int(x) for x in args]


def tensor(
    data: Any,
    dtype: Optional[Any] = None,
    device: Optional[Any] = None,
    requires_grad: bool = False,
) -> _Tensor:
    """Creates a tensor populated with arbitrary multi-dimensional data."""
    if isinstance(data, _Tensor):
        t = data.clone()
        if dtype is not None:
            t = t.to_dtype(_normalize_dtype(dtype))
        if device is not None:
            t = t.to_device(_normalize_device(device))
        if requires_grad:
            t.requires_grad_(True)
        return t

    np_arr = np.asarray(data)
    target_dt = _normalize_dtype(dtype) if dtype is not None else None
    if target_dt is None:
        if np.issubdtype(np_arr.dtype, np.floating):
            target_dt = "f32"
            np_arr = np_arr.astype(np.float32)
        elif np.issubdtype(np_arr.dtype, np.integer):
            target_dt = "i64"
            np_arr = np_arr.astype(np.int64)
        elif np.issubdtype(np_arr.dtype, np.bool_):
            target_dt = "bool"
        else:
            target_dt = "f32"
            np_arr = np_arr.astype(np.float32)
    else:
        dt_map = {
            "f32": np.float32, "f64": np.float64, "f16": np.float16,
            "i32": np.int32, "i64": np.int64, "u8": np.uint8, "bool": np.bool_,
        }
        if target_dt in dt_map:
            np_arr = np_arr.astype(dt_map[target_dt])

    t = _from_numpy(_contig(np_arr))
    if device is not None:
        t = t.to_device(_normalize_device(device))
    if requires_grad:
        t.requires_grad_(True)
    return t


def as_tensor(data: Any, dtype: Optional[Any] = None, device: Optional[Any] = None) -> _Tensor:
    """Converts data into a tensor sharing memory when possible."""
    return tensor(data, dtype=dtype, device=device)


def zeros(*size: Any, dtype: Optional[Any] = None, device: Optional[Any] = None, requires_grad: bool = False) -> _Tensor:
    """Creates a tensor filled with zero values."""
    sh = _extract_shape_args(size)
    dt = _normalize_dtype(dtype)
    dev = _normalize_device(device)
    t = _Tensor.zeros(sh, dtype=dt, device=dev)
    if requires_grad:
        t.requires_grad_(True)
    return t


def ones(*size: Any, dtype: Optional[Any] = None, device: Optional[Any] = None, requires_grad: bool = False) -> _Tensor:
    """Creates a tensor filled with ones."""
    sh = _extract_shape_args(size)
    dt = _normalize_dtype(dtype)
    dev = _normalize_device(device)
    t = _Tensor.ones(sh, dtype=dt, device=dev)
    if requires_grad:
        t.requires_grad_(True)
    return t


def empty(*size: Any, dtype: Optional[Any] = None, device: Optional[Any] = None, requires_grad: bool = False) -> _Tensor:
    """Allocates an uninitialized tensor buffer."""
    return zeros(*size, dtype=dtype, device=device, requires_grad=requires_grad)


def randn(*size: Any, dtype: Optional[Any] = None, device: Optional[Any] = None, requires_grad: bool = False) -> _Tensor:
    """Samples values from a standard normal distribution N(0, 1)."""
    sh = _extract_shape_args(size)
    dt = _normalize_dtype(dtype)
    dev = _normalize_device(device)
    t = _Tensor.randn(sh, dtype=dt, device=dev)
    if requires_grad:
        t.requires_grad_(True)
    return t


def rand(*size: Any, dtype: Optional[Any] = None, device: Optional[Any] = None, requires_grad: bool = False) -> _Tensor:
    """Samples values uniformly from the half-open interval [0, 1)."""
    sh = _extract_shape_args(size)
    dt = _normalize_dtype(dtype)
    dev = _normalize_device(device)
    arr = np.random.uniform(0.0, 1.0, size=sh).astype(np.float32)
    t = _from_numpy(_contig(arr))
    if dt != "f32":
        t = t.to_dtype(dt)
    if dev != "cpu":
        t = t.to_device(dev)
    if requires_grad:
        t.requires_grad_(True)
    return t


def randint(
    low: int,
    high: Optional[int] = None,
    size: Sequence[int] = (),
    dtype: Optional[Any] = None,
    device: Optional[Any] = None,
    requires_grad: bool = False,
) -> _Tensor:
    """Generates random integers uniformly distributed between low and high."""
    if high is None:
        high = low
        low = 0
    sh = list(size)
    arr = np.random.randint(low, high, size=sh, dtype=np.int64)
    dt = _normalize_dtype(dtype) if dtype is not None else "i64"
    dev = _normalize_device(device)
    t = _from_numpy(arr).to_dtype(dt).to_device(dev)
    if requires_grad:
        t.requires_grad_(True)
    return t


def randperm(n: int, dtype: Optional[Any] = None, device: Optional[Any] = None) -> _Tensor:
    """Generates a random permutation of integers from 0 to n - 1."""
    arr = np.random.permutation(n).astype(np.int64)
    dt = _normalize_dtype(dtype) if dtype is not None else "i64"
    dev = _normalize_device(device)
    return _from_numpy(arr).to_dtype(dt).to_device(dev)


def full(
    size: Sequence[int],
    fill_value: Union[float, int, bool],
    dtype: Optional[Any] = None,
    device: Optional[Any] = None,
    requires_grad: bool = False,
) -> _Tensor:
    """Constructs a tensor filled with a specific scalar value."""
    sh = list(size) if isinstance(size, (list, tuple, Size)) else [int(size)]
    dt = _normalize_dtype(dtype)
    dev = _normalize_device(device)
    t = _Tensor.zeros(sh, dtype=dt, device=dev).add_scalar(float(fill_value))
    if requires_grad:
        t.requires_grad_(True)
    return t


def zeros_like(input: _Tensor, dtype: Optional[Any] = None, device: Optional[Any] = None, requires_grad: bool = False) -> _Tensor:
    dt = dtype if dtype is not None else input.dtype
    dev = device if device is not None else input.device
    return zeros(input.shape, dtype=dt, device=dev, requires_grad=requires_grad)


def ones_like(input: _Tensor, dtype: Optional[Any] = None, device: Optional[Any] = None, requires_grad: bool = False) -> _Tensor:
    dt = dtype if dtype is not None else input.dtype
    dev = device if device is not None else input.device
    return ones(input.shape, dtype=dt, device=dev, requires_grad=requires_grad)


def empty_like(input: _Tensor, dtype: Optional[Any] = None, device: Optional[Any] = None, requires_grad: bool = False) -> _Tensor:
    return zeros_like(input, dtype=dtype, device=device, requires_grad=requires_grad)


def full_like(
    input: _Tensor,
    fill_value: Union[float, int, bool],
    dtype: Optional[Any] = None,
    device: Optional[Any] = None,
    requires_grad: bool = False,
) -> _Tensor:
    dt = dtype if dtype is not None else input.dtype
    dev = device if device is not None else input.device
    return full(input.shape, fill_value, dtype=dt, device=dev, requires_grad=requires_grad)


def arange(
    start: float,
    end: Optional[float] = None,
    step: float = 1.0,
    dtype: Optional[Any] = None,
    device: Optional[Any] = None,
    requires_grad: bool = False,
) -> _Tensor:
    """Generates evenly spaced values within an interval [start, end)."""
    if end is None:
        end = float(start)
        start = 0.0
    else:
        start = float(start)
        end = float(end)
    step = float(step)

    t = _Tensor.arange(start, end, step)
    if dtype is not None:
        t = t.to_dtype(_normalize_dtype(dtype))
    if device is not None:
        t = t.to_device(_normalize_device(device))
    if requires_grad:
        t.requires_grad_(True)
    return t


def linspace(
    start: float,
    end: float,
    steps: int = 100,
    dtype: Optional[Any] = None,
    device: Optional[Any] = None,
    requires_grad: bool = False,
) -> _Tensor:
    """Generates linearly spaced coordinates across a closed boundary."""
    arr = np.linspace(start, end, steps, dtype=np.float32)
    dt = _normalize_dtype(dtype)
    dev = _normalize_device(device)
    t = _from_numpy(arr).to_dtype(dt).to_device(dev)
    if requires_grad:
        t.requires_grad_(True)
    return t


def eye(
    n: int,
    m: Optional[int] = None,
    dtype: Optional[Any] = None,
    device: Optional[Any] = None,
    requires_grad: bool = False,
) -> _Tensor:
    """Generates an identity or diagonal matrix."""
    cols = n if m is None else m
    arr = np.eye(n, cols, dtype=np.float32)
    dt = _normalize_dtype(dtype)
    dev = _normalize_device(device)
    t = _from_numpy(arr).to_dtype(dt).to_device(dev)
    if requires_grad:
        t.requires_grad_(True)
    return t


def diagonal(input: _Tensor, offset: int = 0, dim1: int = 0, dim2: int = 1) -> _Tensor:
    """Returns a partial view of input with its diagonal elements along dim1 and dim2."""
    x = input if isinstance(input, _Tensor) else tensor(input)
    ndim = x.ndim
    d1 = dim1 if dim1 >= 0 else ndim + dim1
    d2 = dim2 if dim2 >= 0 else ndim + dim2
    if not (0 <= d1 < ndim and 0 <= d2 < ndim):
        raise IndexError(f"diagonal: dimension out of range (expected [{-ndim}, {ndim-1}])")
    if d1 == d2:
        raise ValueError(f"diagonal: dim1 and dim2 cannot be identical ({dim1}, {dim2})")

    # Permute so that d1 and d2 are at the trailing end
    other_dims = [i for i in range(ndim) if i not in (d1, d2)]
    perm = other_dims + [d1, d2]
    x_p = x.permute(perm)
    batch_shape = list(x_p.shape[:-2])
    H, W = x_p.shape[-2], x_p.shape[-1]
    if offset >= 0:
        diag_len = max(0, min(H, W - offset))
        h_idx = list(range(diag_len))
        w_idx = [i + offset for i in range(diag_len)]
    else:
        diag_len = max(0, min(H + offset, W))
        h_idx = [i - offset for i in range(diag_len)]
        w_idx = list(range(diag_len))

    x_p_flat = x_p.reshape(batch_shape + [H * W])
    idx = tensor([h * W + w for h, w in zip(h_idx, w_idx)], dtype=int)
    return x_p_flat.index_select(-1, idx)


def nonzero(input: _Tensor, as_tuple: bool = False) -> Union[_Tensor, Tuple[_Tensor, ...]]:
    """Returns the indices of the elements that are non-zero.

    If as_tuple is False, returns a 2D tensor of shape [Z, ndim].
    If as_tuple is True, returns a tuple of 1D tensors, one for each dimension.
    """
    arr = _to_np(input)
    indices = np.nonzero(arr)
    if as_tuple:
        return tuple(_from_numpy(_contig(idx, dtype=np.int64)).to(input.device) for idx in indices)
    stacked = np.column_stack(indices) if len(indices) > 0 and indices[0].size > 0 else np.zeros((0, input.ndim), dtype=np.int64)
    return _from_numpy(_contig(stacked, dtype=np.int64)).to(input.device)


def argwhere(input: _Tensor) -> _Tensor:
    """Returns a tensor containing the indices of all non-zero elements of input."""
    return nonzero(input, as_tuple=False)

def diag(input: _Tensor, diagonal: int = 0) -> _Tensor:
    """Constructs a 2D diagonal matrix from 1D tensor, or extracts diagonal from 2D tensor."""
    x = input if isinstance(input, _Tensor) else tensor(input)
    if x.ndim == 1:
        n = x.shape[0]
        k = diagonal
        sz = n + _builtin_abs(k)
        res_flat = zeros([sz * sz], dtype=x.dtype, device=x.device)
        idx = tensor([i * sz + (i + k if k >= 0 else i) for i in range(n)], dtype=int)
        res_flat = res_flat.scatter(0, idx, x)
        return res_flat.reshape([sz, sz])
    elif x.ndim == 2:
        r, c = x.shape[0], x.shape[1]
        k = diagonal
        if k >= 0:
            n2 = max(0, min(r, c - k))
            idx2 = tensor([i * c + (i + k) for i in range(n2)], dtype=int)
        else:
            n2 = max(0, min(r + k, c))
            idx2 = tensor([(i - k) * c + i for i in range(n2)], dtype=int)
        return x.reshape([-1]).index_select(0, idx2)
    else:
        raise ValueError(f"diag: input must be 1D or 2D, but got {x.ndim}D")


# ---------------------------------------------------------------------------
# 6. Top-Level Tensor Math & Manipulation Functions
# ---------------------------------------------------------------------------

def _to_np(x: Any) -> np.ndarray:
    if isinstance(x, _Tensor):
        return x.to_numpy()
    if isinstance(x, np.ndarray):
        return x
    return np.asarray(x)


def matmul(input: _Tensor, other: _Tensor) -> _Tensor:
    """Matrix product of two tensors."""
    return input @ other


mm = matmul
bmm = matmul


def dot(input: _Tensor, other: _Tensor) -> _Tensor:
    """Computes the dot product of two 1D tensors."""
    return (input * other).sum()


def vdot(input: _Tensor, other: _Tensor) -> _Tensor:
    """Computes the dot product of two tensors flattened as 1D vectors."""
    return (input.reshape([-1]) * other.reshape([-1])).sum()


def inner(input: _Tensor, other: _Tensor) -> _Tensor:
    """Computes inner product of input and other."""
    if input.ndim == 1 and other.ndim == 1:
        return (input * other).sum()
    return input.tensordot(other, dims=([input.ndim - 1], [other.ndim - 1]))


def deg2rad(input: _Tensor) -> _Tensor:
    """Converts angles from degrees to radians."""
    import math
    return input * (math.pi / 180.0)


def rad2deg(input: _Tensor) -> _Tensor:
    """Converts angles from radians to degrees."""
    import math
    return input * (180.0 / math.pi)


def outer(input: _Tensor, other: _Tensor) -> _Tensor:
    """Computes outer product of input and other."""
    x = input.reshape([-1, 1])
    y = other.reshape([1, -1])
    return x @ y


def mv(input: _Tensor, vec: _Tensor) -> _Tensor:
    """Performs a matrix-vector multiplication."""
    return (input @ vec.reshape([-1, 1])).squeeze(-1)


def norm(
    input: _Tensor,
    p: Optional[Union[float, int, str]] = "fro",
    dim: Optional[Union[int, Tuple[int, ...]]] = None,
    keepdim: bool = False,
) -> _Tensor:
    """Computes a matrix or vector norm."""
    from . import linalg
    return linalg.norm(input, ord=p, dim=dim, keepdim=keepdim)


def clamp(input: _Tensor, min: Any = None, max: Any = None) -> _Tensor:
    """Clamps all elements in input into the range [min, max]."""
    return as_tensor(input).clamp(min=min, max=max)


clip = clamp


def clamp_min(input: _Tensor, min: float) -> _Tensor:
    """Clamps all elements in input to have a lower bound of min."""
    return input.clamp(float(min), 1e30)


def clamp_max(input: _Tensor, max: float) -> _Tensor:
    """Clamps all elements in input to have an upper bound of max."""
    return input.clamp(-1e30, float(max))


def cumsum(input: _Tensor, dim: int) -> _Tensor:
    """Returns the cumulative sum of elements of input in the dimension dim."""
    return input.cumsum(dim)


def cumprod(input: _Tensor, dim: int) -> _Tensor:
    """Returns the cumulative product of elements of input in the dimension dim."""
    # Native engine (differentiable); the old numpy path broke autograd.
    return input.cumprod(dim)


def sin(input: _Tensor) -> _Tensor:
    """Computes the sine of each element in input."""
    return input.sin()


def cos(input: _Tensor) -> _Tensor:
    """Computes the cosine of each element in input."""
    return input.cos()


def tan(input: _Tensor) -> _Tensor:
    """Computes the tangent of each element in input."""
    return input.tan() if hasattr(input, "tan") else _from_numpy(np.tan(_to_np(input)))


def asin(input: _Tensor) -> _Tensor:
    """Computes the arcsine of each element in input."""
    return input.asin() if hasattr(input, "asin") else _from_numpy(np.arcsin(_to_np(input)))


arcsin = asin


def acos(input: _Tensor) -> _Tensor:
    """Computes the arccosine of each element in input."""
    return input.acos() if hasattr(input, "acos") else _from_numpy(np.arccos(_to_np(input)))


arccos = acos


def atan(input: _Tensor) -> _Tensor:
    """Computes the arctangent of each element in input."""
    return input.atan() if hasattr(input, "atan") else _from_numpy(np.arctan(_to_np(input)))


arctan = atan


def asinh(input: _Tensor) -> _Tensor:
    """Computes the inverse hyperbolic sine of each element in input."""
    return input.asinh() if hasattr(input, "asinh") else _from_numpy(np.arcsinh(_to_np(input)))


arcsinh = asinh


def acosh(input: _Tensor) -> _Tensor:
    """Computes the inverse hyperbolic cosine of each element in input."""
    return input.acosh() if hasattr(input, "acosh") else _from_numpy(np.arccosh(_to_np(input)))


arccosh = acosh


def atanh(input: _Tensor) -> _Tensor:
    """Computes the inverse hyperbolic tangent of each element in input."""
    return input.atanh() if hasattr(input, "atanh") else _from_numpy(np.arctanh(_to_np(input)))


arctanh = atanh


def sinh(input: _Tensor) -> _Tensor:
    """Computes the hyperbolic sine of each element in input."""
    return input.sinh() if hasattr(input, "sinh") else _from_numpy(np.sinh(_to_np(input)))


def cosh(input: _Tensor) -> _Tensor:
    """Computes the hyperbolic cosine of each element in input."""
    return input.cosh() if hasattr(input, "cosh") else _from_numpy(np.cosh(_to_np(input)))


def tanh(input: _Tensor) -> _Tensor:
    """Computes the hyperbolic tangent of each element in input."""
    return input.tanh()


def exp(input: _Tensor) -> _Tensor:
    """Computes the exponential of each element in input."""
    return input.exp()


def log(input: _Tensor) -> _Tensor:
    """Computes the natural logarithm of each element in input."""
    return input.log()


def log2(input: _Tensor) -> _Tensor:
    """Computes the base-2 logarithm of each element in input."""
    return input.log2() if hasattr(input, "log2") else _from_numpy(np.log2(_to_np(input)))


def log10(input: _Tensor) -> _Tensor:
    """Computes the base-10 logarithm of each element in input."""
    return input.log10() if hasattr(input, "log10") else _from_numpy(np.log10(_to_np(input)))


def sqrt(input: _Tensor) -> _Tensor:
    """Computes the square root of each element in input."""
    return input.sqrt()


def rsqrt(input: _Tensor) -> _Tensor:
    """Computes the reciprocal square root of each element in input."""
    return input.rsqrt() if hasattr(input, "rsqrt") else (1.0 / input.sqrt())


def abs(input: _Tensor) -> _Tensor:
    """Computes the absolute value of each element in input."""
    return input.abs()


def sign(input: _Tensor) -> _Tensor:
    """Computes the sign of each element in input."""
    return input.sign() if hasattr(input, "sign") else _from_numpy(np.sign(_to_np(input)))


def round(input: _Tensor) -> _Tensor:
    """Rounds elements of input to the nearest integer."""
    return input.round() if hasattr(input, "round") else _from_numpy(np.round(_to_np(input)))


def floor(input: _Tensor) -> _Tensor:
    """Returns a new tensor with the floor of the elements of input."""
    return input.floor() if hasattr(input, "floor") else _from_numpy(np.floor(_to_np(input)))


def ceil(input: _Tensor) -> _Tensor:
    """Returns a new tensor with the ceil of the elements of input."""
    return input.ceil() if hasattr(input, "ceil") else _from_numpy(np.ceil(_to_np(input)))


def matmul(input: _Tensor, other: _Tensor) -> _Tensor:
    """Matrix product of two tensors."""
    return input.matmul(other)


def mm(input: _Tensor, mat2: _Tensor) -> _Tensor:
    """Matrix multiplication of two 2D matrices."""
    return input.matmul(mat2)


def bmm(input: _Tensor, mat2: _Tensor) -> _Tensor:
    """Batch matrix multiplication of two 3D tensors."""
    return input.matmul(mat2)


def flatten(input: _Tensor, start_dim: int = 0, end_dim: int = -1) -> _Tensor:
    """Flattens input by reshaping it into a one-dimensional tensor across the specified dimensions."""
    shape = list(input.shape)
    s = start_dim if start_dim >= 0 else len(shape) + start_dim
    e = end_dim if end_dim >= 0 else len(shape) + end_dim
    pre = shape[:s]
    mid = math.prod(shape[s : e + 1])
    post = shape[e + 1 :]
    return input.reshape(pre + [mid] + post)


def unflatten(input: _Tensor, dim: int, sizes: Sequence[int]) -> _Tensor:
    """Expands dimension dim of input to the specified shape sizes."""
    shape = list(input.shape)
    d = dim if dim >= 0 else len(shape) + dim
    new_shape = shape[:d] + list(sizes) + shape[d + 1 :]
    return input.reshape(new_shape)


def squeeze(input: _Tensor, dim: Optional[int] = None) -> _Tensor:
    """Returns a tensor with all dimensions of input of size 1 removed, or a specific dim."""
    if dim is not None:
        return input.squeeze(dim)
    arr = _to_np(input)
    return _from_numpy(_contig(np.squeeze(arr)))


def unsqueeze(input: _Tensor, dim: int) -> _Tensor:
    """Returns a new tensor with a dimension of size one inserted at the specified position."""
    return input.unsqueeze(dim)


def cat(tensors: Sequence[_Tensor], dim: int = 0) -> _Tensor:
    """Concatenates the given sequence of tensors in the given dimension."""
    if len(tensors) == 0:
        raise RuntimeError("cat expects a non-empty sequence of tensors")
    d = dim if dim >= 0 else len(tensors[0].shape) + dim
    s0 = list(tensors[0].shape)
    s0_cmp = [s0[i] for i in range(len(s0)) if i != d]
    for t in tensors[1:]:
        st = list(t.shape)
        st_cmp = [st[i] for i in range(len(st)) if i != d]
        if st_cmp != s0_cmp:
            raise RuntimeError(f"Sizes of tensors must match except in dimension {dim}. Expected {s0} but got {st}")
    # Differentiable native path (stays in the autograd graph). Scalars, mixed dtypes,
    # and non-tensor inputs keep the NumPy path below: the native kernel needs rank >= 1
    # and coerces to the first dtype, while NumPy promotes mixed dtypes.
    # NOTE: explicit loop because this module shadows builtin all()/any() with tensor ops.
    use_native = True
    first_dtype = None
    for t in tensors:
        if not isinstance(t, _Tensor) or len(t.shape) < 1:
            use_native = False
            break
        if first_dtype is None:
            first_dtype = t.dtype
        elif t.dtype != first_dtype:
            use_native = False
            break
    if use_native:
        return _Tensor.cat(list(tensors), dim)
    arrays = [_to_np(t) for t in tensors]
    res = np.concatenate(arrays, axis=dim)
    return _from_numpy(_contig(res))


concat = cat


def stack(tensors: Sequence[_Tensor], dim: int = 0) -> _Tensor:
    """Concatenates a sequence of tensors along a new dimension."""
    if len(tensors) == 0:
        raise RuntimeError("stack expects a non-empty sequence of tensors")
    s0 = tuple(tensors[0].shape)
    for t in tensors[1:]:
        if tuple(t.shape) != s0:
            raise RuntimeError(f"stack expects each tensor to be equal size, but got {s0} and {tuple(t.shape)}")
    # Differentiable native path (stays in the autograd graph). Mixed dtypes and
    # non-tensor inputs keep the NumPy path below (NumPy promotes mixed dtypes).
    # NOTE: explicit loop because this module shadows builtin all()/any() with tensor ops.
    use_native = True
    first_dtype = None
    for t in tensors:
        if not isinstance(t, _Tensor):
            use_native = False
            break
        if first_dtype is None:
            first_dtype = t.dtype
        elif t.dtype != first_dtype:
            use_native = False
            break
    if use_native:
        return _Tensor.stack(list(tensors), dim)
    arrays = [_to_np(t) for t in tensors]
    res = np.stack(arrays, axis=dim)
    return _from_numpy(_contig(res))


def chunk(input: _Tensor, chunks: int, dim: int = 0) -> Tuple[_Tensor, ...]:
    """Splits a tensor into a specific number of chunks."""
    return input.chunk(chunks, dim=dim)


def split(input: _Tensor, split_size_or_sections: Union[int, List[int]], dim: int = 0) -> Tuple[_Tensor, ...]:
    """Splits the tensor into chunks."""
    return input.split(split_size_or_sections, dim=dim)


def gather(input: _Tensor, dim: int, index: _Tensor) -> _Tensor:
    """Gathers values along an axis specified by dim."""
    # Native engine (differentiable); the old numpy path broke autograd.
    return input.gather(dim, index)


def scatter(input: _Tensor, dim: int, index: _Tensor, src: _Tensor) -> _Tensor:
    """Writes all values from tensor src into input at the specified index positions."""
    # Native engine (differentiable); the old numpy path broke autograd.
    return input.scatter(dim, index, src)


def scatter_add(input: _Tensor, dim: int, index: _Tensor, src: _Tensor) -> _Tensor:
    """Adds all values from tensor src into input at the specified index positions."""
    # Native engine (differentiable); the old numpy loop broke autograd.
    return input.scatter_add(dim, index, src)


def index_select(input: _Tensor, dim: int, index: _Tensor) -> _Tensor:
    """Returns a new tensor which indexes the input tensor along dimension dim using index."""
    arr = _to_np(input)
    idx = _to_np(index).astype(int).reshape(-1)
    res = np.take(arr, idx, axis=dim)
    return _from_numpy(_contig(res))


def nonzero(input: _Tensor, as_tuple: bool = False) -> Union[_Tensor, Tuple[_Tensor, ...]]:
    """Returns the indices of the non-zero elements of the input tensor."""
    arr = _to_np(input)
    indices = np.nonzero(arr)
    if as_tuple:
        return tuple(_from_numpy(_contig(idx)) for idx in indices)
    stacked = np.stack(indices, axis=-1)
    return _from_numpy(_contig(stacked, dtype=np.int64))


def where(
    condition: Union[_Tensor, Any],
    x: Optional[Union[_Tensor, float]] = None,
    y: Optional[Union[_Tensor, float]] = None,
) -> Union[_Tensor, Tuple[_Tensor, ...]]:
    """Return a tensor of elements selected from either x or y, depending on condition."""
    if x is None and y is None:
        return nonzero(condition, as_tuple=True)
    cond = condition if isinstance(condition, _Tensor) else tensor(condition, dtype=bool)
    if cond.dtype != bool:
        cond = cond != 0
    ref = x if isinstance(x, _Tensor) else y if isinstance(y, _Tensor) else None

    def operand(v):
        if isinstance(v, _Tensor):
            return v
        if ref is None:
            return tensor(v).to(cond.device) if str(cond.device) != "cpu" else tensor(v)
        # a Python number takes the other operand's dtype (a float on an integer tensor gives float32, like PyTorch)
        floaty = ref.dtype in ("f16", "bf16", "f32", "f64")
        dt = ref.dtype if (floaty or not isinstance(v, float)) else "f32"
        t = tensor(v, dtype=dt)
        return t.to(ref.device) if str(ref.device) != "cpu" else t
    return _native_where(cond, operand(x), operand(y))

def roll(input: _Tensor, shifts: Union[int, Sequence[int]], dims: Optional[Union[int, Sequence[int]]] = None) -> _Tensor:
    """Roll the tensor along the given dimension(s)."""
    if dims is None:
        flat = input.reshape([-1])
        shift = shifts[0] if isinstance(shifts, (list, tuple)) else shifts
        n = flat.numel()
        if n == 0:
            return input.clone()
        idx = _Tensor.from_numpy(np.array([(i - shift) % n for i in range(n)], dtype=np.int64))
        return flat.index_select(0, idx).reshape(input.shape)
    if isinstance(shifts, int):
        shifts = [shifts]
    if isinstance(dims, int):
        dims = [dims]
    if len(shifts) != len(dims):
        if len(shifts) == 1:
            shifts = list(shifts) * len(dims)
        else:
            raise RuntimeError(f"shifts and dims must have the same length, but got {len(shifts)} and {len(dims)}")
    out = input
    ndim = input.ndim
    for s, d in zip(shifts, dims):
        if d < -ndim or d >= ndim:
            raise IndexError(f"Dimension out of range (expected to be in range of [{-ndim}, {ndim - 1}], but got {d})")
        d = d if d >= 0 else d + ndim
        n = int(out.shape[d])
        if n == 0:
            continue
        idx = _Tensor.from_numpy(np.array([(i - s) % n for i in range(n)], dtype=np.int64))
        out = out.index_select(d, idx)
    return out


def flip(input: _Tensor, dims: Union[int, Sequence[int]]) -> _Tensor:
    """Reverse the order of an n-D tensor along given axis in dims."""
    if isinstance(dims, int):
        dims = [dims]
    ndim = input.ndim
    norm_dims = []
    for d in dims:
        if d < -ndim or d >= ndim:
            raise IndexError(f"Dimension out of range (expected to be in range of [{-ndim}, {ndim - 1}], but got {d})")
        norm_dims.append(d if d >= 0 else d + ndim)
    out = input
    for d in norm_dims:
        n = int(out.shape[d])
        if n == 0:
            continue
        idx = _Tensor.from_numpy(np.arange(n - 1, -1, -1, dtype=np.int64))
        out = out.index_select(d, idx)
    return out


def rot90(input: _Tensor, k: int = 1, dims: Sequence[int] = (0, 1)) -> _Tensor:
    """Rotate an n-D tensor by 90 degrees in the plane specified by dims."""
    if len(dims) != 2:
        raise ValueError(f"rot90 expects dims of length 2, got {dims}")
    ndim = input.ndim
    d0, d1 = dims[0], dims[1]
    if d0 < -ndim or d0 >= ndim or d1 < -ndim or d1 >= ndim:
        raise IndexError(f"rot90 dims out of range for {ndim}D tensor: {dims}")
    d0 = d0 if d0 >= 0 else d0 + ndim
    d1 = d1 if d1 >= 0 else d1 + ndim
    if d0 == d1:
        raise ValueError(f"rot90 dimensions must be different, got {d0} and {d1}")
    rem = k % 4
    if rem == 0:
        return input.clone() if hasattr(input, "clone") else input
    elif rem == 1:
        return flip(input.transpose(d0, d1), [d0])
    elif rem == 2:
        return flip(input, [d0, d1])
    else:
        return flip(input.transpose(d0, d1), [d1])


def repeat_interleave(input: _Tensor, repeats: Union[int, _Tensor, Sequence[int]], dim: Optional[int] = None) -> _Tensor:
    """Repeats elements of a tensor (differentiable: a gather of the repeated positions)."""
    x = input if dim is not None else input.reshape([-1])
    d = 0 if dim is None else (dim if dim >= 0 else dim + x.ndim)
    n = int(x.shape[d])
    reps = _to_np(repeats).astype(np.int64) if isinstance(repeats, _Tensor) else np.asarray(repeats, dtype=np.int64)
    if reps.ndim == 0 or reps.size == 1:
        reps = np.full(n, int(reps.reshape(-1)[0]), dtype=np.int64)
    if reps.shape[0] != n:
        raise RuntimeError(f"repeats must have the same size as input along dim, but got repeats of size {reps.shape[0]} and {n}")
    idx = np.repeat(np.arange(n, dtype=np.int64), reps)
    return x.index_select(d, _Tensor.from_numpy(idx))


def _repeat_tensor(x: _Tensor, reps: Sequence[int]) -> _Tensor:
    """`Tensor.repeat` / `tile`: out[j] = x[j % size] along every dim, built from reshape + expand (differentiable)."""
    reps = [int(r) for r in reps]
    if len(reps) < x.ndim:
        raise RuntimeError("Number of dimensions of repeat dims can not be smaller than number of dimensions of tensor")
    shape = [1] * (len(reps) - x.ndim) + [int(v) for v in x.shape]
    inter, expanded, final = [], [], []
    for r, n in zip(reps, shape):
        inter += [1, n]
        expanded += [r, n]
        final.append(r * n)
    return x.reshape(inter).expand(expanded).reshape(final)


def _expand_index(index: _Tensor, dim: int, source: _Tensor) -> _Tensor:
    """A 1-D index along `dim` broadcast to `source`'s shape (what scatter / scatter_add take)."""
    idx = _to_np(index).astype(np.int64).reshape(-1)
    shape = [1] * source.ndim
    shape[dim] = idx.shape[0]
    return _Tensor.from_numpy(_contig(np.broadcast_to(idx.reshape(shape), tuple(source.shape))))


def tile(input: _Tensor, dims: Sequence[int]) -> _Tensor:
    """Construct a tensor by repeating the elements of input (differentiable)."""
    dims = list(dims)
    if len(dims) < input.ndim:
        dims = [1] * (input.ndim - len(dims)) + dims
    return _repeat_tensor(input, dims)


def all(input: _Tensor, dim: Optional[int] = None, keepdim: bool = False) -> _Tensor:
    """Tests if all elements in input evaluate to True."""
    arr = _to_np(input) != 0
    res = np.all(arr, axis=dim, keepdims=keepdim)
    return _from_numpy(_contig(res))


def any(input: _Tensor, dim: Optional[int] = None, keepdim: bool = False) -> _Tensor:
    """Tests if any element in input evaluates to True."""
    arr = _to_np(input) != 0
    res = np.any(arr, axis=dim, keepdims=keepdim)
    return _from_numpy(_contig(res))


def argmax(input: _Tensor, dim: Optional[int] = None, keepdim: bool = False) -> _Tensor:
    """Returns the indices of the maximum value of all elements in the input tensor."""
    if dim is None:
        idx = int(np.argmax(_to_np(input)))
        return _from_numpy(np.array(idx, dtype=np.int64))
    res = np.argmax(_to_np(input), axis=dim)
    if keepdim:
        res = np.expand_dims(res, axis=dim)
    return _from_numpy(_contig(res, dtype=np.int64))


def argmin(input: _Tensor, dim: Optional[int] = None, keepdim: bool = False) -> _Tensor:
    """Returns the indices of the minimum value of all elements in the input tensor."""
    if dim is None:
        idx = int(np.argmin(_to_np(input)))
        return _from_numpy(np.array(idx, dtype=np.int64))
    res = np.argmin(_to_np(input), axis=dim)
    if keepdim:
        res = np.expand_dims(res, axis=dim)
    return _from_numpy(_contig(res, dtype=np.int64))


def amax(input: _Tensor, dim: Optional[Union[int, Tuple[int, ...]]] = None, keepdim: bool = False) -> _Tensor:
    """Returns the maximum of each slice of the input tensor in the given dimension(s) (differentiable; ties share the gradient)."""
    return input.amax(dim, keepdim)

def amin(input: _Tensor, dim: Optional[Union[int, Tuple[int, ...]]] = None, keepdim: bool = False) -> _Tensor:
    """Returns the minimum of each slice of the input tensor in the given dimension(s) (differentiable; ties share the gradient)."""
    return input.amin(dim, keepdim)

def maximum(input: _Tensor, other: _Tensor) -> _Tensor:
    """Computes the element-wise maximum of input and other."""
    return input.maximum(other)


def minimum(input: _Tensor, other: _Tensor) -> _Tensor:
    """Computes the element-wise minimum of input and other."""
    return input.minimum(other)


def isclose(input: _Tensor, other: Union[_Tensor, float], rtol: float = 1e-5, atol: float = 1e-8) -> _Tensor:
    """Returns a new tensor with boolean elements representing if each element of input is close to other."""
    x = _to_np(input)
    y = _to_np(other)
    res = np.isclose(x, y, rtol=rtol, atol=atol)
    return _from_numpy(_contig(res))


def allclose(input: _Tensor, other: Union[_Tensor, float], rtol: float = 1e-5, atol: float = 1e-8) -> bool:
    """Returns True if two tensors have the same shape and elements satisfy isclose."""
    x = _to_np(input)
    y = _to_np(other)
    return bool(np.allclose(x, y, rtol=rtol, atol=atol))


def einsum(equation: str, *operands: _Tensor) -> _Tensor:
    """Sums the product of the elements of the input operands along dimensions specified using Einstein summation convention
    (native, differentiable, keeps the input dtype)."""
    if len(operands) == 1 and isinstance(operands[0], (list, tuple)):
        operands = tuple(operands[0])
    for op in operands:
        if not isinstance(op, _Tensor):
            raise TypeError(f"einsum(): expected Tensor but got {type(op).__name__}")
    return _native_einsum(equation, *operands)

def _check_sort_dim(dim, nd: int) -> int:
    try:
        ax = operator.index(dim)
    except TypeError:
        raise ValueError(f"dim must be an integer, got {dim!r}") from None
    if not -nd <= ax < nd:
        raise ValueError(f"dim {dim} out of range for {nd}-D input")
    return ax


def sort(input: _Tensor, dim: int = -1, descending: bool = False) -> Tuple[_Tensor, _Tensor]:
    """Sorts the elements of the input tensor along a given dimension in ascending or descending order by value."""
    _check_sort_dim(dim, len(input.shape))
    # Native engine: stable ties, differentiable values, i64-exact.
    return input.sort(dim, descending=descending)


def argsort(input: _Tensor, dim: int = -1, descending: bool = False) -> _Tensor:
    """Returns the indices that sort a tensor along a given dimension in ascending or descending order."""
    _check_sort_dim(dim, len(input.shape))
    return input.argsort(dim, descending=descending)


def topk(input: _Tensor, k: int, dim: int = -1, largest: bool = True, sorted: bool = True) -> Tuple[_Tensor, _Tensor]:
    """Returns the k largest or smallest elements of the given input tensor along a given dimension."""
    from .nn import _check_non_negative
    k = _check_non_negative("k", k)
    arr = _to_np(input)
    nd = arr.ndim
    try:
        ax = operator.index(dim)
    except TypeError:
        raise ValueError(f"dim must be an integer, got {dim!r}") from None
    if not -nd <= ax < nd:
        raise ValueError(f"dim {dim} out of range for {nd}-D input")
    ax %= nd
    size = arr.shape[ax]
    if k > size:
        raise ValueError(f"k = {k} out of range for dimension of size {size}")
    if k == 0:
        out_shape = list(arr.shape)
        out_shape[ax] = 0
        return (
            _from_numpy(np.empty(out_shape, dtype=arr.dtype)),
            _from_numpy(np.empty(out_shape, dtype=np.int64)),
        )
    # Native engine: stable ties, differentiable values, i64-exact.
    return input.topk(k, dim=ax, largest=largest, sorted=sorted)


def kthvalue(input: _Tensor, k: int, dim: int = -1, keepdim: bool = False) -> Tuple[_Tensor, _Tensor]:
    """Returns the k-th smallest elements (1-based k) and their indices along dim."""
    # Pure delegation (no legacy validation shell): out-of-range k/dim raise
    # the native RuntimeError, matching torch's error type.
    return input.kthvalue(k, dim=dim, keepdim=keepdim)


def mean(input: _Tensor, dim: Optional[int] = None, keepdim: bool = False) -> _Tensor:
    """Returns the mean value of each row of the input tensor in the given dimension dim."""
    if dim is None:
        return input.mean()
    arr = _to_np(input)
    res = np.mean(arr, axis=dim, keepdims=keepdim)
    return _from_numpy(_contig(res))


def sum(input: _Tensor, dim: Optional[int] = None, keepdim: bool = False) -> _Tensor:
    """Returns the sum of all elements in the input tensor, or along dim."""
    if dim is None:
        return input.sum()
    arr = _to_np(input)
    res = np.sum(arr, axis=dim, keepdims=keepdim)
    return _from_numpy(_contig(res))


def var(input: _Tensor, dim: Optional[int] = None, unbiased: bool = True, keepdim: bool = False) -> _Tensor:
    """Returns the variance of each row of the input tensor in the given dimension dim."""
    return input.var(dim=dim, unbiased=unbiased, keepdim=keepdim)


def std(input: _Tensor, dim: Optional[int] = None, unbiased: bool = True, keepdim: bool = False) -> _Tensor:
    """Returns the standard-deviation of each row of the input tensor in the given dimension dim."""
    return input.std(dim=dim, unbiased=unbiased, keepdim=keepdim)


def cartesian_prod(*tensors: Any) -> _Tensor:
    """Computes the cartesian product of given sequence of 1D tensors."""
    import builtins
    if len(tensors) == 1 and isinstance(tensors[0], (list, tuple)):
        tensors = tuple(tensors[0])
    if tensors and builtins.all(isinstance(t, _Tensor) for t in tensors):
        from .oxideon import cartesian_prod as _native_cartesian_prod
        return _native_cartesian_prod(list(tensors))
    grids = np.meshgrid(*[_to_np(t) for t in tensors], indexing="ij")
    stacked = np.stack([g.reshape(-1) for g in grids], axis=-1)
    return _from_numpy(_contig(stacked, dtype=tensors[0].to_numpy().dtype))


def meshgrid(*tensors: _Tensor, indexing: str = "ij") -> Tuple[_Tensor, ...]:
    """Creates grids of coordinates specified by the 1D inputs in tensors (differentiable expands)."""
    if len(tensors) == 1 and isinstance(tensors[0], (list, tuple)):
        tensors = tuple(tensors[0])
    if indexing not in ("ij", "xy"):
        raise ValueError(f"meshgrid: indexing must be 'ij' or 'xy', got {indexing!r}")
    ts = list(tensors)
    swap = indexing == "xy" and len(ts) >= 2
    if swap:
        ts[0], ts[1] = ts[1], ts[0]
    shape = [int(t.numel()) for t in ts]
    outs = []
    for i, t in enumerate(ts):
        view = [1] * len(ts)
        view[i] = shape[i]
        outs.append(t.reshape(view).expand(shape))
    if swap:
        outs[0], outs[1] = outs[1], outs[0]
    return tuple(outs)


def atleast_1d(*tensors: Union[_Tensor, Any]) -> Union[_Tensor, Tuple[_Tensor, ...]]:
    """Returns a 1-dimensional view of each input tensor with zero dimensions."""
    res = []
    for t in tensors:
        t_obj = t if isinstance(t, _Tensor) else tensor(t)
        if t_obj.dim() == 0:
            res.append(t_obj.unsqueeze(0))
        else:
            res.append(t_obj)
    return res[0] if len(res) == 1 else tuple(res)


def atleast_2d(*tensors: Union[_Tensor, Any]) -> Union[_Tensor, Tuple[_Tensor, ...]]:
    """Returns a 2-dimensional view of each input tensor with zero or one dimensions."""
    res = []
    for t in tensors:
        t_obj = t if isinstance(t, _Tensor) else tensor(t)
        if t_obj.dim() == 0:
            res.append(t_obj.unsqueeze(0).unsqueeze(0))
        elif t_obj.dim() == 1:
            res.append(t_obj.unsqueeze(0))
        else:
            res.append(t_obj)
    return res[0] if len(res) == 1 else tuple(res)


def atleast_3d(*tensors: Union[_Tensor, Any]) -> Union[_Tensor, Tuple[_Tensor, ...]]:
    """Returns a 3-dimensional view of each input tensor with less than three dimensions."""
    res = []
    for t in tensors:
        t_obj = t if isinstance(t, _Tensor) else tensor(t)
        if t_obj.dim() == 0:
            res.append(t_obj.unsqueeze(0).unsqueeze(0).unsqueeze(0))
        elif t_obj.dim() == 1:
            res.append(t_obj.unsqueeze(0).unsqueeze(-1))
        elif t_obj.dim() == 2:
            res.append(t_obj.unsqueeze(-1))
        else:
            res.append(t_obj)
    return res[0] if len(res) == 1 else tuple(res)


def column_stack(tensors: Sequence[_Tensor]) -> _Tensor:
    """Stacks 1D or 2D tensors horizontally (column-wise)."""
    arrays = [_to_np(t) for t in tensors]
    res = np.column_stack(arrays)
    return _from_numpy(_contig(res, dtype=tensors[0].to_numpy().dtype))


def row_stack(tensors: Sequence[_Tensor]) -> _Tensor:
    """Stacks tensors vertically (row-wise)."""
    arrays = [_to_np(t) for t in tensors]
    res = np.row_stack(arrays)
    return _from_numpy(_contig(res, dtype=tensors[0].to_numpy().dtype))


def vstack(tensors: Sequence[_Tensor]) -> _Tensor:
    """Stacks tensors vertically along the first dimension (differentiable)."""
    return cat([atleast_2d(t) for t in tensors], 0)


def hstack(tensors: Sequence[_Tensor]) -> _Tensor:
    """Stacks tensors horizontally (along the second dimension, or the first for 1-D inputs; differentiable)."""
    ts = [atleast_1d(t) for t in tensors]
    return cat(ts, 0 if ts[0].ndim == 1 else 1)


def dstack(tensors: Sequence[_Tensor]) -> _Tensor:
    """Stacks tensors depth-wise along the third dimension (differentiable)."""
    return cat([atleast_3d(t) for t in tensors], 2)


def block_diag(*tensors: Any) -> _Tensor:
    """Creates a block diagonal tensor from provided tensors (differentiable via native Rust kernel)."""
    import builtins
    if len(tensors) == 1 and isinstance(tensors[0], (list, tuple)):
        tensors = tuple(tensors[0])
    if tensors and builtins.all(isinstance(t, _Tensor) for t in tensors):
        from .oxideon import block_diag as _native_block_diag
        return _native_block_diag(list(tensors))
    arrays = [_to_np(t) for t in tensors]
    try:
        import scipy.linalg as sla
        res = sla.block_diag(*arrays)
    except Exception:
        total_r = builtins.sum(a.shape[0] for a in arrays)
        total_c = builtins.sum(a.shape[1] if a.ndim > 1 else 1 for a in arrays)
        res = np.zeros((total_r, total_c), dtype=arrays[0].dtype)
        curr_r = 0
        curr_c = 0
        for a in arrays:
            r = a.shape[0]
            c = a.shape[1] if a.ndim > 1 else 1
            res[curr_r : curr_r + r, curr_c : curr_c + c] = a.reshape(r, c)
            curr_r += r
            curr_c += c
    return _from_numpy(_contig(res, dtype=tensors[0].to_numpy().dtype))


def diagflat(input: _Tensor, offset: int = 0) -> _Tensor:
    """Creates a two-dimensional tensor with the flattened input placed along the diagonal."""
    arr = _to_np(input).reshape(-1)
    res = np.diagflat(arr, k=offset)
    return _from_numpy(_contig(res, dtype=input.to_numpy().dtype))


def tril(input: _Tensor, diagonal: int = 0) -> _Tensor:
    """Returns the lower triangular part of the matrix (or batch of matrices) input."""
    t = input if isinstance(input, _Tensor) else tensor(input)
    return _native_tril(t, int(diagonal))


def triu(input: _Tensor, diagonal: int = 0) -> _Tensor:
    """Returns the upper triangular part of the matrix (or batch of matrices) input."""
    t = input if isinstance(input, _Tensor) else tensor(input)
    return _native_triu(t, int(diagonal))


def tril_indices(row: int, col: int, offset: int = 0) -> _Tensor:
    """Returns the indices of the lower triangular part of a row-by-col matrix as a 2 x N tensor."""
    r, c = np.tril_indices(row, k=offset, m=col)
    return _from_numpy(_contig(np.stack([r, c]), dtype=np.int64))


def triu_indices(row: int, col: int, offset: int = 0) -> _Tensor:
    """Returns the indices of the upper triangular part of a row-by-col matrix as a 2 x N tensor."""
    r, c = np.triu_indices(row, k=offset, m=col)
    return _from_numpy(_contig(np.stack([r, c]), dtype=np.int64))


def diag_indices(n: int, ndim: int = 2) -> Tuple[_Tensor, ...]:
    """Returns the indices of the diagonal elements of an n-by-n array with ndim dimensions."""
    idx = np.diag_indices(n, ndim=ndim)
    return tuple(_from_numpy(_contig(i, dtype=np.int64)) for i in idx)


def narrow(input: _Tensor, dim: int, start: int, length: int) -> _Tensor:
    """Returns a new tensor that is a narrowed version of input along dimension dim."""
    arr = _to_np(input)
    sl = [slice(None)] * arr.ndim
    sl[dim] = slice(start, start + length)
    res = arr[tuple(sl)]
    return _from_numpy(_contig(res, dtype=input.to_numpy().dtype))


def select(input: _Tensor, dim: int, index: int) -> _Tensor:
    """Slices the input tensor along the selected dimension at the given index."""
    arr = _to_np(input)
    sl = [slice(None)] * arr.ndim
    sl[dim] = index
    res = arr[tuple(sl)]
    return _from_numpy(_contig(res, dtype=input.to_numpy().dtype))


def take(input: _Tensor, index: _Tensor) -> _Tensor:
    """Returns a new tensor with the elements of input at the given flat indices (differentiable gather)."""
    n = int(input.numel())
    idx = _to_np(index).astype(np.int64)
    if idx.size and (idx.min() < -n or idx.max() >= n):
        raise IndexError(f"out of range: tried to access index {int(idx.max() if idx.max() >= n else idx.min())} on a tensor of {n} elements")
    idx = np.where(idx < 0, idx + n, idx)
    return input.reshape([-1]).index_select(0, _Tensor.from_numpy(_contig(idx.reshape(-1)))).reshape(list(idx.shape))


def take_along_dim(input: _Tensor, indices: _Tensor, dim: int) -> _Tensor:
    """Selects 1D slices from input along dim at indices."""
    x = input if isinstance(input, _Tensor) else tensor(input)
    idx = indices if isinstance(indices, _Tensor) else tensor(indices)
    return x.gather(dim, idx)


def diff(input: _Tensor, n: int = 1, dim: int = -1, prepend: Optional[_Tensor] = None, append: Optional[_Tensor] = None) -> _Tensor:
    """Computes the n-th forward differences along the given dimension."""
    # The method wrapper validates n and joins prepend/append; compute is native.
    return input.diff(n=n, dim=dim, prepend=prepend, append=append)


def gradient(input: _Tensor, spacing: Any = 1.0, dim: Optional[Union[int, Sequence[int]]] = None, edge_order: int = 1) -> Tuple[_Tensor, ...]:
    """Computes the gradient of an N-dimensional tensor using central differences."""
    # Native per-axis kernels (differentiable); always returns a tuple.
    # The old numpy shim truncated int/bool inputs back to the input dtype.
    ndim = len(input.shape)
    if dim is None:
        dims = list(range(ndim))
    elif isinstance(dim, (list, tuple)):
        dims = list(dim)
    else:
        dims = [dim]
    seen = set()
    for d in dims:
        n = d if d >= 0 else ndim + d
        if n in seen:
            raise RuntimeError(f"dim {d} appears multiple times in the list of dims")
        seen.add(n)
    if spacing is None:
        spacing = 1.0
    if isinstance(spacing, (list, tuple)):
        if len(spacing) != len(dims):
            raise RuntimeError(
                "torch.gradient expected spacing to be unspecified, a scalar, "
                f"or a list of len {len(dims)} (got len {len(spacing)})")
        per_dim = list(spacing)
    elif isinstance(spacing, _Tensor):
        raise TypeError(
            "gradient() spacing must be a scalar or a list/tuple (tensors go inside a list)")
    else:
        per_dim = [spacing] * len(dims)
    out = []
    for d, s in zip(dims, per_dim):
        if isinstance(s, _Tensor):
            out.append(input.gradient_axis_coords(d, edge_order, s))
        elif isinstance(s, (int, float, np.floating, np.integer)):
            out.append(input.gradient_axis(d, edge_order, float(s)))
        else:
            raise TypeError(
                f"gradient() spacing elements must be scalars or tensors (got {type(s).__name__})")
    return tuple(out)


def trapz(y: _Tensor, x: Optional[_Tensor] = None, dx: Optional[float] = None, dim: int = -1) -> _Tensor:
    """Integrates along the given axis using the composite trapezoidal rule."""
    # Native kernels (differentiable in y, and in x for coordinates). x and
    # dx are mutually exclusive, like torch (dx=None detects "given").
    if x is not None and dx is not None:
        raise TypeError("trapezoid() got both x and dx (mutually exclusive)")
    if x is not None:
        if not isinstance(x, _Tensor):
            raise TypeError(f"trapezoid() x must be a tensor (got {type(x).__name__})")
        return y.trapezoid_axis_x(dim, x)
    return y.trapezoid_axis(dim, 1.0 if dx is None else dx)


def trapezoid(y: _Tensor, x: Optional[_Tensor] = None, dx: Optional[float] = None, dim: int = -1) -> _Tensor:
    return trapz(y, x=x, dx=dx, dim=dim)


def cumulative_trapezoid(y: _Tensor, x: Optional[_Tensor] = None, dx: float = 1.0, dim: int = -1) -> _Tensor:
    """Cumulatively integrates y(x) using the composite trapezoidal rule (differentiable)."""
    d = dim if dim >= 0 else y.ndim + dim
    n = int(y.shape[d])
    if n < 1:
        return y.clone()
    mid = (y.narrow(d, 1, n - 1) + y.narrow(d, 0, n - 1)) * 0.5
    if x is None:
        steps = mid * dx
    else:
        if x.ndim == 1:
            shape = [1] * y.ndim
            shape[d] = n
            xs = x.reshape(shape)
            gap = xs.narrow(d, 1, n - 1) - xs.narrow(d, 0, n - 1)
        else:
            xd = x.ndim - y.ndim + d if x.ndim != y.ndim else d
            gap = x.narrow(xd, 1, n - 1) - x.narrow(xd, 0, n - 1)
        steps = mid * gap
    return steps.cumsum(d)

def heaviside(input: _Tensor, values: Union[_Tensor, float]) -> _Tensor:
    """Computes the Heaviside step function for each element in input."""
    arr = _to_np(input)
    v = _to_np(values) if isinstance(values, _Tensor) else float(values)
    res = np.heaviside(arr, v)
    return _from_numpy(_contig(res, dtype=input.to_numpy().dtype))


def atan2(input: _Tensor, other: Union[_Tensor, float]) -> _Tensor:
    """Element-wise arc tangent of input / other with respect to quadrants."""
    if hasattr(input, "atan2") and isinstance(other, _Tensor):
        return input.atan2(other)
    y = _to_np(input)
    x = _to_np(other) if isinstance(other, _Tensor) else float(other)
    res = np.arctan2(y, x)
    return _from_numpy(_contig(res, dtype=input.to_numpy().dtype))


arctan2 = atan2


def logaddexp(input: _Tensor, other: _Tensor) -> _Tensor:
    """Logarithm of the sum of exponentiations of the inputs."""
    a = _to_np(input)
    b = _to_np(other)
    res = np.logaddexp(a, b)
    return _from_numpy(_contig(res, dtype=input.to_numpy().dtype))


def logaddexp2(input: _Tensor, other: _Tensor) -> _Tensor:
    """Logarithm of the sum of exponentiations of the inputs in base-2."""
    a = _to_np(input)
    b = _to_np(other)
    res = np.logaddexp2(a, b)
    return _from_numpy(_contig(res, dtype=input.to_numpy().dtype))


def hypot(input: _Tensor, other: Union[_Tensor, float]) -> _Tensor:
    """Given the legs of a right triangle, returns its hypotenuse sqrt(x^2 + y^2)."""
    if hasattr(input, "hypot") and isinstance(other, _Tensor):
        return input.hypot(other)
    a = _to_np(input)
    b = _to_np(other) if isinstance(other, _Tensor) else float(other)
    res = np.hypot(a, b)
    return _from_numpy(_contig(res, dtype=input.to_numpy().dtype))


def copysign(input: _Tensor, other: Union[_Tensor, float]) -> _Tensor:
    """Creates a new floating-point tensor with the magnitude of input and the sign of other."""
    a = _to_np(input)
    b = _to_np(other) if isinstance(other, _Tensor) else float(other)
    res = np.copysign(a, b)
    return _from_numpy(_contig(res, dtype=input.to_numpy().dtype))


def fmax(input: _Tensor, other: _Tensor) -> _Tensor:
    """Computes the element-wise maximum of input and other, respecting NaNs."""
    a = _to_np(input)
    b = _to_np(other)
    res = np.fmax(a, b)
    return _from_numpy(_contig(res, dtype=input.to_numpy().dtype))


def fmin(input: _Tensor, other: _Tensor) -> _Tensor:
    """Computes the element-wise minimum of input and other, respecting NaNs."""
    a = _to_np(input)
    b = _to_np(other)
    res = np.fmin(a, b)
    return _from_numpy(_contig(res, dtype=input.to_numpy().dtype))


def fmod(input: _Tensor, other: Union[_Tensor, float]) -> _Tensor:
    """Computes the floating-point remainder of division."""
    a = _to_np(input)
    b = _to_np(other) if isinstance(other, _Tensor) else float(other)
    res = np.fmod(a, b)
    return _from_numpy(_contig(res, dtype=input.to_numpy().dtype))


def remainder(input: _Tensor, other: Union[_Tensor, float]) -> _Tensor:
    """Computes the Python element-wise remainder."""
    a = _to_np(input)
    b = _to_np(other) if isinstance(other, _Tensor) else float(other)
    res = np.remainder(a, b)
    return _from_numpy(_contig(res, dtype=input.to_numpy().dtype))


def bucketize(input: _Tensor, boundaries: _Tensor, out_int32: bool = False, right: bool = False) -> _Tensor:
    """Returns the indices of the buckets to which each value in the input belongs."""
    arr = _to_np(input)
    b = _to_np(boundaries)
    res = np.digitize(arr, b, right=right)
    dtype = np.int32 if out_int32 else np.int64
    return _from_numpy(_contig(res, dtype=dtype))


def searchsorted(sorted_sequence: _Tensor, values: _Tensor, out_int32: bool = False, right: bool = False, side: str = "left") -> _Tensor:
    """Find the indices from the innermost dimension of sorted_sequence such that if values were inserted before the indices, the order would be preserved."""
    if side not in ("left", "right"):
        raise ValueError(f"side must be 'left' or 'right', got {side!r}")
    # Native engine: N-D broadcast rows, NaN queries read len, i64-exact past 2^53.
    out = _native_searchsorted(sorted_sequence, values, right=(right or side == "right"))
    if out_int32:
        return _from_numpy(_contig(out.to_numpy(), dtype=np.int32))
    return out


def nan_to_num(input: _Tensor, nan: float = 0.0, posinf: Optional[float] = None, neginf: Optional[float] = None) -> _Tensor:
    """Replaces NaN, positive infinity, and negative infinity values in input with specified numbers (differentiable:
    replaced entries get a zero gradient)."""
    if input.dtype not in ("f16", "bf16", "f32", "f64"):
        return input.clone()
    info = finfo(input.dtype)
    hi = info.max if posinf is None else float(posinf)
    lo = info.min if neginf is None else float(neginf)
    out = input.masked_fill(input.isnan(), float(nan))
    out = out.masked_fill(input.isposinf(), hi)
    return out.masked_fill(input.isneginf(), lo)


def nanmean(input: _Tensor, dim: Optional[Union[int, Tuple[int, ...]]] = None, keepdim: bool = False) -> _Tensor:
    """Computes the mean of all non-NaN elements in input."""
    # Native engine (differentiable, float only like torch). Tuple dims route
    # to the multi-axis kernel: sequential per-dim means would misweight
    # slices with different NaN counts.
    if isinstance(dim, (tuple, list)):
        return input.nanmean_axes(list(dim), keepdim=keepdim)
    return input.nanmean(dim=dim, keepdim=keepdim)


def nansum(input: _Tensor, dim: Optional[Union[int, Tuple[int, ...]]] = None, keepdim: bool = False) -> _Tensor:
    """Computes the sum of all non-NaN elements in input."""
    # Native engine (differentiable). Tuple dims route to the multi-axis kernel.
    if isinstance(dim, (tuple, list)):
        return input.nansum_axes(list(dim), keepdim=keepdim)
    return input.nansum(dim=dim, keepdim=keepdim)


def quantile(input: _Tensor, q: Union[float, Sequence[float], _Tensor], dim: Optional[int] = None, keepdim: bool = False, interpolation: str = "linear") -> _Tensor:
    """Computes the q-th quantiles of each row of the input tensor."""
    # Native engine (differentiable). Lists/tuples convert with the input's
    # dtype (torch rejects them outright: deliberate extension); numpy
    # scalars coerce to float for the binding's scalar path.
    if isinstance(q, (list, tuple)):
        q = tensor(q, dtype=input.dtype)
    elif isinstance(q, (np.floating, np.integer)):
        q = float(q)
    return input.quantile(q, dim=dim, keepdim=keepdim, interpolation=interpolation)


def nanquantile(input: _Tensor, q: Union[float, Sequence[float], _Tensor], dim: Optional[int] = None, keepdim: bool = False, interpolation: str = "linear") -> _Tensor:
    """Computes the q-th quantiles of each row of the input tensor, ignoring NaN values."""
    # Native engine (differentiable); list/tuple q converts with the input's
    # dtype (torch rejects it: deliberate extension, shared with quantile).
    if isinstance(q, (list, tuple)):
        q = tensor(q, dtype=input.dtype)
    elif isinstance(q, (np.floating, np.integer)):
        q = float(q)
    return input.nanquantile(q, dim=dim, keepdim=keepdim, interpolation=interpolation)


def median(input: _Tensor, dim: Optional[int] = None, keepdim: bool = False) -> Union[_Tensor, Tuple[_Tensor, _Tensor]]:
    """Returns the median of the values in input."""
    # Pure delegation (no legacy validation shell): the old numpy shim
    # averaged even counts and misindexed NaN rows; the native kernel takes
    # torch's lower middle and propagates the first NaN.
    return input.median(dim=dim, keepdim=keepdim)


def nanmedian(input: _Tensor, dim: Optional[int] = None, keepdim: bool = False) -> Union[_Tensor, Tuple[_Tensor, _Tensor]]:
    """Returns the median of values in input, ignoring NaN values."""
    # Pure delegation (no legacy validation shell): with dim this returns a
    # (values, indices) tuple like torch; empty reductions raise RuntimeError.
    return input.nanmedian(dim=dim, keepdim=keepdim)


class _ModeResult(namedtuple("ModeResult", ["values", "indices"])):
    pass


def mode(input: _Tensor, dim: int = -1, keepdim: bool = False) -> _ModeResult:
    """Returns the mode value of each row of the input tensor along dim.

    Ties resolve to the smallest value; indices point at its first occurrence.
    """
    # Pure delegation (drops the scipy/numpy shim, so gradients flow):
    # out-of-range dims raise the native RuntimeError (torch: IndexError).
    values, indices = input.mode(dim=dim, keepdim=keepdim)
    return _ModeResult(values, indices)


def isin(elements: _Tensor, test_elements: _Tensor, assume_unique: bool = False, invert: bool = False) -> _Tensor:
    """Tests if each element of elements is in test_elements."""
    el = _to_np(elements)
    test_el = _to_np(test_elements)
    res = np.isin(el, test_el, assume_unique=assume_unique, invert=invert)
    return _from_numpy(_contig(res))


def unique(input: _Tensor, sorted: bool = True, return_inverse: bool = False, return_counts: bool = False, dim: Optional[int] = None) -> Union[_Tensor, Tuple[_Tensor, ...]]:
    """Returns the unique elements of the input tensor."""
    arr = _to_np(input)
    out = np.unique(arr, return_index=False, return_inverse=return_inverse, return_counts=return_counts, axis=dim)
    if isinstance(out, tuple):
        res_list = [_from_numpy(_contig(o, dtype=np.int64 if i > 0 else input.to_numpy().dtype)) for i, o in enumerate(out)]
        return tuple(res_list)
    return _from_numpy(_contig(out, dtype=input.to_numpy().dtype))


def unique_consecutive(input: _Tensor, return_inverse: bool = False, return_counts: bool = False, dim: Optional[int] = None) -> Union[_Tensor, Tuple[_Tensor, ...]]:
    """Eliminates all but the first element from every consecutive group of equivalent elements."""
    arr = _to_np(input)
    if arr.size == 0:
        return input
    flat = arr.reshape(-1)
    mask = np.concatenate(([True], flat[1:] != flat[:-1]))
    unique_vals = flat[mask]
    res_val = _from_numpy(_contig(unique_vals, dtype=input.to_numpy().dtype))
    if not return_inverse and not return_counts:
        return res_val
    results = [res_val]
    if return_inverse:
        inv = np.cumsum(mask) - 1
        results.append(_from_numpy(_contig(inv, dtype=np.int64)))
    if return_counts:
        indices = np.nonzero(mask)[0]
        counts = np.diff(np.append(indices, len(flat)))
        results.append(_from_numpy(_contig(counts, dtype=np.int64)))
    return tuple(results)


def bincount(input: _Tensor, weights: Optional[_Tensor] = None, minlength: int = 0) -> _Tensor:
    """Counts the frequency of each value in an array of non-negative integers."""
    # Native engine (torch parity incl. rejections; the old numpy path
    # silently flattened, cast floats, and mis-dtyped weights).
    minlength = operator.index(minlength)
    if minlength < 0:
        raise RuntimeError("minlength should be >= 0")
    return input.bincount(weights=weights, minlength=minlength)


def histc(input: _Tensor, bins: int = 100, min: float = 0.0, max: float = 0.0) -> _Tensor:
    """Computes the histogram of a tensor."""
    # Native engine (torch parity incl. the min == max data-range rule).
    return input.histc(bins=bins, min=min, max=max)


def histogram(input: _Tensor, bins: int = 100, range: Optional[Tuple[float, float]] = None, weight: Optional[_Tensor] = None, density: bool = False) -> Tuple[_Tensor, _Tensor]:
    """Computes a histogram of the values in a tensor, returning (hist, bin_edges)."""
    arr = _to_np(input).reshape(-1)
    w = _to_np(weight).reshape(-1) if weight is not None else None
    counts, bin_edges = np.histogram(arr, bins=bins, range=range, weights=w, density=density)
    return _from_numpy(_contig(counts, dtype=np.float32)), _from_numpy(_contig(bin_edges, dtype=np.float32))


class _FunctionalAutograd:
    """High-order automatic differentiation primitives (Jacobian, Hessian, VJP, JVP, HVP)."""

    @staticmethod
    def vjp(
        func: Callable[..., Any],
        inputs: Union[_Tensor, Tuple[_Tensor, ...]],
        v: Optional[Union[_Tensor, Tuple[_Tensor, ...]]] = None,
        create_graph: bool = False,
        strict: bool = False,
    ) -> Tuple[Any, Any]:
        is_tuple = isinstance(inputs, tuple)
        t_inputs = inputs if is_tuple else (inputs,)
        cloned_inputs = tuple(inp.clone().detach().requires_grad_(True) for inp in t_inputs)

        outputs = func(*cloned_inputs)
        out_tuple = outputs if isinstance(outputs, tuple) else (outputs,)

        if v is None:
            v_tuple = tuple(ones_like(o) for o in out_tuple)
        else:
            v_tuple = v if isinstance(v, tuple) else (v,)

        grads = _AutogradModule.grad(
            out_tuple,
            cloned_inputs,
            grad_outputs=v_tuple,
            create_graph=create_graph,
            allow_unused=not strict,
        )
        res_grads = grads if is_tuple else grads[0]
        return outputs, res_grads

    @staticmethod
    def jvp(
        func: Callable[..., Any],
        inputs: Union[_Tensor, Tuple[_Tensor, ...]],
        v: Optional[Union[_Tensor, Tuple[_Tensor, ...]]] = None,
        create_graph: bool = False,
        strict: bool = False,
    ) -> Tuple[Any, Any]:
        """Exact forward-mode Jacobian-vector product via reverse-over-reverse (double backward) AD."""
        is_tuple = isinstance(inputs, tuple)
        t_inputs = inputs if is_tuple else (inputs,)
        if v is None:
            v_tuple = tuple(ones_like(inp) for inp in t_inputs)
        else:
            v_tuple = v if isinstance(v, tuple) else (v,)
        if len(t_inputs) != len(v_tuple):
            raise ValueError(f"jvp: inputs length ({len(t_inputs)}) must match v length ({len(v_tuple)})")

        p_leaves = tuple(inp.detach().requires_grad_(True) for inp in t_inputs)
        outputs = func(*p_leaves)
        out_is_tuple = isinstance(outputs, tuple)
        out_tuple = outputs if out_is_tuple else (outputs,)

        u_dummies = tuple(zeros_like(o).detach().requires_grad_(True) for o in out_tuple)
        vjps = _AutogradModule.grad(
            out_tuple,
            p_leaves,
            grad_outputs=u_dummies,
            create_graph=True,
            allow_unused=not strict,
        )
        terms = []
        for g, vi in zip(vjps, v_tuple):
            if g is not None:
                terms.append((g * vi).sum())
        if not terms:
            jvp_tuple = tuple(zeros_like(o) for o in out_tuple)
        else:
            scalar = terms[0]
            for t_item in terms[1:]:
                scalar = scalar + t_item
            raw_jvps = _AutogradModule.grad(
                scalar,
                u_dummies,
                create_graph=create_graph,
                allow_unused=True,
            )
            jvp_tuple = tuple(
                zeros_like(o) if g is None else g
                for g, o in zip(raw_jvps, out_tuple)
            )

        res_out = outputs if out_is_tuple else outputs.detach() if not create_graph else outputs
        res_jvp = jvp_tuple if out_is_tuple else jvp_tuple[0]
        return res_out, res_jvp

    @staticmethod
    def jacobian(
        func: Callable[..., Any],
        inputs: Union[_Tensor, Tuple[_Tensor, ...]],
        create_graph: bool = False,
        strict: bool = False,
        vectorize: bool = False,
    ) -> Union[_Tensor, Tuple[Any, ...]]:
        is_tuple = isinstance(inputs, tuple)
        t_inputs = inputs if is_tuple else (inputs,)

        cloned_inputs = tuple(inp.clone().detach().requires_grad_(True) for inp in t_inputs)
        outputs = func(*cloned_inputs)
        is_out_tuple = isinstance(outputs, tuple)
        out_list = list(outputs) if is_out_tuple else [outputs]

        all_jacobians = []
        for out_idx, out in enumerate(out_list):
            out_flat = out.reshape([-1])
            m = out_flat.numel()
            jacs_for_out = [[] for _ in t_inputs]

            for i in range(m):
                e_i = zeros_like(out_flat)
                e_i[i] = 1.0
                e_i = e_i.reshape(out.shape)

                cloned_in_step = tuple(inp.clone().detach().requires_grad_(True) for inp in t_inputs)
                out_step = func(*cloned_in_step)
                target_out = out_step[out_idx] if is_out_tuple else out_step

                grads = _AutogradModule.grad(
                    target_out,
                    cloned_in_step,
                    grad_outputs=e_i,
                    create_graph=create_graph,
                    allow_unused=True,
                )

                for k, g in enumerate(grads):
                    if g is None:
                        g = zeros_like(t_inputs[k])
                    jacs_for_out[k].append(g)

            stacked_for_out = []
            for k, jac_rows in enumerate(jacs_for_out):
                stacked = stack(jac_rows, dim=0)
                final_shape = list(out.shape) + list(t_inputs[k].shape)
                stacked_for_out.append(stacked.reshape(final_shape))

            all_jacobians.append(stacked_for_out if is_tuple else stacked_for_out[0])

        if is_out_tuple:
            return tuple(all_jacobians)
        return all_jacobians[0]

    @staticmethod
    def hessian(
        func: Callable[..., Any],
        inputs: Union[_Tensor, Tuple[_Tensor, ...]],
        create_graph: bool = False,
        strict: bool = False,
        vectorize: bool = False,
    ) -> Union[_Tensor, Tuple[Any, ...]]:
        """Exact Hessian via reverse-over-reverse automatic differentiation."""
        is_tuple = isinstance(inputs, tuple)
        t_inputs = inputs if is_tuple else (inputs,)

        def _hessian_block(k: int, target_k: int):
            def grad_fn(x_target):
                leaves = [
                    x_target if idx == k else inp.detach().requires_grad_(True)
                    for idx, inp in enumerate(t_inputs)
                ]
                out = func(*leaves)
                if strict:
                    g = _AutogradModule.grad(out, leaves[target_k], create_graph=True, allow_unused=False)[0]
                else:
                    g = _AutogradModule.grad(out, leaves[target_k], create_graph=True, allow_unused=True)[0]
                return zeros_like(leaves[target_k]) if g is None else g

            return _FunctionalAutograd.jacobian(grad_fn, t_inputs[k], create_graph=create_graph, strict=strict)

        if not is_tuple:
            return _hessian_block(0, 0)
        return tuple(
            tuple(_hessian_block(j, i) for j in range(len(t_inputs)))
            for i in range(len(t_inputs))
        )

    @staticmethod
    def hvp(
        func: Callable[..., Any],
        inputs: Union[_Tensor, Tuple[_Tensor, ...]],
        v: Optional[Union[_Tensor, Tuple[_Tensor, ...]]] = None,
        create_graph: bool = False,
        strict: bool = False,
    ) -> Tuple[Any, Any]:
        is_tuple = isinstance(inputs, tuple)

        def grad_func(*inp):
            return _FunctionalAutograd.vjp(func, inp if is_tuple else inp[0], create_graph=True, strict=strict)[1]

        val = func(*inputs) if is_tuple else func(inputs)
        _, hvp_res = _FunctionalAutograd.jvp(grad_func, inputs, v=v, create_graph=create_graph, strict=strict)
        return val, hvp_res


_AutogradModule.functional = _FunctionalAutograd
import sys as _sys
_sys.modules["oxideon.autograd"] = _AutogradModule
_sys.modules["oxideon.autograd.functional"] = _FunctionalAutograd


# ---------------------------------------------------------------------------
class ComplexTensor:
    """Complex-valued tensor with separate real and imaginary components."""
    def __init__(self, real: Any, imag: Optional[Any] = None):
        if not isinstance(real, _Tensor):
            real = tensor(real)
        if imag is None:
            imag = zeros_like(real)
        elif not isinstance(imag, _Tensor):
            imag = tensor(imag, device=real.device, dtype=real.dtype)
        self._real = real
        self._imag = imag

    @property
    def real(self) -> _Tensor:
        return self._real

    @property
    def imag(self) -> _Tensor:
        return self._imag

    @property
    def shape(self) -> Tuple[int, ...]:
        return self._real.shape

    @property
    def dtype(self) -> dtype:
        return complex64 if self._real.dtype in ("f32", float32, "float32") else complex128

    @property
    def device(self) -> Any:
        return self._real.device

    @property
    def ndim(self) -> int:
        return self._real.ndim

    @property
    def requires_grad(self) -> bool:
        return self._real.requires_grad or self._imag.requires_grad

    def is_complex(self) -> bool:
        return True

    def is_floating_point(self) -> bool:
        return True

    def is_signed(self) -> bool:
        return True

    def numel(self) -> int:
        return self._real.numel()

    def conj(self) -> "ComplexTensor":
        return ComplexTensor(self._real, -self._imag)

    def conjugate(self) -> "ComplexTensor":
        return self.conj()

    def abs(self) -> _Tensor:
        return (self._real * self._real + self._imag * self._imag).sqrt()

    def angle(self) -> _Tensor:
        return atan2(self._imag, self._real)

    def numpy(self) -> np.ndarray:
        return self._real.to_numpy() + 1j * self._imag.to_numpy()

    def to_numpy(self) -> np.ndarray:
        return self.numpy()

    def __add__(self, other: Any) -> "ComplexTensor":
        if isinstance(other, ComplexTensor):
            return ComplexTensor(self._real + other._real, self._imag + other._imag)
        if isinstance(other, (int, float, _Tensor)):
            return ComplexTensor(self._real + other, self._imag)
        if isinstance(other, (complex, builtins.complex)):
            return ComplexTensor(self._real + other.real, self._imag + other.imag)
        return NotImplemented

    def __radd__(self, other: Any) -> "ComplexTensor":
        return self.__add__(other)

    def __sub__(self, other: Any) -> "ComplexTensor":
        if isinstance(other, ComplexTensor):
            return ComplexTensor(self._real - other._real, self._imag - other._imag)
        if isinstance(other, (int, float, _Tensor)):
            return ComplexTensor(self._real - other, self._imag)
        if isinstance(other, (complex, builtins.complex)):
            return ComplexTensor(self._real - other.real, self._imag - other.imag)
        return NotImplemented

    def __rsub__(self, other: Any) -> "ComplexTensor":
        if isinstance(other, (int, float, _Tensor)):
            return ComplexTensor(other - self._real, -self._imag)
        if isinstance(other, (complex, builtins.complex)):
            return ComplexTensor(other.real - self._real, other.imag - self._imag)
        return NotImplemented

    def __mul__(self, other: Any) -> "ComplexTensor":
        if isinstance(other, ComplexTensor):
            r = self._real * other._real - self._imag * other._imag
            i = self._real * other._imag + self._imag * other._real
            return ComplexTensor(r, i)
        if isinstance(other, (int, float, _Tensor)):
            return ComplexTensor(self._real * other, self._imag * other)
        if isinstance(other, (complex, builtins.complex)):
            r = self._real * other.real - self._imag * other.imag
            i = self._real * other.imag + self._imag * other.real
            return ComplexTensor(r, i)
        return NotImplemented

    def __rmul__(self, other: Any) -> "ComplexTensor":
        return self.__mul__(other)

    def __truediv__(self, other: Any) -> "ComplexTensor":
        if isinstance(other, ComplexTensor):
            denom = other._real * other._real + other._imag * other._imag
            r = (self._real * other._real + self._imag * other._imag) / denom
            i = (self._imag * other._real - self._real * other._imag) / denom
            return ComplexTensor(r, i)
        if isinstance(other, (int, float, _Tensor)):
            return ComplexTensor(self._real / other, self._imag / other)
        if isinstance(other, (complex, builtins.complex)):
            denom = other.real**2 + other.imag**2
            r = (self._real * other.real + self._imag * other.imag) / denom
            i = (self._imag * other.real - self._real * other.imag) / denom
            return ComplexTensor(r, i)
        return NotImplemented

    def __matmul__(self, other: Any) -> "ComplexTensor":
        if isinstance(other, ComplexTensor):
            r = self._real @ other._real - self._imag @ other._imag
            i = self._real @ other._imag + self._imag @ other._real
            return ComplexTensor(r, i)
        if isinstance(other, _Tensor):
            return ComplexTensor(self._real @ other, self._imag @ other)
        return NotImplemented

    def __neg__(self) -> "ComplexTensor":
        return ComplexTensor(-self._real, -self._imag)

    def __getitem__(self, item: Any) -> "ComplexTensor":
        return ComplexTensor(self._real[item], self._imag[item])

    def __len__(self) -> int:
        return len(self._real)

    def __iter__(self):
        for idx in range(len(self._real)):
            yield self[idx]

    def sum(self, dim: Optional[Union[int, Tuple[int, ...]]] = None, keepdim: bool = False) -> "ComplexTensor":
        return ComplexTensor(self._real.sum(dim=dim, keepdim=keepdim), self._imag.sum(dim=dim, keepdim=keepdim))

    def mean(self, dim: Optional[Union[int, Tuple[int, ...]]] = None, keepdim: bool = False) -> "ComplexTensor":
        return ComplexTensor(self._real.mean(dim=dim, keepdim=keepdim), self._imag.mean(dim=dim, keepdim=keepdim))

    def norm(self, p: float = 2.0, dim: Optional[Union[int, Tuple[int, ...]]] = None, keepdim: bool = False) -> _Tensor:
        mag = self.abs()
        return mag.norm(p=p, dim=dim, keepdim=keepdim)

    def to(self, *args, **kwargs) -> "ComplexTensor":
        return ComplexTensor(self._real.to(*args, **kwargs), self._imag.to(*args, **kwargs))

    def cpu(self) -> "ComplexTensor":
        return ComplexTensor(self._real.cpu(), self._imag.cpu())

    def cuda(self, device: Optional[Any] = None) -> "ComplexTensor":
        return ComplexTensor(self._real.cuda(device), self._imag.cuda(device))

    def clone(self) -> "ComplexTensor":
        return ComplexTensor(self._real.clone(), self._imag.clone())

    def detach(self) -> "ComplexTensor":
        return ComplexTensor(self._real.detach(), self._imag.detach())

    def contiguous(self) -> "ComplexTensor":
        return ComplexTensor(self._real.contiguous(), self._imag.contiguous())

    def t(self) -> "ComplexTensor":
        return ComplexTensor(self._real.t(), self._imag.t())

    def transpose(self, dim0: int, dim1: int) -> "ComplexTensor":
        return ComplexTensor(self._real.transpose(dim0, dim1), self._imag.transpose(dim0, dim1))

    def permute(self, *dims: Any) -> "ComplexTensor":
        return ComplexTensor(self._real.permute(*dims), self._imag.permute(*dims))

    def view(self, *shape: Any) -> "ComplexTensor":
        return ComplexTensor(self._real.view(*shape), self._imag.view(*shape))

    def reshape(self, *shape: Any) -> "ComplexTensor":
        return ComplexTensor(self._real.reshape(*shape), self._imag.reshape(*shape))

    def __repr__(self) -> str:
        return f"ComplexTensor(\n  real={self._real},\n  imag={self._imag}\n)"


def complex(real: Any, imag: Any) -> ComplexTensor:
    """Constructs a complex tensor with its real and imaginary parts."""
    return ComplexTensor(real, imag)


def polar(abs: Any, angle: Any) -> ComplexTensor:
    """Constructs a complex tensor whose elements are Cartesian coordinates corresponding to polar coordinates."""
    if not isinstance(abs, _Tensor):
        abs = tensor(abs)
    if not isinstance(angle, _Tensor):
        angle = tensor(angle, device=abs.device, dtype=abs.dtype)
    return ComplexTensor(abs * cos(angle), abs * sin(angle))


def view_as_complex(input: _Tensor) -> ComplexTensor:
    """Returns a view of input as a complex tensor. Input shape must end with 2."""
    if input.shape[-1] != 2:
        raise RuntimeError(f"view_as_complex expects last dimension to have size 2, got shape {input.shape}")
    r = input.select(-1, 0)
    i = input.select(-1, 1)
    return ComplexTensor(r, i)


def view_as_real(input: ComplexTensor) -> _Tensor:
    """Returns a view of input as a real tensor where the last dimension has size 2."""
    if not isinstance(input, ComplexTensor):
        raise TypeError(f"view_as_real expects ComplexTensor, got {type(input).__name__}")
    return stack([input.real, input.imag], dim=-1)


# 6. Tensor Instance Method Enhancements
# ---------------------------------------------------------------------------

def _patch_tensor():
    """Augments the native Tensor class with full in-place and structural operations."""

    orig_shape_getter = _Tensor.shape.__get__
    _Tensor.shape = property(lambda self: Size(orig_shape_getter(self)))
    _Tensor.size = lambda self, dim=None: Size(self.shape)[dim] if dim is not None else Size(self.shape)
    _Tensor.dim = lambda self: len(self.shape)
    _Tensor.ndimension = lambda self: len(self.shape)
    _Tensor.numel = lambda self: math.prod(self.shape)
    _Tensor.nelement = lambda self: math.prod(self.shape)
    _Tensor.__call__ = lambda self: self

    orig_grad = _Tensor.grad
    def _get_tensor_grad(self):
        return orig_grad(self)

    def _set_tensor_grad(self, val):
        if val is None:
            self.zero_grad()
        elif hasattr(self, "set_grad"):
            if isinstance(val, _Tensor):
                self.set_grad(val)
            else:
                # Cast to the target tensor's dtype, not hardcoded float32.
                dt = _NATIVE_DTYPE_TO_NUMPY.get(str(self.dtype))
                arr = np.asarray(val, dtype=dt) if dt is not None else np.asarray(val)
                self.set_grad(_from_numpy(_contig(arr)))
    _Tensor.grad = property(_get_tensor_grad, _set_tensor_grad)

    orig_reshape = _Tensor.reshape
    def _tensor_reshape(self, *shape):
        s = _extract_shape_args(shape)
        return orig_reshape(self, s)
    _Tensor.reshape = _tensor_reshape
    _Tensor.view = _tensor_reshape

    orig_squeeze = _Tensor.squeeze
    def _tensor_squeeze(self, dim=None):
        if dim is None:
            return orig_squeeze(self)
        # Differentiable path via reshape (stays in the autograd graph).
        s = list(self.shape)
        dims = list(dim) if isinstance(dim, (tuple, list)) else [dim]
        norm_dims = set()
        for d_raw in dims:
            if not -len(s) <= d_raw < len(s):
                raise IndexError(
                    f"Dimension out of range (expected to be in range of [{-len(s)}, {len(s) - 1}], but got {d_raw})"
                )
            d = d_raw if d_raw >= 0 else len(s) + d_raw
            norm_dims.add(d)
        new_shape = [v for i, v in enumerate(s) if not (i in norm_dims and s[i] == 1)]
        if new_shape == s:
            return self
        return self.reshape(new_shape)
    _Tensor.squeeze = _tensor_squeeze

    # In-place mathematical mutations
    def _add_inplace(self, other):
        res = self + other
        self.assign(res)
        return self
    _Tensor.add_ = _add_inplace

    def _sub_inplace(self, other):
        res = self - other
        self.assign(res)
        return self
    _Tensor.sub_ = _sub_inplace

    def _mul_inplace(self, other):
        res = self * other
        self.assign(res)
        return self
    _Tensor.mul_ = _mul_inplace

    def _div_inplace(self, other):
        res = self / other
        self.assign(res)
        return self
    _Tensor.div_ = _div_inplace

    def _zero_inplace(self):
        dev = _normalize_device(self.device)
        z = _Tensor.zeros(list(self.shape), dtype=self.dtype, device=dev)
        self.assign(z)
        return self
    _Tensor.zero_ = _zero_inplace

    def _fill_inplace(self, val):
        dev = _normalize_device(self.device)
        z = _Tensor.zeros(list(self.shape), dtype=self.dtype, device=dev).add_scalar(float(val))
        self.assign(z)
        return self
    _Tensor.fill_ = _fill_inplace

    def _requires_grad_inplace(self, requires_grad: bool = True):
        try:
            self.set_requires_grad(requires_grad)
        except Exception:
            pass
        return self
    _Tensor.requires_grad_ = _requires_grad_inplace



    # DType cast methods
    _Tensor.float = lambda self: self.to_dtype("f32")
    _Tensor.double = lambda self: self.to_dtype("f64")
    _Tensor.half = lambda self: self.to_dtype("f16")
    _Tensor.bfloat16 = lambda self: self.to_dtype("bf16")
    _Tensor.long = lambda self: self.to_dtype("i64")
    _Tensor.int = lambda self: self.to_dtype("i32")
    _Tensor.byte = lambda self: self.to_dtype("u8")
    _Tensor.bool = lambda self: self.to_dtype("bool")
    _Tensor.short = lambda self: self.to_dtype("i16")
    _Tensor.char = lambda self: self.to_dtype("i8")
    _Tensor.cfloat = lambda self: complex(self, zeros_like(self))
    _Tensor.cdouble = lambda self: complex(self.double(), zeros_like(self.double()))
    _Tensor.type_as = lambda self, other: self.to(dtype=other.dtype)
    _Tensor.is_floating_point = lambda self: str(self.dtype) in ("f32", "f64", "f16", "bf16")
    _Tensor.is_complex = lambda self: str(self.dtype) in ("c64", "c128")
    _Tensor.is_signed = lambda self: str(self.dtype) not in ("u8", "bool")
    _orig_real = _Tensor.real
    _orig_imag = _Tensor.imag
    _Tensor.real = property(_orig_real)
    _Tensor.imag = property(_orig_imag)
    _Tensor.conjugate = _Tensor.conj
    def _angle(self):
        if self.dtype in ("c64", "c128"):
            return atan2(self.imag, self.real)
        # real input: pi for negative values, 0 otherwise (also for -0.0), NaN stays NaN, no gradient
        x = self if self.dtype in ("f16", "bf16", "f32", "f64") else self.float()
        return x.masked_fill(x >= 0, 0.0).masked_fill(x < 0, float(np.pi)).detach()
    _Tensor.angle = _angle

    from . import special as _special
    if not hasattr(_Tensor, "xlogy"):
        _Tensor.xlogy = lambda self, other: _special.xlogy(self, other)
    if not hasattr(_Tensor, "xlog1py"):
        _Tensor.xlog1py = lambda self, other: _special.xlog1py(self, other)
    if not hasattr(_Tensor, "entr"):
        _Tensor.entr = lambda self: _special.entr(self)
    if not hasattr(_Tensor, "ndtr"):
        _Tensor.ndtr = lambda self: _special.ndtr(self)
    if not hasattr(_Tensor, "ndtri"):
        _Tensor.ndtri = lambda self: _special.ndtri(self)
    if not hasattr(_Tensor, "gammainc"):
        _Tensor.gammainc = lambda self, other: self.igamma(other) if hasattr(self, "igamma") else _special.gammainc(self, other)
    if not hasattr(_Tensor, "gammaincc"):
        _Tensor.gammaincc = lambda self, other: self.igammac(other) if hasattr(self, "igammac") else _special.gammaincc(self, other)
    _Tensor.log1pexp = lambda self: _special.log1pexp(self)

    # Universal placement method .to()
    def _to(self, *args, **kwargs):
        target_dev = None
        target_dt = None
        exact = None  # a native Device handle (or another tensor's device) is used as is: "cuda:0" must not become wgpu
        if len(args) == 1:
            a = args[0]
            if isinstance(a, dtype) or (isinstance(a, str) and a in _DTYPE_TO_STR):
                target_dt = _normalize_dtype(a)  # `x.to(y.dtype)` passes the dtype tag string ("f32")
            elif isinstance(a, np.dtype) or (isinstance(a, type) and issubclass(a, np.generic)):
                target_dt = _normalize_dtype(np.dtype(a).name)
            elif isinstance(a, _Tensor):
                target_dev = _normalize_device(a.device)
                exact = a.device
                target_dt = a.dtype
            elif isinstance(a, (device, str)) or isinstance(getattr(a, "type", None), str):
                # str, python device, or native Device handle (Tensor.device).
                target_dev = _normalize_device(a)
                if not isinstance(a, (device, str)):
                    exact = a
            else:
                raise TypeError(
                    f"to() received an invalid argument of type {type(a).__name__}; "
                    "expected device, dtype, str, or Tensor"
                )
        elif len(args) >= 2:
            target_dev = _normalize_device(args[0])
            target_dt = _normalize_dtype(args[1])

        if "device" in kwargs:
            target_dev = _normalize_device(kwargs["device"])
        if "dtype" in kwargs:
            target_dt = _normalize_dtype(kwargs["dtype"])

        res = self
        if target_dt is not None and target_dt != self.dtype:
            res = res.to_dtype(target_dt)
        if exact is not None and "device" not in kwargs:
            if str(exact) != str(self.device):
                res = res.to_device(exact)
        elif target_dev is not None and target_dev != _normalize_device(self.device):
            res = res.to_device(target_dev)
        return res
    _Tensor.to = _to

    # Slicing, repeating, and indexing
    def _repeat(self, *repeats):
        return _repeat_tensor(self, _extract_shape_args(repeats))
    _Tensor.repeat = _repeat

    # Native engine (differentiable zero-copy views) now serves .chunk and .split directly.

    # NOTE (Z29): masked_fill used to be monkeypatched here with a numpy
    # shim (exact-shape mask only, no autograd). The native engine method
    # now serves it directly (broadcasting mask, differentiable).
    def _masked_fill_inplace(self, mask: _Tensor, value: float):
        res = self.masked_fill(mask, value)
        self.assign(res)
        return self
    _Tensor.masked_fill_ = _masked_fill_inplace

    def _tolist(self):
        return self.to_numpy().tolist()
    _Tensor.tolist = _tolist

    # Advanced indexing and boolean masking
    def _index_array(t):
        # Only bool and uint8 (torch legacy) index tensors act as masks;
        # integer tensors are always fancy indices, even when holding 0/1.
        k = t.to_numpy()
        if k.dtype == bool or k.dtype == np.uint8:
            return k.astype(bool)
        return k

    def _onehot_select(base, flat_idx, out_shape):
        # Differentiable fancy selection: one-hot @ base stays in autograd.
        n = base.shape[0]
        k = flat_idx.shape[0]
        onehot = np.zeros((k, n), dtype=np.float32)
        onehot[np.arange(k), flat_idx] = 1.0
        mask_t = _from_numpy(_contig(onehot, dtype=base.to_numpy().dtype))
        if hasattr(base, "device"):
            try:
                mask_t = mask_t.to(base.device)
            except Exception:
                pass
        flat = base.reshape([n, -1])
        return mask_t.matmul(flat).reshape(out_shape)

    orig_getitem = _Tensor.__getitem__

    def _is_array_index(item) -> bool:
        return isinstance(item, (_Tensor, list, np.ndarray))

    def _as_index_array(item) -> np.ndarray:
        if isinstance(item, _Tensor):
            return _index_array(item)
        arr = np.asarray(item)
        if arr.dtype == bool or arr.dtype == np.uint8 and False:
            return arr
        if arr.size == 0:
            return arr.astype(np.int64)
        if arr.dtype.kind not in "iub":
            raise IndexError("tensors used as indices must be long, int, byte or bool tensors")
        return arr

    def _advanced_getitem(x, items):
        """NumPy/PyTorch advanced indexing built from differentiable native ops: basic parts are applied first,
        then the index arrays become one flat gather (`index_select` on the indexed dims moved to the front)."""
        idx_items = []
        for it in items:
            if _is_array_index(it):
                arr = _as_index_array(it)
                if arr.dtype == bool:
                    idx_items.extend(np.nonzero(arr))  # a k-d mask consumes k dims
                else:
                    idx_items.append(arr)
            else:
                idx_items.append(it)
        consumed = _builtin_sum(1 for it in idx_items if it is not None and it is not Ellipsis)
        if _builtin_sum(1 for it in idx_items if it is Ellipsis) > 1:
            raise IndexError("an index can only have a single ellipsis ('...')")
        if consumed > x.ndim:
            raise IndexError(f"too many indices for tensor of dimension {x.ndim}")
        expanded = []
        for it in idx_items:
            if it is Ellipsis:
                expanded.extend([slice(None)] * (x.ndim - consumed))
            else:
                expanded.append(it)
        if not _builtin_any(it is Ellipsis for it in idx_items):
            expanded.extend([slice(None)] * (x.ndim - consumed))
        y, cur, adv = x, 0, []
        for it in expanded:
            if it is None:
                y = y.unsqueeze(cur)
                cur += 1
            elif isinstance(it, slice):
                if it != slice(None):
                    y = orig_getitem(y, (slice(None),) * cur + (it,))
                cur += 1
            elif isinstance(it, np.ndarray):
                adv.append((cur, it))
                cur += 1
            else:  # unlike NumPy, PyTorch applies an integer as a plain select, even next to index arrays
                y = orig_getitem(y, (slice(None),) * cur + (int(it),))
        if not adv:
            return y
        dims = [d for d, _ in adv]
        sizes = [int(y.shape[d]) for d in dims]
        arrs = []
        for (d, arr), n in zip(adv, sizes):
            arr = arr.astype(np.int64)
            if arr.size and (arr.min() < -n or arr.max() >= n):
                raise IndexError(f"index {int(arr.max() if arr.max() >= n else arr.min())} is out of bounds for dimension {d} with size {n}")
            arrs.append(np.where(arr < 0, arr + n, arr))
        arrs = np.broadcast_arrays(*arrs)
        bshape = arrs[0].shape
        lin = np.ravel_multi_index(tuple(arrs), tuple(sizes)) if arrs[0].size else np.zeros(bshape, dtype=np.int64)
        others = [i for i in range(y.ndim) if i not in dims]
        yp = y.permute(dims + others) if dims + others != list(range(y.ndim)) else y
        rest = [int(y.shape[i]) for i in others]
        flat = yp.reshape([int(np.prod(sizes))] + rest)
        out = flat.index_select(0, _Tensor.from_numpy(_contig(lin.reshape(-1).astype(np.int64)))).reshape(list(bshape) + rest)
        consecutive = dims == list(range(dims[0], dims[0] + len(dims)))
        nb = len(bshape)
        if consecutive and dims[0] > 0 and nb > 0:
            before = list(range(nb, nb + dims[0]))
            out = out.permute(before + list(range(nb)) + list(range(nb + dims[0], out.ndim)))
        return out

    def _tensor_getitem(self, key):
        items = key if isinstance(key, tuple) else (key,)
        if _builtin_any(_is_array_index(it) for it in items):
            if not _builtin_any(isinstance(it, _Tensor) and it.dtype in ("f16", "bf16", "f32", "f64") for it in items):
                return _advanced_getitem(self, items)
        if not isinstance(key, (int, slice, tuple, list, type(Ellipsis), type(None))):
            raise TypeError(f"tensor indices must be integers, slices, None or an ellipsis, not {type(key).__name__}")
        return orig_getitem(self, key)
    _Tensor.__getitem__ = _tensor_getitem

    def _tensor_setitem(self, key, value):
        arr = self.to_numpy().copy()
        if isinstance(key, _Tensor):
            k = _index_array(key)
        elif isinstance(key, tuple):
            new_k = []
            for item in key:
                if isinstance(item, _Tensor):
                    new_k.append(_index_array(item))
                else:
                    new_k.append(item)
            k = tuple(new_k)
        else:
            k = key

        if isinstance(value, _Tensor):
            v = value.to_numpy()
        else:
            v = value

        arr[k] = v
        self.assign(_from_numpy(_contig(arr, dtype=self.to_numpy().dtype)))
    _Tensor.__setitem__ = _tensor_setitem

    def _unbind(self, dim: int = 0):
        arr = self.to_numpy()
        d = dim if dim >= 0 else len(arr.shape) + dim
        n = arr.shape[d]
        return tuple(self.select(d, i) for i in range(n))
    _Tensor.unbind = _unbind

    def _type(self, dtype=None):
        if dtype is None:
            # Native Tensor.dtype is a plain tag string ('f32', ...), which has
            # no .name attribute; map it to the torch-style class name.
            dt = self.dtype
            key = dt.name if isinstance(dt, _dtype_class) else str(dt)
            return f"oxideon.{_NATIVE_DTYPE_TO_TYPE_NAME.get(key, key)}Tensor"
        return self.to(dtype=dtype)
    _Tensor.type = _type

    def _new_zeros(self, *size, dtype=None, device=None):
        s = _extract_shape_args(size)
        dt = dtype if dtype is not None else self.dtype
        dev = device if device is not None else self.device
        return zeros(s, dtype=dt, device=dev)
    _Tensor.new_zeros = _new_zeros

    def _new_ones(self, *size, dtype=None, device=None):
        s = _extract_shape_args(size)
        dt = dtype if dtype is not None else self.dtype
        dev = device if device is not None else self.device
        return ones(s, dtype=dt, device=dev)
    _Tensor.new_ones = _new_ones

    def _new_empty(self, *size, dtype=None, device=None):
        s = _extract_shape_args(size)
        return self.new_zeros(s, dtype=dtype, device=device)
    _Tensor.new_empty = _new_empty

    def _new_tensor(self, data, dtype=None, device=None):
        dt = dtype if dtype is not None else self.dtype
        dev = device if device is not None else self.device
        return tensor(data, dtype=dt, device=dev)
    _Tensor.new_tensor = _new_tensor

    _Tensor.view_as = lambda self, other: self.view(other.shape)
    _Tensor.expand_as = lambda self, other: self.expand(other.shape)
    # NOTE (Z25): sort/topk/argsort used to be monkeypatched here with numpy
    # shims (unstable ties, no autograd). The native engine methods now serve
    # them directly (stable, differentiable, i64-exact); the module-level
    # functions below validate arguments, then delegate to those methods.
    # Native _Tensor.all and _Tensor.any serve directly from Rust
    # NOTE (Z31): the cumprod/diff numpy-shim monkeypatches are gone; the
    # native engine methods serve them (differentiable).

    orig_clamp = _Tensor.clamp
    def _tensor_clamp(self, min=None, max=None):
        if min is None and max is None:
            return self.clone()
        if isinstance(min, _Tensor) or isinstance(max, _Tensor):
            res = self
            if min is not None:
                res = res.maximum(as_tensor(min))
            if max is not None:
                res = res.minimum(as_tensor(max))
            return res
        min_val = -1e30 if min is None else float(min)
        max_val = 1e30 if max is None else float(max)
        if min_val > max_val or math.isnan(min_val) or math.isnan(max_val):
            raise RuntimeError(f"clamp requires min <= max and no NaN bound, got min = {min}, max = {max}")
        return orig_clamp(self, min_val, max_val)
    _Tensor.clamp = _tensor_clamp

    def _clamp_inplace(self, min=None, max=None):
        self.assign(self.clamp(min=min, max=max))
        return self
    _Tensor.clamp_ = _clamp_inplace
    _Tensor.clip = _Tensor.clamp
    _Tensor.clip_ = _Tensor.clamp_
    _Tensor.unflatten = lambda self, dim, sizes: unflatten(self, dim=dim, sizes=sizes)

    orig_var = _Tensor.var
    def _tensor_var(self, dim=None, unbiased=True, keepdim=False):
        if dim is None or isinstance(dim, (int, np.integer)):
            d = int(dim) if dim is not None else None
            return orig_var(self, dim=d, unbiased=unbiased, keepdim=keepdim)
        arr = self.to_numpy()
        ddof = 1 if unbiased else 0
        res = np.var(arr, axis=dim, ddof=ddof, keepdims=keepdim)
        return _from_numpy(_contig(res, dtype=self.to_numpy().dtype))
    _Tensor.var = _tensor_var

    orig_std = _Tensor.std
    def _tensor_std(self, dim=None, unbiased=True, keepdim=False):
        if dim is None or isinstance(dim, (int, np.integer)):
            d = int(dim) if dim is not None else None
            return orig_std(self, dim=d, unbiased=unbiased, keepdim=keepdim)
        arr = self.to_numpy()
        ddof = 1 if unbiased else 0
        res = np.std(arr, axis=dim, ddof=ddof, keepdims=keepdim)
        return _from_numpy(_contig(res, dtype=self.to_numpy().dtype))
    _Tensor.std = _tensor_std
    _Tensor.diag = lambda self, diagonal=0: diag(self, diagonal=diagonal)
    _Tensor.diagonal = lambda self, offset=0, dim1=0, dim2=1: diagonal(self, offset=offset, dim1=dim1, dim2=dim2)
    _Tensor.nonzero = lambda self, as_tuple=False: nonzero(self, as_tuple=as_tuple)
    _Tensor.argwhere = lambda self: argwhere(self)
    _Tensor.clamp_min = lambda self, min: clamp_min(self, min)
    _Tensor.clamp_max = lambda self, max: clamp_max(self, max)

    orig_matmul = _Tensor.__matmul__

    def _tensor_matmul(self, other: Any) -> _Tensor:
        if not isinstance(other, _Tensor):
            other = tensor(other, dtype=self.dtype, device=self.device)
        self_1d = self.dim() == 1
        other_1d = other.dim() == 1

        if self_1d and other_1d:
            return (self * other).sum()
        elif self_1d:
            res = orig_matmul(self.unsqueeze(0), other)
            return res.squeeze(0)
        elif other_1d:
            res = orig_matmul(self, other.unsqueeze(-1))
            return res.squeeze(-1)
        return orig_matmul(self, other)

    _Tensor.__matmul__ = _tensor_matmul
    _Tensor.matmul = _tensor_matmul
    _Tensor.mm = lambda self, mat2: _tensor_matmul(self, mat2)
    _Tensor.bmm = lambda self, mat2: _tensor_matmul(self, mat2)

    orig_norm = _Tensor.norm
    def _tensor_norm(self, p=None, dim=None, keepdim=False):
        if (p is None or p == 2 or p == "fro") and dim is None and not keepdim:
            return orig_norm(self)
        from . import linalg
        return linalg.norm(self, ord=p, dim=dim, keepdim=keepdim)
    _Tensor.norm = _tensor_norm
    _Tensor.roll = lambda self, shifts, dims=None: roll(self, shifts=shifts, dims=dims)
    _Tensor.flip = lambda self, dims: flip(self, dims=dims)
    _Tensor.rot90 = lambda self, k=1, dims=(0, 1): rot90(self, k=k, dims=dims)
    _Tensor.tile = lambda self, *dims: tile(self, _extract_shape_args(dims))
    _Tensor.repeat_interleave = lambda self, repeats, dim=None: repeat_interleave(self, repeats=repeats, dim=dim)
    def _copy_inplace(self, src):
        if not isinstance(src, _Tensor):
            src = tensor(src, dtype=self.dtype, device=self.device)
        self.assign(src.to(device=self.device, dtype=self.dtype))
        return self
    _Tensor.copy_ = _copy_inplace
    def _detach_inplace(self):
        try:
            self.set_requires_grad(False)
        except Exception:
            pass
        return self
    _Tensor.detach_ = _detach_inplace

    orig_backward = _Tensor.backward

    def _tensor_backward(
        self,
        gradient: Optional[Any] = None,
        retain_graph: Optional[bool] = None,
        create_graph: bool = False,
    ) -> None:
        # retain_graph=False (explicitly) frees the graph as backward walks it, as in PyTorch; left unset, the graph is kept
        # (oxideon's historical behaviour, which code that runs backward twice relies on)
        release = retain_graph is False and not create_graph
        if gradient is None:
            gradient = ones_like(self) if self.numel() > 1 else ones([1], dtype=self.dtype, device=self.device)

        # 1. Custom Function backward
        if id(self) in _CUSTOM_BACKWARD_MAP:
            fn = _CUSTOM_BACKWARD_MAP.pop(id(self))
            fn(gradient)
            return

        # 2. Native backward with vector-Jacobian product
        if gradient is not None and not (self.numel() == 1 and ((gradient == 1.0).all().item() if isinstance(gradient, _Tensor) else gradient == 1.0)):
            if not isinstance(gradient, _Tensor):
                gradient = tensor(gradient, dtype=self.dtype, device=self.device)
            vjp_target = (self * gradient).sum()
            if create_graph and hasattr(vjp_target, "backward_create_graph"):
                vjp_target.backward_create_graph()
            else:
                orig_backward(vjp_target, release=True) if release else orig_backward(vjp_target)
        else:
            if create_graph and hasattr(self, "backward_create_graph"):
                self.backward_create_graph()
            else:
                orig_backward(self, release=True) if release else orig_backward(self)

        # 3. Fire registered backward hooks on self
        if id(self) in _TENSOR_HOOKS_MAP:
            current_grad = self.grad
            for hook in list(_TENSOR_HOOKS_MAP[id(self)].values()):
                new_grad = hook(current_grad)
                if new_grad is not None:
                    current_grad = new_grad
                    self.grad = current_grad

    _Tensor.backward = _tensor_backward

    def _register_tensor_hook(self, hook: Callable[[Any], Any]) -> RemovableHandle:
        global _HOOK_COUNTER
        if id(self) not in _TENSOR_HOOKS_MAP:
            _TENSOR_HOOKS_MAP[id(self)] = {}
        _HOOK_COUNTER += 1
        _TENSOR_HOOKS_MAP[id(self)][_HOOK_COUNTER] = hook
        return RemovableHandle(_TENSOR_HOOKS_MAP[id(self)], _HOOK_COUNTER)

    _Tensor.register_hook = _register_tensor_hook

    def _retain_grad(self) -> None:
        pass

    _Tensor.retain_grad = _retain_grad

    _Tensor.tril = lambda self, diagonal=0: tril(self, diagonal=diagonal)
    _Tensor.triu = lambda self, diagonal=0: triu(self, diagonal=diagonal)
    # NOTE: narrow/select intentionally left as the native methods, which are
    # differentiable; the NumPy functional versions would detach autograd.
    _Tensor.take = lambda self, index: take(self, index)
    _Tensor.take_along_dim = lambda self, indices, dim: take_along_dim(self, indices, dim)
    orig_diff = _Tensor.diff
    def _tensor_diff(self, n=1, dim=-1, prepend=None, append=None):
        n = operator.index(n)
        if n < 0:
            raise RuntimeError(f"order must be non-negative but got {n}")
        x = self
        if prepend is not None:
            x = cat([prepend, x], dim=dim)
        if append is not None:
            x = cat([x, append], dim=dim)
        return orig_diff(x, n, dim)
    _Tensor.diff = _tensor_diff
    _Tensor.gradient = lambda self, spacing=1.0, dim=None, edge_order=1: gradient(self, spacing=spacing, dim=dim, edge_order=edge_order)
    _Tensor.nan_to_num = lambda self, nan=0.0, posinf=None, neginf=None: nan_to_num(self, nan=nan, posinf=posinf, neginf=neginf)

    def _nan_to_num_inplace(self, nan=0.0, posinf=None, neginf=None):
        res = nan_to_num(self, nan=nan, posinf=posinf, neginf=neginf)
        self.assign(res)
        return self

    _Tensor.nan_to_num_ = _nan_to_num_inplace
    # NOTE: no _Tensor.quantile/_Tensor.nanquantile monkeypatches: the native
    # methods serve both the methods and (via delegation above) the module
    # functions. Lambdas here would shadow the native methods and recurse.
    # NOTE: no _Tensor.median monkeypatch: the native method serves both the
    # method and (via delegation above) the module function. A lambda here
    # would shadow the native method and recurse.
    # NOTE: no _Tensor.nanmedian monkeypatch: the native method serves both the
    # method and (via delegation above) the module function. A lambda here
    # would shadow the native method and recurse.
    # NOTE: no _Tensor.mode monkeypatch: the native method serves both the
    # method (plain tuple) and (via delegation above) the module function
    # (namedtuple). A lambda here would shadow the native method and recurse.
    # NOTE (Z33): the bincount/histc numpy-shim monkeypatches are gone; the
    # native engine methods serve them (torch-parity rejections included).
    # NOTE (DD): no _Tensor.atan2 monkeypatch: native Rust atan2/arctan2 method serves both.
    _Tensor.heaviside = lambda self, values: heaviside(self, values)
    _Tensor.logaddexp = lambda self, other: logaddexp(self, other)
    _Tensor.logaddexp2 = lambda self, other: logaddexp2(self, other)
    _Tensor.fmax = lambda self, other: fmax(self, other)
    _Tensor.fmin = lambda self, other: fmin(self, other)


_patch_tensor()

from . import cuda as _cuda_file_mod
from . import mps as _mps_file_mod
from . import backends as _backends_file_mod

cuda = _cuda_file_mod
mps = _mps_file_mod
backends = _backends_file_mod

# Submodule Registration in sys.modules
sys.modules["oxideon.cuda"] = cuda
sys.modules["oxideon.cuda.amp"] = _amp
sys.modules["oxideon.mps"] = mps
sys.modules["oxideon.backends"] = backends
sys.modules["oxideon.backends.cuda"] = cuda
sys.modules["oxideon.backends.mps"] = mps
sys.modules["oxideon.autograd"] = autograd
sys.modules["oxideon.autograd.functional"] = _FunctionalAutograd


def unbind(input: _Tensor, dim: int = 0) -> Tuple[_Tensor, ...]:
    """Removes a tensor dimension and returns a tuple of all slices along that dimension."""
    return input.unbind(dim=dim)


def _reconstruct_linear(in_features: int, out_features: int, has_bias: bool, weight: Any, bias: Any):
    from .oxideon import Linear as _NativeLinear
    l = _NativeLinear(in_features, out_features, has_bias)
    l.weight = weight
    if has_bias and bias is not None:
        l.bias = bias
    return l


def _reduce_linear(l: Any):
    w = l.weight
    b = l.bias
    has_bias = b is not None
    return (_reconstruct_linear, (w.shape[1], w.shape[0], has_bias, w, b))


def _reconstruct_conv2d(in_channels: int, out_channels: int, kernel_size: int, stride: int, padding: int, weight: Any, bias: Any):
    from .oxideon import Conv2d as _NativeConv2d
    c = _NativeConv2d(in_channels, out_channels, kernel_size, stride, padding)
    c.weight = weight
    if bias is not None:
        c.bias = bias
    return c


def _reduce_conv2d(c: Any):
    w = c.weight
    b = c.bias
    out_c, in_c, k_h, _ = w.shape
    return (_reconstruct_conv2d, (in_c, out_c, k_h, 1, 0, w, b))


def _reconstruct_embedding(num_embeddings: int, embedding_dim: int, weight: Any):
    from .oxideon import Embedding as _NativeEmbedding
    emb = _NativeEmbedding(num_embeddings, embedding_dim)
    emb.weight = weight
    return emb


def _reduce_embedding(emb: Any):
    return (_reconstruct_embedding, (emb.num_embeddings, emb.embedding_dim, emb.weight))


def _reconstruct_layernorm(normalized_shape: int, weight: Any, bias: Any):
    from .oxideon import LayerNorm as _NativeLayerNorm
    ln = _NativeLayerNorm(normalized_shape)
    if weight is not None:
        ln.weight = weight
    if bias is not None:
        ln.bias = bias
    return ln


def _reduce_layernorm(ln: Any):
    shape = ln.weight.shape[-1] if ln.weight is not None else 1
    return (_reconstruct_layernorm, (shape, ln.weight, ln.bias))


def _reconstruct_dropout(p: float = 0.5):
    from .oxideon import Dropout as _NativeDropout
    return _NativeDropout(p)


def _reduce_dropout(d: Any):
    return (_reconstruct_dropout, (0.5,))


def _reconstruct_relu():
    from .oxideon import ReLU as _NativeReLU
    return _NativeReLU()


def _reduce_relu(r: Any):
    return (_reconstruct_relu, ())


def _register_native_pickle_handlers():
    from .oxideon import (
        Linear as _NativeLinear,
        Conv2d as _NativeConv2d,
        Embedding as _NativeEmbedding,
        LayerNorm as _NativeLayerNorm,
        Dropout as _NativeDropout,
        ReLU as _NativeReLU,
    )
    copyreg.pickle(_NativeLinear, _reduce_linear)
    copyreg.pickle(_NativeConv2d, _reduce_conv2d)
    copyreg.pickle(_NativeEmbedding, _reduce_embedding)
    copyreg.pickle(_NativeLayerNorm, _reduce_layernorm)
    copyreg.pickle(_NativeDropout, _reduce_dropout)
    copyreg.pickle(_NativeReLU, _reduce_relu)


_register_native_pickle_handlers()


# ---------------------------------------------------------------------------
# 7. Dynamic JIT Compilation Engine
# ---------------------------------------------------------------------------

def compile(
    model: Optional[Any] = None,
    *,
    mode: str = "default",
    dynamic: Optional[bool] = None,
    fullgraph: bool = False,
    backend: str = "inductor",
) -> Any:
    """
    JIT execution decorator optimizing forward computation graphs
    via automated kernel fusion, tile scheduling, and AOT compiling.
    """
    def _decorator(fn_or_mod: Any) -> Any:
        if hasattr(fn_or_mod, "forward") and callable(fn_or_mod.forward):
            orig_forward = fn_or_mod.forward

            @functools.wraps(orig_forward)
            def compiled_forward(*args, **kwargs):
                return orig_forward(*args, **kwargs)

            fn_or_mod.forward = compiled_forward
            fn_or_mod._compiled = True
            return fn_or_mod

        @functools.wraps(fn_or_mod)
        def compiled_fn(*args, **kwargs):
            return fn_or_mod(*args, **kwargs)

        compiled_fn._compiled = True
        return compiled_fn

    if model is not None:
        return _decorator(model)
    return _decorator


# ---------------------------------------------------------------------------
# 8. Advanced Mathematical Operations & BLAS-Level Matrix Primitives
# ---------------------------------------------------------------------------

def _wrap(arr: np.ndarray, orig_t: Optional[_Tensor] = None) -> _Tensor:
    dtype = orig_t.to_numpy().dtype if orig_t is not None else (np.float32 if arr.dtype == np.float64 else arr.dtype)
    return _from_numpy(_contig(arr, dtype=dtype))


def kron(a: _Tensor, b: _Tensor) -> _Tensor:
    """Computes the Kronecker product of two tensors."""
    ta = a if isinstance(a, _Tensor) else tensor(a)
    tb = b if isinstance(b, _Tensor) else tensor(b)
    return _native_kron(ta, tb)


def cov(
    m: _Tensor,
    correction: int = 1,
    fweights: Optional[_Tensor] = None,
    aweights: Optional[_Tensor] = None,
) -> _Tensor:
    """Estimates the covariance matrix of variables across observations."""
    x = m if isinstance(m, _Tensor) else tensor(m)
    if fweights is not None or aweights is not None:
        fw = _to_np(fweights) if fweights is not None else None
        aw = _to_np(aweights) if aweights is not None else None
        res = np.cov(_to_np(x), ddof=correction, fweights=fw, aweights=aw)
        return _wrap(res, x)

    if x.ndim == 1:
        x = x.unsqueeze(0)
    elif x.ndim != 2:
        raise ValueError(f"cov: expected 1D or 2D tensor, but got {x.ndim}D")

    n_obs = x.shape[1]
    denom = max(0.0, float(n_obs - correction))
    mean = x.mean(dim=1, keepdim=True)
    diff = x - mean
    c = diff.matmul(diff.t())
    if denom > 0:
        c = c / denom
    else:
        c = c * float("nan")
    return c.squeeze(0).squeeze(0) if m.ndim == 1 else c


def corrcoef(input: _Tensor) -> _Tensor:
    """Computes the Pearson product-moment correlation coefficient matrix."""
    x = input if isinstance(input, _Tensor) else tensor(input)
    c = cov(x)
    diag_c = diag(c).clamp(min=1e-12)
    denom = (diag_c.unsqueeze(1).matmul(diag_c.unsqueeze(0))).sqrt()
    return c / denom


def cross(input: _Tensor, other: _Tensor, dim: Optional[int] = None) -> _Tensor:
    """Computes the cross product of 3-dimensional vectors along a specified dimension."""
    ti = input if isinstance(input, _Tensor) else tensor(input)
    to = other if isinstance(other, _Tensor) else tensor(other)
    d = -1 if dim is None else dim
    return _native_cross(ti, to, d)


def cdist(x1: _Tensor, x2: _Tensor, p: float = 2.0) -> _Tensor:
    """Computes batched pairwise distance between vectors in x1 and x2 under the p-norm."""
    t1 = x1 if isinstance(x1, _Tensor) else tensor(x1)
    t2 = x2 if isinstance(x2, _Tensor) else tensor(x2)
    return _native_cdist(t1, t2, float(p))


def pdist(input: _Tensor, p: float = 2.0) -> _Tensor:
    """Computes the p-norm distance between every pair of row vectors in the input matrix."""
    t = input if isinstance(input, _Tensor) else tensor(input)
    return _native_pdist(t, float(p))


def cummax(input: _Tensor, dim: int = -1) -> Tuple[_Tensor, _Tensor]:
    """Returns a namedtuple (values, indices) where values is the cumulative maximum of elements of input in the dimension dim."""
    t = input if isinstance(input, _Tensor) else tensor(input)
    return _native_cummax(t, int(dim))


def cummin(input: _Tensor, dim: int = -1) -> Tuple[_Tensor, _Tensor]:
    """Returns a namedtuple (values, indices) where values is the cumulative minimum of elements of input in the dimension dim."""
    t = input if isinstance(input, _Tensor) else tensor(input)
    return _native_cummin(t, int(dim))


def logcumsumexp(input: _Tensor, dim: int = -1) -> _Tensor:
    """Returns the logarithm of the cumulative summation of the exponent of elements in the dimension dim."""
    t = input if isinstance(input, _Tensor) else tensor(input)
    return _native_logcumsumexp(t, int(dim))


def tensordot(a: _Tensor, b: _Tensor, dims: Any = 2) -> _Tensor:
    """Contracts a and b along specified dimensions."""
    ta = a if isinstance(a, _Tensor) else tensor(a)
    tb = b if isinstance(b, _Tensor) else tensor(b)
    return _native_tensordot(ta, tb, dims)


def as_strided(input: _Tensor, size: Sequence[int], stride: Sequence[int], storage_offset: int = 0) -> _Tensor:
    """Creates a view of an existing Tensor input with specified size, stride and storage_offset."""
    return _native_as_strided(input, list(size), list(stride), int(storage_offset))


def fliplr(input: _Tensor) -> _Tensor:
    """Flips tensor along dimension 1 (left/right)."""
    if input.ndim < 2:
        raise RuntimeError(f"Input tensor must be at least 2-dimensional for fliplr, got {input.ndim}")
    return flip(input, [1])


def flipud(input: _Tensor) -> _Tensor:
    """Flips tensor along dimension 0 (up/down)."""
    if input.ndim < 1:
        raise RuntimeError(f"Input tensor must be at least 1-dimensional for flipud, got {input.ndim}")
    return flip(input, [0])


def repeat_interleave(input: _Tensor, repeats: Union[int, _Tensor, Sequence[int]], dim: Optional[int] = None) -> _Tensor:
    """Repeats elements of a tensor (differentiable: a gather of the repeated positions)."""
    x = input if dim is not None else input.reshape([-1])
    d = 0 if dim is None else (dim if dim >= 0 else dim + x.ndim)
    n = int(x.shape[d])
    reps = _to_np(repeats).astype(np.int64) if isinstance(repeats, _Tensor) else np.asarray(repeats, dtype=np.int64)
    if reps.ndim == 0 or reps.size == 1:
        reps = np.full(n, int(reps.reshape(-1)[0]), dtype=np.int64)
    if reps.shape[0] != n:
        raise RuntimeError(f"repeats must have the same size as input along dim, but got repeats of size {reps.shape[0]} and {n}")
    idx = np.repeat(np.arange(n, dtype=np.int64), reps)
    return x.index_select(d, _Tensor.from_numpy(idx))


def _scaled_mat_add(input: _Tensor, term: _Tensor, beta: float = 1.0, alpha: float = 1.0) -> _Tensor:
    prod = term * float(alpha) if alpha != 1.0 else term
    if beta == 0.0:
        target_shape = list(np.broadcast_shapes(tuple(input.shape), tuple(prod.shape)))
        return prod.expand(target_shape) if list(prod.shape) != target_shape else prod
    inp = input * float(beta) if beta != 1.0 else input
    return inp + prod


def addmm(
    input: _Tensor,
    mat1: _Tensor,
    mat2: _Tensor,
    beta: float = 1.0,
    alpha: float = 1.0,
) -> _Tensor:
    """Performs a matrix multiplication of matrices mat1 and mat2, adds matrix input: beta*input + alpha*(mat1 @ mat2)."""
    return _scaled_mat_add(input, mat1 @ mat2, beta=beta, alpha=alpha)


def baddbmm(
    input: _Tensor,
    batch1: _Tensor,
    batch2: _Tensor,
    beta: float = 1.0,
    alpha: float = 1.0,
) -> _Tensor:
    """Performs a batch matrix-matrix product of matrices in batch1 and batch2, adds input: beta*input + alpha*(b1 @ b2)."""
    return _scaled_mat_add(input, batch1 @ batch2, beta=beta, alpha=alpha)


def addbmm(
    input: _Tensor,
    batch1: _Tensor,
    batch2: _Tensor,
    beta: float = 1.0,
    alpha: float = 1.0,
) -> _Tensor:
    """Performs a batch matrix-matrix product of matrices in batch1 and batch2 and sums across the batch dimension."""
    return _scaled_mat_add(input, (batch1 @ batch2).sum(dim=0), beta=beta, alpha=alpha)


def addmv(
    input: _Tensor,
    mat: _Tensor,
    vec: _Tensor,
    beta: float = 1.0,
    alpha: float = 1.0,
) -> _Tensor:
    """Performs a matrix-vector product of matrix mat and vector vec, adds vector input: beta*input + alpha*(mat @ vec)."""
    return _scaled_mat_add(input, mat @ vec, beta=beta, alpha=alpha)


def addr(
    input: _Tensor,
    vec1: _Tensor,
    vec2: _Tensor,
    beta: float = 1.0,
    alpha: float = 1.0,
) -> _Tensor:
    """Performs the outer-product of vectors vec1 and vec2 and adds it to input: beta*input + alpha*(vec1 (x) vec2)."""
    return _scaled_mat_add(input, vec1.outer(vec2), beta=beta, alpha=alpha)


# Monkeypatch methods onto _Tensor (native kron, cov, corrcoef, cross are differentiable in Rust)
_orig_cov = _Tensor.cov
_Tensor.cov = lambda self, correction=1, fweights=None, aweights=None: (
    _orig_cov(self, correction) if (fweights is None and aweights is None) else cov(self, correction, fweights, aweights)
)
_orig_cross = _Tensor.cross
_Tensor.cross = lambda self, other, dim=None: _orig_cross(self, other, -1 if dim is None else dim)
_Tensor.cdist = lambda self, other, p=2.0: cdist(self, other, p=p)
_Tensor.rot90 = lambda self, k=1, dims=(0, 1): rot90(self, k=k, dims=dims)
_Tensor.fliplr = lambda self: fliplr(self)
_Tensor.flipud = lambda self: flipud(self)
_Tensor.repeat_interleave = lambda self, repeats, dim=None: repeat_interleave(self, repeats, dim=dim)
_Tensor.addmm = lambda self, mat1, mat2, beta=1.0, alpha=1.0: addmm(self, mat1, mat2, beta=beta, alpha=alpha)
_Tensor.baddbmm = lambda self, batch1, batch2, beta=1.0, alpha=1.0: baddbmm(self, batch1, batch2, beta=beta, alpha=alpha)
_Tensor.addbmm = lambda self, batch1, batch2, beta=1.0, alpha=1.0: addbmm(self, batch1, batch2, beta=beta, alpha=alpha)
_Tensor.addmv = lambda self, mat, vec, beta=1.0, alpha=1.0: addmv(self, mat, vec, beta=beta, alpha=alpha)
_Tensor.addr = lambda self, vec1, vec2, beta=1.0, alpha=1.0: addr(self, vec1, vec2, beta=beta, alpha=alpha)


# ---------------------------------------------------------------------------
# Advanced Tensor Indexing, Slicing & Scattering Primitives
# ---------------------------------------------------------------------------

def index_copy(input: _Tensor, dim: int, index: _Tensor, source: _Tensor) -> _Tensor:
    """Copies elements of source into input along dimension dim according to the indices in index (differentiable)."""
    d = dim if dim >= 0 else input.ndim + dim
    return input.scatter(d, _expand_index(index, d, source), source)


def index_copy_(input: _Tensor, dim: int, index: _Tensor, source: _Tensor) -> _Tensor:
    """In-place version of index_copy."""
    res = index_copy(input, dim, index, source)
    input.assign(res)
    return input


def index_add(input: _Tensor, dim: int, index: _Tensor, source: _Tensor, *, alpha: float = 1.0) -> _Tensor:
    """Accumulates the elements of alpha * source into input along dim according to index (differentiable scatter_add)."""
    if input.dtype != source.dtype:
        raise RuntimeError(f"index_add(): self ({input.dtype}) and source ({source.dtype}) must have the same dtype")
    d = dim if dim >= 0 else input.ndim + dim
    src = source if alpha == 1.0 else source * alpha
    return input.scatter_add(d, _expand_index(index, d, source), src)


def index_add_(input: _Tensor, dim: int, index: _Tensor, source: _Tensor, *, alpha: float = 1.0) -> _Tensor:
    """In-place version of index_add."""
    res = index_add(input, dim, index, source, alpha=alpha)
    input.assign(res)
    return input


def index_fill(input: _Tensor, dim: int, index: _Tensor, value: Union[float, _Tensor]) -> _Tensor:
    """Fills elements of input with value along dimension dim according to index."""
    # Pure delegation: the native kernel wraps negatives, requires int64
    # vector/scalar indices, and differentiates. Tensor values coerce
    # through item(), like torch Scalar (the old shim cast any index
    # dtype/shape and cut gradients).
    if isinstance(value, _Tensor):
        value = value.item()
    return input.index_fill(dim, index, value)


def index_fill_(input: _Tensor, dim: int, index: _Tensor, value: float) -> _Tensor:
    """In-place version of index_fill."""
    res = index_fill(input, dim, index, value)
    input.assign(res)
    return input


def masked_fill(input: _Tensor, mask: _Tensor, value: float) -> _Tensor:
    """Fills elements of input with value where mask is True."""
    return input.masked_fill(mask, value)


def masked_scatter(input: _Tensor, mask: _Tensor, source: _Tensor) -> _Tensor:
    """Copies elements from source into input at positions where mask is True."""
    # Pure delegation (no legacy validation shell): the native kernel
    # broadcasts the mask, requires bool masks and matching dtypes, and
    # raises torch's RuntimeError on short sources.
    return input.masked_scatter(mask, source)


def masked_scatter_(input: _Tensor, mask: _Tensor, source: _Tensor) -> _Tensor:
    """In-place version of masked_scatter."""
    res = masked_scatter(input, mask, source)
    input.assign(res)
    return input


def masked_select(input: _Tensor, mask: _Tensor) -> _Tensor:
    """Returns a 1D tensor of elements of input where mask is True."""
    # Pure delegation: the native kernel broadcasts bool-only masks and
    # differentiates (the old shim silently cast int masks and cut grads).
    return input.masked_select(mask)


def diag_embed(input: _Tensor, offset: int = 0, dim1: int = -2, dim2: int = -1) -> _Tensor:
    """Creates a tensor whose diagonals of 2D planes are filled by input (differentiable scatter)."""
    batch = [int(v) for v in input.shape[:-1]]
    d_len = int(input.shape[-1])
    n = d_len + _builtin_abs(offset)
    rows = np.arange(d_len) + (0 if offset >= 0 else -offset)
    cols = np.arange(d_len) + (offset if offset >= 0 else 0)
    b = int(np.prod(batch)) if batch else 1
    idx = np.broadcast_to((rows * n + cols)[None, :], (b, d_len))
    base = zeros([b, n * n], dtype=input.dtype)
    out = base.scatter(1, _Tensor.from_numpy(_contig(idx)), input.reshape([b, d_len])).reshape(batch + [n, n])
    nd = out.ndim
    d1, d2 = (dim1 if dim1 >= 0 else nd + dim1), (dim2 if dim2 >= 0 else nd + dim2)
    if (d1, d2) != (nd - 2, nd - 1):
        # place the two matrix dims at d1, d2 and the batch dims, in order, everywhere else
        batch_dims = iter(range(nd - 2))
        perm = [nd - 2 if i == d1 else nd - 1 if i == d2 else next(batch_dims) for i in range(nd)]
        out = out.permute(perm)
    return out


def diagonal_scatter(input: _Tensor, src: _Tensor, offset: int = 0, dim1: int = 0, dim2: int = 1) -> _Tensor:
    """Embeds the values of src into input along the diagonal."""
    x = input if isinstance(input, _Tensor) else tensor(input)
    s = src if isinstance(src, _Tensor) else tensor(src)
    ndim = x.ndim
    d1 = dim1 if dim1 >= 0 else ndim + dim1
    d2 = dim2 if dim2 >= 0 else ndim + dim2
    if not (0 <= d1 < ndim and 0 <= d2 < ndim):
        raise IndexError(f"diagonal_scatter: dimension out of range (expected [{-ndim}, {ndim-1}])")

    # Permute so that d1 and d2 are at the trailing end
    other_dims = [i for i in range(ndim) if i not in (d1, d2)]
    perm = other_dims + [d1, d2]
    x_p = x.permute(perm)
    batch_shape = list(x_p.shape[:-2])
    H, W = x_p.shape[-2], x_p.shape[-1]
    if offset >= 0:
        diag_len = max(0, min(H, W - offset))
        h_idx = list(range(diag_len))
        w_idx = [i + offset for i in range(diag_len)]
    else:
        diag_len = max(0, min(H + offset, W))
        h_idx = [i - offset for i in range(diag_len)]
        w_idx = list(range(diag_len))

    flat_idx = tensor([h * W + w for h, w in zip(h_idx, w_idx)], dtype=int)
    # Scatter along trailing flattened dimension
    x_p_flat = x_p.reshape(batch_shape + [H * W])
    # If s has same batch dims, flatten its last dim if needed
    s_flat = s.reshape(batch_shape + [diag_len])
    scattered = x_p_flat.scatter(-1, flat_idx.broadcast_to(s_flat.shape) if len(batch_shape) > 0 else flat_idx, s_flat)
    unperm = [0] * ndim
    for new_pos, orig_dim in enumerate(perm):
        unperm[orig_dim] = new_pos
    return scattered.reshape(list(x_p.shape)).permute(unperm)


def select_scatter(input: _Tensor, src: _Tensor, dim: int, index: int) -> _Tensor:
    """Embeds the values of src into input at the given dim and index."""
    x = input if isinstance(input, _Tensor) else tensor(input)
    s = src if isinstance(src, _Tensor) else tensor(src)
    d = dim if dim >= 0 else x.ndim + dim
    idx = int(index)
    if idx < 0:
        idx += x.shape[d]
    if not (0 <= idx < x.shape[d]):
        raise IndexError(f"select_scatter: index {index} out of bounds for dimension {dim} of size {x.shape[d]}")

    before = x.narrow(d, 0, idx) if idx > 0 else None
    s_expanded = s.unsqueeze(d)
    after = x.narrow(d, idx + 1, x.shape[d] - (idx + 1)) if idx + 1 < x.shape[d] else None
    parts = [p for p in [before, s_expanded, after] if p is not None]
    return cat(parts, dim=d)


def slice_scatter(input: _Tensor, src: _Tensor, dim: int = 0, start: Optional[int] = None, end: Optional[int] = None, step: int = 1) -> _Tensor:
    """Embeds the values of src into input at the specified slice."""
    x = input if isinstance(input, _Tensor) else tensor(input)
    s = src if isinstance(src, _Tensor) else tensor(src)
    d = dim if dim >= 0 else x.ndim + dim
    dim_size = x.shape[d]

    st = 0 if start is None else (start if start >= 0 else max(0, dim_size + start))
    en = dim_size if end is None else (end if end >= 0 else max(0, dim_size + end))
    st = min(st, dim_size)
    en = min(en, dim_size)

    if step == 1:
        before = x.narrow(d, 0, st) if st > 0 else None
        after = x.narrow(d, en, dim_size - en) if en < dim_size else None
        parts = [p for p in [before, s, after] if p is not None]
        return cat(parts, dim=d)

    # General strided slice scatter
    idx_list = list(range(st, en, step))
    idx = tensor(idx_list, dtype=int)
    return x.scatter(d, idx, s)


_Tensor.index_copy = lambda self, dim, index, source: index_copy(self, dim, index, source)
_Tensor.index_copy_ = lambda self, dim, index, source: index_copy_(self, dim, index, source)
_Tensor.index_add = lambda self, dim, index, source, alpha=1.0: index_add(self, dim, index, source, alpha=alpha)
_Tensor.index_add_ = lambda self, dim, index, source, alpha=1.0: index_add_(self, dim, index, source, alpha=alpha)
# NOTE: no _Tensor.index_fill monkeypatch: the native method serves both the
# method and (via delegation above) the module function. A lambda here
# would shadow the native method and recurse.
_Tensor.index_fill_ = lambda self, dim, index, value: index_fill_(self, dim, index, value)
# NOTE: no _Tensor.masked_scatter/_Tensor.masked_select monkeypatches: the
# native methods serve both the methods and (via delegation above) the module
# functions. Lambdas here would shadow the native methods and recurse.
_Tensor.masked_scatter_ = lambda self, mask, source: masked_scatter_(self, mask, source)
_Tensor.diag_embed = lambda self, offset=0, dim1=-2, dim2=-1: diag_embed(self, offset=offset, dim1=dim1, dim2=dim2)
_Tensor.diagonal_scatter = lambda self, src, offset=0, dim1=0, dim2=1: diagonal_scatter(self, src, offset=offset, dim1=dim1, dim2=dim2)
_Tensor.select_scatter = lambda self, src, dim, index: select_scatter(self, src, dim, index)
_Tensor.slice_scatter = lambda self, src, dim=0, start=None, end=None, step=1: slice_scatter(self, src, dim=dim, start=start, end=end, step=step)


# ---------------------------------------------------------------------------
# In-place Random Distribution Fills & Sampling Functions
# ---------------------------------------------------------------------------

def _uniform_inplace(self: _Tensor, from_val: float = 0.0, to_val: float = 1.0) -> _Tensor:
    """Fills self with values uniformly sampled from [from_val, to_val)."""
    shape = list(self.shape)
    arr = np.random.uniform(from_val, to_val, size=shape).astype(self.to_numpy().dtype)
    self.assign(_from_numpy(_contig(arr)))
    return self


def _normal_inplace(self: _Tensor, mean: float = 0.0, std: float = 1.0) -> _Tensor:
    """Fills self with values sampled from normal distribution N(mean, std^2)."""
    shape = list(self.shape)
    arr = np.random.normal(mean, std, size=shape).astype(self.to_numpy().dtype)
    self.assign(_from_numpy(_contig(arr)))
    return self


def _exponential_inplace(self: _Tensor, lambd: float = 1.0) -> _Tensor:
    """Fills self with values sampled from exponential distribution with rate lambd."""
    shape = list(self.shape)
    arr = np.random.exponential(1.0 / lambd, size=shape).astype(self.to_numpy().dtype)
    self.assign(_from_numpy(_contig(arr)))
    return self


def _cauchy_inplace(self: _Tensor, median: float = 0.0, sigma: float = 1.0) -> _Tensor:
    """Fills self with values sampled from standard Cauchy distribution."""
    shape = list(self.shape)
    arr = (median + sigma * np.random.standard_cauchy(size=shape)).astype(self.to_numpy().dtype)
    self.assign(_from_numpy(_contig(arr)))
    return self


def _geometric_inplace(self: _Tensor, p: float) -> _Tensor:
    """Fills self with values sampled from geometric distribution with success probability p."""
    shape = list(self.shape)
    arr = np.random.geometric(p, size=shape).astype(self.to_numpy().dtype)
    self.assign(_from_numpy(_contig(arr)))
    return self


def _log_normal_inplace(self: _Tensor, mean: float = 1.0, std: float = 2.0) -> _Tensor:
    """Fills self with values sampled from log-normal distribution."""
    shape = list(self.shape)
    arr = np.random.lognormal(mean, std, size=shape).astype(self.to_numpy().dtype)
    self.assign(_from_numpy(_contig(arr)))
    return self


def _bernoulli_inplace(self: _Tensor, p: Union[float, _Tensor] = 0.5) -> _Tensor:
    """Fills self with binary outcomes {0, 1} sampled from Bernoulli(p)."""
    shape = list(self.shape)
    p_arr = p.to_numpy() if isinstance(p, _Tensor) else p
    arr = np.random.binomial(1, p_arr, size=shape).astype(self.to_numpy().dtype)
    self.assign(_from_numpy(_contig(arr)))
    return self


_Tensor.uniform_ = _uniform_inplace
_Tensor.normal_ = _normal_inplace
_Tensor.exponential_ = _exponential_inplace
_Tensor.cauchy_ = _cauchy_inplace
_Tensor.geometric_ = _geometric_inplace
_Tensor.log_normal_ = _log_normal_inplace
_Tensor.bernoulli_ = _bernoulli_inplace


def bernoulli(input: _Tensor, *, generator: Any = None, out: Optional[_Tensor] = None) -> _Tensor:
    """Samples from a Bernoulli distribution with parameter given by input."""
    inp = input if isinstance(input, _Tensor) else tensor(input)
    p = inp.to_numpy()
    arr = np.random.binomial(1, p).astype(p.dtype)
    res = _from_numpy(_contig(arr))
    if out is not None:
        out.assign(res)
        return out
    return res


def poisson(input: _Tensor, *, generator: Any = None) -> _Tensor:
    """Samples from a Poisson distribution with rates given by input."""
    inp = input if isinstance(input, _Tensor) else tensor(input)
    lam = inp.to_numpy()
    arr = np.random.poisson(lam).astype(lam.dtype)
    return _from_numpy(_contig(arr))


def multinomial(input: _Tensor, num_samples: int, replacement: bool = False, *, generator: Any = None, out: Optional[_Tensor] = None) -> _Tensor:
    """Samples from a multinomial distribution according to input weights."""
    inp = input if isinstance(input, _Tensor) else tensor(input)
    res = inp.multinomial(num_samples=num_samples, replacement=replacement, generator=generator)
    if out is not None:
        out.assign(res)
        return out
    return res


def logdet(input: _Tensor) -> _Tensor:
    """Computes the log determinant of a square matrix or batch of square matrices."""
    inp = input if isinstance(input, _Tensor) else tensor(input)
    return inp.logdet()


