"""Comprehensive Model Quantization Engine for Oxideon.

Implements high-performance integer and floating-point quantization:
  - Per-tensor uniform affine quantization (INT8/UINT8)
  - Per-channel / per-row dynamic and static quantization
  - QuantizedLinear (per-tensor INT8)
  - PerChannelQuantizedLinear (per-channel INT8/INT4)
  - QuantizedEmbedding (memory-efficient on-demand row dequantization)
  - SmoothQuant: activation outlier migration for W8A8 linear layers
  - FP8 Scaled GEMM and FP8Linear (E4M3 and E5M2)
  - quantize_dynamic: recursive module quantization with policy configuration
"""

from typing import Any, Dict, Optional, Sequence, Tuple, Union
import numpy as np

from .oxideon import Tensor as _Tensor, from_numpy as _from_numpy, Embedding as _Embedding
from .nn import Module, Parameter, Linear


def quantize_per_tensor(
    tensor: Union[_Tensor, np.ndarray],
    scale: float,
    zero_point: int = 0,
    dtype: str = "qint8",
) -> Tuple[_Tensor, float, int]:
    """Quantizes a floating point tensor into integer tensor using uniform affine quantization."""
    if float(scale) <= 0.0:
        raise ValueError(f"scale must be > 0, got {scale}")
    arr = tensor.to_numpy() if isinstance(tensor, _Tensor) else np.asarray(tensor, dtype=np.float32)
    q = np.round(arr / scale) + zero_point
    if dtype in ("qint8", "int8"):
        q_clamped = np.clip(q, -128, 127).astype(np.int8)
    elif dtype in ("quint8", "uint8"):
        q_clamped = np.clip(q, 0, 255).astype(np.uint8)
    elif dtype in ("qint4", "int4"):
        q_clamped = np.clip(q, -8, 7).astype(np.int8)
    elif dtype in ("quint4", "uint4"):
        q_clamped = np.clip(q, 0, 15).astype(np.uint8)
    else:
        raise ValueError(f"Unsupported quantization dtype: {dtype}")
    return _from_numpy(np.ascontiguousarray(q_clamped.astype(np.float32))), float(scale), int(zero_point)


def dequantize(
    q_tensor: Union[_Tensor, np.ndarray],
    scale: float,
    zero_point: int = 0,
) -> _Tensor:
    """Dequantizes an integer tensor back into a floating point tensor."""
    arr = q_tensor.to_numpy() if isinstance(q_tensor, _Tensor) else np.asarray(q_tensor)
    deq = (arr.astype(np.float32) - float(zero_point)) * float(scale)
    return _from_numpy(np.ascontiguousarray(deq, dtype=np.float32))


def quantize_per_channel(
    tensor: Union[_Tensor, np.ndarray],
    scales: Union[_Tensor, np.ndarray, Sequence[float]],
    zero_points: Union[_Tensor, np.ndarray, Sequence[int]],
    axis: int = 0,
    dtype: str = "qint8",
) -> Tuple[_Tensor, np.ndarray, np.ndarray]:
    """Quantizes a tensor along a specified channel axis with individual scales and zero-points."""
    arr = tensor.to_numpy() if isinstance(tensor, _Tensor) else np.asarray(tensor, dtype=np.float32)
    axis = axis % arr.ndim
    s_arr = scales.to_numpy().astype(np.float32) if isinstance(scales, _Tensor) else np.asarray(scales, dtype=np.float32)
    zp_arr = zero_points.to_numpy().astype(np.int32) if isinstance(zero_points, _Tensor) else np.asarray(zero_points, dtype=np.int32)

    if s_arr.shape != (arr.shape[axis],) or zp_arr.shape != (arr.shape[axis],):
        raise ValueError(
            f"scales ({s_arr.shape}) and zero_points ({zp_arr.shape}) must match tensor.shape[{axis}] ({arr.shape[axis]})"
        )
    if np.any(s_arr <= 0.0):
        raise ValueError("All quantization scales must be strictly positive (> 0.0)")

    shape = [1] * arr.ndim
    shape[axis] = arr.shape[axis]
    s_b = s_arr.reshape(shape)
    zp_b = zp_arr.reshape(shape)

    q = np.round(arr / s_b) + zp_b
    if dtype in ("qint8", "int8"):
        q_clamped = np.clip(q, -128, 127).astype(np.int8)
    elif dtype in ("quint8", "uint8"):
        q_clamped = np.clip(q, 0, 255).astype(np.uint8)
    elif dtype in ("qint4", "int4"):
        q_clamped = np.clip(q, -8, 7).astype(np.int8)
    elif dtype in ("quint4", "uint4"):
        q_clamped = np.clip(q, 0, 15).astype(np.uint8)
    else:
        raise ValueError(f"Unsupported dtype for per-channel quantization: {dtype}")

    return _from_numpy(np.ascontiguousarray(q_clamped.astype(np.float32))), s_arr, zp_arr


def dequantize_per_channel(
    q_tensor: Union[_Tensor, np.ndarray],
    scales: Union[_Tensor, np.ndarray, Sequence[float]],
    zero_points: Union[_Tensor, np.ndarray, Sequence[int]],
    axis: int = 0,
) -> _Tensor:
    """Dequantizes a per-channel quantized tensor back to full-precision float."""
    arr = q_tensor.to_numpy() if isinstance(q_tensor, _Tensor) else np.asarray(q_tensor, dtype=np.float32)
    axis = axis % arr.ndim
    s_arr = scales.to_numpy().astype(np.float32) if isinstance(scales, _Tensor) else np.asarray(scales, dtype=np.float32)
    zp_arr = zero_points.to_numpy().astype(np.float32) if isinstance(zero_points, _Tensor) else np.asarray(zero_points, dtype=np.float32)

    shape = [1] * arr.ndim
    shape[axis] = arr.shape[axis]
    s_b = s_arr.reshape(shape)
    zp_b = zp_arr.reshape(shape)

    deq = (arr.astype(np.float32) - zp_b) * s_b
    return _from_numpy(np.ascontiguousarray(deq, dtype=np.float32))


def quantize_per_channel_dynamic(
    tensor: Union[_Tensor, np.ndarray],
    axis: int = 0,
    dtype: str = "qint8",
    symmetric: bool = True,
) -> Tuple[_Tensor, np.ndarray, np.ndarray]:
    """Dynamically computes optimal scales and zero-points per slice along `axis` and quantizes."""
    arr = tensor.to_numpy() if isinstance(tensor, _Tensor) else np.asarray(tensor, dtype=np.float32)
    axis = axis % arr.ndim
    reduce_axes = tuple(i for i in range(arr.ndim) if i != axis)

    if dtype in ("qint8", "int8"):
        qmin, qmax = -128, 127
    elif dtype in ("quint8", "uint8"):
        qmin, qmax = 0, 255
    elif dtype in ("qint4", "int4"):
        qmin, qmax = -8, 7
    elif dtype in ("quint4", "uint4"):
        qmin, qmax = 0, 15
    else:
        raise ValueError(f"Unsupported dtype: {dtype}")

    if arr.size == 0:
        n_ch = arr.shape[axis]
        return (
            _from_numpy(np.empty_like(arr, dtype=np.float32)),
            np.ones(n_ch, dtype=np.float32),
            np.zeros(n_ch, dtype=np.int32),
        )

    if symmetric:
        amax = np.max(np.abs(arr), axis=reduce_axes)
        max_q = float(max(abs(qmin), abs(qmax)))
        scales = np.maximum(amax / max_q, 1e-8).astype(np.float32)
        zero_points = np.zeros_like(scales, dtype=np.int32)
    else:
        mins = np.minimum(np.min(arr, axis=reduce_axes), 0.0)
        maxs = np.maximum(np.max(arr, axis=reduce_axes), 0.0)
        scales = np.maximum((maxs - mins) / float(qmax - qmin), 1e-8).astype(np.float32)
        zero_points = np.round(qmin - mins / scales).astype(np.int32)
        zero_points = np.clip(zero_points, qmin, qmax)

    return quantize_per_channel(arr, scales, zero_points, axis=axis, dtype=dtype)


class QuantizedLinear(Module):
    """Linear layer with 8-bit quantized weights and floating point bias."""

    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.has_bias = bias

        self.qweight = None
        self.scale = 1.0
        self.zero_point = 0
        self.bias = Parameter(_Tensor.zeros([out_features])) if bias else None

    @classmethod
    def from_float(cls, float_linear: Any) -> "QuantizedLinear":
        """Quantizes an existing floating-point Linear layer into a QuantizedLinear layer."""
        in_f = float_linear.weight.shape[1] if len(float_linear.weight.shape) > 1 else float_linear.weight.shape[0]
        out_f = float_linear.weight.shape[0]
        has_b = getattr(float_linear, "bias", None) is not None

        q = cls(in_f, out_f, bias=has_b)
        w_np = float_linear.weight.to_numpy() if hasattr(float_linear.weight, "to_numpy") else np.asarray(float_linear.weight)

        if w_np.size == 0:
            q.scale = 1.0
            q.zero_point = 0
            q.qweight = np.empty_like(w_np, dtype=np.int8)
        else:
            w_min = float(np.min(w_np))
            w_max = float(np.max(w_np))
            q.scale = max(float((w_max - w_min) / 255.0), 1e-8)
            q.zero_point = int(np.clip(np.round(-w_min / q.scale) - 128, -128, 127))
            q.qweight = np.clip(np.round(w_np / q.scale) + q.zero_point, -128, 127).astype(np.int8)

        if has_b and float_linear.bias is not None:
            b_np = float_linear.bias.to_numpy() if hasattr(float_linear.bias, "to_numpy") else np.asarray(float_linear.bias)
            q.bias = Parameter(_from_numpy(np.ascontiguousarray(b_np, dtype=np.float32)))

        return q

    def forward(self, x: _Tensor) -> _Tensor:
        w_dequant = (self.qweight.astype(np.float32) - self.zero_point) * self.scale
        w_tensor = _from_numpy(np.ascontiguousarray(w_dequant, dtype=np.float32))

        out = x @ w_tensor.t()
        if self.bias is not None:
            out = out + self.bias
        return out

    def state_dict(self) -> Dict[str, Any]:
        """Returns state dictionary containing quantized parameters and quantization scale/zero_point."""
        sd: Dict[str, Any] = {
            "qweight": self.qweight.copy() if self.qweight is not None else None,
            "scale": float(self.scale),
            "zero_point": int(self.zero_point),
        }
        if self.bias is not None:
            sd["bias"] = self.bias.to_numpy().copy()
        return sd

    def load_state_dict(self, state_dict: Dict[str, Any], strict: bool = True) -> None:
        """Loads state dictionary into QuantizedLinear."""
        if "qweight" in state_dict:
            self.qweight = np.asarray(state_dict["qweight"], dtype=np.int8).copy()
        if "scale" in state_dict:
            self.scale = float(state_dict["scale"])
        if "zero_point" in state_dict:
            self.zero_point = int(state_dict["zero_point"])
        if "bias" in state_dict and state_dict["bias"] is not None:
            b_arr = np.asarray(state_dict["bias"], dtype=np.float32)
            self.bias = Parameter(_from_numpy(np.ascontiguousarray(b_arr)))

    def __repr__(self):
        return f"QuantizedLinear(in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None}, scale={self.scale:.6f})"


class PerChannelQuantizedLinear(Module):
    """Linear layer with per-channel (per-output-row) quantized weights."""

    def __init__(self, in_features: int, out_features: int, bias: bool = True, dtype: str = "qint8"):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dtype = dtype
        self.qweight = None
        self.scales = np.ones(out_features, dtype=np.float32)
        self.zero_points = np.zeros(out_features, dtype=np.int32)
        self.bias = Parameter(_Tensor.zeros([out_features])) if bias else None

    @classmethod
    def from_float(cls, float_linear: Any, dtype: str = "qint8", symmetric: bool = True) -> "PerChannelQuantizedLinear":
        in_f = float_linear.weight.shape[1] if len(float_linear.weight.shape) > 1 else float_linear.weight.shape[0]
        out_f = float_linear.weight.shape[0]
        has_b = getattr(float_linear, "bias", None) is not None

        layer = cls(in_f, out_f, bias=has_b, dtype=dtype)
        w = float_linear.weight.to_numpy() if hasattr(float_linear.weight, "to_numpy") else np.asarray(float_linear.weight)

        q_w, scales, zps = quantize_per_channel_dynamic(w, axis=0, dtype=dtype, symmetric=symmetric)
        layer.qweight = q_w.to_numpy().astype(np.int8)
        layer.scales = scales
        layer.zero_points = zps

        if has_b and float_linear.bias is not None:
            b_np = float_linear.bias.to_numpy() if hasattr(float_linear.bias, "to_numpy") else np.asarray(float_linear.bias)
            layer.bias = Parameter(_from_numpy(np.ascontiguousarray(b_np, dtype=np.float32)))

        return layer

    def forward(self, x: _Tensor) -> _Tensor:
        # Dequantize row by row: (w_row - zp_row) * scale_row
        w_dequant = (self.qweight.astype(np.float32) - self.zero_points[:, None]) * self.scales[:, None]
        w_tensor = _from_numpy(np.ascontiguousarray(w_dequant, dtype=np.float32))

        out = x @ w_tensor.t()
        if self.bias is not None:
            out = out + self.bias
        return out

    def __repr__(self):
        return f"PerChannelQuantizedLinear(in_features={self.in_features}, out_features={self.out_features}, dtype={self.dtype}, bias={self.bias is not None})"


class QuantizedEmbedding(Module):
    """Memory-efficient embedding table with quantized representations and on-demand dequantization."""

    def __init__(self, num_embeddings: int, embedding_dim: int, dtype: str = "qint8"):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.dtype = dtype
        self.qweight = None
        self.scales = np.ones(num_embeddings, dtype=np.float32)
        self.zero_points = np.zeros(num_embeddings, dtype=np.int32)

    @classmethod
    def from_float(cls, float_emb: Any, dtype: str = "qint8", symmetric: bool = True) -> "QuantizedEmbedding":
        num_emb = float_emb.weight.shape[0]
        emb_dim = float_emb.weight.shape[1]
        layer = cls(num_emb, emb_dim, dtype=dtype)
        w = float_emb.weight.to_numpy() if hasattr(float_emb.weight, "to_numpy") else np.asarray(float_emb.weight)

        q_w, scales, zps = quantize_per_channel_dynamic(w, axis=0, dtype=dtype, symmetric=symmetric)
        layer.qweight = q_w.to_numpy().astype(np.int8)
        layer.scales = scales
        layer.zero_points = zps
        return layer

    def forward(self, indices: Union[_Tensor, np.ndarray]) -> _Tensor:
        idx_np = indices.to_numpy().astype(np.int64) if isinstance(indices, _Tensor) else np.asarray(indices, dtype=np.int64)
        flat_idx = idx_np.flatten()

        # On-demand row dequantization: only dequantize fetched embedding vectors
        q_rows = self.qweight[flat_idx].astype(np.float32)
        s_rows = self.scales[flat_idx, None]
        zp_rows = self.zero_points[flat_idx, None].astype(np.float32)

        deq_rows = (q_rows - zp_rows) * s_rows
        out_shape = list(idx_np.shape) + [self.embedding_dim]
        return _from_numpy(np.ascontiguousarray(deq_rows.reshape(out_shape), dtype=np.float32))

    def __repr__(self):
        return f"QuantizedEmbedding(num_embeddings={self.num_embeddings}, embedding_dim={self.embedding_dim}, dtype={self.dtype})"


def smooth_quant(
    linear: Linear,
    act_max: Union[_Tensor, np.ndarray],
    alpha: float = 0.5,
) -> Tuple[Linear, _Tensor]:
    """Applies SmoothQuant activation outlier migration to a Linear layer.

    Computes channel-wise smoothing scales:
      s_j = (act_max_j^alpha) / (weight_max_j^(1 - alpha))
    Then scales weights by s and returns a transformed Linear layer along with smoothing factors s.
    During inference, activations are scaled by diag(s)^-1 prior to matrix multiplication.
    """
    if alpha < 0.0 or alpha > 1.0:
        raise ValueError(f"alpha must be in [0, 1], got {alpha}")

    a_max = act_max.to_numpy() if isinstance(act_max, _Tensor) else np.asarray(act_max, dtype=np.float32)
    w_mat = linear.weight.to_numpy() if hasattr(linear.weight, "to_numpy") else np.asarray(linear.weight, dtype=np.float32)

    # Channel-wise max of weight along in_features (columns, axis 0)
    w_max = np.max(np.abs(w_mat), axis=0)

    # s_j = (a_max_j^alpha) / (w_max_j^(1 - alpha))
    numerator = np.power(np.maximum(a_max, 1e-5), alpha)
    denominator = np.power(np.maximum(w_max, 1e-5), 1.0 - alpha)
    scales = np.clip(numerator / np.maximum(denominator, 1e-8), 1e-4, 1e4).astype(np.float32)

    # W_smoothed = W * diag(s) (each column j multiplied by scales[j])
    smoothed_w = w_mat * scales[None, :]

    in_features = linear.weight.shape[1] if len(linear.weight.shape) > 1 else linear.weight.shape[0]
    out_features = linear.weight.shape[0]
    new_linear = Linear(in_features, out_features, bias=linear.bias is not None)
    new_linear.weight = Parameter(_from_numpy(np.ascontiguousarray(smoothed_w, dtype=np.float32)))
    if linear.bias is not None:
        new_linear.bias = Parameter(linear.bias.clone())

    return new_linear, _from_numpy(np.ascontiguousarray(scales, dtype=np.float32))


# FP8 Constants
FP8_E4M3_MAX = 448.0
FP8_E5M2_MAX = 57344.0


def fp8_gemm(
    a: _Tensor,
    b: _Tensor,
    scale_a: float = 1.0,
    scale_b: float = 1.0,
    format: str = "e4m3",
) -> _Tensor:
    """Scaled matrix multiplication for FP8 simulated tensors: out = (A @ B) * (scale_a * scale_b)."""
    max_val = FP8_E4M3_MAX if format == "e4m3" else FP8_E5M2_MAX
    step = 1.0 / 8.0 if format == "e4m3" else 1.0 / 4.0

    a_np = a.to_numpy()
    b_np = b.to_numpy()

    # Emulate FP8 representation
    a_q = np.clip(np.round(a_np / step) * step, -max_val, max_val)
    b_q = np.clip(np.round(b_np / step) * step, -max_val, max_val)

    y = (a_q @ b_q) * (float(scale_a) * float(scale_b))
    return _from_numpy(np.ascontiguousarray(y, dtype=np.float32))


class FP8Linear(Module):
    """Linear layer using simulated FP8 E4M3 or E5M2 weights."""

    def __init__(self, in_features: int, out_features: int, bias: bool = True, format: str = "e4m3"):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.format = format.lower()
        if self.format not in ("e4m3", "e5m2"):
            raise ValueError(f"Unsupported FP8 format: {format}")

        self.qweight = None
        self.scale = 1.0
        self.bias = Parameter(_Tensor.zeros([out_features])) if bias else None

    @classmethod
    def from_float(cls, float_linear: Any, format: str = "e4m3") -> "FP8Linear":
        in_f = float_linear.weight.shape[1] if len(float_linear.weight.shape) > 1 else float_linear.weight.shape[0]
        out_f = float_linear.weight.shape[0]
        has_b = getattr(float_linear, "bias", None) is not None

        layer = cls(in_f, out_f, bias=has_b, format=format)
        w = float_linear.weight.to_numpy() if hasattr(float_linear.weight, "to_numpy") else np.asarray(float_linear.weight)

        max_val = FP8_E4M3_MAX if layer.format == "e4m3" else FP8_E5M2_MAX
        amax = float(np.max(np.abs(w))) if w.size > 0 else 1.0
        layer.scale = amax / max_val if amax > 0.0 else 1.0

        inv_scale = 1.0 / layer.scale
        step = 1.0 / 8.0 if layer.format == "e4m3" else 1.0 / 4.0
        scaled = np.clip(w * inv_scale, -max_val, max_val)
        layer.qweight = np.round(scaled / step) * step

        if has_b and float_linear.bias is not None:
            b_np = float_linear.bias.to_numpy() if hasattr(float_linear.bias, "to_numpy") else np.asarray(float_linear.bias)
            layer.bias = Parameter(_from_numpy(np.ascontiguousarray(b_np, dtype=np.float32)))

        return layer

    def forward(self, x: _Tensor) -> _Tensor:
        w_deq = self.qweight * self.scale
        w_tensor = _from_numpy(np.ascontiguousarray(w_deq, dtype=np.float32))

        out = x @ w_tensor.t()
        if self.bias is not None:
            out = out + self.bias
        return out

    def __repr__(self):
        return f"FP8Linear(in_features={self.in_features}, out_features={self.out_features}, format={self.format}, scale={self.scale:.6f})"


def quantize_dynamic(
    model: Module,
    dtype: str = "qint8",
    per_channel: bool = False,
    quantize_embeddings: bool = True,
) -> Module:
    """Recursively replaces all `Linear` (and optionally `Embedding`) modules in `model` with quantized modules.

    Args:
        model: Target Oxideon Module.
        dtype: Quantization target ('qint8', 'fp8_e4m3', 'fp8_e5m2').
        per_channel: Whether to use per-channel row-wise quantization for linear weights.
        quantize_embeddings: Whether to also quantize Embedding layers into QuantizedEmbedding.
    """
    valid_dtypes = ("qint8", "int8", "quint8", "uint8", "qint4", "int4", "fp8_e4m3", "fp8_e5m2")
    if dtype not in valid_dtypes:
        raise ValueError(f"Unsupported dtype: {dtype}. Valid dtypes are {valid_dtypes}")

    modules_dict = getattr(model, "_modules", {})
    for name, child in list(modules_dict.items()):
        cls_name = child.__class__.__name__
        if isinstance(child, Linear) or cls_name == "Linear":
            if dtype.startswith("fp8"):
                fmt = "e4m3" if "e4m3" in dtype else "e5m2"
                ql = FP8Linear.from_float(child, format=fmt)
            elif per_channel:
                ql = PerChannelQuantizedLinear.from_float(child, dtype=dtype)
            else:
                ql = QuantizedLinear.from_float(child)
            modules_dict[name] = ql
            if hasattr(model, name):
                try:
                    setattr(model, name, ql)
                except Exception:
                    pass
        elif quantize_embeddings and (isinstance(child, _Embedding) or cls_name == "Embedding"):
            qe = QuantizedEmbedding.from_float(child, dtype=dtype)
            modules_dict[name] = qe
            if hasattr(model, name):
                try:
                    setattr(model, name, qe)
                except Exception:
                    pass
        else:
            quantize_dynamic(child, dtype=dtype, per_channel=per_channel, quantize_embeddings=quantize_embeddings)
    return model


__all__ = [
    "QuantizedLinear",
    "PerChannelQuantizedLinear",
    "QuantizedEmbedding",
    "FP8Linear",
    "quantize_dynamic",
    "quantize_per_tensor",
    "dequantize",
    "quantize_per_channel",
    "dequantize_per_channel",
    "quantize_per_channel_dynamic",
    "smooth_quant",
    "fp8_gemm",
]
