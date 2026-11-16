"""
Advanced Model Quantization for Oxideon.
Includes FP8 formats (E4M3, E5M2), block-wise 4-bit weight-only quantization (AWQ/GPTQ style),
OCP Microscaling MXFP4, and Quantization-Aware Training (QAT) modules with Straight-Through Estimators (STE).
"""
from typing import Tuple
import numpy as np
from .oxideon import Tensor, from_numpy as _from_numpy
from .nn import Module, Parameter, _check_positive
from .kquants import (
    QuantizedKBlock,
    Q2KQuantizer,
    Q3KQuantizer,
    Q5KQuantizer,
    Q6KQuantizer,
    quantize_k_quant,
    dequantize_k_quant,
    quantize_ternary_bitnet,
    dequantize_ternary_bitnet,
    BitLinear,
)

# FP8 constants
# E4M3: 1 sign bit, 4 exponent bits (bias=7), 3 mantissa bits; max normal = 448.0
FP8_E4M3_MAX = 448.0
# E5M2: 1 sign bit, 5 exponent bits (bias=15), 2 mantissa bits; max normal = 57344.0
FP8_E5M2_MAX = 57344.0


def quantize_fp8_e4m3(x: Tensor) -> Tuple[Tensor, float]:
    """Simulates FP8 E4M3 quantization with dynamic tensor scaling."""
    arr = x.to_numpy()
    if arr.size == 0:
        raise ValueError("Input tensor cannot be empty")
    amax = float(np.max(np.abs(arr))) + 1e-12
    scale = FP8_E4M3_MAX / amax

    scaled = np.clip(arr * scale, -FP8_E4M3_MAX, FP8_E4M3_MAX)
    quantized = np.round(scaled * 8.0) / 8.0
    return _from_numpy(quantized.astype(np.float32)), float(scale)


def dequantize_fp8_e4m3(x_q: Tensor, scale: float) -> Tensor:
    """Dequantizes FP8 E4M3 tensor back to standard precision."""
    arr = x_q.to_numpy()
    return _from_numpy((arr / scale).astype(np.float32))


def quantize_fp8_e5m2(x: Tensor) -> Tuple[Tensor, float]:
    """Simulates FP8 E5M2 quantization with dynamic tensor scaling."""
    arr = x.to_numpy()
    if arr.size == 0:
        raise ValueError("Input tensor cannot be empty")
    amax = float(np.max(np.abs(arr))) + 1e-12
    scale = FP8_E5M2_MAX / amax

    scaled = np.clip(arr * scale, -FP8_E5M2_MAX, FP8_E5M2_MAX)
    quantized = np.round(scaled * 4.0) / 4.0
    return _from_numpy(quantized.astype(np.float32)), float(scale)


def dequantize_fp8_e5m2(x_q: Tensor, scale: float) -> Tensor:
    """Dequantizes FP8 E5M2 tensor back to standard precision."""
    arr = x_q.to_numpy()
    return _from_numpy((arr / scale).astype(np.float32))


class WeightOnlyInt4Quantizer:
    """Block-wise 4-bit integer weight-only quantization with group scales and zero-points."""
    @staticmethod
    def quantize(weight: Tensor, group_size: int = 32) -> Tuple[Tensor, Tensor, Tensor]:
        if group_size < 1:
            raise ValueError(f"group_size must be >= 1, got {group_size}")
        w = weight.to_numpy()
        if w.size == 0:
            raise ValueError("weight tensor cannot be empty")
        orig_shape = w.shape
        w_flat = w.flatten()
        n = len(w_flat)
        pad_len = (group_size - (n % group_size)) % group_size
        if pad_len > 0:
            w_flat = np.pad(w_flat, (0, pad_len))

        groups = w_flat.reshape(-1, group_size)
        g_min = np.min(groups, axis=1, keepdims=True)
        g_max = np.max(groups, axis=1, keepdims=True)

        scales = (g_max - g_min) / 15.0 + 1e-8
        zeros = np.round(-g_min / scales)

        q_groups = np.clip(np.round(groups / scales + zeros), 0, 15).astype(np.int8)

        return (
            _from_numpy(q_groups.flatten()[:n].reshape(orig_shape).astype(np.float32)),
            _from_numpy(scales.flatten().astype(np.float32)),
            _from_numpy(zeros.flatten().astype(np.float32))
        )

    @staticmethod
    def dequantize(q_weight: Tensor, scales: Tensor, zeros: Tensor, group_size: int = 32) -> Tensor:
        q = q_weight.to_numpy()
        orig_shape = q.shape
        q_flat = q.flatten()
        n = len(q_flat)
        pad_len = (group_size - (n % group_size)) % group_size
        if pad_len > 0:
            q_flat = np.pad(q_flat, (0, pad_len))

        s = scales.to_numpy()
        z = zeros.to_numpy()

        groups = q_flat.reshape(-1, group_size)
        deq_groups = (groups - z[:, None]) * s[:, None]
        deq_flat = deq_groups.flatten()[:n]
        return _from_numpy(deq_flat.reshape(orig_shape).astype(np.float32))


class QATLinear(Module):
    """Linear layer with Straight-Through Estimator (STE) fake-quantization for QAT."""
    def __init__(self, in_features: int, out_features: int, num_bits: int = 8):
        super().__init__()
        if num_bits < 2:
            raise ValueError(f"num_bits must be at least 2, got {num_bits}")
        if in_features <= 0 or out_features <= 0:
            raise ValueError(f"in_features and out_features must be positive, got {in_features}, {out_features}")
        self.in_features = in_features
        self.out_features = out_features
        self.num_bits = num_bits
        self.qmax = (1 << (num_bits - 1)) - 1
        self.qmin = -(1 << (num_bits - 1))

        w = (np.random.randn(out_features, in_features) * np.sqrt(2.0 / in_features)).astype(np.float32)
        b = np.zeros(out_features, dtype=np.float32)
        self.weight = Parameter(_from_numpy(w))
        self.bias = Parameter(_from_numpy(b))

    def _fake_quantize(self, arr: np.ndarray) -> np.ndarray:
        amax = float(np.max(np.abs(arr))) + 1e-8
        scale = amax / float(self.qmax)
        q = np.clip(np.round(arr / scale), self.qmin, self.qmax)
        return (q * scale).astype(np.float32)

    def forward(self, x: Tensor) -> Tensor:
        w_np = self.weight.to_numpy()
        w_q_np = self._fake_quantize(w_np)
        w_q_t = _from_numpy(np.ascontiguousarray(w_q_np))
        # Straight-Through Estimator: forward uses w_q_t, backward flows to self.weight
        w_ste = self.weight + (w_q_t - self.weight).detach()

        orig_shape = list(x.shape)
        x_2d = x.reshape([-1, self.in_features])
        out_2d = x_2d.matmul(w_ste.transpose(0, 1)) + self.bias.reshape([1, self.out_features])
        return out_2d.reshape(orig_shape[:-1] + [self.out_features])


# OCP Microscaling MXFP4: E2M1 private elements (sign + 2-bit exp + 1-bit mantissa,
# bias 1, no inf/NaN) with a shared E8M0 power-of-two scale per block.
MXFP4_LEVELS = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
MXFP4_MAX = 6.0
MXFP4_EXP_MIN = -127
MXFP4_EXP_MAX = 127


def _mxfp4_scales(amax: np.ndarray) -> np.ndarray:
    exp = np.floor(np.log2(np.maximum(amax, 1e-30) / MXFP4_MAX))
    exp = np.clip(exp, MXFP4_EXP_MIN, MXFP4_EXP_MAX)
    return np.power(2.0, exp)


def quantize_mxfp4(weight: Tensor, block_size: int = 32) -> Tuple[Tensor, Tensor]:
    """OCP MXFP4 quantization: E2M1 nibbles + E8M0 power-of-two scales per block."""
    block_size = _check_positive("block_size", block_size)
    w = weight.to_numpy().astype(np.float64)
    if np.isnan(w).any():
        raise ValueError("quantize_mxfp4 refuses NaN entries")
    orig_shape = w.shape
    flat = w.flatten()
    n = len(flat)
    pad_len = (block_size - (n % block_size)) % block_size
    if pad_len:
        flat = np.pad(flat, (0, pad_len))
    groups = flat.reshape(-1, block_size)
    finite = np.where(np.isfinite(groups), np.abs(groups), 0.0)
    scales = _mxfp4_scales(finite.max(axis=1))
    scaled = np.clip(groups / scales[:, None], -MXFP4_MAX, MXFP4_MAX)
    levels = np.array(MXFP4_LEVELS, dtype=np.float64)
    mag = np.abs(np.abs(scaled)[..., None] - levels).argmin(axis=-1)
    codes = mag + (scaled < 0).astype(np.int64) * 8
    codes = codes.flatten()[:n].reshape(orig_shape)
    return (
        _from_numpy(codes.astype(np.float32)),
        _from_numpy(scales.astype(np.float32)),
    )


def dequantize_mxfp4(q_weight: Tensor, scales: Tensor, block_size: int = 32) -> Tensor:
    """Reconstructs weights from MXFP4 ``(codes, scales)`` pairs."""
    block_size = _check_positive("block_size", block_size)
    codes = q_weight.to_numpy()
    if not np.all((codes == np.round(codes)) & (codes >= 0) & (codes <= 15)):
        raise ValueError("dequantize_mxfp4 needs integer nibble codes within 0..15")
    scales_np = scales.to_numpy().astype(np.float64)
    if not np.all(np.isfinite(scales_np)) or (scales_np <= 0).any():
        raise ValueError("dequantize_mxfp4 needs finite positive scales")
    orig_shape = codes.shape
    flat = codes.flatten().astype(np.int64)
    n = len(flat)
    pad_len = (block_size - (n % group_size if 'group_size' in locals() else n % block_size)) % block_size
    if pad_len:
        flat = np.pad(flat, (0, pad_len))
    groups = flat.reshape(-1, block_size)
    if groups.shape[0] != len(scales_np):
        raise ValueError(
            f"{len(scales_np)} scales do not cover {groups.shape[0]} blocks"
        )
    levels = np.array(MXFP4_LEVELS, dtype=np.float64)
    mag = levels[groups & 7]
    sign = np.where(groups & 8, -1.0, 1.0)
    deq = (sign * mag * scales_np[:, None]).flatten()[:n]
    return _from_numpy(deq.reshape(orig_shape).astype(np.float32))
