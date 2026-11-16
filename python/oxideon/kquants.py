"""
Ultra-Low-Bit K-Quants (Q2_K, Q3_K, Q5_K, Q6_K) and 1.58-Bit Ternary BitNet for Oxideon.
Implements GGUF/llama.cpp-grade block-quantization algorithms with sub-byte bit-packing and BitNet ternary linear layers.
"""
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
import numpy as np

from .oxideon import Tensor, from_numpy as _from_numpy
from .nn import Module, Parameter


@dataclass
class QuantizedKBlock:
    """Container for block-quantized weights."""
    quant_type: str
    orig_shape: List[int]
    block_size: int
    bits: int
    data: np.ndarray      # Quantized indices or packed bytes
    scales: np.ndarray    # Float scale per block
    mins: np.ndarray      # Float minimum per block
    packed_data: Optional[np.ndarray] = None


def _pack_bits(q_flat: np.ndarray, bits: int) -> np.ndarray:
    """Packs an array of B-bit unsigned integers [0, 2^B - 1] into a compact uint8 byte stream."""
    if len(q_flat) == 0:
        return np.zeros(0, dtype=np.uint8)
    # Expand each value into `bits` little-endian bits and pack 8 bits per byte
    shifts = np.arange(bits, dtype=np.uint8)
    bit_matrix = ((q_flat[:, None] >> shifts[None, :]) & 1).astype(np.uint8)
    return np.packbits(bit_matrix.flatten(), bitorder="little")


def _unpack_bits(packed: np.ndarray, bits: int, count: int) -> np.ndarray:
    """Unpacks `count` B-bit unsigned integers from a compact uint8 byte stream."""
    if count == 0:
        return np.zeros(0, dtype=np.uint8)
    raw_bits = np.unpackbits(packed, bitorder="little")[: count * bits]
    bit_matrix = raw_bits.reshape(count, bits)
    shifts = (1 << np.arange(bits, dtype=np.uint8)).astype(np.uint8)
    return np.sum(bit_matrix * shifts[None, :], axis=1, dtype=np.uint8)


def _block_quantize(weight_np: np.ndarray, bits: int, block_size: int = 32) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Helper to perform block-wise affine quantization to B bits [0, 2^B - 1]."""
    if block_size < 1:
        raise ValueError("block_size must be >= 1")
    w_flat = weight_np.flatten().astype(np.float32)
    n = len(w_flat)
    if n == 0:
        return np.zeros((0, block_size), dtype=np.uint8), np.zeros(0, dtype=np.float32), np.zeros(0, dtype=np.float32), 0
    pad_len = (block_size - (n % block_size)) % block_size
    if pad_len > 0:
        w_flat = np.pad(w_flat, (0, pad_len))

    blocks = w_flat.reshape(-1, block_size)
    b_min = np.min(blocks, axis=1, keepdims=True)
    b_max = np.max(blocks, axis=1, keepdims=True)

    max_val = (1 << bits) - 1
    scales = (b_max - b_min) / float(max_val)
    scales[scales == 0.0] = 1e-8

    q_blocks = np.clip(np.round((blocks - b_min) / scales), 0, max_val).astype(np.uint8)
    return q_blocks, scales.flatten(), b_min.flatten(), n


def _block_dequantize(q_blocks: np.ndarray, scales: np.ndarray, mins: np.ndarray, orig_shape: Sequence[int], n: int, block_size: int = 32) -> np.ndarray:
    """Helper to reconstruct float array from block-quantized indices, scales, and mins."""
    if n == 0:
        return np.zeros(orig_shape, dtype=np.float32)
    blocks_recon = q_blocks.astype(np.float32) * scales[:, None] + mins[:, None]
    w_flat = blocks_recon.flatten()[:n]
    return w_flat.reshape(orig_shape)


class Q2KQuantizer:
    """2-bit block quantizer with 4 values packed per uint8 byte."""
    BITS = 2

    @classmethod
    def quantize(cls, weight: Tensor, block_size: int = 32) -> QuantizedKBlock:
        w_np = weight.to_numpy()
        q_blocks, scales, mins, _ = _block_quantize(w_np, cls.BITS, block_size)
        packed = _pack_bits(q_blocks.flatten(), cls.BITS)
        return QuantizedKBlock(
            quant_type="q2_k",
            orig_shape=list(w_np.shape),
            block_size=block_size,
            bits=cls.BITS,
            data=packed,
            scales=scales,
            mins=mins,
        )

    @classmethod
    def dequantize(cls, block: QuantizedKBlock) -> Tensor:
        n = int(np.prod(block.orig_shape))
        num_blocks = len(block.scales)
        total_padded = num_blocks * block.block_size
        unpacked = _unpack_bits(block.data, cls.BITS, total_padded) if block.data.ndim == 1 else block.data.flatten()[:total_padded]
        q_blocks = unpacked.reshape(num_blocks, block.block_size) if num_blocks > 0 else np.zeros((0, block.block_size), dtype=np.uint8)
        recon = _block_dequantize(q_blocks, block.scales, block.mins, block.orig_shape, n, block.block_size)
        return _from_numpy(recon.astype(np.float32))


class Q3KQuantizer:
    """3-bit block quantizer with compact 3-bit sub-byte bit-packing."""
    BITS = 3

    @classmethod
    def quantize(cls, weight: Tensor, block_size: int = 32) -> QuantizedKBlock:
        w_np = weight.to_numpy()
        q_blocks, scales, mins, _ = _block_quantize(w_np, cls.BITS, block_size)
        packed = _pack_bits(q_blocks.flatten(), cls.BITS)
        return QuantizedKBlock(
            quant_type="q3_k",
            orig_shape=list(w_np.shape),
            block_size=block_size,
            bits=cls.BITS,
            data=q_blocks,
            scales=scales,
            mins=mins,
            packed_data=packed,
        )

    @classmethod
    def dequantize(cls, block: QuantizedKBlock) -> Tensor:
        n = int(np.prod(block.orig_shape))
        num_blocks = len(block.scales)
        total_padded = num_blocks * block.block_size
        if block.data.size >= total_padded and block.data.ndim > 1:
            q_blocks = block.data
        elif block.data.size >= total_padded:
            q_blocks = block.data.flatten()[:total_padded].reshape(num_blocks, block.block_size)
        elif block.packed_data is not None:
            unpacked = _unpack_bits(block.packed_data, cls.BITS, total_padded)
            q_blocks = unpacked.reshape(num_blocks, block.block_size)
        else:
            unpacked = _unpack_bits(block.data, cls.BITS, total_padded)
            q_blocks = unpacked.reshape(num_blocks, block.block_size) if num_blocks > 0 else np.zeros((0, block.block_size), dtype=np.uint8)
        recon = _block_dequantize(q_blocks, block.scales, block.mins, block.orig_shape, n, block.block_size)
        return _from_numpy(recon.astype(np.float32))


class Q5KQuantizer:
    """5-bit block quantizer (32 levels per block, compact 5-bit packing)."""
    BITS = 5

    @classmethod
    def quantize(cls, weight: Tensor, block_size: int = 32) -> QuantizedKBlock:
        w_np = weight.to_numpy()
        q_blocks, scales, mins, _ = _block_quantize(w_np, cls.BITS, block_size)
        packed = _pack_bits(q_blocks.flatten(), cls.BITS)
        return QuantizedKBlock(
            quant_type="q5_k",
            orig_shape=list(w_np.shape),
            block_size=block_size,
            bits=cls.BITS,
            data=q_blocks,
            scales=scales,
            mins=mins,
            packed_data=packed,
        )

    @classmethod
    def dequantize(cls, block: QuantizedKBlock) -> Tensor:
        n = int(np.prod(block.orig_shape))
        num_blocks = len(block.scales)
        total_padded = num_blocks * block.block_size
        if block.data.size >= total_padded and block.data.ndim > 1:
            q_blocks = block.data
        elif block.data.size >= total_padded:
            q_blocks = block.data.flatten()[:total_padded].reshape(num_blocks, block.block_size)
        elif block.packed_data is not None:
            unpacked = _unpack_bits(block.packed_data, cls.BITS, total_padded)
            q_blocks = unpacked.reshape(num_blocks, block.block_size)
        else:
            unpacked = _unpack_bits(block.data, cls.BITS, total_padded)
            q_blocks = unpacked.reshape(num_blocks, block.block_size) if num_blocks > 0 else np.zeros((0, block.block_size), dtype=np.uint8)
        recon = _block_dequantize(q_blocks, block.scales, block.mins, block.orig_shape, n, block.block_size)
        return _from_numpy(recon.astype(np.float32))


class Q6KQuantizer:
    """6-bit block quantizer (64 levels per block, compact 6-bit packing)."""
    BITS = 6

    @classmethod
    def quantize(cls, weight: Tensor, block_size: int = 32) -> QuantizedKBlock:
        w_np = weight.to_numpy()
        q_blocks, scales, mins, _ = _block_quantize(w_np, cls.BITS, block_size)
        packed = _pack_bits(q_blocks.flatten(), cls.BITS)
        return QuantizedKBlock(
            quant_type="q6_k",
            orig_shape=list(w_np.shape),
            block_size=block_size,
            bits=cls.BITS,
            data=q_blocks,
            scales=scales,
            mins=mins,
            packed_data=packed,
        )

    @classmethod
    def dequantize(cls, block: QuantizedKBlock) -> Tensor:
        n = int(np.prod(block.orig_shape))
        num_blocks = len(block.scales)
        total_padded = num_blocks * block.block_size
        if block.data.size >= total_padded and block.data.ndim > 1:
            q_blocks = block.data
        elif block.data.size >= total_padded:
            q_blocks = block.data.flatten()[:total_padded].reshape(num_blocks, block.block_size)
        elif block.packed_data is not None:
            unpacked = _unpack_bits(block.packed_data, cls.BITS, total_padded)
            q_blocks = unpacked.reshape(num_blocks, block.block_size)
        else:
            unpacked = _unpack_bits(block.data, cls.BITS, total_padded)
            q_blocks = unpacked.reshape(num_blocks, block.block_size) if num_blocks > 0 else np.zeros((0, block.block_size), dtype=np.uint8)
        recon = _block_dequantize(q_blocks, block.scales, block.mins, block.orig_shape, n, block.block_size)
        return _from_numpy(recon.astype(np.float32))


def quantize_ternary_bitnet(weight: Tensor) -> Tuple[Tensor, float]:
    """
    Quantizes weights into ternary values {-1, 0, 1} (Wang et al., 2024 - BitNet b1.58).
    Scale gamma = mean(abs(W)).
    W_ternary = clip(round(W / gamma), -1, 1).
    """
    w = weight.to_numpy()
    gamma = float(np.mean(np.abs(w))) + 1e-9
    w_scaled = w / gamma
    w_ternary = np.clip(np.round(w_scaled), -1.0, 1.0)
    return _from_numpy(w_ternary.astype(np.int8).astype(np.float32)), gamma


def dequantize_ternary_bitnet(ternary_weight: Tensor, scale: float) -> Tensor:
    """Reconstructs approximate full-precision weights from ternary representation."""
    tw = ternary_weight.to_numpy()
    return _from_numpy((tw * scale).astype(np.float32))


class BitLinear(Module):
    """
    BitNet 1.58-bit Linear Layer (Wang et al., 2024) with Straight-Through Estimator (STE) autograd.
    Uses ternary weights {-1, 0, 1} and dynamic 8-bit activation quantization.
    """
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        w = np.random.randn(out_features, in_features).astype(np.float32) * np.sqrt(2.0 / in_features)
        self.weight = Parameter(_from_numpy(w.astype(np.float32)))
        if bias:
            b = np.zeros(out_features, dtype=np.float32)
            self.bias = Parameter(_from_numpy(b))
        else:
            self.bias = None

    def forward(self, x: Tensor) -> Tensor:
        w_np = self.weight.to_numpy()
        x_np = x.to_numpy()

        # 1. Quantize weights to {-1, 0, 1} * gamma with STE
        gamma = float(np.mean(np.abs(w_np))) + 1e-9
        w_ternary = np.clip(np.round(w_np / gamma), -1.0, 1.0) * gamma
        w_diff = _from_numpy((w_ternary - w_np).astype(np.float32))
        w_ste = self.weight + w_diff

        # 2. Quantize activations to 8-bit absmax [-128, 127] with STE
        eta = float(np.max(np.abs(x_np))) + 1e-9
        x_dequant = np.clip(np.round(x_np * (127.0 / eta)), -128.0, 127.0) * (eta / 127.0)
        x_diff = _from_numpy((x_dequant - x_np).astype(np.float32))
        x_ste = x + x_diff

        out = x_ste.matmul(w_ste.transpose(0, 1))
        if self.bias is not None:
            out = out + self.bias
        return out


_QUANT_REGISTRY = {
    "q2": Q2KQuantizer,
    "q2_k": Q2KQuantizer,
    "q3": Q3KQuantizer,
    "q3_k": Q3KQuantizer,
    "q5": Q5KQuantizer,
    "q5_k": Q5KQuantizer,
    "q6": Q6KQuantizer,
    "q6_k": Q6KQuantizer,
}


def quantize_k_quant(weight: Tensor, quant_type: str = "q3_k", block_size: int = 32) -> QuantizedKBlock:
    """Unified entrypoint for GGUF-grade k-quants (q2_k, q3_k, q5_k, q6_k)."""
    q_key = quant_type.lower()
    if q_key not in _QUANT_REGISTRY:
        raise ValueError(f"Unsupported quant_type '{quant_type}'. Choose from: {list(_QUANT_REGISTRY.keys())}")
    quantizer = _QUANT_REGISTRY[q_key]
    return quantizer.quantize(weight, block_size=block_size)


def dequantize_k_quant(block: QuantizedKBlock) -> Tensor:
    """Unified dequantization entrypoint."""
    q_key = block.quant_type.lower()
    if q_key not in _QUANT_REGISTRY:
        raise ValueError(f"Unknown block quant_type '{block.quant_type}'")
    quantizer = _QUANT_REGISTRY[q_key]
    return quantizer.dequantize(block)
