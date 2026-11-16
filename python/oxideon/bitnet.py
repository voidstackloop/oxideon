"""BitNet b1.58: Trainable 1.58-Bit Ternary Linear Layer and Quantizers.

Implements BitNet b1.58 (Wang et al., 2024; Ma et al., 2024):
- WeightQuantizerSTE: Ternary {-1, 0, +1} weight quantization with Straight-Through Estimator
- AbsmaxQuantizer: Symmetric 8-bit integer activation scaling
- BitLinear158b: Trainable replacement for Linear layers in 1-bit LLMs
- pack_ternary_weights / unpack_ternary_weights: 2-bit Base-4 ternary weight compression (4 weights per byte)
"""

from typing import Optional, Tuple
import numpy as np

from .oxideon import Tensor as _Tensor, from_numpy as _from_numpy
from .nn import Module, Parameter, LayerNorm


class WeightQuantizerSTE:
    """Straight-Through Estimator (STE) for ternary {-1, 0, +1} weight quantization."""

    @staticmethod
    def forward(w: np.ndarray) -> Tuple[np.ndarray, float]:
        """Quantizes weight matrix to {-1, 0, +1} scaled by average absolute magnitude."""
        gamma = float(np.mean(np.abs(w))) + 1e-8
        scaled_w = w / gamma
        # Round to nearest integer {-1, 0, +1} and clip
        quantized = np.clip(np.round(scaled_w), -1.0, 1.0)
        return quantized, gamma


class AbsmaxQuantizer:
    """Symmetric per-tensor absmax activation quantizer to 8-bit integers."""

    @staticmethod
    def forward(x: np.ndarray, num_bits: int = 8) -> Tuple[np.ndarray, float]:
        if num_bits < 2:
            raise ValueError(f"num_bits must be >= 2, got {num_bits}")
        q_max = 2 ** (num_bits - 1) - 1  # 127 for int8
        scale = float(np.max(np.abs(x))) + 1e-8
        quantized = np.clip(np.round(x * (q_max / scale)), -q_max, q_max)
        return quantized, scale


def pack_ternary_weights(w_ternary: np.ndarray) -> np.ndarray:
    """Packs ternary weights in {-1, 0, +1} into uint8 (4 weights per byte, 2 bits each)."""
    flat = np.round(np.clip(w_ternary, -1.0, 1.0)).astype(np.int8).reshape(-1)
    # Map {-1, 0, +1} -> {0, 1, 2}
    codes = (flat + 1).astype(np.uint8)
    rem = len(codes) % 4
    if rem != 0:
        codes = np.pad(codes, (0, 4 - rem), constant_values=1)
    codes_4 = codes.reshape(-1, 4)
    packed = (
        codes_4[:, 0]
        | (codes_4[:, 1] << 2)
        | (codes_4[:, 2] << 4)
        | (codes_4[:, 3] << 6)
    ).astype(np.uint8)
    return packed


def unpack_ternary_weights(packed: np.ndarray, orig_shape: Tuple[int, ...]) -> np.ndarray:
    """Unpacks uint8 2-bit packed ternary weights back to float32 in {-1.0, 0.0, +1.0}."""
    p = np.asarray(packed, dtype=np.uint8).reshape(-1)
    c0 = p & 0x03
    c1 = (p >> 2) & 0x03
    c2 = (p >> 4) & 0x03
    c3 = (p >> 6) & 0x03
    codes = np.stack([c0, c1, c2, c3], axis=-1).reshape(-1)
    total = int(np.prod(orig_shape))
    vals = (codes[:total].astype(np.float32) - 1.0).reshape(orig_shape)
    return vals


class BitLinear158b(Module):
    """Trainable BitNet b1.58 Linear Layer.

    In the forward pass:
    - Weights are quantized to {-1, 0, +1} ternary values.
    - Activations are quantized to 8-bit integers via absmax scaling.
    - In the backward pass, gradients flow through the continuous latent weight matrix via STE.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = False,
        eps: float = 1e-5,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.eps = eps

        # Continuous latent weight parameter
        init_scale = 1.0 / np.sqrt(in_features)
        w_init = np.random.uniform(-init_scale, init_scale, (out_features, in_features)).astype(np.float32)
        self.weight = Parameter(_from_numpy(w_init))

        if bias:
            b_init = np.zeros(out_features, dtype=np.float32)
            self.bias: Optional[Parameter] = Parameter(_from_numpy(b_init))
        else:
            self.bias = None

        self.norm = LayerNorm(in_features)

    def forward(self, x: _Tensor) -> _Tensor:
        """Forward pass with 8-bit activation and ternary weight STE computation."""
        # 1. Normalize input activations
        normed_x = self.norm(x)
        x_np = normed_x.to_numpy()

        # 2. Absmax quantize activations to int8 with STE
        x_q, x_scale = AbsmaxQuantizer.forward(x_np, num_bits=8)
        x_dequant_np = (x_q * (x_scale / 127.0)).astype(np.float32)
        x_q_tensor = _from_numpy(np.ascontiguousarray(x_dequant_np))
        x_ste = normed_x + (x_q_tensor - normed_x).detach()

        # 3. Quantize latent weights to ternary {-1, 0, +1} with STE
        w_np = self.weight.to_numpy()
        w_q, w_scale = WeightQuantizerSTE.forward(w_np)
        w_dequant_np = (w_q * w_scale).astype(np.float32)
        w_q_tensor = _from_numpy(np.ascontiguousarray(w_dequant_np))
        w_ste = self.weight + (w_q_tensor - self.weight).detach()

        # 4. Differentiable linear projection
        orig_shape = list(x_ste.shape)
        x_2d = x_ste.reshape([-1, self.in_features])
        out_2d = x_2d.matmul(w_ste.transpose(0, 1))

        if self.bias is not None:
            out_2d = out_2d + self.bias.reshape([1, self.out_features])

        out_shape = orig_shape[:-1] + [self.out_features]
        return out_2d.reshape(out_shape)

    def get_ternary_weights(self) -> Tuple[np.ndarray, float]:
        """Returns the discrete ternary weight array {-1, 0, +1} and its scalar scaling factor gamma."""
        w_np = self.weight.to_numpy()
        return WeightQuantizerSTE.forward(w_np)


__all__ = [
    "WeightQuantizerSTE",
    "AbsmaxQuantizer",
    "BitLinear158b",
    "pack_ternary_weights",
    "unpack_ternary_weights",
]
