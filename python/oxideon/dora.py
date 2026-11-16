"""DoRA (Weight-Decomposed Low-Rank Adaptation) and LoRA+ fine-tuning algorithms.

Implements:
  - DoRALinear: Weight-Decomposed Low-Rank Adaptation (Liu et al., ICML 2024)
    Decomposes weights into magnitude and directional components:
    $$W = m \\frac{V + \\Delta V}{\\|V + \\Delta V\\|_c}$$
  - LoRAPlusLinear: Low-rank adaptation with asymmetric learning rate dynamics (Hayou et al., 2024)
"""

from typing import Optional, Any
import math
import numpy as np

from .oxideon import Tensor as _Tensor, from_numpy as _from_numpy
from .nn import Module, Parameter


class DoRALinear(Module):
    """Weight-Decomposed Low-Rank Adaptation (DoRA) Linear layer."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        r: int = 8,
        lora_alpha: float = 16.0,
        bias: bool = True,
        base_linear: Optional[Any] = None,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.r = r
        self.lora_alpha = lora_alpha
        self.scaling = lora_alpha / float(r) if r > 0 else 1.0
        self.merged = False

        if base_linear is not None:
            self.weight = base_linear.weight
            self.bias = getattr(base_linear, "bias", None)
        else:
            self.weight = Parameter(_Tensor.randn([out_features, in_features]))
            self.bias = Parameter(_Tensor.zeros([out_features])) if bias else None

        if hasattr(self.weight, "requires_grad_"):
            self.weight.requires_grad_(False)
        elif hasattr(self.weight, "requires_grad"):
            self.weight.requires_grad = False

        if self.bias is not None:
            if hasattr(self.bias, "requires_grad_"):
                self.bias.requires_grad_(False)
            elif hasattr(self.bias, "requires_grad"):
                self.bias.requires_grad = False

        w_np = self.weight.to_numpy()
        initial_m = np.linalg.norm(w_np, axis=1, keepdims=True).astype(np.float32)  # [out_features, 1]
        self.magnitude = Parameter(_from_numpy(initial_m), requires_grad=True)

        if r > 0:
            std_a = 1.0 / math.sqrt(r)
            a_init = np.random.normal(0.0, std_a, size=(r, in_features)).astype(np.float32)
            b_init = np.zeros((out_features, r), dtype=np.float32)

            self.lora_A = Parameter(_from_numpy(np.ascontiguousarray(a_init)), requires_grad=True)
            self.lora_B = Parameter(_from_numpy(np.ascontiguousarray(b_init)), requires_grad=True)
        else:
            self.lora_A = None
            self.lora_B = None

    def _get_weight_tensor(self) -> _Tensor:
        w_base = self.weight.detach()
        if self.r > 0 and not self.merged and self.lora_A is not None and self.lora_B is not None:
            delta_v = self.lora_B.matmul(self.lora_A) * float(self.scaling)
            v = w_base + delta_v
        else:
            v = w_base

        norm_v = ((v * v).sum(dim=1, keepdim=True) + 1e-12).sqrt()
        v_normed = v / norm_v
        return self.magnitude * v_normed

    def _get_weight(self) -> np.ndarray:
        return self._get_weight_tensor().to_numpy()

    def forward(self, x: _Tensor) -> _Tensor:
        if self.merged:
            out = x @ self.weight.t()
            if self.bias is not None:
                out = out + self.bias
            return out

        w_tensor = self._get_weight_tensor()
        out = x @ w_tensor.t()

        if self.bias is not None:
            out = out + self.bias
        return out

    def merge(self):
        """Merges DoRA directional and magnitude parameters into the base weight."""
        if not self.merged:
            merged_w = self._get_weight()
            self.weight = Parameter(_from_numpy(np.ascontiguousarray(merged_w.astype(np.float32))), requires_grad=False)
            self.merged = True


class LoRAPlusLinear(Module):
    """LoRA+ linear layer with asymmetric scaling ratio between A and B matrices (Hayou et al., 2024)."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        r: int = 8,
        lora_alpha: float = 16.0,
        b_ratio: float = 4.0,
        bias: bool = True,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.r = r
        self.lora_alpha = lora_alpha
        self.b_ratio = b_ratio
        self.scaling = (lora_alpha / float(r)) if r > 0 else 1.0

        self.weight = Parameter(_Tensor.randn([out_features, in_features]), requires_grad=False)
        self.bias = Parameter(_Tensor.zeros([out_features]), requires_grad=False) if bias else None

        if r > 0:
            std_a = 1.0 / math.sqrt(r)
            a_init = np.random.normal(0.0, std_a, size=(r, in_features)).astype(np.float32)
            b_init = np.zeros((out_features, r), dtype=np.float32)

            self.lora_A = Parameter(_from_numpy(np.ascontiguousarray(a_init)), requires_grad=True)
            self.lora_B = Parameter(_from_numpy(np.ascontiguousarray(b_init)), requires_grad=True)
        else:
            self.lora_A = None
            self.lora_B = None

    def forward(self, x: _Tensor) -> _Tensor:
        result = x @ self.weight.t()
        if self.bias is not None:
            result = result + self.bias

        if self.r > 0:
            lora_out = (x @ self.lora_A.t()) @ self.lora_B.t()
            result = result + (lora_out * (self.scaling * self.b_ratio))

        return result


__all__ = [
    "DoRALinear",
    "LoRAPlusLinear",
]
