"""Quantization-Aware Training (QAT) fake-quantization modules for Oxideon AO."""

from typing import Optional, Tuple
import numpy as np

from ...oxideon import Tensor as _Tensor, from_numpy as _from_numpy
from ...nn import Module
from .observer import MinMaxObserver, ObserverBase


class FakeQuantize(Module):
    """Simulates lower-precision integer quantization during floating-point training (QAT).

    Forward applies fake quantization, backward uses the straight-through estimator (STE).
    """

    def __init__(
        self,
        observer: Optional[ObserverBase] = None,
        quant_min: int = -128,
        quant_max: int = 127,
        dtype: str = "qint8",
    ):
        super().__init__()
        if quant_min >= quant_max:
            raise ValueError(f"quant_min ({quant_min}) must be < quant_max ({quant_max})")
        self.activation_post_process = observer or MinMaxObserver(dtype=dtype)
        obs_qmin = getattr(self.activation_post_process, "qmin", quant_min)
        obs_qmax = getattr(self.activation_post_process, "qmax", quant_max)
        if obs_qmin != quant_min or obs_qmax != quant_max:
            raise ValueError(
                f"observer range [{obs_qmin}, {obs_qmax}] must match "
                f"quant_min/quant_max [{quant_min}, {quant_max}]"
            )
        self.quant_min = quant_min
        self.quant_max = quant_max
        self.scale = 1.0
        self.zero_point = 0
        self.enabled = True

    def forward(self, x: _Tensor) -> _Tensor:
        if not self.enabled:
            return x

        # Update calibration parameters
        self.scale, self.zero_point = self.activation_post_process.calculate_qparams(x)

        # Quantize and dequantize in floating-point
        arr = x.to_numpy()
        q = np.clip(np.round(arr / self.scale) + self.zero_point, self.quant_min, self.quant_max)
        dq = (q - self.zero_point) * self.scale
        dq_t = _from_numpy(np.ascontiguousarray(dq, dtype=x.to_numpy().dtype))

        # Straight-Through Estimator (STE): forward yields dq_t, backward flows gradient directly through x
        return x + (dq_t - x).detach()

    def extra_repr(self) -> str:
        return f"fake_quant_enabled={self.enabled}, quant_min={self.quant_min}, quant_max={self.quant_max}, scale={self.scale:.6f}, zero_point={self.zero_point}"


__all__ = ["FakeQuantize"]
