"""Calibration observers for post-training quantization and QAT in Oxideon AO."""

from typing import Tuple, Union
import numpy as np

from ...oxideon import Tensor as _Tensor, from_numpy as _from_numpy
from ...nn import Module


class ObserverBase(Module):
    """Base class for all calibration observers."""

    def __init__(self, dtype: str = "qint8", qscheme: str = "per_tensor_affine"):
        super().__init__()
        if dtype not in ("qint8", "quint8"):
            raise ValueError(f"dtype must be 'qint8' or 'quint8', got {dtype!r}")
        if qscheme != "per_tensor_affine":
            raise ValueError(f"only qscheme 'per_tensor_affine' is supported, got {qscheme!r}")
        self.dtype = dtype
        self.qscheme = qscheme
        if dtype == "qint8":
            self.qmin = -128
            self.qmax = 127
        else:  # quint8
            self.qmin = 0
            self.qmax = 255

    def forward(self, x: _Tensor) -> _Tensor:
        self.calculate_qparams(x)
        return x

    def calculate_qparams(self, x: _Tensor) -> Tuple[float, int]:
        raise NotImplementedError


class MinMaxObserver(ObserverBase):
    """Observer computing quantization scale and zero_point from min and max observed values."""

    def __init__(self, dtype: str = "qint8", qscheme: str = "per_tensor_affine"):
        super().__init__(dtype=dtype, qscheme=qscheme)
        self.min_val = float("inf")
        self.max_val = float("-inf")
        self.scale = 1.0
        self.zero_point = 0

    def calculate_qparams(self, x: _Tensor) -> Tuple[float, int]:
        arr = x.to_numpy()
        if arr.size == 0:
            raise ValueError("Input tensor cannot be empty")
        cur_min = float(np.min(arr))
        cur_max = float(np.max(arr))
        self.min_val = min(self.min_val, cur_min)
        self.max_val = max(self.max_val, cur_max)

        # Ensure range includes 0
        min_val = min(0.0, self.min_val)
        max_val = max(0.0, self.max_val)

        if max_val == min_val:
            self.scale = 1.0
            self.zero_point = 0
        else:
            self.scale = (max_val - min_val) / float(self.qmax - self.qmin)
            initial_zero_point = self.qmin - round(min_val / self.scale)
            self.zero_point = int(np.clip(initial_zero_point, self.qmin, self.qmax))

        return self.scale, self.zero_point


class MovingAverageMinMaxObserver(MinMaxObserver):
    """Observer tracking running min and max values with exponential moving average (EMA)."""

    def __init__(self, averaging_constant: float = 0.01, dtype: str = "qint8", qscheme: str = "per_tensor_affine"):
        super().__init__(dtype=dtype, qscheme=qscheme)
        if not (0.0 <= averaging_constant <= 1.0):
            raise ValueError(f"averaging_constant must be between 0.0 and 1.0, got {averaging_constant}")
        self.averaging_constant = averaging_constant
        self._first_observed = False

    def calculate_qparams(self, x: _Tensor) -> Tuple[float, int]:
        arr = x.to_numpy()
        if arr.size == 0:
            raise ValueError("Input tensor cannot be empty")
        cur_min = float(np.min(arr))
        cur_max = float(np.max(arr))

        if not self._first_observed:
            self.min_val = cur_min
            self.max_val = cur_max
            self._first_observed = True
        else:
            c = self.averaging_constant
            self.min_val = self.min_val + c * (cur_min - self.min_val)
            self.max_val = self.max_val + c * (cur_max - self.max_val)

        min_val = min(0.0, self.min_val)
        max_val = max(0.0, self.max_val)

        if max_val == min_val:
            self.scale = 1.0
            self.zero_point = 0
        else:
            self.scale = (max_val - min_val) / float(self.qmax - self.qmin)
            initial_zero_point = self.qmin - round(min_val / self.scale)
            self.zero_point = int(np.clip(initial_zero_point, self.qmin, self.qmax))

        return self.scale, self.zero_point


__all__ = [
    "ObserverBase",
    "MinMaxObserver",
    "MovingAverageMinMaxObserver",
]
