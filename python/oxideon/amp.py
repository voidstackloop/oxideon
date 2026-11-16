"""Automatic Mixed Precision (AMP) training support for Oxideon."""

import contextlib
import math
from typing import Any, Dict, Optional
import numpy as np

from .oxideon import Tensor as _Tensor, from_numpy as _from_numpy


class _AmpState:
    enabled: bool = False
    dtype: str = "f16"


_amp_state = _AmpState()


@contextlib.contextmanager
def autocast(enabled: bool = True, dtype: str = "f16"):
    """Context manager for automatic mixed precision execution."""
    prev_enabled = _amp_state.enabled
    prev_dtype = _amp_state.dtype
    _amp_state.enabled = enabled
    _amp_state.dtype = dtype
    try:
        yield
    finally:
        _amp_state.enabled = prev_enabled
        _amp_state.dtype = prev_dtype


def is_autocast_enabled() -> bool:
    return _amp_state.enabled


def get_autocast_dtype() -> str:
    return _amp_state.dtype


class GradScaler:
    """Dynamic gradient scaling for half-precision mixed-precision training.

    Prevents numeric underflow in FP16 gradients during backpropagation.
    """

    def __init__(
        self,
        init_scale: float = 65536.0,
        growth_factor: float = 2.0,
        backoff_factor: float = 0.5,
        growth_interval: int = 2000,
        enabled: bool = True,
    ):
        self.scale_factor = float(init_scale)
        self.growth_factor = float(growth_factor)
        self.backoff_factor = float(backoff_factor)
        self.growth_interval = growth_interval
        self.enabled = enabled

        self._growth_tracker = 0
        self._found_inf = False
        self._unscaled_optimizers = set()

    def scale(self, loss: Any) -> Any:
        """Multiplies loss by scale factor before backprop."""
        if not self.enabled:
            return loss
        return loss * self.scale_factor

    def _iter_optimizer_params(self, optimizer: Any):
        if hasattr(optimizer, "param_groups"):
            for group in optimizer.param_groups:
                for p in group.get("params", []):
                    yield p
        elif hasattr(optimizer, "parameters"):
            params = optimizer.parameters() if callable(optimizer.parameters) else optimizer.parameters
            for p in params:
                yield p

    def unscale_(self, optimizer: Any) -> None:
        """Divides gradients of optimizer's parameters by scale factor and checks for Inf/NaN."""
        if not self.enabled or id(optimizer) in self._unscaled_optimizers:
            return
        self._found_inf = False
        inv_scale = 1.0 / self.scale_factor

        for p in self._iter_optimizer_params(optimizer):
            g = p.grad() if hasattr(p, "grad") and callable(p.grad) else getattr(p, "grad", None)
            if g is not None:
                g_np = g.to_numpy() if isinstance(g, _Tensor) else np.asarray(g, dtype=np.float32)
                if not np.all(np.isfinite(g_np)):
                    self._found_inf = True
                unscaled = _from_numpy(np.ascontiguousarray(g_np * inv_scale, dtype=np.float32))
                try:
                    p.grad = unscaled
                except AttributeError:
                    if hasattr(p, "set_grad"):
                        p.set_grad(unscaled)

        self._unscaled_optimizers.add(id(optimizer))

    def step(self, optimizer: Any) -> bool:
        """Unscales gradients and steps the optimizer if no infinite or NaN values are found."""
        if not self.enabled:
            optimizer.step()
            return True

        self.unscale_(optimizer)

        if not self._found_inf:
            optimizer.step()
            return True
        return False

    def update(self, new_scale: Optional[float] = None):
        """Updates the scale factor based on gradient validity."""
        self._unscaled_optimizers.clear()
        if not self.enabled:
            return
        if new_scale is not None:
            self.scale_factor = float(new_scale)
            self._growth_tracker = 0
            self._found_inf = False
            return

        if self._found_inf:
            self.scale_factor *= self.backoff_factor
            self._growth_tracker = 0
        else:
            self._growth_tracker += 1
            if self._growth_tracker >= self.growth_interval:
                self.scale_factor *= self.growth_factor
                self._growth_tracker = 0
        self._found_inf = False

    def get_scale(self) -> float:
        return self.scale_factor

    def state_dict(self) -> Dict[str, Any]:
        return {
            "scale": self.scale_factor,
            "growth_factor": self.growth_factor,
            "backoff_factor": self.backoff_factor,
            "growth_interval": self.growth_interval,
            "_growth_tracker": self._growth_tracker,
        }

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        self.scale_factor = float(state_dict["scale"])
        self.growth_factor = float(state_dict["growth_factor"])
        self.backoff_factor = float(state_dict["backoff_factor"])
        self.growth_interval = int(state_dict["growth_interval"])
        self._growth_tracker = int(state_dict.get("_growth_tracker", 0))


__all__ = [
    "autocast",
    "is_autocast_enabled",
    "get_autocast_dtype",
    "GradScaler",
]
