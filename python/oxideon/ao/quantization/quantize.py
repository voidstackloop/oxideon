"""Model preparation and conversion workflows for PTQ and QAT in Oxideon AO."""

from typing import Any, Dict, Optional
import copy

from ...oxideon import Tensor as _Tensor
from ...nn import Module, Linear, Conv2d
from .observer import ObserverBase, MinMaxObserver
from .fake_quantize import FakeQuantize
from .qconfig import QConfig, get_default_qconfig, get_default_qat_qconfig


class _QuantizedLayerWrapper(Module):
    """Wraps a layer with activation and weight observers/fake-quantizers for QAT."""

    def __init__(self, layer: Any, act_obs: Any, weight_obs: Any):
        super().__init__()
        self.layer = layer
        self.activation_post_process = act_obs
        self.weight_fake_quant = weight_obs

    def forward(self, x: _Tensor) -> _Tensor:
        orig_weight = getattr(self.layer, "weight", None)
        try:
            if orig_weight is not None and self.weight_fake_quant is not None:
                self.layer.weight = self.weight_fake_quant(orig_weight)
            out = self.layer(x)
            return self.activation_post_process(out)
        finally:
            if orig_weight is not None:
                self.layer.weight = orig_weight


class _QuantizedInferenceLayer(Module):
    """Represents a frozen quantized layer for deployment."""

    def __init__(self, layer: Any, act_scale: float, act_zp: int, weight_scale: float, weight_zp: int):
        super().__init__()
        self.layer = layer
        self.act_scale = act_scale
        self.act_zero_point = act_zp
        self.weight_scale = weight_scale
        self.weight_zero_point = weight_zp

    def forward(self, x: _Tensor) -> _Tensor:
        return self.layer(x)


def prepare(
    model: Module,
    qconfig: Optional[QConfig] = None,
    qconfig_dict: Optional[Dict[str, QConfig]] = None,
    inplace: bool = False,
) -> Module:
    """Prepares a model for calibration or QAT by wrapping layers with observers or fake-quant modules."""
    if not inplace:
        model = copy.deepcopy(model)

    default_qcfg = qconfig or getattr(model, "qconfig", None) or get_default_qat_qconfig()

    def _insert_observers(mod: Module, prefix: str = ""):
        for name, sub in list(getattr(mod, "_modules", {}).items()):
            fqn = f"{prefix}{name}"
            if isinstance(sub, _QuantizedLayerWrapper):
                continue
            if isinstance(sub, (Linear, Conv2d)) or sub.__class__.__name__ in ("Linear", "Conv2d"):
                qcfg = (
                    (qconfig_dict.get(fqn) if qconfig_dict else None)
                    or (qconfig_dict.get(name) if qconfig_dict else None)
                    or default_qcfg
                )
                act_obs = qcfg.activation() if callable(qcfg.activation) else copy.deepcopy(qcfg.activation)
                weight_obs = qcfg.weight() if callable(qcfg.weight) else copy.deepcopy(qcfg.weight)
                wrapped = _QuantizedLayerWrapper(sub, act_obs, weight_obs)
                setattr(mod, name, wrapped)
            elif isinstance(sub, Module):
                _insert_observers(sub, fqn + ".")

    _insert_observers(model)
    return model


def convert(model: Module, inplace: bool = False) -> Module:
    """Converts a calibrated or QAT model into a quantized inference model."""
    if not inplace:
        model = copy.deepcopy(model)

    def _freeze_and_convert(mod: Module):
        for name, sub in list(getattr(mod, "_modules", {}).items()):
            if isinstance(sub, _QuantizedLayerWrapper):
                act_scale = getattr(sub.activation_post_process, "scale", 1.0)
                act_zp = getattr(sub.activation_post_process, "zero_point", 0)
                weight_scale = getattr(sub.weight_fake_quant, "scale", 1.0)
                weight_zp = getattr(sub.weight_fake_quant, "zero_point", 0)
                q_inf = _QuantizedInferenceLayer(sub.layer, act_scale, act_zp, weight_scale, weight_zp)
                setattr(mod, name, q_inf)
            elif isinstance(sub, Module):
                _freeze_and_convert(sub)

    _freeze_and_convert(model)
    model.eval()
    return model


__all__ = [
    "prepare",
    "convert",
]
