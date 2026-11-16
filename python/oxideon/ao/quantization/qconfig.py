"""Quantization configuration (QConfig) for Oxideon AO."""

from typing import Any, Callable, NamedTuple, Optional

from .observer import MinMaxObserver, MovingAverageMinMaxObserver, ObserverBase
from .fake_quantize import FakeQuantize


class QConfig(NamedTuple):
    """Encapsulates the observer or fake_quantize module constructors for activations and weights."""
    activation: Any
    weight: Any


def get_default_qconfig(backend: str = "fbgemm") -> QConfig:
    """Returns the default Post-Training Quantization (PTQ) QConfig."""
    if backend not in ("fbgemm", "qnnpack"):
        raise ValueError(f"unknown quantization backend {backend!r} (expected 'fbgemm' or 'qnnpack')")
    if backend == "qnnpack":
        return QConfig(
            activation=lambda: MinMaxObserver(dtype="quint8"),
            weight=lambda: MinMaxObserver(dtype="qint8"),
        )
    return QConfig(
        activation=lambda: MinMaxObserver(dtype="qint8"),
        weight=lambda: MinMaxObserver(dtype="qint8"),
    )


def get_default_qat_qconfig(backend: str = "fbgemm") -> QConfig:
    """Returns the default Quantization-Aware Training (QAT) QConfig."""
    if backend not in ("fbgemm", "qnnpack"):
        raise ValueError(f"unknown quantization backend {backend!r} (expected 'fbgemm' or 'qnnpack')")
    if backend == "qnnpack":
        return QConfig(
            activation=lambda: FakeQuantize(observer=MovingAverageMinMaxObserver(dtype="quint8"), quant_min=0, quant_max=255, dtype="quint8"),
            weight=lambda: FakeQuantize(observer=MinMaxObserver(dtype="qint8"), quant_min=-128, quant_max=127, dtype="qint8"),
        )
    return QConfig(
        activation=lambda: FakeQuantize(observer=MovingAverageMinMaxObserver(dtype="qint8"), quant_min=-128, quant_max=127, dtype="qint8"),
        weight=lambda: FakeQuantize(observer=MinMaxObserver(dtype="qint8"), quant_min=-128, quant_max=127, dtype="qint8"),
    )


__all__ = [
    "QConfig",
    "get_default_qconfig",
    "get_default_qat_qconfig",
]
