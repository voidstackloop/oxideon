"""Quantization-aware training (QAT), Post-training quantization (PTQ), and observers for Oxideon."""

from .observer import ObserverBase, MinMaxObserver, MovingAverageMinMaxObserver
from .fake_quantize import FakeQuantize
from .qconfig import QConfig, get_default_qconfig, get_default_qat_qconfig
from .quantize import prepare, convert

__all__ = [
    "ObserverBase",
    "MinMaxObserver",
    "MovingAverageMinMaxObserver",
    "FakeQuantize",
    "QConfig",
    "get_default_qconfig",
    "get_default_qat_qconfig",
    "prepare",
    "convert",
]
