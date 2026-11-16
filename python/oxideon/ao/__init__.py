"""Oxideon Architecture Optimization (AO) suite.

Provides quantization (PTQ, QAT, FakeQuantize, Observers), sparsification,
and low-bit model compilation optimizations.
"""

from . import quantization
import sys

__path__ = []
sys.modules["oxideon.ao.quantization"] = quantization

__all__ = [
    "quantization",
]
