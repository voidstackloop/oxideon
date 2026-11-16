"""No builtin shadowing: module-level tensor ops must not hijack builtins.

Regression tests: ``primitives.py`` defines tensor ops named ``any``/``all``/
``sum``/``abs``/``round``, which shadow the builtins module-wide. Internal
uses that meant the builtins broke:
- ``_tensor_getitem`` used builtin ``any`` -> every tuple getitem silently
  took the NumPy-copy branch and detached from autograd.
- ``diag_embed`` used builtin ``abs`` -> crashed on every call.
- ``Function.apply`` used builtin ``any`` -> custom-op outputs always
  required grad even with no grad inputs.
"""
import numpy as np
import pytest

torch = pytest.importorskip("torch")

import oxideon as ox


def test_tuple_getitem_preserves_grad():
    s = ox.from_numpy(np.arange(24, dtype=np.float32).reshape(4, 6)).requires_grad_(True)
    v = s[1:3, 1:4]
    assert v.requires_grad is True
    assert list(v.strides) == [6, 1]
    v.sum().backward()
    g = s.grad.to_numpy()
    expect = np.zeros((4, 6), dtype=np.float32)
    expect[1:3, 1:4] = 1.0
    assert np.array_equal(g, expect)


def test_step_getitem_preserves_grad():
    s = ox.from_numpy(np.arange(24, dtype=np.float32).reshape(4, 6)).requires_grad_(True)
    v = s[:, ::2]
    assert v.requires_grad is True
    v.sum().backward()
    g = s.grad.to_numpy()
    expect = np.zeros((4, 6), dtype=np.float32)
    expect[:, ::2] = 1.0
    assert np.array_equal(g, expect)


def test_diag_embed_runs():
    d = ox.from_numpy(np.array([1.0, 2.0, 3.0], dtype=np.float32))
    m = ox.diag_embed(d)
    assert m.to_numpy().tolist() == [[1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 3.0]]
    m2 = ox.diag_embed(d, offset=1)
    assert m2.shape == ox.Size([4, 4])
    assert m2.to_numpy()[0, 1] == 1.0


def test_size_one_keeps_shape_and_grad():
    s = ox.from_numpy(np.arange(16, dtype=np.float32).reshape(4, 4)).requires_grad_(True)
    v = s[1:2, 2:3]
    assert list(v.shape) == [1, 1]
    assert v.requires_grad is True
    v.sum().backward()
    g = s.grad.to_numpy()
    expect = np.zeros((4, 4), dtype=np.float32)
    expect[1:2, 2:3] = 1.0
    assert np.array_equal(g, expect)


def test_custom_function_no_over_tracking():
    class Double(ox.Function):
        @staticmethod
        def forward(ctx, x):
            return x * 2.0

        @staticmethod
        def backward(ctx, g):
            return g * 2.0

    x = ox.from_numpy(np.ones([3], dtype=np.float32))
    assert x.requires_grad is False
    y = Double.apply(x)
    assert y.requires_grad is False
    assert y.to_numpy().tolist() == [2.0, 2.0, 2.0]
