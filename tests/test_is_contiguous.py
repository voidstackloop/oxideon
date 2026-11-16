"""Tensor.is_contiguous() must report real layout contiguity (torch parity).

Regression test: the Python patch layer used to overwrite the native method
with ``lambda self: True``, so transposed views wrongly reported contiguous.
"""
import numpy as np
import pytest

torch = pytest.importorskip("torch")

import oxideon as ox


def ot(arr):
    return ox.from_numpy(np.ascontiguousarray(arr))


# (name, oxideon view fn, torch view fn or None, shape, expected)
CASES = [
    ("plain-2d", lambda t: t, lambda t: t, (3, 4), True),
    ("plain-3d", lambda t: t, lambda t: t, (2, 3, 4), True),
    ("transpose", lambda t: t.transpose(0, 1), lambda t: t.t(), (3, 4), False),
    ("transpose-3d", lambda t: t.transpose(0, 2), lambda t: t.transpose(0, 2), (2, 3, 4), False),
    ("strided-slice", lambda t: t[:, ::2], lambda t: t[:, ::2], (4, 6), False),
    ("offset-slice", lambda t: t[1:, 1:], lambda t: t[1:, 1:], (4, 6), False),
    ("unsqueeze", lambda t: t.unsqueeze(0), lambda t: t.unsqueeze(0), (3, 4), True),
    ("single-elem", lambda t: t, lambda t: t, (1, 1), True),
    ("empty", lambda t: t, lambda t: t, (0, 5), True),
    ("size1-dim", lambda t: t, lambda t: t, (1, 5, 1), True),
]


@pytest.mark.parametrize("name,ox_fn,torch_fn,shape,expected", CASES)
def test_is_contiguous(name, ox_fn, torch_fn, shape, expected):
    a = np.arange(int(np.prod(shape)), dtype=np.float32).reshape(shape)
    o = ox_fn(ot(a))
    assert o.is_contiguous() is expected
    if torch_fn is not None:
        t = torch_fn(torch.from_numpy(a.copy()))
        assert o.is_contiguous() is t.is_contiguous()


def test_transpose_reports_non_contiguous():
    t = ox.from_numpy(np.arange(12, dtype=np.float32).reshape(3, 4))
    assert t.is_contiguous() is True
    assert t.transpose(0, 1).is_contiguous() is False
