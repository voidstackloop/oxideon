"""Tests of oxideon.tensor_methods: the PyTorch-style Tensor methods (method forms, comparison and division aliases,
in-place variants) against NumPy, plus the in-place safety rules."""
import math

import numpy as np
import pytest

import oxideon as ox
from oxideon import tensor_methods

rng = np.random.default_rng(0)


def T(a):
    return ox.from_numpy(np.ascontiguousarray(a))


def N(t):
    return t.to_numpy()


def close(t, ref, tol=1e-5):
    np.testing.assert_allclose(N(t), ref, rtol=tol, atol=tol)


X = rng.standard_normal((3, 4)).astype(np.float32)
Y = rng.standard_normal((3, 4)).astype(np.float32)
POS = (np.abs(X) + 0.5).astype(np.float32)


def test_methods_are_attached_and_do_not_replace_existing_ones():
    assert len(tensor_methods.ADDED) >= 140
    for name in tensor_methods.ADDED:
        assert getattr(ox.Tensor, name).__module__ == "oxideon.tensor_methods" or callable(getattr(ox.Tensor, name))
    assert "add_" not in tensor_methods.ADDED and "exp" not in tensor_methods.ADDED  # already provided elsewhere


# ---- method forms
def test_method_forms_match_functions():
    x = T(X)
    close(x.logsumexp(1), np.log(np.exp(X.astype(np.float64)).sum(1)))
    close(x.movedim(0, 1), X.T)
    close(x.swapaxes(0, 1), X.T)
    close(x.ravel(), X.ravel())
    close(x.msort(), np.sort(X, 0))
    close(x.lerp(T(Y), 0.3), X + 0.3 * (Y - X))
    close(x.amax(1), X.max(1))
    close(x.amin(0), X.min(0))
    close(x.sgn(), np.sign(X))
    close(x.negative(), -X)
    assert [tuple(p.shape) for p in x.tensor_split(3, 1)] == [(3, 2), (3, 1), (3, 1)]
    assert x.isposinf().to_numpy().sum() == 0
    v = T(np.array([1.0, 2.0, 3.0], np.float32))
    close(v.outer(v), np.outer([1, 2, 3], [1, 2, 3]))
    assert v.dot(v).item() == 14.0
    close(T(X[:, :3]).mv(v), X[:, :3] @ np.array([1, 2, 3], np.float32))
    assert T(np.arange(9, dtype=np.float32).reshape(3, 3)).trace().item() == 12.0
    with pytest.raises(RuntimeError):
        v.trace()
    assert x.allclose(T(X + 1e-9))
    close(x.where(T(X > 0), T(Y)), np.where(X > 0, X, Y))
    u = T(np.array([3, 1, 3, 2, 1], np.int64)).unique()
    assert N(u).tolist() == [1, 2, 3]


# ---- comparisons and division
@pytest.mark.parametrize("name,op", [("eq", np.equal), ("ne", np.not_equal), ("lt", np.less), ("le", np.less_equal),
                                     ("gt", np.greater), ("ge", np.greater_equal), ("not_equal", np.not_equal),
                                     ("greater", np.greater), ("greater_equal", np.greater_equal), ("less", np.less),
                                     ("less_equal", np.less_equal)])
def test_comparisons(name, op):
    assert N(getattr(T(X), name)(T(Y))).tolist() == op(X, Y).tolist()
    assert N(getattr(T(X), name)(0.0)).tolist() == op(X, 0.0).tolist()


def test_equal():
    assert T(X).equal(T(X.copy()))
    assert not T(X).equal(T(Y))
    assert not T(X).equal(T(X[:2]))


def test_division_family():
    i = T(np.array([7, -7, 6], np.int64))
    j = T(np.array([2, 2, 4], np.int64))
    assert N(i.true_divide(j)).tolist() == [3.5, -3.5, 1.5]
    assert N(i.divide(j)).tolist() == [3.5, -3.5, 1.5]
    assert N(i.divide(j, rounding_mode="trunc")).tolist() == [3, -3, 1]
    assert N(i.divide(j, rounding_mode="floor")).tolist() == [3, -4, 1]
    assert N(i.floor_divide(j)).tolist() == [3, -4, 1]
    close(T(X).floor_divide(T(POS)), np.floor(X / POS))
    close(T(X).divide(T(POS), rounding_mode="trunc"), np.trunc(X / POS))
    with pytest.raises(ValueError):
        T(X).divide(T(POS), rounding_mode="round")
    close(T(POS).reciprocal(), 1 / POS)
    assert N(j.reciprocal()).tolist() == [0.5, 0.5, 0.25]
    close(T(X).multiply(T(Y)), X * Y)
    close(T(X).subtract(T(Y), alpha=2.0), X - 2 * Y)
    close(T(X).absolute(), np.abs(X))


def test_dist():
    d = X - Y
    assert math.isclose(T(X).dist(T(Y)).item(), np.linalg.norm(d.ravel()), rel_tol=1e-5)
    assert math.isclose(T(X).dist(T(Y), 1).item(), np.abs(d).sum(), rel_tol=1e-5)
    assert math.isclose(T(X).dist(T(Y), float("inf")).item(), np.abs(d).max(), rel_tol=1e-6)
    assert T(X).dist(T(Y), 0).item() == np.count_nonzero(d)


def test_shape_helpers():
    x = T(X)
    assert tuple(x.reshape_as(T(np.zeros((4, 3), np.float32))).shape) == (4, 3)
    assert x.is_same_size(T(Y)) and not x.is_same_size(T(X[:1]))
    assert x.element_size() == 4 and T(np.zeros(2, np.int64)).element_size() == 8
    close(x.sum_to_size(1, 4), X.sum(0, keepdims=True))
    close(x.sum_to_size(4), X.sum(0))
    close(x.sum_to_size(3, 1), X.sum(1, keepdims=True))
    with pytest.raises(RuntimeError):
        x.sum_to_size(2, 4)
    parts = x.split_with_sizes([1, 3], 1)
    assert [tuple(p.shape) for p in parts] == [(3, 1), (3, 3)]
    with pytest.raises(RuntimeError):
        x.split_with_sizes([1, 1], 1)
    f = x.new_full((2, 2), 7.0)
    assert f.dtype == "f32" and N(f).tolist() == [[7, 7], [7, 7]]
    assert T(np.array([2.0], np.float32)).is_nonzero()
    with pytest.raises(RuntimeError):
        x.is_nonzero()


# ---- in-place
UNARY = {
    "exp_": np.exp, "log_": np.log, "log2_": np.log2, "log10_": np.log10, "log1p_": np.log1p, "expm1_": np.expm1,
    "sqrt_": np.sqrt, "rsqrt_": lambda a: 1 / np.sqrt(a), "abs_": np.abs, "absolute_": np.abs, "sign_": np.sign,
    "sgn_": np.sign, "floor_": np.floor, "ceil_": np.ceil, "trunc_": np.trunc, "fix_": np.trunc,
    "frac_": lambda a: a - np.trunc(a), "square_": np.square, "sigmoid_": lambda a: 1 / (1 + np.exp(-a)),
    "tanh_": np.tanh, "relu_": lambda a: np.maximum(a, 0), "sin_": np.sin, "cos_": np.cos, "tan_": np.tan,
    "atan_": np.arctan, "sinh_": np.sinh, "cosh_": np.cosh, "neg_": np.negative, "negative_": np.negative,
    "reciprocal_": lambda a: 1 / a, "round_": np.round,
}


@pytest.mark.parametrize("name", sorted(UNARY))
def test_unary_inplace(name):
    data = POS if name in ("log_", "log1p_", "log2_", "log10_", "sqrt_", "rsqrt_", "reciprocal_") else X
    x = T(data.copy())
    alias = x  # every handle sees the update
    out = getattr(x, name)()
    assert out is x
    close(alias, UNARY[name](data.astype(np.float64)), 1e-5)


def test_asin_acos_erf_inplace():
    a = (X / 4).astype(np.float32)
    for name, ref in (("asin_", np.arcsin), ("acos_", np.arccos)):
        x = T(a.copy())
        getattr(x, name)()
        close(x, ref(a))
    x = T(a.copy())
    x.erf_()
    close(x, [[math.erf(v) for v in row] for row in a])


BINARY = {
    "pow_": (lambda x, y: x.pow_(2.0), lambda a, b: a ** 2),
    "clamp_min_": (lambda x, y: x.clamp_min_(0.1), lambda a, b: np.maximum(a, 0.1)),
    "clamp_max_": (lambda x, y: x.clamp_max_(0.1), lambda a, b: np.minimum(a, 0.1)),
    "lerp_": (lambda x, y: x.lerp_(y, 0.25), lambda a, b: a + 0.25 * (b - a)),
    "addcmul_": (lambda x, y: x.addcmul_(y, y, value=0.5), lambda a, b: a + 0.5 * b * b),
    "addcdiv_": (lambda x, y: x.addcdiv_(y, y + 3, value=2.0), lambda a, b: a + 2 * b / (b + 3)),
    "remainder_": (lambda x, y: x.remainder_(0.7), lambda a, b: np.mod(a, 0.7)),
    "fmod_": (lambda x, y: x.fmod_(0.7), lambda a, b: np.fmod(a, 0.7)),
    "atan2_": (lambda x, y: x.atan2_(y), np.arctan2),
    "hypot_": (lambda x, y: x.hypot_(y), np.hypot),
    "copysign_": (lambda x, y: x.copysign_(y), np.copysign),
    "multiply_": (lambda x, y: x.multiply_(y), np.multiply),
    "subtract_": (lambda x, y: x.subtract_(y, alpha=3.0), lambda a, b: a - 3 * b),
    "true_divide_": (lambda x, y: x.true_divide_(y + 5), lambda a, b: a / (b + 5)),
    "divide_": (lambda x, y: x.divide_(y + 5, rounding_mode="floor"), lambda a, b: np.floor(a / (b + 5))),
    "floor_divide_": (lambda x, y: x.floor_divide_(y + 5), lambda a, b: np.floor(a / (b + 5))),
    "tril_": (lambda x, y: x.tril_(), np.tril and (lambda a, b: np.tril(a))),
    "triu_": (lambda x, y: x.triu_(1), lambda a, b: np.triu(a, 1)),
    "eq_": (lambda x, y: x.eq_(y), lambda a, b: (a == b).astype(np.float32)),
    "gt_": (lambda x, y: x.gt_(y), lambda a, b: (a > b).astype(np.float32)),
    "le_": (lambda x, y: x.le_(0.0), lambda a, b: (a <= 0).astype(np.float32)),
}


@pytest.mark.parametrize("name", sorted(BINARY))
def test_binary_inplace(name):
    call, ref = BINARY[name]
    x, y = T(X.copy()), T(Y.copy())
    assert call(x, y) is x
    close(x, ref(X.astype(np.float64), Y.astype(np.float64)), 1e-5)


def test_integer_and_bool_inplace():
    a = T(np.array([12, 5, -3], np.int64))
    a.bitwise_and_(T(np.array([10, 3, 6], np.int64)))
    assert N(a).tolist() == [8, 1, 4]
    a.bitwise_not_()
    assert N(a).tolist() == [-9, -2, -5]
    b = T(np.array([True, False, True]))
    b.logical_and_(T(np.array([True, True, False])))
    assert N(b).tolist() == [True, False, False]
    b.logical_not_()
    assert N(b).tolist() == [False, True, True]


def test_inplace_refuses_shape_change_and_float_into_int():
    x = T(X.copy())
    with pytest.raises(RuntimeError):
        x.lerp_(T(np.ones((2, 3, 4), np.float32)), 0.5)  # broadcasting would change the shape
    i = T(np.array([1, 2, 3], np.int64))
    with pytest.raises(RuntimeError):
        i.sqrt_()
    with pytest.raises(RuntimeError):
        i.true_divide_(T(np.array([2, 2, 2], np.int64)))


def test_inplace_refuses_tensors_that_require_grad_unless_no_grad():
    w = T(X.copy()).requires_grad_(True)
    with pytest.raises(RuntimeError):
        w.exp_()
    with ox.no_grad():
        w.clamp_min_(0.0)
    close(w.detach(), np.maximum(X, 0))


def test_inplace_refuses_non_contiguous_views():
    x = T(X.copy())
    with pytest.raises(RuntimeError):
        x.t().exp_()
