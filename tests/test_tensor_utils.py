"""Tests of oxideon.tensor_utils: every function against NumPy (or the definition) on small inputs."""
import itertools
import math

import numpy as np
import pytest

import oxideon as ox

rng = np.random.default_rng(0)


def T(a):
    return ox.from_numpy(np.ascontiguousarray(a))


def N(t):
    return t.to_numpy()


def close(t, ref, tol=1e-5):
    np.testing.assert_allclose(N(t), ref, rtol=tol, atol=tol)


X = rng.standard_normal((2, 3, 4)).astype(np.float32)


# ---- shape and layout
def test_ravel():
    close(ox.ravel(T(X)), X.ravel())


@pytest.mark.parametrize("src,dst", [(0, 2), (2, 0), ((0, 1), (1, 2)), (-1, 0)])
def test_movedim_and_moveaxis(src, dst):
    ref = np.moveaxis(X, src, dst)
    close(ox.movedim(T(X), src, dst), ref)
    close(ox.moveaxis(T(X), src, dst), ref)


def test_swapaxes_and_swapdims():
    close(ox.swapaxes(T(X), 0, 2), np.swapaxes(X, 0, 2))
    close(ox.swapdims(T(X), 1, 2), np.swapaxes(X, 1, 2))


@pytest.mark.parametrize("spec", [3, 5, [1, 3], [3, 1], [-1]])
def test_tensor_split_matches_array_split(spec):
    parts = ox.tensor_split(T(X), spec, dim=2)
    ref = np.array_split(X, spec, axis=2) if isinstance(spec, int) else np.split(X, spec, axis=2)
    assert len(parts) == len(ref)
    for p, r in zip(parts, ref):
        assert tuple(p.shape) == r.shape
        close(p, r)


def test_hsplit_vsplit_dsplit():
    for f, ax in ((ox.vsplit, 0), (ox.hsplit, 1), (ox.dsplit, 2)):
        for p, r in zip(f(T(X), 2), np.array_split(X, 2, axis=ax)):
            close(p, r)
    assert [tuple(p.shape) for p in ox.hsplit(T(np.arange(6, dtype=np.float32)), 3)] == [(2,)] * 3
    with pytest.raises(ValueError):
        ox.vsplit(T(np.arange(3, dtype=np.float32)), 3)
    with pytest.raises(ValueError):
        ox.dsplit(T(np.ones((2, 2), np.float32)), 2)


def test_broadcast_helpers():
    assert ox.broadcast_shapes((2, 1), (3,)) == (2, 3)
    assert ox.broadcast_shapes((1, 5), (4, 1), (4, 5)) == (4, 5)
    a, b = ox.broadcast_tensors(T(np.ones((2, 1), np.float32)), T(np.arange(3, dtype=np.float32)))
    assert tuple(a.shape) == tuple(b.shape) == (2, 3)
    assert tuple(ox.broadcast_to(T(np.ones(3, np.float32)), (4, 3)).shape) == (4, 3)
    with pytest.raises(Exception):
        ox.broadcast_shapes((2,), (3,))


@pytest.mark.parametrize("r,rep", [(2, False), (3, False), (2, True), (1, False)])
def test_combinations(r, rep):
    v = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    pick = itertools.combinations_with_replacement if rep else itertools.combinations
    ref = np.array(list(pick(v.tolist(), r)), dtype=np.float32)
    close(ox.combinations(T(v), r, rep), ref)
    assert tuple(ox.combinations(T(v), 5).shape)[0] == 0


def test_msort():
    a = rng.standard_normal((5, 3)).astype(np.float32)
    close(ox.msort(T(a)), np.sort(a, axis=0))


# ---- elementwise
@pytest.mark.parametrize("w", [0.0, 0.25, 0.5, 0.75, 1.0])
def test_lerp_scalar_weight(w):
    a, b = rng.standard_normal(7).astype(np.float32), rng.standard_normal(7).astype(np.float32)
    close(ox.lerp(T(a), T(b), w), a + w * (b - a))
    assert np.array_equal(N(ox.lerp(T(a), T(b), 1.0)), b)  # exact at the end point


def test_lerp_tensor_weight():
    a, b, w = (rng.standard_normal(9).astype(np.float32) for _ in range(3))
    w = np.clip(w, 0, 1)
    close(ox.lerp(T(a), T(b), T(w)), a + w * (b - a))


def test_float_power_is_float64():
    out = ox.float_power(T(np.array([2, 3], dtype=np.float32)), 2)
    assert out.dtype == "f64"
    close(out, [4.0, 9.0])
    close(ox.float_power(2, T(np.array([1.0, 3.0], dtype=np.float32))), [2.0, 8.0])


def test_signbit():
    v = np.array([-1.0, 0.0, -0.0, 2.0, -np.inf], dtype=np.float32)
    assert N(ox.signbit(T(v))).tolist() == [True, False, True, False, True]


def test_gcd_lcm():
    a, b = np.array([12, -18, 0, 7], dtype=np.int64), np.array([18, 12, 5, 0], dtype=np.int64)
    assert N(ox.gcd(T(a), T(b))).tolist() == np.gcd(a, b).tolist()
    assert N(ox.lcm(T(a), T(b))).tolist() == np.lcm(a, b).tolist()


def test_ldexp_and_frexp_roundtrip():
    x = rng.standard_normal(20).astype(np.float32) * 100
    m, e = ox.frexp(T(x))
    assert N(e).dtype == np.int32
    assert np.all((np.abs(N(m)) >= 0.5) & (np.abs(N(m)) < 1))
    assert np.array_equal(N(ox.ldexp(m, e)), x)
    close(ox.ldexp(T(np.array([1.0, 3.0], dtype=np.float32)), 3), [8.0, 24.0])


def test_nextafter():
    a = np.array([1.0, -1.0, 0.0], dtype=np.float32)
    b = np.array([2.0, -2.0, 1.0], dtype=np.float32)
    assert np.array_equal(N(ox.nextafter(T(a), T(b))), np.nextafter(a, b))


def test_unary_aliases():
    v = np.array([-2.7, -0.5, 0.0, 0.5, 2.7], dtype=np.float32)
    close(ox.sgn(T(v)), np.sign(v))
    close(ox.positive(T(v)), v)
    close(ox.negative(T(v)), -v)
    close(ox.trunc(T(v)), np.trunc(v))
    close(ox.fix(T(v)), np.trunc(v))
    close(ox.frac(T(v)), v - np.trunc(v))
    close(ox.square(T(v)), v * v)


def test_addcmul_addcdiv():
    a, b, c = (rng.standard_normal(6).astype(np.float32) + 3 for _ in range(3))
    close(ox.addcmul(T(a), T(b), T(c), value=0.5), a + 0.5 * b * c)
    close(ox.addcdiv(T(a), T(b), T(c), value=2.0), a + 2.0 * b / c)


def test_lgamma_and_mvlgamma():
    x = np.array([0.5, 1.0, 2.5, 7.0], dtype=np.float32)
    close(ox.lgamma(T(x)), [math.lgamma(v) for v in x])
    x3 = np.array([2.0, 3.5, 6.0], dtype=np.float32)
    ref = [3 * 2 / 4 * math.log(math.pi) + sum(math.lgamma(v - i / 2) for i in range(3)) for v in x3]
    close(ox.mvlgamma(T(x3), 3), ref)
    close(ox.mvlgamma(T(x3), 1), [math.lgamma(v) for v in x3])
    with pytest.raises(ValueError):
        ox.mvlgamma(T(np.array([0.5], dtype=np.float32)), 3)


def test_logspace_and_vander():
    close(ox.logspace(0, 3, 4), [1, 10, 100, 1000], 1e-4)
    close(ox.logspace(0, 3, 4, base=2.0), [1, 2, 4, 8])
    v = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    close(ox.vander(T(v)), np.vander(v))
    close(ox.vander(T(v), 4, increasing=True), np.vander(v, 4, increasing=True))


# ---- predicates
def test_float_predicates():
    v = np.array([1.0, np.nan, np.inf, -np.inf], dtype=np.float32)
    assert N(ox.isnan(T(v))).tolist() == [False, True, False, False]
    assert N(ox.isinf(T(v))).tolist() == [False, False, True, True]
    assert N(ox.isfinite(T(v))).tolist() == [True, False, False, False]
    assert N(ox.isposinf(T(v))).tolist() == [False, False, True, False]
    assert N(ox.isneginf(T(v))).tolist() == [False, False, False, True]
    assert N(ox.isreal(T(v))).tolist() == [True] * 4


def test_is_tensor_numel():
    assert ox.is_tensor(T(X)) and not ox.is_tensor(X) and not ox.is_tensor(3)
    assert ox.numel(T(X)) == 24


def test_result_type():
    f32, f64, i64, i32 = (T(np.zeros(1, dtype=d)) for d in (np.float32, np.float64, np.int64, np.int32))
    assert ox.result_type(f32, f64) == "f64"
    assert ox.result_type(i32, i64) == "i64"
    assert ox.result_type(i64, f32) == "f32"  # like PyTorch: the float category wins
    assert ox.result_type(i32, f64) == "f64"
    assert ox.result_type(f32, 2.0) == "f32"
    assert ox.result_type(i32, 2.5) == "f32"
    assert ox.result_type(i32, 2) == "i32"


# ---- reductions
@pytest.mark.parametrize("dim,keep", [(None, False), (1, False), (1, True), (-1, False), ((0, 2), False), ((0, 2), True)])
def test_logsumexp(dim, keep):
    ref = np.log(np.sum(np.exp(X.astype(np.float64)), axis=dim, keepdims=keep))
    close(ox.logsumexp(T(X), dim, keep), ref, 1e-5)


def test_logsumexp_does_not_overflow_and_handles_minus_inf():
    big = np.array([[1000.0, 1000.0], [-np.inf, -np.inf]], dtype=np.float32)
    out = N(ox.logsumexp(T(big), 1))
    assert abs(out[0] - (1000 + math.log(2))) < 1e-3
    assert out[1] == -np.inf


def test_logsumexp_is_differentiable():
    x = T(X[0]).detach().requires_grad_(True)
    ox.logsumexp(x, 1).sum().backward()
    ref = np.exp(X[0] - np.log(np.sum(np.exp(X[0]), axis=1, keepdims=True)))
    close(x.grad, ref)


def test_cumulative_logsumexp():
    ref = np.log(np.cumsum(np.exp(X.astype(np.float64)), axis=1))
    close(ox.cumulative_logsumexp(T(X), 1), ref, 1e-5)


@pytest.mark.parametrize("dim", [None, 1, (0, 2)])
@pytest.mark.parametrize("unbiased", [True, False])
def test_var_mean_std_mean(dim, unbiased):
    ddof = 1 if unbiased else 0
    v, m = ox.var_mean(T(X), dim, unbiased)
    close(v, np.var(X.astype(np.float64), axis=dim, ddof=ddof))
    close(m, np.mean(X.astype(np.float64), axis=dim))
    s, m2 = ox.std_mean(T(X), dim, unbiased)
    close(s, np.std(X.astype(np.float64), axis=dim, ddof=ddof))
    close(m2, m)


def test_var_mean_keepdim_and_correction():
    v, m = ox.var_mean(T(X), 1, keepdim=True, correction=0)
    assert tuple(v.shape) == tuple(m.shape) == (2, 1, 4)
    v2, _ = ox.var_mean(T(X), (0, 2), keepdim=True)
    assert tuple(v2.shape) == (1, 3, 1)


def test_aminmax():
    r = ox.aminmax(T(X))
    assert r.min.item() == X.min() and r.max.item() == X.max()
    r = ox.aminmax(T(X), dim=1)
    close(r.min, X.min(axis=1))
    close(r.max, X.max(axis=1))
    assert tuple(ox.aminmax(T(X), dim=1, keepdim=True).max.shape) == (2, 1, 4)


def test_count_nonzero():
    a = np.array([[0, 1, 2], [0, 0, 3]], dtype=np.float32)
    assert ox.count_nonzero(T(a)).item() == 3
    assert N(ox.count_nonzero(T(a), 1)).tolist() == [2, 1]
    assert N(ox.count_nonzero(T(a), 0)).tolist() == [0, 1, 2]


def test_renorm():
    a = rng.standard_normal((4, 5)).astype(np.float32) * 3
    out = N(ox.renorm(T(a), 2, 0, 1.0))
    norms = np.linalg.norm(a, axis=1)
    for i in range(4):
        expect = a[i] if norms[i] <= 1 else a[i] / (norms[i] + 1e-7)
        np.testing.assert_allclose(out[i], expect, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("reduce", ["sum", "prod", "mean", "amax", "amin"])
@pytest.mark.parametrize("include_self", [True, False])
def test_scatter_reduce(reduce, include_self):
    x = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    src = np.array([5.0, 6.0, 7.0, 8.0, 9.0], dtype=np.float32)
    idx = np.array([0, 1, 0, 1, 0], dtype=np.int64)
    out = N(ox.scatter_reduce(T(x), 0, T(idx), T(src), reduce, include_self=include_self))
    ref = x.copy()
    for t in (0, 1):
        vals = list(src[idx == t]) + ([x[t]] if include_self else [])
        ref[t] = {"sum": sum, "prod": np.prod, "mean": np.mean, "amax": max, "amin": min}[reduce](vals)
    np.testing.assert_allclose(out, ref, rtol=1e-6)
    with pytest.raises(ValueError):
        ox.scatter_reduce(T(x), 0, T(idx), T(src), "median")


def test_histogramdd():
    pts = rng.random((200, 2)).astype(np.float32)
    h, edges = ox.histogramdd(pts, 4, range=[0, 1, 0, 1])
    ref, _ = np.histogramdd(pts, bins=4, range=[(0, 1), (0, 1)])
    close(h, ref)
    assert len(edges) == 2 and all(tuple(e.shape) == (5,) for e in edges)
    assert h.sum().item() == 200


# ---- linalg
A = np.array([[4.0, 1.0, 0.5], [1.0, 3.0, 0.25], [0.5, 0.25, 2.0]], dtype=np.float32)


def test_linalg_top_level():
    close(ox.det(T(A)), np.linalg.det(A), 1e-4)
    close(ox.inverse(T(A)), np.linalg.inv(A), 1e-4)
    s = ox.slogdet(T(A))
    sign, logdet = np.linalg.slogdet(A)
    close(s[0], sign)
    close(s[1], logdet, 1e-4)
    close(ox.matrix_power(T(A), 3), np.linalg.matrix_power(A, 3), 1e-3)
    close(ox.matrix_power(T(A), -1), np.linalg.inv(A), 1e-4)
    close(ox.pinverse(T(A)), np.linalg.pinv(A), 1e-3)
    close(ox.matrix_exp(T(np.zeros((2, 2), np.float32))), np.eye(2))


def test_chain_matmul_and_ger():
    a, b, c = (rng.standard_normal((3, 3)).astype(np.float32) for _ in range(3))
    close(ox.chain_matmul(T(a), T(b), T(c)), a @ b @ c, 1e-4)
    close(ox.chain_matmul(T(a)), a)
    u, v = np.arange(1, 4, dtype=np.float32), np.arange(1, 6, dtype=np.float32)
    close(ox.ger(T(u), T(v)), np.outer(u, v))
    with pytest.raises(ValueError):
        ox.chain_matmul()


def test_lu_and_cholesky_solve():
    p, l, u = ox.lu(T(A))
    close(p.matmul(l).matmul(u), A, 1e-4)
    chol = np.linalg.cholesky(A.astype(np.float64)).astype(np.float32)
    b = rng.standard_normal((3, 2)).astype(np.float32)
    close(ox.cholesky_solve(T(b), T(chol)), np.linalg.solve(A, b), 1e-3)


# ---- logical and bitwise
def test_logical_ops():
    a, b = np.array([True, True, False, False]), np.array([True, False, True, False])
    assert N(ox.logical_and(T(a), T(b))).tolist() == (a & b).tolist()
    assert N(ox.logical_or(T(a), T(b))).tolist() == (a | b).tolist()
    assert N(ox.logical_xor(T(a), T(b))).tolist() == (a ^ b).tolist()
    assert N(ox.logical_not(T(a))).tolist() == (~a).tolist()


def test_bitwise_ops():
    a, b = np.array([0b1100, 5, -3], dtype=np.int64), np.array([0b1010, 3, 6], dtype=np.int64)
    assert N(ox.bitwise_and(T(a), T(b))).tolist() == (a & b).tolist()
    assert N(ox.bitwise_or(T(a), T(b))).tolist() == (a | b).tolist()
    assert N(ox.bitwise_xor(T(a), T(b))).tolist() == (a ^ b).tolist()
    assert N(ox.bitwise_not(T(a))).tolist() == (~a).tolist()


def test_every_export_is_reachable_from_the_package():
    from oxideon import tensor_utils
    missing = [n for n in tensor_utils.__all__ if getattr(ox, n, None) is not getattr(tensor_utils, n)]
    assert not missing, missing
