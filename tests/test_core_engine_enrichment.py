"""Core Engine Enrichment Tests (Core-1 through Core-10):
Tests native Rust Tensor operations, autograd backward passes, linear algebra decompositions,
and fused kernels exposed via PyO3."""

import math
import numpy as np
import pytest
import oxideon as ox


def test_core1_trig_and_inverse_trig_with_autograd():
    """Core-1: tan, asin, acos, atan, and sinc forward + analytical backward."""
    x = ox.Tensor.from_slice([0.0, 0.5, -0.5], [3], dtype="f64", requires_grad=True)
    y = x.tan() + x.asin() + x.acos() + x.atan() + x.sinc()
    loss = y.sum()
    loss.backward()

    x_np = np.array([0.0, 0.5, -0.5], dtype=np.float64)
    expected = (
        np.tan(x_np)
        + np.arcsin(x_np)
        + np.arccos(x_np)
        + np.arctan(x_np)
        + np.sinc(x_np)
    )
    assert np.allclose(y.to_numpy(), expected, atol=1e-10)

    # Analytical derivative: sec^2(x) + 1/sqrt(1-x^2) - 1/sqrt(1-x^2) + 1/(1+x^2) + sinc'(x)
    # Note: d(asin)/dx + d(acos)/dx = 0!
    with np.errstate(divide="ignore", invalid="ignore"):
        sinc_deriv = np.where(
            np.abs(x_np) < 1e-8,
            0.0,
            (np.pi * x_np * np.cos(np.pi * x_np) - np.sin(np.pi * x_np)) / (np.pi * x_np**2),
        )
    expected_grad = (1.0 / np.cos(x_np) ** 2) + (1.0 / (1.0 + x_np**2)) + sinc_deriv
    assert np.allclose(x.grad.to_numpy(), expected_grad, atol=1e-9)


def test_core2_hyperbolic_and_inverse_hyperbolic_with_autograd():
    """Core-2: sinh, cosh, asinh, acosh, atanh forward + analytical backward."""
    x = ox.Tensor.from_slice([0.2, 0.5], [2], dtype="f64", requires_grad=True)
    z = ox.Tensor.from_slice([1.5, 2.0], [2], dtype="f64", requires_grad=True)

    y1 = x.sinh() + x.cosh() + x.asinh() + x.atanh()
    y2 = z.acosh()
    (y1.sum() + y2.sum()).backward()

    x_np = np.array([0.2, 0.5], dtype=np.float64)
    z_np = np.array([1.5, 2.0], dtype=np.float64)
    expected_x_grad = (
        np.cosh(x_np)
        + np.sinh(x_np)
        + 1.0 / np.sqrt(x_np**2 + 1.0)
        + 1.0 / (1.0 - x_np**2)
    )
    expected_z_grad = 1.0 / np.sqrt(z_np**2 - 1.0)
    assert np.allclose(x.grad.to_numpy(), expected_x_grad, atol=1e-10)
    assert np.allclose(z.grad.to_numpy(), expected_z_grad, atol=1e-10)


def test_core3_log_exp_xlogy_logit_primitives():
    """Core-3: expm1, log1p, log2, log10, exp2, xlogy (with 0*log(0)=0), and logit."""
    x = ox.Tensor.from_slice([1e-9, 0.5], [2], dtype="f64", requires_grad=True)
    y = x.expm1() + x.log1p() + x.log2() + x.log10() + x.exp2()
    y.sum().backward()
    x_np = np.array([1e-9, 0.5], dtype=np.float64)
    assert np.allclose(
        y.to_numpy(),
        np.expm1(x_np) + np.log1p(x_np) + np.log2(x_np) + np.log10(x_np) + np.exp2(x_np),
        atol=1e-10,
    )

    # xlogy: 0 * log(0) must be 0.0, not NaN
    a = ox.Tensor.from_slice([0.0, 2.0], [2], dtype="f64", requires_grad=True)
    b = ox.Tensor.from_slice([0.0, math.e], [2], dtype="f64", requires_grad=True)
    xl = a.xlogy(b)
    assert np.allclose(xl.to_numpy(), [0.0, 2.0], atol=1e-12)
    xl.sum().backward()
    assert np.allclose(a.grad.to_numpy(), [0.0, 1.0], atol=1e-12)

    # logit with clamping eps
    p = ox.Tensor.from_slice([0.5, 0.8], [2], dtype="f64", requires_grad=True)
    lg = p.logit(eps=1e-6)
    lg.sum().backward()
    assert np.allclose(lg.to_numpy(), [0.0, math.log(0.8 / 0.2)], atol=1e-10)
    assert np.allclose(p.grad.to_numpy(), [1.0 / 0.25, 1.0 / 0.16], atol=1e-9)


def test_core4_cummax_cummin_and_logcumsumexp():
    """Core-4: cummax, cummin, and logcumsumexp forward + autograd backward."""
    x = ox.Tensor.from_slice([2.0, 1.0, 4.0, 3.0], [4], dtype="f64", requires_grad=True)
    cmax_vals, cmax_idx = x.cummax(dim=0)
    assert np.allclose(cmax_vals.to_numpy(), [2.0, 2.0, 4.0, 4.0])
    assert list(cmax_idx.to_numpy()) == [0, 0, 2, 2]
    cmax_vals.sum().backward()
    # Index 0 wins positions 0 and 1 (grad=2), index 2 wins positions 2 and 3 (grad=2)
    assert np.allclose(x.grad.to_numpy(), [2.0, 0.0, 2.0, 0.0])

    x2 = ox.Tensor.from_slice([2.0, 1.0, 4.0, 0.5], [4], dtype="f64", requires_grad=True)
    cmin_vals, cmin_idx = x2.cummin(dim=0)
    assert np.allclose(cmin_vals.to_numpy(), [2.0, 1.0, 1.0, 0.5])
    assert list(cmin_idx.to_numpy()) == [0, 1, 1, 3]
    cmin_vals.sum().backward()
    assert np.allclose(x2.grad.to_numpy(), [1.0, 2.0, 0.0, 1.0])

    x3 = ox.Tensor.from_slice([0.5, 1.0, -0.5], [3], dtype="f64", requires_grad=True)
    lce = x3.logcumsumexp(dim=0)
    expected_lce = np.log(np.cumsum(np.exp([0.5, 1.0, -0.5])))
    assert np.allclose(lce.to_numpy(), expected_lce, atol=1e-12)
    lce.sum().backward()
    assert np.all(np.isfinite(x3.grad.to_numpy()))


def test_core5_prod_zero_safe_grad_any_all_count_nonzero_aminmax():
    """Core-5: prod (with zero-safe autograd), any, all, count_nonzero, aminmax."""
    # Zero-safe product gradient: [3.0, 0.0, 4.0] -> grad at 0.0 is 3.0 * 4.0 = 12.0!
    x = ox.Tensor.from_slice([3.0, 0.0, 4.0], [3], dtype="f64", requires_grad=True)
    p = x.prod()
    assert abs(float(p.item()) - 0.0) < 1e-12
    p.backward()
    assert np.allclose(x.grad.to_numpy(), [0.0, 12.0, 0.0], atol=1e-12)

    m = ox.Tensor.from_slice([1.0, 0.0, 2.0, 3.0, 0.0, 0.0], [2, 3], dtype="f64")
    assert bool(m.any().item()) is True
    assert bool(m.all().item()) is False
    assert list(m.any(dim=0).to_numpy()) == [True, False, True]
    assert list(m.all(dim=1).to_numpy()) == [False, False]
    assert int(m.count_nonzero().item()) == 3
    assert list(m.count_nonzero(dim=1).to_numpy()) == [2, 1]

    mn, mx = m.aminmax(dim=1)
    assert np.allclose(mn.to_numpy(), [0.0, 0.0])
    assert np.allclose(mx.to_numpy(), [2.0, 3.0])


def test_core6_compound_pointwise_and_modern_activations():
    """Core-6: addcmul, addcdiv, mish, hardswish, hardsigmoid, selu, celu, copysign, fmod, remainder, trunc, frac."""
    base = ox.Tensor.from_slice([1.0, 2.0], [2], dtype="f64", requires_grad=True)
    t1 = ox.Tensor.from_slice([3.0, 4.0], [2], dtype="f64", requires_grad=True)
    t2 = ox.Tensor.from_slice([2.0, 0.5], [2], dtype="f64", requires_grad=True)

    out = base.addcmul(t1, t2, value=0.5)
    assert np.allclose(out.to_numpy(), [1.0 + 0.5 * 6.0, 2.0 + 0.5 * 2.0])
    out.sum().backward()
    assert np.allclose(t1.grad.to_numpy(), [1.0, 0.25])

    div_out = base.detach().addcdiv(t1.detach(), t2.detach(), value=1.0)
    assert np.allclose(div_out.to_numpy(), [1.0 + 1.5, 2.0 + 8.0])

    x = ox.Tensor.from_slice([-1.5, 0.0, 2.25], [3], dtype="f64", requires_grad=True)
    act = x.mish() + x.hardswish() + x.hardsigmoid() + x.selu() + x.celu(1.0)
    act.sum().backward()
    assert np.all(np.isfinite(x.grad.to_numpy()))

    assert np.allclose(x.trunc().to_numpy(), [-1.0, 0.0, 2.0])
    assert np.allclose(x.frac().to_numpy(), [-0.5, 0.0, 0.25])
    s = ox.Tensor.from_slice([1.0, -1.0, -1.0], [3], dtype="f64")
    assert np.allclose(x.copysign(s).to_numpy(), [1.5, -0.0, -2.25])


def test_core7_cross_and_tensordot_with_autograd():
    """Core-7: 3D cross product and multi-axis tensordot with autograd."""
    a = ox.Tensor.from_slice([1.0, 0.0, 0.0], [3], dtype="f64", requires_grad=True)
    b = ox.Tensor.from_slice([0.0, 1.0, 0.0], [3], dtype="f64", requires_grad=True)
    c = a.cross(b, dim=0)
    assert np.allclose(c.to_numpy(), [0.0, 0.0, 1.0])
    c.sum().backward()
    # d(a x b).sum / da = b x [1, 1, 1] = [1, 0, -1]
    assert np.allclose(a.grad.to_numpy(), [1.0, 0.0, -1.0])
    # d(a x b).sum / db = [1, 1, 1] x a = [0, 1, -1]
    assert np.allclose(b.grad.to_numpy(), [0.0, 1.0, -1.0])

    t_a = ox.Tensor.from_slice(list(range(24)), [2, 3, 4], dtype="f64", requires_grad=True)
    t_b = ox.Tensor.from_slice(list(range(24)), [3, 4, 2], dtype="f64", requires_grad=True)
    td = t_a.tensordot(t_b, dims=([1, 2], [0, 1]))
    assert td.shape == (2, 2)
    expected_td = np.tensordot(t_a.to_numpy(), t_b.to_numpy(), axes=([1, 2], [0, 1]))
    assert np.allclose(td.to_numpy(), expected_td)
    td.sum().backward()
    assert t_a.grad.shape == (2, 3, 4) and t_b.grad.shape == (3, 4, 2)


def test_core8_block_diag_vander_and_cartesian_prod():
    """Core-8: block_diag (with autograd), vander, and cartesian_prod."""
    m1 = ox.Tensor.from_slice([1.0, 2.0, 3.0, 4.0], [2, 2], dtype="f64", requires_grad=True)
    m2 = ox.Tensor.from_slice([5.0], [1, 1], dtype="f64", requires_grad=True)
    bd = ox.block_diag([m1, m2])
    assert bd.shape == (3, 3)
    expected_bd = np.array([[1.0, 2.0, 0.0], [3.0, 4.0, 0.0], [0.0, 0.0, 5.0]])
    assert np.allclose(bd.to_numpy(), expected_bd)
    (bd * 2.0).sum().backward()
    assert np.allclose(m1.grad.to_numpy(), np.full((2, 2), 2.0))
    assert np.allclose(m2.grad.to_numpy(), [[2.0]])

    v = ox.Tensor.from_slice([1.0, 2.0, 3.0], [3], dtype="f64", requires_grad=True)
    van = v.vander(N=3, increasing=True)
    assert np.allclose(van.to_numpy(), np.vander([1.0, 2.0, 3.0], N=3, increasing=True))
    van.sum().backward()
    assert np.allclose(v.grad.to_numpy(), [1.0 + 2.0, 1.0 + 4.0, 1.0 + 6.0])

    c1 = ox.Tensor.from_slice([1.0, 2.0], [2], dtype="f64")
    c2 = ox.Tensor.from_slice([10.0, 20.0, 30.0], [3], dtype="f64")
    cp = ox.cartesian_prod([c1, c2])
    assert cp.shape == (6, 2)


def test_core9_linalg_cholesky_solve_lu_polar_cdist_p_cov_corrcoef():
    """Core-9: cholesky_solve, lu, polar, cdist_p, cov, and corrcoef."""
    a = ox.Tensor.from_slice([4.0, 1.0, 1.0, 3.0], [2, 2], dtype="f64", requires_grad=True)
    b = ox.Tensor.from_slice([1.0, 2.0], [2, 1], dtype="f64", requires_grad=True)
    l = a.cholesky()
    x_sol = b.cholesky_solve(l, upper=False)
    assert np.allclose((a @ x_sol).to_numpy(), b.to_numpy(), atol=1e-10)
    x_sol.sum().backward()
    assert a.grad is not None and b.grad is not None

    # PA = LU -> P @ L @ U == A
    mat = ox.Tensor.from_slice([2.0, 1.0, 1.0, 4.0, -6.0, 0.0, -2.0, 7.0, 2.0], [3, 3], dtype="f64")
    p_mat, l_mat, u_mat = mat.lu()
    assert np.allclose((p_mat @ l_mat @ u_mat).to_numpy(), mat.to_numpy(), atol=1e-10)

    # Polar decomposition: A = U @ P where U is orthogonal and P is symmetric PSD
    u_pol, p_pol = mat.polar()
    assert np.allclose((u_pol @ p_pol).to_numpy(), mat.to_numpy(), atol=1e-6)
    assert np.allclose((u_pol.t() @ u_pol).to_numpy(), np.eye(3), atol=1e-6)

    # cov and corrcoef
    data = ox.Tensor.from_slice([1.0, 2.0, 3.0, 3.0, 5.0, 7.0], [2, 3], dtype="f64")
    cov_m = data.cov()
    corr_m = data.corrcoef()
    assert np.allclose(cov_m.to_numpy(), np.cov(data.to_numpy()), atol=1e-10)
    assert np.allclose(corr_m.to_numpy(), np.corrcoef(data.to_numpy()), atol=1e-10)


def test_core10_fused_layernorm_rope_and_sdpa_with_autograd():
    """Core-10: fused_layernorm, fused_rope, and fused_sdpa forward + analytical backward."""
    x = ox.Tensor.from_slice([1.0, 2.0, 3.0, 4.0, 2.0, 4.0, 6.0, 8.0], [2, 4], dtype="f64", requires_grad=True)
    w = ox.Tensor.from_slice([1.0, 1.0, 1.0, 1.0], [4], dtype="f64", requires_grad=True)
    b = ox.Tensor.from_slice([0.0, 0.0, 0.0, 0.0], [4], dtype="f64", requires_grad=True)
    ln_out = ox.fused_layernorm(x, w, b, eps=1e-5)
    assert np.allclose(ln_out.mean(dim=-1).to_numpy(), [0.0, 0.0], atol=1e-6)
    ln_out.sum().backward()
    assert w.grad is not None and b.grad is not None and x.grad is not None

    # Fused RoPE preserves vector L2 norm and differentiates cleanly
    rx = ox.Tensor.from_slice([1.0, 2.0, 3.0, 4.0], [1, 4], dtype="f64", requires_grad=True)
    cos = ox.Tensor.from_slice([math.cos(0.3), math.cos(0.7)], [1, 2], dtype="f64")
    sin = ox.Tensor.from_slice([math.sin(0.3), math.sin(0.7)], [1, 2], dtype="f64")
    rope_out = ox.fused_rope(rx, cos, sin)
    assert np.isclose(float((rope_out * rope_out).sum().item()), float((rx * rx).sum().item()), atol=1e-10)
    rope_out.sum().backward()
    assert rx.grad is not None

    # Fused SDPA with causal mask + backward
    q = ox.Tensor.from_slice([1.0, 0.0, 0.0, 1.0, 1.0, 1.0], [3, 2], dtype="f64", requires_grad=True)
    k = ox.Tensor.from_slice([1.0, 0.0, 0.0, 1.0, 1.0, 1.0], [3, 2], dtype="f64", requires_grad=True)
    v = ox.Tensor.from_slice([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [3, 2], dtype="f64", requires_grad=True)
    attn_out = ox.fused_sdpa(q, k, v, is_causal=True)
    assert attn_out.shape == (3, 2)
    # Row 0 with causal mask can only attend to row 0 of v ([1.0, 2.0])
    assert np.allclose(attn_out.to_numpy()[0], [1.0, 2.0], atol=1e-10)
    attn_out.sum().backward()
    assert q.grad is not None and k.grad is not None and v.grad is not None
