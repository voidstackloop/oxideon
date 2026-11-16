import math
import numpy as np
import pytest
import oxideon as ox


def test_xlogy_xlog1py_log1pexp():
    # xlogy(0, 0) == 0, xlogy(0, -5) == 0, xlogy(2, 3) == 2 * ln(3)
    x = ox.tensor([0.0, 0.0, 2.0, 3.0])
    y = ox.tensor([0.0, -5.0, 3.0, math.e])
    res = ox.special.xlogy(x, y)
    np_res = res.to_numpy()
    assert np_res[0] == 0.0
    assert np_res[1] == 0.0
    assert np.isclose(np_res[2], 2.0 * math.log(3.0))
    assert np.isclose(np_res[3], 3.0)

    # Tensor method
    res_m = x.xlogy(y)
    assert np.allclose(res_m.to_numpy(), np_res)

    # xlog1py
    res1p = ox.special.xlog1py(x, y)
    np1p = res1p.to_numpy()
    assert np1p[0] == 0.0
    assert np1p[1] == 0.0
    assert np.isclose(np1p[2], 2.0 * math.log(4.0))

    # Tensor method
    assert np.allclose(x.xlog1py(y).to_numpy(), np1p)

    # log1pexp
    z = ox.tensor([-10.0, 0.0, 10.0, 50.0])
    res_exp = ox.special.log1pexp(z)
    np_exp = res_exp.to_numpy()
    assert np.isclose(np_exp[0], math.log1p(math.exp(-10.0)))
    assert np.isclose(np_exp[1], math.log(2.0))
    assert np.isclose(np_exp[2], math.log1p(math.exp(10.0)))
    assert np.isclose(np_exp[3], 50.0)  # for large x, log(1+e^x) ~ x

    # Tensor method
    assert np.allclose(z.log1pexp().to_numpy(), np_exp)


def test_gammainc_and_gammaincc():
    a = ox.tensor([0.5, 1.0, 2.0, 5.0])
    x = ox.tensor([1.0, 2.0, 3.0, 2.0])

    p = ox.special.gammainc(a, x)
    q = ox.special.gammaincc(a, x)
    np_p = p.to_numpy()
    np_q = q.to_numpy()

    # P(a, x) + Q(a, x) == 1
    assert np.allclose(np_p + np_q, 1.0, atol=1e-7)

    # Known analytical values:
    # P(0.5, 1.0) == erf(sqrt(1.0)) == erf(1.0)
    assert np.isclose(np_p[0], math.erf(1.0), atol=1e-7)
    # P(1.0, 2.0) == 1 - exp(-2.0)
    assert np.isclose(np_p[1], 1.0 - math.exp(-2.0), atol=1e-7)
    # P(2.0, 3.0) == 1 - (1 + 3) * exp(-3.0)
    assert np.isclose(np_p[2], 1.0 - 4.0 * math.exp(-3.0), atol=1e-7)

    # Tensor methods
    assert np.allclose(a.gammainc(x).to_numpy(), np_p)
    assert np.allclose(a.gammaincc(x).to_numpy(), np_q)


def test_orthogonal_polynomials():
    x = ox.tensor([-1.0, -0.5, 0.0, 0.5, 1.0])
    x_np = x.to_numpy()

    # Chebyshev T: T_2(x) = 2x^2 - 1, T_3(x) = 4x^3 - 3x
    t2 = ox.special.chebyshev_polynomial_t(x, 2)
    assert np.allclose(t2.to_numpy(), 2.0 * x_np**2 - 1.0)
    t3 = ox.special.chebyshev_polynomial_t(x, 3)
    assert np.allclose(t3.to_numpy(), 4.0 * x_np**3 - 3.0 * x_np)

    # Chebyshev U: U_2(x) = 4x^2 - 1, U_3(x) = 8x^3 - 4x
    u2 = ox.special.chebyshev_polynomial_u(x, 2)
    assert np.allclose(u2.to_numpy(), 4.0 * x_np**2 - 1.0)
    u3 = ox.special.chebyshev_polynomial_u(x, 3)
    assert np.allclose(u3.to_numpy(), 8.0 * x_np**3 - 4.0 * x_np)

    # Chebyshev V: V_1(x) = 2x - 1, V_2(x) = 4x^2 - 2x - 1
    v1 = ox.special.chebyshev_polynomial_v(x, 1)
    assert np.allclose(v1.to_numpy(), 2.0 * x_np - 1.0)
    v2 = ox.special.chebyshev_polynomial_v(x, 2)
    assert np.allclose(v2.to_numpy(), 4.0 * x_np**2 - 2.0 * x_np - 1.0)

    # Chebyshev W: W_1(x) = 2x + 1, W_2(x) = 4x^2 + 2x - 1
    w1 = ox.special.chebyshev_polynomial_w(x, 1)
    assert np.allclose(w1.to_numpy(), 2.0 * x_np + 1.0)
    w2 = ox.special.chebyshev_polynomial_w(x, 2)
    assert np.allclose(w2.to_numpy(), 4.0 * x_np**2 + 2.0 * x_np - 1.0)

    # Laguerre L: L_1(x) = 1 - x, L_2(x) = 0.5 * (x^2 - 4x + 2)
    l1 = ox.special.laguerre_polynomial_l(x, 1)
    assert np.allclose(l1.to_numpy(), 1.0 - x_np)
    l2 = ox.special.laguerre_polynomial_l(x, 2)
    assert np.allclose(l2.to_numpy(), 0.5 * (x_np**2 - 4.0 * x_np + 2.0))


def test_bessel_and_airy():
    x = ox.tensor([1.0, 2.0, 5.0])
    x_np = x.to_numpy()

    y0 = ox.special.bessel_y0(x)
    assert np.isclose(y0.to_numpy()[0], 0.0882569642, atol=1e-4)

    y1 = ox.special.bessel_y1(x)
    assert np.isclose(y1.to_numpy()[0], -0.7812128213, atol=1e-3)

    k0 = ox.special.modified_bessel_k0(x)
    assert np.isclose(k0.to_numpy()[0], 0.4210244382, atol=1e-4)

    k1 = ox.special.modified_bessel_k1(x)
    assert np.isclose(k1.to_numpy()[0], 0.6019072302, atol=1e-3)

    # Scaled K0: exp(x) * K0(x)
    sk0 = ox.special.scaled_modified_bessel_k0(x)
    assert np.allclose(sk0.to_numpy(), np.exp(x_np) * k0.to_numpy())

    # Airy Ai(0) = 1 / (3^(2/3) * Gamma(2/3)) ~ 0.355028
    zero_t = ox.tensor([0.0])
    ai_zero = ox.special.airy_ai(zero_t)
    assert np.isclose(ai_zero.to_numpy()[0], 0.35502805, atol=1e-5)


def test_complex_tensor_extended_operations():
    r = ox.tensor([[1.0, 2.0], [3.0, 4.0]])
    im = ox.tensor([[0.5, 1.5], [2.5, 3.5]])
    c = ox.complex(r, im)

    # Slicing and indexing
    assert len(c) == 2
    c0 = c[0]
    assert c0.shape == (2,)
    assert c0.real.to_numpy().tolist() == [1.0, 2.0]
    assert c0.imag.to_numpy().tolist() == [0.5, 1.5]

    # Iteration
    rows = list(c)
    assert len(rows) == 2
    assert rows[1].real.to_numpy().tolist() == [3.0, 4.0]

    # Reductions: sum and mean
    s = c.sum(dim=0)
    assert np.allclose(s.real.to_numpy(), [4.0, 6.0])
    assert np.allclose(s.imag.to_numpy(), [3.0, 5.0])

    m = c.mean(dim=0)
    assert np.allclose(m.real.to_numpy(), [2.0, 3.0])
    assert np.allclose(m.imag.to_numpy(), [1.5, 2.5])

    # Norm
    nm = c.norm()
    expected_norm = np.linalg.norm(r.to_numpy() + 1j * im.to_numpy())
    assert np.isclose(nm.item(), expected_norm, atol=1e-5)

    # Transpose and reshape
    ct = c.t()
    assert ct.shape == (2, 2)
    assert ct.real.to_numpy().tolist() == [[1.0, 3.0], [2.0, 4.0]]

    flat_c = c.view(-1)
    assert flat_c.shape == (4,)
    assert flat_c.real.to_numpy().tolist() == [1.0, 2.0, 3.0, 4.0]

    # clone and detach
    c_clone = c.clone()
    assert np.allclose(c_clone.numpy(), c.numpy())
    c_det = c.detach()
    assert not c_det.requires_grad


def test_fft_complex_tensor_integration():
    r = ox.tensor([1.0, 2.0, 3.0, 4.0], dtype=ox.float32)
    im = ox.tensor([0.0, 1.0, -1.0, 0.0], dtype=ox.float32)
    c = ox.complex(r, im)

    # ox.fft.fft on ComplexTensor returns ComplexTensor
    spec = ox.fft.fft(c)
    assert spec.is_complex() is True
    assert isinstance(spec, ox.ComplexTensor)

    # Verify against numpy fft
    np_c = r.to_numpy() + 1j * im.to_numpy()
    np_fft = np.fft.fft(np_c)
    assert np.allclose(spec.numpy(), np_fft, atol=1e-5)

    # Inverse FFT returns original complex tensor
    rec = ox.fft.ifft(spec)
    assert np.allclose(rec.numpy(), np_c, atol=1e-5)

    # view_as_real and view_as_complex round-trip
    real_mat = ox.fft.view_as_real(c)
    assert real_mat.shape == (4, 2)
    c_back = ox.fft.view_as_complex(real_mat)
    assert np.allclose(c_back.numpy(), np_c)
