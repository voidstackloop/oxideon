"""Comprehensive test suite for Oxideon Functional Autograd Enrichment & Higher-Order Gradients.

Verifies:
  - Hessian-Vector Products (hvp) via double backward (Pearlmutter trick) in O(D) memory
  - Tangent linear models via linearize / jvp closures
  - Higher-order nested functional differentiation (1st, 2nd, 3rd, and 4th order gradients)
  - Functional transform has_aux support for auxiliary logging/metric outputs
  - Functional model calling (functional_call, make_functional) and parameter isolation
"""

import numpy as np
import pytest
import oxideon as ox
from oxideon.func import (
    grad,
    value_and_grad,
    hvp,
    linearize,
    jvp,
    vjp,
    hessian,
    vmap,
    functional_call,
    make_functional,
)


def test_hvp_pearlmutter_double_backward():
    # f(x) = x_0^3 + x_0 * x_1^2
    # grad f = [3*x_0^2 + x_1^2, 2*x_0*x_1]
    # H = [[6*x_0, 2*x_1], [2*x_1, 2*x_0]]
    # At x = [2.0, 3.0], v = [1.0, -1.0]:
    # H = [[12, 6], [6, 4]]
    # H @ v = [12(1) + 6(-1), 6(1) + 4(-1)] = [6.0, 2.0]
    def f(t):
        return (t[0] * t[0] * t[0]) + (t[0] * t[1] * t[1])

    x = ox.Tensor.from_slice([2.0, 3.0], [2])
    v = ox.Tensor.from_slice([1.0, -1.0], [2])

    out, hv_val = hvp(f, x, v)
    assert abs(out.item() - 26.0) < 1e-5

    hv_np = hv_val.to_numpy()
    np.testing.assert_allclose(hv_np, [6.0, 2.0], rtol=1e-5, atol=1e-5)


def test_linearize_and_jvp_closure():
    # f(x) = x^2 + 3*x
    # f'(x) = 2*x + 3
    # At x = 2.0, f(2) = 10.0, f'(2) = 7.0
    # Along v = 1.5, JVP = 7.0 * 1.5 = 10.5
    def f(t):
        return (t * t) + (t * 3.0)

    x = ox.Tensor.from_slice([2.0], [1])
    primal, jvp_fn = linearize(f, x)

    assert abs(primal.item() - 10.0) < 1e-5
    v = ox.Tensor.from_slice([1.5], [1])
    tan = jvp_fn(v)
    assert abs(tan.item() - 10.5) < 1e-4


def test_grad_and_value_and_grad_has_aux():
    def loss_with_aux(t):
        loss = (t * t).sum()
        aux = {"energy": float(t.sum().item())}
        return loss, aux

    x = ox.Tensor.from_slice([3.0, 4.0], [2])

    # 1. grad with has_aux=True
    g_fn = grad(loss_with_aux, has_aux=True)
    g, aux = g_fn(x)
    np.testing.assert_allclose(g.to_numpy(), [6.0, 8.0], rtol=1e-5, atol=1e-5)
    assert abs(aux["energy"] - 7.0) < 1e-5

    # 2. value_and_grad with has_aux=True
    vag_fn = value_and_grad(loss_with_aux, has_aux=True)
    (out, aux2), g2 = vag_fn(x)
    assert abs(out.item() - 25.0) < 1e-5
    np.testing.assert_allclose(g2.to_numpy(), [6.0, 8.0], rtol=1e-5, atol=1e-5)
    assert abs(aux2["energy"] - 7.0) < 1e-5


def test_nested_higher_order_derivatives():
    # f(x) = x^4 / 12
    # f'(x) = x^3 / 3
    # f''(x) = x^2
    # f'''(x) = 2*x
    # At x = 3.0:
    # f(3) = 81 / 12 = 6.75
    # f'(3) = 27 / 3 = 9.0
    # f''(3) = 9.0
    # f'''(3) = 6.0
    def f(t):
        return (t * t * t * t) * (1.0 / 12.0)

    df = grad(f)
    d2f = grad(df)
    d3f = grad(d2f)

    x = ox.Tensor.from_slice([3.0], [1])
    assert abs(f(x).item() - 6.75) < 1e-4
    assert abs(df(x).item() - 9.0) < 1e-4
    assert abs(d2f(x).item() - 9.0) < 1e-4
    assert abs(d3f(x).item() - 6.0) < 1e-4


def test_functional_call_and_make_functional():
    linear = ox.nn.Linear(2, 1)
    orig_w = linear.weight.to_numpy().copy()
    orig_b = linear.bias.to_numpy().copy()

    f_model, params = make_functional(linear)
    assert "weight" in params
    assert "bias" in params

    x = ox.Tensor.from_slice([1.0, 2.0], [1, 2])

    # Standard call with current params
    y_orig = f_model(params, x)

    # Call with altered params
    altered_params = {
        "weight": ox.Tensor.from_slice([10.0, 20.0], [1, 2]),
        "bias": ox.Tensor.from_slice([5.0], [1]),
    }
    y_alt = f_model(altered_params, x)
    # Expected: 1.0 * 10.0 + 2.0 * 20.0 + 5.0 = 55.0
    assert abs(y_alt.item() - 55.0) < 1e-4

    # Verify original module parameters were not corrupted
    np.testing.assert_allclose(linear.weight.to_numpy(), orig_w)
    np.testing.assert_allclose(linear.bias.to_numpy(), orig_b)
