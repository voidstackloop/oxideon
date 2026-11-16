"""Classical numerical methods with ox.numerical.

Covers adaptive quadrature, Gauss-Legendre rules, 1-D interpolation,
scalar minimization, curve fitting, and the physical-constants table.
Run: python examples/science_numerical_methods.py
"""

import numpy as np

import oxideon as ox


def main() -> None:
    # --- Adaptive quadrature (SciPy-quad-compatible defaults) ---
    val, err = ox.numerical.quad(np.sin, 0.0, np.pi)
    print(f"int_0^pi sin(x) dx = {val:.12f}  (+/- {err:.2e})")
    assert abs(val - 2.0) < 1e-9

    # --- Fixed-order Gauss-Legendre ---
    nodes, weights = ox.numerical.gauss_legendre(5)
    print(f"GL(5) nodes: {np.asarray(nodes).round(4)}")
    q5 = ox.numerical.fixed_quad(lambda x: x**4, 0.0, 1.0)
    print(f"fixed_quad x^4 on [0,1] = {q5:.12f}  (exact 0.2)")
    assert abs(q5 - 0.2) < 1e-12
    assert abs(float(np.sum(weights)) - 2.0) < 1e-12

    # --- Interpolation: piecewise cubic and cubic spline ---
    xs = [0.0, 1.0, 2.0, 3.0]
    ys = [0.0, 1.0, 0.0, 1.0]
    interp = ox.numerical.Interp1D(xs, ys, kind="linear")
    spline = ox.numerical.CubicSpline(xs, ys)
    print(f"interp(0.5) = {float(interp(0.5)):.6f}, spline(0.5) = {float(spline(0.5)):.6f}")
    assert abs(float(interp(1.0)) - 1.0) < 1e-12
    assert abs(float(spline(2.0)) - 0.0) < 1e-12

    # --- Scalar minimization over a bracket ---
    res = ox.numerical.minimize_scalar(lambda x: (x - 2.0) ** 2 + 1.0, 0.0, 5.0)
    print(f"minimize_scalar: x = {res.x:.9f}, fun = {res.fun:.9f}, success = {res.success}")
    assert res.success and abs(res.x - 2.0) < 1e-6 and abs(res.fun - 1.0) < 1e-9

    # --- Nonlinear least-squares curve fitting ---
    xdata = np.linspace(0.0, 2.0, 9)
    ydata = 3.0 * np.exp(-0.7 * xdata)
    popt, pcov = ox.numerical.curve_fit(
        lambda x, a, b: a * np.exp(-b * x), xdata, ydata, p0=[2.0, 0.5]
    )
    print(f"curve_fit: a = {popt[0]:.6f}, b = {popt[1]:.6f}")
    assert abs(popt[0] - 3.0) < 1e-4 and abs(popt[1] - 0.7) < 1e-4
    assert pcov.shape == (2, 2)

    # --- Physical constants ---
    c = ox.numerical.physical_constants["speed of light in vacuum"]
    print(f"c = {c.value} {c.unit}  (uncertainty {c.uncertainty})")
    assert c.value == 299792458.0

    print("science_numerical_methods: OK")


if __name__ == "__main__":
    main()
