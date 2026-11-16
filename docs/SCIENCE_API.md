# Classical Science API (`ox.numerical`, `ox.stats`, `ox.physics`)

Pure-Python numerical core, classical statistics, and physics simulators.
Zero required runtime dependencies beyond NumPy and the `oxideon` core —
SciPy/PyTorch appear only as *test oracles*, never as imports.

Stability: **Tier 3 — Provisional** (see `docs/API_STABILITY.md`):
behavior is tested (71 tests in `tests/test_numerical.py`,
`tests/test_stats.py`, `tests/test_physics.py`, several cross-checked
against SciPy), but names and signatures may still evolve.

---

## Table of Contents

1. [Numerical core (`ox.numerical`)](#1-numerical-core-oxnumerical)
2. [Classical statistics (`ox.stats`)](#2-classical-statistics-oxstats)
3. [Physics simulators (`ox.physics`)](#3-physics-simulators-oxphysics)
4. [Error contract](#4-error-contract)

---

## 1. Numerical core (`ox.numerical`)

Quadrature, interpolation, scalar optimization, least-squares curve
fitting, and a CODATA-style physical-constants table.

```python
import numpy as np
import oxideon as ox

# Adaptive quadrature: (value, estimated error), SciPy-quad-compatible defaults
val, err = ox.numerical.quad(np.sin, 0.0, np.pi)   # ~2.0
print(val, err)

# Fixed-order Gauss-Legendre, plus nodes/weights access
xs, ws = ox.numerical.gauss_legendre(5)
print(ox.numerical.fixed_quad(lambda x: x**4, 0.0, 1.0))  # ~0.2

# 1-D interpolation (linear / nearest / slinear / ...) and cubic splines
f = ox.numerical.Interp1D([0.0, 1.0, 2.0], [0.0, 1.0, 0.0], kind="linear")
s = ox.numerical.CubicSpline([0.0, 1.0, 2.0], [0.0, 1.0, 0.0])
print(float(f(0.5)), float(s(0.5)))

# Scalar minimization over a bracket ('brent' or 'golden')
res = ox.numerical.minimize_scalar(lambda x: (x - 2.0) ** 2 + 1.0, 0.0, 5.0)
print(res.x, res.fun, res.success)  # ~2.0  ~1.0  True

# Nonlinear least squares: (optimal params, covariance estimate)
popt, pcov = ox.numerical.curve_fit(
    lambda x, a, b: a * np.exp(-b * x),
    np.linspace(0, 2, 9), [3.0, 2.5, 2.0, 1.7, 1.4, 1.1, 0.9, 0.8, 0.6],
    p0=[2.0, 0.5],
)
print(popt)

# Physical constants: mapping name -> Constant(value, unit, uncertainty)
c = ox.numerical.physical_constants["speed of light in vacuum"]
print(c.value, c.unit)  # 299792458.0  m s^-1
```

Function reference:

| Function | Purpose |
|---|---|
| `quad(f, a, b, epsabs, epsrel, limit)` | Adaptive 1-D quadrature → `(value, abserr)` |
| `dblquad(f, a, b, gfun, hfun, ...)` | Adaptive 2-D quadrature over `y in [gfun(x), hfun(x)]` |
| `fixed_quad(f, a, b, n=5)` | N-point Gauss-Legendre quadrature |
| `gauss_legendre(n)` | `(nodes, weights)` on `[-1, 1]` |
| `Interp1D(x, y, kind, bounds_error, fill_value)` | 1-D interpolant (callable; `linear`/`nearest`/`slinear`/`zero`/`next`/`previous`) |
| `CubicSpline(x, y, bc_type, extrapolate)` | Cubic spline (callable) |
| `minimize_scalar(f, a, b, method, tol, maxiter)` | Bounded scalar minimizer → `ScalarOptResult(x, fun, nfev, nit, success)` |
| `curve_fit(f, xdata, ydata, p0, sigma, maxiter, tol, absolute_sigma)` | Nonlinear least squares → `(popt, pcov)` |
| `physical_constants` / `Constant` | Name → `(value, unit, uncertainty)` table |

## 2. Classical statistics (`ox.stats`)

Hypothesis tests, moments, ECDF, and multiple-testing correction.
No SciPy at runtime: reference distributions are implemented in-module
(regularized gamma/beta continued fractions) and cross-checked against
SciPy in tests.

```python
import numpy as np
import oxideon as ox

rng = np.random.default_rng(0)
a = rng.normal(0.0, 1.0, 200)
b = rng.normal(0.5, 1.0, 200)

# t-tests: one-sample, independent (Student/Welch), paired
print(ox.stats.ttest_1samp(a, popmean=0.0))
print(ox.stats.ttest_ind(a, b, equal_var=False))
print(ox.stats.ttest_rel(a, a + rng.normal(0, 0.1, 200)))

# Chi-square: goodness-of-fit and contingency tables
print(ox.stats.chisquare([16, 18, 16, 14, 12, 12]))
print(ox.stats.chi2_contingency([[10, 10, 20], [20, 20, 10]]))

# Correlation with p-values
print(ox.stats.pearsonr(a, b))
print(ox.stats.spearmanr(a, b))

# Non-parametric: Mann-Whitney U, Kolmogorov-Smirnov vs Normal
print(ox.stats.mannwhitneyu(a, b))
print(ox.stats.kstest(rng.normal(size=300), "norm"))

# Moments and normality, ECDF, multiple-testing correction
print(ox.stats.skew(a), ox.stats.kurtosis(a), ox.stats.jarque_bera(a))
e = ox.stats.ecdf(a)
print(e.quantiles[:3], e([-1.0, 0.0, 1.0]))  # step function, callable
rejected, adjusted = ox.stats.p_adjust([0.01, 0.04, 0.30], method="holm")
print(rejected, adjusted)
```

Function reference:

| Function | Returns |
|---|---|
| `ttest_1samp(a, popmean, alternative)` | `TtestResult(statistic, pvalue, df)` |
| `ttest_ind(a, b, equal_var, alternative)` | `TtestResult(...)` (`equal_var=False` → Welch) |
| `ttest_rel(a, b, alternative)` | `TtestResult(...)` (paired) |
| `chisquare(f_obs, f_exp, ddof)` | `Chi2Result(statistic, pvalue, dof, expected_freq)` |
| `chi2_contingency(observed, correction)` | `Chi2Result(...)` (Yates for 2×2 when `correction`) |
| `pearsonr(x, y)` / `spearmanr(x, y)` | `CorrResult(statistic, pvalue)` |
| `mannwhitneyu(x, y, continuity, alternative)` | `MannWhitneyResult(statistic, pvalue)` |
| `kstest(x, cdf="norm", args)` | `KSTestResult(statistic, pvalue)` (asymptotic) |
| `skew(x, bias)` / `kurtosis(x, fisher, bias)` | `float` |
| `jarque_bera(x)` | `JarqueBeraResult(statistic, pvalue)` |
| `ecdf(x)` | `ECDF` (callable; `.quantiles`, `.probabilities`) |
| `p_adjust(pvals, method, alpha)` | `(rejected, adjusted)`; `bonferroni`/`holm`/`bh`/`fdr_bh`/`none` |

## 3. Physics simulators (`ox.physics`)

Explicit PDE solvers (Rust-backed, returning `ox.Tensor`), N-body
gravity, and symplectic mechanics. 1-D counterparts live in the
pre-existing `ox.ode_pde_solvers` module and serve as cross-check
oracles in tests.

```python
import numpy as np
import oxideon as ox

# 2-D wave equation, fixed zero boundaries (CFL-checked: c*dt/dx must be stable)
u0 = np.zeros((21, 21))
u0[10, 10] = 1.0
u = ox.physics.Wave2DEquation.solve(u0, c=1.0, dx=0.05, dy=0.05, dt=0.01, num_steps=5)
print(type(u), u.to_numpy().shape)  # ox.Tensor (21, 21)

# 2-D diffusion, insulated (Neumann) edges, mass-conserving
v = ox.physics.Diffusion2DEquation.solve(u0, alpha=0.05, dx=0.05, dy=0.05, dt=0.01, num_steps=10)
print(abs(v.to_numpy().sum() - u0.sum()))  # ~1e-5 (f32 solver)

# Circular two-body orbit: velocity + period helpers, then integrate
m, r = 1000.0, 10.0
vc = ox.physics.circular_velocity(m, r)
T = ox.physics.orbital_period(r, m)
print(vc, T)  # sqrt(G m / r), 2 pi sqrt(r^3 / G m)
pos = np.array([[0.0, 0.0], [r, 0.0]])
vel = np.array([[0.0, 0.0], [0.0, vc]])
t, p_traj, v_traj = ox.physics.simulate_nbody(
    pos, vel, [m, 1.0], dt=T / 200, num_steps=200, method="verlet")
print(ox.physics.nbody_energy(p_traj.to_numpy()[-1], v_traj.to_numpy()[-1], [m, 1.0]))

# Generic symplectic (leapfrog) integration of Hamilton's equations
tq, q_traj, p_traj = ox.physics.hamiltonian_leapfrog(
    grad_potential=lambda q: q, q0=[1.0], p0=[0.0], dt=0.01, num_steps=100)
print(q_traj.to_numpy()[-1])  # ~cos(1)
```

Function reference:

| Function | Purpose |
|---|---|
| `Wave2DEquation.solve(u0, v0, c, dx, dy, dt, num_steps)` | 2-D wave equation → `ox.Tensor` (CFL-validated) |
| `Diffusion2DEquation.solve(u0, alpha, dx, dy, dt, num_steps)` | 2-D heat equation → `ox.Tensor` (stability-validated) |
| `simulate_nbody(pos, vel, masses, dt, num_steps, g, softening, method)` | Gravity integration → `(times, positions, velocities)` trajectories |
| `nbody_energy / nbody_momentum(...)` | Total energy / momentum (conservation checks) |
| `hamiltonian_leapfrog(grad_potential, q0, p0, mass, dt, num_steps)` | Symplectic integration → `(times, positions, momenta)` trajectories |
| `circular_velocity(m_central, r, g)` / `orbital_period(a, m_central, g)` | Kepler helpers |

## 4. Error contract

All three modules validate eagerly and raise descriptive errors:

- bad shapes / empty inputs / non-finite bounds → `ValueError`
- wrong types (e.g. non-callable `f`, unknown `method`/`kind`) → `TypeError` / `ValueError` naming the supported set
- unstable PDE steps (CFL / diffusion-number violations) → `ValueError` before any compute
- iterative routines report non-convergence in-band (`ScalarOptResult.success is False`)
  rather than raising, so callers can branch on it.
