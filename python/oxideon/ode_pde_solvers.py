"""Numerical ODE and PDE Solvers for Scientific Machine Learning.

Implements:
36. RungeKutta4: Classical 4th-order explicit Runge-Kutta integrator (RK4) with Tensor autograd support
37. DormandPrince54: Adaptive step size 5(4) embedded pair ODE solver (DOPRI5)
38. SymplecticEuler: Symplectic phase-space preserving integrator for Hamiltonian systems
39. Burgers1DEquation: Viscous 1D non-linear Burgers' partial differential equation solver
40. Wave1DEquation: 1D hyperbolic wave equation solver with FDTD and CFL stability
41. Diffusion1DEquation: 1D parabolic heat/diffusion equation solver with stability check
"""

from typing import Callable, List, Tuple, Optional, Union, Any
import math
import numpy as np

from .oxideon import Tensor as _Tensor, from_numpy as _from_numpy


class RungeKutta4:
    """Classical 4th-order explicit Runge-Kutta integrator for ODEs."""

    @staticmethod
    def step(f: Callable[[float, Any], Any], t: float, y: Any, dt: float) -> Any:
        """Computes single RK4 step: y_{n+1} = y_n + dt/6 * (k1 + 2*k2 + 2*k3 + k4)"""
        k1 = f(t, y)
        k2 = f(t + 0.5 * dt, y + k1 * (0.5 * dt))
        k3 = f(t + 0.5 * dt, y + k2 * (0.5 * dt))
        k4 = f(t + dt, y + k3 * dt)
        return y + (k1 + k2 * 2.0 + k3 * 2.0 + k4) * (dt / 6.0)

    @staticmethod
    def integrate(
        f: Callable[[float, Any], Any],
        y0: Union[_Tensor, np.ndarray],
        t_span: Tuple[float, float],
        steps: int = 100,
    ) -> Tuple[np.ndarray, _Tensor]:
        """Integrates dy/dt = f(t, y) over t_span = (t0, t1)."""
        if steps < 1:
            raise ValueError(f"steps must be >= 1, got {steps}")
        t0, t1 = float(t_span[0]), float(t_span[1])
        dt = (t1 - t0) / float(steps)

        if isinstance(y0, _Tensor) and getattr(y0, "requires_grad", False):
            # Differentiable Tensor trajectory integration
            y_curr_t = y0
            t_hist = [t0]
            y_hist_t: List[_Tensor] = [y_curr_t.unsqueeze(0)]
            curr_t = t0
            for _ in range(steps):
                y_curr_t = RungeKutta4.step(f, curr_t, y_curr_t, dt)
                curr_t += dt
                t_hist.append(curr_t)
                y_hist_t.append(y_curr_t.unsqueeze(0))
            return np.array(t_hist, dtype=np.float32), _Tensor.cat(y_hist_t, dim=0)

        y_curr = y0.to_numpy().astype(np.float64) if isinstance(y0, _Tensor) else np.array(y0, dtype=np.float64)
        t_hist = [t0]
        y_hist = [y_curr.copy()]

        curr_t = t0
        for _ in range(steps):
            y_curr = RungeKutta4.step(f, curr_t, y_curr, dt)
            curr_t += dt
            t_hist.append(curr_t)
            y_hist.append(y_curr.copy())

        return np.array(t_hist, dtype=np.float32), _from_numpy(np.array(y_hist, dtype=np.float32))


class DormandPrince54:
    """Adaptive step-size 5th order Dormand-Prince integrator (DOPRI5) with 4th order error estimator."""

    C = [0.0, 1.0 / 5.0, 3.0 / 10.0, 4.0 / 5.0, 8.0 / 9.0, 1.0, 1.0]
    A = [
        [],
        [1.0 / 5.0],
        [3.0 / 40.0, 9.0 / 40.0],
        [44.0 / 45.0, -56.0 / 15.0, 32.0 / 9.0],
        [19372.0 / 6561.0, -25360.0 / 2187.0, 64448.0 / 6561.0, -212.0 / 729.0],
        [9017.0 / 3168.0, -355.0 / 33.0, 46732.0 / 5247.0, 49.0 / 176.0, -5103.0 / 18656.0],
        [35.0 / 384.0, 0.0, 500.0 / 1113.0, 125.0 / 192.0, -2187.0 / 6784.0, 11.0 / 84.0],
    ]
    B5 = [35.0 / 384.0, 0.0, 500.0 / 1113.0, 125.0 / 192.0, -2187.0 / 6784.0, 11.0 / 84.0, 0.0]
    B4 = [
        5179.0 / 57600.0,
        0.0,
        7571.0 / 16695.0,
        393.0 / 640.0,
        -92097.0 / 339200.0,
        187.0 / 2100.0,
        1.0 / 40.0,
    ]

    @staticmethod
    def integrate(
        f: Callable[[float, np.ndarray], np.ndarray],
        y0: Union[_Tensor, np.ndarray],
        t_span: Tuple[float, float],
        rtol: float = 1e-4,
        atol: float = 1e-6,
        dt_init: float = 0.01,
        dt_min: float = 1e-6,
        dt_max: float = 0.5,
    ) -> Tuple[np.ndarray, _Tensor]:
        y = y0.to_numpy().astype(np.float64) if isinstance(y0, _Tensor) else np.array(y0, dtype=np.float64)
        t0, t1 = t_span
        t = t0
        dt = min(dt_init, t1 - t0)

        t_vals = [t]
        y_vals = [y.copy()]

        while t < t1 - 1e-12:
            if t + dt > t1:
                dt = t1 - t

            k = [f(t, y)]
            for i in range(1, 7):
                ti = t + DormandPrince54.C[i] * dt
                yi = y + dt * sum(DormandPrince54.A[i][j] * k[j] for j in range(i))
                k.append(f(ti, yi))

            y5 = y + dt * sum(DormandPrince54.B5[i] * k[i] for i in range(7))
            y4 = y + dt * sum(DormandPrince54.B4[i] * k[i] for i in range(7))

            err = float(np.max(np.abs(y5 - y4) / (atol + rtol * np.maximum(np.abs(y), np.abs(y5)))))

            if err <= 1.0 or dt <= dt_min:
                t += dt
                y = y5
                t_vals.append(t)
                y_vals.append(y.copy())
                factor = 0.9 * (1.0 / (err + 1e-8)) ** 0.2
            else:
                factor = 0.9 * (1.0 / (err + 1e-8)) ** 0.25

            factor = max(0.2, min(5.0, factor))
            dt = max(dt_min, min(dt_max, dt * factor))

        return np.array(t_vals, dtype=np.float32), _from_numpy(np.array(y_vals, dtype=np.float32))


class SymplecticEuler:
    """Symplectic Euler (semi-implicit Euler) for Hamiltonian mechanics H(q, p)."""

    @staticmethod
    def integrate(
        grad_v: Callable[[np.ndarray], np.ndarray],
        grad_t: Callable[[np.ndarray], np.ndarray],
        q0: Union[_Tensor, np.ndarray],
        p0: Union[_Tensor, np.ndarray],
        t_span: Tuple[float, float],
        steps: int = 100,
    ) -> Tuple[np.ndarray, _Tensor, _Tensor]:
        q = q0.to_numpy().astype(np.float64) if isinstance(q0, _Tensor) else np.array(q0, dtype=np.float64)
        p = p0.to_numpy().astype(np.float64) if isinstance(p0, _Tensor) else np.array(p0, dtype=np.float64)
        t0, t1 = t_span
        dt = (t1 - t0) / steps

        t_hist = [t0]
        q_hist = [q.copy()]
        p_hist = [p.copy()]

        curr_t = t0
        for _ in range(steps):
            p = p - dt * grad_v(q)
            q = q + dt * grad_t(p)
            curr_t += dt
            t_hist.append(curr_t)
            q_hist.append(q.copy())
            p_hist.append(p.copy())

        return (
            np.array(t_hist, dtype=np.float32),
            _from_numpy(np.array(q_hist, dtype=np.float32)),
            _from_numpy(np.array(p_hist, dtype=np.float32)),
        )


class Burgers1DEquation:
    """1D Viscous Burgers' PDE solver: du/dt + u * du/dx = nu * d2u/dx2."""

    @staticmethod
    def solve(
        u0: Union[_Tensor, np.ndarray],
        dx: float = 0.05,
        dt: float = 0.001,
        nu: float = 0.07,
        num_steps: int = 50,
    ) -> _Tensor:
        """Solves 1D viscous Burgers equation with periodic boundary conditions."""
        u = (u0.to_numpy() if isinstance(u0, _Tensor) else np.array(u0, dtype=np.float64)).copy()
        dtdx = dt / dx
        nu_factor = nu * (dt / (dx ** 2))
        for _ in range(num_steps):
            u_im1 = np.roll(u, 1)
            u_ip1 = np.roll(u, -1)
            convection = np.where(
                u >= 0.0,
                u * dtdx * (u - u_im1),
                u * dtdx * (u_ip1 - u),
            )
            diffusion = nu_factor * (u_ip1 - 2.0 * u + u_im1)
            u = u - convection + diffusion

        return _from_numpy(u.astype(np.float32))


class Wave1DEquation:
    """1D Acoustic/Hyperbolic Wave Equation solver: d2u/dt2 = c^2 * d2u/dx2."""

    @staticmethod
    def solve(
        u0: Union[_Tensor, np.ndarray],
        v0: Optional[Union[_Tensor, np.ndarray]] = None,
        c: float = 1.0,
        dx: float = 0.05,
        dt: float = 0.01,
        num_steps: int = 50,
    ) -> _Tensor:
        """Explicit FDTD solver for 1D wave equation with fixed boundary conditions."""
        u_curr = (u0.to_numpy() if isinstance(u0, _Tensor) else np.array(u0, dtype=np.float64)).copy()
        nx = len(u_curr)

        cfl = c * dt / dx
        if cfl > 1.0:
            raise ValueError(f"CFL stability violated: c*dt/dx = {cfl:.3f} > 1.0")

        c2 = cfl ** 2

        if v0 is not None:
            v_curr = v0.to_numpy() if isinstance(v0, _Tensor) else np.array(v0, dtype=np.float64)
        else:
            v_curr = np.zeros(nx, dtype=np.float64)

        u_prev = np.zeros_like(u_curr)
        u_prev[1:-1] = (
            u_curr[1:-1]
            - dt * v_curr[1:-1]
            + 0.5 * c2 * (u_curr[2:] - 2.0 * u_curr[1:-1] + u_curr[:-2])
        )

        for _ in range(num_steps):
            u_next = np.zeros_like(u_curr)
            u_next[1:-1] = (
                2.0 * u_curr[1:-1]
                - u_prev[1:-1]
                + c2 * (u_curr[2:] - 2.0 * u_curr[1:-1] + u_curr[:-2])
            )
            u_next[0] = 0.0
            u_next[-1] = 0.0

            u_prev = u_curr
            u_curr = u_next

        return _from_numpy(u_curr.astype(np.float32))


class Diffusion1DEquation:
    """1D Heat/Diffusion Equation solver: du/dt = alpha * d2u/dx2."""

    @staticmethod
    def solve(
        u0: Union[_Tensor, np.ndarray],
        alpha: float = 0.01,
        dx: float = 0.05,
        dt: float = 0.01,
        num_steps: int = 50,
    ) -> _Tensor:
        """Solves 1D heat diffusion with Neumann (insulated) boundary conditions."""
        u = (u0.to_numpy() if isinstance(u0, _Tensor) else np.array(u0, dtype=np.float64)).copy()
        if len(u) < 3:
            raise ValueError(f"Diffusion1DEquation requires at least 3 grid points, got {len(u)}")

        r = alpha * dt / (dx ** 2)
        if r > 0.5:
            raise ValueError(f"Diffusion stability violated: alpha*dt/dx^2 = {r:.3f} > 0.5")

        for _ in range(num_steps):
            u_next = np.empty_like(u)
            u_next[1:-1] = u[1:-1] + r * (u[2:] - 2.0 * u[1:-1] + u[:-2])
            u_next[0] = u_next[1]
            u_next[-1] = u_next[-2]
            u = u_next

        return _from_numpy(u.astype(np.float32))
