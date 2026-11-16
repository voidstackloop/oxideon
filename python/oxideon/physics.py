"""Classical physics simulations: 2-D fields, N-body gravity, Hamiltonian mechanics.

Implements:
69. Wave2DEquation: 2-D acoustic wave equation (FDTD, CFL-gated)
70. Diffusion2DEquation: 2-D heat/diffusion equation (FTCS, stability-gated)
71. simulate_nbody: Gravitational N-body (velocity Verlet or RK4) with diagnostics
72. hamiltonian_leapfrog: Generic kick-drift-kick integrator for separable Hamiltonians
73. circular_velocity / orbital_period: Two-body Kepler helpers

Conventions mirror `ode_pde_solvers`: float64 compute, float32 Tensor outputs,
descriptive ValueErrors (no silent instability: CFL/diffusion numbers are gated).
"""

from typing import Any, Callable, Optional, Tuple, Union
import math
import operator
import numpy as np

from .oxideon import Tensor as _Tensor, from_numpy as _from_numpy
from .ode_pde_solvers import RungeKutta4


__all__ = [
    "Wave2DEquation",
    "Diffusion2DEquation",
    "simulate_nbody",
    "nbody_energy",
    "nbody_momentum",
    "hamiltonian_leapfrog",
    "circular_velocity",
    "orbital_period",
]


def _check_positive(value: Any, label: str, name: str, allow_zero: bool = False) -> float:
    try:
        v = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name}: {label} must be a real scalar, got {value!r}") from exc
    if not math.isfinite(v) or (v < 0.0 if allow_zero else v <= 0.0):
        bound = "non-negative" if allow_zero else "positive"
        raise ValueError(f"{name}: {label} must be finite and {bound}, got {value!r}")
    return v


def _check_num_steps(num_steps: int, name: str) -> int:
    if isinstance(num_steps, bool):
        raise ValueError(f"{name}: num_steps must be a positive integer, got {num_steps!r}")
    try:
        steps = operator.index(num_steps)
    except TypeError as exc:
        raise TypeError(f"{name}: num_steps must be an integer, got {num_steps!r}") from exc
    if steps < 1:
        raise ValueError(f"{name}: num_steps must be a positive integer, got {num_steps!r}")
    return steps


def _as_2d_field(u: Any, name: str) -> np.ndarray:
    arr = u.to_numpy() if isinstance(u, _Tensor) else np.asarray(u, dtype=np.float64)
    arr = np.asarray(arr, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[0] < 3 or arr.shape[1] < 3:
        raise ValueError(f"{name}: need a 2-D field with shape >= (3, 3), got {arr.shape}")
    if bool(np.any(~np.isfinite(arr))):
        raise ValueError(f"{name}: initial field must be finite")
    return arr.copy()


class Wave2DEquation:
    """2-D acoustic wave equation: d2u/dt2 = c^2 (d2u/dx2 + d2u/dy2), fixed zero boundaries."""

    @staticmethod
    def solve(
        u0: Union[_Tensor, np.ndarray],
        v0: Optional[Union[_Tensor, np.ndarray]] = None,
        c: float = 1.0,
        dx: float = 0.05,
        dy: float = 0.05,
        dt: float = 0.01,
        num_steps: int = 50,
    ) -> _Tensor:
        """Explicit FDTD solver gated by the 2-D CFL condition."""
        u_curr = _as_2d_field(u0, "Wave2DEquation")
        ny, nx = u_curr.shape
        c = _check_positive(c, "c", "Wave2DEquation")
        dx = _check_positive(dx, "dx", "Wave2DEquation")
        dy = _check_positive(dy, "dy", "Wave2DEquation")
        dt = _check_positive(dt, "dt", "Wave2DEquation")
        num_steps = _check_num_steps(num_steps, "Wave2DEquation")
        cfl = c * dt * math.sqrt(1.0 / dx ** 2 + 1.0 / dy ** 2)
        if cfl > 1.0:
            raise ValueError(f"Wave2DEquation: CFL stability violated: {cfl:.3f} > 1.0")
        cx = (c * dt / dx) ** 2
        cy = (c * dt / dy) ** 2

        if v0 is not None:
            v_curr = _as_2d_field(v0, "Wave2DEquation")
            if v_curr.shape != u_curr.shape:
                raise ValueError(f"Wave2DEquation: v0 shape {v_curr.shape} must match u0 {u_curr.shape}")
        else:
            v_curr = np.zeros_like(u_curr)

        u_prev = np.zeros_like(u_curr)
        u_prev[1:-1, 1:-1] = (
            u_curr[1:-1, 1:-1]
            - dt * v_curr[1:-1, 1:-1]
            + 0.5 * (cx * (u_curr[2:, 1:-1] - 2.0 * u_curr[1:-1, 1:-1] + u_curr[:-2, 1:-1])
                      + cy * (u_curr[1:-1, 2:] - 2.0 * u_curr[1:-1, 1:-1] + u_curr[1:-1, :-2]))
        )
        for _ in range(num_steps):
            u_next = np.zeros_like(u_curr)
            u_next[1:-1, 1:-1] = (
                2.0 * u_curr[1:-1, 1:-1]
                - u_prev[1:-1, 1:-1]
                + cx * (u_curr[2:, 1:-1] - 2.0 * u_curr[1:-1, 1:-1] + u_curr[:-2, 1:-1])
                + cy * (u_curr[1:-1, 2:] - 2.0 * u_curr[1:-1, 1:-1] + u_curr[1:-1, :-2])
            )
            u_prev = u_curr
            u_curr = u_next
        return _from_numpy(u_curr.astype(np.float32))


class Diffusion2DEquation:
    """2-D heat/diffusion equation: du/dt = alpha (d2u/dx2 + d2u/dy2), insulated (Neumann) edges."""

    @staticmethod
    def solve(
        u0: Union[_Tensor, np.ndarray],
        alpha: float = 0.01,
        dx: float = 0.05,
        dy: float = 0.05,
        dt: float = 0.01,
        num_steps: int = 50,
    ) -> _Tensor:
        """Explicit FTCS solver gated by the 2-D diffusion stability limit."""
        u = _as_2d_field(u0, "Diffusion2DEquation")
        alpha = _check_positive(alpha, "alpha", "Diffusion2DEquation")
        dx = _check_positive(dx, "dx", "Diffusion2DEquation")
        dy = _check_positive(dy, "dy", "Diffusion2DEquation")
        dt = _check_positive(dt, "dt", "Diffusion2DEquation")
        num_steps = _check_num_steps(num_steps, "Diffusion2DEquation")
        r = alpha * dt * (1.0 / dx ** 2 + 1.0 / dy ** 2)
        if r > 0.5:
            raise ValueError(f"Diffusion2DEquation: stability violated: {r:.3f} > 0.5")
        rx = alpha * dt / dx ** 2
        ry = alpha * dt / dy ** 2
        for _ in range(num_steps):
            u_next = np.empty_like(u)
            u_next[1:-1, 1:-1] = (
                u[1:-1, 1:-1]
                + rx * (u[2:, 1:-1] - 2.0 * u[1:-1, 1:-1] + u[:-2, 1:-1])
                + ry * (u[1:-1, 2:] - 2.0 * u[1:-1, 1:-1] + u[1:-1, :-2])
            )
            u_next[0, :] = u_next[1, :]
            u_next[-1, :] = u_next[-2, :]
            u_next[:, 0] = u_next[:, 1]
            u_next[:, -1] = u_next[:, -2]
            u = u_next
        return _from_numpy(u.astype(np.float32))


def _as_bodies(pos: Any, vel: Any, masses: Any, name: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    p = pos.to_numpy() if isinstance(pos, _Tensor) else np.asarray(pos, dtype=np.float64)
    v = vel.to_numpy() if isinstance(vel, _Tensor) else np.asarray(vel, dtype=np.float64)
    m = masses.to_numpy() if isinstance(masses, _Tensor) else np.asarray(masses, dtype=np.float64)
    p = np.asarray(p, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    m = np.asarray(m, dtype=np.float64).ravel()
    if p.ndim != 2 or v.ndim != 2 or p.shape != v.shape:
        raise ValueError(f"{name}: pos/vel must be matching (n, d) arrays, got {p.shape} and {v.shape}")
    if m.shape[0] != p.shape[0]:
        raise ValueError(f"{name}: masses length {m.shape[0]} must match {p.shape[0]} bodies")
    if p.shape[0] < 1 or p.shape[1] < 1:
        raise ValueError(f"{name}: need at least one body in at least 1-D, got shape {p.shape}")
    if bool(np.any(~np.isfinite(p))) or bool(np.any(~np.isfinite(v))):
        raise ValueError(f"{name}: pos/vel must be finite")
    if bool(np.any(~np.isfinite(m))) or bool(np.any(m < 0.0)):
        raise ValueError(f"{name}: masses must be finite and non-negative")
    return p.copy(), v.copy(), m.copy()


def _nbody_accel(p: np.ndarray, m: np.ndarray, g: float, soft2: float) -> np.ndarray:
    diff = p[None, :, :] - p[:, None, :]
    dist2 = np.sum(diff * diff, axis=2) + soft2
    off_diag = dist2[~np.eye(dist2.shape[0], dtype=bool)]
    if off_diag.shape[0] > 0 and bool(np.any(off_diag == 0.0)):
        raise ValueError("simulate_nbody: coincident bodies with softening=0; use softening>0")
    np.fill_diagonal(dist2, np.inf)
    inv3 = dist2 ** -1.5
    w = inv3 * m[None, :]
    return g * np.sum(diff * w[:, :, None], axis=1)


def nbody_energy(pos: Any, vel: Any, masses: Any, g: float = 1.0, softening: float = 0.0) -> float:
    """Total (kinetic + potential) energy of a gravitational N-body state."""
    p, v, m = _as_bodies(pos, vel, masses, "nbody_energy")
    g = _check_positive(g, "G", "nbody_energy")
    softening = _check_positive(softening, "softening", "nbody_energy", allow_zero=True)
    ke = 0.5 * float(np.sum(m * np.sum(v * v, axis=1)))
    diff = p[None, :, :] - p[:, None, :]
    dist = np.sqrt(np.sum(diff * diff, axis=2) + softening * softening)
    i, j = np.triu_indices(p.shape[0], k=1)
    if i.shape[0] > 0 and bool(np.any(dist[i, j] == 0.0)):
        raise ValueError("nbody_energy: coincident bodies with softening=0 have infinite potential; use softening>0")
    pe = -g * float(np.sum(m[i] * m[j] / dist[i, j])) if i.shape[0] > 0 else 0.0
    return ke + pe


def nbody_momentum(pos: Any, vel: Any, masses: Any) -> np.ndarray:
    """Total linear momentum vector of an N-body state."""
    _, v, m = _as_bodies(pos, vel, masses, "nbody_momentum")
    return np.sum(m[:, None] * v, axis=0)


def simulate_nbody(
    pos: Any,
    vel: Any,
    masses: Any,
    dt: float,
    num_steps: int,
    g: float = 1.0,
    softening: float = 0.0,
    method: str = "verlet",
) -> Tuple[np.ndarray, _Tensor, _Tensor]:
    """Gravitational N-body integration ('verlet' symplectic, or 'rk4').

    Returns `(times, positions, velocities)` with trajectories of shape
    (num_steps + 1, n, d).
    """
    p, v, m = _as_bodies(pos, vel, masses, "simulate_nbody")
    dt = _check_positive(dt, "dt", "simulate_nbody")
    num_steps = _check_num_steps(num_steps, "simulate_nbody")
    g = _check_positive(g, "G", "simulate_nbody")
    softening = _check_positive(softening, "softening", "simulate_nbody", allow_zero=True)
    if method not in ("verlet", "rk4"):
        raise ValueError(f"simulate_nbody: method must be 'verlet' or 'rk4', got {method!r}")
    soft2 = softening * softening
    n, d = p.shape
    t_hist = [0.0]
    p_hist = [p.copy()]
    v_hist = [v.copy()]
    if method == "verlet":
        acc = _nbody_accel(p, m, g, soft2)
        for _ in range(num_steps):
            v = v + acc * (dt / 2.0)
            p = p + v * dt
            acc = _nbody_accel(p, m, g, soft2)
            v = v + acc * (dt / 2.0)
            if bool(np.any(~np.isfinite(p))) or bool(np.any(~np.isfinite(v))):
                raise ValueError("simulate_nbody: close encounter produced non-finite state; use softening>0")
            t_hist.append(t_hist[-1] + dt)
            p_hist.append(p.copy())
            v_hist.append(v.copy())
    else:
        state = np.concatenate([p.ravel(), v.ravel()])

        def deriv(_t: float, y: np.ndarray) -> np.ndarray:
            pp = y[: n * d].reshape(n, d)
            vv = y[n * d :].reshape(n, d)
            return np.concatenate([vv.ravel(), _nbody_accel(pp, m, g, soft2).ravel()])

        t_cur = 0.0
        for _ in range(num_steps):
            state = RungeKutta4.step(deriv, t_cur, state, dt)
            if bool(np.any(~np.isfinite(state))):
                raise ValueError("simulate_nbody: close encounter produced non-finite state; use softening>0")
            t_cur += dt
            t_hist.append(t_cur)
            p_hist.append(state[: n * d].reshape(n, d).copy())
            v_hist.append(state[n * d :].reshape(n, d).copy())
    t_arr = np.array(t_hist, dtype=np.float64)
    p_arr = np.array(p_hist, dtype=np.float32)
    v_arr = np.array(v_hist, dtype=np.float32)
    return t_arr, _from_numpy(p_arr), _from_numpy(v_arr)


def hamiltonian_leapfrog(
    grad_potential: Callable[[np.ndarray], np.ndarray],
    q0: Any,
    p0: Any,
    mass: Union[float, np.ndarray] = 1.0,
    dt: float = 0.01,
    num_steps: int = 100,
) -> Tuple[np.ndarray, _Tensor, _Tensor]:
    """Kick-drift-kick leapfrog for separable H(q, p) = |p|^2/2m + U(q).

    `grad_potential(q)` returns dU/dq with q's shape. Returns `(times,
    positions, momenta)`.
    """
    q = q0.to_numpy() if isinstance(q0, _Tensor) else np.asarray(q0, dtype=np.float64)
    pp = p0.to_numpy() if isinstance(p0, _Tensor) else np.asarray(p0, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64).copy()
    pp = np.asarray(pp, dtype=np.float64).copy()
    if q.shape != pp.shape or q.ndim != 1 or q.shape[0] < 1:
        raise ValueError(f"hamiltonian_leapfrog: q0/p0 must be matching 1-D arrays, got {q.shape} and {pp.shape}")
    if bool(np.any(~np.isfinite(q))) or bool(np.any(~np.isfinite(pp))):
        raise ValueError("hamiltonian_leapfrog: q0/p0 must be finite")
    try:
        m_arr = np.broadcast_to(np.asarray(mass, dtype=np.float64), q.shape).copy()
    except ValueError as exc:
        raise ValueError(f"hamiltonian_leapfrog: mass {mass!r} cannot broadcast to {q.shape}") from exc
    if bool(np.any(~np.isfinite(m_arr))) or bool(np.any(m_arr <= 0.0)):
        raise ValueError("hamiltonian_leapfrog: mass must be finite and positive")
    dt = _check_positive(dt, "dt", "hamiltonian_leapfrog")
    num_steps = _check_num_steps(num_steps, "hamiltonian_leapfrog")

    def grad(qq: np.ndarray) -> np.ndarray:
        try:
            g = np.asarray(grad_potential(qq), dtype=np.float64)
        except Exception as exc:
            raise ValueError(f"hamiltonian_leapfrog: grad_potential failed ({exc})") from exc
        if g.shape != qq.shape:
            raise ValueError(f"hamiltonian_leapfrog: grad_potential must preserve shape {qq.shape}, got {g.shape}")
        if bool(np.any(~np.isfinite(g))):
            raise ValueError("hamiltonian_leapfrog: grad_potential returned non-finite values")
        return g

    t_hist = [0.0]
    q_hist = [q.copy()]
    p_hist = [pp.copy()]
    for _ in range(num_steps):
        pp = pp - grad(q) * (dt / 2.0)
        q = q + (pp / m_arr) * dt
        pp = pp - grad(q) * (dt / 2.0)
        t_hist.append(t_hist[-1] + dt)
        q_hist.append(q.copy())
        p_hist.append(pp.copy())
    return (np.array(t_hist, dtype=np.float64),
            _from_numpy(np.array(q_hist, dtype=np.float32)),
            _from_numpy(np.array(p_hist, dtype=np.float32)))


def circular_velocity(m_central: float, r: float, g: float = 1.0) -> float:
    """Circular orbit speed at radius `r` around mass `m_central` (G = 1 default)."""
    m_central = _check_positive(m_central, "m_central", "circular_velocity")
    r = _check_positive(r, "r", "circular_velocity")
    g = _check_positive(g, "G", "circular_velocity")
    return math.sqrt(g * m_central / r)


def orbital_period(a: float, m_central: float, g: float = 1.0) -> float:
    """Kepler period for semi-major axis `a` around mass `m_central`."""
    a = _check_positive(a, "a", "orbital_period")
    m_central = _check_positive(m_central, "m_central", "orbital_period")
    g = _check_positive(g, "G", "orbital_period")
    return 2.0 * math.pi * math.sqrt(a ** 3 / (g * m_central))
