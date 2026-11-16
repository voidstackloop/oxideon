"""Tests for oxideon.physics: 2-D fields, N-body gravity, leapfrog mechanics.

Oracles are analytic identities, conservation laws, and cross-checks
against the 1-D solvers in oxideon.ode_pde_solvers.
"""

import math

import numpy as np
import pytest

from oxideon.ode_pde_solvers import Diffusion1DEquation, Wave1DEquation
from oxideon.physics import (
    Diffusion2DEquation,
    Wave2DEquation,
    circular_velocity,
    hamiltonian_leapfrog,
    nbody_energy,
    nbody_momentum,
    orbital_period,
    simulate_nbody,
)


# --- 2-D wave ---------------------------------------------------------------


def test_wave2d_matches_wave1d_interior():
    nx = ny = 41
    f = np.sin(np.pi * np.linspace(0.0, 1.0, nx))
    u0 = np.tile(f[None, :], (ny, 1))
    out2 = Wave2DEquation.solve(u0, c=1.0, dx=0.05, dy=0.05, dt=0.01, num_steps=5).to_numpy()
    out1 = Wave1DEquation.solve(f, c=1.0, dx=0.05, dt=0.01, num_steps=5).to_numpy()
    # y-edge waves travel ~1 cell in 5 steps; the center row is 20 away.
    np.testing.assert_allclose(out2[20], out1, rtol=1e-5, atol=1e-7)
    assert bool(np.all(out2[0] == 0.0)) and bool(np.all(out2[:, 0] == 0.0))


def test_wave2d_v0_and_validation():
    u0 = np.zeros((11, 13))
    u0[5, 6] = 1.0
    out = Wave2DEquation.solve(u0, v0=np.zeros((11, 13)), num_steps=3).to_numpy()
    assert out.shape == (11, 13) and bool(np.all(np.isfinite(out)))
    with pytest.raises(ValueError, match="CFL"):
        Wave2DEquation.solve(u0, c=1.0, dx=0.05, dy=0.05, dt=0.05, num_steps=1)
    with pytest.raises(ValueError, match="2-D"):
        Wave2DEquation.solve(np.zeros(10))
    with pytest.raises(ValueError, match="v0 shape"):
        Wave2DEquation.solve(u0, v0=np.zeros((5, 5)))
    with pytest.raises(ValueError, match="positive"):
        Wave2DEquation.solve(u0, c=-1.0)


# --- 2-D diffusion -----------------------------------------------------------


def test_diffusion2d_mass_and_spreading():
    n = 61
    x = (np.arange(n) - n // 2) * 0.1
    X, Y = np.meshgrid(x, x)
    s0 = 0.5
    u0 = np.exp(-(X**2 + Y**2) / (2 * s0**2))
    alpha, dt, steps = 0.1, 0.02, 100
    out = Diffusion2DEquation.solve(u0, alpha=alpha, dx=0.1, dy=0.1, dt=dt, num_steps=steps).to_numpy()
    assert float(out.sum()) == pytest.approx(float(u0.sum()), rel=1e-3)
    marg = out.sum(axis=0)
    marg /= marg.sum()
    mean = float(np.sum(marg * x))
    var = float(np.sum(marg * (x - mean) ** 2))
    assert var == pytest.approx(s0**2 + 2 * alpha * dt * steps, rel=0.05)


def test_diffusion2d_matches_diffusion1d():
    f = np.sin(np.pi * np.linspace(0.0, 1.0, 31)) ** 2 + 0.5
    u0 = np.tile(f[None, :], (31, 1))
    out2 = Diffusion2DEquation.solve(u0, alpha=0.05, dx=0.05, dy=0.05, dt=0.01, num_steps=10).to_numpy()
    out1 = Diffusion1DEquation.solve(f, alpha=0.05, dx=0.05, dt=0.01, num_steps=10).to_numpy()
    np.testing.assert_allclose(out2[15], out1, rtol=1e-5, atol=1e-7)


def test_diffusion2d_validation():
    u0 = np.zeros((9, 9))
    with pytest.raises(ValueError, match="stability"):
        Diffusion2DEquation.solve(u0, alpha=1.0, dx=0.05, dy=0.05, dt=0.01, num_steps=1)
    with pytest.raises(ValueError, match="2-D"):
        Diffusion2DEquation.solve(np.zeros((2, 2)))
    with pytest.raises(ValueError, match="positive"):
        Diffusion2DEquation.solve(u0, alpha=0.0)


# --- N-body ------------------------------------------------------------------


def test_nbody_accel_two_body_analytic():
    from oxideon.physics import _nbody_accel

    p = np.array([[0.0, 0.0], [3.0, 4.0]])
    m = np.array([2.0, 5.0])
    a = _nbody_accel(p, m, 1.0, 0.0)
    np.testing.assert_allclose(a[0], [0.12, 0.16], rtol=1e-12)
    np.testing.assert_allclose(a[1], [-0.048, -0.064], rtol=1e-12)
    # n != d shape check + Newton's third law (mass-weighted sum vanishes).
    p3 = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 2.0]])
    a3 = _nbody_accel(p3, np.array([1.0, 2.0, 3.0]), 1.0, 0.0)
    assert a3.shape == (3, 2)
    np.testing.assert_allclose(a3.T @ np.array([1.0, 2.0, 3.0]), [0.0, 0.0], atol=1e-12)


def _circular_binary():
    m_c, m_o, r = 100.0, 1.0, 10.0
    v_rel = math.sqrt((m_c + m_o) / r)
    pos = np.array([[-m_o * r / (m_c + m_o), 0.0], [m_c * r / (m_c + m_o), 0.0]])
    vel = np.array([[0.0, -v_rel * m_o / (m_c + m_o)], [0.0, v_rel * m_c / (m_c + m_o)]])
    period = 2.0 * math.pi * math.sqrt(r**3 / (m_c + m_o))
    return pos, vel, np.array([m_c, m_o]), period


def test_nbody_circular_orbit_closes():
    pos, vel, masses, period = _circular_binary()
    steps = 4000
    dt = period / steps
    t, p_traj, v_traj = simulate_nbody(pos, vel, masses, dt, steps, method="verlet")
    p = p_traj.to_numpy()
    assert p.shape == (steps + 1, 2, 2)
    sep0 = pos[1] - pos[0]
    sep1 = p[-1, 1] - p[-1, 0]
    assert float(np.linalg.norm(sep1 - sep0)) < 5e-3
    e0 = nbody_energy(pos, vel, masses)
    e1 = nbody_energy(p[-1], v_traj.to_numpy()[-1], masses)
    assert abs(e1 - e0) / abs(e0) < 1e-4
    mom0 = nbody_momentum(pos, vel, masses)
    mom1 = nbody_momentum(p[-1], v_traj.to_numpy()[-1], masses)
    np.testing.assert_allclose(mom1, mom0, atol=1e-6)
    assert float(t[-1]) == pytest.approx(period, rel=1e-9)


def test_nbody_rk4_single_body_and_softening():
    pos, vel, masses, period = _circular_binary()
    steps = 1000
    _, p_traj, _ = simulate_nbody(pos, vel, masses, period / steps, steps, method="rk4")
    p = p_traj.to_numpy()
    assert float(np.linalg.norm((p[-1, 1] - p[-1, 0]) - (pos[1] - pos[0]))) < 1e-3
    _, ps, _ = simulate_nbody(np.array([[0.0, 0.0]]), np.array([[1.0, 2.0]]),
                              np.array([5.0]), 0.1, 10, method="verlet")
    np.testing.assert_allclose(ps.to_numpy()[-1, 0], [1.0, 2.0], rtol=1e-6)
    t, p_soft, _ = simulate_nbody(pos, vel, masses, period / 4000, 4000,
                                  softening=0.1, method="verlet")
    assert bool(np.all(np.isfinite(p_soft.to_numpy())))
    e0 = nbody_energy(pos, vel, masses, softening=0.1)
    e1 = nbody_energy(p_soft.to_numpy()[-1], np.zeros((2, 2)), masses, softening=0.1)
    assert math.isfinite(e0) and math.isfinite(e1)


def test_nbody_collision_guards_and_validation():
    with pytest.raises(ValueError, match="coincident"):
        simulate_nbody(np.array([[0.0, 0.0], [0.0, 0.0]]), np.zeros((2, 2)),
                       np.array([1.0, 1.0]), 0.001, 10, method="verlet")
    with pytest.raises(ValueError, match="coincident"):
        simulate_nbody(np.array([[0.0, 0.0], [0.0, 0.0]]), np.zeros((2, 2)),
                       np.array([1.0, 1.0]), 0.001, 10, method="rk4")
    with pytest.raises(ValueError, match="coincident"):
        nbody_energy(np.array([[0.0, 0.0], [0.0, 0.0]]), np.zeros((2, 2)), np.array([1.0, 1.0]))
    with pytest.raises(ValueError, match="matching"):
        simulate_nbody(np.zeros((2, 2)), np.zeros((3, 2)), np.ones(2), 0.01, 5)
    with pytest.raises(ValueError, match="non-negative"):
        simulate_nbody(np.zeros((2, 2)), np.zeros((2, 2)), np.array([1.0, -1.0]), 0.01, 5)
    with pytest.raises(ValueError, match="method"):
        simulate_nbody(np.zeros((2, 2)), np.zeros((2, 2)), np.ones(2), 0.01, 5, method="euler")


# --- leapfrog ------------------------------------------------------------------


def test_leapfrog_harmonic_oscillator():
    dt, steps = 0.05, 500
    t, q, p = hamiltonian_leapfrog(lambda x: x, np.array([1.0]), np.array([0.0]),
                                   mass=1.0, dt=dt, num_steps=steps)
    qq = q.to_numpy().ravel()
    pp = p.to_numpy().ravel()
    np.testing.assert_allclose(qq, np.cos(t), atol=0.02)
    energy = 0.5 * (qq**2 + pp**2)
    assert bool(np.all(np.abs(energy - 0.5) < 5e-3))
    assert abs(energy[-1] - energy[0]) < 1e-3


def test_leapfrog_pendulum_trapped_and_validation():
    t, q, _ = hamiltonian_leapfrog(np.sin, np.array([0.3]), np.array([0.0]), dt=0.02, num_steps=1000)
    qq = q.to_numpy().ravel()
    assert bool(np.all(np.abs(qq) < 0.35))
    crossings = int(np.sum(qq[:-1] * qq[1:] < 0.0))
    assert crossings >= 6
    with pytest.raises(ValueError, match="matching"):
        hamiltonian_leapfrog(lambda x: x, np.array([1.0]), np.array([0.0, 0.0]))
    with pytest.raises(ValueError, match="mass"):
        hamiltonian_leapfrog(lambda x: x, np.array([1.0]), np.array([0.0]), mass=0.0)
    with pytest.raises(ValueError, match="grad_potential"):
        hamiltonian_leapfrog(lambda x: np.zeros(2), np.array([1.0]), np.array([0.0]), num_steps=2)


# --- Kepler helpers --------------------------------------------------------------


def test_kepler_identities_and_validation():
    m, r = 100.0, 10.0
    v = circular_velocity(m, r)
    assert v**2 == pytest.approx(m / r, rel=1e-12)
    period = orbital_period(r, m)
    assert v == pytest.approx(2.0 * math.pi * r / period, rel=1e-12)
    assert period**2 == pytest.approx(4.0 * math.pi**2 * r**3 / m, rel=1e-12)
    with pytest.raises(ValueError, match="positive"):
        circular_velocity(m, 0.0)
    with pytest.raises(ValueError, match="positive"):
        orbital_period(-1.0, m)
