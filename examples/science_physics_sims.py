"""Physics simulators with ox.physics.

Covers the 2-D wave and diffusion PDE solvers, N-body gravity with a
circular-orbit energy check, and symplectic (leapfrog) mechanics.
Run: python examples/science_physics_sims.py
"""

import numpy as np

import oxideon as ox


def main() -> None:
    # --- 2-D wave equation: plucked membrane, fixed zero boundaries ---
    u0 = np.zeros((21, 21))
    u0[10, 10] = 1.0
    u = ox.physics.Wave2DEquation.solve(u0, c=1.0, dx=0.05, dy=0.05, dt=0.01, num_steps=5)
    arr = u.to_numpy()
    print(f"wave: shape = {arr.shape}, peak = {arr.max():.4f} (started at 1.0)")
    assert arr.shape == (21, 21) and 0.0 < arr.max() < 1.0

    # --- 2-D diffusion: insulated edges conserve mass, peak spreads ---
    v = ox.physics.Diffusion2DEquation.solve(
        u0, alpha=0.05, dx=0.05, dy=0.05, dt=0.01, num_steps=10
    ).to_numpy()
    print(f"diffusion: mass = {v.sum():.9f} (started at 1.0), peak = {v.max():.4f}")
    assert abs(v.sum() - 1.0) < 2e-4  # f32 solver; suite bar is rel=1e-3
    assert v.max() < 1.0

    # --- N-body: circular two-body orbit, one period, energy conserved ---
    m_central, r = 1000.0, 10.0
    vc = ox.physics.circular_velocity(m_central, r)
    period = ox.physics.orbital_period(r, m_central)
    print(f"orbit: v_circ = {vc:.6f}, period = {period:.6f}")
    pos = np.array([[0.0, 0.0], [r, 0.0]])
    vel = np.array([[0.0, 0.0], [0.0, vc]])
    masses = [m_central, 1.0]
    e0 = ox.physics.nbody_energy(pos, vel, masses)
    t, p_traj, v_traj = ox.physics.simulate_nbody(
        pos, vel, masses, dt=period / 400, num_steps=400, method="verlet"
    )
    p_last = p_traj.to_numpy()[-1]
    v_last = v_traj.to_numpy()[-1]
    e1 = ox.physics.nbody_energy(p_last, v_last, masses)
    print(f"nbody: E0 = {e0:.6f}, E1 = {e1:.6f}, |dE/E| = {abs(e1 - e0) / abs(e0):.2e}")
    print(f"nbody: orbit closure error = {np.linalg.norm(p_last[1] - pos[1]):.4f} (r = {r})")
    assert abs(e1 - e0) / abs(e0) < 1e-3
    assert np.linalg.norm(p_last[1] - pos[1]) < 0.5

    # --- Symplectic leapfrog: unit harmonic oscillator, q(1) ~ cos(1) ---
    tq, q_traj, p_traj = ox.physics.hamiltonian_leapfrog(
        grad_potential=lambda q: q, q0=[1.0], p0=[0.0], dt=0.01, num_steps=100
    )
    q1 = float(q_traj.to_numpy()[-1].ravel()[0])
    print(f"leapfrog: q(1) = {q1:.6f} (cos(1) = {np.cos(1.0):.6f})")
    assert abs(q1 - np.cos(1.0)) < 1e-3

    print("science_physics_sims: OK")


if __name__ == "__main__":
    main()
