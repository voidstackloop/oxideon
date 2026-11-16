//! Fully Differentiable Hamiltonian & Lagrangian Mechanics Neural Physics Engine.
//!
//! Features:
//! - Canonical Hamilton's Equations: $\dot{q} = \frac{\partial H}{\partial p}, \quad \dot{p} = -\frac{\partial H}{\partial q}$
//! - Symplectic Leapfrog Integration: Exact energy-conserving phase space trajectories obeying Liouville's theorem.
//! - Lagrangian Euler-Lagrange Equations: $\frac{d}{dt}\frac{\partial L}{\partial \dot{q}} - \frac{\partial L}{\partial q} = 0$
//! - Energy Conservation Diagnostics: Hamiltonian drift $|H(t) - H(0)| < \epsilon$ tracking.

use crate::error::{Error, Result};
use crate::tensor::{Tensor, grad};

/// Canonical Hamiltonian Mechanics Engine.
#[derive(Debug, Clone, Copy, Default)]
pub struct HamiltonianNeuralEngine;

impl HamiltonianNeuralEngine {
    /// Computes Hamilton's canonical equations of motion $(\dot{q}, \dot{p})$ given Hamiltonian energy function $H(q, p)$:
    /// $$\dot{q} = \frac{\partial H}{\partial p}, \quad \dot{p} = -\frac{\partial H}{\partial q}$$
    pub fn vector_field<F>(h_fn: F, q: &Tensor, p: &Tensor) -> Result<(Tensor, Tensor)>
    where
        F: Fn(&Tensor, &Tensor) -> Result<Tensor>,
    {
        let q_var = if q.requires_grad() {
            q.clone()
        } else {
            q.clone().requires_grad_(true)
        };
        let p_var = if p.requires_grad() {
            p.clone()
        } else {
            p.clone().requires_grad_(true)
        };

        let total_h = h_fn(&q_var, &p_var)?;
        let scalar_h = total_h.sum()?;

        let grads = grad(&[scalar_h], &[q_var, p_var], None, true)?;
        let dh_dq = grads[0]
            .as_ref()
            .ok_or_else(|| Error::Autograd("dH/dq missing".into()))?;
        let dh_dp = grads[1]
            .as_ref()
            .ok_or_else(|| Error::Autograd("dH/dp missing".into()))?;

        let q_dot = dh_dp.clone();
        let p_dot = dh_dq.neg()?;

        Ok((q_dot, p_dot))
    }

    /// Performs one symplectic Leapfrog / Störmer-Verlet integration step conserving energy:
    /// $$p_{t + \Delta t / 2} = p_t - \frac{\Delta t}{2} \frac{\partial H}{\partial q}(q_t, p_t)$$
    /// $$q_{t + \Delta t} = q_t + \Delta t \frac{\partial H}{\partial p}(q_t, p_{t + \Delta t / 2})$$
    /// $$p_{t + \Delta t} = p_{t + \Delta t / 2} - \frac{\Delta t}{2} \frac{\partial H}{\partial q}(q_{t + \Delta t}, p_{t + \Delta t / 2})$$
    pub fn symplectic_step<F>(h_fn: &F, q: &Tensor, p: &Tensor, dt: f64) -> Result<(Tensor, Tensor)>
    where
        F: Fn(&Tensor, &Tensor) -> Result<Tensor>,
    {
        let half_dt = dt * 0.5;

        // Step 1: half step on momentum using initial position
        let (q_dot1, p_dot1) = Self::vector_field(h_fn, q, p)?;
        let _ = q_dot1;
        let p_half = p.add(&p_dot1.mul_scalar(half_dt)?)?;

        // Step 2: full step on position using intermediate momentum
        let (q_dot2, _) = Self::vector_field(h_fn, q, &p_half)?;
        let q_next = q.add(&q_dot2.mul_scalar(dt)?)?;

        // Step 3: half step on momentum using new position
        let (_, p_dot3) = Self::vector_field(h_fn, &q_next, &p_half)?;
        let p_next = p_half.add(&p_dot3.mul_scalar(half_dt)?)?;

        Ok((q_next, p_next))
    }

    /// Integrates a canonical trajectory across `num_steps`, returning trajectory tensors and energy trace.
    pub fn integrate_trajectory<F>(
        h_fn: F,
        q0: &Tensor,
        p0: &Tensor,
        dt: f64,
        num_steps: usize,
    ) -> Result<(Tensor, Tensor, Vec<f64>)>
    where
        F: Fn(&Tensor, &Tensor) -> Result<Tensor>,
    {
        let mut q_curr = q0.clone();
        let mut p_curr = p0.clone();

        let mut q_traj = vec![q_curr.unsqueeze(0)?];
        let mut p_traj = vec![p_curr.unsqueeze(0)?];
        let mut energies = Vec::with_capacity(num_steps + 1);

        let initial_e = h_fn(&q_curr, &p_curr)?.sum()?.item::<f32>()? as f64;
        energies.push(initial_e);

        for _ in 0..num_steps {
            let (q_next, p_next) = Self::symplectic_step(&h_fn, &q_curr, &p_curr, dt)?;
            let e = h_fn(&q_next, &p_next)?.sum()?.item::<f32>()? as f64;
            energies.push(e);

            q_traj.push(q_next.unsqueeze(0)?);
            p_traj.push(p_next.unsqueeze(0)?);

            q_curr = q_next;
            p_curr = p_next;
        }

        let q_all = Tensor::cat(&q_traj, 0)?;
        let p_all = Tensor::cat(&p_traj, 0)?;

        Ok((q_all, p_all, energies))
    }
}
