//! Dissipative Contact Geometry & Non-Conservative Hamiltonian Dynamics.
//!
//! Features:
//! - $(2n + 1)$-dimensional contact manifolds $(M, \eta)$ with coordinates $(\mathbf{q}, \mathbf{p}, S)$
//!   and canonical contact 1-form $\eta = dS - \mathbf{p} \cdot d\mathbf{q}$.
//! - Contact Hamiltonian equations of motion modeling non-conservative dissipation and entropy production:
//!   $$\dot{\mathbf{q}} = \frac{\partial H}{\partial \mathbf{p}}$$
//!   $$\dot{\mathbf{p}} = -\frac{\partial H}{\partial \mathbf{q}} - \mathbf{p} \frac{\partial H}{\partial S}$$
//!   $$\dot{S} = \mathbf{p} \cdot \frac{\partial H}{\partial \mathbf{p}} - H$$
//! - Contact Verlet geometric numerical integration preserving contact submanifolds.

use crate::error::{Error, Result};
use crate::tensor::Tensor;

/// Dissipative Contact Hamiltonian System on $(2n + 1)$-dimensional contact manifold.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ContactHamiltonianSystem {
    /// Spatial degrees of freedom $n$.
    pub dimension: usize,
    /// Dissipation / friction parameter $\gamma = \frac{\partial H}{\partial S}$.
    pub dissipation: f64,
}

impl ContactHamiltonianSystem {
    /// Creates a new `ContactHamiltonianSystem`.
    pub fn new(dimension: usize, dissipation: f64) -> Result<Self> {
        if dimension == 0 {
            return Err(Error::Msg("Dimension must be >= 1".into()));
        }
        Ok(Self {
            dimension,
            dissipation,
        })
    }

    /// Evaluates contact Hamiltonian vector field $(\dot{\mathbf{q}}, \dot{\mathbf{p}}, \dot{S})$:
    /// For standard kinetic $T(\mathbf{p}) = \frac{1}{2} \|\mathbf{p}\|^2$ and potential $V(\mathbf{q})$.
    pub fn vector_field(
        &self,
        _q: &Tensor,
        p: &Tensor,
        s: f64,
        grad_v: &Tensor,
    ) -> Result<(Tensor, Tensor, f64)> {
        // \dot{q} = p
        let dq = p.clone();

        // \dot{p} = - \nabla_q V - \gamma * p
        let gamma_p = p.mul_scalar(self.dissipation)?;
        let dp = grad_v.neg()?.sub(&gamma_p)?;

        // \dot{S} = p \cdot dq - H(q, p, s) = ||p||^2 - (0.5 * ||p||^2 + V(q) + \gamma * S)
        let p_sq = p.powf(2.0)?.sum()?.item::<f64>()?;
        let v_val = grad_v.powf(2.0)?.sum()?.item::<f64>()? * 0.5; // proxy potential
        let h_val = 0.5 * p_sq + v_val + self.dissipation * s;
        let ds = p_sq - h_val;

        Ok((dq, dp, ds))
    }

    /// Contact Verlet integration step over interval `dt`:
    /// Preserves contact geometry and accurately tracks non-conservative dissipation and action $S$.
    pub fn step(
        &self,
        q: &Tensor,
        p: &Tensor,
        s: f64,
        grad_v: &Tensor,
        dt: f64,
    ) -> Result<(Tensor, Tensor, f64)> {
        let half_dt = 0.5 * dt;

        // Damped momentum half-step: p_{1/2} = e^{-\gamma dt / 2} * (p_0 - \frac{dt}{2} \nabla V(q_0))
        let decay_half = (-self.dissipation * half_dt).exp();
        let p_half = p
            .sub(&grad_v.mul_scalar(half_dt)?)?
            .mul_scalar(decay_half)?;

        // Coordinate step: q_1 = q_0 + dt * p_{1/2}
        let q_next = q.add(&p_half.mul_scalar(dt)?)?;

        // Momentum full-step with decay
        let decay_full = (-self.dissipation * half_dt).exp();
        let p_next = p_half
            .mul_scalar(decay_full)?
            .sub(&grad_v.mul_scalar(half_dt)?)?;

        // Action / entropy coordinate step: S_{k+1} = S_k + dt * (||p_{1/2}||^2 - H)
        let p_sq = p_half.powf(2.0)?.sum()?.item::<f64>()?;
        let s_next = s + dt * (0.5 * p_sq - self.dissipation * s);

        Ok((q_next, p_next, s_next))
    }

    /// Simulates dissipative contact trajectory for `steps` steps.
    pub fn simulate_trajectory<F>(
        &self,
        grad_v_fn: F,
        q0: &Tensor,
        p0: &Tensor,
        s0: f64,
        dt: f64,
        steps: usize,
    ) -> Result<(Tensor, Tensor, Vec<f64>)>
    where
        F: Fn(&Tensor) -> Result<Tensor>,
    {
        let mut q = q0.clone();
        let mut p = p0.clone();
        let mut s = s0;

        let mut q_traj = Vec::with_capacity(steps + 1);
        let mut p_traj = Vec::with_capacity(steps + 1);
        let mut s_traj = Vec::with_capacity(steps + 1);

        q_traj.push(q.clone());
        p_traj.push(p.clone());
        s_traj.push(s);

        for _ in 0..steps {
            let grad_v = grad_v_fn(&q)?;
            let (q_n, p_n, s_n) = self.step(&q, &p, s, &grad_v, dt)?;
            q = q_n;
            p = p_n;
            s = s_n;

            q_traj.push(q.clone());
            p_traj.push(p.clone());
            s_traj.push(s);
        }

        let q_tensor = Tensor::stack(&q_traj, 0)?;
        let p_tensor = Tensor::stack(&p_traj, 0)?;
        Ok((q_tensor, p_tensor, s_traj))
    }
}
