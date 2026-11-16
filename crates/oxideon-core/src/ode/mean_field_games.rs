//! Mean-Field Games (MFG): Coupled Backward Hamilton-Jacobi-Bellman (HJB) &
//! Forward Fokker-Planck-Kolmogorov (FPK) Nash Equilibrium Solvers (Lasry & Lions, 2006).
//!
//! ### 1. Lasry-Lions Mean-Field Game System
//! In a continuum of rational agents over a spatial graph or 1D periodic ring, the Nash equilibrium
//! is governed by the forward-backward coupled system:
//! • **Backward Hamilton-Jacobi-Bellman (HJB)** for value function $u(x, t)$:
//! $$-\partial_t u - \nu \Delta u + \frac{1}{2}|\nabla u|^2 = V(x) + \gamma_{\text{cong}} \, m(x, t)$$
//! • **Forward Fokker-Planck-Kolmogorov (FPK)** for agent density $m(x, t)$ under optimal feedback control $\alpha^*(x, t) = -\nabla u(x, t)$:
//! $$\partial_t m - \nu \Delta m - \operatorname{div}(m \nabla u) = 0$$
//!
//! ### 2. Cole-Hopf Transformation & Crowd Congestion Aversion
//! For quadratic Hamiltonian $H(p) = \frac{1}{2}|p|^2$ and viscosity $\nu > 0$, the Cole-Hopf variable
//! $\Phi(x, t) = \exp\left(-\frac{u(x, t)}{2\nu}\right)$ linearizes the HJB step into a reaction-diffusion equation.
//! Positive congestion coupling $\gamma_{\text{cong}} > 0$ penalizes high-density regions, driving agents
//! to disperse toward low-cost basins.

use crate::error::{Error, Result};

/// Coupled 1D Forward-Backward Mean-Field Game (HJB-FPK) Solver on a periodic domain.
#[derive(Debug, Clone)]
pub struct MeanFieldGameSolver1D {
    /// Number of spatial grid nodes $N$.
    pub num_nodes: usize,
    /// Viscosity / idiosyncratic diffusion coefficient $\nu > 0$.
    pub viscosity: f64,
    /// Crowd congestion aversion coupling $\gamma_{\text{cong}} \ge 0$.
    pub congestion_coupling: f64,
    /// Spatial potential landscape $V(x)$ of length $N$.
    pub potential: Vec<f64>,
}

impl MeanFieldGameSolver1D {
    /// Constructs a 1D Mean-Field Game solver.
    pub fn new(
        num_nodes: usize,
        viscosity: f64,
        congestion_coupling: f64,
        potential: Vec<f64>,
    ) -> Result<Self> {
        if num_nodes < 3 {
            return Err(Error::Msg("num_nodes must be >= 3".to_string()));
        }
        if !viscosity.is_finite() || viscosity <= 0.0 {
            return Err(Error::Msg(format!(
                "viscosity must be finite and > 0, got {viscosity}"
            )));
        }
        if potential.len() != num_nodes {
            return Err(Error::Msg(
                "potential length must equal num_nodes".to_string(),
            ));
        }
        Ok(Self {
            num_nodes,
            viscosity,
            congestion_coupling,
            potential,
        })
    }

    /// Performs one backward HJB step from $u(t + \Delta t)$ to $u(t)$ given current population density $m(t)$
    /// and terminal/next value $u_{\text{next}}$, returning `(u_curr, optimal_drift_alpha)`.
    pub fn backward_hjb_step(
        &self,
        u_next: &[f64],
        m_curr: &[f64],
        dt: f64,
        dx: f64,
    ) -> Result<(Vec<f64>, Vec<f64>)> {
        let n = self.num_nodes;
        if u_next.len() != n || m_curr.len() != n {
            return Err(Error::Msg(
                "State vector length mismatch in backward_hjb_step".to_string(),
            ));
        }
        if !dx.is_finite() || dx <= 0.0 {
            return Err(Error::Msg(format!(
                "backward_hjb_step requires finite dx > 0, got {dx}"
            )));
        }
        if !dt.is_finite() || dt < 0.0 {
            return Err(Error::Msg(format!(
                "backward_hjb_step requires finite dt >= 0, got {dt}"
            )));
        }
        if self.potential.len() != n {
            return Err(Error::Msg(format!(
                "backward_hjb_step potential length {} disagrees with {n} nodes",
                self.potential.len()
            )));
        }
        let mut u_curr = vec![0.0; n];
        let mut alpha = vec![0.0; n];

        for i in 0..n {
            let ip = (i + 1) % n;
            let im = (i + n - 1) % n;

            let du_dx = (u_next[ip] - u_next[im]) / (2.0 * dx);
            let lap_u = (u_next[ip] - 2.0 * u_next[i] + u_next[im]) / (dx * dx);
            let running_cost = self.potential[i] + self.congestion_coupling * m_curr[i];

            // Backward in time: u(t) = u(t + dt) + dt * (nu * Lap(u) - 0.5 * |du/dx|^2 + V + gamma * m)
            u_curr[i] =
                u_next[i] + dt * (self.viscosity * lap_u - 0.5 * du_dx * du_dx + running_cost);
            // Optimal control alpha* = -du/dx
            alpha[i] = -du_dx;
        }
        Ok((u_curr, alpha))
    }

    /// Performs one forward Fokker-Planck-Kolmogorov (FPK) step evolving density $m(t)$ to $m(t + \Delta t)$
    /// under drift $\alpha(x)$ while strictly conserving total probability mass $\sum_i m_i = 1$.
    pub fn forward_fpk_step(
        &self,
        m_curr: &[f64],
        alpha: &[f64],
        dt: f64,
        dx: f64,
    ) -> Result<Vec<f64>> {
        let n = self.num_nodes;
        if m_curr.len() != n || alpha.len() != n {
            return Err(Error::Msg(
                "Vector length mismatch in forward_fpk_step".to_string(),
            ));
        }
        if !dx.is_finite() || dx <= 0.0 {
            return Err(Error::Msg(format!(
                "forward_fpk_step requires finite dx > 0, got {dx}"
            )));
        }
        if !dt.is_finite() || dt < 0.0 {
            return Err(Error::Msg(format!(
                "forward_fpk_step requires finite dt >= 0, got {dt}"
            )));
        }
        let mut m_next = vec![0.0; n];
        for i in 0..n {
            let ip = (i + 1) % n;
            let im = (i + n - 1) % n;

            let lap_m = (m_curr[ip] - 2.0 * m_curr[i] + m_curr[im]) / (dx * dx);
            let div_flux = (m_curr[ip] * alpha[ip] - m_curr[im] * alpha[im]) / (2.0 * dx);

            m_next[i] = (m_curr[i] + dt * (self.viscosity * lap_m - div_flux)).max(1e-12);
        }
        let sum: f64 = m_next.iter().sum();
        let target_sum: f64 = m_curr.iter().sum();
        if sum > 0.0 {
            for val in &mut m_next {
                *val *= target_sum / sum;
            }
        }
        Ok(m_next)
    }

    /// Solves for the stationary / fixed-point Mean-Field Nash Equilibrium over `num_iters` forward-backward sweeps,
    /// returning `(equilibrium_density_m, equilibrium_value_u, nash_social_cost)`.
    pub fn solve_nash_equilibrium(
        &self,
        m_init: &[f64],
        u_terminal: &[f64],
        dt: f64,
        dx: f64,
        num_iters: usize,
    ) -> Result<(Vec<f64>, Vec<f64>, f64)> {
        let n = self.num_nodes;
        if m_init.len() != n || u_terminal.len() != n {
            return Err(Error::Msg(
                "Initial/terminal vector length mismatch".to_string(),
            ));
        }
        crate::limits::check_iterations("num_iters", num_iters)?;

        let mut m = m_init.to_vec();
        let mut u = u_terminal.to_vec();

        for _ in 0..num_iters {
            let (u_new, alpha) = self.backward_hjb_step(&u, &m, dt, dx)?;
            let m_new = self.forward_fpk_step(&m, &alpha, dt, dx)?;

            // Damped fixed-point update for stability
            for (u_i, &un_i) in u.iter_mut().zip(u_new.iter()) {
                *u_i = 0.5 * (*u_i) + 0.5 * un_i;
            }
            for (m_i, &mn_i) in m.iter_mut().zip(m_new.iter()) {
                *m_i = 0.5 * (*m_i) + 0.5 * mn_i;
            }
        }

        let mut social_cost = 0.0;
        for (i, &m_i) in m.iter().enumerate().take(n) {
            social_cost += m_i * (self.potential[i] + 0.5 * self.congestion_coupling * m_i);
        }
        Ok((m, u, social_cost))
    }
}
