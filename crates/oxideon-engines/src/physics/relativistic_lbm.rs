//! Relativistic Lattice Boltzmann Method (RLBM) & Anderson-Witting BGK Kinetic Transport.
//!
//! ### 1. Relativistic Boltzmann BGK Equation
//! The relativistic phase-space distribution $f(x^\mu, p^\mu)$ evolves according to the Anderson-Witting
//! Bhatnagar-Gross-Krook (BGK) kinetic equation:
//! $$p^\mu \partial_\mu f = -\frac{p_\mu u^\mu}{c^2 \tau_{\text{rel}}} \left(f - f^{\text{eq}}\right)$$
//! where $\tau_{\text{rel}}$ is the relaxation time and $f^{\text{eq}}$ is the local Jüttner equilibrium.
//!
//! ### 2. D1Q3 Relativistic Lattice Streaming & Collision
//! On a 1D periodic lattice with discrete velocities $c_0 = 0, c_{+1} = +1, c_{-1} = -1$ and weights
//! $w_0 = 2/3, w_{\pm 1} = 1/6$:
//! • Macroscopic density: $n(x) = f_0 + f_{+1} + f_{-1}$
//! • Macroscopic momentum flux: $J(x) = f_{+1} - f_{-1}$, velocity $u(x) = J(x) / n(x)$
//! • Equilibrium distributions:
//! $$f_i^{\text{eq}}(n, u) = w_i n \left(1 + 3 c_i u + \frac{9}{2} c_i^2 u^2 - \frac{3}{2} u^2\right)$$

use crate::error::{Error, Result};

/// 1D Relativistic D1Q3 Lattice Boltzmann (RLBM) Kinetic Solver.
#[derive(Debug, Clone)]
pub struct RelativisticLbm1D {
    /// Number of lattice sites $N$.
    pub num_sites: usize,
    /// BGK collision relaxation parameter $\omega = \Delta t / \tau \in (0, 2)$.
    pub omega: f64,
    /// Rest population $f_0(x)$.
    pub f0: Vec<f64>,
    /// Right-moving population $f_{+1}(x)$.
    pub f_plus: Vec<f64>,
    /// Left-moving population $f_{-1}(x)$.
    pub f_minus: Vec<f64>,
}

impl RelativisticLbm1D {
    /// Initializes a 1D D1Q3 RLBM lattice from macroscopic density $\rho(x)$ and velocity $u(x)$.
    pub fn new(rho: &[f64], velocity: &[f64], omega: f64) -> Result<Self> {
        let n = rho.len();
        if n < 3 || velocity.len() != n {
            return Err(Error::Msg(
                "Invalid lattice size or mismatched velocity vector".to_string(),
            ));
        }
        // NaN fails both comparisons, so it needs an explicit finiteness check.
        if omega <= 0.0 || omega >= 2.0 || !omega.is_finite() {
            return Err(Error::Msg(
                "BGK relaxation omega must be in (0, 2)".to_string(),
            ));
        }

        let mut f0 = vec![0.0; n];
        let mut f_plus = vec![0.0; n];
        let mut f_minus = vec![0.0; n];

        for i in 0..n {
            let (eq0, eqp, eqm) = Self::equilibrium(rho[i], velocity[i]);
            f0[i] = eq0;
            f_plus[i] = eqp;
            f_minus[i] = eqm;
        }

        Ok(Self {
            num_sites: n,
            omega,
            f0,
            f_plus,
            f_minus,
        })
    }

    #[inline]
    fn equilibrium(rho: f64, u: f64) -> (f64, f64, f64) {
        let u2 = u * u;
        let eq0 = (2.0 / 3.0) * rho * (1.0 - 1.5 * u2);
        let eqp = (1.0 / 6.0) * rho * (1.0 + 3.0 * u + 3.0 * u2);
        let eqm = (1.0 / 6.0) * rho * (1.0 - 3.0 * u + 3.0 * u2);
        (eq0, eqp, eqm)
    }

    /// Executes one full BGK collision + exact lattice streaming step.
    pub fn step(&mut self) {
        let n = self.num_sites;

        // 1. BGK Collision
        for i in 0..n {
            let rho = self.f0[i] + self.f_plus[i] + self.f_minus[i];
            let u = if rho > 1e-15 {
                ((self.f_plus[i] - self.f_minus[i]) / rho).clamp(-0.5, 0.5)
            } else {
                0.0
            };
            let (eq0, eqp, eqm) = Self::equilibrium(rho, u);
            self.f0[i] += self.omega * (eq0 - self.f0[i]);
            self.f_plus[i] += self.omega * (eqp - self.f_plus[i]);
            self.f_minus[i] += self.omega * (eqm - self.f_minus[i]);
        }

        // 2. Exact Periodic Streaming
        let mut next_plus = vec![0.0; n];
        let mut next_minus = vec![0.0; n];
        for i in 0..n {
            next_plus[(i + 1) % n] = self.f_plus[i];
            next_minus[(i + n - 1) % n] = self.f_minus[i];
        }
        self.f_plus = next_plus;
        self.f_minus = next_minus;
    }

    /// Computes total conserved particle mass $\sum_x (f_0 + f_{+1} + f_{-1})$.
    pub fn total_mass(&self) -> f64 {
        (0..self.num_sites)
            .map(|i| self.f0[i] + self.f_plus[i] + self.f_minus[i])
            .sum()
    }

    /// Computes total conserved relativistic momentum $\sum_x (f_{+1} - f_{-1})$.
    pub fn total_momentum(&self) -> f64 {
        (0..self.num_sites)
            .map(|i| self.f_plus[i] - self.f_minus[i])
            .sum()
    }
}
