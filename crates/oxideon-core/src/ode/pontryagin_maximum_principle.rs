//! Pontryagin Maximum Principle (PMP), Symplectic Costate Flow & Minimum-Time / Minimum-Energy Bang-Bang Control (Pontryagin et al., 1962).
//!
//! ### 1. Control Hamiltonian $\mathcal{H}(x, p, u)$ & Optimal Switching Function
//! For a control-affine dynamics $\dot{x} = f(x) + g(x) u$ with control constraint $|u(t)| \le u_{\max}$ and running cost
//! $L(x, u) = \frac{1}{2} r u^2$, maximizing Pontryagin's Hamiltonian $\mathcal{H}(x, p, u) = p^T (f(x) + g(x) u) - \frac{1}{2} r u^2$
//! yields the saturated optimal control:
//! $$u^*(x, p) = \operatorname{clamp}\left(\frac{p^T g(x)}{r}, -u_{\max}, u_{\max}\right)$$

use crate::error::{Error, Result};

/// Pontryagin Maximum Principle & Optimal Costate Switching Engine.
#[derive(Debug, Clone, Copy)]
pub struct PontryaginMaximumEngine {
    /// Control effort penalty $r > 0$.
    pub penalty_r: f64,
    /// Control saturation bound $u_{\max} > 0$.
    pub u_max: f64,
}

impl PontryaginMaximumEngine {
    /// Creates a new Pontryagin Maximum Principle engine.
    pub fn new(penalty_r: f64, u_max: f64) -> Result<Self> {
        if penalty_r <= 0.0 || u_max <= 0.0 {
            return Err(Error::Msg("penalty_r and u_max must be > 0".to_string()));
        }
        Ok(Self { penalty_r, u_max })
    }

    /// Computes the optimal Pontryagin control $u^*(p, g)$ and maximized Hamiltonian $\mathcal{H}^*(x, p)$.
    pub fn optimal_control_and_hamiltonian(
        &self,
        drift_dot_p: f64,
        control_switching_pg: f64,
    ) -> (f64, f64) {
        let u_opt = (control_switching_pg / self.penalty_r).clamp(-self.u_max, self.u_max);
        let h_opt =
            drift_dot_p + control_switching_pg * u_opt - 0.5 * self.penalty_r * u_opt * u_opt;
        (u_opt, h_opt)
    }
}
