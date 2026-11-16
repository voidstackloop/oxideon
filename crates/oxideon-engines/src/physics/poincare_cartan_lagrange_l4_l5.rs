//! Circular Restricted Three-Body Problem (CR3BP), Jacobi Integral $C_J$ & Gascheau-Routh Trojan Stability $\mu_R \approx 0.03852$ (Lagrange, 1772; Gascheau, 1843).
//!
//! ### 1. Jacobi Constant $C_J$ & Equilateral Lagrange Points $L_4, L_5$
//! In the synodic frame rotating with primary masses $1 - \mu$ at $(-\mu, 0)$ and $\mu$ at $(1 - \mu, 0)$, the effective potential is
//! $\Omega(x, y) = \frac{1}{2}(x^2 + y^2) + \frac{1 - \mu}{r_1} + \frac{\mu}{r_2}$ and the conserved **Jacobi Constant** is $C_J = 2 \Omega - v^2$.
//! The equilateral Trojan points $L_4, L_5$ at $(1/2 - \mu, \pm \sqrt{3}/2)$ are linearly spectrally stable iff $27 \mu (1 - \mu) < 1$, i.e.:
//! $$\mu < \mu_{\mathrm{Routh}} = \frac{1}{2}\left(1 - \frac{\sqrt{69}}{9}\right) \approx 0.0385208965$$

use crate::error::{Error, Result};

/// Circular Restricted Three-Body Problem (CR3BP) & Lagrange Point Stability Engine.
#[derive(Debug, Clone, Copy)]
pub struct CircularRestrictedThreeBodyEngine {
    /// Mass ratio $\mu = m_2 / (m_1 + m_2) \in (0, 0.5]$.
    pub mass_ratio_mu: f64,
}

impl CircularRestrictedThreeBodyEngine {
    /// Creates a new CR3BP Celestial Mechanics engine.
    pub fn new(mass_ratio_mu: f64) -> Result<Self> {
        if !(0.0..=0.5).contains(&mass_ratio_mu) || mass_ratio_mu == 0.0 {
            return Err(Error::Msg("mass_ratio_mu must be in (0, 0.5]".to_string()));
        }
        Ok(Self { mass_ratio_mu })
    }

    /// Evaluates the conserved Jacobi Integral $C_J = x^2 + y^2 + \frac{2(1-\mu)}{r_1} + \frac{2\mu}{r_2} - (v_x^2 + v_y^2)$ and checks Routh stability of $L_4, L_5$.
    pub fn jacobi_constant_and_routh_stable(&self, pos: [f64; 2], vel: [f64; 2]) -> (f64, bool) {
        let [x, y] = pos;
        let mu = self.mass_ratio_mu;
        let r1 = ((x + mu).powi(2) + y * y).sqrt().max(1e-15);
        let r2 = ((x - 1.0 + mu).powi(2) + y * y).sqrt().max(1e-15);
        let cj = x * x + y * y + 2.0 * (1.0 - mu) / r1 + 2.0 * mu / r2
            - (vel[0] * vel[0] + vel[1] * vel[1]);
        let routh_stable = 27.0 * mu * (1.0 - mu) < 1.0;
        (cj, routh_stable)
    }
}
