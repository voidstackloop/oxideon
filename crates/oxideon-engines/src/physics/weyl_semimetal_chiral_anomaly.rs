//! Weyl Semimetals, Berry Curvature Monopoles, Adler-Bell-Jackiw Chiral Anomaly & Negative Magnetoresistance (Son & Spivak, 2013).
//!
//! ### 1. Chiral Pumping Between Weyl Nodes & Quadratic Magnetoconductivity $\sigma_{zz}(B)$
//! Parallel electric and magnetic fields $\mathbf{E} \parallel \mathbf{B}$ pump chiral charge between opposite-chirality
//! Weyl nodes at rate $\partial_t (n_+ - n_-) = \frac{e^2}{2 \pi^2 \hbar^2} \mathbf{E} \cdot \mathbf{B}$, generating the
//! Son-Spivak positive longitudinal magnetoconductivity:
//! $$\sigma_{zz}(B) = \sigma_0 + \frac{e^4 v_F^3 \tau_a}{4 \pi^2 \hbar^4 \mu^2} B^2$$

use crate::error::{Error, Result};
use std::f64::consts::PI;

/// Weyl Semimetal Chiral Anomaly & Longitudinal Magnetoconductivity Engine.
#[derive(Debug, Clone, Copy)]
pub struct WeylChiralAnomalyEngine {
    /// Zero-field Drude conductivity $\sigma_0 > 0$.
    pub sigma_0: f64,
    /// Inter-valley chiral relaxation time $\tau_a > 0$.
    pub chiral_tau_a: f64,
    /// Fermi velocity $v_F > 0$.
    pub fermi_velocity: f64,
    /// Chemical potential $\mu > 0$.
    pub chemical_potential_mu: f64,
}

impl WeylChiralAnomalyEngine {
    /// Creates a new Weyl Semimetal Chiral Anomaly engine.
    pub fn new(
        sigma_0: f64,
        chiral_tau_a: f64,
        fermi_velocity: f64,
        chemical_potential_mu: f64,
    ) -> Result<Self> {
        if sigma_0 <= 0.0
            || chiral_tau_a <= 0.0
            || fermi_velocity <= 0.0
            || chemical_potential_mu <= 0.0
        {
            return Err(Error::Msg(
                "All Weyl chiral anomaly parameters must be > 0".to_string(),
            ));
        }
        Ok(Self {
            sigma_0,
            chiral_tau_a,
            fermi_velocity,
            chemical_potential_mu,
        })
    }

    /// Evaluates the longitudinal magnetoconductivity $\sigma_{zz}(B) = \sigma_0 + C_{\text{chiral}} B^2$ in natural units ($e = \hbar = 1$).
    pub fn longitudinal_magnetoconductivity(&self, magnetic_b: f64) -> f64 {
        let c_chiral = self.fermi_velocity.powi(3) * self.chiral_tau_a
            / (4.0 * PI * PI * self.chemical_potential_mu.powi(2));
        self.sigma_0 + c_chiral * magnetic_b * magnetic_b
    }
}
