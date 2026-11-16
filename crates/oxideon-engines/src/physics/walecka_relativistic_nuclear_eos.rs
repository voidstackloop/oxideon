//! Walecka Relativistic Mean-Field ($\sigma$-$\omega$ QHD-I) Nuclear Matter Equation of State (Walecka, 1974; Serot & Walecka, 1986).
//!
//! ### 1. Dirac Effective Nucleon Mass $M^*$ & Vector Repulsion Shift
//! In relativistic mean-field quantum hadrodynamics (QHD-I), scalar $\sigma$-meson attraction lowers the Dirac effective nucleon mass
//! $M^* = M_N - \frac{g_\sigma^2}{m_\sigma^2} \rho_S$, while vector $\omega$-meson exchange shifts the single-particle energy by $+\frac{g_\omega^2}{m_\omega^2} \rho_B$.

use crate::error::{Error, Result};

/// Walecka Relativistic Mean-Field ($\sigma$-$\omega$) Nuclear EOS Engine.
#[derive(Debug, Clone, Copy)]
pub struct WaleckaRelativisticMeanFieldNuclear {
    /// Bare nucleon mass $M_N > 0$ (MeV, $\approx 939$ MeV).
    pub bare_mass_mev: f64,
    /// Scalar coupling-to-mass ratio $C_\sigma^2 = g_\sigma^2 / m_\sigma^2 > 0$ ($\mathrm{MeV \cdot fm^3}$).
    pub c_sigma_sq: f64,
    /// Vector coupling-to-mass ratio $C_\omega^2 = g_\omega^2 / m_\omega^2 > 0$ ($\mathrm{MeV \cdot fm^3}$).
    pub c_omega_sq: f64,
}

impl Default for WaleckaRelativisticMeanFieldNuclear {
    fn default() -> Self {
        Self {
            bare_mass_mev: 939.0,
            c_sigma_sq: 2670.0,
            c_omega_sq: 1950.0,
        }
    }
}

impl WaleckaRelativisticMeanFieldNuclear {
    /// Creates a new Walecka Relativistic Mean-Field Nuclear engine.
    pub fn new(bare_mass_mev: f64, c_sigma_sq: f64, c_omega_sq: f64) -> Result<Self> {
        if bare_mass_mev <= 0.0 || c_sigma_sq <= 0.0 || c_omega_sq <= 0.0 {
            return Err(Error::Msg(
                "All Walecka QHD parameters must be > 0".to_string(),
            ));
        }
        Ok(Self {
            bare_mass_mev,
            c_sigma_sq,
            c_omega_sq,
        })
    }

    /// Evaluates the Dirac effective nucleon mass $M^* = M_N - C_\sigma^2 \rho_S$ and vector potential $V_0 = C_\omega^2 \rho_B$ (in MeV) at scalar density $\rho_S$ and baryon density $\rho_B$ ($\mathrm{fm^{-3}}$).
    pub fn effective_mass_and_vector_potential(
        &self,
        rho_scalar_fm3: f64,
        rho_baryon_fm3: f64,
    ) -> Result<(f64, f64)> {
        if rho_scalar_fm3 < 0.0 || rho_baryon_fm3 < 0.0 {
            return Err(Error::Msg("Densities must be >= 0".to_string()));
        }
        let m_star = (self.bare_mass_mev - self.c_sigma_sq * rho_scalar_fm3).max(1.0);
        let v_omega = self.c_omega_sq * rho_baryon_fm3;
        Ok((m_star, v_omega))
    }
}
