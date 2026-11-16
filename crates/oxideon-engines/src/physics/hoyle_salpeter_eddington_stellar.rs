//! Eddington Luminosity Limit $L_{\mathrm{Edd}}$ & Hoyle-Salpeter Triple-Alpha Nucleosynthesis Resonance (Eddington, 1926; Salpeter, 1952; Hoyle, 1954).
//!
//! ### 1. Eddington Radiation-Pressure Limit & Triple-Alpha Temperature Sensitivity
//! Balancing outward Thomson electron-scattering radiation force against gravity yields $L_{\mathrm{Edd}} \approx 3.2 \times 10^4 (M / M_\odot) L_\odot$,
//! while Hoyle's $0^+$ resonance at $E_r = 379.5\,\mathrm{keV}$ above $3\alpha$ threshold drives $r_{3\alpha} \propto \rho^2 Y^3 T^{-3} \exp(-44.027 / T_8)$.

use crate::error::{Error, Result};

/// Eddington Luminosity Limit & Hoyle-Salpeter Triple-Alpha Engine.
#[derive(Debug, Clone, Copy)]
pub struct HoyleSalpeterEddingtonStellar {
    /// Stellar mass in solar masses $M / M_\odot > 0$.
    pub mass_solar: f64,
}

impl Default for HoyleSalpeterEddingtonStellar {
    fn default() -> Self {
        Self { mass_solar: 10.0 }
    }
}

impl HoyleSalpeterEddingtonStellar {
    /// Creates a new Hoyle-Salpeter-Eddington Stellar engine.
    pub fn new(mass_solar: f64) -> Result<Self> {
        if mass_solar <= 0.0 || !mass_solar.is_finite() {
            return Err(Error::Msg("mass_solar must be > 0".to_string()));
        }
        Ok(Self { mass_solar })
    }

    /// Evaluates Eddington luminosity $L_{\mathrm{Edd}} / L_\odot = 3.2 \times 10^4 (M / M_\odot)$ and logarithmic temperature sensitivity $\nu_{3\alpha} = \partial \ln \epsilon_{3\alpha} / \partial \ln T = 44.027 / T_8 - 3$ at temperature $T_8 = T / 10^8\,\mathrm{K}$.
    pub fn eddington_luminosity_and_triple_alpha_exponent(
        &self,
        temp_t8: f64,
    ) -> Result<(f64, f64)> {
        if temp_t8 <= 0.0 {
            return Err(Error::Msg("Temperature T_8 must be > 0".to_string()));
        }
        let l_edd_solar = 3.2e4 * self.mass_solar;
        let nu_3alpha = 44.027 / temp_t8 - 3.0;
        Ok((l_edd_solar, nu_3alpha))
    }
}
