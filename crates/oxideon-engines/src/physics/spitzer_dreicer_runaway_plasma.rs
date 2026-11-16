//! Spitzer Fully Ionized Plasma Resistivity $\eta \propto T_e^{-3/2}$ & Dreicer Runaway Electron Field (Spitzer & Härm, 1953; Dreicer, 1959).
//!
//! ### 1. Spitzer Resistivity Scaling & Dreicer Critical Electric Field $E_D$
//! Because Coulomb collision frequency scales as $\nu_{ei} \propto n_e Z_{\mathrm{eff}} \ln \Lambda / T_e^{3/2}$, Spitzer resistivity drops as $T_e^{-3/2}$,
//! while electrons accelerated above the Dreicer field $E_D \propto n_e \ln \Lambda / T_e$ experience runaway acceleration.

use crate::error::{Error, Result};

/// Spitzer Plasma Resistivity & Dreicer Runaway Electron Engine.
#[derive(Debug, Clone, Copy)]
pub struct SpitzerDreicerPlasmaTransport {
    /// Effective ion charge state $Z_{\mathrm{eff}} \ge 1$.
    pub z_eff: f64,
    /// Coulomb logarithm $\ln \Lambda > 0$ (typically $15$–$20$).
    pub coulomb_log: f64,
}

impl Default for SpitzerDreicerPlasmaTransport {
    fn default() -> Self {
        Self {
            z_eff: 1.0,
            coulomb_log: 17.0,
        }
    }
}

impl SpitzerDreicerPlasmaTransport {
    /// Creates a new Spitzer-Dreicer Plasma Transport engine.
    pub fn new(z_eff: f64, coulomb_log: f64) -> Result<Self> {
        if z_eff < 1.0 || coulomb_log <= 0.0 {
            return Err(Error::Msg(
                "z_eff >= 1 and coulomb_log > 0 required".to_string(),
            ));
        }
        Ok(Self { z_eff, coulomb_log })
    }

    /// Evaluates Spitzer parallel resistivity $\eta_\parallel$ (in $\Omega \cdot \mathrm{m}$) and normalized Dreicer field $E_D \propto n_{20} \ln\Lambda / T_{\mathrm{keV}}$.
    pub fn spitzer_resistivity_and_dreicer_field(
        &self,
        temp_kev: f64,
        density_1e20_m3: f64,
    ) -> Result<(f64, f64)> {
        if temp_kev <= 0.0 || density_1e20_m3 <= 0.0 {
            return Err(Error::Msg(
                "temp_kev and density_1e20_m3 must be > 0".to_string(),
            ));
        }
        // Standard Spitzer formula: eta_parallel approx 1.65e-9 * Z_eff * ln(Lambda) / T_keV^(3/2) Ohm*m
        let eta_ohm_m = 1.65e-9 * self.z_eff * self.coulomb_log / temp_kev.powf(1.5);
        // Dreicer field E_D approx 2.6 * n_20 * ln(Lambda) / T_keV V/m
        let e_dreicer = 2.6 * density_1e20_m3 * self.coulomb_log / temp_kev;
        Ok((eta_ohm_m, e_dreicer))
    }
}
