//! Förster Resonance Energy Transfer (FRET) Dipole-Dipole $R_0^6 / r^6$ Spectroscopic Ruler & Exciton Coupling (Förster, 1948).
//!
//! ### 1. Förster Radius $R_0$ & Sixth-Power Distance Efficiency $E(r)$
//! Non-radiative dipole-dipole exciton transfer between donor $D$ and acceptor $A$ at separation $r > 0$ with
//! Förster critical distance $R_0 > 0$ has rate $k_{\mathrm{FRET}}(r) = \tau_D^{-1} (R_0 / r)^6$ and quantum efficiency:
//! $$E(r) = \frac{R_0^6}{R_0^6 + r^6}$$

use crate::error::{Error, Result};

/// Förster Resonance Energy Transfer (FRET) Spectroscopic Ruler Engine.
#[derive(Debug, Clone, Copy)]
pub struct ForsterResonanceEnergyEngine {
    /// Förster 50%-efficiency distance $R_0 > 0$ (in $\mathrm{nm}$).
    pub forster_r0: f64,
    /// Donor fluorescence lifetime $\tau_D > 0$ (in $\mathrm{ns}$).
    pub donor_lifetime_tau: f64,
}

impl ForsterResonanceEnergyEngine {
    /// Creates a new Förster Resonance Energy Transfer engine.
    pub fn new(forster_r0: f64, donor_lifetime_tau: f64) -> Result<Self> {
        if forster_r0 <= 0.0 || donor_lifetime_tau <= 0.0 {
            return Err(Error::Msg(
                "forster_r0 and donor_lifetime_tau must be > 0".to_string(),
            ));
        }
        Ok(Self {
            forster_r0,
            donor_lifetime_tau,
        })
    }

    /// Computes the FRET efficiency $E(r) \in (0, 1)$ and transfer rate $k_{\mathrm{FRET}}(r)$ at donor-acceptor separation $r > 0$.
    pub fn efficiency_and_rate(&self, distance_r: f64) -> Result<(f64, f64)> {
        if distance_r <= 0.0 {
            return Err(Error::Msg("distance_r must be > 0".to_string()));
        }
        let r0_6 = self.forster_r0.powi(6);
        let r6 = distance_r.powi(6);
        let eff = r0_6 / (r0_6 + r6);
        let rate = (r0_6 / r6) / self.donor_lifetime_tau;
        Ok((eff, rate))
    }
}
