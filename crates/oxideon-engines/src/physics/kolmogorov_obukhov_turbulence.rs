//! Kolmogorov K41 Energy Cascade $E(k) = C_K \varepsilon^{2/3} k^{-5/3}$ & K62 Log-Normal Intermittency (Kolmogorov, 1941/1962; Obukhov, 1962).
//!
//! ### 1. Inertial-Range Spectrum & K62 Structure Function Exponents $\zeta_p = p/3 - \mu p(p - 3)/18$
//! In isotropic homogeneous turbulence with mean dissipation rate $\varepsilon$ and intermittency parameter $\mu \approx 0.25$:
//! $$E(k) = C_K \varepsilon^{2/3} k^{-5/3}, \qquad \zeta_p = \frac{p}{3} - \frac{\mu p(p - 3)}{18}, \qquad \eta_K = \left(\frac{\nu^3}{\varepsilon}\right)^{1/4}$$

use crate::error::{Error, Result};

/// Kolmogorov K41/K62 Turbulence Spectrum & Intermittency Engine.
#[derive(Debug, Clone, Copy)]
pub struct KolmogorovK41ObukhovIntermittency {
    /// Kolmogorov constant $C_K \approx 1.5$.
    pub kolmogorov_ck: f64,
    /// K62 intermittency coefficient $\mu \in [0, 0.5]$.
    pub intermittency_mu: f64,
}

impl Default for KolmogorovK41ObukhovIntermittency {
    fn default() -> Self {
        Self {
            kolmogorov_ck: 1.5,
            intermittency_mu: 0.25,
        }
    }
}

impl KolmogorovK41ObukhovIntermittency {
    /// Creates a new Kolmogorov K41/K62 Turbulence engine.
    pub fn new(kolmogorov_ck: f64, intermittency_mu: f64) -> Result<Self> {
        if kolmogorov_ck <= 0.0 || !(0.0..=1.0).contains(&intermittency_mu) {
            return Err(Error::Msg(
                "kolmogorov_ck > 0 and intermittency_mu in [0, 1] required".to_string(),
            ));
        }
        Ok(Self {
            kolmogorov_ck,
            intermittency_mu,
        })
    }

    /// Evaluates the K41 spectral density $E(k)$ and the K62 structure-function scaling exponent $\zeta_p$.
    pub fn energy_spectrum_and_zeta_p(
        &self,
        eps_dissipation: f64,
        wavenumber_k: f64,
        order_p: f64,
    ) -> Result<(f64, f64)> {
        if eps_dissipation <= 0.0 || wavenumber_k <= 0.0 || order_p < 0.0 {
            return Err(Error::Msg(
                "eps_dissipation > 0, wavenumber_k > 0, order_p >= 0 required".to_string(),
            ));
        }
        let ek =
            self.kolmogorov_ck * eps_dissipation.powf(2.0 / 3.0) * wavenumber_k.powf(-5.0 / 3.0);
        let zeta_p = order_p / 3.0 - self.intermittency_mu * order_p * (order_p - 3.0) / 18.0;
        Ok((ek, zeta_p))
    }
}
