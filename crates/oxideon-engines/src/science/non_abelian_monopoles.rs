//! 't Hooft–Polyakov Magnetic Monopoles, Prasad-Sommerfield BPS Limit & Dirac Quantization ('t Hooft, 1974; Polyakov, 1974).
//!
//! ### 1. Bogomolny-Prasad-Sommerfield (BPS) Monopole Mass & Profile
//! In an $SU(2)$ Georgi-Glashow gauge-Higgs theory broken to $U(1)$ at vacuum expectation value $v > 0$,
//! a magnetic monopole of topological winding number $n \in \mathbb{Z}$ and magnetic charge $g_m = \frac{4\pi n}{e}$
//! satisfies the Bogomolny mass bound:
//! $$M_{\text{BPS}} = \frac{4 \pi v}{e} |n| = v \, |g_m|$$
//! In the Prasad-Sommerfield limit ($\lambda \to 0$), the exact analytical Higgs profile is $H(\xi) = \xi \coth(\xi) - 1$ ($\xi = e v r$).

use crate::error::{Error, Result};
use std::f64::consts::PI;

/// 't Hooft–Polyakov BPS Magnetic Monopole & Prasad-Sommerfield Profile Engine.
#[derive(Debug, Clone, Copy)]
pub struct BpsMagneticMonopoleEngine {
    /// Gauge coupling $e > 0$.
    pub gauge_e: f64,
    /// Higgs vacuum expectation value $v > 0$.
    pub higgs_vev: f64,
}

impl BpsMagneticMonopoleEngine {
    /// Creates a new BPS Magnetic Monopole engine.
    pub fn new(gauge_e: f64, higgs_vev: f64) -> Result<Self> {
        if gauge_e <= 0.0 || higgs_vev <= 0.0 {
            return Err(Error::Msg("gauge_e and higgs_vev must be > 0".to_string()));
        }
        Ok(Self { gauge_e, higgs_vev })
    }

    /// Computes the exact BPS monopole mass $M_{\text{BPS}} = \frac{4 \pi v}{e} |n|$.
    pub fn bps_monopole_mass(&self, winding_n: i64) -> f64 {
        4.0 * PI * self.higgs_vev * (winding_n.abs() as f64) / self.gauge_e
    }

    /// Evaluates the normalized Prasad-Sommerfield Higgs field ratio $\phi(r) / v = \frac{\xi \coth(\xi) - 1}{\xi} = \coth(\xi) - \frac{1}{\xi}$
    /// where $\xi = e v r > 0$.
    pub fn prasad_sommerfield_higgs_profile(&self, radius_r: f64) -> f64 {
        let xi = self.gauge_e * self.higgs_vev * radius_r;
        if xi < 1e-6 {
            xi / 3.0
        } else {
            (1.0 / xi.tanh()) - (1.0 / xi)
        }
    }
}
