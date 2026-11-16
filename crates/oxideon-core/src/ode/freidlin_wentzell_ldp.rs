//! Freidlin-Wentzell Large Deviations Principle, Quasi-Potential & Eyring-Kramers Escape Rate (Freidlin & Wentzell, 1970).
//!
//! ### 1. Eyring-Kramers Metastable Escape Rate Across Potential Barrier $\Delta V$
//! For an overdamped Langevin diffusion $dX_t = -V'(X_t) dt + \sqrt{2 \varepsilon} dW_t$ in a double-well potential
//! with local minimum $x_{\min}$ ($V''(x_{\min}) > 0$) and saddle barrier $x_{\text{sad}}$ ($V''(x_{\text{sad}}) < 0$),
//! the Freidlin-Wentzell quasi-potential is $W = 2 \Delta V$ and the Eyring-Kramers mean exit time is:
//! $$\mathbb{E}[\tau_\varepsilon] = \frac{2 \pi}{\sqrt{V''(x_{\min}) |V''(x_{\text{sad}})|}} \exp\left(\frac{\Delta V}{\varepsilon}\right)$$

use crate::error::{Error, Result};
use std::f64::consts::PI;

/// Freidlin-Wentzell Large Deviations & Eyring-Kramers Metastable Escape Engine.
#[derive(Debug, Clone, Copy, Default)]
pub struct FreidlinWentzellLdpEngine;

impl FreidlinWentzellLdpEngine {
    /// Creates a new Freidlin-Wentzell Large Deviations engine.
    pub fn new() -> Self {
        Self
    }

    /// Evaluates the exact Eyring-Kramers mean escape time $\mathbb{E}[\tau_\varepsilon]$ over barrier $\Delta V > 0$ at noise intensity $\varepsilon > 0$.
    pub fn eyring_kramers_escape_time(
        barrier_delta_v: f64,
        noise_eps: f64,
        curvature_well: f64,
        curvature_saddle_abs: f64,
    ) -> Result<f64> {
        if barrier_delta_v <= 0.0
            || noise_eps <= 0.0
            || curvature_well <= 0.0
            || curvature_saddle_abs <= 0.0
        {
            return Err(Error::Msg(
                "All Eyring-Kramers parameters must be > 0".to_string(),
            ));
        }
        let prefactor = 2.0 * PI / (curvature_well * curvature_saddle_abs).sqrt();
        Ok(prefactor * (barrier_delta_v / noise_eps).exp())
    }
}
