//! Thurston Hyperbolic Dehn Surgery Theorem & Lobachevsky-Milnor Ideal Tetrahedron Volume (Thurston, 1979).
//!
//! ### 1. Hyperbolic Dehn Filling Volume Asymptotics
//! Under $(p, q)$ Dehn filling on a 1-cusped hyperbolic 3-manifold $M$ with cusp shape $\tau$ and cusp area $A$,
//! the volume decrease is asymptotically (Neumann-Zagier, 1985):
//! $$\Delta \mathrm{Vol}(M_{p,q}) = \mathrm{Vol}(M) - \mathrm{Vol}(M_{p,q}) \approx \frac{\pi^2}{L(p, q)^2}$$
//! where $L(p, q)$ is the normalized Euclidean length of the $(p,q)$ slope on the cusp torus.

use crate::error::{Error, Result};
use std::f64::consts::PI;

/// Thurston Hyperbolic Dehn Surgery & Ideal Tetrahedron Volume Engine.
#[derive(Debug, Clone, Copy)]
pub struct ThurstonHyperbolicDehnEngine {
    /// Cusped manifold volume $\mathrm{Vol}(M)$ (e.g., $2 V_0 \approx 2.0298832$ for the figure-eight knot complement).
    pub cusped_volume: f64,
}

impl Default for ThurstonHyperbolicDehnEngine {
    fn default() -> Self {
        Self {
            cusped_volume: 2.029883212819307,
        }
    }
}

impl ThurstonHyperbolicDehnEngine {
    /// Creates a new Thurston Hyperbolic Dehn Surgery engine.
    pub fn new(cusped_volume: f64) -> Result<Self> {
        if cusped_volume <= 0.0 || !cusped_volume.is_finite() {
            return Err(Error::Msg("cusped_volume must be > 0".to_string()));
        }
        Ok(Self { cusped_volume })
    }

    /// Evaluates the Neumann-Zagier asymptotic volume after $(p, q)$ Dehn surgery with normalized slope length squared $L^2 = p^2 + c q^2$.
    pub fn dehn_filled_volume(&self, p: i64, q: i64, cusp_aspect_c: f64) -> Result<(f64, f64)> {
        if (p == 0 && q == 0) || cusp_aspect_c <= 0.0 {
            return Err(Error::Msg(
                "Invalid Dehn surgery slope (p, q) or cusp aspect".to_string(),
            ));
        }
        let l_sq = (p as f64).powi(2) + cusp_aspect_c * (q as f64).powi(2);
        let delta_vol = (PI * PI) / l_sq;
        let filled_vol = (self.cusped_volume - delta_vol).max(0.0);
        Ok((filled_vol, delta_vol))
    }
}
