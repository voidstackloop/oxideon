//! Slepian-Wolf Lossless Distributed Source Coding & Wyner-Ziv Side-Information Rate-Distortion (Slepian & Wolf, 1973; Wyner & Ziv, 1976).
//!
//! ### 1. Slepian-Wolf Achievable Rate Region: $R_X \ge H(X|Y)$, $R_Y \ge H(Y|X)$, $R_X + R_Y \ge H(X, Y)$
//! Separate encoders observing correlated sources $(X, Y)$ can achieve the same sum-rate $H(X, Y)$ as a joint encoder:
//! $$R_X \ge H(X | Y) = H(X, Y) - H(Y), \qquad R_Y \ge H(Y | X) = H(X, Y) - H(X), \qquad R_X + R_Y \ge H(X, Y)$$

use crate::error::{Error, Result};

/// Slepian-Wolf & Wyner-Ziv Distributed Source Coding Engine.
#[derive(Debug, Clone, Copy)]
pub struct SlepianWolfRateRegionEngine {
    /// Marginal entropy $H(X) > 0$.
    pub hx: f64,
    /// Marginal entropy $H(Y) > 0$.
    pub hy: f64,
    /// Joint entropy $H(X, Y)$.
    pub hxy: f64,
}

impl Default for SlepianWolfRateRegionEngine {
    fn default() -> Self {
        Self {
            hx: 1.0,
            hy: 1.0,
            hxy: 1.5,
        }
    }
}

impl SlepianWolfRateRegionEngine {
    /// Creates a new Slepian-Wolf Distributed Coding engine.
    pub fn new(hx: f64, hy: f64, hxy: f64) -> Result<Self> {
        if hx <= 0.0 || hy <= 0.0 || hxy < hx.max(hy) || hxy > hx + hy + 1e-12 {
            return Err(Error::Msg(
                "valid entropies max(hx, hy) <= hxy <= hx + hy required".to_string(),
            ));
        }
        Ok(Self { hx, hy, hxy })
    }

    /// Evaluates the conditional entropies $(H(X|Y), H(Y|X))$ and checks if a rate pair $(R_X, R_Y)$ lies inside the Slepian-Wolf region.
    pub fn conditional_rates_and_achievability(&self, rx: f64, ry: f64) -> (f64, f64, bool) {
        let hx_given_y = self.hxy - self.hy;
        let hy_given_x = self.hxy - self.hx;
        let ok =
            rx + 1e-12 >= hx_given_y && ry + 1e-12 >= hy_given_x && (rx + ry) + 1e-12 >= self.hxy;
        (hx_given_y, hy_given_x, ok)
    }
}
