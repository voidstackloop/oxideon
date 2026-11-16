//! Guruswami-Sudan Algebraic List Decoding Radius $1 - \sqrt{R}$ for Reed-Solomon Codes (Sudan, 1997; Guruswami & Sudan, 1999).
//!
//! ### 1. Johnson-Guruswami-Sudan Decoding Fraction $\tau_{\mathrm{GS}}(R) = 1 - \sqrt{R}$ vs Unique Decoding $(1 - R)/2$
//! For a Reed-Solomon code $[n, k = R n]_q$ of rate $R \in (0, 1)$, polynomial interpolation with multiplicity $s \to \infty$ list-decodes up to fraction of errors:
//! $$\tau_{\mathrm{GS}}(R) = 1 - \sqrt{R} > \frac{1 - R}{2} = \tau_{\mathrm{unique}}(R)$$

use crate::error::{Error, Result};

/// Guruswami-Sudan Reed-Solomon List-Decoding Engine.
#[derive(Debug, Clone, Copy)]
pub struct GuruswamiSudanListDecoder {
    /// Code rate $R = k / n \in (0, 1)$.
    pub code_rate_r: f64,
}

impl Default for GuruswamiSudanListDecoder {
    fn default() -> Self {
        Self { code_rate_r: 0.25 }
    }
}

impl GuruswamiSudanListDecoder {
    /// Creates a new Guruswami-Sudan List Decoder engine.
    pub fn new(code_rate_r: f64) -> Result<Self> {
        if !(0.0..1.0).contains(&code_rate_r) || code_rate_r == 0.0 {
            return Err(Error::Msg("code_rate_r must be in (0, 1)".to_string()));
        }
        Ok(Self { code_rate_r })
    }

    /// Evaluates the Guruswami-Sudan list-decoding radius $1 - \sqrt{R}$, unique decoding radius $(1 - R)/2$, and the list-decoding gain.
    pub fn gs_radius_unique_radius_and_gain(&self) -> (f64, f64, f64) {
        let gs = 1.0 - self.code_rate_r.sqrt();
        let uniq = 0.5 * (1.0 - self.code_rate_r);
        (gs, uniq, gs - uniq)
    }
}
