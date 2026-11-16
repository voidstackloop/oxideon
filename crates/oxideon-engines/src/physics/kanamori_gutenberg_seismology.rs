//! Kanamori Moment Magnitude $M_w$, Gutenberg-Richter $b$-Value & Omori-Utsu Aftershock Law (Gutenberg & Richter, 1944; Kanamori, 1977).
//!
//! ### 1. Seismic Moment Magnitude & Omori-Utsu Decay
//! Given scalar seismic moment $M_0 = \mu A D$ (in $\mathrm{N \cdot m}$), Hanks & Kanamori (1979) define:
//! $$M_w = \frac{2}{3} \log_{10}(M_0) - 6.07, \qquad n(t) = \frac{K}{(t + c)^p}$$

use crate::error::{Error, Result};

/// Kanamori Moment Magnitude, Gutenberg-Richter & Omori Aftershock Engine.
#[derive(Debug, Clone, Copy)]
pub struct KanamoriGutenbergSeismicEngine {
    /// Gutenberg-Richter $b$-value ($b > 0$, typically $\approx 1.0$).
    pub b_value: f64,
}

impl Default for KanamoriGutenbergSeismicEngine {
    fn default() -> Self {
        Self { b_value: 1.0 }
    }
}

impl KanamoriGutenbergSeismicEngine {
    /// Creates a new Kanamori-Gutenberg Seismic engine.
    pub fn new(b_value: f64) -> Result<Self> {
        if b_value <= 0.0 || !b_value.is_finite() {
            return Err(Error::Msg(
                "Gutenberg-Richter b_value must be > 0".to_string(),
            ));
        }
        Ok(Self { b_value })
    }

    /// Evaluates Kanamori moment magnitude $M_w = \frac{2}{3} \log_{10}(M_0) - 6.07$ and Omori aftershock rate $K / (t + c)^p$.
    pub fn moment_magnitude_and_omori_rate(
        &self,
        seismic_moment_nm: f64,
        time_days: f64,
        omori_k: f64,
        omori_c: f64,
        omori_p: f64,
    ) -> Result<(f64, f64)> {
        if seismic_moment_nm <= 0.0
            || time_days < 0.0
            || omori_k <= 0.0
            || omori_c <= 0.0
            || omori_p <= 0.0
        {
            return Err(Error::Msg(
                "Invalid seismic moment or Omori parameters".to_string(),
            ));
        }
        let mw = (2.0 / 3.0) * seismic_moment_nm.log10() - 6.07;
        let rate = omori_k / (time_days + omori_c).powf(omori_p);
        Ok((mw, rate))
    }
}
