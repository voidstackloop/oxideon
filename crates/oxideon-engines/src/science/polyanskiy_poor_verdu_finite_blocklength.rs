//! Polyanskiy-Poor-Verdú Finite-Blocklength Normal Approximation & Channel Dispersion $V$ (Polyanskiy, Poor & Verdú, 2010).
//!
//! ### 1. Non-Asymptotic Maximal Coding Rate $R(n, \epsilon) \approx C - \sqrt{\frac{V}{n}} Q^{-1}(\epsilon) + \frac{\log_2 n}{2 n}$
//! For an AWGN channel with SNR $P$, capacity $C = \frac{1}{2}\log_2(1 + P)$, and AWGN dispersion:
//! $$V(P) = \frac{P(P + 2)}{2(P + 1)^2} (\log_2 e)^2$$

use crate::error::{Error, Result};
use std::f64::consts::LN_2;

/// Polyanskiy-Poor-Verdú Finite-Blocklength Channel Dispersion Engine.
#[derive(Debug, Clone, Copy)]
pub struct PolyanskiyFiniteBlocklengthRate {
    /// Linear SNR $P > 0$.
    pub snr_p: f64,
}

impl Default for PolyanskiyFiniteBlocklengthRate {
    fn default() -> Self {
        Self { snr_p: 3.0 }
    }
}

impl PolyanskiyFiniteBlocklengthRate {
    /// Creates a new Polyanskiy-Poor-Verdú Finite-Blocklength engine.
    pub fn new(snr_p: f64) -> Result<Self> {
        if snr_p <= 0.0 || !snr_p.is_finite() {
            return Err(Error::Msg("snr_p must be > 0".to_string()));
        }
        Ok(Self { snr_p })
    }

    /// Evaluates the AWGN channel dispersion $V(P)$ ($\mathrm{bits}^2/    ext{channel use}$) and normal-approximation rate $R(n, \epsilon)$ given $Q^{-1}(\epsilon) > 0$.
    pub fn dispersion_and_finite_blocklength_rate(
        &self,
        blocklength_n: usize,
        q_inv_eps: f64,
    ) -> Result<(f64, f64)> {
        if blocklength_n < 10 || q_inv_eps <= 0.0 {
            return Err(Error::Msg(
                "blocklength_n >= 10 and q_inv_eps > 0 required".to_string(),
            ));
        }
        let p = self.snr_p;
        let log2_e = 1.0 / LN_2;
        let v_disp = (p * (p + 2.0) / (2.0 * (p + 1.0) * (p + 1.0))) * log2_e * log2_e;
        let c = 0.5 * (1.0 + p).log2();
        let n_f = blocklength_n as f64;
        let r_n = (c - (v_disp / n_f).sqrt() * q_inv_eps + 0.5 * n_f.log2() / n_f).max(0.0);
        Ok((v_disp, r_n))
    }
}
