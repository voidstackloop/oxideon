//! Shannon-Hartley AWGN Channel Capacity $C = B \log_2(1 + S/N)$ & Parallel Spectral Water-Filling (Shannon, 1948/1949).
//!
//! ### 1. Parallel Gaussian Channel Water-Filling Power Allocation $P_i^* = (\nu - N_i)_+$
//! Over $K$ independent subchannels with noise variances $N_1, \dots, N_K$ under total power budget $\sum_i P_i \le P_{\mathrm{tot}}$:
//! $$P_i^* = \max(0, \nu - N_i), \qquad C = \sum_{i=1}^K \frac{1}{2} \log_2\!\left(1 + \frac{P_i^*}{N_i}\right)$$

use crate::error::{Error, Result};

/// Shannon-Hartley AWGN Capacity & Spectral Water-Filling Engine.
#[derive(Debug, Clone, Copy)]
pub struct ShannonHartleyWaterfilling {
    /// Channel bandwidth $B > 0$ (Hz).
    pub bandwidth_hz: f64,
}

impl Default for ShannonHartleyWaterfilling {
    fn default() -> Self {
        Self { bandwidth_hz: 1.0 }
    }
}

impl ShannonHartleyWaterfilling {
    /// Creates a new Shannon-Hartley & Water-Filling engine.
    pub fn new(bandwidth_hz: f64) -> Result<Self> {
        if bandwidth_hz <= 0.0 || !bandwidth_hz.is_finite() {
            return Err(Error::Msg("bandwidth_hz must be > 0".to_string()));
        }
        Ok(Self { bandwidth_hz })
    }

    /// Evaluates the AWGN capacity $C = B \log_2(1 + \mathrm{SNR})$ and spectral efficiency $\eta = C / B$ (bits/s/Hz).
    pub fn capacity_and_spectral_efficiency(&self, snr_linear: f64) -> Result<(f64, f64)> {
        if snr_linear < 0.0 || !snr_linear.is_finite() {
            return Err(Error::Msg("snr_linear must be >= 0".to_string()));
        }
        let eff = (1.0 + snr_linear).log2();
        Ok((self.bandwidth_hz * eff, eff))
    }
}
