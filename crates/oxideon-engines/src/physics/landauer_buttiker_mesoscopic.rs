//! Landauer-Büttiker Mesoscopic Ballistic Conductance Quantization & Büttiker Shot-Noise Fano Factor (Landauer, 1957; Büttiker, 1990).
//!
//! ### 1. Landauer Formula & Partition Shot Noise Suppression
//! Given channel transmission probabilities $T_n \in [0, 1]$ and spin-degenerate conductance quantum $G_0 = 2 e^2 / h$,
//! the two-terminal conductance and zero-temperature shot-noise Fano factor $F \in [0, 1]$ are:
//! $$G = G_0 \sum_n T_n, \qquad F = \frac{\sum_n T_n (1 - T_n)}{\sum_n T_n}$$

use crate::error::{Error, Result};

/// Landauer-Büttiker Mesoscopic Conductance & Shot-Noise Fano Factor Engine.
#[derive(Debug, Clone, Copy, Default)]
pub struct LandauerButtikerBallisticTransport;

impl LandauerButtikerBallisticTransport {
    /// Creates a new Landauer-Büttiker Ballistic Transport engine.
    pub fn new() -> Self {
        Self
    }

    /// Evaluates normalized conductance $G / G_0 = \sum T_n$ and Büttiker shot-noise Fano factor $F = \sum T_n(1 - T_n) / \sum T_n$.
    pub fn conductance_and_fano_factor(transmissions: &[f64]) -> Result<(f64, f64)> {
        if transmissions.is_empty() || transmissions.iter().any(|&t| !(0.0..=1.0).contains(&t)) {
            return Err(Error::Msg(
                "transmissions must be non-empty in [0, 1]".to_string(),
            ));
        }
        let g_sum: f64 = transmissions.iter().sum();
        if g_sum <= 0.0 {
            return Ok((0.0, 0.0));
        }
        let noise_sum: f64 = transmissions.iter().map(|&t| t * (1.0 - t)).sum();
        Ok((g_sum, noise_sum / g_sum))
    }
}
