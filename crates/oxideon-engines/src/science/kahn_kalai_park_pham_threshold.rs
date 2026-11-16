//! Kahn-Kalai Conjecture (Park-Pham Theorem) on Expectation vs Critical Thresholds (Kahn & Kalai, 2007; Park & Pham, 2022).
//!
//! ### 1. Park-Pham Theorem: $p_c(\mathcal{F}) \le C \, q(\mathcal{F}) \log \ell(\mathcal{F})$
//! For any nontrivial monotone hypergraph property $\mathcal{F}$ with maximum minimal-element size $\ell(\mathcal{F})$ and fractional expectation threshold $q(\mathcal{F})$:
//! $$q(\mathcal{F}) \le p_c(\mathcal{F}) \le K_{\mathrm{PP}} \, q(\mathcal{F}) \log_2(\ell(\mathcal{F}) + 1)$$

use crate::error::{Error, Result};

/// Kahn-Kalai / Park-Pham Expectation-Threshold Duality Engine.
#[derive(Debug, Clone, Copy)]
pub struct KahnKalaiParkPhamThreshold {
    /// Maximum witness size $\ell(\mathcal{F}) \ge 2$.
    pub max_witness_size_l: usize,
}

impl Default for KahnKalaiParkPhamThreshold {
    fn default() -> Self {
        Self {
            max_witness_size_l: 64,
        }
    }
}

impl KahnKalaiParkPhamThreshold {
    /// Creates a new Kahn-Kalai / Park-Pham Threshold engine.
    pub fn new(max_witness_size_l: usize) -> Result<Self> {
        if max_witness_size_l < 2 {
            return Err(Error::Msg("max_witness_size_l must be >= 2".to_string()));
        }
        Ok(Self { max_witness_size_l })
    }

    /// Evaluates the logarithmic gap factor $\log_2(\ell + 1)$ and the Park-Pham upper bound on the critical threshold $p_c(\mathcal{F})$.
    pub fn log_factor_and_critical_threshold_bound(
        &self,
        expectation_threshold_q: f64,
    ) -> Result<(f64, f64)> {
        if !(0.0..=1.0).contains(&expectation_threshold_q) || expectation_threshold_q == 0.0 {
            return Err(Error::Msg(
                "expectation_threshold_q must be in (0, 1]".to_string(),
            ));
        }
        let log_f = ((self.max_witness_size_l + 1) as f64).log2();
        let pc_ub = (expectation_threshold_q * log_f).min(1.0);
        Ok((log_f, pc_ub))
    }
}
