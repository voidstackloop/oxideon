//! Stochastic Chemical Master Equation (CME), Schnakenberg Network Entropy Production & Thermodynamic Uncertainty Relation (Schnakenberg, 1976; Barato & Seifert, 2015).
//!
//! ### 1. Schnakenberg Entropy Production Rate & Barato-Seifert TUR Bound
//! For a continuous-time Markov jump network with forward and backward transition fluxes $J_{ij}^+ = p_i w_{ij} > 0$
//! and $J_{ij}^- = p_j w_{ji} > 0$, the steady-state entropy production rate is:
//! $$\sigma = \sum_{i < j} (J_{ij}^+ - J_{ij}^-) \ln\left(\frac{J_{ij}^+}{J_{ij}^-}\right) \ge 0$$
//! And any integrated current $J$ with mean $\langle J \rangle$ and variance $\operatorname{Var}(J)$ obeys the
//! **Thermodynamic Uncertainty Relation (TUR)**: $\frac{\operatorname{Var}(J)}{\langle J \rangle^2} \sigma \ge 2 k_B$.

use crate::error::{Error, Result};

/// Schnakenberg Master Equation Entropy Production & TUR Engine.
#[derive(Debug, Clone, Copy, Default)]
pub struct ChemicalMasterEntropyEngine;

impl ChemicalMasterEntropyEngine {
    /// Creates a new Schnakenberg Chemical Master Entropy engine.
    pub fn new() -> Self {
        Self
    }

    /// Computes the non-negative Schnakenberg entropy production rate $\sigma = \sum (J^+ - J^-) \ln(J^+ / J^-) \ge 0$ across edges.
    pub fn schnakenberg_entropy_production(
        forward_fluxes: &[f64],
        backward_fluxes: &[f64],
    ) -> Result<f64> {
        if forward_fluxes.is_empty()
            || forward_fluxes.len() != backward_fluxes.len()
            || forward_fluxes.iter().any(|&j| j <= 0.0)
            || backward_fluxes.iter().any(|&j| j <= 0.0)
        {
            return Err(Error::Msg(
                "Positive equal-length forward and backward edge fluxes required".to_string(),
            ));
        }
        let mut sigma = 0.0;
        for (jp, jm) in forward_fluxes.iter().zip(backward_fluxes.iter()) {
            sigma += (jp - jm) * (jp / jm).ln();
        }
        Ok(sigma)
    }
}
