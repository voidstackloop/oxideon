//! Price Equation of Evolutionary Covariance & Hamilton's Rule of Inclusive Fitness (Hamilton, 1964; Price, 1970).
//!
//! ### 1. Exact Price Covariance-Transmission Decomposition
//! The generational change in mean trait value $\Delta \bar{z}$ decomposes into natural selection covariance and transmission bias:
//! $$\bar{w} \Delta \bar{z} = \mathrm{Cov}(w_i, z_i) + \mathbb{E}[w_i \Delta z_i]$$

use crate::error::{Error, Result};

/// Price Evolutionary Equation & Hamilton Inclusive Fitness Engine.
#[derive(Debug, Clone, Copy, Default)]
pub struct PriceHamiltonEvolutionEngine;

impl PriceHamiltonEvolutionEngine {
    /// Creates a new Price-Hamilton Evolution engine.
    pub fn new() -> Self {
        Self
    }

    /// Evaluates the exact Price decomposition $(\Delta \bar{z}, \mathrm{Cov}(w, z)/\bar{w}, \mathbb{E}[w \Delta z]/\bar{w})$ and checks Hamilton's kin-selection rule $r B > C$.
    pub fn price_decomposition_and_hamilton(
        fitness_w: &[f64],
        trait_z: &[f64],
        delta_z: &[f64],
        relatedness_r: f64,
        benefit_b: f64,
        cost_c: f64,
    ) -> Result<(f64, f64, f64, bool)> {
        let n = fitness_w.len();
        if n == 0 || trait_z.len() != n || delta_z.len() != n {
            return Err(Error::Msg(
                "Non-empty equal-length slices required".to_string(),
            ));
        }
        let nf = n as f64;
        let w_bar: f64 = fitness_w.iter().sum::<f64>() / nf;
        if w_bar <= 0.0 {
            return Err(Error::Msg("Mean fitness w_bar must be > 0".to_string()));
        }
        let z_bar: f64 = trait_z.iter().sum::<f64>() / nf;
        let mut cov_wz = 0.0;
        let mut exp_w_dz = 0.0;
        for i in 0..n {
            cov_wz += (fitness_w[i] - w_bar) * (trait_z[i] - z_bar);
            exp_w_dz += fitness_w[i] * delta_z[i];
        }
        cov_wz /= nf;
        exp_w_dz /= nf;
        let selection_term = cov_wz / w_bar;
        let transmission_term = exp_w_dz / w_bar;
        Ok((
            selection_term + transmission_term,
            selection_term,
            transmission_term,
            relatedness_r * benefit_b > cost_c,
        ))
    }
}
