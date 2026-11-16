//! Huang-Kueng-Preskill Classical Shadow Tomography & Median-of-Means Estimators (Huang, Kueng & Preskill, 2020).
//!
//! ### 1. Classical Shadows via Randomized Pauli Measurements
//! Measuring an $n$-qubit quantum state $\rho$ in random single-qubit Pauli bases $U = \bigotimes_{i=1}^n U_i$
//! and inverting the depolarizing channel $\mathcal{M}^{-1}_1(\sigma) = 3 \sigma - I_2$ yields an unbiased classical
//! snapshot $\hat{\rho} = \bigotimes_{i=1}^n (3 U_i^\dagger |b_i\rangle\langle b_i| U_i - I_2)$ with $\mathbb{E}[\hat{\rho}] = \rho$.
//! Predicting $M$ arbitrary $k$-local Pauli observables to accuracy $\epsilon$ requires only:
//! $$N_{\text{shadows}} = \mathcal{O}\left(\frac{3^k \ln M}{\epsilon^2}\right)$$

use crate::error::{Error, Result};

/// Huang-Kueng-Preskill Classical Shadow Tomography Engine.
#[derive(Debug, Clone, Copy, Default)]
pub struct ClassicalShadowTomography;

impl ClassicalShadowTomography {
    /// Creates a new Classical Shadow Tomography engine.
    pub fn new() -> Self {
        Self
    }

    /// Computes the Huang-Kueng-Preskill sample complexity bound $N = \lceil 34 \cdot 3^k \ln(2 M / \delta) / \epsilon^2 \rceil$
    /// for $M$ observables of Pauli weight $k$.
    pub fn sample_complexity_bound(
        pauli_weight_k: u32,
        num_observables_m: usize,
        eps: f64,
        failure_delta: f64,
    ) -> Result<u64> {
        if num_observables_m == 0
            || !eps.is_finite()
            || eps <= 0.0
            || !(0.0..1.0).contains(&failure_delta)
        {
            return Err(Error::Msg(
                "Invalid shadow tomography parameters".to_string(),
            ));
        }
        let shadow_norm_sq = 3.0_f64.powi(pauli_weight_k as i32);
        let n = 34.0 * shadow_norm_sq * (2.0 * (num_observables_m as f64) / failure_delta).ln()
            / (eps * eps);
        Ok(n.ceil() as u64)
    }

    /// Computes the robust Median-of-Means estimator over $K$ blocks of snapshot expectations.
    pub fn median_of_means(samples: &[f64], num_blocks_k: usize) -> Result<f64> {
        if samples.is_empty() || num_blocks_k == 0 || samples.len() < num_blocks_k {
            return Err(Error::Msg(
                "Invalid samples or num_blocks_k in median_of_means".to_string(),
            ));
        }
        let block_size = samples.len() / num_blocks_k;
        let mut means = Vec::with_capacity(num_blocks_k);
        for b in 0..num_blocks_k {
            let slice = &samples[b * block_size..(b + 1) * block_size];
            let m: f64 = slice.iter().sum::<f64>() / (block_size as f64);
            means.push(m);
        }
        means.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        Ok(means[num_blocks_k / 2])
    }
}
