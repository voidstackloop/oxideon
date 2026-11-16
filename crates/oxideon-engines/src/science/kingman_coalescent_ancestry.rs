//! Kingman $n$-Coalescent Genealogy, Expected $T_{\mathrm{MRCA}}$ & Watterson $\theta_W$ Estimator (Kingman, 1982; Watterson, 1975).
//!
//! ### 1. Coalescent Waiting Times & Total Branch Length
//! In Kingman's coalescent for a sample of $n \ge 2$ lineages in coalescent time units ($2 N_e$ generations),
//! the waiting time with $k$ lineages has rate $\binom{k}{2}$, yielding:
//! $$\mathbb{E}[T_{\mathrm{MRCA}}] = 2\!\left(1 - \frac{1}{n}\right), \qquad \mathbb{E}[L_{\mathrm{total}}] = 2 H_{n-1} = 2 \sum_{k=1}^{n-1} \frac{1}{k}$$

use crate::error::{Error, Result};

/// Kingman $n$-Coalescent & Watterson Nucleotide Diversity Engine.
#[derive(Debug, Clone, Copy)]
pub struct KingmanCoalescentEngine {
    /// Sample size $n \ge 2$.
    pub sample_size_n: usize,
}

impl Default for KingmanCoalescentEngine {
    fn default() -> Self {
        Self { sample_size_n: 10 }
    }
}

impl KingmanCoalescentEngine {
    /// Creates a new Kingman Coalescent engine.
    pub fn new(sample_size_n: usize) -> Result<Self> {
        if sample_size_n < 2 {
            return Err(Error::Msg("Sample size n must be >= 2".to_string()));
        }
        Ok(Self { sample_size_n })
    }

    /// Evaluates expected $\mathbb{E}[T_{\mathrm{MRCA}}] = 2(1 - 1/n)$, expected total tree length $2 H_{n-1}$, and Watterson's estimator $\hat{\theta}_W = S / H_{n-1}$.
    pub fn tmrca_tree_length_and_watterson(&self, segregating_sites_s: usize) -> (f64, f64, f64) {
        let n = self.sample_size_n as f64;
        let e_tmrca = 2.0 * (1.0 - 1.0 / n);
        let h_n_minus_1: f64 = (1..self.sample_size_n).map(|k| 1.0 / (k as f64)).sum();
        let e_total_length = 2.0 * h_n_minus_1;
        let theta_w = (segregating_sites_s as f64) / h_n_minus_1;
        (e_tmrca, e_total_length, theta_w)
    }
}
