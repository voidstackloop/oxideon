//! Robinson-Schensted-Knuth (RSK) Correspondence & Frame-Robinson-Thrall Hook-Length Formula (Frame, Robinson & Thrall, 1954).
//!
//! ### 1. Hook-Length Formula for Standard Young Tableaux $f^\lambda$
//! For a partition $\lambda \vdash n$, the dimension $f^\lambda$ of the irreducible symmetric group representation $S^\lambda$ (and number of SYT) is:
//! $$f^\lambda = \frac{n!}{\prod_{(i, j) \in \lambda} h_\lambda(i, j)}, \qquad \sum_{\lambda \vdash n} (f^\lambda)^2 = n!$$

use crate::error::{Error, Result};

/// RSK Correspondence & Frame-Robinson-Thrall Hook-Length Formula Engine.
#[derive(Debug, Clone, Copy, Default)]
pub struct HookLengthYoungTableauxEngine;

impl HookLengthYoungTableauxEngine {
    /// Creates a new Hook-Length Young Tableaux engine.
    pub fn new() -> Self {
        Self
    }

    /// Evaluates the number of Standard Young Tableaux $f^\lambda$ for partition $\lambda_1 \ge \lambda_2 \ge \dots > 0$ via the exact hook-length product.
    pub fn count_standard_young_tableaux(partition: &[usize]) -> Result<u64> {
        if partition.is_empty()
            || partition.contains(&0)
            || partition.windows(2).any(|w| w[0] < w[1])
        {
            return Err(Error::Msg(
                "partition must be non-empty, positive, and non-increasing".to_string(),
            ));
        }
        let n: usize = partition.iter().sum();
        if n > 20 {
            return Err(Error::Msg(
                "Partition size n must be <= 20 for exact u64 factorial".to_string(),
            ));
        }
        let n_fact: u64 = (1..=n as u64).product();
        let mut hook_prod = 1u64;
        for (i, &row_len) in partition.iter().enumerate() {
            for j in 0..row_len {
                let arm = row_len - 1 - j;
                let leg = partition[(i + 1)..].iter().filter(|&&r| r > j).count();
                let hook = (arm + leg + 1) as u64;
                hook_prod = hook_prod.saturating_mul(hook);
            }
        }
        Ok(n_fact / hook_prod)
    }
}
