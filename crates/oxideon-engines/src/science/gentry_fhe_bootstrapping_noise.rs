//! Gentry Fully Homomorphic Encryption (FHE) Multiplicative Noise Growth & Modulus Switching (Gentry, 2009; Brakerski, Gentry & Vaikuntanathan, 2012).
//!
//! ### 1. BGV/BFV Multiplicative Noise Expansion & Decryption Correctness Bound $\|e_{\mathrm{final}}\|_\infty < q / (2 t)$
//! For plaintext modulus $t$, ciphertext modulus $q$, and ring expansion factor $\gamma_{\mathrm{ring}} \approx 2 \sqrt{N}$, a depth-$L$ circuit without modulus switching grows noise as $(t \gamma_{\mathrm{ring}} B_{\mathrm{clean}})^{2^L}$, whereas BGV modulus switching reduces growth to linear in $L$.

use crate::error::{Error, Result};

/// Gentry FHE & BGV Noise Budget Engine.
#[derive(Debug, Clone, Copy)]
pub struct GentryFheNoiseGrowthEngine {
    /// Ring degree $N$ (power of two $\ge 1024$).
    pub ring_degree_n: usize,
    /// Plaintext modulus $t \ge 2$.
    pub plaintext_modulus_t: u64,
}

impl Default for GentryFheNoiseGrowthEngine {
    fn default() -> Self {
        Self {
            ring_degree_n: 4096,
            plaintext_modulus_t: 65537,
        }
    }
}

impl GentryFheNoiseGrowthEngine {
    /// Creates a new Gentry FHE Noise Growth engine.
    pub fn new(ring_degree_n: usize, plaintext_modulus_t: u64) -> Result<Self> {
        if ring_degree_n < 16 || plaintext_modulus_t < 2 {
            return Err(Error::Msg(
                "ring_degree_n >= 16 and plaintext_modulus_t >= 2 required".to_string(),
            ));
        }
        Ok(Self {
            ring_degree_n,
            plaintext_modulus_t,
        })
    }

    /// Evaluates the remaining noise budget $\log_2(q / (2 t \|e\|))$ in bits and checks if decryption succeeds ($> 0$).
    pub fn remaining_noise_budget_bits(
        &self,
        log2_q: f64,
        current_noise_norm: f64,
    ) -> Result<(f64, bool)> {
        if log2_q <= 2.0 || current_noise_norm <= 0.0 {
            return Err(Error::Msg(
                "log2_q > 2 and current_noise_norm > 0 required".to_string(),
            ));
        }
        let budget =
            log2_q - 1.0 - (self.plaintext_modulus_t as f64).log2() - current_noise_norm.log2();
        Ok((budget, budget > 0.0))
    }
}
