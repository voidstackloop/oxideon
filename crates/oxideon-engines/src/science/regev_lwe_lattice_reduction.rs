//! Regev Learning With Errors (LWE) Worst-Case Lattice Reduction & Gaussian Error Modulus Ratio (Regev, 2005/2009).
//!
//! ### 1. Quantum Reduction from $\mathrm{GapSVP}_\gamma$ & $\mathrm{SIVP}_\gamma$ to $\mathrm{LWE}_{n, q, \alpha}$
//! For lattice dimension $n$, modulus $q$, and discrete Gaussian error width $\alpha q > 2 \sqrt{n}$:
//! $$\sigma = \alpha q > 2 \sqrt{n} \implies \gamma_{\mathrm{LWE}}(n) = \tilde{O}(n / \alpha)$$

use crate::error::{Error, Result};

/// Regev Learning With Errors (LWE) Lattice Security Engine.
#[derive(Debug, Clone, Copy)]
pub struct RegevLweLatticeSecurity {
    /// Lattice dimension $n \ge 16$.
    pub dim_n: usize,
    /// Prime or power-of-two modulus $q \ge 256$.
    pub modulus_q: u64,
}

impl Default for RegevLweLatticeSecurity {
    fn default() -> Self {
        Self {
            dim_n: 512,
            modulus_q: 3329,
        }
    }
}

impl RegevLweLatticeSecurity {
    /// Creates a new Regev LWE Lattice Security engine.
    pub fn new(dim_n: usize, modulus_q: u64) -> Result<Self> {
        if dim_n < 16 || modulus_q < 16 {
            return Err(Error::Msg(
                "dim_n >= 16 and modulus_q >= 16 required".to_string(),
            ));
        }
        Ok(Self { dim_n, modulus_q })
    }

    /// Evaluates the noise standard deviation $\sigma = \alpha q$, Regev worst-case approximation factor $n / \alpha$, and checks $\alpha q > 2\sqrt{n}$.
    pub fn noise_std_approx_factor_and_regev_check(
        &self,
        alpha_noise_rate: f64,
    ) -> Result<(f64, f64, bool)> {
        if !(0.0..1.0).contains(&alpha_noise_rate) || alpha_noise_rate == 0.0 {
            return Err(Error::Msg("alpha_noise_rate must be in (0, 1)".to_string()));
        }
        let sigma = alpha_noise_rate * (self.modulus_q as f64);
        let gamma_approx = (self.dim_n as f64) / alpha_noise_rate;
        let valid = sigma > 2.0 * (self.dim_n as f64).sqrt();
        Ok((sigma, gamma_approx, valid))
    }
}
