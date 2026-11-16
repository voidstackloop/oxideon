//! Ryan Williams Algorithmic Method: SAT-to-Circuit Lower Bound Duality ($\mathsf{NEXP} \not\subset \mathsf{ACC}^0$, Williams, 2011/2014).
//!
//! ### 1. Williams Algorithmic Speedup vs. Superpolynomial Circuit Lower Bounds
//! If $\mathcal{C}$-SAT on $n$-input circuits of size $n^k$ can be solved in deterministic time $O(2^n / n^c)$ for every $c$
//! (via the Polynomial Method and fast rectangular matrix multiplication), then $\mathsf{NEXP} \not\subset \mathcal{C}$.

use crate::error::{Error, Result};

/// Ryan Williams SAT-to-Circuit Duality & Beigel-Tarui $\mathsf{SYM}^+$ Polynomial Degree Engine.
#[derive(Debug, Clone, Copy)]
pub struct WilliamsSatCircuitLowerBound {
    /// Number of input Boolean variables $n \ge 2$.
    pub num_vars_n: usize,
}

impl Default for WilliamsSatCircuitLowerBound {
    fn default() -> Self {
        Self { num_vars_n: 20 }
    }
}

impl WilliamsSatCircuitLowerBound {
    /// Creates a new Williams SAT-to-Circuit Duality engine.
    pub fn new(num_vars_n: usize) -> Result<Self> {
        if num_vars_n < 2 {
            return Err(Error::Msg("num_vars_n must be >= 2".to_string()));
        }
        Ok(Self { num_vars_n })
    }

    /// Evaluates the exhaustive $2^n$ search cost vs. Williams split-and-evaluate matrix-multiplication cost $2^{n - \ell} + 2^{\omega \ell / 3}$ for split size $\ell < n$.
    pub fn speedup_factor(
        &self,
        split_vars_l: usize,
        omega_exponent: f64,
    ) -> Result<(f64, f64, bool)> {
        if split_vars_l == 0
            || split_vars_l >= self.num_vars_n
            || !(2.0..=3.0).contains(&omega_exponent)
        {
            return Err(Error::Msg(
                "Invalid split_vars_l or matrix multiplication exponent omega".to_string(),
            ));
        }
        let n = self.num_vars_n as f64;
        let l = split_vars_l as f64;
        let brute_force = 2.0_f64.powf(n);
        let fast_eval = 2.0_f64.powf(n - l) + 2.0_f64.powf(omega_exponent * l / 3.0);
        let speedup = brute_force / fast_eval;
        Ok((fast_eval, speedup, speedup > 1.0))
    }
}
