//! O-Minimal Structures $\mathbb{R}_{\mathrm{an,exp}}$ (Wilkie, 1996; van den Dries, 1998) & Khovanskii Pfaffian Fewnomial Bound.
//!
//! ### 1. Khovanskii Bound on Connected Components of Pfaffian Varieties
//! For a Pfaffian chain of order $r$ and degree $\alpha$, a system of Pfaffian equations of degree $\beta$ in $\mathbb{R}^n$
//! has finitely many connected components (establishing O-minimality of $\mathbb{R}_{\mathrm{exp}}$), bounded by:
//! $$B(n, r, \alpha, \beta) = 2^{r(r-1)/2} \beta^n (\min(n, r) \alpha + 2 \beta - 1)^{n - 1} ((n + 1) \alpha + 2 \beta - 1)^r$$

use crate::error::{Error, Result};

/// O-Minimal Cell Decomposition & Khovanskii Pfaffian Component Bound Engine.
#[derive(Debug, Clone, Copy)]
pub struct OMinimalPfaffianCellEngine {
    /// Ambient dimension $n \ge 1$.
    pub dim_n: usize,
    /// Pfaffian chain length / order $r \ge 1$.
    pub chain_order_r: usize,
}

impl Default for OMinimalPfaffianCellEngine {
    fn default() -> Self {
        Self {
            dim_n: 2,
            chain_order_r: 2,
        }
    }
}

impl OMinimalPfaffianCellEngine {
    /// Creates a new O-Minimal Pfaffian Cell engine.
    pub fn new(dim_n: usize, chain_order_r: usize) -> Result<Self> {
        if dim_n == 0 || chain_order_r == 0 {
            return Err(Error::Msg(
                "dim_n >= 1 and chain_order_r >= 1 required".to_string(),
            ));
        }
        Ok(Self {
            dim_n,
            chain_order_r,
        })
    }

    /// Evaluates the Khovanskii upper bound on the number of connected components for chain degree $\alpha \ge 1$ and polynomial degree $\beta \ge 1$.
    pub fn khovanskii_component_bound(&self, alpha: usize, beta: usize) -> Result<f64> {
        if alpha == 0 || beta == 0 {
            return Err(Error::Msg(
                "Degrees alpha and beta must be >= 1".to_string(),
            ));
        }
        let n = self.dim_n as f64;
        let r = self.chain_order_r as f64;
        let a = alpha as f64;
        let b = beta as f64;
        let m = n.min(r);
        let bound = 2.0_f64.powf(0.5 * r * (r - 1.0))
            * b.powf(n)
            * (m * a + 2.0 * b - 1.0).powf(n - 1.0)
            * ((n + 1.0) * a + 2.0 * b - 1.0).powf(r);
        Ok(bound)
    }
}
