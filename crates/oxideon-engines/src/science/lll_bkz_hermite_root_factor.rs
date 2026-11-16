//! Lenstra-Lenstra-Lovász (LLL) & Block Korkine-Zolotarev (BKZ) Root Hermite Factor $\delta_0(\beta)$ (Lenstra, Lenstra & Lovász, 1982; Schnorr, 1987).
//!
//! ### 1. LLL Approximation Factor $2^{(n-1)/2}$ & Asymptotic BKZ-$\beta$ Root Hermite Factor
//! In dimension $n$, LLL finds a nonzero lattice vector within $(2 / \sqrt{4 \delta - 1})^{n-1}$ of $\lambda_1(\Lambda)$, while BKZ with blocksize $\beta \ge 50$ achieves root Hermite factor:
//! $$\delta_0(\beta) \approx \left(\frac{\beta}{2 \pi e} (\pi \beta)^{1/\beta}\right)^{\frac{1}{2(\beta - 1)}}, \qquad \|\mathbf{b}_1\| \approx \delta_0(\beta)^n \det(\Lambda)^{1/n}$$

use crate::error::{Error, Result};
use std::f64::consts::{E, PI};

/// LLL & BKZ Lattice Basis Reduction Root Hermite Factor Engine.
#[derive(Debug, Clone, Copy)]
pub struct LllBkzLatticeReduction {
    /// BKZ blocksize $\beta \ge 36$.
    pub blocksize_beta: usize,
}

impl Default for LllBkzLatticeReduction {
    fn default() -> Self {
        Self {
            blocksize_beta: 100,
        }
    }
}

impl LllBkzLatticeReduction {
    /// Creates a new LLL/BKZ Lattice Reduction engine.
    pub fn new(blocksize_beta: usize) -> Result<Self> {
        if blocksize_beta < 36 {
            return Err(Error::Msg("blocksize_beta must be >= 36".to_string()));
        }
        Ok(Self { blocksize_beta })
    }

    /// Evaluates the Chen-Nguyen asymptotic BKZ root Hermite factor $\delta_0(\beta)$ and Core-SVP bit security $0.292 \beta$.
    pub fn root_hermite_factor_and_core_svp_bits(&self) -> (f64, f64) {
        let b = self.blocksize_beta as f64;
        let base = (b / (2.0 * PI * E)) * (PI * b).powf(1.0 / b);
        let delta0 = base.powf(1.0 / (2.0 * (b - 1.0)));
        let bits = 0.292 * b;
        (delta0, bits)
    }
}
