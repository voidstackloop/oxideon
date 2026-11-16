//! Lions Concentration-Compactness Principle & Aubin-Talenti Critical Sobolev Bubble Energy (Aubin, 1976; Talenti, 1976; Lions, 1984).
//!
//! ### 1. Critical Sobolev Exponent $2^* = 2n / (n - 2)$ & Mountain-Pass Energy Threshold
//! In $\mathbb{R}^n$ ($n \ge 3$), the critical Sobolev embedding $H^1(\mathbb{R}^n) \hookrightarrow L^{2^*}(\mathbb{R}^n)$ loses compactness due to dilation invariance
//! $u_\lambda(x) = \lambda^{(n-2)/2} u(\lambda x)$. By Lions' Concentration-Compactness Principle, Palais-Smale sequences are precompact below the first bubble threshold $c^* = \frac{1}{n} S_n^{n/2}$.

use crate::error::{Error, Result};

/// Lions Concentration-Compactness & Critical Sobolev Palais-Smale Engine.
#[derive(Debug, Clone, Copy)]
pub struct LionsConcentrationCompactness {
    /// Spatial dimension $n \ge 3$.
    pub dim_n: usize,
    /// Best Sobolev constant $S_n = \inf \|\nabla u\|_2^2 / \|u\|_{2^*}^2 > 0$.
    pub sobolev_sn: f64,
}

impl Default for LionsConcentrationCompactness {
    fn default() -> Self {
        Self {
            dim_n: 3,
            sobolev_sn: 5.4779,
        }
    }
}

impl LionsConcentrationCompactness {
    /// Creates a new Lions Concentration-Compactness engine.
    pub fn new(dim_n: usize, sobolev_sn: f64) -> Result<Self> {
        if dim_n < 3 || sobolev_sn <= 0.0 {
            return Err(Error::Msg(
                "dim_n >= 3 and sobolev_sn > 0 required".to_string(),
            ));
        }
        Ok(Self { dim_n, sobolev_sn })
    }

    /// Evaluates the critical Sobolev exponent $2^* = 2n/(n-2)$, Brézis-Nirenberg / Aubin-Talenti bubble energy $c^* = \frac{1}{n} S_n^{n/2}$, and compactness check ($E < c^*$).
    pub fn critical_exponent_and_bubble_threshold(&self, energy_level: f64) -> (f64, f64, bool) {
        let n = self.dim_n as f64;
        let two_star = 2.0 * n / (n - 2.0);
        let c_star = self.sobolev_sn.powf(0.5 * n) / n;
        (two_star, c_star, energy_level < c_star)
    }
}
