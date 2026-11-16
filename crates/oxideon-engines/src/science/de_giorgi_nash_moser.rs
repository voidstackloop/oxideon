//! De Giorgi–Nash–Moser Elliptic Regularity, Oscillation Decay & Harnack Inequality (De Giorgi, 1957; Nash, 1958; Moser, 1961).
//!
//! ### 1. Oscillation Contraction & Hölder Exponent $\alpha \in (0, 1)$
//! For uniformly elliptic divergence-form equations $-\nabla \cdot (A(x) \nabla u) = 0$ with $\lambda I \le A(x) \le \Lambda I$,
//! $\mathrm{osc}_{B_r}(u) \le (1 - \theta) \mathrm{osc}_{B_{2r}}(u)$, yielding interior $C^{0, \alpha}$ Hölder continuity with $\alpha = -\log_2(1 - \theta)$ and Harnack constant $C_H = (\Lambda / \lambda)^{c \sqrt{n}}$.

use crate::error::{Error, Result};

/// De Giorgi–Nash–Moser Elliptic Hölder Regularity & Harnack Engine.
#[derive(Debug, Clone, Copy)]
pub struct DeGiorgiNashMoserRegularity {
    /// Lower ellipticity bound $\lambda > 0$.
    pub lambda_min: f64,
    /// Upper ellipticity bound $\Lambda \ge \lambda$.
    pub lambda_max: f64,
}

impl Default for DeGiorgiNashMoserRegularity {
    fn default() -> Self {
        Self {
            lambda_min: 1.0,
            lambda_max: 2.0,
        }
    }
}

impl DeGiorgiNashMoserRegularity {
    /// Creates a new De Giorgi–Nash–Moser Regularity engine.
    pub fn new(lambda_min: f64, lambda_max: f64) -> Result<Self> {
        if lambda_min <= 0.0 || lambda_max < lambda_min {
            return Err(Error::Msg(
                "0 < lambda_min <= lambda_max required".to_string(),
            ));
        }
        Ok(Self {
            lambda_min,
            lambda_max,
        })
    }

    /// Evaluates the ellipticity condition number $\kappa = \Lambda / \lambda$, effective Hölder exponent $\alpha = -\log_2(1 - (2\kappa)^{-n})$, and Harnack constant $(\Lambda/\lambda)^{\sqrt{n}}$.
    pub fn holder_exponent_and_harnack_constant(&self, dim_n: usize) -> Result<(f64, f64)> {
        if dim_n == 0 {
            return Err(Error::Msg("dim_n must be >= 1".to_string()));
        }
        let kappa = self.lambda_max / self.lambda_min;
        let theta = (2.0 * kappa).powf(-(dim_n as f64));
        let alpha = -(1.0 - theta).log2();
        let c_harnack = kappa.powf((dim_n as f64).sqrt());
        Ok((alpha, c_harnack))
    }
}
