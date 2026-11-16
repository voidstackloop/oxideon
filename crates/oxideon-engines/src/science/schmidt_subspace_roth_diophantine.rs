//! Roth's Theorem on Diophantine Approximation $|\alpha - p/q| > c / q^{2+\varepsilon}$ & Schmidt Subspace Theorem (Roth, 1955; Schmidt, 1972).
//!
//! ### 1. Irrationality Measure $\mu(\alpha) = 2$ for Algebraic Numbers & Liouville Bound
//! By Roth's theorem, every irrational algebraic number $\alpha \in \overline{\mathbb{Q}} \setminus \mathbb{Q}$ of degree $d \ge 2$ has exact irrationality exponent $\mu(\alpha) = 2$, refining Liouville's bound $|\alpha - p/q| \ge c / q^d$.

use crate::error::{Error, Result};

/// Roth's Theorem & Schmidt Subspace Diophantine Engine.
#[derive(Debug, Clone, Copy)]
pub struct RothSchmidtDiophantineEngine {
    /// Algebraic degree $d \ge 2$ of $\alpha$.
    pub algebraic_degree_d: usize,
}

impl Default for RothSchmidtDiophantineEngine {
    fn default() -> Self {
        Self {
            algebraic_degree_d: 3,
        }
    }
}

impl RothSchmidtDiophantineEngine {
    /// Creates a new Roth-Schmidt Diophantine engine.
    pub fn new(algebraic_degree_d: usize) -> Result<Self> {
        if algebraic_degree_d < 2 {
            return Err(Error::Msg("algebraic_degree_d must be >= 2".to_string()));
        }
        Ok(Self { algebraic_degree_d })
    }

    /// Evaluates the Roth irrationality measure $\mu(\alpha) = 2.0$, the Liouville exponent $d$, and the Roth lower-bound scale $q^{-(2 + \varepsilon)}$.
    pub fn roth_exponent_and_scale(
        &self,
        denominator_q: u64,
        eps: f64,
    ) -> Result<(f64, usize, f64)> {
        if denominator_q < 2 || eps <= 0.0 {
            return Err(Error::Msg(
                "denominator_q >= 2 and eps > 0 required".to_string(),
            ));
        }
        let scale = (denominator_q as f64).powf(-(2.0 + eps));
        Ok((2.0, self.algebraic_degree_d, scale))
    }
}
