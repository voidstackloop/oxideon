//! Lund-Fortnow-Karloff-Nisan (LFKN) Algebraic Sum-Check Protocol & $\mathsf{IP} = \mathsf{PSPACE}$ (Lund, Fortnow, Karloff & Nisan, 1992; Shamir, 1992).
//!
//! ### 1. Schwartz-Zippel Soundness Error Bound $m d / |\mathbb{F}|$ Over $m$-Variate Degree-$d$ Polynomials
//! To verify $H = \sum_{x_1, \dots, x_m \in \{0,1\}} g(x_1, \dots, x_m)$ where $g$ has individual degree $\le d$ in each variable over $\mathbb{F}_q$:
//! - Rounds: $m$
//! - Soundness error against any cheating prover $P^*$: $\epsilon_{\mathrm{sound}} \le \frac{m d}{|\mathbb{F}_q|}$

use crate::error::{Error, Result};

/// LFKN Sum-Check Protocol & Shamir IP=PSPACE Soundness Engine.
#[derive(Debug, Clone, Copy)]
pub struct SumcheckInteractiveProof {
    /// Number of variables / rounds $m \ge 1$.
    pub num_vars_m: usize,
    /// Maximum individual degree $d \ge 1$.
    pub max_degree_d: usize,
}

impl Default for SumcheckInteractiveProof {
    fn default() -> Self {
        Self {
            num_vars_m: 32,
            max_degree_d: 3,
        }
    }
}

impl SumcheckInteractiveProof {
    /// Creates a new LFKN Sum-Check Protocol engine.
    pub fn new(num_vars_m: usize, max_degree_d: usize) -> Result<Self> {
        if num_vars_m == 0 || max_degree_d == 0 {
            return Err(Error::Msg(
                "num_vars_m and max_degree_d must be >= 1".to_string(),
            ));
        }
        Ok(Self {
            num_vars_m,
            max_degree_d,
        })
    }

    /// Evaluates the total prover communication words $m (d + 1)$ and the Schwartz-Zippel soundness error upper bound $m d / q$.
    pub fn communication_words_and_soundness_error(
        &self,
        field_size_q: f64,
    ) -> Result<(usize, f64)> {
        if field_size_q <= (self.num_vars_m * self.max_degree_d) as f64 {
            return Err(Error::Msg(
                "field_size_q must exceed num_vars_m * max_degree_d".to_string(),
            ));
        }
        let words = self.num_vars_m * (self.max_degree_d + 1);
        let err = (self.num_vars_m * self.max_degree_d) as f64 / field_size_q;
        Ok((words, err))
    }
}
