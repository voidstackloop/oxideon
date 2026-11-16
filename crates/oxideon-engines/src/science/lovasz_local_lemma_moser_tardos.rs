//! Lovász Local Lemma (LLL) & Moser-Tardos Constructive Resampling Algorithm (Erdős & Lovász, 1975; Moser & Tardos, 2010).
//!
//! ### 1. Symmetric Shearer-Lovász Condition $e p (d + 1) \le 1$ & Expected Resampling Steps
//! Given bad events $A_1, \dots, A_m$ each occurring with probability $\le p$ and depending on at most $d$ other events:
//! $$e p (d + 1) \le 1 \implies \mathbb{P}\!\left(\bigcap_{i=1}^m \overline{A_i}\right) > 0, \qquad \mathbb{E}[T_{\mathrm{MT}}] \le \sum_{i=1}^m \frac{x_i}{1 - x_i}$$
//! where $x_i = 1 / (d + 1)$ in the symmetric setting gives expected resampling steps $\le m / d$.

use crate::error::{Error, Result};
use std::f64::consts::E;

/// Lovász Local Lemma & Moser-Tardos Constructive Resampling Engine.
#[derive(Debug, Clone, Copy)]
pub struct LovaszLocalLemmaMoserTardos {
    /// Number of bad events $m \ge 1$.
    pub num_clauses_m: usize,
    /// Maximum dependency degree $d \ge 1$.
    pub max_dependency_d: usize,
}

impl Default for LovaszLocalLemmaMoserTardos {
    fn default() -> Self {
        Self {
            num_clauses_m: 100,
            max_dependency_d: 7,
        }
    }
}

impl LovaszLocalLemmaMoserTardos {
    /// Creates a new Lovász Local Lemma & Moser-Tardos engine.
    pub fn new(num_clauses_m: usize, max_dependency_d: usize) -> Result<Self> {
        if num_clauses_m == 0 || max_dependency_d == 0 {
            return Err(Error::Msg(
                "num_clauses_m and max_dependency_d must be >= 1".to_string(),
            ));
        }
        Ok(Self {
            num_clauses_m,
            max_dependency_d,
        })
    }

    /// Evaluates the LLL criterion $e p (d + 1)$, whether avoidance is guaranteed ($\le 1$), and the Moser-Tardos expected resampling bound $m / d$.
    pub fn lll_parameter_and_expected_resamplings(
        &self,
        event_prob_p: f64,
    ) -> Result<(f64, bool, f64)> {
        if !(0.0..1.0).contains(&event_prob_p) {
            return Err(Error::Msg("event_prob_p must be in [0, 1)".to_string()));
        }
        let lll_val = E * event_prob_p * (self.max_dependency_d as f64 + 1.0);
        let expected_steps = self.num_clauses_m as f64 / self.max_dependency_d as f64;
        Ok((lll_val, lll_val <= 1.0, expected_steps))
    }
}
