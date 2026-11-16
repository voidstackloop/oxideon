//! Rockafellar-Uryasev Conditional Value-at-Risk ($\mathrm{CVaR}_\alpha$) Convex Optimization & Artzner Coherent Risk (Rockafellar & Uryasev, 2000).
//!
//! ### 1. Variational Representation of Expected Shortfall ($\mathrm{CVaR}_\alpha$)
//! For loss random variable $L$ and confidence level $\alpha \in (0, 1)$, Rockafellar & Uryasev proved the convex program identity:
//! $$\mathrm{CVaR}_\alpha(L) = \min_{\zeta \in \mathbb{R}} \left\{ \zeta + \frac{1}{1 - \alpha} \mathbb{E}[(L - \zeta)_+] \right\}$$
//! where the optimal threshold $\zeta^*$ equals $\mathrm{VaR}_\alpha(L)$.

use crate::error::{Error, Result};

/// Rockafellar-Uryasev Conditional Value-at-Risk ($\mathrm{CVaR}_\alpha$) Engine.
#[derive(Debug, Clone, Copy)]
pub struct RockafellarUryasevCVaREngine {
    /// Tail confidence level $\alpha \in (0, 1)$ (e.g., $0.95$).
    pub confidence_alpha: f64,
}

impl Default for RockafellarUryasevCVaREngine {
    fn default() -> Self {
        Self {
            confidence_alpha: 0.95,
        }
    }
}

impl RockafellarUryasevCVaREngine {
    /// Creates a new Rockafellar-Uryasev CVaR engine.
    pub fn new(confidence_alpha: f64) -> Result<Self> {
        if !(0.0..1.0).contains(&confidence_alpha) || confidence_alpha == 0.0 {
            return Err(Error::Msg("confidence_alpha must be in (0, 1)".to_string()));
        }
        Ok(Self { confidence_alpha })
    }

    /// Evaluates empirical $(\mathrm{VaR}_\alpha, \mathrm{CVaR}_\alpha)$ via the Rockafellar-Uryasev variational formula over scenario losses.
    pub fn empirical_var_and_cvar(&self, scenario_losses: &[f64]) -> Result<(f64, f64)> {
        if scenario_losses.is_empty() {
            return Err(Error::Msg("scenario_losses must be non-empty".to_string()));
        }
        let mut sorted = scenario_losses.to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let n = sorted.len();
        let idx = ((self.confidence_alpha * (n as f64)).ceil() as usize)
            .saturating_sub(1)
            .min(n - 1);
        let var_zeta = sorted[idx];
        let excess_mean: f64 =
            sorted.iter().map(|&l| (l - var_zeta).max(0.0)).sum::<f64>() / (n as f64);
        let cvar = var_zeta + excess_mean / (1.0 - self.confidence_alpha);
        Ok((var_zeta, cvar))
    }
}
