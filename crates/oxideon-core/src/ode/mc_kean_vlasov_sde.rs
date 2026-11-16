//! McKean-Vlasov Mean-Field SDEs, Sznitman Propagation of Chaos & Granular Media Free Energy (McKean, 1966; Sznitman, 1991).
//!
//! ### 1. Quadratic Kuramoto / Harmonics McKean-Vlasov Interaction $dX_t = -\kappa (X_t - \mathbb{E}\[X_t\]) dt + \sigma dB_t$
//! For $N$ weakly interacting particles $dX_t^{i,N} = -\frac{\kappa}{N}\sum_{j=1}^N (X_t^{i,N} - X_t^{j,N}) dt + \sigma dB_t^i$,
//! Sznitman's **Propagation of Chaos** proves that the empirical measure $\mu_t^N = \frac{1}{N} \sum \delta_{X_t^i}$ converges in
//! Wasserstein distance $W_2(\mu_t^N, \mu_t) = \mathcal{O}(N^{-1/2})$ to the McKean-Vlasov law with variance:
//! $$\operatorname{Var}(X_t) = \operatorname{Var}(X_0) e^{-2 \kappa t} + \frac{\sigma^2}{2 \kappa}\left(1 - e^{-2 \kappa t}\right)$$

use crate::error::{Error, Result};

/// McKean-Vlasov Mean-Field SDE & Propagation of Chaos Engine.
#[derive(Debug, Clone, Copy)]
pub struct McKeanVlasovPropagationChaos {
    /// Mean-reverting interaction coupling $\kappa > 0$.
    pub interaction_kappa: f64,
    /// Diffusion coefficient $\sigma > 0$.
    pub sigma: f64,
}

impl McKeanVlasovPropagationChaos {
    /// Creates a new McKean-Vlasov Mean-Field SDE engine.
    pub fn new(interaction_kappa: f64, sigma: f64) -> Result<Self> {
        if interaction_kappa <= 0.0 || sigma <= 0.0 {
            return Err(Error::Msg(
                "interaction_kappa and sigma must be > 0".to_string(),
            ));
        }
        Ok(Self {
            interaction_kappa,
            sigma,
        })
    }

    /// Evaluates the exact McKean-Vlasov variance $\operatorname{Var}(X_t)$ at time $t \ge 0$ from initial variance $v_0 \ge 0$.
    pub fn mean_field_variance(&self, v0: f64, t: f64) -> Result<f64> {
        if v0 < 0.0 || !v0.is_finite() || t < 0.0 || !t.is_finite() {
            return Err(Error::Msg(format!(
                "mean_field_variance needs finite v0 >= 0 and t >= 0, got v0={v0}, t={t}"
            )));
        }
        let decay = (-2.0 * self.interaction_kappa * t).exp();
        let v_inf = (self.sigma * self.sigma) / (2.0 * self.interaction_kappa);
        Ok(v0 * decay + v_inf * (1.0 - decay))
    }
}
