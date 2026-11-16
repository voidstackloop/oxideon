//! Jarzynski Nonequilibrium Work Equality $\langle e^{-\beta W} \rangle = e^{-\beta \Delta F}$ & Crooks Fluctuation Theorem (Jarzynski, 1997; Crooks, 1999).
//!
//! ### 1. Exact Free Energy Recovery from Non-Equilibrium Work Trajectories
//! For a system driven arbitrarily far from thermal equilibrium at inverse temperature $\beta = 1 / (k_B T)$,
//! Jarzynski's equality relates the exponential average of irreversible work $W$ to the equilibrium Helmholtz free energy difference $\Delta F$:
//! $$\Delta F = -\frac{1}{\beta} \ln \left\langle e^{-\beta W} \right\rangle \le \langle W \rangle$$

use crate::error::{Error, Result};

/// Jarzynski Nonequilibrium Equality & Crooks Fluctuation Engine.
#[derive(Debug, Clone, Copy)]
pub struct JarzynskiCrooksEngine {
    /// Inverse thermal energy $\beta = 1 / (k_B T) > 0$.
    pub beta: f64,
}

impl JarzynskiCrooksEngine {
    /// Creates a new Jarzynski-Crooks engine.
    pub fn new(beta: f64) -> Result<Self> {
        if beta <= 0.0 {
            return Err(Error::Msg("beta must be > 0".to_string()));
        }
        Ok(Self { beta })
    }

    /// Computes the Jarzynski free energy estimate $\Delta F$ and dissipated work $W_{\text{diss}} = \langle W \rangle - \Delta F \ge 0$
    /// using the numerically stable log-sum-exp trick.
    pub fn jarzynski_free_energy_and_dissipation(
        &self,
        work_samples: &[f64],
    ) -> Result<(f64, f64)> {
        if work_samples.is_empty() {
            return Err(Error::Msg("work_samples must not be empty".to_string()));
        }
        let n = work_samples.len() as f64;
        let mean_w = work_samples.iter().sum::<f64>() / n;
        let min_w = work_samples.iter().cloned().fold(f64::INFINITY, f64::min);
        let sum_exp: f64 = work_samples
            .iter()
            .map(|&w| (-self.beta * (w - min_w)).exp())
            .sum();
        let delta_f = min_w - (sum_exp / n).ln() / self.beta;
        Ok((delta_f, (mean_w - delta_f).max(0.0)))
    }
}
