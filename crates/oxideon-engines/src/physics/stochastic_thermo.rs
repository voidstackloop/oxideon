//! Non-Equilibrium Stochastic Thermodynamics: Jarzynski Equality, Crooks Fluctuation Theorem &
//! Landauer Information-Erasure Bounds.
//!
//! ### 1. Jarzynski Equality (1997)
//! For an arbitrary far-from-equilibrium finite-time driving protocol $\lambda(t)$ ($0 \le t \le \tau$)
//! at inverse temperature $\beta = (k_B T)^{-1}$, the equilibrium Helmholtz free energy difference
//! $\Delta F = F_B - F_A$ is recovered exactly from the exponential average of non-equilibrium work $W$:
//! $$\left\langle e^{-\beta W} \right\rangle = e^{-\beta \Delta F} \implies \Delta F = -\frac{1}{\beta} \ln \left( \frac{1}{N} \sum_{k=1}^N e^{-\beta W_k} \right)$$
//! By Jensen's inequality, $\langle W \rangle \ge \Delta F$, with mean dissipated work
//! $W_{\text{diss}} = \langle W \rangle - \Delta F = \beta^{-1} \langle \Delta S_{\text{tot}} \rangle \ge 0$.
//!
//! ### 2. Crooks Fluctuation Theorem (1999)
//! The ratio of forward work distribution $P_F(+W)$ to time-reversed work distribution $P_R(-W)$ obeys:
//! $$\frac{P_F(W)}{P_R(-W)} = \exp\Big(\beta (W - \Delta F)\Big)$$
//!
//! ### 3. Landauer's Principle (1961)
//! Erasing $\Delta I$ bits of Shannon entropy into a thermal reservoir at temperature $T$ requires
//! minimum dissipated heat:
//! $$Q_{\text{Landauer}} = (\Delta I \cdot \ln 2) \, k_B T = \frac{\Delta I \cdot \ln 2}{\beta}$$

use crate::error::{Error, Result};

/// Summary report of non-equilibrium work ensemble analysis via Jarzynski's Equality.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct JarzynskiReport {
    /// Arithmetic mean of irreversible work $\langle W \rangle$.
    pub mean_work: f64,
    /// Equilibrium Helmholtz free energy difference $\Delta F$ via Jarzynski exponential estimator.
    pub jarzynski_delta_f: f64,
    /// Mean dissipated work $W_{\text{diss}} = \langle W \rangle - \Delta F \ge 0$.
    pub dissipated_work: f64,
    /// Mean total entropy production $\langle \Delta S_{\text{tot}} \rangle = \beta W_{\text{diss}} \ge 0$.
    pub mean_entropy_production: f64,
}

/// Non-Equilibrium Stochastic Thermodynamics & Fluctuation Theorem Engine.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct StochasticThermoEngine {
    /// Inverse thermal energy $\beta = 1 / (k_B T) > 0$.
    pub beta: f64,
}

impl StochasticThermoEngine {
    /// Constructs a stochastic thermodynamics engine at inverse temperature $\beta > 0$.
    pub fn new(beta: f64) -> Result<Self> {
        if beta <= 0.0 {
            return Err(Error::Msg(format!(
                "Inverse temperature beta must be > 0, got {beta}"
            )));
        }
        Ok(Self { beta })
    }

    /// Evaluates Jarzynski's Equality on a sample of non-equilibrium trajectory work values $W_1, \dots, W_N$
    /// using numerically stable log-sum-exp reduction.
    pub fn evaluate_jarzynski(&self, work_samples: &[f64]) -> Result<JarzynskiReport> {
        if work_samples.is_empty() {
            return Err(Error::Msg("work_samples cannot be empty".to_string()));
        }
        let n = work_samples.len() as f64;
        let mean_work = work_samples.iter().sum::<f64>() / n;

        // Log-sum-exp for -beta * W_k
        let mut max_arg = f64::NEG_INFINITY;
        for &w in work_samples {
            max_arg = max_arg.max(-self.beta * w);
        }
        let mut sum_exp = 0.0;
        for &w in work_samples {
            sum_exp += (-self.beta * w - max_arg).exp();
        }
        let log_mean_exp = max_arg + (sum_exp / n).ln();
        let jarzynski_delta_f = -log_mean_exp / self.beta;
        let dissipated_work = (mean_work - jarzynski_delta_f).max(0.0);
        let mean_entropy_production = self.beta * dissipated_work;

        Ok(JarzynskiReport {
            mean_work,
            jarzynski_delta_f,
            dissipated_work,
            mean_entropy_production,
        })
    }

    /// Computes the exact Crooks Fluctuation Theorem log-probability ratio $\ln\left(\frac{P_F(W)}{P_R(-W)}\right) = \beta (W - \Delta F)$.
    pub fn crooks_log_ratio(&self, work_w: f64, delta_f: f64) -> f64 {
        self.beta * (work_w - delta_f)
    }

    /// Computes the Landauer minimum heat bound $Q_{\min} = \frac{\Delta I \ln 2}{\beta}$ for erasing `bits_erased` bits.
    pub fn landauer_erasure_bound(&self, bits_erased: f64) -> f64 {
        bits_erased * std::f64::consts::LN_2 / self.beta
    }
}
