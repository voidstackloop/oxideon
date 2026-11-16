//! Monod-Wyman-Changeux (MWC) Concerted Allosteric Cooperativity Model (Monod, Wyman & Changeux, 1965).
//!
//! ### 1. MWC Fractional Saturation $\bar{Y}(\alpha)$ for $n$-Subunit Oligomers
//! With tense-to-relaxed conformational equilibrium constant $L = \[T_0\]/\[R_0\] > 0$, microscopic affinity ratio $c = K_R / K_T \in \[0, 1\]$,
//! and normalized ligand concentration $\alpha = \[X\] / K_R \ge 0$:
//! $$\bar{Y}(\alpha) = \frac{\alpha (1 + \alpha)^{n-1} + L c \alpha (1 + c \alpha)^{n-1}}{(1 + \alpha)^n + L (1 + c \alpha)^n}$$

use crate::error::{Error, Result};

/// Monod-Wyman-Changeux (MWC) Concerted Allosteric Cooperativity Engine.
#[derive(Debug, Clone, Copy)]
pub struct MonodWymanChangeuxAllostery {
    /// Number of oligomeric subunits $n \ge 2$ (e.g. $n = 4$ for hemoglobin).
    pub subunits_n: usize,
    /// Allosteric T/R equilibrium constant $L > 0$.
    pub allosteric_l: f64,
    /// Affinity ratio $c = K_R / K_T \in [0, 1]$.
    pub ratio_c: f64,
}

impl Default for MonodWymanChangeuxAllostery {
    fn default() -> Self {
        Self {
            subunits_n: 4,
            allosteric_l: 1000.0,
            ratio_c: 0.01,
        }
    }
}

impl MonodWymanChangeuxAllostery {
    /// Creates a new Monod-Wyman-Changeux Allostery engine.
    pub fn new(subunits_n: usize, allosteric_l: f64, ratio_c: f64) -> Result<Self> {
        if subunits_n < 2 || allosteric_l <= 0.0 || !(0.0..=1.0).contains(&ratio_c) {
            return Err(Error::Msg(
                "Invalid MWC parameters (n >= 2, L > 0, c in [0, 1])".to_string(),
            ));
        }
        Ok(Self {
            subunits_n,
            allosteric_l,
            ratio_c,
        })
    }

    /// Evaluates fractional ligand saturation $\bar{Y}(\alpha)$ and relaxed-state population fraction $\bar{R}(\alpha)$ at normalized concentration $\alpha \ge 0$.
    pub fn saturation_and_relaxed_fraction(&self, alpha: f64) -> Result<(f64, f64)> {
        if alpha < 0.0 {
            return Err(Error::Msg("alpha must be >= 0".to_string()));
        }
        let n = self.subunits_n as i32;
        let r_term = (1.0 + alpha).powi(n);
        let t_term = self.allosteric_l * (1.0 + self.ratio_c * alpha).powi(n);
        let denom = r_term + t_term;
        let y_num = alpha * (1.0 + alpha).powi(n - 1)
            + self.allosteric_l * self.ratio_c * alpha * (1.0 + self.ratio_c * alpha).powi(n - 1);
        Ok((y_num / denom, r_term / denom))
    }
}
