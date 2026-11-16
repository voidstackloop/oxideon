//! Penington-Almheiri-Engelhardt-Molfetas-Maxfield Quantum Extremal Surface (QES) Island Rule & Page Curve (Page, 1993; Penington, 2020; Almheiri et al., 2019).
//!
//! ### 1. Island Formula $S(\mathbf{R}) = \min \operatorname{ext}_I \left[\frac{\operatorname{Area}(\partial I)}{4 G_N} + S_{\mathrm{semi}}(\mathbf{R} \cup I)\right]$
//! For an evaporating black hole of initial Bekenstein-Hawking entropy $S_{\mathrm{BH}}(0) = S_0$ radiating at constant rate $\Gamma$, the unitary radiation entropy follows the Page curve:
//! $$S_{\mathrm{rad}}(t) = \min\!\left(\Gamma t, \; S_0 - \Gamma t\right), \qquad t_{\mathrm{Page}} = \frac{S_0}{2 \Gamma}$$

use crate::error::{Error, Result};

/// Quantum Extremal Surface (QES) Island & Unitary Page Curve Engine.
#[derive(Debug, Clone, Copy)]
pub struct QuantumExtremalSurfaceIslandPage {
    /// Initial Bekenstein-Hawking entropy $S_0 > 0$.
    pub initial_bh_entropy_s0: f64,
    /// Hawking entropy emission rate $\Gamma > 0$.
    pub emission_rate_gamma: f64,
}

impl Default for QuantumExtremalSurfaceIslandPage {
    fn default() -> Self {
        Self {
            initial_bh_entropy_s0: 100.0,
            emission_rate_gamma: 2.0,
        }
    }
}

impl QuantumExtremalSurfaceIslandPage {
    /// Creates a new Quantum Extremal Surface & Page Curve engine.
    pub fn new(initial_bh_entropy_s0: f64, emission_rate_gamma: f64) -> Result<Self> {
        if initial_bh_entropy_s0 <= 0.0 || emission_rate_gamma <= 0.0 {
            return Err(Error::Msg(
                "initial_bh_entropy_s0 and emission_rate_gamma must be > 0".to_string(),
            ));
        }
        Ok(Self {
            initial_bh_entropy_s0,
            emission_rate_gamma,
        })
    }

    /// Evaluates the Page time $t_{\mathrm{Page}} = S_0 / (2\Gamma)$, unitary QES radiation entropy $S_{\mathrm{rad}}(t)$, and whether the entanglement island dominates ($t > t_{\mathrm{Page}}$).
    pub fn page_time_entropy_and_island_dominance(&self, time_t: f64) -> Result<(f64, f64, bool)> {
        if time_t < 0.0 || !time_t.is_finite() {
            return Err(Error::Msg("time_t must be >= 0".to_string()));
        }
        let t_page = self.initial_bh_entropy_s0 / (2.0 * self.emission_rate_gamma);
        let s_hawking = self.emission_rate_gamma * time_t;
        let s_island = (self.initial_bh_entropy_s0 - s_hawking).max(0.0);
        let s_rad = s_hawking.min(s_island);
        Ok((t_page, s_rad, s_island < s_hawking))
    }
}
