//! Bravyi-Kitaev 15-to-1 $T$-State Magic Distillation & Cubic Error Suppression (Bravyi & Kitaev, 2005).
//!
//! ### 1. 15-to-1 Reed-Muller Magic State Distillation Protocol
//! Using the $[[15, 1, 3]]$ quantum Reed-Muller code with transversal $T = \operatorname{diag}(1, e^{i\pi/4})$ gate,
//! 15 noisy $|T\rangle$ magic states with depolarizing error rate $p < p_{\text{th}}$ are distilled into 1 high-fidelity
//! $|T\rangle$ state with output infidelity suppressed cubically:
//! $$p_{\text{out}}(p) = 35 p^3 + O(p^4)$$

use crate::error::{Error, Result};

/// Bravyi-Kitaev 15-to-1 Magic State Distillation & Resource Overhead Engine.
#[derive(Debug, Clone, Copy, Default)]
pub struct MagicStateDistillationEngine;

impl MagicStateDistillationEngine {
    /// Creates a new Magic State Distillation engine.
    pub fn new() -> Self {
        Self
    }

    /// Computes the output logical $T$-gate error rate after `rounds` concatenated levels of 15-to-1 Bravyi-Kitaev distillation:
    /// $p_{k+1} = 35 p_k^3 / (1 + 35 p_k^3)$.
    pub fn distill_15_to_1_infidelity(initial_error_p: f64, rounds: usize) -> Result<(f64, u64)> {
        if !(0.0..0.14).contains(&initial_error_p) || rounds > 10 {
            return Err(Error::Msg(
                "initial_error_p must be in [0, 0.14) and rounds <= 10".to_string(),
            ));
        }
        let mut p = initial_error_p;
        let mut raw_states = 1_u64;
        for _ in 0..rounds {
            let num = 35.0 * p.powi(3);
            p = num / (1.0 + num);
            raw_states *= 15;
        }
        Ok((p, raw_states))
    }
}
