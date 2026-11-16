//! Eisenberg-Noe Systemic Interbank Clearing Vector & Contagion Fixed-Point Theorem (Eisenberg & Noe, 2001).
//!
//! ### 1. Clearing Payment Fixed-Point Map $\Phi(p)$
//! In an $n$-bank network with external cash flows $e_i \ge 0$, total obligations $\bar{p}_i > 0$, and relative liability matrix $\Pi_{ij}$,
//! the unique clearing payment vector $p^*$ under limited liability and proportional priority solves:
//! $$p_i^* = \min\!\left(\bar{p}_i, \, e_i + \sum_{j=1}^{n} \Pi_{ji} p_j^*\right)$$

use crate::error::{Error, Result};

/// Eisenberg-Noe Interbank Systemic Contagion & Clearing Vector Engine.
#[derive(Debug, Clone, Copy, Default)]
pub struct EisenbergNoeSystemicClearing;

impl EisenbergNoeSystemicClearing {
    /// Creates a new Eisenberg-Noe Systemic Clearing engine.
    pub fn new() -> Self {
        Self
    }

    /// Computes the greatest fixed-point clearing vector $p^*$ via monotone Tarski iteration for a 2-bank symmetric ring with external assets $(e_1, e_2)$ and total obligations $(\bar{p}_1, \bar{p}_2)$.
    pub fn solve_two_bank_ring_clearing(
        e1: f64,
        e2: f64,
        pbar1: f64,
        pbar2: f64,
        cross_share_alpha: f64,
    ) -> Result<(f64, f64, usize)> {
        if e1 < 0.0
            || e2 < 0.0
            || pbar1 <= 0.0
            || pbar2 <= 0.0
            || !(0.0..1.0).contains(&cross_share_alpha)
        {
            return Err(Error::Msg(
                "Invalid interbank clearing parameters".to_string(),
            ));
        }
        let mut p1 = pbar1;
        let mut p2 = pbar2;
        for _ in 0..100 {
            let next1 = pbar1.min(e1 + cross_share_alpha * p2);
            let next2 = pbar2.min(e2 + cross_share_alpha * p1);
            if (next1 - p1).abs() + (next2 - p2).abs() < 1e-12 {
                p1 = next1;
                p2 = next2;
                break;
            }
            p1 = next1;
            p2 = next2;
        }
        let defaults = usize::from(p1 < pbar1 - 1e-9) + usize::from(p2 < pbar2 - 1e-9);
        Ok((p1, p2, defaults))
    }
}
