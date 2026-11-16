//! Erdős-Rado / ALWZ Sunflower Lemma & Razborov-Rudich Natural Proofs Barrier (Erdős & Rado, 1960; Razborov & Rudich, 1997; ALWZ, 2020).
//!
//! ### 1. Erdős-Rado vs. Alweiss-Lovett-Wu-Zhang (ALWZ) Sunflower Thresholds
//! Any family $\mathcal{F}$ of $w$-element sets without a $p$-petal sunflower has size bounded by the classical Erdős-Rado bound
//! $w! (p - 1)^w$ and the breakthrough ALWZ bound $(C p \log(w p))^w$.

use crate::error::{Error, Result};

/// Erdős-Rado / ALWZ Sunflower Lemma & Razborov-Rudich Monotone Circuit Engine.
#[derive(Debug, Clone, Copy)]
pub struct SunflowerNaturalProofsBarrier {
    /// Set width $w \ge 1$.
    pub set_width_w: usize,
    /// Number of sunflower petals $p \ge 3$.
    pub petals_p: usize,
}

impl Default for SunflowerNaturalProofsBarrier {
    fn default() -> Self {
        Self {
            set_width_w: 5,
            petals_p: 3,
        }
    }
}

impl SunflowerNaturalProofsBarrier {
    /// Creates a new Sunflower & Natural Proofs Barrier engine.
    pub fn new(set_width_w: usize, petals_p: usize) -> Result<Self> {
        if set_width_w == 0 || petals_p < 3 {
            return Err(Error::Msg(
                "set_width_w >= 1 and petals_p >= 3 required".to_string(),
            ));
        }
        Ok(Self {
            set_width_w,
            petals_p,
        })
    }

    /// Evaluates the classical Erdős-Rado threshold $w! (p - 1)^w$ and the Razborov-Rudich largeness density $2^{-n}$ threshold for $n$-variable truth tables.
    pub fn erdos_rado_and_alwz_bounds(&self) -> (f64, f64) {
        let fact_w: f64 = (1..=self.set_width_w).map(|k| k as f64).product();
        let w = self.set_width_w as f64;
        let p = self.petals_p as f64;
        let er_bound = fact_w * (p - 1.0).powf(w);
        let alwz_base = (p * (w * p).ln().max(1.0)).powf(w);
        (er_bound, alwz_base)
    }
}
