//! CRISPR-Cas9 R-Loop Zipping Free-Energy Landscape & Off-Target Mismatch Discrimination (Jinek et al., 2012; Klein, Depken et al., 2018).
//!
//! ### 1. Hybridization Free-Energy Barrier & Cleavage Probability
//! During 20-nt guide-RNA R-loop invasion from the PAM, each matched base pair stabilizes the R-loop by $-\epsilon_{\mathrm{match}} < 0$
//! while a mismatch at position $m$ incurs penalty $+\Delta \epsilon_{\mathrm{mm}} > 0$, yielding Boltzmann cleavage probability $P_{\mathrm{cleave}} = (1 + e^{\Delta G_{\mathrm{tot}} / k_B T})^{-1}$.

use crate::error::{Error, Result};

/// CRISPR-Cas9 R-Loop Zipping & Off-Target Specificity Engine.
#[derive(Debug, Clone, Copy)]
pub struct CrisprCas9RLoopKinetics {
    /// Per-base-pair R-loop stabilization energy $\epsilon_{\mathrm{match}} > 0$ (in $k_B T$ units).
    pub match_gain_kt: f64,
    /// Per-mismatch energetic penalty $\Delta \epsilon_{\mathrm{mm}} > 0$ (in $k_B T$ units).
    pub mismatch_penalty_kt: f64,
}

impl Default for CrisprCas9RLoopKinetics {
    fn default() -> Self {
        Self {
            match_gain_kt: 0.5,
            mismatch_penalty_kt: 4.0,
        }
    }
}

impl CrisprCas9RLoopKinetics {
    /// Creates a new CRISPR-Cas9 R-Loop Kinetics engine.
    pub fn new(match_gain_kt: f64, mismatch_penalty_kt: f64) -> Result<Self> {
        if match_gain_kt <= 0.0 || mismatch_penalty_kt <= 0.0 {
            return Err(Error::Msg(
                "match_gain_kt and mismatch_penalty_kt must be > 0".to_string(),
            ));
        }
        Ok(Self {
            match_gain_kt,
            mismatch_penalty_kt,
        })
    }

    /// Evaluates the net R-loop free energy $\Delta G / k_B T$ and cleavage probability $1 / (1 + e^{\Delta G / k_B T})$ for guide length $L$ with $m$ mismatches.
    pub fn rloop_free_energy_and_cleavage_prob(
        &self,
        guide_len_l: usize,
        num_mismatches_m: usize,
    ) -> Result<(f64, f64)> {
        if guide_len_l == 0 || num_mismatches_m > guide_len_l {
            return Err(Error::Msg(
                "0 <= num_mismatches_m <= guide_len_l and guide_len_l >= 1 required".to_string(),
            ));
        }
        let matched = (guide_len_l - num_mismatches_m) as f64;
        let dg =
            -matched * self.match_gain_kt + (num_mismatches_m as f64) * self.mismatch_penalty_kt;
        let p_cleave = 1.0 / (1.0 + dg.exp());
        Ok((dg, p_cleave))
    }
}
