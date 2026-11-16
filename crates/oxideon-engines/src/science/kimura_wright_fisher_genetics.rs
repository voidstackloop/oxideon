//! Wright-Fisher Diffusion & Kimura Fixation Probability in Population Genetics (Wright, 1931; Kimura, 1962).
//!
//! ### 1. Kimura Exact Diffusion Fixation Probability
//! For a diploid population of effective size $N_e$, selection coefficient $s$, and initial allele frequency $p \in (0, 1)$,
//! the probability of ultimate fixation under genic selection is:
//! $$u(p) = \frac{1 - e^{-4 N_e s p}}{1 - e^{-4 N_e s}}$$

use crate::error::{Error, Result};

/// Wright-Fisher Diffusion & Kimura Fixation Probability Engine.
#[derive(Debug, Clone, Copy)]
pub struct KimuraWrightFisherEngine {
    /// Effective diploid population size $N_e > 0$.
    pub effective_pop_ne: f64,
}

impl Default for KimuraWrightFisherEngine {
    fn default() -> Self {
        Self {
            effective_pop_ne: 1000.0,
        }
    }
}

impl KimuraWrightFisherEngine {
    /// Creates a new Kimura Wright-Fisher engine.
    pub fn new(effective_pop_ne: f64) -> Result<Self> {
        if effective_pop_ne <= 0.0 || !effective_pop_ne.is_finite() {
            return Err(Error::Msg("effective_pop_ne must be > 0".to_string()));
        }
        Ok(Self { effective_pop_ne })
    }

    /// Evaluates Kimura's fixation probability $u(p)$ for selection coefficient $s$ and initial frequency $p \in [0, 1]$.
    pub fn fixation_probability(&self, selection_s: f64, initial_freq_p: f64) -> Result<f64> {
        if !(0.0..=1.0).contains(&initial_freq_p) {
            return Err(Error::Msg("initial_freq_p must be in [0, 1]".to_string()));
        }
        let gamma = 4.0 * self.effective_pop_ne * selection_s;
        if gamma.abs() < 1e-10 {
            Ok(initial_freq_p)
        } else {
            Ok((1.0 - (-gamma * initial_freq_p).exp()) / (1.0 - (-gamma).exp()))
        }
    }
}
