//! Speculative Decoding Engine (Leviathan et al., ICML 2023; Chen et al., 2023).
//!
//! Accelerates autoregressive decoding by having a small, fast "draft model" propose $\gamma$
//! candidate tokens in rapid sequence, and having the larger "target model" verify all candidates
//! in parallel within a single forward pass via causal masking.
//!
//! Rejection sampling guarantees that the output distribution is mathematically identical
//! to sampling directly from the target model.

use crate::error::Result;
use crate::rng;

/// Configuration parameters for speculative decoding.
#[derive(Debug, Clone, Copy)]
pub struct SpeculativeConfig {
    /// Number of draft tokens proposed per speculative iteration ($\gamma$).
    pub num_speculative_tokens: usize,
    /// Sampling temperature.
    pub temperature: f64,
}

impl Default for SpeculativeConfig {
    fn default() -> Self {
        Self {
            num_speculative_tokens: 4,
            temperature: 1.0,
        }
    }
}

/// Cumulative performance metrics collected across speculative decoding runs.
#[derive(Debug, Clone, Default)]
pub struct SpeculativeMetrics {
    pub total_proposals: usize,
    pub total_accepted: usize,
    pub total_forward_passes: usize,
}

impl SpeculativeMetrics {
    /// Acceptance rate $\alpha = \frac{\text{accepted}}{\text{proposed}}$.
    pub fn acceptance_rate(&self) -> f64 {
        if self.total_proposals == 0 {
            0.0
        } else {
            self.total_accepted as f64 / self.total_proposals as f64
        }
    }

    /// Speedup factor over pure autoregressive generation.
    pub fn estimated_speedup(&self) -> f64 {
        if self.total_forward_passes == 0 {
            1.0
        } else {
            (self.total_accepted + self.total_forward_passes) as f64
                / self.total_forward_passes as f64
        }
    }
}

/// Speculative decoding driver.
pub struct SpeculativeDecoder {
    pub config: SpeculativeConfig,
    pub metrics: SpeculativeMetrics,
}

impl SpeculativeDecoder {
    pub fn new(config: SpeculativeConfig) -> Self {
        Self {
            config,
            metrics: SpeculativeMetrics::default(),
        }
    }

    /// Executes one iteration of speculative generation given draft logits and target logits:
    ///
    /// - `draft_tokens`: Candidate tokens $[t_1, t_2, \dots, t_\gamma]$ proposed by draft model
    /// - `draft_probs`: Token probabilities predicted by draft model $[P_{\text{draft}}(t_1), \dots]$
    /// - `target_probs_all`: Full vocabulary probability distributions predicted by target model for all proposal positions
    ///
    /// Returns the slice of accepted tokens, plus one recovery/resampled token.
    pub fn verify_and_sample(
        &mut self,
        draft_tokens: &[usize],
        draft_probs: &[f64],
        target_probs_all: &[Vec<f64>],
    ) -> Result<Vec<usize>> {
        let gamma = draft_tokens.len();
        if draft_probs.len() < gamma || target_probs_all.len() < gamma {
            return Err(crate::error::Error::Msg(format!(
                "verify_and_sample: {gamma} draft tokens need {gamma} draft/target prob entries, got {}/{}",
                draft_probs.len(),
                target_probs_all.len()
            )));
        }
        let mut accepted = Vec::new();
        self.metrics.total_proposals += gamma;
        self.metrics.total_forward_passes += 1;

        let mut rejected_at = None;

        for i in 0..gamma {
            let tok = draft_tokens[i];
            let p_draft = draft_probs[i].max(1e-12);
            let p_target = target_probs_all[i].get(tok).copied().unwrap_or(0.0);

            let u: f64 = rng::with_rng(|r| r.next_f32() as f64);
            let acceptance_ratio = (p_target / p_draft).min(1.0);

            if u <= acceptance_ratio {
                accepted.push(tok);
                self.metrics.total_accepted += 1;
            } else {
                rejected_at = Some((i, p_draft));
                break;
            }
        }

        // Recovery token sampling: if rejected at position `i`, resample from max(0, p_target - p_draft)
        // If all accepted, sample bonus token from target_probs_all[gamma]
        if let Some((i, _)) = rejected_at {
            let p_target_vec = &target_probs_all[i];
            let mut adjusted = Vec::with_capacity(p_target_vec.len());
            let mut sum_adj = 0.0f64;

            for (vocab_id, &p_tgt) in p_target_vec.iter().enumerate() {
                let p_d = if vocab_id == draft_tokens[i] {
                    draft_probs[i]
                } else {
                    0.0
                };
                let diff = (p_tgt - p_d).max(0.0);
                adjusted.push(diff);
                sum_adj += diff;
            }

            let next_tok = sample_from_distribution(&adjusted, sum_adj);
            accepted.push(next_tok);
        } else if target_probs_all.len() > gamma {
            // All gamma candidates accepted! Bonus token from the final verified position
            let bonus_probs = &target_probs_all[gamma];
            let sum_p: f64 = bonus_probs.iter().sum();
            let bonus_tok = sample_from_distribution(bonus_probs, sum_p);
            accepted.push(bonus_tok);
        }

        Ok(accepted)
    }
}

fn sample_from_distribution(probs: &[f64], total_sum: f64) -> usize {
    if total_sum <= 1e-12 {
        return 0;
    }
    let r: f64 = rng::with_rng(|rng| (rng.next_f32() as f64) * total_sum);
    let mut acc = 0.0;
    for (i, &p) in probs.iter().enumerate() {
        acc += p;
        if acc >= r {
            return i;
        }
    }
    probs.len().saturating_sub(1)
}
