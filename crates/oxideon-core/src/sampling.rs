//! Generation sampling strategies, dynamic temperature, top-k/top-p filtering,
//! repetition/presence/frequency penalties, logit bias, and reproducible seeds.

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};

use crate::error::{Error, Result};
use crate::rng::{Generator, with_rng};
use crate::tensor::Tensor;

/// Configuration for autoregressive sampling and logit post-processing.
#[derive(Debug)]
pub struct SamplingConfig {
    /// Softmax temperature $T \ge 0$. If $T \le 10^{-6}$ or greedy, selects argmax. Default: 1.0.
    pub temperature: f64,
    /// Top-K sampling: retains only the top $K$ logits. Default: None.
    pub top_k: Option<usize>,
    /// Top-P (nucleus) sampling: retains the smallest set of tokens whose cumulative probability $\ge P$. Default: None.
    pub top_p: Option<f64>,
    /// Min-P sampling: filters tokens whose probability is $< \text{min\_p} \times p_{\max}$. Default: None.
    pub min_p: Option<f64>,
    /// Repetition penalty applied to tokens present in history (> 1.0 discourages repetition). Default: 1.0.
    pub repetition_penalty: f64,
    /// Presence penalty: additive penalty applied once per token present in history. Default: 0.0.
    pub presence_penalty: f64,
    /// Frequency penalty: additive penalty scaled by count of token in history. Default: 0.0.
    pub frequency_penalty: f64,
    /// Explicit RNG seed for reproducible generation per-request. Default: None.
    /// Without an explicit generator, repeated calls advance an internal
    /// counter mixed into the seed, forming a reproducible stream.
    pub seed: Option<u64>,
    /// Draw counter for the seed-only path. Atomic (not Cell) so the
    /// config stays Send + Sync for the Python bindings; Clone copies
    /// the count and PartialEq ignores it.
    pub seed_counter: AtomicU64,
    /// Additive biases for specific token IDs.
    pub logit_bias: HashMap<usize, f64>,
    /// Maximum new tokens to generate. Default: 128.
    pub max_new_tokens: usize,
    /// Early stopping tokens (e.g. EOS). Default: \[2\].
    pub stop_tokens: Vec<usize>,
}

impl Clone for SamplingConfig {
    fn clone(&self) -> Self {
        Self {
            temperature: self.temperature,
            top_k: self.top_k,
            top_p: self.top_p,
            min_p: self.min_p,
            repetition_penalty: self.repetition_penalty,
            presence_penalty: self.presence_penalty,
            frequency_penalty: self.frequency_penalty,
            seed: self.seed,
            seed_counter: AtomicU64::new(self.seed_counter.load(Ordering::Relaxed)),
            logit_bias: self.logit_bias.clone(),
            max_new_tokens: self.max_new_tokens,
            stop_tokens: self.stop_tokens.clone(),
        }
    }
}

impl PartialEq for SamplingConfig {
    fn eq(&self, other: &Self) -> bool {
        self.temperature == other.temperature
            && self.top_k == other.top_k
            && self.top_p == other.top_p
            && self.min_p == other.min_p
            && self.repetition_penalty == other.repetition_penalty
            && self.presence_penalty == other.presence_penalty
            && self.frequency_penalty == other.frequency_penalty
            && self.seed == other.seed
            && self.logit_bias == other.logit_bias
            && self.max_new_tokens == other.max_new_tokens
            && self.stop_tokens == other.stop_tokens
    }
}

impl Default for SamplingConfig {
    fn default() -> Self {
        Self {
            temperature: 1.0,
            top_k: None,
            top_p: None,
            min_p: None,
            repetition_penalty: 1.0,
            presence_penalty: 0.0,
            frequency_penalty: 0.0,
            seed: None,
            seed_counter: AtomicU64::new(0),
            logit_bias: HashMap::new(),
            max_new_tokens: 128,
            stop_tokens: vec![2],
        }
    }
}

impl SamplingConfig {
    pub fn new() -> Self {
        Self::default()
    }

    /// Creates a greedy decoding configuration (temperature = 0.0).
    pub fn greedy() -> Self {
        Self {
            temperature: 0.0,
            ..Self::default()
        }
    }

    pub fn with_temperature(mut self, temperature: f64) -> Self {
        self.temperature = temperature.max(0.0);
        self
    }

    pub fn with_top_k(mut self, top_k: usize) -> Self {
        self.top_k = Some(top_k);
        self
    }

    pub fn with_top_p(mut self, top_p: f64) -> Self {
        self.top_p = Some(top_p);
        self
    }

    pub fn with_min_p(mut self, min_p: f64) -> Self {
        self.min_p = Some(min_p);
        self
    }

    pub fn with_repetition_penalty(mut self, penalty: f64) -> Self {
        self.repetition_penalty = penalty;
        self
    }

    pub fn with_presence_penalty(mut self, penalty: f64) -> Self {
        self.presence_penalty = penalty;
        self
    }

    pub fn with_frequency_penalty(mut self, penalty: f64) -> Self {
        self.frequency_penalty = penalty;
        self
    }

    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = Some(seed);
        self
    }

    pub fn with_max_new_tokens(mut self, max: usize) -> Self {
        self.max_new_tokens = max;
        self
    }

    pub fn with_stop_tokens(mut self, stop_tokens: Vec<usize>) -> Self {
        self.stop_tokens = stop_tokens;
        self
    }

    pub fn with_logit_bias(mut self, token: usize, bias: f64) -> Self {
        self.logit_bias.insert(token, bias);
        self
    }

    /// Transforms raw 1D logits `[vocab_size]` and draws a sampled token index.
    pub fn sample_token(
        &self,
        logits: &Tensor,
        history: Option<&[usize]>,
        generator: Option<&mut Generator>,
    ) -> Result<usize> {
        let vocab = logits.numel();
        if vocab == 0 {
            return Err(Error::Msg("sample_token: empty logits tensor".into()));
        }
        let mut vals = logits.to_vec::<f64>()?;

        // 1. Apply logit bias
        if !self.logit_bias.is_empty() {
            for (&tok, &bias) in &self.logit_bias {
                if tok < vocab {
                    vals[tok] += bias;
                }
            }
        }

        // 2. Apply repetition, presence, frequency penalties
        if let Some(hist) = history {
            if !hist.is_empty()
                && (self.repetition_penalty != 1.0
                    || self.presence_penalty != 0.0
                    || self.frequency_penalty != 0.0)
            {
                let mut counts = HashMap::new();
                for &tok in hist {
                    *counts.entry(tok).or_insert(0usize) += 1;
                }
                for (&tok, &cnt) in &counts {
                    if tok < vocab {
                        let z = vals[tok];
                        let z_rep = if self.repetition_penalty != 1.0 {
                            if z > 0.0 {
                                z / self.repetition_penalty
                            } else {
                                z * self.repetition_penalty
                            }
                        } else {
                            z
                        };
                        vals[tok] =
                            z_rep - self.presence_penalty - (self.frequency_penalty * (cnt as f64));
                    }
                }
            }
        }

        // 3. Check temperature
        if self.temperature <= 1e-6 {
            let mut best_idx = 0;
            let mut best_val = f64::NEG_INFINITY;
            for (idx, &v) in vals.iter().enumerate() {
                if v > best_val {
                    best_val = v;
                    best_idx = idx;
                }
            }
            return Ok(best_idx);
        }

        let inv_temp = 1.0 / self.temperature;
        for v in &mut vals {
            *v *= inv_temp;
        }

        // 4. Sort indices descending. `total_cmp` is a valid total order with
        // NaN ranking first, so NaN logits behave deterministically.
        let mut order: Vec<usize> = (0..vocab).collect();
        order.sort_by(|&a, &b| vals[b].total_cmp(&vals[a]));

        // 5. Top-K filtering (HF: `top_k <= 0` disables filtering instead of
        // collapsing to argmax; oversized k is naturally a no-op).
        if let Some(k) = self.top_k {
            let k = k.min(vocab);
            if k > 0 {
                for &idx in &order[k..] {
                    vals[idx] = f64::NEG_INFINITY;
                }
            }
        }

        // 6. Softmax over remaining candidates
        let max_val = vals.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let mut probs: Vec<f64> = vals.iter().map(|&v| (v - max_val).exp()).collect();
        let sum_p: f64 = probs.iter().sum();
        if sum_p <= 0.0 || sum_p.is_nan() {
            return Ok(order[0]);
        }
        for p in &mut probs {
            *p /= sum_p;
        }

        // 7. Top-P (nucleus) filtering. HF warper order runs top-p BEFORE min-p
        // (the two do not commute once renormalized). The first rank whose
        // cumulative mass exceeds `top_p` is KEPT and everything after it is
        // dropped — including rank 0, so a lone dominant token collapses the
        // nucleus to itself (HF shifts the cumsum mask the same way). The old
        // `rank > 0` guard skipped the rank-0 cut and leaked a second token.
        // `top_p <= 0` keeps only the argmax (empty nucleus); `top_p >= 1`
        // is a no-op.
        if let Some(top_p) = self.top_p {
            if top_p < 1.0 {
                if top_p <= 0.0 {
                    for &idx in &order[1..] {
                        probs[idx] = 0.0;
                    }
                    probs[order[0]] = 1.0;
                } else {
                    let mut cum = 0.0f64;
                    for (rank, &idx) in order.iter().enumerate() {
                        cum += probs[idx];
                        // the nucleus is the smallest set whose mass is >= top_p (a token is kept only while the mass
                        // above it is < top_p, like transformers' TopPLogitsWarper); the tolerance absorbs rounding
                        // when the mass lands exactly on top_p
                        if cum >= top_p - 1e-12 {
                            for &drop_idx in &order[rank + 1..] {
                                probs[drop_idx] = 0.0;
                            }
                            break;
                        }
                    }
                    let sum_nucleus: f64 = probs.iter().sum();
                    if sum_nucleus > 0.0 {
                        for p in &mut probs {
                            *p /= sum_nucleus;
                        }
                    } else {
                        probs[order[0]] = 1.0;
                    }
                }
            }
        }

        // 8. Min-P filtering: drops tokens with `p < min_p * p_max`. `min_p <= 0`
        // is a no-op; `min_p >= 1` saturates the threshold at `p_max` so only
        // the max (plus exact ties) survives.
        if let Some(min_p) = self.min_p {
            if min_p > 0.0 {
                let p_max = probs[order[0]];
                let thresh = min_p.min(1.0) * p_max;
                for p in &mut probs {
                    if *p < thresh {
                        *p = 0.0;
                    }
                }
                let sum_filtered: f64 = probs.iter().sum();
                if sum_filtered > 0.0 {
                    for p in &mut probs {
                        *p /= sum_filtered;
                    }
                } else {
                    probs[order[0]] = 1.0;
                }
            }
        }

        // 9. Draw sample using Generator or global RNG
        let rand_val = if let Some(g) = generator {
            g.next_f64()
        } else if let Some(s) = self.seed {
            // Advance a per-config counter so repeated seeded calls without
            // an explicit generator draw an advancing reproducible stream.
            // (Rebuilding Generator::with_seed(s) on every call repeats the
            // identical first draw forever instead of sampling the tail.)
            let n = self.seed_counter.fetch_add(1, Ordering::Relaxed);
            let mut local_gen =
                Generator::with_seed(s.wrapping_add(n.wrapping_mul(0x9E3779B97F4A7C15)));
            local_gen.next_f64()
        } else {
            with_rng(|r| r.next_f32() as f64)
        };

        let mut acc = 0.0f64;
        for (idx, &p) in probs.iter().enumerate() {
            acc += p;
            if rand_val <= acc && p > 0.0 {
                return Ok(idx);
            }
        }

        Ok(order[0])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn logits(data: &[f64]) -> Tensor {
        Tensor::from_slice(data, &[data.len()]).unwrap()
    }

    #[test]
    fn greedy_picks_argmax_first_tie_and_rejects_empty() {
        let cfg = SamplingConfig::greedy();
        let x = logits(&[1.0, 5.0, 3.0]);
        assert_eq!(cfg.sample_token(&x, None, None).unwrap(), 1);
        // ties keep the first index (strict comparison)
        let tied = logits(&[2.0, 2.0, 2.0]);
        assert_eq!(cfg.sample_token(&tied, None, None).unwrap(), 0);
        let empty = Tensor::from_slice(&[] as &[f64], &[0]).unwrap();
        assert!(cfg.sample_token(&empty, None, None).is_err());
    }

    #[test]
    fn greedy_skips_nan_logits() {
        // Deliberate divergence from argmax-first-NaN: a single NaN logit
        // must not nuke greedy decoding; the best finite token wins.
        let cfg = SamplingConfig::greedy();
        let x = logits(&[1.0, f64::NAN, 3.0]);
        assert_eq!(cfg.sample_token(&x, None, None).unwrap(), 2);
        let all_nan = logits(&[f64::NAN, f64::NAN]);
        assert_eq!(cfg.sample_token(&all_nan, None, None).unwrap(), 0);
    }

    #[test]
    fn top_k_restricts_support_and_reproduces_with_seed() {
        let cfg = SamplingConfig::new().with_top_k(2).with_temperature(1.0);
        let x = logits(&[1.0, 5.0, 3.0, 0.0]);
        // the top-2 set is {1, 2}: every draw must land inside it
        let mut rng = Generator::with_seed(7);
        for _ in 0..50 {
            let tok = cfg.sample_token(&x, None, Some(&mut rng)).unwrap();
            assert!(tok == 1 || tok == 2, "outside top-k: {tok}");
        }
        // k = 1 collapses to the argmax deterministically
        let cfg1 = SamplingConfig::new().with_top_k(1).with_temperature(1.0);
        let mut gen1 = Generator::with_seed(11);
        for _ in 0..10 {
            assert_eq!(cfg1.sample_token(&x, None, Some(&mut gen1)).unwrap(), 1);
        }
        // same seed restarts the same stream
        let mut a = Generator::with_seed(99);
        let mut b = Generator::with_seed(99);
        let sa: Vec<usize> = (0..5)
            .map(|_| cfg.sample_token(&x, None, Some(&mut a)).unwrap())
            .collect();
        let sb: Vec<usize> = (0..5)
            .map(|_| cfg.sample_token(&x, None, Some(&mut b)).unwrap())
            .collect();
        assert_eq!(sa, sb);
    }

    #[test]
    fn repetition_penalty_can_flip_the_greedy_winner() {
        let cfg = SamplingConfig::greedy().with_repetition_penalty(3.0);
        let x = logits(&[5.0, 4.0]);
        assert_eq!(cfg.sample_token(&x, Some(&[0]), None).unwrap(), 1);
        assert_eq!(cfg.sample_token(&x, None, None).unwrap(), 0);
    }

    #[test]
    fn nan_logit_sorts_first_deterministically() {
        // total order: NaN ranks above everything, so top_k=1 keeps it and
        // the NaN-poisoned distribution deterministically falls back to it.
        let cfg = SamplingConfig::new().with_top_k(1).with_temperature(1.0);
        let x = logits(&[1.0, f64::NAN, 0.0]);
        let mut rng = Generator::with_seed(3);
        assert_eq!(cfg.sample_token(&x, None, Some(&mut rng)).unwrap(), 1);
    }

    /// HF parity: when the top token alone already exceeds `top_p`, the nucleus
    /// collapses to that single token (HF shifts the cumsum mask so rank 0 is kept
    /// and everything after the first exceeding rank is dropped). The old
    /// `rank > 0` guard skipped the rank-0 cut and leaked a second token.
    #[test]
    fn top_p_keeps_single_token_when_it_alone_exceeds_p() {
        let cfg = SamplingConfig::new().with_top_p(0.4).with_temperature(1.0);
        // softmax is ~[0.5, 0.5, 0, 0]: top_p=0.4 < p_max.
        let x = logits(&[2.0, 2.0, -10.0, -10.0]);
        let mut rng = Generator::with_seed(21);
        for _ in 0..50 {
            assert_eq!(cfg.sample_token(&x, None, Some(&mut rng)).unwrap(), 0);
        }
    }

    /// HF warper order is top-p *then* min-p (sequential warpers in the HF logits
    /// processor list). With probs [0.6, 0.3, 0.1], top_p=0.5 cuts to the top
    /// token before min_p=0.4 ever sees the distribution, so every draw is 0;
    /// min-p-first would renormalize to [2/3, 1/3] and leak token 1.
    #[test]
    fn combined_top_p_and_min_p_follow_hf_warper_order() {
        let cfg = SamplingConfig::new()
            .with_top_p(0.5)
            .with_min_p(0.4)
            .with_temperature(1.0);
        let x = logits(&[6.0f64.ln(), 3.0f64.ln(), 1.0f64.ln()]);
        let mut rng = Generator::with_seed(22);
        for _ in 0..50 {
            assert_eq!(cfg.sample_token(&x, None, Some(&mut rng)).unwrap(), 0);
        }
    }

    /// HF parity for degenerate knobs: `top_k=0` disables filtering (HF guards
    /// `top_k > 0`; clamping 0 to 1 would silently collapse to argmax).
    #[test]
    fn top_k_zero_disables_filtering() {
        let cfg = SamplingConfig::new().with_top_k(0).with_temperature(1.0);
        let x = logits(&[3.0, 2.0, 1.0, 0.0]);
        let mut rng = Generator::with_seed(23);
        let mut seen = [false; 4];
        for _ in 0..200 {
            let tok = cfg.sample_token(&x, None, Some(&mut rng)).unwrap();
            seen[tok] = true;
        }
        assert!(
            seen[1] && seen[2],
            "top_k=0 must keep full support: {seen:?}"
        );
    }

    /// Degenerate-clamp contract, HF-compatible: `min_p=1.0` keeps only the max
    /// (`p >= p_max`), `top_p=0.0` keeps only the argmax (empty nucleus), while
    /// `top_p=1.0` and oversized `top_k` are no-ops (full support).
    #[test]
    fn degenerate_clamps_match_hf() {
        let x = logits(&[1.0, 2.0, 3.0]);
        let max_only = SamplingConfig::new().with_min_p(1.0).with_temperature(1.0);
        let mut rng = Generator::with_seed(24);
        for _ in 0..20 {
            assert_eq!(max_only.sample_token(&x, None, Some(&mut rng)).unwrap(), 2);
        }
        let nucleus_empty = SamplingConfig::new().with_top_p(0.0).with_temperature(1.0);
        let mut rng = Generator::with_seed(25);
        for _ in 0..20 {
            assert_eq!(
                nucleus_empty
                    .sample_token(&x, None, Some(&mut rng))
                    .unwrap(),
                2
            );
        }
        for cfg in [
            SamplingConfig::new().with_top_p(1.0).with_temperature(1.0),
            SamplingConfig::new().with_top_k(99).with_temperature(1.0),
        ] {
            let mut rng = Generator::with_seed(26);
            let mut seen = [false; 3];
            for _ in 0..200 {
                let tok = cfg.sample_token(&x, None, Some(&mut rng)).unwrap();
                seen[tok] = true;
            }
            assert!(seen.iter().all(|&s| s), "full support expected: {seen:?}");
        }
    }

    /// Presence/frequency penalties (OpenAI subtractive convention) and logit bias
    /// shift greedy decoding; out-of-vocab bias/history ids are ignored, not errors.
    #[test]
    fn presence_frequency_and_bias_math() {
        let x = logits(&[3.0, 3.0]);
        let pres = SamplingConfig::greedy().with_presence_penalty(1.0);
        assert_eq!(pres.sample_token(&x, Some(&[0]), None).unwrap(), 1);
        assert_eq!(pres.sample_token(&x, None, None).unwrap(), 0);
        let freq = SamplingConfig::greedy().with_frequency_penalty(1.0);
        assert_eq!(freq.sample_token(&x, Some(&[0, 0, 0]), None).unwrap(), 1);
        let bias = SamplingConfig::greedy().with_logit_bias(1, 5.0);
        assert_eq!(bias.sample_token(&x, None, None).unwrap(), 1);
        // out-of-range ids are skipped silently (no panic, no error)
        let oob = SamplingConfig::greedy().with_logit_bias(99, 100.0);
        assert_eq!(oob.sample_token(&x, Some(&[77]), None).unwrap(), 0);
    }

    /// Without any filter, a NaN logit poisons the whole softmax; the fallback is
    /// the deterministic first-ranked token (NaN sorts first), never a panic.
    #[test]
    fn unfiltered_nan_falls_back_to_first_ranked() {
        let cfg = SamplingConfig::new().with_temperature(1.0);
        let x = logits(&[1.0, f64::NAN, 0.0]);
        let mut rng = Generator::with_seed(27);
        for _ in 0..10 {
            assert_eq!(cfg.sample_token(&x, None, Some(&mut rng)).unwrap(), 1);
        }
    }

    /// Negative temperatures are meaningless; the builder clamps to greedy (0.0).
    #[test]
    fn negative_temperature_clamps_to_greedy() {
        let cfg = SamplingConfig::new().with_temperature(-2.0);
        assert_eq!(cfg.temperature, 0.0);
        let x = logits(&[1.0, 5.0, 3.0]);
        assert_eq!(cfg.sample_token(&x, None, None).unwrap(), 1);
    }
}
