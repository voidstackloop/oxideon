//! Spike-Timing-Dependent Plasticity (STDP) for Neuromorphic Synaptic Learning.
//!
//! Implements local, biologically grounded, unsupervised synaptic adaptation:
//! - Exponential decay traces for pre- and post-synaptic spike trains:
//!   $$\tau_+ \frac{d x_i}{dt} = -x_i + S_{\text{pre}, i}(t)$$
//!   $$\tau_- \frac{d y_j}{dt} = -y_j + S_{\text{post}, j}(t)$$
//! - Asymmetric Hebbian STDP learning rule:
//!   - Long-Term Potentiation (LTP) when pre fires before post: $\Delta W_{ji} = A_+ \cdot x_i$ upon $S_{\text{post}, j} = 1$
//!   - Long-Term Depression (LTD) when post fires before pre: $\Delta W_{ji} = -A_- \cdot y_j$ upon $S_{\text{pre}, i} = 1$
//! - Hard or soft weight clamping to $[w_{\min}, w_{\max}]$.

use crate::dtype::DType;
use crate::error::{Error, Result};
use crate::tensor::Tensor;

/// Configuration hyperparameters for Spike-Timing-Dependent Plasticity.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct STDPConfig {
    /// Potentiation amplitude $A_+ > 0$.
    pub a_plus: f64,
    /// Depression amplitude $A_- > 0$.
    pub a_minus: f64,
    /// Presynaptic trace decay time constant $\tau_+$.
    pub tau_plus: f64,
    /// Postsynaptic trace decay time constant $\tau_-$.
    pub tau_minus: f64,
    /// Minimum synaptic weight bound.
    pub w_min: f64,
    /// Maximum synaptic weight bound.
    pub w_max: f64,
    /// Simulation timestep $dt$.
    pub dt: f64,
}

impl Default for STDPConfig {
    fn default() -> Self {
        Self {
            a_plus: 0.01,
            a_minus: 0.012,
            tau_plus: 20.0,
            tau_minus: 20.0,
            w_min: 0.0,
            w_max: 1.0,
            dt: 1.0,
        }
    }
}

/// Neuromorphic Synapse with online trace-based STDP learning.
#[derive(Debug, Clone)]
pub struct STDPSynapse {
    pub in_features: usize,
    pub out_features: usize,
    pub weights: Tensor,
    pub trace_pre: Tensor,
    pub trace_post: Tensor,
    pub config: STDPConfig,
}

impl STDPSynapse {
    /// Creates a new STDP synapse with uniformly initialized weights in $[w_{\min}, w_{\max}]$.
    pub fn new(in_features: usize, out_features: usize, config: STDPConfig) -> Result<Self> {
        let initial_w = (config.w_min + config.w_max) / 2.0;
        let cells = crate::layout::checked_numel(&[in_features, out_features])?;
        let weights = Tensor::from_vec(
            crate::layout::try_filled(cells, initial_w)?,
            &[out_features, in_features],
        )?;
        let trace_pre = Tensor::zeros(&[in_features], DType::F64, &weights.device())?;
        let trace_post = Tensor::zeros(&[out_features], DType::F64, &weights.device())?;

        Ok(Self {
            in_features,
            out_features,
            weights,
            trace_pre,
            trace_post,
            config,
        })
    }

    /// Computes the forward postsynaptic current $I = W \cdot S_{\text{pre}}$.
    pub fn forward(&self, pre_spikes: &Tensor) -> Result<Tensor> {
        let s = if pre_spikes.dtype() != self.weights.dtype() {
            pre_spikes.to_dtype(self.weights.dtype())?
        } else {
            pre_spikes.clone()
        };

        if s.ndim() == 1 {
            let s_col = s.unsqueeze(1)?;
            let current = self.weights.matmul(&s_col)?;
            current.squeeze()
        } else {
            // [Batch, In] -> [Batch, Out] via W^T
            let wt = self.weights.t()?;
            s.matmul(&wt)
        }
    }

    /// Performs one online STDP update step given binary/rate spikes from pre and post populations.
    pub fn step_stdp(&mut self, pre_spikes: &Tensor, post_spikes: &Tensor) -> Result<()> {
        let pre_vec = pre_spikes.to_vec::<f64>()?;
        let post_vec = post_spikes.to_vec::<f64>()?;

        if pre_vec.len() != self.in_features || post_vec.len() != self.out_features {
            return Err(Error::Msg(format!(
                "STDP shape mismatch: expected in={}, out={}, got pre={}, post={}",
                self.in_features,
                self.out_features,
                pre_vec.len(),
                post_vec.len()
            )));
        }
        // Weights/traces are pub: the update below indexes them by feature
        // count, so a short swap would read out of bounds.
        let need_w = self
            .out_features
            .checked_mul(self.in_features)
            .ok_or_else(|| Error::Msg("STDP weight size overflows".into()))?;
        if self.weights.numel() != need_w
            || self.trace_pre.numel() != self.in_features
            || self.trace_post.numel() != self.out_features
        {
            return Err(Error::Msg(format!(
                "STDP state mismatch: weights numel={} (need {}), traces {}/{} (need {}/{})",
                self.weights.numel(),
                need_w,
                self.trace_pre.numel(),
                self.trace_post.numel(),
                self.in_features,
                self.out_features
            )));
        }
        // `clamp` panics when min > max (or NaN): reject inverted bounds.
        if self.config.w_min.is_nan()
            || self.config.w_max.is_nan()
            || self.config.w_min > self.config.w_max
        {
            return Err(Error::Msg(format!(
                "STDP needs w_min <= w_max, got {} > {}",
                self.config.w_min, self.config.w_max
            )));
        }

        // Trace decay factors: exp(-dt / tau)
        let decay_pre = (-self.config.dt / self.config.tau_plus).exp();
        let decay_post = (-self.config.dt / self.config.tau_minus).exp();

        let mut trace_pre_vec = self.trace_pre.to_vec::<f64>()?;
        let mut trace_post_vec = self.trace_post.to_vec::<f64>()?;
        let mut w_vec = self.weights.to_vec::<f64>()?;

        // 1. Decay traces
        for tp in &mut trace_pre_vec {
            *tp *= decay_pre;
        }
        for tp in &mut trace_post_vec {
            *tp *= decay_post;
        }

        // 2. Synaptic plasticity update
        // W is [out_features, in_features] -> index = j * in_features + i
        for j in 0..self.out_features {
            let s_post = post_vec[j];
            for i in 0..self.in_features {
                let s_pre = pre_vec[i];
                let idx = j * self.in_features + i;

                // LTP: post fires, potentiation proportional to pre trace
                let dw_ltp = if s_post > 0.0 {
                    self.config.a_plus * trace_pre_vec[i] * s_post
                } else {
                    0.0
                };

                // LTD: pre fires, depression proportional to post trace
                let dw_ltd = if s_pre > 0.0 {
                    self.config.a_minus * trace_post_vec[j] * s_pre
                } else {
                    0.0
                };

                let updated_w =
                    (w_vec[idx] + dw_ltp - dw_ltd).clamp(self.config.w_min, self.config.w_max);
                w_vec[idx] = updated_w;
            }
        }

        // 3. Accumulate new spikes into traces
        for (tp, &sp) in trace_pre_vec.iter_mut().zip(pre_vec.iter()) {
            *tp += sp;
        }
        for (tp, &sp) in trace_post_vec.iter_mut().zip(post_vec.iter()) {
            *tp += sp;
        }

        // 4. Update tensors
        self.weights = Tensor::from_vec(w_vec, &[self.out_features, self.in_features])?;
        self.trace_pre = Tensor::from_vec(trace_pre_vec, &[self.in_features])?;
        self.trace_post = Tensor::from_vec(trace_post_vec, &[self.out_features])?;

        Ok(())
    }

    /// Resets synaptic traces to zero without modifying learned weights.
    pub fn reset_traces(&mut self) -> Result<()> {
        self.trace_pre = Tensor::zeros(&[self.in_features], DType::F64, &self.weights.device())?;
        self.trace_post = Tensor::zeros(&[self.out_features], DType::F64, &self.weights.device())?;
        Ok(())
    }
}
