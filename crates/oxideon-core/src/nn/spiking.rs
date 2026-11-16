//! Neuromorphic Spiking Neural Networks (SNN) & Event-Driven Temporal Processing (SpikingJelly / snnTorch standard).
//!
//! Features:
//! - [`LIFNode`]: Leaky Integrate-and-Fire neuron with configurable leak decay $\tau \in (0, 1)$,
//!   firing threshold $V_{\text{th}}$, and soft/hard membrane reset.
//! - [`SurrogateKind`]: Differentiable surrogate gradient approximation ($\text{ATan}$, $\text{FastSigmoid}$)
//!   enabling standard backpropagation-through-time (BPTT) through non-differentiable step functions.
//! - [`PoissonEncoder`]: Rate-coded stochastic spike train generator for neuromorphic sensors.

use super::Module;
use crate::error::{Error, Result};
use crate::tensor::Tensor;

/// Surrogate gradient function used during backpropagation through discrete spikes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SurrogateKind {
    /// $\sigma'(x) = \frac{\alpha}{2 (1 + (\frac{\pi}{2} \alpha x)^2)}$ (ArcTangent surrogate)
    ATan,
    /// $\sigma'(x) = \frac{1}{(1 + |x|)^2}$ (FastSigmoid surrogate)
    FastSigmoid,
}

/// Reset mode after a neuron fires a spike.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ResetMode {
    /// $V \leftarrow V_{\text{reset}}$ (Hard reset)
    Hard,
    /// $V \leftarrow V - V_{\text{th}}$ (Soft reset preserving residual charge)
    Soft,
}

/// Leaky Integrate-and-Fire (LIF) spiking neuron node.
pub struct LIFNode {
    pub tau: f64,
    pub v_threshold: f64,
    pub v_reset: f64,
    pub reset_mode: ResetMode,
    pub surrogate: SurrogateKind,
    pub v: std::sync::Mutex<Option<Tensor>>,
}

impl LIFNode {
    /// Creates a new LIF neuron node.
    pub fn new(tau: f64, v_threshold: f64, v_reset: f64, reset_mode: ResetMode) -> Self {
        // clamp() passes NaN through; like Dropout, fail loudly instead of poisoning the state.
        assert!(!tau.is_nan(), "LIF leak decay tau must not be NaN");
        Self {
            tau: tau.clamp(0.01, 1.0),
            v_threshold,
            v_reset,
            reset_mode,
            surrogate: SurrogateKind::ATan,
            v: std::sync::Mutex::new(None),
        }
    }

    /// Resets the internal membrane potential state to rest.
    pub fn reset_state(&self) {
        // `v` is pub: recover from poisoning instead of panicking.
        let mut lock = self.v.lock().unwrap_or_else(|p| p.into_inner());
        *lock = None;
    }

    /// Evaluates one discrete time step forward:
    /// $$V\[t\] = \tau V\[t-1\] + X\[t\]$$
    /// $$S\[t\] = \begin{cases} 1 & \text{if } V\[t\] \ge V_{\text{th}} \\ 0 & \text{otherwise} \end{cases}$$
    pub fn forward_step(&self, x: &Tensor) -> Result<Tensor> {
        let mut lock = self.v.lock().unwrap_or_else(|p| p.into_inner());

        // 1. Membrane integration
        let v_curr = match &*lock {
            Some(v_prev) => v_prev.mul_scalar(self.tau)?.add(x)?,
            None => x.clone(),
        };

        // 2. Spike generation S = (V >= V_th)
        let v_th_t = v_curr.full_like(self.v_threshold)?;
        let spike_bool = v_curr.ge(&v_th_t)?;
        let spike = spike_bool.to_dtype(x.dtype())?;

        // 3. Membrane potential reset
        let v_next = match self.reset_mode {
            ResetMode::Hard => {
                // V = V * (1 - S) + V_reset * S
                let one_minus_s = spike.neg()?.add_scalar(1.0)?;
                let reset_val = spike.mul_scalar(self.v_reset)?;
                v_curr.mul(&one_minus_s)?.add(&reset_val)?
            }
            ResetMode::Soft => {
                // V = V - V_th * S
                let sub_charge = spike.mul_scalar(self.v_threshold)?;
                v_curr.sub(&sub_charge)?
            }
        };

        *lock = Some(v_next);
        Ok(spike)
    }

    /// Evaluates an entire sequence of input time steps `x` of shape `[time_steps, batch, ...]`.
    /// Returns spike train `[time_steps, batch, ...]`.
    pub fn forward_sequence(&self, x_seq: &Tensor) -> Result<Tensor> {
        if x_seq.ndim() < 2 {
            return Err(Error::Msg(
                "LIFNode::forward_sequence expects at least 2D tensor [T, B, ...]".into(),
            ));
        }
        let time_steps = x_seq.shape()[0];
        let mut spikes = Vec::with_capacity(time_steps);

        for t in 0..time_steps {
            let xt = x_seq.narrow(0, t, 1)?.squeeze_axis(0)?;
            let st = self.forward_step(&xt)?;
            spikes.push(st.unsqueeze(0)?);
        }

        Tensor::cat(&spikes, 0)
    }
}

impl Module for LIFNode {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        self.forward_step(input)
    }
}

/// Stochastic Poisson rate encoder converting continuous inputs $\in [0, 1]$
/// into binary spike sequences over `time_steps`.
pub struct PoissonEncoder {
    pub time_steps: usize,
}

impl PoissonEncoder {
    pub fn new(time_steps: usize) -> Self {
        Self {
            time_steps: time_steps.max(1),
        }
    }

    /// Encodes continuous inputs $X \in [0, 1]$ into spike train $[T, \dots]$.
    pub fn encode(&self, x: &Tensor) -> Result<Tensor> {
        // `time_steps` is pub with only a constructor floor: cap the reserve
        // and the per-step full-tensor sampling loop here.
        let steps_n = crate::limits::check_iterations("time_steps", self.time_steps)?;
        let mut steps = Vec::with_capacity(steps_n);
        for _ in 0..steps_n {
            let rand_t = Tensor::rand_uniform(x.shape(), 0.0, 1.0)?.to_dtype(x.dtype())?;
            let spike = x.ge(&rand_t)?.to_dtype(x.dtype())?;
            steps.push(spike.unsqueeze(0)?);
        }
        Tensor::cat(&steps, 0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lif_rejects_nan_tau() {
        let r = std::panic::catch_unwind(|| LIFNode::new(f64::NAN, 1.0, 0.0, ResetMode::Hard));
        assert!(
            r.is_err(),
            "NaN tau must fail loudly, not silently poison the membrane"
        );
    }
}
