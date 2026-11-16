//! Neuromorphic Spiking Graph Neural Network (Spiking-GNN) Layer.
//!
//! Features:
//! - Event-driven graph message passing using Leaky Integrate-and-Fire (LIF) neurons:
//!   $$S_i\[t\] = \Theta(V_i\[t\] - V_{\text{th}})$$
//!   $$V_i\[t+1\] = \beta V_i\[t\] (1 - S_i\[t\]) + \sum_{j \in \mathcal{N}(i)} \frac{1}{\sqrt{d_i d_j}} W S_j\[t\] + I_{\text{ext}, i}\[t\]$$
//! - Ultra-low power, zero-multiplication sparse binary spike aggregation across graph topology.
//! - Multi-timestep temporal simulation with membrane potential reset and spike rate readouts.

use crate::error::{Error, Result};
use crate::nn::{Linear, Module};
use crate::tensor::Tensor;

/// Spiking Graph Neural Network Layer with Leaky Integrate-and-Fire dynamics.
pub struct SpikingGnnLayer {
    pub in_dim: usize,
    pub out_dim: usize,
    pub v_threshold: f64,
    pub decay: f64,
    pub weight: Linear,
}

impl SpikingGnnLayer {
    /// Creates a new `SpikingGnnLayer`.
    pub fn new(in_dim: usize, out_dim: usize, v_threshold: f64, decay: f64) -> Result<Self> {
        let weight = Linear::new(in_dim, out_dim)?;
        Ok(Self {
            in_dim,
            out_dim,
            v_threshold: v_threshold.max(1e-4),
            decay: decay.clamp(0.0, 1.0),
            weight,
        })
    }

    /// Single discrete time step update across all graph nodes:
    /// - `v_prev`: Membrane potentials of shape `[num_nodes, out_dim]`.
    /// - `spikes_prev`: Binary spikes emitted at previous time step of shape `[num_nodes, in_dim]`.
    /// - `edges`: Graph adjacency edges `(src, dst)`.
    /// - `ext_input`: Optional external injected current `[num_nodes, out_dim]`.
    ///
    /// Returns `(v_next, spikes_current)`.
    pub fn step(
        &self,
        v_prev: &Tensor,
        spikes_prev: &Tensor,
        edges: &[(usize, usize)],
        ext_input: Option<&Tensor>,
    ) -> Result<(Tensor, Tensor)> {
        if v_prev.ndim() < 1 {
            return Err(Error::Msg(format!(
                "SpikingGnnLayer::step needs [nodes, out_dim] potentials, got {:?}",
                v_prev.shape()
            )));
        }
        let num_nodes = v_prev.shape()[0];
        let out_d = self.out_dim;

        // 1. Transform incoming spikes through synaptic weights W * S_j
        let synaptic_spikes = self.weight.forward(spikes_prev)?;
        // Row agreement (spikes vs potentials) and column agreement (weight
        // vs tamperable out_dim): every index below rests on these.
        let syn_shape = synaptic_spikes.shape();
        if syn_shape.len() != 2 || syn_shape[0] != num_nodes || syn_shape[1] != out_d {
            return Err(Error::ShapeMismatch {
                op: "SpikingGnnLayer::step",
                expected: vec![num_nodes, out_d],
                got: syn_shape.to_vec(),
            });
        }
        let syn_data = synaptic_spikes.to_vec::<f64>()?;

        // 2. Normalized graph neighborhood aggregation
        let mut degrees = vec![1.0f64; num_nodes]; // self-loops
        let mut adj: Vec<Vec<usize>> = vec![Vec::new(); num_nodes];
        for &(src, dst) in edges {
            if src >= num_nodes || dst >= num_nodes {
                return Err(Error::IndexOutOfBounds {
                    op: "SpikingGnnLayer::step",
                    index: src.max(dst) as isize,
                    dim_size: num_nodes,
                });
            }
            adj[dst].push(src);
            degrees[dst] += 1.0;
        }

        let mut aggregated = vec![0.0f64; num_nodes * out_d];
        for dst in 0..num_nodes {
            // Self contribution
            let self_norm = 1.0 / degrees[dst].sqrt();
            for k in 0..out_d {
                aggregated[dst * out_d + k] += self_norm * syn_data[dst * out_d + k];
            }

            // Neighbor contributions
            for &src in &adj[dst] {
                let norm = 1.0 / (degrees[dst] * degrees[src]).sqrt();
                for k in 0..out_d {
                    aggregated[dst * out_d + k] += norm * syn_data[src * out_d + k];
                }
            }
        }

        let syn_current =
            Tensor::from_vec(aggregated, &[num_nodes, out_d])?.to_dtype(v_prev.dtype())?;

        // 3. Leaky integration: V_{t+1} = \beta * V_t + I_syn + I_ext
        let mut v_decayed = v_prev.mul_scalar(self.decay)?;
        v_decayed = v_decayed.add(&syn_current)?;
        if let Some(ext) = ext_input {
            v_decayed = v_decayed.add(&ext.to_dtype(v_prev.dtype())?)?;
        }

        // 4. Fire condition: S_i = 1 if V_i >= V_th else 0
        let v_data = v_decayed.to_vec::<f64>()?;
        let mut s_data = vec![0.0f64; num_nodes * out_d];
        let mut v_reset = vec![0.0f64; num_nodes * out_d];

        for idx in 0..v_data.len() {
            let val = v_data[idx];
            if val >= self.v_threshold {
                s_data[idx] = 1.0;
                v_reset[idx] = 0.0; // Hard reset to resting potential
            } else {
                s_data[idx] = 0.0;
                v_reset[idx] = val;
            }
        }

        let v_next = Tensor::from_vec(v_reset, &[num_nodes, out_d])?.to_dtype(v_prev.dtype())?;
        let spikes_curr =
            Tensor::from_vec(s_data, &[num_nodes, out_d])?.to_dtype(v_prev.dtype())?;

        Ok((v_next, spikes_curr))
    }

    /// Simulates multi-timestep Spiking GNN temporal dynamics over `time_steps`.
    /// Returns `(all_spikes [time_steps, num_nodes, out_dim], final_v [num_nodes, out_dim])`.
    pub fn forward_temporal(
        &self,
        input_spikes: &Tensor,
        edges: &[(usize, usize)],
        time_steps: usize,
    ) -> Result<(Tensor, Tensor)> {
        if input_spikes.ndim() < 1 {
            return Err(Error::Msg(format!(
                "SpikingGnnLayer::forward_temporal needs [nodes, in_dim] spikes, got {:?}",
                input_spikes.shape()
            )));
        }
        let num_nodes = input_spikes.shape()[0];
        // Each step runs a full graph pass; an uncapped count never returns.
        let time_steps = crate::limits::check_iterations("time_steps", time_steps)?;
        let state_cells = num_nodes.checked_mul(self.out_dim).ok_or_else(|| {
            Error::Msg(format!(
                "spiking state size {num_nodes} * {} overflows",
                self.out_dim
            ))
        })?;
        let mut v = Tensor::from_vec(
            crate::layout::try_filled(state_cells, 0.0f64)?,
            &[num_nodes, self.out_dim],
        )?
        .to_dtype(input_spikes.dtype())?;

        let mut spike_history: Vec<Tensor> = Vec::with_capacity(time_steps);

        for _ in 0..time_steps {
            let (v_n, s_n) = self.step(&v, input_spikes, edges, None)?;
            v = v_n;
            spike_history.push(s_n);
        }

        let all_spikes = Tensor::stack(&spike_history, 0)?;
        Ok((all_spikes, v))
    }

    pub fn parameters(&self) -> Vec<Tensor> {
        self.weight.parameters()
    }
}
