//! Neuromorphic Event-Driven Spike Routing & Synaptic Delay Plasticity.
//!
//! # Theoretical Framework
//!
//! 1. **Continuous-Time Event-Driven Spike Routing**:
//!    Spikes are discrete events scheduled in a priority queue.
//!    The arrival time at target neuron $v$ is:
//!    $$t_{\text{arrival}} = t_{\text{spike}} + \tau_{uv}$$
//!
//! 2. **Continuous-Time Leaky Integrate-and-Fire (LIF / SRM0)**:
//!    Between spike arrivals at times $t_{\text{prev}}$ and $t_{\text{curr}}$, membrane potential decays:
//!    $$V_v(t_{\text{curr}}) = V_{\text{rest}} + (V_v(t_{\text{prev}}) - V_{\text{rest}}) \exp\left(-\frac{t_{\text{curr}} - t_{\text{prev}}}{\tau_m}\right)$$
//!    On spike arrival with synaptic weight $W_{uv}$:
//!    $$V_v(t_{\text{curr}}^+) = V_v(t_{\text{curr}}) + W_{uv}$$
//!    If $V_v(t_{\text{curr}}^+) \ge V_{\text{thresh}}$ and $t_{\text{curr}} \ge t_{\text{ref\_end}}$,
//!    the neuron fires an action potential at $t_{\text{curr}}$, resets to $V_{\text{reset}}$, and sets
//!    $t_{\text{ref\_end}} = t_{\text{curr}} + \tau_{\text{ref}}$.
//!
//! 3. **Spike-Timing-Dependent Delay Plasticity (STDDP)**:
//!    Axonal and dendritic conduction delays $\tau_{uv} \in [\tau_{\min}, \tau_{\max}]$ adapt dynamically:
//!    $$\Delta t = t_{\text{post}} - (t_{\text{pre}} + \tau_{uv})$$
//!    $$\Delta \tau_{uv} = \eta_\tau \cdot \operatorname{sign}(\Delta t) \exp\left(-\frac{|\Delta t|}{\tau_\delta}\right)$$
//!    If $\Delta t > 0$, presynaptic spike arrived slightly early; delay increases to lock into resonance.
//!    If $\Delta t < 0$, presynaptic spike arrived late; delay decreases.
//!    This enables self-organized emergence of **polychronous neuronal groups** (Izhikevich, 2006).

use crate::error::{Error, Result};
use std::cmp::Ordering;
use std::collections::BinaryHeap;

/// A discrete spike event in continuous simulation time.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SpikeEvent {
    /// Time when the presynaptic neuron emitted the spike.
    pub timestamp: f64,
    /// Presynaptic neuron index.
    pub source: usize,
    /// Postsynaptic neuron index.
    pub target: usize,
    /// Synaptic efficacy weight $W_{uv}$.
    pub weight: f64,
    /// Axonal conduction delay $\tau_{uv}$.
    pub delay: f64,
}

impl SpikeEvent {
    /// Constructs a new spike event.
    pub fn new(timestamp: f64, source: usize, target: usize, weight: f64, delay: f64) -> Self {
        Self {
            timestamp,
            source,
            target,
            weight,
            delay: delay.max(0.0),
        }
    }

    /// Continuous physical time when the action potential reaches the postsynaptic soma:
    ///
    /// $$t_{\text{arrival}} = t_{\text{timestamp}} + \tau_{\text{delay}}$$
    #[inline]
    pub fn arrival_time(&self) -> f64 {
        self.timestamp + self.delay
    }
}

impl Eq for SpikeEvent {}

impl PartialOrd for SpikeEvent {
    #[inline]
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for SpikeEvent {
    #[inline]
    fn cmp(&self, other: &Self) -> Ordering {
        // Min-heap ordering: smaller arrival time has higher priority
        other
            .arrival_time()
            .partial_cmp(&self.arrival_time())
            .unwrap_or(Ordering::Equal)
    }
}

/// Continuous-time asynchronous priority queue event router.
#[derive(Debug, Clone, Default)]
pub struct EventDrivenSpikeRouter {
    queue: BinaryHeap<SpikeEvent>,
    /// Number of routed spike events over the simulation history.
    pub events_routed: usize,
}

impl EventDrivenSpikeRouter {
    /// Creates a new empty spike router.
    pub fn new() -> Self {
        Self {
            queue: BinaryHeap::new(),
            events_routed: 0,
        }
    }

    /// Schedules an incoming spike event.
    pub fn schedule(&mut self, event: SpikeEvent) {
        self.queue.push(event);
    }

    /// Returns the number of pending events waiting in the queue.
    pub fn pending_count(&self) -> usize {
        self.queue.len()
    }

    /// Peeks at the earliest arrival time in the queue, if any.
    pub fn next_arrival_time(&self) -> Option<f64> {
        self.queue.peek().map(|e| e.arrival_time())
    }

    /// Pops the earliest event from the queue.
    pub fn pop_next(&mut self) -> Option<SpikeEvent> {
        let ev = self.queue.pop();
        if ev.is_some() {
            self.events_routed += 1;
        }
        ev
    }

    /// Clears all scheduled events from the queue.
    pub fn clear(&mut self) {
        self.queue.clear();
    }
}

/// Polychronous neural layer with learnable synaptic weights and axonal conduction delays.
#[derive(Debug, Clone)]
pub struct PolychronousNeuralLayer {
    /// Number of input neurons.
    pub num_inputs: usize,
    /// Number of output neurons.
    pub num_outputs: usize,
    /// Synaptic weights: shape `[num_inputs * num_outputs]`.
    pub weights: Vec<f64>,
    /// Axonal conduction delays: shape `[num_inputs * num_outputs]` in seconds.
    pub delays: Vec<f64>,
    /// Maximum allowable delay $\tau_{\max}$.
    pub max_delay: f64,
    /// Resting membrane potential $V_{\text{rest}}$.
    pub v_rest: f64,
    /// Firing threshold $V_{\text{thresh}}$.
    pub v_thresh: f64,
    /// Reset potential $V_{\text{reset}}$.
    pub v_reset: f64,
    /// Membrane time constant $\tau_m$ in seconds.
    pub tau_m: f64,
    /// Absolute refractory period $\tau_{\text{ref}}$ in seconds.
    pub tau_ref: f64,
}

impl PolychronousNeuralLayer {
    /// Creates a new polychronous layer with randomized or uniform weights and delays.
    pub fn new(num_inputs: usize, num_outputs: usize, max_delay: f64) -> Result<Self> {
        if num_inputs == 0 || num_outputs == 0 {
            return Err(Error::Msg("Layer dimensions must be > 0".to_string()));
        }
        // NaN passes a bare `<= 0` check, then panics `clamp` below.
        if max_delay.is_nan() || max_delay <= 0.0 {
            return Err(Error::Msg("max_delay must be > 0".to_string()));
        }

        let total = crate::layout::checked_numel(&[num_inputs, num_outputs])?;
        // Default initialized with positive excitory weights
        let weights = crate::layout::try_filled(total, 1.0 / (num_inputs as f64).sqrt())?;
        // Uniformly distributed delays between 0 and max_delay
        let mut delays = crate::layout::try_with_capacity(total)?;
        delays.extend((0..total).map(|i| (i as f64 / total as f64) * max_delay));

        Ok(Self {
            num_inputs,
            num_outputs,
            weights,
            delays,
            max_delay,
            v_rest: 0.0,
            v_thresh: 1.0,
            v_reset: 0.0,
            tau_m: 0.020,   // 20 ms
            tau_ref: 0.002, // 2 ms
        })
    }

    /// Sets explicit synaptic weights of length `num_inputs * num_outputs`.
    pub fn set_weights(&mut self, weights: Vec<f64>) -> Result<()> {
        // Dims are pub: the product overflows for absurd tampered dims.
        let expected = self
            .num_inputs
            .checked_mul(self.num_outputs)
            .ok_or_else(|| Error::Msg("Layer size overflows".to_string()))?;
        if weights.len() != expected {
            return Err(Error::Msg(format!(
                "Weight size {} does not match expected {}",
                weights.len(),
                expected
            )));
        }
        self.weights = weights;
        Ok(())
    }

    /// Sets explicit axonal delays of length `num_inputs * num_outputs`.
    pub fn set_delays(&mut self, delays: Vec<f64>) -> Result<()> {
        let expected = self
            .num_inputs
            .checked_mul(self.num_outputs)
            .ok_or_else(|| Error::Msg("Layer size overflows".to_string()))?;
        if delays.len() != expected {
            return Err(Error::Msg(format!(
                "Delays size {} does not match expected {}",
                delays.len(),
                expected
            )));
        }
        // `max_delay` is pub: NaN (or negative) panics `clamp`.
        if self.max_delay.is_nan() || self.max_delay <= 0.0 {
            return Err(Error::Msg("max_delay must be > 0".to_string()));
        }
        self.delays = delays
            .into_iter()
            .map(|d| d.clamp(0.0, self.max_delay))
            .collect();
        Ok(())
    }

    /// Performs exact event-driven forward propagation for a given train of input spikes:
    /// `input_spikes`: slice of `(t_spike, input_neuron_idx)`.
    /// Returns output spikes: `Vec<(t_spike, output_neuron_idx)>`.
    pub fn forward_event_driven(
        &self,
        input_spikes: &[(f64, usize)],
        t_max: f64,
    ) -> Result<Vec<(f64, usize)>> {
        // Tables and dims are pub: validate the consistency the indexing
        // below assumes (a short table indexed OOB; absurd dims hang).
        let need = self
            .num_inputs
            .checked_mul(self.num_outputs)
            .ok_or_else(|| Error::Msg("Layer size overflows".to_string()))?;
        if self.weights.len() != need || self.delays.len() != need {
            return Err(Error::Msg(format!(
                "Layer tables have {}/{} entries, need {}",
                self.weights.len(),
                self.delays.len(),
                need
            )));
        }
        let mut router = EventDrivenSpikeRouter::new();

        // 1. Schedule all input spikes into the event queue with their individual connection delays
        for &(t_spike, in_idx) in input_spikes {
            if in_idx >= self.num_inputs {
                return Err(Error::Msg(format!(
                    "Input neuron index {in_idx} out of range (num_inputs = {})",
                    self.num_inputs
                )));
            }
            if t_spike < 0.0 || t_spike > t_max {
                continue;
            }

            for out_idx in 0..self.num_outputs {
                let conn_id = in_idx * self.num_outputs + out_idx;
                let w = self.weights[conn_id];
                let d = self.delays[conn_id];
                router.schedule(SpikeEvent::new(t_spike, in_idx, out_idx, w, d));
            }
        }

        // Postsynaptic state tracking
        let mut v_mem = vec![self.v_rest; self.num_outputs];
        let mut last_event_time = vec![0.0; self.num_outputs];
        let mut ref_until = vec![0.0; self.num_outputs];
        let mut output_spikes = Vec::new();

        // 2. Process events in causal arrival order
        while let Some(event) = router.pop_next() {
            let t_arr = event.arrival_time();
            if t_arr > t_max {
                break;
            }

            let out_idx = event.target;
            let dt = t_arr - last_event_time[out_idx];
            if dt > 0.0 {
                // Continuous leaky decay
                let decay = (-dt / self.tau_m).exp();
                v_mem[out_idx] = self.v_rest + (v_mem[out_idx] - self.v_rest) * decay;
                last_event_time[out_idx] = t_arr;
            }

            // Only integrate if not in absolute refractory period
            if t_arr >= ref_until[out_idx] {
                v_mem[out_idx] += event.weight;

                // Firing check
                if v_mem[out_idx] >= self.v_thresh {
                    output_spikes.push((t_arr, out_idx));
                    v_mem[out_idx] = self.v_reset;
                    ref_until[out_idx] = t_arr + self.tau_ref;
                }
            }
        }

        Ok(output_spikes)
    }

    /// Adapts conduction delays and synaptic weights using Spike-Timing-Dependent Delay Plasticity (STDDP).
    pub fn adapt_stddp(
        &mut self,
        input_spikes: &[(f64, usize)],
        output_spikes: &[(f64, usize)],
        lr_delay: f64,
        lr_weight: f64,
        tau_delta: f64,
    ) -> Result<()> {
        if tau_delta <= 0.0 {
            return Ok(());
        }
        // Spike indices come straight from the caller: an unvalidated index
        // reads/writes out of bounds (or overflows the connection product).
        for &(_, pre_idx) in input_spikes {
            if pre_idx >= self.num_inputs {
                return Err(Error::Msg(format!(
                    "STDDP pre index {pre_idx} out of range (num_inputs = {})",
                    self.num_inputs
                )));
            }
        }
        for &(_, post_idx) in output_spikes {
            if post_idx >= self.num_outputs {
                return Err(Error::Msg(format!(
                    "STDDP post index {post_idx} out of range (num_outputs = {})",
                    self.num_outputs
                )));
            }
        }
        if self.max_delay.is_nan() || self.max_delay <= 0.0 {
            return Err(Error::Msg("max_delay must be > 0".to_string()));
        }
        // Tables are pub too: bound the connection id against their length.
        let need = self
            .num_inputs
            .checked_mul(self.num_outputs)
            .ok_or_else(|| Error::Msg("Layer size overflows".to_string()))?;
        if self.weights.len() != need || self.delays.len() != need {
            return Err(Error::Msg(format!(
                "Layer tables have {}/{} entries, need {}",
                self.weights.len(),
                self.delays.len(),
                need
            )));
        }

        for &(t_post, post_idx) in output_spikes {
            for &(t_pre, pre_idx) in input_spikes {
                let conn_id = pre_idx * self.num_outputs + post_idx;
                let current_delay = self.delays[conn_id];
                let t_arrival = t_pre + current_delay;
                let delta_t = t_post - t_arrival;

                if delta_t.abs() < 5.0 * tau_delta {
                    // STDDP delay update: shift delay so presynaptic arrival aligns with postsynaptic firing
                    let delay_grad = (delta_t.signum()) * (-delta_t.abs() / tau_delta).exp();
                    self.delays[conn_id] =
                        (current_delay + lr_delay * delay_grad).clamp(0.0, self.max_delay);

                    // Standard STDP weight update: potentiate if pre preceded post
                    let weight_grad = if delta_t > 0.0 {
                        (-delta_t / tau_delta).exp()
                    } else {
                        -0.5 * (delta_t.abs() / tau_delta).exp()
                    };
                    self.weights[conn_id] =
                        (self.weights[conn_id] + lr_weight * weight_grad).max(0.0);
                }
            }
        }
        Ok(())
    }
}
