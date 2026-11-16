//! Wilson-Cowan Excitatory-Inhibitory Cortical Neural Population Dynamics & Gamma Oscillations (Wilson & Cowan, 1972).
//!
//! ### 1. Coupled Excitatory-Inhibitory Population Firing Rate ODEs
//! Let $E(t), I(t) \in [0, 1]$ denote the fraction of active excitatory and inhibitory neurons in a cortical column:
//! $$\tau_E \frac{dE}{dt} = -E + \sigma(w_{EE} E - w_{EI} I + P_E), \qquad \tau_I \frac{dI}{dt} = -I + \sigma(w_{IE} E - w_{II} I + P_I)$$
//! where $\sigma(x) = (1 + e^{-x})^{-1}$ is the logistic population response function.

/// Wilson-Cowan Cortical Excitatory-Inhibitory Population Engine.
#[derive(Debug, Clone, Copy)]
pub struct WilsonCowanCorticalEngine {
    /// Synaptic coupling matrix $[w_{EE}, w_{EI}, w_{IE}, w_{II}]$.
    pub weights: [f64; 4],
}

impl WilsonCowanCorticalEngine {
    /// Creates a new Wilson-Cowan cortical microcircuit engine.
    pub fn new(weights: [f64; 4]) -> Self {
        Self { weights }
    }

    fn sigmoid(x: f64) -> f64 {
        1.0 / (1.0 + (-x).exp())
    }

    /// Computes one Euler step $(E_{t+dt}, I_{t+dt})$ of the Wilson-Cowan cortical column with $\tau_E = \tau_I = 1$.
    pub fn step_ei(&self, e: f64, i: f64, p_e: f64, p_i: f64, dt: f64) -> (f64, f64) {
        let [wee, wei, wie, wii] = self.weights;
        let de = -e + Self::sigmoid(wee * e - wei * i + p_e);
        let di = -i + Self::sigmoid(wie * e - wii * i + p_i);
        (e + dt * de, i + dt * di)
    }
}
