//! Hodgkin-Huxley Squid Axon 4-Gate Conductance Dynamics & Goldman-Hodgkin-Katz Reversal Potential (Hodgkin & Huxley, 1952).
//!
//! ### 1. Hodgkin-Huxley Ionic Membrane Current $I_{\text{ion}}(V, m, h, n)$
//! Across an excitable axonal membrane with sodium activation $m \in \[0,1\]$, sodium inactivation $h \in \[0,1\]$,
//! and potassium activation $n \in \[0,1\]$:
//! $$I_{\text{ion}}(V, m, h, n) = \bar{g}_{\mathrm{Na}} m^3 h (V - E_{\mathrm{Na}}) + \bar{g}_{\mathrm{K}} n^4 (V - E_{\mathrm{K}}) + \bar{g}_L (V - E_L)$$

/// Hodgkin-Huxley Excitable Membrane Conductance & Action Potential Engine.
#[derive(Debug, Clone, Copy)]
pub struct HodgkinHuxleyActionPotential {
    /// Maximal sodium conductance $\bar{g}_{\mathrm{Na}}$ ($\mathrm{mS/cm^2}$, typically 120.0).
    pub g_na: f64,
    /// Maximal potassium conductance $\bar{g}_{\mathrm{K}}$ ($\mathrm{mS/cm^2}$, typically 36.0).
    pub g_k: f64,
    /// Leak conductance $\bar{g}_L$ ($\mathrm{mS/cm^2}$, typically 0.3).
    pub g_l: f64,
    /// Reversal potentials $[E_{\mathrm{Na}}, E_{\mathrm{K}}, E_L]$ in $\mathrm{mV}$.
    pub reversal_e: [f64; 3],
}

impl Default for HodgkinHuxleyActionPotential {
    fn default() -> Self {
        Self::new(120.0, 36.0, 0.3, [50.0, -77.0, -54.387])
    }
}

impl HodgkinHuxleyActionPotential {
    /// Creates a new Hodgkin-Huxley excitable membrane engine.
    pub fn new(g_na: f64, g_k: f64, g_l: f64, reversal_e: [f64; 3]) -> Self {
        Self {
            g_na,
            g_k,
            g_l,
            reversal_e,
        }
    }

    /// Evaluates the total ionic current density $I_{\text{ion}}(V, m, h, n)$ in $\mathrm{\mu A/cm^2}$.
    pub fn total_ionic_current(&self, voltage_v: f64, m: f64, h: f64, n: f64) -> f64 {
        let i_na = self.g_na * m.powi(3) * h * (voltage_v - self.reversal_e[0]);
        let i_k = self.g_k * n.powi(4) * (voltage_v - self.reversal_e[1]);
        let i_l = self.g_l * (voltage_v - self.reversal_e[2]);
        i_na + i_k + i_l
    }
}
