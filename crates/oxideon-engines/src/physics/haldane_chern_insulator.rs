//! Haldane Quantum Anomalous Hall Effect on a Honeycomb Lattice & Quantized Chern Number $C \in \{-1, 0, +1\}$ (Haldane, 1988).
//!
//! ### 1. Dirac Mass at Valley Points $\mathbf{K}, \mathbf{K}'$ & Topological Phase Diagram
//! With Semenoff sublattice mass $M$ and complex next-nearest-neighbor hopping $t_2 e^{i \phi}$, the effective Dirac masses
//! at the two Brillouin zone valleys $\mathbf{K}_\pm$ are:
//! $$m_\pm = M \mp 3 \sqrt{3} \, t_2 \sin \phi$$
//! yielding the TKNN / Haldane first Chern number:
//! $$C = \frac{1}{2}\left(\operatorname{sgn}(m_-) - \operatorname{sgn}(m_+)\right) \in \{-1, 0, +1\}$$

/// Haldane Honeycomb Chern Insulator & Quantum Anomalous Hall Engine.
#[derive(Debug, Clone, Copy)]
pub struct HaldaneHoneycombModel {
    /// Sublattice Semenoff symmetry-breaking mass $M$.
    pub semenoff_mass_m: f64,
    /// Next-nearest-neighbor hopping amplitude $t_2$.
    pub nnn_hopping_t2: f64,
    /// Haldane magnetic flux phase $\phi \in [-\pi, \pi]$.
    pub flux_phi: f64,
}

impl HaldaneHoneycombModel {
    /// Creates a new Haldane Honeycomb Chern Insulator model.
    pub fn new(semenoff_mass_m: f64, nnn_hopping_t2: f64, flux_phi: f64) -> Self {
        Self {
            semenoff_mass_m,
            nnn_hopping_t2,
            flux_phi,
        }
    }

    /// Computes the valley Dirac masses $(m_+, m_-)$ and exact topological Chern number $C \in \{-1, 0, +1\}$.
    pub fn valley_masses_and_chern_number(&self) -> (f64, f64, i32) {
        let haldane_gap = 3.0 * 3.0_f64.sqrt() * self.nnn_hopping_t2 * self.flux_phi.sin();
        let m_plus = self.semenoff_mass_m - haldane_gap;
        let m_minus = self.semenoff_mass_m + haldane_gap;
        let s_plus = if m_plus > 0.0 {
            1
        } else if m_plus < 0.0 {
            -1
        } else {
            0
        };
        let s_minus = if m_minus > 0.0 {
            1
        } else if m_minus < 0.0 {
            -1
        } else {
            0
        };
        (m_plus, m_minus, (s_minus - s_plus) / 2)
    }
}
