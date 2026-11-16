//! Møller-Plesset Second-Order Perturbation Theory (MP2) & Grimme Spin-Component-Scaled SCS-MP2 (Møller & Plesset, 1934; Grimme, 2003).
//!
//! ### 1. Opposite-Spin & Same-Spin MP2 Pair Correlation Energy & Grimme SCS-MP2
//! Separating the second-order Møller-Plesset correlation energy into opposite-spin ($E_{\mathrm{OS}}^{(2)}$) and same-spin ($E_{\mathrm{SS}}^{(2)}$)
//! pair contributions, Grimme's **Spin-Component-Scaled MP2 (SCS-MP2)** achieves near-CCSD accuracy via:
//! $$E_{\mathrm{SCS-MP2}}^{(2)} = \frac{6}{5} E_{\mathrm{OS}}^{(2)} + \frac{1}{3} E_{\mathrm{SS}}^{(2)}$$

/// Møller-Plesset MP2 & Grimme Spin-Component-Scaled (SCS-MP2) Engine.
#[derive(Debug, Clone, Copy, Default)]
pub struct MollerPlessetMp2Engine;

impl MollerPlessetMp2Engine {
    /// Creates a new Møller-Plesset MP2 engine.
    pub fn new() -> Self {
        Self
    }

    /// Evaluates the canonical MP2 correlation energy $E_{\mathrm{OS}} + E_{\mathrm{SS}}$ and Grimme's SCS-MP2 energy $\frac{6}{5} E_{\mathrm{OS}} + \frac{1}{3} E_{\mathrm{SS}}$.
    pub fn canonical_and_scs_mp2_energy(e_opposite_spin: f64, e_same_spin: f64) -> (f64, f64) {
        let e_mp2 = e_opposite_spin + e_same_spin;
        let e_scs = 1.2 * e_opposite_spin + (1.0 / 3.0) * e_same_spin;
        (e_mp2, e_scs)
    }
}
