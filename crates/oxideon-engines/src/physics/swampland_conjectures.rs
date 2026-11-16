//! Quantum Gravity Swampland Program: Weak Gravity Conjecture (WGC), Ooguri-Vafa Distance Conjecture & de Sitter Bounds.
//!
//! ### 1. Arkani-Hamed–Motl–Nicolis–Vafa Weak Gravity Conjecture (2006)
//! In any consistent quantum gravity theory coupled to a $U(1)$ gauge field of coupling $g$, there must exist
//! a super-extremal charged state of mass $m$ and integer charge $q$ satisfying (in Planck units $M_P = 1$):
//! $$\frac{g \, q}{m} \ge \frac{1}{\sqrt{2}} \quad \Longleftrightarrow \quad m \le \sqrt{2} \, g \, q \, M_P$$
//! so that extremal black holes can decay without leaving behind an infinite tower of stable remnants.
//!
//! ### 2. Ooguri-Vafa Swampland Distance Conjecture
//! Travelling geodesic distance $\Delta \phi \gg 1$ in moduli space brings down an infinite tower of states
//! with exponentially light mass scale $M(\phi) = M_0 e^{-\alpha \Delta \phi}$.

use std::f64::consts::FRAC_1_SQRT_2;

/// Quantum Gravity Swampland WGC & Distance Conjecture Verifier.
#[derive(Debug, Clone, Copy, Default)]
pub struct SwamplandVerifierEngine;

impl SwamplandVerifierEngine {
    /// Creates a new Swampland verifier engine.
    pub fn new() -> Self {
        Self
    }

    /// Checks whether a particle of `(mass, charge, gauge_coupling_g)` satisfies the Weak Gravity Conjecture
    /// $g q / m \ge 1 / \sqrt{2}$ in Planck units.
    pub fn satisfies_weak_gravity_conjecture(mass: f64, charge_q: f64, gauge_g: f64) -> bool {
        if mass <= 0.0 || charge_q <= 0.0 || gauge_g <= 0.0 {
            return false;
        }
        (gauge_g * charge_q / mass) >= FRAC_1_SQRT_2 - 1e-12
    }

    /// Computes the Ooguri-Vafa Kaluza-Klein / winding tower mass scale $M(\Delta \phi) = M_0 e^{-\alpha \Delta \phi}$.
    pub fn distance_conjecture_tower_mass(m0: f64, alpha: f64, geodesic_dist: f64) -> f64 {
        m0 * (-alpha * geodesic_dist).exp()
    }
}
