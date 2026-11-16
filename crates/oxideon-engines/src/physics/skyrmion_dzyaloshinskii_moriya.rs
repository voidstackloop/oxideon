//! Chiral Magnetic Skyrmions, Dzyaloshinskii-Moriya Interaction (DMI), Topological Charge $Q$ & Thiele Hall Angle (Skyrme, 1962; Thiele, 1973).
//!
//! ### 1. Topological Skyrmion Number $Q \in \mathbb{Z}$ & Thiele Skyrmion Hall Angle
//! Under spin-transfer torque velocity $\mathbf{v}_s$, Thiele's collective coordinate equation
//! $\mathbf{G} \times (\mathbf{v}_d - \mathbf{v}_s) + \mathcal{D}(\alpha \mathbf{v}_d - \beta \mathbf{v}_s) = 0$ with gyrocoupling
//! $G = 4 \pi Q$ and dissipative tensor $D$ yields the transverse **Skyrmion Hall Angle**:
//! $$\tan \theta_{\mathrm{SkHE}} = \frac{v_{d,\perp}}{v_{d,\parallel}} = \frac{G (\alpha - \beta)}{G^2 + \alpha \beta D^2}$$

use std::f64::consts::PI;

/// Chiral Magnetic Skyrmion Topology & Thiele Skyrmion Hall Effect Engine.
#[derive(Debug, Clone, Copy)]
pub struct MagneticSkyrmionEngine {
    /// Topological Pontryagin skyrmion charge $Q \in \mathbb{Z}$ (typically $Q = \pm 1$).
    pub topological_q: i32,
    /// Shape dissipation factor $D > 0$.
    pub dissipation_d: f64,
}

impl MagneticSkyrmionEngine {
    /// Creates a new Magnetic Skyrmion engine.
    pub fn new(topological_q: i32, dissipation_d: f64) -> Self {
        Self {
            topological_q,
            dissipation_d,
        }
    }

    /// Computes the Thiele Skyrmion Hall tangent $\tan \theta_{\mathrm{SkHE}} = \frac{G(\alpha - \beta)}{G^2 + \alpha \beta D^2}$ ($G = 4 \pi Q$).
    pub fn thiele_skyrmion_hall_tangent(&self, gilbert_alpha: f64, nonadiabatic_beta: f64) -> f64 {
        let g = 4.0 * PI * (self.topological_q as f64);
        let denom =
            g * g + gilbert_alpha * nonadiabatic_beta * self.dissipation_d * self.dissipation_d;
        if denom.abs() < 1e-15 {
            0.0
        } else {
            g * (gilbert_alpha - nonadiabatic_beta) / denom
        }
    }
}
