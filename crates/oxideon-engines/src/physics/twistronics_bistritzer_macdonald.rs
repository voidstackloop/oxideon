//! Bistritzer-MacDonald Continuum Model of Twisted Bilayer Graphene (TBG) & Flat-Band Magic Angle $\theta_m \approx 1.05^\circ$ (Bistritzer & MacDonald, 2011).
//!
//! ### 1. Renormalized Dirac Velocity $v^*(\alpha)$ & First Magic Angle $\alpha_1 = 1 / \sqrt{3}$
//! In the Bistritzer-MacDonald moiré continuum model with interlayer tunneling $w$ and moiré momentum $k_\theta = 2 k_D \sin(\theta / 2)$,
//! the dimensionless coupling $\alpha = \frac{w}{\hbar v_F k_\theta}$ renormalizes the Dirac Fermi velocity at leading order via:
//! $$\frac{v^*(\alpha)}{v_F} = \frac{1 - 3 \alpha^2}{1 + 6 \alpha^2}$$
//! which vanishes identically ($v^* = 0$, flat band!) at the **First Magic Angle** $\alpha_1 = \frac{1}{\sqrt{3}} \approx 0.57735$ ($\theta_{\text{magic}} \approx 1.05^\circ$).

/// Bistritzer-MacDonald Twisted Bilayer Graphene (TBG) Magic Angle Engine.
#[derive(Debug, Clone, Copy, Default)]
pub struct MagicAngleGrapheneEngine;

impl MagicAngleGrapheneEngine {
    /// Creates a new Magic-Angle Twisted Bilayer Graphene engine.
    pub fn new() -> Self {
        Self
    }

    /// Evaluates the Bistritzer-MacDonald renormalized Fermi velocity ratio $v^*(\alpha) / v_F = \frac{1 - 3 \alpha^2}{1 + 6 \alpha^2}$.
    pub fn renormalized_velocity_ratio(alpha: f64) -> f64 {
        let a2 = alpha * alpha;
        (1.0 - 3.0 * a2) / (1.0 + 6.0 * a2)
    }

    /// Returns the exact leading-order dimensionless magic parameter $\alpha_1 = 1 / \sqrt{3}$ where $v^*(\alpha_1) = 0$.
    pub fn first_magic_alpha() -> f64 {
        1.0 / 3.0_f64.sqrt()
    }
}
