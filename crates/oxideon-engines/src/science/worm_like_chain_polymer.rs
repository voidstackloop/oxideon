//! Kratky-Porod Worm-Like Chain (WLC) Semiflexible Biopolymer & Marko-Siggia DNA Force-Extension Law (Marko & Siggia, 1995).
//!
//! ### 1. Exact Kratky-Porod End-to-End Mean-Square Distance & Marko-Siggia Entropic Elasticity
//! For a semiflexible biopolymer (dsDNA / cytoskeleton filament) of contour length $L_0 > 0$ and persistence length $L_p > 0$:
//! $$\langle R^2 \rangle = 2 L_p L_0 \left[1 - \frac{L_p}{L_0}\left(1 - e^{-L_0 / L_p}\right)\right]$$
//! Under optical-tweezers extension $z / L_0 \in [0, 1)$, the Marko-Siggia entropic stretching force is:
//! $$\frac{F(z) L_p}{k_B T} = \frac{1}{4 (1 - z / L_0)^2} - \frac{1}{4} + \frac{z}{L_0}$$

use crate::error::{Error, Result};

/// Kratky-Porod Worm-Like Chain (WLC) & Marko-Siggia DNA Elasticity Engine.
#[derive(Debug, Clone, Copy)]
pub struct KratkyPorodPolymerEngine {
    /// Persistence length $L_p > 0$ ($\approx 50\,\mathrm{nm}$ for double-stranded DNA).
    pub persistence_lp: f64,
    /// Total contour length $L_0 > 0$.
    pub contour_l0: f64,
}

impl KratkyPorodPolymerEngine {
    /// Creates a new Kratky-Porod Worm-Like Chain engine.
    pub fn new(persistence_lp: f64, contour_l0: f64) -> Result<Self> {
        if persistence_lp <= 0.0 || contour_l0 <= 0.0 {
            return Err(Error::Msg(
                "persistence_lp and contour_l0 must be > 0".to_string(),
            ));
        }
        Ok(Self {
            persistence_lp,
            contour_l0,
        })
    }

    /// Computes the exact Kratky-Porod mean-square end-to-end distance $\langle R^2 \rangle$.
    pub fn mean_square_end_to_end(&self) -> f64 {
        let ratio = self.persistence_lp / self.contour_l0;
        2.0 * self.persistence_lp * self.contour_l0 * (1.0 - ratio * (1.0 - (-1.0 / ratio).exp()))
    }

    /// Evaluates the dimensionless Marko-Siggia stretching force $F L_p / (k_B T)$ at relative extension $x = z / L_0 \in [0, 1)$.
    pub fn marko_siggia_dimensionless_force(relative_extension: f64) -> Result<f64> {
        if !(0.0..1.0).contains(&relative_extension) {
            return Err(Error::Msg(
                "relative_extension z/L0 must be in [0, 1)".to_string(),
            ));
        }
        let u = 1.0 - relative_extension;
        Ok(0.25 / (u * u) - 0.25 + relative_extension)
    }
}
