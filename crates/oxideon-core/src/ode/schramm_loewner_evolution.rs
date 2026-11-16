//! Schramm-Loewner Evolution ($\mathrm{SLE}_\kappa$), Beffara Hausdorff Dimension & Rohde-Schramm Duality (Schramm, 2000; Beffara, 2008).
//!
//! ### 1. Chordal Loewner Differential Equation & Critical Conformal Curves
//! Driven by $\xi_t = \sqrt{\kappa} B_t$ on the upper half-plane $\mathbb{H}$, chordal $\mathrm{SLE}_\kappa$ evolves conformal maps via:
//! $$\partial_t g_t(z) = \frac{2}{g_t(z) - \sqrt{\kappa} B_t}, \qquad g_0(z) = z$$
//! • **Beffara's Theorem (2008)**: The Hausdorff dimension of the $\mathrm{SLE}_\kappa$ trace is $d_H(\kappa) = \min(1 + \kappa / 8, 2)$.
//! • **Phase Transitions**: Simple curves for $\kappa \in (0, 4]$, self-touching for $\kappa \in (4, 8)$, space-filling for $\kappa \ge 8$.

use crate::error::{Error, Result};

/// Chordal Schramm-Loewner Evolution ($\mathrm{SLE}_\kappa$) & Conformal Fractal Engine.
#[derive(Debug, Clone, Copy)]
pub struct SchrammLoewnerEngine {
    /// Diffusivity parameter $\kappa > 0$ (e.g., $\kappa = 2$ LERW, $\kappa = 8/3$ SAW, $\kappa = 3$ Ising, $\kappa = 6$ Percolation).
    pub kappa: f64,
}

impl SchrammLoewnerEngine {
    /// Creates a new $\mathrm{SLE}_\kappa$ engine.
    pub fn new(kappa: f64) -> Result<Self> {
        if kappa <= 0.0 {
            return Err(Error::Msg("SLE parameter kappa must be > 0".to_string()));
        }
        Ok(Self { kappa })
    }

    /// Computes Beffara's exact Hausdorff dimension $d_H = \min(1 + \kappa / 8, 2)$ and Duplantier's dual $\tilde{\kappa} = 16 / \kappa$.
    pub fn hausdorff_dimension_and_dual(&self) -> (f64, f64) {
        ((1.0 + self.kappa / 8.0).min(2.0), 16.0 / self.kappa)
    }

    /// Computes the central charge $c(\kappa) = \frac{(3 \kappa - 8)(6 - \kappa)}{2 \kappa}$ of the associated CFT.
    pub fn cft_central_charge(&self) -> f64 {
        (3.0 * self.kappa - 8.0) * (6.0 - self.kappa) / (2.0 * self.kappa)
    }
}
