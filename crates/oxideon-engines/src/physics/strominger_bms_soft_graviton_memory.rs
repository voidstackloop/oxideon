//! Strominger Infrared Triangle: BMS Supertranslations, Weinberg Soft Graviton Theorem & Christodoulou Gravitational Wave Memory (Weinberg, 1965; Strominger, 2014/2017).
//!
//! ### 1. Leading Soft Factor $S^{(0)} \propto \omega^{-1}$ & Permanent Strain Memory $\Delta h_+(\mathcal{I}^+)$
//! As soft radiative frequency $\omega \to 0$, the tree-level Weinberg soft factor scales universally as $S^{(0)}(\omega) = C_{\mathrm{soft}} / \omega$, whose Fourier transform produces a permanent DC step $\Delta h_{\mathrm{mem}} = C_{\mathrm{soft}} / r$ in the asymptotic metric at future null infinity $\mathcal{I}^+$, canonically conjugate to BMS supertranslations.

use crate::error::{Error, Result};

/// Strominger BMS-Soft-Memory Infrared Triangle Engine.
#[derive(Debug, Clone, Copy)]
pub struct StromingerSoftTheoremMemoryTriangle {
    /// Kinematic Weinberg soft factor numerator $\mathcal{K}_{\mathrm{soft}} > 0$.
    pub soft_numerator_k: f64,
}

impl Default for StromingerSoftTheoremMemoryTriangle {
    fn default() -> Self {
        Self {
            soft_numerator_k: 4.0,
        }
    }
}

impl StromingerSoftTheoremMemoryTriangle {
    /// Creates a new Strominger Infrared Triangle engine.
    pub fn new(soft_numerator_k: f64) -> Result<Self> {
        if soft_numerator_k <= 0.0 || !soft_numerator_k.is_finite() {
            return Err(Error::Msg("soft_numerator_k must be > 0".to_string()));
        }
        Ok(Self { soft_numerator_k })
    }

    /// Evaluates the Weinberg soft amplitude pole $\mathcal{K}_{\mathrm{soft}} / \omega$ and permanent gravitational-wave memory strain $\Delta h = \mathcal{K}_{\mathrm{soft}} / r$.
    pub fn soft_pole_and_memory_strain(
        &self,
        soft_freq_omega: f64,
        distance_r: f64,
    ) -> Result<(f64, f64)> {
        if soft_freq_omega <= 0.0 || distance_r <= 0.0 {
            return Err(Error::Msg(
                "soft_freq_omega and distance_r must be > 0".to_string(),
            ));
        }
        Ok((
            self.soft_numerator_k / soft_freq_omega,
            self.soft_numerator_k / distance_r,
        ))
    }
}
