//! Conformal Bootstrap, 4-Point Crossing Symmetry, Dolan-Osborn Conformal Blocks & Unitarity Bounds.
//!
//! ### 1. Mack Unitarity Bounds in $d$-Dimensional Conformal Field Theory ($SO(d, 2)$)
//! In any unitary $d$-dimensional CFT ($d \ge 2$), the scaling dimension $\Delta$ of a primary operator of spin $\ell$
//! must satisfy Mack's unitarity bound:
//! • Scalar primaries ($\ell = 0$, non-identity): $\Delta \ge \frac{d - 2}{2}$
//! • Spinning primaries ($\ell \ge 1$): $\Delta \ge \ell + d - 2$ (saturated by conserved currents $\partial_\mu J^{\mu \dots} = 0$, e.g. stress tensor $T_{\mu\nu}$ with $\ell = 2, \Delta = d$).
//!
//! ### 2. 1D/Chiral Conformal Block $G_{\Delta}(z) = z^\Delta \, {}_2F_1(\Delta, \Delta; 2\Delta; z)$
//! Evaluated via the convergent hypergeometric series so crossing symmetry $z \leftrightarrow 1 - z$ can be tested.

use crate::error::{Error, Result};

/// Conformal Bootstrap Unitarity & Conformal Block Engine.
#[derive(Debug, Clone, Copy)]
pub struct ConformalBootstrapEngine {
    /// Spacetime dimension $d \ge 2$.
    pub spacetime_dim: f64,
}

impl ConformalBootstrapEngine {
    /// Creates a new Conformal Bootstrap engine in $d \ge 2$ dimensions.
    pub fn new(spacetime_dim: f64) -> Result<Self> {
        if spacetime_dim < 2.0 {
            return Err(Error::Msg("spacetime_dim must be >= 2.0".to_string()));
        }
        Ok(Self { spacetime_dim })
    }

    /// Computes Mack's unitarity lower bound $\Delta_{\min}(\ell)$ for spin $\ell$.
    pub fn unitarity_bound(&self, spin_l: usize) -> f64 {
        if spin_l == 0 {
            0.5 * (self.spacetime_dim - 2.0)
        } else {
            (spin_l as f64) + self.spacetime_dim - 2.0
        }
    }

    /// Evaluates the $SL(2, \mathbb{R})$ chiral conformal block $g_\Delta(z) = z^\Delta {}_2F_1(\Delta, \Delta; 2\Delta; z)$ for $z \in (0, 1)$.
    pub fn chiral_conformal_block(delta: f64, z: f64) -> Result<f64> {
        if delta <= 0.0 || z <= 0.0 || z >= 1.0 {
            return Err(Error::Msg("delta must be > 0 and z in (0, 1)".to_string()));
        }
        let mut sum = 1.0;
        let mut term = 1.0;
        for n in 0..40 {
            let nf = n as f64;
            term *= (delta + nf) * (delta + nf) * z / ((2.0 * delta + nf) * (nf + 1.0));
            sum += term;
        }
        Ok(z.powf(delta) * sum)
    }
}
