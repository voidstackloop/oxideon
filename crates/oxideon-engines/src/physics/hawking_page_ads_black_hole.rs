//! Hawking-Page $\mathrm{AdS}$ Black Hole Phase Transition & Witten Confinement-Deconfinement Duality (Hawking & Page, 1983; Witten, 1998).
//!
//! ### 1. $\mathrm{AdS}_4$ Schwarzschild Temperature & Free Energy Sign Change at $r_+ = L$
//! A Schwarzschild black hole of horizon radius $r_+ > 0$ in Anti-de Sitter space of curvature radius $L > 0$ has
//! Hawking temperature and Euclidean on-shell free energy ($G = 1$):
//! $$T(r_+) = \frac{L^2 + 3 r_+^2}{4 \pi L^2 r_+}, \qquad F(r_+) = \frac{r_+ (L^2 - r_+^2)}{4 L^2}$$
//! Notice that $F(r_+) < 0$ iff $r_+ > L$ ($T > T_{\mathrm{HP}} = \frac{1}{\pi L}$): above the **Hawking-Page Temperature** $T_{\mathrm{HP}}$,
//! thermal $\mathrm{AdS}$ tunnels into a large $\mathrm{AdS}$ black hole (dual to the $\mathcal{N}=4$ SYM deconfined quark-gluon plasma)!

use crate::error::{Error, Result};
use std::f64::consts::PI;

/// Hawking-Page $\mathrm{AdS}$ Black Hole Phase Transition Engine.
#[derive(Debug, Clone, Copy)]
pub struct HawkingPagePhaseTransition {
    /// $\mathrm{AdS}$ curvature radius $L > 0$.
    pub ads_radius_l: f64,
}

impl HawkingPagePhaseTransition {
    /// Creates a new Hawking-Page phase transition engine.
    pub fn new(ads_radius_l: f64) -> Result<Self> {
        if ads_radius_l <= 0.0 {
            return Err(Error::Msg("ads_radius_l must be > 0".to_string()));
        }
        Ok(Self { ads_radius_l })
    }

    /// Computes the Hawking temperature $T(r_+)$, Euclidean free energy $F(r_+)$, and whether the large black hole phase dominates ($F < 0$, i.e., $r_+ > L$).
    pub fn temperature_free_energy_and_deconfined(&self, r_plus: f64) -> Result<(f64, f64, bool)> {
        if r_plus <= 0.0 {
            return Err(Error::Msg("r_plus must be > 0".to_string()));
        }
        let l2 = self.ads_radius_l * self.ads_radius_l;
        let r2 = r_plus * r_plus;
        let temp = (l2 + 3.0 * r2) / (4.0 * PI * l2 * r_plus);
        let free_e = r_plus * (l2 - r2) / (4.0 * l2);
        Ok((temp, free_e, free_e < 0.0))
    }
}
