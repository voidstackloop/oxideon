//! Goeppert-Mayer & Jensen Nuclear Shell Model Spin-Orbit Splitting & Magic Numbers (Goeppert-Mayer, 1949; Haxel, Jensen & Suess, 1949).
//!
//! ### 1. Spin-Orbit Doublet Splitting $\Delta E_{ls}(\ell)$ & Intruder Shell Closure
//! Strong attractive nuclear spin-orbit coupling $-V_{ls}(r) \mathbf{\ell} \cdot \mathbf{s}$ splits levels $j = \ell \pm 1/2$ by:
//! $$\Delta E_{ls}(\ell) = E(\ell - 1/2) - E(\ell + 1/2) = C_{ls} \frac{2 \ell + 1}{2} \hbar \omega_0$$
//! pushing the high-$\ell$ intruder state $j = \ell_{\max} + 1/2$ down to generate the magic closures $2, 8, 20, 28, 50, 82, 126$.

use crate::error::{Error, Result};

/// Goeppert-Mayer & Jensen Nuclear Spin-Orbit Shell Model Engine.
#[derive(Debug, Clone, Copy)]
pub struct MayerJensenNuclearShellMagic {
    /// Spin-orbit coupling strength $C_{ls} > 0$ (MeV).
    pub spin_orbit_cls_mev: f64,
}

impl Default for MayerJensenNuclearShellMagic {
    fn default() -> Self {
        Self {
            spin_orbit_cls_mev: 1.2,
        }
    }
}

impl MayerJensenNuclearShellMagic {
    /// Creates a new Mayer-Jensen Nuclear Shell engine.
    pub fn new(spin_orbit_cls_mev: f64) -> Result<Self> {
        if spin_orbit_cls_mev <= 0.0 || !spin_orbit_cls_mev.is_finite() {
            return Err(Error::Msg("spin_orbit_cls_mev must be > 0".to_string()));
        }
        Ok(Self { spin_orbit_cls_mev })
    }

    /// Evaluates the spin-orbit doublet splitting $\Delta E_{ls} = C_{ls} (\ell + 1/2)$, intruder subshell degeneracy $2 j_+ + 1 = 2 \ell + 2$, and checks if nucleon count $N$ is one of $(2, 8, 20, 28, 50, 82, 126)$.
    pub fn doublet_splitting_and_magic_check(
        &self,
        orbital_l: usize,
        nucleon_count: usize,
    ) -> (f64, usize, bool) {
        let splitting = self.spin_orbit_cls_mev * ((orbital_l as f64) + 0.5);
        let j_plus_degeneracy = 2 * orbital_l + 2;
        let is_magic = matches!(nucleon_count, 2 | 8 | 20 | 28 | 50 | 82 | 126);
        (splitting, j_plus_degeneracy, is_magic)
    }
}
