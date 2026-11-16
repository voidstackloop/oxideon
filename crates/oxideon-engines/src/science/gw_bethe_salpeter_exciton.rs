//! Hedin's $GW$ Quasiparticle Self-Energy $\Sigma = i G W$ & Bethe-Salpeter Exciton Optical Gap (Hedin, 1965; Strinati, 1988).
//!
//! ### 1. Linearized $G_0 W_0$ Quasiparticle Energy & Wannier-Mott Exciton Binding
//! Expanding Hedin's self-energy $\Sigma_n(\omega)$ around the Kohn-Sham eigenvalue $\varepsilon_n^{\mathrm{KS}}$ with
//! renormalization weight $Z_n = \left(1 - \left.\frac{\partial \operatorname{Re} \Sigma_n}{\partial \omega}\right|_{\varepsilon_n^{\mathrm{KS}}}\right)^{-1} \in (0, 1]$ yields:
//! $$E_n^{\mathrm{QP}} = \varepsilon_n^{\mathrm{KS}} + Z_n \left\langle \psi_n \left| \operatorname{Re}\Sigma(\varepsilon_n^{\mathrm{KS}}) - V_{xc} \right| \psi_n \right\rangle$$

use crate::error::{Error, Result};

/// Hedin $G_0 W_0$ Quasiparticle & Bethe-Salpeter Exciton Engine.
#[derive(Debug, Clone, Copy, Default)]
pub struct HedinGwBetheSalpeterEngine;

impl HedinGwBetheSalpeterEngine {
    /// Creates a new Hedin $GW$ & Bethe-Salpeter engine.
    pub fn new() -> Self {
        Self
    }

    /// Evaluates the linearized $G_0 W_0$ quasiparticle energy $E^{\mathrm{QP}}$ and renormalization weight $Z \in (0, 1]$.
    pub fn g0w0_quasiparticle_energy(
        ks_eigenvalue: f64,
        sigma_minus_vxc: f64,
        dsigma_domega: f64,
    ) -> Result<(f64, f64)> {
        if dsigma_domega > 0.0 {
            return Err(Error::Msg(
                "dsigma_domega must be <= 0 for physical quasiparticle weight Z in (0, 1]"
                    .to_string(),
            ));
        }
        let z = 1.0 / (1.0 - dsigma_domega);
        let eqp = ks_eigenvalue + z * sigma_minus_vxc;
        Ok((eqp, z))
    }
}
