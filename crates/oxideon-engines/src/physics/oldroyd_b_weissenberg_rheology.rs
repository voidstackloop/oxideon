//! Oldroyd-B Upper-Convected Maxwell Viscoelasticity, Weissenberg Number $\mathrm{Wi}$ & First Normal Stress Difference $N_1$ (Oldroyd, 1950).
//!
//! ### 1. Steady Simple Shear Stress & Rod-Climbing Normal Stress $N_1$
//! For solvent viscosity $\eta_s$, polymer viscosity $\eta_p$, and polymer relaxation time $\lambda$ under steady shear rate $\dot{\gamma}$:
//! $$\mathrm{Wi} = \lambda \dot{\gamma}, \qquad \sigma_{xy} = (\eta_s + \eta_p) \dot{\gamma}, \qquad N_1 = \sigma_{xx} - \sigma_{yy} = 2 \eta_p \lambda \dot{\gamma}^2$$

use crate::error::{Error, Result};

/// Oldroyd-B Viscoelastic Rheology & Weissenberg Normal-Stress Engine.
#[derive(Debug, Clone, Copy)]
pub struct OldroydBViscoelasticRheology {
    /// Solvent viscosity $\eta_s \ge 0$.
    pub eta_solvent: f64,
    /// Polymer viscosity $\eta_p > 0$.
    pub eta_polymer: f64,
    /// Polymer relaxation time $\lambda > 0$.
    pub relaxation_lambda: f64,
}

impl Default for OldroydBViscoelasticRheology {
    fn default() -> Self {
        Self {
            eta_solvent: 1.0,
            eta_polymer: 4.0,
            relaxation_lambda: 0.5,
        }
    }
}

impl OldroydBViscoelasticRheology {
    /// Creates a new Oldroyd-B Viscoelastic Rheology engine.
    pub fn new(eta_solvent: f64, eta_polymer: f64, relaxation_lambda: f64) -> Result<Self> {
        if eta_solvent < 0.0 || eta_polymer <= 0.0 || relaxation_lambda <= 0.0 {
            return Err(Error::Msg(
                "eta_solvent >= 0 and eta_polymer, relaxation_lambda > 0 required".to_string(),
            ));
        }
        Ok(Self {
            eta_solvent,
            eta_polymer,
            relaxation_lambda,
        })
    }

    /// Evaluates the Weissenberg number $\mathrm{Wi} = \lambda \dot{\gamma}$, shear stress $\sigma_{xy}$, and first normal stress difference $N_1 = 2 \eta_p \lambda \dot{\gamma}^2$.
    pub fn weissenberg_shear_and_normal_stress(
        &self,
        shear_rate_gamma_dot: f64,
    ) -> (f64, f64, f64) {
        let wi = self.relaxation_lambda * shear_rate_gamma_dot.abs();
        let sigma_xy = (self.eta_solvent + self.eta_polymer) * shear_rate_gamma_dot;
        let n1 = 2.0 * self.eta_polymer * self.relaxation_lambda * shear_rate_gamma_dot.powi(2);
        (wi, sigma_xy, n1)
    }
}
