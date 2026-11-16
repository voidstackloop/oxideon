//! Finite-Strain Hyperelasticity, Mooney-Rivlin & Ogden Strain Energy Potentials $W(\lambda_1, \lambda_2, \lambda_3)$ (Mooney, 1940; Ogden, 1972).
//!
//! ### 1. Incompressible Mooney-Rivlin & Neo-Hookean Principal Stretch Energy
//! For principal stretches $(\lambda_1, \lambda_2, \lambda_3)$ with first and second Cauchy-Green invariants
//! $I_1 = \lambda_1^2 + \lambda_2^2 + \lambda_3^2$ and $I_2 = \lambda_1^2 \lambda_2^2 + \lambda_2^2 \lambda_3^2 + \lambda_3^2 \lambda_1^2$,
//! the Mooney-Rivlin strain energy density is:
//! $$W_{\mathrm{MR}}(\lambda_1, \lambda_2, \lambda_3) = C_{10} (I_1 - 3) + C_{01} (I_2 - 3)$$

use crate::error::{Error, Result};

/// Mooney-Rivlin & Ogden Finite-Strain Hyperelasticity Engine.
#[derive(Debug, Clone, Copy)]
pub struct HyperelasticStrainEnergyEngine {
    /// Neo-Hookean shear modulus coefficient $C_{10} \ge 0$.
    pub c10: f64,
    /// Second-invariant Mooney-Rivlin coefficient $C_{01} \ge 0$.
    pub c01: f64,
}

impl HyperelasticStrainEnergyEngine {
    /// Creates a new Mooney-Rivlin Hyperelastic engine.
    pub fn new(c10: f64, c01: f64) -> Result<Self> {
        if c10 < 0.0 || c01 < 0.0 || (c10 + c01) <= 0.0 {
            return Err(Error::Msg(
                "Requires c10, c01 >= 0 and c10 + c01 > 0".to_string(),
            ));
        }
        Ok(Self { c10, c01 })
    }

    /// Evaluates the incompressible uniaxial tension strain energy $W(\lambda)$ and Cauchy true stress $\sigma_{11}(\lambda) = 2(\lambda^2 - \lambda^{-1})(C_{10} + C_{01} \lambda^{-1})$
    /// for axial stretch $\lambda_1 = \lambda > 0$ ($\lambda_2 = \lambda_3 = \lambda^{-1/2}$).
    pub fn uniaxial_energy_and_cauchy_stress(&self, stretch_lambda: f64) -> Result<(f64, f64)> {
        if stretch_lambda <= 0.0 {
            return Err(Error::Msg("stretch_lambda must be > 0".to_string()));
        }
        let l2 = stretch_lambda * stretch_lambda;
        let inv_l = 1.0 / stretch_lambda;
        let i1 = l2 + 2.0 * inv_l;
        let i2 = 2.0 * stretch_lambda + 1.0 / l2;
        let w = self.c10 * (i1 - 3.0) + self.c01 * (i2 - 3.0);
        let sigma = 2.0 * (l2 - inv_l) * (self.c10 + self.c01 * inv_l);
        Ok((w, sigma))
    }
}
