//! McLerran-Venugopalan Color Glass Condensate (CGC) & Small-$x$ Gluon Saturation Scale $Q_s^2(x, A)$ (McLerran & Venugopalan, 1994; Golec-Biernat & Wüsthoff, 1998).
//!
//! ### 1. Nuclear Gluon Saturation Momentum $Q_s^2(x, A)$ & Color Dipole Cross Section
//! At small Bjorken-$x$ in a nucleus of mass number $A$, gluon recombination balances BFKL splitting at the saturation scale:
//! $$Q_s^2(x, A) = Q_0^2 A^{1/3} \left(\frac{x_0}{x}\right)^\lambda, \qquad \sigma_{\mathrm{dip}}(r, x) = \sigma_0 \left(1 - \exp\!\left(-\frac{r^2 Q_s^2(x, A)}{4}\right)\right)$$

use crate::error::{Error, Result};

/// McLerran-Venugopalan Color Glass Condensate & Saturation Scale Engine.
#[derive(Debug, Clone, Copy)]
pub struct ColorGlassCondensateSaturation {
    /// Reference saturation scale $Q_0^2 > 0$ ($\mathrm{GeV}^2$).
    pub q0_sq_gev2: f64,
    /// Reference Bjorken $x_0 \in (0, 1)$.
    pub x0: f64,
    /// Small-$x$ growth exponent $\lambda \approx 0.288$.
    pub lambda_exp: f64,
}

impl Default for ColorGlassCondensateSaturation {
    fn default() -> Self {
        Self {
            q0_sq_gev2: 1.0,
            x0: 3.0e-4,
            lambda_exp: 0.288,
        }
    }
}

impl ColorGlassCondensateSaturation {
    /// Creates a new Color Glass Condensate Saturation engine.
    pub fn new(q0_sq_gev2: f64, x0: f64, lambda_exp: f64) -> Result<Self> {
        if q0_sq_gev2 <= 0.0 || !(0.0..1.0).contains(&x0) || x0 == 0.0 || lambda_exp <= 0.0 {
            return Err(Error::Msg("Invalid CGC saturation parameters".to_string()));
        }
        Ok(Self {
            q0_sq_gev2,
            x0,
            lambda_exp,
        })
    }

    /// Evaluates nuclear saturation scale $Q_s^2(x, A)$ and GBW dipole scattering probability $1 - \exp(-r^2 Q_s^2 / 4)$.
    pub fn saturation_scale_and_dipole_scattering(
        &self,
        bjorken_x: f64,
        mass_number_a: u32,
        dipole_r_gev_inv: f64,
    ) -> Result<(f64, f64)> {
        if !(0.0..1.0).contains(&bjorken_x)
            || bjorken_x == 0.0
            || mass_number_a == 0
            || dipole_r_gev_inv < 0.0
        {
            return Err(Error::Msg(
                "Invalid bjorken_x, mass_number_a, or dipole_r".to_string(),
            ));
        }
        let qs_sq = self.q0_sq_gev2
            * (mass_number_a as f64).cbrt()
            * (self.x0 / bjorken_x).powf(self.lambda_exp);
        let n_dip = 1.0 - (-0.25 * dipole_r_gev_inv.powi(2) * qs_sq).exp();
        Ok((qs_sq, n_dip))
    }
}
