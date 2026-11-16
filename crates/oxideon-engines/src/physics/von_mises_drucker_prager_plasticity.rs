//! $J_2$ Von Mises Deviatoric Plasticity & Wilkins Radial Return Mapping with Isotropic Hardening (von Mises, 1913; Wilkins, 1964).
//!
//! ### 1. Exact Radial Return Algorithm in Deviatoric Stress Space
//! Given a trial deviatoric stress norm $\|\mathbf{s}^{\mathrm{trial}}\|$ and equivalent Von Mises trial stress
//! $q^{\mathrm{trial}} = \sqrt{\frac{3}{2}} \|\mathbf{s}^{\mathrm{trial}}\|$, plastic yielding occurs if $f^{\mathrm{trial}} = q^{\mathrm{trial}} - \sigma_y > 0$,
//! with exact plastic multiplier increment (under shear modulus $G$ and hardening modulus $H \ge 0$):
//! $$\Delta \gamma = \frac{q^{\mathrm{trial}} - \sigma_y}{3 G + H}, \qquad q_{n+1} = \sigma_y + H \Delta \gamma$$

use crate::error::{Error, Result};

/// $J_2$ Von Mises Plasticity & Radial Return Mapping Engine.
#[derive(Debug, Clone, Copy)]
pub struct J2PlasticityReturnMapping {
    /// Shear modulus $G > 0$.
    pub shear_g: f64,
    /// Initial yield stress $\sigma_y > 0$.
    pub yield_stress: f64,
    /// Linear isotropic hardening modulus $H \ge 0$.
    pub hardening_h: f64,
}

impl J2PlasticityReturnMapping {
    /// Creates a new $J_2$ Von Mises Radial Return engine.
    pub fn new(shear_g: f64, yield_stress: f64, hardening_h: f64) -> Result<Self> {
        if shear_g <= 0.0 || yield_stress <= 0.0 || hardening_h < 0.0 {
            return Err(Error::Msg(
                "shear_g > 0, yield_stress > 0, and hardening_h >= 0 required".to_string(),
            ));
        }
        Ok(Self {
            shear_g,
            yield_stress,
            hardening_h,
        })
    }

    /// Performs Wilkins radial return on trial Von Mises stress $q^{\mathrm{trial}} \ge 0$, returning `(updated_vm_stress, plastic_strain_increment)`.
    pub fn radial_return_step(&self, trial_von_mises_q: f64) -> Result<(f64, f64)> {
        if !trial_von_mises_q.is_finite() || trial_von_mises_q < 0.0 {
            return Err(Error::Msg(format!(
                "radial_return_step requires trial_von_mises_q >= 0, got {trial_von_mises_q}"
            )));
        }
        if trial_von_mises_q <= self.yield_stress {
            Ok((trial_von_mises_q, 0.0))
        } else {
            let dgamma =
                (trial_von_mises_q - self.yield_stress) / (3.0 * self.shear_g + self.hardening_h);
            let q_new = self.yield_stress + self.hardening_h * dgamma;
            Ok((q_new, dgamma))
        }
    }
}
