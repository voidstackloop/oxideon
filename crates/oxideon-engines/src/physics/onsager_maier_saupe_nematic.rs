//! Onsager Hard-Rod & Landau-de Gennes / Maier-Saupe Nematic Liquid Crystal Phase Transition (Onsager, 1949; Maier & Saupe, 1958).
//!
//! ### 1. Landau-de Gennes Free Energy & First-Order Isotropic-Nematic Jump
//! For scalar nematic order parameter $S = \langle P_2(\cos \theta)\rangle \in [-1/2, 1]$, the Landau-de Gennes expansion is:
//! $$f(S) = \frac{1}{2} A (T - T^*) S^2 - \frac{1}{3} B S^3 + \frac{1}{4} C S^4$$
//! The first-order clearing transition occurs at $T_{NI} = T^* + \frac{2 B^2}{9 A C}$ with order-parameter jump $S_{NI} = \frac{2 B}{3 C}$.

use crate::error::{Error, Result};

/// Onsager / Landau-de Gennes Nematic Liquid Crystal Engine.
#[derive(Debug, Clone, Copy)]
pub struct OnsagerMaierSaupeNematic {
    /// Quadratic stiffness $A > 0$.
    pub coeff_a: f64,
    /// Cubic asymmetry coefficient $B > 0$.
    pub coeff_b: f64,
    /// Quartic stabilization coefficient $C > 0$.
    pub coeff_c: f64,
}

impl Default for OnsagerMaierSaupeNematic {
    fn default() -> Self {
        Self {
            coeff_a: 1.0,
            coeff_b: 1.5,
            coeff_c: 2.5,
        }
    }
}

impl OnsagerMaierSaupeNematic {
    /// Creates a new Onsager / Landau-de Gennes Nematic engine.
    pub fn new(coeff_a: f64, coeff_b: f64, coeff_c: f64) -> Result<Self> {
        if coeff_a <= 0.0 || coeff_b <= 0.0 || coeff_c <= 0.0 {
            return Err(Error::Msg(
                "Landau-de Gennes coefficients A, B, C must be > 0".to_string(),
            ));
        }
        Ok(Self {
            coeff_a,
            coeff_b,
            coeff_c,
        })
    }

    /// Evaluates the nematic-isotropic clearing temperature shift $\Delta T_{NI} = T_{NI} - T^* = 2 B^2 / (9 A C)$ and clearing order-parameter jump $S_{NI} = 2 B / (3 C)$.
    pub fn clearing_temperature_shift_and_order_jump(&self) -> (f64, f64) {
        let delta_t_ni = 2.0 * self.coeff_b.powi(2) / (9.0 * self.coeff_a * self.coeff_c);
        let s_ni = 2.0 * self.coeff_b / (3.0 * self.coeff_c);
        (delta_t_ni, s_ni)
    }
}
