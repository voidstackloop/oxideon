//! Prandtl-Blasius Laminar Boundary Layer ($C_f = 0.664 / \sqrt{\mathrm{Re}_x}$) & von Kármán Log-Law of the Wall (Prandtl, 1904; Blasius, 1908; von Kármán, 1930).
//!
//! ### 1. Blasius Skin Friction & Turbulent Log-Law $u^+ = \frac{1}{\kappa} \ln y^+ + B$
//! Over a flat plate at local Reynolds number $\mathrm{Re}_x = U_\infty x / \nu$:
//! $$C_f(x) = \frac{0.66411}{\sqrt{\mathrm{Re}_x}}, \qquad u^+(y^+) = \frac{1}{\kappa} \ln y^+ + B \quad (\kappa \approx 0.41, B \approx 5.2)$$

use crate::error::{Error, Result};

/// Prandtl-Blasius Boundary Layer & von Kármán Log-Law Engine.
#[derive(Debug, Clone, Copy)]
pub struct PrandtlBlasiusVonKarmanBoundary {
    /// von Kármán constant $\kappa \approx 0.41$.
    pub kappa_karman: f64,
    /// Smooth-wall log-law intercept $B \approx 5.2$.
    pub intercept_b: f64,
}

impl Default for PrandtlBlasiusVonKarmanBoundary {
    fn default() -> Self {
        Self {
            kappa_karman: 0.41,
            intercept_b: 5.2,
        }
    }
}

impl PrandtlBlasiusVonKarmanBoundary {
    /// Creates a new Prandtl-Blasius & von Kármán Boundary Layer engine.
    pub fn new(kappa_karman: f64, intercept_b: f64) -> Result<Self> {
        if kappa_karman <= 0.0 || !intercept_b.is_finite() {
            return Err(Error::Msg("kappa_karman must be > 0".to_string()));
        }
        Ok(Self {
            kappa_karman,
            intercept_b,
        })
    }

    /// Evaluates the Blasius laminar skin friction $C_f(\mathrm{Re}_x)$ and turbulent log-law velocity $u^+(y^+)$ for $y^+ \ge 30$.
    pub fn blasius_cf_and_log_law_u_plus(&self, re_x: f64, y_plus: f64) -> Result<(f64, f64)> {
        if re_x <= 0.0 || y_plus <= 1.0 {
            return Err(Error::Msg("re_x > 0 and y_plus > 1 required".to_string()));
        }
        let cf = 0.66411 / re_x.sqrt();
        let u_plus = y_plus.ln() / self.kappa_karman + self.intercept_b;
        Ok((cf, u_plus))
    }
}
