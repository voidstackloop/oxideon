//! Kruskal-Shafranov MHD Kink Safety Factor $q(a) > 1$ & Troyon Beta Limit (Kruskal, 1954; Shafranov, 1956; Troyon, 1984).
//!
//! ### 1. Edge Safety Factor $q(a)$ & Troyon Maximum Toroidal Beta
//! For a tokamak of major radius $R_0$, minor radius $a$, toroidal field $B_T$, and poloidal field $B_p = \mu_0 I_p / (2 \pi a)$,
//! suppressing the external $m=1, n=1$ current-driven kink instability requires $q(a) = \frac{a B_T}{R_0 B_p} > 1$, while Troyon's limit bounds $\beta_{\mathrm{max}}(\%) = \beta_N \frac{I_p(\mathrm{MA})}{a(\mathrm{m}) B_T(\mathrm{T})}$.

use crate::error::{Error, Result};

/// Kruskal-Shafranov MHD Kink Safety Factor & Troyon Beta Limit Engine.
#[derive(Debug, Clone, Copy)]
pub struct KruskalShafranovTroyonMhd {
    /// Major radius $R_0 > 0$ (m).
    pub major_r0: f64,
    /// Minor radius $a > 0$ (m).
    pub minor_a: f64,
    /// Toroidal magnetic field $B_T > 0$ (T).
    pub b_toroidal: f64,
}

impl Default for KruskalShafranovTroyonMhd {
    fn default() -> Self {
        Self {
            major_r0: 6.2,
            minor_a: 2.0,
            b_toroidal: 5.3,
        }
    }
}

impl KruskalShafranovTroyonMhd {
    /// Creates a new Kruskal-Shafranov-Troyon MHD engine.
    pub fn new(major_r0: f64, minor_a: f64, b_toroidal: f64) -> Result<Self> {
        if major_r0 <= 0.0 || minor_a <= 0.0 || minor_a >= major_r0 || b_toroidal <= 0.0 {
            return Err(Error::Msg(
                "0 < minor_a < major_r0 and b_toroidal > 0 required".to_string(),
            ));
        }
        Ok(Self {
            major_r0,
            minor_a,
            b_toroidal,
        })
    }

    /// Evaluates the cylindrical edge safety factor $q_{\mathrm{cyl}}(a) = \frac{5 a^2 B_T}{R_0 I_p(\mathrm{MA})}$, Troyon beta limit $\beta_{\mathrm{max}}(\\%) = 2.8 \frac{I_p}{a B_T}$, and Kruskal-Shafranov stability ($q > 2$).
    pub fn safety_factor_and_troyon_beta(
        &self,
        plasma_current_ma: f64,
    ) -> Result<(f64, f64, bool)> {
        if plasma_current_ma <= 0.0 {
            return Err(Error::Msg("plasma_current_ma must be > 0".to_string()));
        }
        let q_a =
            5.0 * self.minor_a.powi(2) * self.b_toroidal / (self.major_r0 * plasma_current_ma);
        let beta_max_pct = 2.8 * plasma_current_ma / (self.minor_a * self.b_toroidal);
        Ok((q_a, beta_max_pct, q_a > 2.0))
    }
}
