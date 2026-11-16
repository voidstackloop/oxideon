//! Caffarelli-Silvestre Degenerate Harmonic Extension for the Fractional Laplacian $(-\Delta)^s$ (Caffarelli & Silvestre, 2007).
//!
//! ### 1. Dirichlet-to-Neumann Extension in $\mathbb{R}^{n+1}_+$ with Weight $y^{1 - 2s}$
//! For $s \in (0, 1)$, solving $\mathrm{div}(y^{1 - 2s} \nabla U(x, y)) = 0$ on $\mathbb{R}^n \times (0, \infty)$ with $U(x, 0) = u(x)$
//! realizes the nonlocal fractional Laplacian via the local weighted Neumann trace:
//! $$(-\Delta)^s u(x) = -d_s \lim_{y \to 0^+} y^{1 - 2s} \partial_y U(x, y), \qquad \widehat{(-\Delta)^s u}(\xi) = |\xi|^{2s} \hat{u}(\xi)$$

use crate::error::{Error, Result};

/// Caffarelli-Silvestre Fractional Laplacian Extension Engine.
#[derive(Debug, Clone, Copy)]
pub struct CaffarelliSilvestreFractionalExt {
    /// Fractional power $s \in (0, 1)$.
    pub fractional_s: f64,
}

impl Default for CaffarelliSilvestreFractionalExt {
    fn default() -> Self {
        Self { fractional_s: 0.5 }
    }
}

impl CaffarelliSilvestreFractionalExt {
    /// Creates a new Caffarelli-Silvestre Fractional Extension engine.
    pub fn new(fractional_s: f64) -> Result<Self> {
        if !(0.0..1.0).contains(&fractional_s) || fractional_s == 0.0 {
            return Err(Error::Msg("fractional_s must be in (0, 1)".to_string()));
        }
        Ok(Self { fractional_s })
    }

    /// Evaluates the Muckenhoupt $A_2$ weight exponent $a = 1 - 2s \in (-1, 1)$ and the multiplier $|\xi|^{2s} e^{-|\xi| y}$ at frequency $|\xi| > 0$ and height $y \ge 0$.
    /// This is the exact Neumann-trace multiplier at $y = 0$ (the symbol of $(-\Delta)^s$) and for every $y$ when $s = 1/2$ (Poisson extension);
    /// for other $s$ and $y > 0$ the exact multiplier involves a modified Bessel function $K_s$, which is not evaluated here.
    pub fn muckenhoupt_exponent_and_fourier_multiplier(
        &self,
        xi_norm: f64,
        height_y: f64,
    ) -> Result<(f64, f64)> {
        if xi_norm <= 0.0 || height_y < 0.0 {
            return Err(Error::Msg(
                "xi_norm > 0 and height_y >= 0 required".to_string(),
            ));
        }
        let a_exp = 1.0 - 2.0 * self.fractional_s;
        let symbol = xi_norm.powf(2.0 * self.fractional_s) * (-xi_norm * height_y).exp();
        Ok((a_exp, symbol))
    }
}
