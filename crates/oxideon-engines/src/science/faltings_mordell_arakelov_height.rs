//! Faltings' Theorem (Mordell Conjecture) & Faltings-Arakelov Stable Modular Height (Faltings, 1983; Parshin, 1968).
//!
//! ### 1. Finiteness of Rational Points $X(K)$ for $g \ge 2$ & Néron-Tate Canonical Height
//! For a smooth projective curve $X / K$ of genus $g \ge 2$, Faltings proved $|X(K)| < \infty$ by bounding the Faltings height
//! $h_F(A)$ of the Jacobian $A = \operatorname{Jac}(X)$ across isogenies:
//! $$h_F(B) = h_F(A) + \frac{1}{2} \ln \deg(\phi) - \frac{1}{[K:\mathbb{Q}]} \ln |\Omega_{\phi}|$$

use crate::error::{Error, Result};

/// Faltings' Theorem & Arakelov Modular Height Engine.
#[derive(Debug, Clone, Copy)]
pub struct FaltingsMordellArakelovEngine {
    /// Curve genus $g$.
    pub genus_g: usize,
}

impl Default for FaltingsMordellArakelovEngine {
    fn default() -> Self {
        Self { genus_g: 2 }
    }
}

impl FaltingsMordellArakelovEngine {
    /// Creates a new Faltings-Mordell Arakelov engine.
    pub fn new(genus_g: usize) -> Self {
        Self { genus_g }
    }

    /// Evaluates whether $|X(K)|$ is unconditionally finite by Faltings ($g \ge 2$), the Euler characteristic $2 - 2g$, and the isogeny Faltings-height shift $\frac{1}{2}\ln d$.
    pub fn is_mordell_finite_and_isogeny_height_shift(
        &self,
        isogeny_degree: u64,
    ) -> Result<(bool, i64, f64)> {
        if isogeny_degree == 0 {
            return Err(Error::Msg("isogeny_degree must be >= 1".to_string()));
        }
        let finite = self.genus_g >= 2;
        let chi = 2 - 2 * (self.genus_g as i64);
        let shift = 0.5 * (isogeny_degree as f64).ln();
        Ok((finite, chi, shift))
    }
}
