//! Goppa Algebraic-Geometry (AG) Codes, Riemann-Roch Dimension & Tsfasman-Vlăduț-Zink Bound (Goppa, 1981; Tsfasman, Vlăduț & Zink, 1982).
//!
//! ### 1. Riemann-Roch Lower Bound on $[n, k, d]_q$ Goppa AG Codes & TVZ Asymptotic Bound
//! For a smooth projective curve $X / \mathbb{F}_q$ of genus $g$ with $n$ rational points and divisor $G$ ($2g - 2 < \deg(G) < n$),
//! the evaluation AG code $C_L(D, G)$ has dimension $k = \deg(G) + 1 - g$ and minimum distance $d \ge n - \deg(G)$, satisfying:
//! $$k + d \ge n + 1 - g$$
//! Using modular Shimura/Drinfeld towers attaining the Drinfeld-Vlăduț bound $N / g \to \sqrt{q} - 1$, the **TVZ Bound**
//! $R + \delta \ge 1 - \frac{1}{\sqrt{q} - 1}$ beats the Gilbert-Varshamov bound for all $q = p^{2m} \ge 49$!

use crate::error::{Error, Result};

/// Goppa Algebraic-Geometry (AG) Code & Tsfasman-Vlăduț-Zink Bound Engine.
#[derive(Debug, Clone, Copy)]
pub struct TsfasmanVladutZinkAgCode {
    /// Curve genus $g \ge 0$.
    pub genus_g: usize,
    /// Number of rational evaluation points $n$.
    pub length_n: usize,
    /// Divisor degree $\deg(G)$ in $(2g - 2, n)$.
    pub divisor_deg: usize,
}

impl TsfasmanVladutZinkAgCode {
    /// Creates a new Goppa Algebraic-Geometry code engine.
    pub fn new(genus_g: usize, length_n: usize, divisor_deg: usize) -> Result<Self> {
        if divisor_deg
            .checked_add(2)
            .is_none_or(|s| s <= genus_g.saturating_mul(2))
            || divisor_deg >= length_n
        {
            return Err(Error::Msg(
                "Requires 2g - 2 < divisor_deg < length_n".to_string(),
            ));
        }
        Ok(Self {
            genus_g,
            length_n,
            divisor_deg,
        })
    }

    /// Computes the exact Riemann-Roch dimension $k = \deg(G) + 1 - g$ and design minimum distance $d^* = n - \deg(G)$.
    pub fn riemann_roch_k_and_design_distance(&self) -> (usize, usize) {
        (
            self.divisor_deg + 1 - self.genus_g,
            self.length_n - self.divisor_deg,
        )
    }

    /// Evaluates the asymptotic Tsfasman-Vlăduț-Zink rate lower bound $R_{\mathrm{TVZ}}(\delta, q) = 1 - \delta - \frac{1}{\sqrt{q} - 1}$.
    pub fn tvz_asymptotic_rate(relative_distance_delta: f64, alphabet_q: f64) -> Result<f64> {
        if !relative_distance_delta.is_finite() || !(0.0..=1.0).contains(&relative_distance_delta) {
            return Err(Error::Msg(format!(
                "tvz_asymptotic_rate requires relative_distance_delta in [0, 1], got {relative_distance_delta}"
            )));
        }
        if !alphabet_q.is_finite() || alphabet_q <= 1.0 {
            return Err(Error::Msg(format!(
                "tvz_asymptotic_rate requires alphabet_q > 1, got {alphabet_q}"
            )));
        }
        Ok(1.0 - relative_distance_delta - 1.0 / (alphabet_q.sqrt() - 1.0))
    }
}
