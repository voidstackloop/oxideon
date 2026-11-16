//! Kontsevich Motivic Integration, Grothendieck Ring of Varieties $K_0(\mathrm{Var}_k)$ & Hodge-Deligne Polynomials.
//!
//! ### 1. Grothendieck Ring $K_0(\mathrm{Var}_k)$ & Lefschetz Motive $\mathbb{L} = [\mathbb{A}^1_k]$
//! The Hodge-Deligne realization $E: K_0(\mathrm{Var}_{\mathbb{C}}) \to \mathbb{Z}[u, v]$ is a ring homomorphism sending
//! the affine line to the Lefschetz motive $E(\mathbb{L}) = u v$ and projective space to:
//! $$E([\mathbb{P}^n]) = 1 + (uv) + (uv)^2 + \cdots + (uv)^n$$
//!
//! ### 2. Kontsevich Motivic Igusa Zeta Integral
//! For an $A_m$ surface singularity or monomial ideal of multiplicity $a$ and log-discrepancy $k$, the geometric
//! motivic measure evaluates to $\mu_{\text{mot}} = \frac{\mathbb{L} - 1}{\mathbb{L}^{a+k} - \mathbb{L}^k}$ in $\mathcal{M}_{\text{loc}} = K_0(\mathrm{Var})[\mathbb{L}^{-1}]$.

use crate::error::{Error, Result};

/// Kontsevich Motivic Measure & Hodge-Deligne Polynomial Realization Engine.
#[derive(Debug, Clone, Copy, Default)]
pub struct MotivicHodgeEngine;

impl MotivicHodgeEngine {
    /// Creates a new Motivic Hodge engine.
    pub fn new() -> Self {
        Self
    }

    /// Evaluates the Hodge-Deligne polynomial $E([\mathbb{P}^n]; u, v) = \sum_{k=0}^n (u v)^k$ of projective $n$-space.
    /// Beyond a million terms the geometric series is summed in closed form instead of term by term.
    pub fn projective_space_motive(dim_n: usize, u: f64, v: f64) -> f64 {
        let l = u * v;
        if dim_n > 1_000_000 {
            return if l == 1.0 {
                dim_n as f64 + 1.0
            } else {
                (1.0 - l.powf(dim_n as f64 + 1.0)) / (1.0 - l)
            };
        }
        let mut sum = 0.0;
        let mut power = 1.0;
        for _ in 0..=dim_n {
            sum += power;
            power *= l;
        }
        sum
    }

    /// Evaluates the Hodge-Deligne polynomial of the Grassmannian $G(2, 4)$:
    /// $[G(2, 4)] = 1 + \mathbb{L} + 2\mathbb{L}^2 + \mathbb{L}^3 + \mathbb{L}^4$.
    pub fn grassmannian_g24_motive(u: f64, v: f64) -> f64 {
        let l = u * v;
        1.0 + l + 2.0 * l * l + l.powi(3) + l.powi(4)
    }

    /// Evaluates the localized Kontsevich motivic arc integral $\int_{\mathcal{L}_\infty(\mathbb{A}^d)} \mathbb{L}^{-\mathrm{ord}_t(x_1^a)} d\mu_{\text{mot}}$
    /// at Lefschetz parameter $L > 1$.
    pub fn kontsevich_arc_integral(dim_d: u32, order_a: u32, lefschetz_l: f64) -> Result<f64> {
        if lefschetz_l <= 1.0 || dim_d == 0 || order_a == 0 {
            return Err(Error::Msg(
                "lefschetz_l must be > 1 and dim_d, order_a >= 1".to_string(),
            ));
        }
        let l_minus_1 = 1.0 - lefschetz_l.powi(-(dim_d as i32));
        let denom = 1.0 - lefschetz_l.powi(-((dim_d + order_a) as i32));
        Ok(l_minus_1 / denom)
    }
}
