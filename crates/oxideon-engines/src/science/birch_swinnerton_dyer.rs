//! Birch and Swinnerton-Dyer (BSD) Conjecture, Néron-Tate Height Regulator & Shafarevich-Tate Group $\mathrm{Sha}(E/\mathbb{Q})$.
//!
//! ### 1. Exact BSD Leading Taylor Coefficient Formula at $s = 1$
//! For an elliptic curve $E/\mathbb{Q}$ of algebraic Mordell-Weil rank $r = \operatorname{rk} E(\mathbb{Q})$,
//! the Birch and Swinnerton-Dyer Conjecture states $\operatorname{ord}_{s=1} L(E, s) = r$ and:
//! $$\lim_{s \to 1} \frac{L(E, s)}{(s - 1)^r} = \frac{\#\mathrm{Sha}(E/\mathbb{Q}) \cdot \Omega_E \cdot \operatorname{Reg}(E/\mathbb{Q}) \cdot \prod_p c_p}{\left(\# E(\mathbb{Q})_{\text{tors}}\right)^2}$$

use crate::error::{Error, Result};

/// Birch and Swinnerton-Dyer (BSD) Conjecture & Néron-Tate Regulator Engine.
#[derive(Debug, Clone, Copy, Default)]
pub struct BirchSwinnertonDyerEngine;

impl BirchSwinnertonDyerEngine {
    /// Creates a new Birch and Swinnerton-Dyer engine.
    pub fn new() -> Self {
        Self
    }

    /// Evaluates the predicted BSD leading $L$-series coefficient $\frac{L^{(r)}(E, 1)}{r!}$ from arithmetic invariants.
    pub fn bsd_leading_coefficient(
        sha_order: u64,
        real_period_omega: f64,
        neron_tate_regulator: f64,
        tamagawa_product: u64,
        torsion_order: u64,
    ) -> Result<f64> {
        if sha_order == 0
            || real_period_omega <= 0.0
            || neron_tate_regulator <= 0.0
            || tamagawa_product == 0
            || torsion_order == 0
        {
            return Err(Error::Msg(
                "All BSD arithmetic invariants must be strictly positive".to_string(),
            ));
        }
        let num = (sha_order as f64)
            * real_period_omega
            * neron_tate_regulator
            * (tamagawa_product as f64);
        let den = (torsion_order as f64).powi(2);
        Ok(num / den)
    }
}
