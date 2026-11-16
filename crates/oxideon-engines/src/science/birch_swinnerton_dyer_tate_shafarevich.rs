//! Birch and Swinnerton-Dyer (BSD) Conjecture Leading Coefficient & Tate-Shafarevich Group $\Sha(E/\mathbb{Q})$ (Birch & Swinnerton-Dyer, 1965; Gross & Zagier, 1986; Kolyvagin, 1989).
//!
//! ### 1. Exact BSD Leading Coefficient Formula at $s = 1$
//! For an elliptic curve $E / \mathbb{Q}$ of algebraic/analytic rank $r$:
//! $$\lim_{s \to 1} \frac{L(E, s)}{(s - 1)^r} = \frac{\Omega_E \cdot R_E \cdot |\Sha(E/\mathbb{Q})| \cdot \prod_p c_p}{|E(\mathbb{Q})_{\mathrm{tors}}|^2}$$

use crate::error::{Error, Result};

/// Birch and Swinnerton-Dyer (BSD) Leading Term & Sha Invariant Engine.
#[derive(Debug, Clone, Copy)]
pub struct BirchSwinnertonDyerLeadingTerm {
    /// Real period $\Omega_E > 0$.
    pub real_period_omega: f64,
    /// Torsion subgroup order $|E(\mathbb{Q})_{\mathrm{tors}}| \ge 1$.
    pub torsion_order: u64,
}

impl Default for BirchSwinnertonDyerLeadingTerm {
    fn default() -> Self {
        Self {
            real_period_omega: 3.0599,
            torsion_order: 1,
        }
    }
}

impl BirchSwinnertonDyerLeadingTerm {
    /// Creates a new BSD Leading Term engine.
    pub fn new(real_period_omega: f64, torsion_order: u64) -> Result<Self> {
        if real_period_omega <= 0.0 || torsion_order == 0 {
            return Err(Error::Msg(
                "real_period_omega > 0 and torsion_order >= 1 required".to_string(),
            ));
        }
        Ok(Self {
            real_period_omega,
            torsion_order,
        })
    }

    /// Evaluates the BSD leading $L$-series Taylor coefficient $L^{(r)}(E, 1) / r!$ given regulator $R_E$, $|\Sha|$, and Tamagawa product $\prod c_p$.
    pub fn bsd_leading_coefficient(
        &self,
        regulator_re: f64,
        sha_order: u64,
        tamagawa_product: u64,
    ) -> Result<f64> {
        if regulator_re <= 0.0 || sha_order == 0 || tamagawa_product == 0 {
            return Err(Error::Msg(
                "regulator_re > 0, sha_order >= 1, tamagawa_product >= 1 required".to_string(),
            ));
        }
        let num =
            self.real_period_omega * regulator_re * (sha_order as f64) * (tamagawa_product as f64);
        let den = (self.torsion_order * self.torsion_order) as f64;
        Ok(num / den)
    }
}
