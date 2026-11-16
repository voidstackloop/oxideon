//! Balog-Szemerédi-Gowers (BSG) Theorem & Freiman-Ruzsa Doubling Structure (Freiman, 1973; Balog & Szemerédi, 1994; Gowers, 2001).
//!
//! ### 1. Doubling Constant $K = |A + A| / |A|$ & Additive Energy $E(A, B)$
//! For finite $A \subset G$ in an abelian group, the additive energy is $E(A) = |\{(a_1, a_2, a_3, a_4) \in A^4 : a_1 + a_2 = a_3 + a_4\}| \ge |A|^4 / |A + A|$.
//! By Plünnecke-Ruzsa inequalities, $|k A - l A| \le K^{k + l} |A|$.

use crate::error::{Error, Result};

/// Balog-Szemerédi-Gowers & Freiman-Ruzsa Additive Combinatorics Engine.
#[derive(Debug, Clone, Copy)]
pub struct BalogSzemerediGowersFreiman {
    /// Cardinality $|A| \ge 1$.
    pub set_size_a: usize,
}

impl Default for BalogSzemerediGowersFreiman {
    fn default() -> Self {
        Self { set_size_a: 100 }
    }
}

impl BalogSzemerediGowersFreiman {
    /// Creates a new Balog-Szemerédi-Gowers & Freiman engine.
    pub fn new(set_size_a: usize) -> Result<Self> {
        if set_size_a == 0 {
            return Err(Error::Msg("set_size_a must be >= 1".to_string()));
        }
        Ok(Self { set_size_a })
    }

    /// Evaluates the doubling constant $K = |A + A| / |A|$, Cauchy-Schwarz lower bound on additive energy $|A|^3 / K$, and Plünnecke-Ruzsa bound $K^4 |A|$ on $|2A - 2A|$.
    pub fn doubling_energy_and_plunnecke_bound(
        &self,
        sumset_size: usize,
    ) -> Result<(f64, f64, f64)> {
        if sumset_size < self.set_size_a {
            return Err(Error::Msg("sumset_size must be >= set_size_a".to_string()));
        }
        let a = self.set_size_a as f64;
        let k = sumset_size as f64 / a;
        let energy_lb = (a * a * a) / k;
        let plunnecke_2a_2a = k.powi(4) * a;
        Ok((k, energy_lb, plunnecke_2a_2a))
    }
}
