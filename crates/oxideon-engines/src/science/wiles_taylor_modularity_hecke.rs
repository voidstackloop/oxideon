//! Wiles-Taylor-Breuil-Conrad-Diamond Modularity Theorem $R = \mathbb{T}$ & Ramanujan-Petersson Hecke Bound (Wiles, 1995; Breuil, Conrad, Diamond & Taylor, 2001).
//!
//! ### 1. Galois Deformation Ring Isomorphism $R \cong \mathbb{T}$ & Hasse-Deligne Weight-$k$ Hecke Bound $|a_p(f)| \le 2 p^{(k-1)/2}$
//! Every elliptic curve $E / \mathbb{Q}$ of conductor $N$ is modular ($a_p(E) = p + 1 - \# E(\mathbb{F}_p) = a_p(f)$ for a weight-$2$ newform $f \in S_2(\Gamma_0(N))$),
//! satisfying $|a_p(f)| \le 2 \sqrt{p}$.

use crate::error::{Error, Result};

/// Wiles-Taylor Modularity $R = \mathbb{T}$ & Hecke Eigenvalue Engine.
#[derive(Debug, Clone, Copy)]
pub struct WilesTaylorModularityHecke {
    /// Modular form weight $k \ge 2$.
    pub weight_k: u32,
}

impl Default for WilesTaylorModularityHecke {
    fn default() -> Self {
        Self { weight_k: 2 }
    }
}

impl WilesTaylorModularityHecke {
    /// Creates a new Wiles-Taylor Modularity & Hecke engine.
    pub fn new(weight_k: u32) -> Result<Self> {
        if weight_k < 2 {
            return Err(Error::Msg("weight_k must be >= 2".to_string()));
        }
        Ok(Self { weight_k })
    }

    /// Evaluates the Ramanujan-Petersson-Deligne bound $2 p^{(k-1)/2}$ and normalized Sato-Tate angle $\cos \theta_p = a_p / (2 p^{(k-1)/2})$.
    pub fn ramanujan_bound_and_sato_tate_cos(
        &self,
        prime_p: u64,
        hecke_ap: f64,
    ) -> Result<(f64, f64, bool)> {
        if prime_p < 2 {
            return Err(Error::Msg("prime_p must be >= 2".to_string()));
        }
        if self.weight_k < 2 {
            return Err(Error::Msg(format!(
                "weight_k must be >= 2, got {}",
                self.weight_k
            )));
        }
        let bound = 2.0 * (prime_p as f64).powf(0.5 * (self.weight_k - 1) as f64);
        let cos_theta = hecke_ap / bound;
        Ok((bound, cos_theta, hecke_ap.abs() <= bound + 1e-12))
    }
}
