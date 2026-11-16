//! Fargues-Fontaine Curve $X_{\mathrm{FF}}$ & Fargues-Scholze Geometrization of the Local Langlands Correspondence (Fargues & Fontaine, 2018; Fargues & Scholze, 2021).
//!
//! ### 1. Harder-Narasimhan Slope Classification of Vector Bundles $\mathcal{O}(\lambda) = \mathcal{O}(d / h)$ on $X_{\mathrm{FF}}$
//! Every vector bundle $\mathcal{E}$ on the Fargues-Fontaine curve $X_{\mathrm{FF}}$ decomposes into stable bundles $\bigoplus_i \mathcal{O}(\lambda_i)$ indexed by rational slopes $\lambda_i = d_i / h_i \in \mathbb{Q}$ ($\gcd(d_i, h_i) = 1$, $h_i \ge 1$) of degree $d_i$ and rank $h_i$.

use crate::error::{Error, Result};

/// Fargues-Fontaine Curve Vector Bundle Slope & Harder-Narasimhan Engine.
#[derive(Debug, Clone, Copy)]
pub struct FarguesFontaineCurveEngine {
    /// Bundle degree $d \in \mathbb{Z}$.
    pub degree_d: i64,
    /// Bundle rank $h \ge 1$.
    pub rank_h: u64,
}

impl Default for FarguesFontaineCurveEngine {
    fn default() -> Self {
        Self {
            degree_d: 1,
            rank_h: 2,
        }
    }
}

impl FarguesFontaineCurveEngine {
    /// Creates a new Fargues-Fontaine Curve Bundle engine.
    pub fn new(degree_d: i64, rank_h: u64) -> Result<Self> {
        if rank_h == 0 {
            return Err(Error::Msg("rank_h must be >= 1".to_string()));
        }
        Ok(Self { degree_d, rank_h })
    }

    /// Evaluates the Harder-Narasimhan slope $\mu(\mathcal{E}) = d / h$ and checks whether $\mathcal{E}$ is semistable of slope $0$ (étale local system locus).
    pub fn hn_slope_and_is_etale_semistable(&self) -> (f64, bool) {
        let slope = self.degree_d as f64 / self.rank_h as f64;
        (slope, self.degree_d == 0)
    }
}
