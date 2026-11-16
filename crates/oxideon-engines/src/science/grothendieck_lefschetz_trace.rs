//! Grothendieck-Lefschetz Trace Formula & Deligne's Weil Conjectures Purity (Grothendieck, 1965; Deligne, 1974).
//!
//! ### 1. alternating Étale Frobenius Trace & Hasse-Weil Bound $|\# X(\mathbb{F}_{q^n}) - (q^n + 1)| \le 2 g q^{n/2}$
//! For a smooth projective curve $X / \mathbb{F}_q$ of genus $g$ with Frobenius eigenvalues $\alpha_j = \sqrt{q} e^{i \theta_j}$ on $H^1_{\text{ét}}(X_{\bar{\mathbb{F}}_q}, \mathbb{Q}_\ell)$:
//! $$\# X(\mathbb{F}_{q^n}) = q^n + 1 - \sum_{j=1}^{2g} \alpha_j^n, \qquad \left|\# X(\mathbb{F}_{q^n}) - (q^n + 1)\right| \le 2 g q^{n/2}$$

use crate::error::{Error, Result};

/// Grothendieck-Lefschetz Trace Formula & Deligne Weil Bound Engine.
#[derive(Debug, Clone, Copy)]
pub struct GrothendieckLefschetzTraceFormula {
    /// Prime power $q \ge 2$.
    pub field_size_q: u64,
    /// Curve genus $g$.
    pub genus_g: usize,
}

impl Default for GrothendieckLefschetzTraceFormula {
    fn default() -> Self {
        Self {
            field_size_q: 7,
            genus_g: 1,
        }
    }
}

impl GrothendieckLefschetzTraceFormula {
    /// Creates a new Grothendieck-Lefschetz Trace Formula engine.
    pub fn new(field_size_q: u64, genus_g: usize) -> Result<Self> {
        if field_size_q < 2 {
            return Err(Error::Msg("field_size_q must be >= 2".to_string()));
        }
        Ok(Self {
            field_size_q,
            genus_g,
        })
    }

    /// Evaluates the Hasse-Weil-Deligne upper bound $q^n + 1 + 2 g q^{n/2}$ and lower bound on $\# X(\mathbb{F}_{q^n})$.
    pub fn hasse_weil_deligne_point_bounds(&self, degree_n: u32) -> Result<(f64, f64)> {
        if degree_n == 0 {
            return Err(Error::Msg("degree_n must be >= 1".to_string()));
        }
        let n = i32::try_from(degree_n)
            .map_err(|_| Error::Msg("degree_n exceeds i32 range".to_string()))?;
        let qn = (self.field_size_q as f64).powi(n);
        let dev = 2.0 * (self.genus_g as f64) * qn.sqrt();
        Ok(((qn + 1.0 - dev).max(0.0), qn + 1.0 + dev))
    }
}
