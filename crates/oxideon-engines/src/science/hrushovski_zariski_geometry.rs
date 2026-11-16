//! Hrushovski-Zilber Zariski Geometries & Hrushovski Pre-Dimension Amalgamation (Hrushovski, 1993; Hrushovski & Zilber, 1996).
//!
//! ### 1. Hrushovski Pre-Dimension $\delta(A) = \alpha |V(A)| - |E(A)|$
//! In Hrushovski's ab initio construction of strongly minimal sets and Zariski geometries, a finite relational structure $A = (V, E)$
//! carries the pre-dimension $\delta(A) = \alpha |V| - |E|$. A substructure $A \le B$ is self-sufficient (strong) iff $\delta(A) \le \delta(B')$ for all $A \subseteq B' \subseteq B$.

use crate::error::{Error, Result};

/// Hrushovski-Zilber Zariski Geometry & Pre-Dimension Amalgamation Engine.
#[derive(Debug, Clone, Copy)]
pub struct HrushovskiZariskiGeometry {
    /// Irrational or rational weight $\alpha > 0$ in $\delta(A) = \alpha |V| - |E|$.
    pub alpha: f64,
}

impl Default for HrushovskiZariskiGeometry {
    fn default() -> Self {
        Self { alpha: 2.0 }
    }
}

impl HrushovskiZariskiGeometry {
    /// Creates a new Hrushovski-Zilber Zariski Geometry engine.
    pub fn new(alpha: f64) -> Result<Self> {
        if alpha <= 0.0 || !alpha.is_finite() {
            return Err(Error::Msg(
                "Pre-dimension weight alpha must be > 0".to_string(),
            ));
        }
        Ok(Self { alpha })
    }

    /// Evaluates the Hrushovski pre-dimension $\delta(A) = \alpha |V| - |E|$ and checks if the extension $A \subseteq B$ has non-negative relative pre-dimension $\delta(B/A) \ge 0$.
    pub fn predimension_and_strong_extension(
        &self,
        v_a: usize,
        e_a: usize,
        v_b: usize,
        e_b: usize,
    ) -> Result<(f64, f64, bool)> {
        if v_b < v_a || e_b < e_a {
            return Err(Error::Msg("Extension B must contain A".to_string()));
        }
        let delta_a = self.alpha * (v_a as f64) - (e_a as f64);
        let delta_b = self.alpha * (v_b as f64) - (e_b as f64);
        let rel = delta_b - delta_a;
        Ok((delta_a, rel, rel >= -1e-12))
    }
}
