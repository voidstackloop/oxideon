//! Szemerédi Regularity Lemma, Discrepancy $\varepsilon$-Regular Pairs & Green-Tao Arithmetic Progressions (Szemerédi, 1975; Green & Tao, 2004).
//!
//! ### 1. $\varepsilon$-Regular Bipartite Edge Density & Triangle Counting Lemma
//! Given disjoint vertex subsets $(A, B)$ with edge density $d(A, B) = \frac{e(A, B)}{|A| |B|}$, if $(A, B)$, $(B, C)$, and $(A, C)$ are
//! $\varepsilon$-regular with densities $d_{AB}, d_{BC}, d_{AC} \ge 2\varepsilon$, the Triangle Counting Lemma guarantees at least
//! $(1 - 2\varepsilon)(d_{AB} - \varepsilon)(d_{BC} - \varepsilon)(d_{AC} - \varepsilon) |A||B||C|$ triangles.

use crate::error::{Error, Result};

/// Szemerédi Regularity Lemma & Triangle Counting / Green-Tao Engine.
#[derive(Debug, Clone, Copy)]
pub struct SzemerediGreenTaoRegularity {
    /// Regularity discrepancy tolerance $\varepsilon \in (0, 0.5)$.
    pub epsilon: f64,
}

impl Default for SzemerediGreenTaoRegularity {
    fn default() -> Self {
        Self { epsilon: 0.05 }
    }
}

impl SzemerediGreenTaoRegularity {
    /// Creates a new Szemerédi-Green-Tao Regularity engine.
    pub fn new(epsilon: f64) -> Result<Self> {
        if !(0.0..0.5).contains(&epsilon) || epsilon == 0.0 {
            return Err(Error::Msg("epsilon must be in (0, 0.5)".to_string()));
        }
        Ok(Self { epsilon })
    }

    /// Evaluates the Triangle Counting Lemma lower bound on triangle density $(1 - 2\varepsilon)(d - \varepsilon)^3$ for symmetric tripartite density $d > 2\varepsilon$.
    pub fn triangle_density_lower_bound(&self, edge_density_d: f64) -> Result<f64> {
        if edge_density_d <= 2.0 * self.epsilon || edge_density_d > 1.0 {
            return Err(Error::Msg(
                "edge_density_d must be in (2*epsilon, 1]".to_string(),
            ));
        }
        let lb = (1.0 - 2.0 * self.epsilon) * (edge_density_d - self.epsilon).powi(3);
        Ok(lb)
    }
}
