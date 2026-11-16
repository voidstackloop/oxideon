//! Erdős-Rényi-Bollobás Giant Component Phase Transition in $G(n, c/n)$ (Erdős & Rényi, 1960; Bollobás, 1984).
//!
//! ### 1. Survival Fraction $\beta = 1 - e^{-c \beta}$ & Susceptibility Divergence $(1 - c)^{-1}$
//! In $G(n, p = c / n)$, for $c < 1$ the expected cluster size is $\chi(c) = 1 / (1 - c)$ and largest component is $O(\log n)$;
//! for $c > 1$, a unique giant component of size $\beta(c) n$ emerges where $\beta \in (0, 1)$ solves:
//! $$\beta = 1 - \exp(-c \beta)$$

use crate::error::{Error, Result};

/// Erdős-Rényi-Bollobás Random Graph Phase Transition Engine.
#[derive(Debug, Clone, Copy)]
pub struct ErdosRenyiBollobasPhaseTransition {
    /// Number of vertices $n \ge 10$.
    pub num_vertices_n: usize,
}

impl Default for ErdosRenyiBollobasPhaseTransition {
    fn default() -> Self {
        Self {
            num_vertices_n: 10_000,
        }
    }
}

impl ErdosRenyiBollobasPhaseTransition {
    /// Creates a new Erdős-Rényi-Bollobás Phase Transition engine.
    pub fn new(num_vertices_n: usize) -> Result<Self> {
        if num_vertices_n < 10 {
            return Err(Error::Msg("num_vertices_n must be >= 10".to_string()));
        }
        Ok(Self { num_vertices_n })
    }

    /// Computes the asymptotic giant component fraction $\beta(c)$ via fixed-point iteration and expected giant size $\beta(c) n$.
    pub fn giant_component_fraction_and_size(&self, mean_degree_c: f64) -> Result<(f64, f64)> {
        if mean_degree_c <= 0.0 || !mean_degree_c.is_finite() {
            return Err(Error::Msg("mean_degree_c must be > 0".to_string()));
        }
        if mean_degree_c <= 1.0 {
            return Ok((0.0, (self.num_vertices_n as f64).ln()));
        }
        let mut beta = 1.0 - (-mean_degree_c).exp();
        for _ in 0..80 {
            beta = 1.0 - (-mean_degree_c * beta).exp();
        }
        Ok((beta, beta * self.num_vertices_n as f64))
    }
}
