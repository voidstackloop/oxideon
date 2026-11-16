//! May-Wigner Random Ecosystem Complexity-Stability Criterion & Girko Circular Law (May, 1972).
//!
//! ### 1. May's Stability Inequality
//! For an ecological community of $S$ species with self-regulation $-d < 0$, connectance $C \in (0, 1]$,
//! and interaction variance $\sigma^2$, the community Jacobian $J = -d I + M$ is asymptotically stable with high probability iff:
//! $$\sigma \sqrt{S C} < d$$

use crate::error::{Error, Result};

/// May-Wigner Random Ecosystem Stability & Circular Law Spectral Engine.
#[derive(Debug, Clone, Copy)]
pub struct MayWignerComplexityStability {
    /// Self-regulation damping rate $d > 0$.
    pub self_damping_d: f64,
}

impl Default for MayWignerComplexityStability {
    fn default() -> Self {
        Self {
            self_damping_d: 1.0,
        }
    }
}

impl MayWignerComplexityStability {
    /// Creates a new May-Wigner Ecosystem Stability engine.
    pub fn new(self_damping_d: f64) -> Result<Self> {
        if self_damping_d <= 0.0 || !self_damping_d.is_finite() {
            return Err(Error::Msg("self_damping_d must be > 0".to_string()));
        }
        Ok(Self { self_damping_d })
    }

    /// Evaluates the Girko circular spectral radius $R = \sigma \sqrt{S C}$, leading Lyapunov abscissa $R - d$, and May stability boolean.
    pub fn spectral_radius_and_stability(
        &self,
        num_species_s: usize,
        connectance_c: f64,
        interaction_sigma: f64,
    ) -> Result<(f64, f64, bool)> {
        if num_species_s == 0 || !(0.0..=1.0).contains(&connectance_c) || interaction_sigma < 0.0 {
            return Err(Error::Msg(
                "Invalid ecosystem complexity parameters".to_string(),
            ));
        }
        let radius = interaction_sigma * ((num_species_s as f64) * connectance_c).sqrt();
        let abscissa = radius - self.self_damping_d;
        Ok((radius, abscissa, abscissa < 0.0))
    }
}
