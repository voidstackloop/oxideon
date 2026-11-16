//! Huisken Mean Curvature Flow (MCF), Shrinking Hyperspheres & Colding-Minicozzi Entropy (Huisken, 1984; Colding & Minicozzi, 2012).
//!
//! ### 1. Huisken's Convex Hypersurface Theorem
//! An $n$-dimensional sphere $S^n(R(t)) \subset \mathbb{R}^{n+1}$ evolving by Mean Curvature Flow $\partial_t X = -H \mathbf{N}$
//! ($H = n / R$) satisfies $\frac{dR}{dt} = -\frac{n}{R}$, giving:
//! $$R(t) = \sqrt{R_0^2 - 2 n t}, \qquad T_{\text{ext}} = \frac{R_0^2}{2 n}$$

use super::ln_unit_sphere_area;
use crate::error::{Error, Result};

/// Huisken Mean Curvature Flow & Colding-Minicozzi Entropy Engine.
#[derive(Debug, Clone, Copy)]
pub struct MeanCurvatureFlowEngine {
    /// Hypersurface dimension $n \ge 1$.
    pub dim_n: usize,
    /// Initial radius $R_0 > 0$.
    pub initial_radius: f64,
}

impl MeanCurvatureFlowEngine {
    /// Creates a new Mean Curvature Flow engine.
    pub fn new(dim_n: usize, initial_radius: f64) -> Result<Self> {
        if dim_n == 0 || initial_radius <= 0.0 {
            return Err(Error::Msg(
                "dim_n >= 1 and initial_radius > 0 required".to_string(),
            ));
        }
        Ok(Self {
            dim_n,
            initial_radius,
        })
    }

    /// Computes Huisken's sphere extinction time $T_{\text{ext}} = R_0^2 / (2 n)$.
    pub fn extinction_time(&self) -> f64 {
        (self.initial_radius * self.initial_radius) / (2.0 * (self.dim_n as f64))
    }

    /// Evaluates the Colding-Minicozzi Gaussian surface entropy $\lambda(S^n) = \left(\frac{n}{2 \pi e}\right)^{n/2} \mathrm{Area}(S^n_1)$ for every $n \ge 1$,
    /// e.g. $\lambda(S^1) = \sqrt{2 \pi / e}$ and $\lambda(S^2) = 4 / e$. It decreases strictly toward $\sqrt{2}$ as $n \to \infty$ (Stone).
    pub fn colding_minicozzi_sphere_entropy(&self) -> f64 {
        let n = self.dim_n as f64;
        // Log space: the factor (n / 2 pi e)^(n/2) overflows and the sphere area underflows long before their product does.
        (0.5 * n * (n / (2.0 * std::f64::consts::PI * std::f64::consts::E)).ln()
            + ln_unit_sphere_area(self.dim_n))
        .exp()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::{E, PI};

    fn entropy(n: usize) -> f64 {
        MeanCurvatureFlowEngine::new(n, 1.0)
            .unwrap()
            .colding_minicozzi_sphere_entropy()
    }

    #[test]
    fn known_sphere_entropies() {
        assert!((entropy(1) - (2.0 * PI / E).sqrt()).abs() < 1e-12);
        assert!((entropy(2) - 4.0 / E).abs() < 1e-12);
        assert!((entropy(3) - (3.0 / (2.0 * PI * E)).powf(1.5) * 2.0 * PI * PI).abs() < 1e-12);
    }

    /// Stone: lambda(S^n) decreases strictly and stays above sqrt(2). A wrong sphere area breaks this for n >= 3.
    #[test]
    fn entropy_decreases_toward_sqrt2() {
        let mut prev = f64::INFINITY;
        for n in 1..=30 {
            let l = entropy(n);
            assert!(l < prev && l > 2f64.sqrt(), "n={n}: {l}");
            prev = l;
        }
    }

    /// Log-space evaluation stays finite for huge n (the naive product is inf * 0) and converges to sqrt(2).
    #[test]
    fn huge_dimension_converges_to_sqrt2() {
        assert!((entropy(1_000_000) - 2f64.sqrt()).abs() < 1e-3);
        assert!(entropy(usize::MAX / 4).is_finite());
    }
}
