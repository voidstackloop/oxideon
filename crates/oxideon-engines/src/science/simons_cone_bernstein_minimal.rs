//! Simons Minimal Cone $C_{m, m} \subset \mathbb{R}^{2m}$ & Bombieri-De Giorgi-Giusti Bernstein Counterexample (Simons, 1968; Bombieri, De Giorgi & Giusti, 1969).
//!
//! ### 1. Second Variation Stability of the Clifford Cone $C(S^{m-1} \times S^{m-1}) \subset \mathbb{R}^{2m}$
//! On the cone $C^n \subset \mathbb{R}^{n+1}$ over a minimal hypersurface $\Sigma^{n-1} \subset S^n$ with $r^2 |A|^2 = n - 1$ (the Simons cone $C(S^p \times S^p)$, $n = 2p + 1$),
//! the Jacobi stability inequality $\int (|\nabla \varphi|^2 - |A|^2 \varphi^2) \ge 0$ holds iff the Hardy constant $(n - 2)^2 / 4 \ge n - 1$, i.e. $n \ge 4 + 2\sqrt{2}$, so $n \ge 7$ (ambient dimension $n + 1 \ge 8$).

use crate::error::{Error, Result};

/// Simons Minimal Cone Stability & Bernstein Theorem Dimension Engine.
#[derive(Debug, Clone, Copy)]
pub struct SimonsMinimalConeBernstein {
    /// Ambient Euclidean dimension $N = n + 1 \ge 3$ for minimal hypersurface $\Sigma^n \subset \mathbb{R}^{n+1}$.
    pub ambient_dim_n_plus_1: usize,
}

impl Default for SimonsMinimalConeBernstein {
    fn default() -> Self {
        Self {
            ambient_dim_n_plus_1: 8,
        }
    }
}

impl SimonsMinimalConeBernstein {
    /// Creates a new Simons Minimal Cone & Bernstein engine.
    pub fn new(ambient_dim_n_plus_1: usize) -> Result<Self> {
        if ambient_dim_n_plus_1 < 3 {
            return Err(Error::Msg("ambient_dim_n_plus_1 must be >= 3".to_string()));
        }
        Ok(Self {
            ambient_dim_n_plus_1,
        })
    }

    /// Evaluates the Jacobi-Hardy stability gap $\frac{(n - 2)^2}{4} - (n - 1)$ for a minimal cone $C^{n} \subset \mathbb{R}^{n+1}$ ($n = N - 1$), returning `(gap, is_stable_cone_possible, bernstein_holds)`.
    pub fn jacobi_hardy_gap_and_bernstein_status(&self) -> (f64, bool, bool) {
        let n = (self.ambient_dim_n_plus_1 - 1) as f64;
        // Hardy constant on R^n is (n - 2)^2 / 4; Simons cone over S^p x S^p (n = 2p + 1) has r^2 |A|^2 = n - 1 = 2p
        let gap = 0.25 * (n - 2.0).powi(2) - (n - 1.0);
        let stable_singular_cone = self.ambient_dim_n_plus_1 >= 8;
        let bernstein_theorem_holds = self.ambient_dim_n_plus_1 <= 8;
        (gap, stable_singular_cone, bernstein_theorem_holds)
    }
}
