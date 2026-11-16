//! Macdonald Symmetric Polynomials, Haiman Diagonal Harmonics & $(q, t)$-Catalan Numbers (Macdonald, 1988; Haiman, 2001).
//!
//! ### 1. Carlitz-Riordan $q$-Catalan & $(q, t)$-Catalan Specialization
//! The bigraded Hilbert series of the alternating component of diagonal harmonics $\mathrm{DH}_n$ is the $(q, t)$-Catalan polynomial $C_n(q, t)$,
//! which at $t = 1$ specializes to the area-enumerating Carlitz $q$-Catalan recurrence $C_n(q) = \sum_{k=0}^{n-1} q^k C_k(q) C_{n-1-k}(q)$.

use crate::error::{Error, Result};

/// Macdonald-Haiman Diagonal Harmonics & $q$-Catalan Engine.
#[derive(Debug, Clone, Copy)]
pub struct MacdonaldSymmetricCatalanEngine {
    /// Order $n \ge 1$ of the Catalan / Diagonal Harmonics space $\mathrm{DH}_n$.
    pub order_n: usize,
}

impl Default for MacdonaldSymmetricCatalanEngine {
    fn default() -> Self {
        Self { order_n: 4 }
    }
}

impl MacdonaldSymmetricCatalanEngine {
    /// Creates a new Macdonald Symmetric Catalan engine.
    pub fn new(order_n: usize) -> Result<Self> {
        if order_n == 0 || order_n > 15 {
            return Err(Error::Msg("order_n must be in 1..=15".to_string()));
        }
        Ok(Self { order_n })
    }

    /// Evaluates the Carlitz $q$-Catalan value $C_n(q)$ and Haiman's total dimension $(n + 1)^{n - 1}$ of the diagonal harmonics module $\mathrm{DH}_n$.
    pub fn q_catalan_and_haiman_dimension(&self, q: f64) -> (f64, u64) {
        let mut c = vec![0.0; self.order_n + 1];
        c[0] = 1.0;
        for m in 1..=self.order_n {
            let mut sum = 0.0;
            for k in 0..m {
                sum += q.powi(k as i32) * c[k] * c[m - 1 - k];
            }
            c[m] = sum;
        }
        let haiman_dim = ((self.order_n + 1) as u64).pow((self.order_n - 1) as u32);
        (c[self.order_n], haiman_dim)
    }
}
