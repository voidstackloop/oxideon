//! Tropical Spectral Theory, Karp's Minimum Cycle Mean Eigenvalue & Legendre-Fenchel Duals.
//!
//! ### 1. Tropical Eigenvalue (Minimum Cycle Mean)
//! In the min-plus tropical semiring $(\mathbb{R} \cup \{+\infty\}, \oplus = \min, \otimes = +)$,
//! an $n \times n$ matrix $A$ satisfies the tropical eigenvector equation $A \otimes v = \lambda \otimes v$, i.e.:
//! $$\min_{1 \le j \le n} (A_{ij} + v_j) = \lambda + v_i$$
//! For any strongly connected weighted directed graph, the unique tropical eigenvalue $\lambda(A)$ is
//! the **Minimum Cycle Mean**, computed exactly in $O(n^3)$ time by **Karp's Theorem (1978)**:
//! $$\lambda(A) = \min_{1 \le i \le n} \max_{0 \le k < n} \frac{d_n(i) - d_k(i)}{n - k}$$
//! where $d_k(i)$ is the minimum weight of a $k$-edge path reaching vertex $i$.
//!
//! ### 2. Discrete Legendre-Fenchel Transform (Newton Polytope Support Function)
//! Given a tropical polynomial support $(x_m, c_m)$, its Legendre-Fenchel convex dual at slope $p$ is:
//! $$f^*(p) = \max_m (p \cdot x_m - c_m)$$

use crate::error::{Error, Result};

/// Tropical Spectral Eigenvalue & Legendre-Fenchel Newton Polytope Engine.
#[derive(Debug, Clone)]
pub struct TropicalSpectralEngine {
    /// Dimension $n$ of the square tropical matrix.
    pub n: usize,
    /// Flattened $n \times n$ weight matrix $A_{ij}$.
    pub matrix: Vec<f64>,
}

impl TropicalSpectralEngine {
    /// Constructs a tropical spectral engine for an $n \times n$ matrix.
    pub fn new(n: usize, matrix: Vec<f64>) -> Result<Self> {
        let need = n.checked_mul(n);
        if n == 0 || need != Some(matrix.len()) {
            return Err(Error::Msg("Invalid tropical matrix dimensions".to_string()));
        }
        Ok(Self { n, matrix })
    }

    fn check_square(&self, op: &'static str) -> Result<usize> {
        let n = self.n;
        match n.checked_mul(n) {
            Some(need) if n > 0 && self.matrix.len() == need => Ok(n),
            _ => Err(Error::Msg(format!(
                "{op}: need a square {n}x{n} matrix, got {} entries",
                self.matrix.len()
            ))),
        }
    }

    /// Computes the exact unique tropical eigenvalue $\lambda(A)$ (minimum cycle mean) via Karp's Algorithm.
    pub fn karp_minimum_cycle_mean(&self) -> Result<f64> {
        let n = self.check_square("karp_minimum_cycle_mean")?;
        let mut d = vec![vec![f64::INFINITY; n]; n + 1];
        d[0].fill(0.0);

        for k in 1..=n {
            for v in 0..n {
                let mut best = f64::INFINITY;
                for u in 0..n {
                    let w = self.matrix[u * n + v];
                    if d[k - 1][u].is_finite() && w.is_finite() {
                        best = best.min(d[k - 1][u] + w);
                    }
                }
                d[k][v] = best;
            }
        }

        let mut min_over_v = f64::INFINITY;
        for v in 0..n {
            if !d[n][v].is_finite() {
                continue;
            }
            let mut max_over_k = f64::NEG_INFINITY;
            for (k, d_k) in d.iter().enumerate().take(n) {
                if d_k[v].is_finite() {
                    let ratio = (d[n][v] - d_k[v]) / ((n - k) as f64);
                    max_over_k = max_over_k.max(ratio);
                }
            }
            min_over_v = min_over_v.min(max_over_k);
        }
        Ok(min_over_v)
    }

    /// Computes one min-plus matrix-vector multiplication $(A \otimes x)_i = \min_j (A_{ij} + x_j)$.
    pub fn min_plus_matvec(&self, x: &[f64]) -> Result<Vec<f64>> {
        let n = self.check_square("min_plus_matvec")?;
        if x.len() != n {
            return Err(Error::Msg(
                "Vector dimension mismatch in min_plus_matvec".to_string(),
            ));
        }
        let mut out = vec![f64::INFINITY; self.n];
        for (i, out_i) in out.iter_mut().enumerate().take(self.n) {
            let mut best = f64::INFINITY;
            for (j, &x_j) in x.iter().enumerate().take(self.n) {
                best = best.min(self.matrix[i * self.n + j] + x_j);
            }
            *out_i = best;
        }
        Ok(out)
    }

    /// Evaluates the discrete Legendre-Fenchel conjugate $f^*(p) = \max_m (p \cdot x_m - c_m)$.
    pub fn legendre_fenchel_conjugate(points: &[f64], coeffs: &[f64], slope_p: f64) -> Result<f64> {
        if points.is_empty() || points.len() != coeffs.len() {
            return Err(Error::Msg(
                "Points and coeffs must have equal non-zero length".to_string(),
            ));
        }
        let mut max_val = f64::NEG_INFINITY;
        for (&x, &c) in points.iter().zip(coeffs.iter()) {
            max_val = max_val.max(slope_p * x - c);
        }
        Ok(max_val)
    }
}
