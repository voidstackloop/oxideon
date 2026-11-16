//! Derived Algebraic Geometry, Differential Graded Lie Algebras (DGLA) & Maurer-Cartan Deformation Functor.
//!
//! ### 1. Differential Graded Lie Algebras (DGLA)
//! A Differential Graded Lie Algebra $(L^\bullet = \bigoplus_i L^i, \; d, \; [\cdot, \cdot])$ consists of a $\mathbb{Z}$-graded
//! vector space equipped with a degree-$+1$ differential $d: L^i \to L^{i+1}$ ($d^2 = 0$) and a graded Lie bracket
//! satisfying graded antisymmetry and the graded Leibniz rule:
//! $$d[x, y] = [dx, y] + (-1)^{|x|} [x, dy]$$
//!
//! ### 2. Maurer-Cartan Equation & Deligne-Goldman-Millson Deformation Functor
//! In Kontsevich's & Lurie's Derived Deformation Theory, formal deformations are governed by degree-$1$ elements
//! $\alpha \in L^1$ satisfying the **Maurer-Cartan Equation**:
//! $$\mathcal{MC}(\alpha) := d\alpha + \frac{1}{2} [\alpha, \alpha] = 0 \in L^2$$
//! For a 1-form connection $\alpha$ with nilpotent structure constants, $\mathcal{MC}(\alpha)$ is the non-abelian
//! curvature 2-form $F_\alpha$, and flat connections satisfy $\|\mathcal{MC}(\alpha)\|_2 = 0$.

use crate::error::{Error, Result};

/// Differential Graded Lie Algebra (DGLA) & Maurer-Cartan Deformation Engine.
#[derive(Debug, Clone)]
pub struct DglaMaurerCartanEngine {
    /// Dimension $n$ of the degree-1 and degree-2 graded spaces $L^1, L^2$.
    pub dim: usize,
    /// Differential matrix $d_1: L^1 \to L^2$ of shape $n \times n$ (flattened).
    pub d1_matrix: Vec<f64>,
    /// Symmetric bilinear bracket tensor $B^k_{ij}$ mapping $L^1 \otimes L^1 \to L^2$ (remembering $[x, y] = [y, x]$ for odd degree $|x|=|y|=1$).
    pub bracket_tensor: Vec<f64>,
}

impl DglaMaurerCartanEngine {
    /// Creates a new DGLA Maurer-Cartan engine of dimension `dim`.
    pub fn new(dim: usize, d1_matrix: Vec<f64>, bracket_tensor: Vec<f64>) -> Result<Self> {
        if dim == 0 {
            return Err(Error::Msg(
                "DglaMaurerCartanEngine requires dim >= 1".to_string(),
            ));
        }
        let sq = dim
            .checked_mul(dim)
            .ok_or_else(|| Error::Msg(format!("dim^2 overflows usize for dim={dim}")))?;
        let cube = sq
            .checked_mul(dim)
            .ok_or_else(|| Error::Msg(format!("dim^3 overflows usize for dim={dim}")))?;
        if d1_matrix.len() != sq || bracket_tensor.len() != cube {
            return Err(Error::Msg("Invalid DGLA tensor dimensions".to_string()));
        }
        Ok(Self {
            dim,
            d1_matrix,
            bracket_tensor,
        })
    }

    /// Evaluates the Maurer-Cartan curvature vector $\mathcal{MC}(\alpha) = d\alpha + \frac{1}{2}[\alpha, \alpha] \in L^2$.
    pub fn maurer_cartan_curvature(&self, alpha: &[f64]) -> Result<Vec<f64>> {
        let n = self.dim;
        if alpha.len() != n {
            return Err(Error::Msg(
                "Degree-1 element alpha dimension mismatch".to_string(),
            ));
        }
        let mut curv = vec![0.0; n];
        for (k, curv_k) in curv.iter_mut().enumerate().take(n) {
            let mut d_alpha_k = 0.0;
            let mut bracket_k = 0.0;
            for (i, &a_i) in alpha.iter().enumerate().take(n) {
                d_alpha_k += self.d1_matrix[k * n + i] * a_i;
                for (j, &a_j) in alpha.iter().enumerate().take(n) {
                    bracket_k += self.bracket_tensor[k * n * n + i * n + j] * a_i * a_j;
                }
            }
            *curv_k = d_alpha_k + 0.5 * bracket_k;
        }
        Ok(curv)
    }

    /// Computes the Euclidean norm $\|d\alpha + \frac{1}{2}[\alpha, \alpha]\|_2$ of the Maurer-Cartan residual.
    pub fn maurer_cartan_residual_norm(&self, alpha: &[f64]) -> Result<f64> {
        let curv = self.maurer_cartan_curvature(alpha)?;
        Ok(curv.iter().map(|x| x * x).sum::<f64>().sqrt())
    }

    /// Performs $K$ steps of gradient descent on the Maurer-Cartan deformation action $\frac{1}{2}\|\mathcal{MC}(\alpha)\|_2^2$.
    pub fn solve_maurer_cartan_flow(
        &self,
        alpha_init: &[f64],
        lr: f64,
        steps: usize,
    ) -> Result<Vec<f64>> {
        let n = self.dim;
        if alpha_init.len() != n {
            return Err(Error::Msg("alpha_init dimension mismatch".to_string()));
        }
        let mut alpha = alpha_init.to_vec();
        let eps = 1e-7;
        for _ in 0..steps {
            let f0 = 0.5 * self.maurer_cartan_residual_norm(&alpha)?.powi(2);
            if f0 < 1e-24 {
                break;
            }
            let mut grad = vec![0.0; n];
            for (i, g_i) in grad.iter_mut().enumerate().take(n) {
                alpha[i] += eps;
                let fp = 0.5 * self.maurer_cartan_residual_norm(&alpha)?.powi(2);
                alpha[i] -= 2.0 * eps;
                let fm = 0.5 * self.maurer_cartan_residual_norm(&alpha)?.powi(2);
                alpha[i] += eps;
                *g_i = (fp - fm) / (2.0 * eps);
            }
            for (a_i, &g_i) in alpha.iter_mut().zip(grad.iter()) {
                *a_i -= lr * g_i;
            }
        }
        Ok(alpha)
    }
}
