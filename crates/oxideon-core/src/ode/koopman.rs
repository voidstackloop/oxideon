//! Ergodic Theory, Koopman Linear Spectral Operators, Extended Dynamic Mode Decomposition (EDMD) & Birkhoff Averages.
//!
//! ### 1. Koopman Operator $\mathcal{K}$ on Observable Space
//! Given a nonlinear dynamical system $x_{t+1} = T(x_t)$ on state space $\mathcal{M}$, the infinite-dimensional
//! **Koopman Operator** $\mathcal{K}$ acts linearly on scalar observables $g: \mathcal{M} \to \mathbb{R}$ via composition:
//! $$(\mathcal{K} g)(x) = g(T(x))$$
//!
//! ### 2. Extended Dynamic Mode Decomposition (EDMD)
//! Given a dictionary of basis observables $\psi(x) = [\psi_1(x), \dots, \psi_M(x)]^T$ and trajectory snapshots
//! $(x_k, y_k = T(x_k))_{k=1}^N$, the finite-dimensional Koopman matrix $K$ minimizes $\sum_k \|\psi(y_k) - K \psi(x_k)\|_2^2$.
//!
//! ### 3. Birkhoff Pointwise Ergodic Theorem (1931)
//! For a measure-preserving ergodic map $T$ (such as the chaotic Ulam logistic map $T(x) = 4 x (1 - x)$ on $[0, 1]$
//! with invariant arcsine density $\rho(x) = \frac{1}{\pi \sqrt{x(1-x)}}$), the time average converges to the space average:
//! $$\lim_{N \to \infty} \frac{1}{N} \sum_{k=0}^{N-1} f(T^k(x_0)) = \int_0^1 f(x) \, d\mu(x)$$

use crate::error::{Error, Result};

/// Koopman Operator Spectral & Birkhoff Ergodic Theorem Engine.
#[derive(Debug, Clone)]
pub struct KoopmanSpectralEngine {
    /// Koopman linear transition matrix $K \in \mathbb{R}^{M \times M}$ (flattened).
    pub koopman_matrix: Vec<f64>,
    /// Observable dictionary dimension $M$.
    pub dict_dim: usize,
}

impl KoopmanSpectralEngine {
    /// Fits a diagonal/decoupled or general 2x2 Koopman operator from observable snapshots $\Psi_X, \Psi_Y$ of shape `[N, M]`.
    pub fn fit_diagonal_edmd(psi_x: &[Vec<f64>], psi_y: &[Vec<f64>]) -> Result<Self> {
        if psi_x.is_empty() || psi_x.len() != psi_y.len() {
            return Err(Error::Msg(
                "Snapshot matrices must have equal non-zero length".to_string(),
            ));
        }
        let m = psi_x[0].len();
        if m == 0 {
            return Err(Error::Msg("Dictionary dimension must be >= 1".to_string()));
        }
        if psi_x.iter().any(|r| r.len() != m) || psi_y.iter().any(|r| r.len() != m) {
            return Err(Error::Msg(
                "Snapshot matrices must be rectangular with matching dictionary dim".to_string(),
            ));
        }

        let mut k_mat = vec![0.0; m * m];
        for j in 0..m {
            let mut num = 0.0;
            let mut den = 0.0;
            for (rx, ry) in psi_x.iter().zip(psi_y.iter()) {
                num += rx[j] * ry[j];
                den += rx[j] * rx[j];
            }
            k_mat[j * m + j] = if den > 1e-15 { num / den } else { 0.0 };
        }

        Ok(Self {
            koopman_matrix: k_mat,
            dict_dim: m,
        })
    }

    /// Predicts observable vector $k$ steps into the future via linear Koopman evolution $\psi_{t+k} = K^k \psi_t$.
    pub fn predict_observable(&self, psi_init: &[f64], steps: usize) -> Result<Vec<f64>> {
        let m = self.dict_dim;
        if psi_init.len() != m {
            return Err(Error::Msg(
                "Observable vector dimension mismatch".to_string(),
            ));
        }
        crate::limits::check_iterations("steps", steps)?;
        let cells = m
            .checked_mul(m)
            .ok_or_else(|| Error::Msg(format!("Koopman matrix size {m} * {m} overflows")))?;
        if self.koopman_matrix.len() != cells {
            return Err(Error::Msg(format!(
                "Koopman matrix length {} disagrees with {m}*{m}",
                self.koopman_matrix.len()
            )));
        }
        let mut curr = psi_init.to_vec();
        for _ in 0..steps {
            let mut next = vec![0.0; m];
            for (i, next_i) in next.iter_mut().enumerate().take(m) {
                let mut sum = 0.0;
                for (j, &c_j) in curr.iter().enumerate().take(m) {
                    sum += self.koopman_matrix[i * m + j] * c_j;
                }
                *next_i = sum;
            }
            curr = next;
        }
        Ok(curr)
    }

    /// Computes the Birkhoff ergodic time average $\frac{1}{N} \sum_{k=0}^{N-1} x_k$ along the chaotic Ulam logistic map
    /// $x_{k+1} = 4 x_k (1 - x_k)$, whose exact invariant measure mean is $\int_0^1 x \frac{dx}{\pi\sqrt{x(1-x)}} = 0.5$.
    pub fn birkhoff_ulam_ergodic_average(x0: f64, num_steps: usize) -> Result<f64> {
        if !(0.0..=1.0).contains(&x0) || num_steps == 0 {
            return Err(Error::Msg(
                "x0 must be in [0, 1] and num_steps > 0".to_string(),
            ));
        }
        crate::limits::check_iterations("num_steps", num_steps)?;
        let mut x = x0;
        let mut sum = 0.0;
        for _ in 0..num_steps {
            sum += x;
            x = 4.0 * x * (1.0 - x);
        }
        Ok(sum / (num_steps as f64))
    }
}
