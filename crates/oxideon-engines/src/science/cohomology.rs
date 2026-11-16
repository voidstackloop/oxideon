//! Algebraic Topology: Simplicial Cohomology, Hodge Laplacians & Helmholtz-Hodge Decomposition.
//!
//! Provides combinatorial Hodge theory on simplicial complexes:
//! - Boundary operators $\partial_k: C_k \to C_{k-1}$ satisfying $\partial_{k-1} \circ \partial_k = 0$.
//! - Coboundary operators $d^k = \partial_{k+1}^T: C^k \to C^{k+1}$ satisfying $d^{k+1} \circ d^k = 0$.
//! - Combinatorial $k$-th Hodge Laplacian:
//!   $$\Delta_k = d^{k-1} (d^{k-1})^T + (d^k)^T d^k = \partial_k^T \partial_k + \partial_{k+1} \partial_{k+1}^T$$
//! - Harmonic cochains $\ker(\Delta_k) \cong H^k(K; \mathbb{R})$ (Simplicial Hodge Theorem).
//! - Hodge-Helmholtz cochain decomposition of any $k$-cochain $\omega \in C^k$:
//!   $$\omega = d^{k-1} \alpha + \omega_{\text{harmonic}} + (d^k)^T \beta$$

use crate::error::{Error, Result};
use crate::tensor::Tensor;

/// Oriented simplex represented as a sorted vector of non-negative vertex IDs.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct OrientedSimplex {
    pub vertices: Vec<usize>,
}

impl OrientedSimplex {
    /// Creates a new oriented simplex with vertices sorted.
    pub fn new(mut vertices: Vec<usize>) -> Self {
        vertices.sort_unstable();
        Self { vertices }
    }

    /// Dimension of the simplex ($k = |V| - 1$).
    pub fn dim(&self) -> usize {
        self.vertices.len().saturating_sub(1)
    }

    /// Computes the faces with alternating orientations $(-1)^i [v_0, \dots, \hat{v}_i, \dots, v_k]$.
    pub fn faces(&self) -> Vec<(OrientedSimplex, f64)> {
        let n = self.vertices.len();
        if n <= 1 {
            return Vec::new();
        }
        let mut result = Vec::with_capacity(n);
        for i in 0..n {
            let mut face_v = self.vertices.clone();
            face_v.remove(i);
            let sign = if i % 2 == 0 { 1.0 } else { -1.0 };
            result.push((OrientedSimplex::new(face_v), sign));
        }
        result
    }
}

/// Simplicial complex with oriented cells up to maximum dimension.
#[derive(Debug, Clone)]
pub struct SimplicialComplex {
    /// Simplices grouped by dimension: `simplices[k]` contains $k$-simplices.
    pub simplices: Vec<Vec<OrientedSimplex>>,
}

impl SimplicialComplex {
    /// Creates a simplicial complex from a collection of maximal simplices.
    pub fn from_simplices(maximal: &[OrientedSimplex]) -> Self {
        let max_dim = maximal.iter().map(|s| s.dim()).max().unwrap_or(0);
        let mut grouped: Vec<std::collections::BTreeSet<OrientedSimplex>> =
            vec![std::collections::BTreeSet::new(); max_dim + 1];

        // Closure under subset / face operations
        for s in maximal {
            Self::add_closure(s, &mut grouped);
        }

        let simplices: Vec<Vec<OrientedSimplex>> = grouped
            .into_iter()
            .map(|set| set.into_iter().collect())
            .collect();

        Self { simplices }
    }

    fn add_closure(
        s: &OrientedSimplex,
        grouped: &mut [std::collections::BTreeSet<OrientedSimplex>],
    ) {
        let k = s.dim();
        if grouped[k].insert(s.clone()) && k > 0 {
            for (face, _) in s.faces() {
                Self::add_closure(&face, grouped);
            }
        }
    }

    /// Returns the number of $k$-simplices (rank of $C_k$).
    pub fn num_simplices(&self, k: usize) -> usize {
        self.simplices.get(k).map(|v| v.len()).unwrap_or(0)
    }

    /// Computes the boundary operator matrix $\partial_k \in \mathbb{R}^{|C_{k-1}| \times |C_k|}$.
    pub fn boundary_matrix(&self, k: usize) -> Result<Tensor> {
        if k == 0 || k >= self.simplices.len() {
            return Err(Error::Msg(format!("Invalid boundary dimension k={k}")));
        }

        let c_k = &self.simplices[k];
        let c_prev = &self.simplices[k - 1];
        let rows = c_prev.len();
        let cols = c_k.len();

        let mut data = vec![0.0f64; rows * cols];

        for (j, simplex) in c_k.iter().enumerate() {
            for (face, sign) in simplex.faces() {
                if let Some(i) = c_prev.iter().position(|f| f == &face) {
                    data[i * cols + j] = sign;
                }
            }
        }

        Tensor::from_vec(data, &[rows, cols])
    }

    /// Computes the coboundary operator matrix $d^{k-1} = \partial_k^T \in \mathbb{R}^{|C_k| \times |C_{k-1}|}$.
    pub fn coboundary_matrix(&self, k_minus_1: usize) -> Result<Tensor> {
        let k = k_minus_1.checked_add(1).ok_or_else(|| {
            Error::Msg("coboundary_matrix: dimension overflows usize".to_string())
        })?;
        let b = self.boundary_matrix(k)?;
        b.t()
    }

    /// Computes the combinatorial $k$-th Hodge Laplacian:
    /// $$\Delta_k = \partial_k^T \partial_k + \partial_{k+1} \partial_{k+1}^T \in \mathbb{R}^{|C_k| \times |C_k|}$$
    pub fn hodge_laplacian(&self, k: usize) -> Result<Tensor> {
        let n_k = self.num_simplices(k);
        if n_k == 0 {
            return Err(Error::Msg(format!("No simplices of dimension k={k}")));
        }

        let mut delta_k = vec![0.0f64; n_k * n_k];

        // Down-Laplacian: \Delta_{k, \text{down}} = \partial_k^T \partial_k (for k > 0)
        if k > 0 {
            let b_k = self.boundary_matrix(k)?;
            let b_k_t = b_k.t()?;
            let down = b_k_t.matmul(&b_k)?;
            let down_vec = down.to_vec::<f64>()?;
            for (i, v) in down_vec.into_iter().enumerate() {
                delta_k[i] += v;
            }
        }

        // Up-Laplacian: \Delta_{k, \text{up}} = \partial_{k+1} \partial_{k+1}^T (for k < max_dim)
        if k + 1 < self.simplices.len() {
            let b_k_plus_1 = self.boundary_matrix(k + 1)?;
            let b_t = b_k_plus_1.t()?;
            let up = b_k_plus_1.matmul(&b_t)?;
            let up_vec = up.to_vec::<f64>()?;
            for (i, v) in up_vec.into_iter().enumerate() {
                delta_k[i] += v;
            }
        }

        Tensor::from_vec(delta_k, &[n_k, n_k])
    }

    /// Computes the $k$-th Betti number $\beta_k = \dim(H^k(K; \mathbb{R})) = \dim(\ker(\Delta_k))$.
    pub fn betti_number(&self, k: usize, tolerance: f64) -> Result<usize> {
        let delta = self.hodge_laplacian(k)?;
        let n = self.num_simplices(k);
        let mat = delta.to_vec::<f64>()?;

        // Gaussian elimination with full pivoting to find rank of Delta_k
        let mut a = mat;
        let mut rank = 0;
        let tol = tolerance.max(1e-10);

        for col in 0..n {
            // Find pivot
            let mut max_val = 0.0f64;
            let mut pivot_row = rank;
            for r in rank..n {
                let val = a[r * n + col].abs();
                if val > max_val {
                    max_val = val;
                    pivot_row = r;
                }
            }

            if max_val < tol {
                continue;
            }

            // Swap rows
            for c in 0..n {
                a.swap(rank * n + c, pivot_row * n + c);
            }

            let pivot = a[rank * n + col];
            for r in (rank + 1)..n {
                let factor = a[r * n + col] / pivot;
                for c in col..n {
                    a[r * n + c] -= factor * a[rank * n + c];
                }
            }

            rank += 1;
            if rank == n {
                break;
            }
        }

        // By Hodge theorem: dim(ker(Delta_k)) = n_k - rank(Delta_k)
        Ok(n.saturating_sub(rank))
    }

    /// Performs the Hodge-Helmholtz decomposition of a 1-cochain (flow on edges) $\omega \in C^1$:
    /// $$\omega = d^0 \alpha + \omega_{\text{harmonic}} + \partial_2 \beta$$
    /// where:
    /// - $d^0 \alpha$: exact / gradient component (curl-free).
    /// - $\partial_2 \beta$: coexact / curl component (divergence-free).
    /// - $\omega_{\text{harmonic}}$: harmonic component satisfying $\Delta_1 \omega_{\text{harmonic}} = 0$.
    pub fn helmholtz_hodge_decomposition(
        &self,
        omega: &Tensor,
        max_iters: usize,
        lr: f64,
    ) -> Result<(Tensor, Tensor, Tensor)> {
        let n1 = self.num_simplices(1);
        if omega.shape() != [n1] && omega.shape() != [n1, 1] {
            return Err(Error::Msg(format!(
                "Expected 1-cochain of shape [{n1}], got {:?}",
                omega.shape()
            )));
        }

        let flat_omega = omega.to_vec::<f64>()?;

        // If no triangles exist, up-Laplacian is zero
        let b1 = self.boundary_matrix(1)?; // [n0, n1]
        let d0 = b1.t()?; // [n1, n0]

        let delta1 = self.hodge_laplacian(1)?;
        let delta1_mat = delta1.to_vec::<f64>()?;

        // Solve for harmonic component via diffusion damping (heat flow):
        // \omega_{harm} = \lim_{t \to \infty} e^{-t \Delta_1} \omega
        let mut harm = flat_omega.clone();
        for _ in 0..max_iters {
            let mut d_harm = vec![0.0f64; n1];
            for i in 0..n1 {
                let mut sum = 0.0;
                for j in 0..n1 {
                    sum += delta1_mat[i * n1 + j] * harm[j];
                }
                d_harm[i] = sum;
            }
            for i in 0..n1 {
                harm[i] -= lr * d_harm[i];
            }
        }

        // Gradient / exact projection: d^0 (d^0^T d^0)^+ d^0^T \omega
        let d0_mat = d0.to_vec::<f64>()?;
        let n0 = self.num_simplices(0);
        let mut exact = vec![0.0f64; n1];

        if n0 > 0 {
            // Compute d0^T * omega = b1 * omega
            let mut div = vec![0.0f64; n0];
            let b1_mat = b1.to_vec::<f64>()?;
            for i in 0..n0 {
                for j in 0..n1 {
                    div[i] += b1_mat[i * n1 + j] * flat_omega[j];
                }
            }

            // Approximate potential \alpha via Jacobi relaxation on \Delta_0 = b1 * d0
            let delta0 = b1.matmul(&d0)?;
            let delta0_mat = delta0.to_vec::<f64>()?;
            let mut alpha = vec![0.0f64; n0];

            for _ in 0..max_iters {
                let mut res = vec![0.0f64; n0];
                for i in 0..n0 {
                    let mut sum = 0.0;
                    for j in 0..n0 {
                        sum += delta0_mat[i * n0 + j] * alpha[j];
                    }
                    res[i] = div[i] - sum;
                }
                for i in 0..n0 {
                    alpha[i] += lr * res[i];
                }
            }

            // exact = d0 * alpha
            for i in 0..n1 {
                for j in 0..n0 {
                    exact[i] += d0_mat[i * n0 + j] * alpha[j];
                }
            }
        }

        // Coexact / curl component: curl = \omega - \omega_{harm} - \omega_{exact}
        let mut coexact = vec![0.0f64; n1];
        for i in 0..n1 {
            coexact[i] = flat_omega[i] - harm[i] - exact[i];
        }

        let t_exact = Tensor::from_vec(exact, &[n1])?;
        let t_harm = Tensor::from_vec(harm, &[n1])?;
        let t_coexact = Tensor::from_vec(coexact, &[n1])?;

        Ok((t_exact, t_harm, t_coexact))
    }
}
