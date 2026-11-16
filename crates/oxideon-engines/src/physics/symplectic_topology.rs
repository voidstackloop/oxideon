//! Symplectic Topology, Gromov's Non-Squeezing Theorem, Symplectic Capacities &
//! Hamiltonian Floer Action Functionals.
//!
//! ### 1. Standard Symplectic Structure & $Sp(2n, \mathbb{R})$
//! On phase space $\mathbb{R}^{2n}$ with coordinates $(q_1, p_1, \dots, q_n, p_n)$, the canonical symplectic
//! matrix is $J = \bigoplus_{k=1}^n \begin{pmatrix} 0 & 1 \\ -1 & 0 \end{pmatrix}$.
//! A matrix $M \in \mathbb{R}^{2n \times 2n}$ is symplectic ($M \in Sp(2n, \mathbb{R})$) iff:
//! $$M^T J M = J$$
//!
//! ### 2. Gromov's Non-Squeezing Theorem (1985) & Symplectic Capacities
//! Given a symplectic ellipsoid $E(R_1, \dots, R_n) = \left\{ \sum_{k=1}^n \frac{q_k^2 + p_k^2}{R_k^2} \le 1 \right\}$,
//! its **Gromov Width / Hofer-Zehnder Symplectic Capacity** is:
//! $$c_G(E) = \pi \min_{1 \le k \le n} R_k^2$$
//! Whereas Liouville phase-space volume is $\operatorname{Vol}(E) = \frac{\pi^n}{n!} \prod_{k=1}^n R_k^2$.
//! By **Gromov's Non-Squeezing Theorem**, a ball $B^{2n}(R)$ can be symplectically embedded into a symplectic
//! cylinder $Z^{2n}(r) = D^2(r) \times \mathbb{R}^{2n-2}$ if and only if $c_G(B^{2n}(R)) \le c_G(Z^{2n}(r))$, i.e. $R \le r$.
//!
//! ### 3. Hamiltonian Floer Action Functional
//! For a 1-periodic phase-space loop $\gamma: S^1 \to \mathbb{R}^{2n}$ and Hamiltonian $H(q, p) = \frac{1}{2}\sum_k \omega_k (q_k^2 + p_k^2)$:
//! $$\mathcal{A}_H(\gamma) = \oint_\gamma \mathbf{p} \cdot d\mathbf{q} - \int_0^1 H(\gamma(t)) \, dt$$
//! Critical points $\delta \mathcal{A}_H(\gamma) = 0$ are exactly the 1-periodic classical Hamiltonian orbits.

use crate::error::{Error, Result};
use std::f64::consts::PI;

/// Symplectic Topology, Capacity & Floer Action Engine on $2n$-dimensional phase space.
#[derive(Debug, Clone)]
pub struct SymplecticTopologyEngine {
    /// Number of canonical conjugate pairs $n$ (phase space dimension is $2n$).
    pub num_dof: usize,
}

/// Maximum degrees of freedom: dim $2n$ matrices are $O(n^3)$ work and
/// $O(n^2)$ memory, so unbounded grids would hang or abort before computing.
pub const MAX_SYMPLECTIC_DOF: usize = 1024;

impl SymplecticTopologyEngine {
    /// Constructs a symplectic topology engine for $n$ degrees of freedom ($\dim = 2n$).
    pub fn new(num_dof: usize) -> Result<Self> {
        if num_dof == 0 || num_dof > MAX_SYMPLECTIC_DOF {
            return Err(Error::Msg(format!(
                "num_dof must be in 1..={MAX_SYMPLECTIC_DOF}, got {num_dof}"
            )));
        }
        Ok(Self { num_dof })
    }

    /// Phase-space dimension $2n$, checked (`num_dof` is public, so methods
    /// cannot trust the constructor ran).
    fn dim(&self) -> Result<usize> {
        self.num_dof
            .checked_mul(2)
            .ok_or_else(|| Error::Msg(format!("symplectic dim 2*{} overflows", self.num_dof)))
    }

    /// Computes the Frobenius norm residual $\|M^T J M - J\|_F$ measuring departure from $Sp(2n, \mathbb{R})$.
    pub fn symplecticity_defect(&self, matrix: &[f64]) -> Result<f64> {
        let dim = self.dim()?;
        let dim_sq = dim
            .checked_mul(dim)
            .ok_or_else(|| Error::Msg(format!("symplectic matrix size {dim}^2 overflows")))?;
        if matrix.len() != dim_sq {
            return Err(Error::Msg(format!(
                "Matrix length {} != (2n)^2 = {}",
                matrix.len(),
                dim_sq
            )));
        }

        // Construct canonical J
        let mut j_mat = vec![0.0; dim_sq];
        for k in 0..self.num_dof {
            let iq = 2 * k;
            let ip = 2 * k + 1;
            j_mat[iq * dim + ip] = 1.0;
            j_mat[ip * dim + iq] = -1.0;
        }

        // Compute JM = J * M
        let mut jm = vec![0.0; dim_sq];
        for i in 0..dim {
            for j in 0..dim {
                let mut s = 0.0;
                for k in 0..dim {
                    s += j_mat[i * dim + k] * matrix[k * dim + j];
                }
                jm[i * dim + j] = s;
            }
        }

        // Compute M^T * JM - J
        let mut sq_err = 0.0;
        for i in 0..dim {
            for j in 0..dim {
                let mut s = 0.0;
                for k in 0..dim {
                    s += matrix[k * dim + i] * jm[k * dim + j];
                }
                let diff = s - j_mat[i * dim + j];
                sq_err += diff * diff;
            }
        }
        Ok(sq_err.sqrt())
    }

    /// Computes the Gromov width / Hofer-Zehnder symplectic capacity $c_G(E) = \pi \min_k R_k^2$
    /// of a symplectic ellipsoid with semi-axes $(R_1, \dots, R_n)$.
    pub fn ellipsoid_gromov_capacity(&self, radii: &[f64]) -> Result<f64> {
        if radii.len() != self.num_dof {
            return Err(Error::Msg("Radii length must equal num_dof".to_string()));
        }
        let mut min_r2 = f64::INFINITY;
        for &r in radii {
            if r <= 0.0 {
                return Err(Error::Msg("Ellipsoid radii must be > 0".to_string()));
            }
            min_r2 = min_r2.min(r * r);
        }
        Ok(PI * min_r2)
    }

    /// Computes the Liouville volume $\operatorname{Vol}(E) = \frac{\pi^n}{n!} \prod_{k=1}^n R_k^2$.
    pub fn ellipsoid_liouville_volume(&self, radii: &[f64]) -> Result<f64> {
        if radii.len() != self.num_dof {
            return Err(Error::Msg("Radii length must equal num_dof".to_string()));
        }
        let mut prod_r2 = 1.0;
        for &r in radii {
            if r <= 0.0 {
                return Err(Error::Msg("Ellipsoid radii must be > 0".to_string()));
            }
            prod_r2 *= r * r;
        }
        let n_fact: f64 = (1..=self.num_dof).map(|k| k as f64).product();
        let n_i32 = i32::try_from(self.num_dof)
            .map_err(|_| Error::Msg(format!("num_dof {} exceeds i32 range", self.num_dof)))?;
        Ok(PI.powi(n_i32) * prod_r2 / n_fact)
    }

    /// Verifies Gromov's Non-Squeezing Theorem for embedding a ball $B^{2n}(R)$ into cylinder $Z^{2n}(r)$:
    /// returns `(can_embed_symplectically, ball_capacity, cylinder_capacity)`.
    pub fn verify_gromov_non_squeezing(
        &self,
        ball_radius_r: f64,
        cylinder_radius_r: f64,
    ) -> Result<(bool, f64, f64)> {
        if ball_radius_r <= 0.0 || cylinder_radius_r <= 0.0 {
            return Err(Error::Msg("Radii must be > 0".to_string()));
        }
        let c_ball = PI * ball_radius_r * ball_radius_r;
        let c_cyl = PI * cylinder_radius_r * cylinder_radius_r;
        Ok((c_ball <= c_cyl + 1e-12, c_ball, c_cyl))
    }

    /// Evaluates the discrete Hamiltonian Floer action functional on a closed loop of $N$ phase-space
    /// points `loop_points` (each of length $2n$) under harmonic frequencies `omegas` (length $n$):
    ///
    /// $$\mathcal{A}_H(\gamma) = \oint_\gamma \mathbf{p} \cdot d\mathbf{q} - \int_0^1 H(\gamma(t)) \, dt$$
    pub fn floer_action_functional(&self, loop_points: &[Vec<f64>], omegas: &[f64]) -> Result<f64> {
        let m = loop_points.len();
        let dim = self.dim()?;
        if m < 3 {
            return Err(Error::Msg(
                "Loop requires at least 3 discretized points".to_string(),
            ));
        }
        if omegas.len() != self.num_dof {
            return Err(Error::Msg("omegas length must equal num_dof".to_string()));
        }
        for pt in loop_points {
            if pt.len() != dim {
                return Err(Error::Msg("Loop point dimension mismatch".to_string()));
            }
        }

        let dt = 1.0 / (m as f64);
        let mut symplectic_area = 0.0;
        let mut hamiltonian_integral = 0.0;

        for (step, curr) in loop_points.iter().enumerate().take(m) {
            let next = &loop_points[(step + 1) % m];

            for (k, &omega_k) in omegas.iter().enumerate().take(self.num_dof) {
                let q_curr = curr[2 * k];
                let p_curr = curr[2 * k + 1];
                let q_next = next[2 * k];
                let p_next = next[2 * k + 1];

                // Midpoint rule for p * dq
                let p_mid = 0.5 * (p_curr + p_next);
                let dq = q_next - q_curr;
                symplectic_area += p_mid * dq;

                // Harmonic Hamiltonian H_k = 0.5 * omega_k * (q^2 + p^2)
                hamiltonian_integral += 0.5 * omega_k * (q_curr * q_curr + p_curr * p_curr) * dt;
            }
        }

        Ok(symplectic_area - hamiltonian_integral)
    }
}
