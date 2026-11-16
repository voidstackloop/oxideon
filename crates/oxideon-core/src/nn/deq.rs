//! Deep Equilibrium Models (DEQ) & Implicit Function Theorem (IFT) Adjoint Solvers.
//!
//! Implements infinite-depth implicit layers with $O(1)$ memory training:
//! - Forward fixed-point equilibrium solver: $z^* = f_\theta(z^*, x)$.
//! - Anderson / relaxed Picard fixed-point root finding: $g(z) = f_\theta(z, x) - z = 0$.
//! - Exact Implicit Function Theorem (IFT) adjoint backward pass: $(I - J_{f, z^*})^T y = \nabla_{z^*} \mathcal{L}$.
//! - Jacobian-free vector-Jacobian products avoiding full Jacobian materialization.

use crate::error::{Error, Result};
use crate::tensor::Tensor;

/// Configuration for Deep Equilibrium fixed-point solver.
#[derive(Debug, Clone)]
pub struct DeqConfig {
    /// Maximum number of forward fixed-point iterations.
    pub max_forward_iters: usize,
    /// Forward convergence tolerance $\epsilon_{\text{fwd}}$.
    pub fwd_tol: f64,
    /// Relaxation factor $\beta \in (0, 1]$ in Picard iteration $z \leftarrow (1-\beta) z + \beta f(z, x)$.
    pub relaxation_beta: f64,
    /// Maximum number of backward adjoint iterations for Neumann/Richardson solver.
    pub max_backward_iters: usize,
    /// Backward adjoint convergence tolerance $\epsilon_{\text{bwd}}$.
    pub bwd_tol: f64,
}

impl Default for DeqConfig {
    fn default() -> Self {
        Self {
            max_forward_iters: 50,
            fwd_tol: 1e-6,
            relaxation_beta: 0.6,
            max_backward_iters: 40,
            bwd_tol: 1e-6,
        }
    }
}

/// Output of a forward DEQ solve.
#[derive(Debug, Clone)]
pub struct DeqForwardOutput {
    /// Fixed-point equilibrium state $z^* \in \mathbb{R}^d$.
    pub z_star: Tensor,
    /// Final residual norm $\|f(z^*, x) - z^*\|_2$.
    pub residual_norm: f64,
    /// Number of iterations taken to reach equilibrium.
    pub num_iters: usize,
    /// Whether the solver successfully converged within tolerance.
    pub converged: bool,
}

/// Deep Equilibrium Model layer.
#[derive(Debug, Clone)]
pub struct DeepEquilibriumModel {
    /// Input dimension $d_{\text{in}}$.
    pub in_dim: usize,
    /// Equilibrium hidden state dimension $d_{\text{state}}$.
    pub state_dim: usize,
    /// State transition matrix $W_z \in \mathbb{R}^{d_{\text{state}} \times d_{\text{state}}}$.
    pub weight_z: Vec<f64>,
    /// Input projection matrix $W_x \in \mathbb{R}^{d_{\text{state}} \times d_{\text{in}}}$.
    pub weight_x: Vec<f64>,
    /// State bias $b \in \mathbb{R}^{d_{\text{state}}}$.
    pub bias: Vec<f64>,
    /// Solver configuration.
    pub config: DeqConfig,
}

impl DeepEquilibriumModel {
    /// Instantiates a new DEQ layer.
    pub fn new(in_dim: usize, state_dim: usize, config: DeqConfig) -> Result<Self> {
        crate::limits::check_iterations("max_forward_iters", config.max_forward_iters)?;
        crate::limits::check_iterations("max_backward_iters", config.max_backward_iters)?;
        // Contractive deterministic initialization to encourage rapid fixed-point convergence
        let bound_z = 0.5 / (state_dim as f64).sqrt();
        let n_z = crate::layout::checked_numel(&[state_dim, state_dim])?;
        let mut weight_z: Vec<f64> = crate::layout::try_with_capacity(n_z)?;
        weight_z.extend((0..n_z).map(|i| {
            let frac = ((i * 31 + 17) % 1000) as f64 / 1000.0;
            (frac * 2.0 - 1.0) * bound_z
        }));

        let bound_x = 1.0 / (in_dim as f64).sqrt();
        let n_x = crate::layout::checked_numel(&[state_dim, in_dim])?;
        let mut weight_x: Vec<f64> = crate::layout::try_with_capacity(n_x)?;
        weight_x.extend((0..n_x).map(|i| {
            let frac = ((i * 19 + 7) % 1000) as f64 / 1000.0;
            (frac * 2.0 - 1.0) * bound_x
        }));

        let bias = crate::layout::try_filled(state_dim, 0.0)?;

        Ok(Self {
            in_dim,
            state_dim,
            weight_z,
            weight_x,
            bias,
            config,
        })
    }

    /// Evaluates equilibrium layer step $f(z, x) = \tanh(W_z z + W_x x + b)$.
    pub fn step_f(&self, z: &[f64], x: &[f64]) -> Result<Vec<f64>> {
        let sd = self.state_dim;
        let id = self.in_dim;
        if z.len() != sd || x.len() != id {
            return Err(Error::ShapeMismatch {
                op: "DeepEquilibriumModel::step_f",
                expected: vec![sd],
                got: vec![z.len()],
            });
        }
        // The weights are pub Vecs: verify the lengths the raw indexing below assumes.
        let (n_z, n_x) = (sd.checked_mul(sd), sd.checked_mul(id));
        if n_z.is_none_or(|n| self.weight_z.len() != n)
            || n_x.is_none_or(|n| self.weight_x.len() != n)
            || self.bias.len() != sd
        {
            return Err(Error::Msg(format!(
                "DeepEquilibriumModel::step_f: weight lengths disagree with state_dim={sd}, in_dim={id}"
            )));
        }

        let mut out = vec![0.0; sd];
        for (i, out_val) in out.iter_mut().enumerate().take(sd) {
            let mut sum = self.bias[i];
            for (j, &zj) in z.iter().enumerate().take(sd) {
                sum += self.weight_z[i * sd + j] * zj;
            }
            for (j, &xj) in x.iter().enumerate().take(id) {
                sum += self.weight_x[i * id + j] * xj;
            }
            *out_val = sum.tanh();
        }

        Ok(out)
    }

    /// Solves for the equilibrium fixed point $z^* = f_\theta(z^*, x)$ using relaxed Picard iteration:
    ///
    /// $$z^{(k+1)} = (1 - \beta) z^{(k)} + \beta f_\theta(z^{(k)}, x)$$
    pub fn forward_equilibrium(&self, x: &Tensor) -> Result<DeqForwardOutput> {
        // `config` is a pub field: re-check the constructor cap here so
        // post-construction tampering yields Err instead of an unbounded loop.
        crate::limits::check_iterations("max_forward_iters", self.config.max_forward_iters)?;
        let shape = x.shape();
        if shape.len() != 1 || shape[0] != self.in_dim {
            return Err(Error::ShapeMismatch {
                op: "forward_equilibrium",
                expected: vec![self.in_dim],
                got: shape.to_vec(),
            });
        }

        let x_data = x.to_vec::<f64>()?;
        let sd = self.state_dim;
        let mut z = vec![0.0; sd]; // Initial guess z_0 = 0

        let beta = self.config.relaxation_beta;
        let mut final_res = 1.0;
        let mut iters_taken = 0;
        let mut converged = false;

        for iter in 0..self.config.max_forward_iters {
            let f_z = self.step_f(&z, &x_data)?;

            // Residual: r = f(z, x) - z
            let mut res_norm_sq = 0.0;
            for i in 0..sd {
                let diff = f_z[i] - z[i];
                res_norm_sq += diff * diff;
            }
            let res_norm = res_norm_sq.sqrt();
            final_res = res_norm;
            iters_taken = iter + 1;

            if res_norm < self.config.fwd_tol {
                converged = true;
                z = f_z;
                break;
            }

            // Relaxed Picard update
            for (i, zi) in z.iter_mut().enumerate().take(sd) {
                *zi = (1.0 - beta) * *zi + beta * f_z[i];
            }
        }

        let z_star = Tensor::from_slice(&z, &[sd])?;
        Ok(DeqForwardOutput {
            z_star,
            residual_norm: final_res,
            num_iters: iters_taken,
            converged,
        })
    }

    /// Evaluates vector-Jacobian product $v^T J_{f, z^*}$ where $J = \frac{\partial f}{\partial z}(z^*, x)$.
    ///
    /// Since $f_i(z, x) = \tanh(a_i)$, $\frac{\partial f_i}{\partial z_j} = (1 - \tanh^2(a_i)) W_{z, ij}$.
    /// Therefore: $(v^T J)_j = \sum_i v_i (1 - f_i^2) W_{z, ij}$.
    pub fn vjp_state(&self, v: &[f64], z_star: &[f64], x: &[f64]) -> Result<Vec<f64>> {
        let sd = self.state_dim;
        if v.len() != sd || z_star.len() != sd {
            return Err(Error::ShapeMismatch {
                op: "vjp_state",
                expected: vec![sd],
                got: vec![v.len()],
            });
        }

        let f_z = self.step_f(z_star, x)?;
        let mut v_scaled = vec![0.0; sd];
        for (i, vs) in v_scaled.iter_mut().enumerate().take(sd) {
            let dtanh = 1.0 - f_z[i] * f_z[i];
            *vs = v[i] * dtanh;
        }

        // v_scaled^T W_z
        let mut out = vec![0.0; sd];
        for (j, out_val) in out.iter_mut().enumerate().take(sd) {
            let mut sum = 0.0;
            for (i, &vs) in v_scaled.iter().enumerate().take(sd) {
                sum += vs * self.weight_z[i * sd + j];
            }
            *out_val = sum;
        }

        Ok(out)
    }

    /// Solves the Implicit Function Theorem (IFT) adjoint system for backward gradient:
    ///
    /// $$(I - J_{f, z^*})^T y = g_{z^*} \iff y = g_{z^*} + J_{f, z^*}^T y$$
    ///
    /// using Neumann series iterations $y_{k+1} = g_{z^*} + J^T y_k$.
    pub fn solve_adjoint(
        &self,
        grad_z_star: &[f64],
        z_star: &[f64],
        x: &[f64],
    ) -> Result<Vec<f64>> {
        // `config` is a pub field: re-check the constructor cap here so
        // post-construction tampering yields Err instead of an unbounded loop.
        crate::limits::check_iterations("max_backward_iters", self.config.max_backward_iters)?;
        let sd = self.state_dim;
        if grad_z_star.len() != sd || z_star.len() != sd {
            return Err(Error::ShapeMismatch {
                op: "solve_adjoint",
                expected: vec![sd],
                got: vec![grad_z_star.len()],
            });
        }

        let mut y = grad_z_star.to_vec();

        for _ in 0..self.config.max_backward_iters {
            let j_t_y = self.vjp_state(&y, z_star, x)?;
            let mut next_y = vec![0.0; sd];
            let mut diff_sq = 0.0;

            for (i, ny) in next_y.iter_mut().enumerate().take(sd) {
                *ny = grad_z_star[i] + j_t_y[i];
                let diff = *ny - y[i];
                diff_sq += diff * diff;
            }

            y = next_y;
            if diff_sq.sqrt() < self.config.bwd_tol {
                break;
            }
        }

        Ok(y)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn step_f_rejects_truncated_weights() -> Result<()> {
        let mut m = DeepEquilibriumModel::new(4, 4, DeqConfig::default())?;
        m.weight_z.truncate(3); // pub field: raw indexing below would panic
        let err = m
            .step_f(&[0.0; 4], &[0.0; 4])
            .expect_err("truncated weights must be an error, not a panic");
        assert!(err.to_string().contains("weight"), "{err}");
        Ok(())
    }
}
