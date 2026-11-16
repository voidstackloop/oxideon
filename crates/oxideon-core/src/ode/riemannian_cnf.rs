//! Continuous Normalizing Flows on Riemannian Manifolds (Riemannian CNF).
//!
//! Implements continuous probability flows and exact log-density evaluation over curved
//! Riemannian manifolds (e.g. Poincaré ball model of hyperbolic geometry $\mathbb{H}^d$):
//! - Riemannian divergence: $\operatorname{div}_{\mathcal{M}}(v) = \operatorname{div}_{\mathbb{R}^d}(v) + d \cdot \langle v, \nabla \log \lambda(x) \rangle$.
//! - Exact continuous change of variables: $\frac{d}{dt} \log p_t(x_t) = -\operatorname{div}_{\mathcal{M}}(v_t)(x_t)$.
//! - Adaptive/Euler-Heun geometric ODE integration preserving manifold boundary constraints.

use crate::error::{Error, Result};
use crate::tensor::Tensor;

/// Trajectory output from Riemannian Continuous Normalizing Flow integration.
#[derive(Debug, Clone)]
pub struct RiemannianTrajectory {
    /// Final pushed-forward sample $x_1 \in \mathcal{M}$.
    pub x_1: Tensor,
    /// Cumulative change in log-density: $\Delta \log p = \log p(x_1) - \log p(x_0)$.
    pub delta_log_p: f64,
    /// Trajectory path positions across integration time steps.
    pub path: Tensor,
    /// Timestamps $t \in [t_0, t_1]$.
    pub timestamps: Vec<f64>,
}

/// Continuous Normalizing Flow configured over the Poincaré ball model $\mathbb{B}^d = \{x \in \mathbb{R}^d : \|x\| < 1\}$.
#[derive(Debug, Clone)]
pub struct PoincareCnfFlow {
    /// Dimension of the manifold $d$.
    pub dim: usize,
    /// Flow velocity network weights $W_v \in \mathbb{R}^{d \times d}$.
    pub weight_v: Vec<f64>,
    /// Time-dependent frequency parameters $\omega \in \mathbb{R}^d$.
    pub omega: Vec<f64>,
    /// Boundary safety clamp radius $r_{\max} < 1.0$ (e.g. 0.95).
    pub max_radius: f64,
}

impl PoincareCnfFlow {
    /// Creates a new Poincaré ball CNF with dimension $d$.
    pub fn new(dim: usize) -> Result<Self> {
        if dim == 0 {
            return Err(Error::Msg("PoincareCnfFlow dim must be >= 1".into()));
        }
        // Deterministic initialization
        let bound = (1.0 / dim as f64).sqrt();
        let cells = crate::layout::checked_numel(&[dim, dim])?;
        let mut weight_v: Vec<f64> = crate::layout::try_with_capacity(cells)?;
        weight_v.extend((0..cells).map(|i| {
            let frac = ((i * 37 + 19) % 1000) as f64 / 1000.0;
            (frac * 2.0 - 1.0) * bound
        }));

        let mut omega: Vec<f64> = crate::layout::try_with_capacity(dim)?;
        omega.extend((0..dim).map(|i| 1.0 + ((i * 13) % 5) as f64 * 0.5));

        Ok(Self {
            dim,
            weight_v,
            omega,
            max_radius: 0.95,
        })
    }

    /// Evaluates conformal factor $\lambda(x) = \frac{2}{1 - \|x\|^2}$.
    pub fn conformal_factor(&self, x: &[f64]) -> f64 {
        let r2: f64 = x.iter().map(|&v| v * v).sum();
        let clamped_r2 = r2.min(self.max_radius * self.max_radius);
        2.0 / (1.0 - clamped_r2).max(1e-6)
    }

    /// Evaluates gradient of log conformal factor:
    /// $\nabla \log \lambda(x) = \frac{2x}{1 - \|x\|^2}$.
    pub fn grad_log_conformal(&self, x: &[f64]) -> Vec<f64> {
        let r2: f64 = x.iter().map(|&v| v * v).sum();
        let clamped_r2 = r2.min(self.max_radius * self.max_radius);
        let factor = 2.0 / (1.0 - clamped_r2).max(1e-6);

        x.iter().map(|&val| factor * val).collect()
    }

    /// Verifies the velocity-field buffers agree with the (public,
    /// tamperable) dimension before any indexing.
    fn check_field_layout(&self, x: &[f64]) -> Result<()> {
        let d = self.dim;
        if x.len() != d {
            return Err(Error::Msg(format!(
                "CNF state length {} disagrees with dim {d}",
                x.len()
            )));
        }
        let cells = d
            .checked_mul(d)
            .ok_or_else(|| Error::Msg(format!("CNF weight size {d} * {d} overflows")))?;
        if self.weight_v.len() != cells {
            return Err(Error::Msg(format!(
                "CNF weight length {} disagrees with {d}*{d}",
                self.weight_v.len()
            )));
        }
        if self.omega.len() != d {
            return Err(Error::Msg(format!(
                "CNF omega length {} disagrees with dim {d}",
                self.omega.len()
            )));
        }
        Ok(())
    }

    /// Evaluates time-dependent Riemannian velocity field $v_t(x) \in T_x \mathbb{B}^d$:
    ///
    /// $$v_t(x) = \tanh(W_v x) \odot \cos(\omega t)$$
    pub fn velocity(&self, x: &[f64], t: f64) -> Result<Vec<f64>> {
        let d = self.dim;
        self.check_field_layout(x)?;
        let mut v = vec![0.0; d];

        for (i, v_val) in v.iter_mut().enumerate().take(d) {
            let mut sum = 0.0;
            for (j, &xj) in x.iter().enumerate().take(d) {
                sum += self.weight_v[i * d + j] * xj;
            }
            let time_mod = (self.omega[i] * t).cos();
            *v_val = sum.tanh() * time_mod;
        }

        Ok(v)
    }

    /// Computes the Riemannian divergence $\operatorname{div}_{\mathcal{M}}(v)(x)$ on the Poincaré manifold:
    ///
    /// $$\operatorname{div}_{\mathcal{M}}(v) = \operatorname{div}_{\mathbb{R}^d}(v) + d \cdot \langle v, \nabla \log \lambda(x) \rangle$$
    ///
    /// where $\operatorname{div}_{\mathbb{R}^d}(v) = \sum_{i=1}^d \frac{\partial v^i}{\partial x^i} = \sum_{i=1}^d (1 - \tanh^2(W_{v, i} x)) W_{ii} \cos(\omega_i t)$.
    pub fn riemannian_divergence(&self, x: &[f64], t: f64) -> Result<f64> {
        let d = self.dim;
        self.check_field_layout(x)?;

        // 1. Euclidean divergence trace: sum_i d v_i / d x_i
        let mut euc_div = 0.0;
        for i in 0..d {
            let mut sum = 0.0;
            for (j, &xj) in x.iter().enumerate().take(d) {
                sum += self.weight_v[i * d + j] * xj;
            }
            let tanh_val = sum.tanh();
            let d_tanh = 1.0 - tanh_val * tanh_val;
            let time_mod = (self.omega[i] * t).cos();
            euc_div += d_tanh * self.weight_v[i * d + i] * time_mod;
        }

        // 2. Metric curvature correction: d * <v, grad log lambda>
        let v = self.velocity(x, t)?;
        let grad_log_lam = self.grad_log_conformal(x);
        let inner_prod: f64 = v
            .iter()
            .zip(grad_log_lam.iter())
            .map(|(&vi, &gi)| vi * gi)
            .sum();

        let curvature_term = (d as f64) * inner_prod;

        Ok(euc_div + curvature_term)
    }

    /// Integrates the Riemannian CNF forward from $t=0$ to $t=T$ given initial sample $x_0 \in \mathbb{B}^d$:
    ///
    /// Computes trajectory path and cumulative log-density adjustment $\Delta \log p = \int_0^T -\operatorname{div}_{\mathcal{M}}(v_t) dt$.
    pub fn integrate(
        &self,
        x_0: &Tensor,
        num_steps: usize,
        total_time: f64,
    ) -> Result<RiemannianTrajectory> {
        let d = self.dim;
        let shape = x_0.shape();
        if shape.len() != 1 || shape[0] != d {
            return Err(Error::ShapeMismatch {
                op: "PoincareCnfFlow::integrate",
                expected: vec![d],
                got: shape.to_vec(),
            });
        }

        if num_steps == 0 {
            return Err(Error::Msg(
                "PoincareCnfFlow::integrate requires num_steps > 0".into(),
            ));
        }
        if !total_time.is_finite() {
            return Err(Error::Msg(format!(
                "PoincareCnfFlow::integrate requires finite total_time, got {total_time}"
            )));
        }
        let num_steps = crate::limits::check_iterations("num_steps", num_steps)?;
        let mut cur_x = x_0.to_vec::<f64>()?;
        let dt = total_time / (num_steps as f64);

        let mut delta_log_p = 0.0;
        let cells = (num_steps + 1).checked_mul(d).ok_or_else(|| {
            Error::Msg(format!("CNF path size ({num_steps} + 1) * {d} overflows"))
        })?;
        let mut path_data: Vec<f64> = crate::layout::try_with_capacity(cells)?;
        let mut timestamps: Vec<f64> = crate::layout::try_with_capacity(num_steps + 1)?;

        path_data.extend_from_slice(&cur_x);
        timestamps.push(0.0);

        for step in 0..num_steps {
            let t = (step as f64) * dt;

            // Heun / RK2 predictor-corrector on Riemannian manifold
            let v1 = self.velocity(&cur_x, t)?;
            let div1 = self.riemannian_divergence(&cur_x, t)?;

            // Predictor
            let mut pred_x = vec![0.0; d];
            for i in 0..d {
                pred_x[i] = cur_x[i] + dt * v1[i];
            }
            self.clamp_to_manifold(&mut pred_x);

            // Corrector
            let v2 = self.velocity(&pred_x, t + dt)?;
            let div2 = self.riemannian_divergence(&pred_x, t + dt)?;

            for i in 0..d {
                cur_x[i] += 0.5 * dt * (v1[i] + v2[i]);
            }
            self.clamp_to_manifold(&mut cur_x);

            // Density change: d(log p) / dt = - div_M(v)
            let avg_div = 0.5 * (div1 + div2);
            delta_log_p -= dt * avg_div;

            path_data.extend_from_slice(&cur_x);
            timestamps.push(t + dt);
        }

        let x_1 = Tensor::from_slice(&cur_x, &[d])?;
        let path = Tensor::from_slice(&path_data, &[num_steps + 1, d])?;

        Ok(RiemannianTrajectory {
            x_1,
            delta_log_p,
            path,
            timestamps,
        })
    }

    /// Enforces the Poincaré ball constraint $\|x\| \le r_{\max} < 1$.
    fn clamp_to_manifold(&self, x: &mut [f64]) {
        let r: f64 = x.iter().map(|&v| v * v).sum::<f64>().sqrt();
        if r > self.max_radius {
            let scale = self.max_radius / r;
            for val in x {
                *val *= scale;
            }
        }
    }
}
