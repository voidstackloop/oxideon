//! General Riemannian Geometric Manifolds & Cartan Affine Connections.
//!
//! Provides differential geometric flows on non-Euclidean manifolds:
//! - General metric tensor $g_{ij}(x)$ and inverse metric tensor $g^{ij}(x)$.
//! - Levi-Civita affine connection & Christoffel symbols of the second kind:
//!   $$\Gamma^k_{ij} = \frac{1}{2} g^{kl} \left( \frac{\partial g_{il}}{\partial x^j} + \frac{\partial g_{jl}}{\partial x^i} - \frac{\partial g_{ij}}{\partial x^l} \right)$$
//! - Geodesic exponential flow integrating second-order geodesic differential equations:
//!   $$\ddot{x}^k + \Gamma^k_{ij} \dot{x}^i \dot{x}^j = 0$$
//! - Parallel transport of tangent vectors $v \in T_x \mathcal{M}$ along geodesic curves.

use crate::error::{Error, Result};
use crate::tensor::Tensor;

/// A smooth Riemannian manifold characterized by a local metric tensor field.
#[derive(Debug, Clone)]
pub struct RiemannianManifold {
    pub dimension: usize,
}

impl RiemannianManifold {
    pub fn new(dimension: usize) -> Self {
        Self { dimension }
    }

    /// Evaluates the metric tensor $g(x) \in \mathbb{R}^{d \times d}$ at position $x$.
    ///
    /// Default metric: Conformal metric $g_{ij}(x) = \phi(x) \delta_{ij}$ where $\phi(x) = \frac{4}{(1 - \|x\|^2)^2}$
    /// (Poincaré ball metric) or general smooth coordinate-dependent positive definite metric.
    pub fn metric_tensor<F>(&self, x: &Tensor, metric_fn: &F) -> Result<Tensor>
    where
        F: Fn(&Tensor) -> Result<Tensor>,
    {
        metric_fn(x)
    }

    /// Computes the Christoffel symbols of the second kind $\Gamma^k_{ij}$ numerically via finite differences.
    ///
    /// Returns a tensor of shape `[d, d, d]` where indices are `[k, i, j]`.
    pub fn christoffel_symbols<F>(&self, x: &Tensor, metric_fn: &F, eps: f64) -> Result<Tensor>
    where
        F: Fn(&Tensor) -> Result<Tensor>,
    {
        let d = self.dimension;
        // d^3 workspaces below; gate the dimension before any d*d arithmetic.
        let cells3 = crate::limits::check_elements("christoffel workspace", &[d, d, d])?;
        let g = metric_fn(x)?;
        let g_inv = g.inv()?;

        let g_inv_vec = g_inv.to_vec::<f64>()?;
        let x_vec = x.to_vec::<f64>()?;
        if x_vec.len() != d {
            return Err(Error::Msg(format!(
                "christoffel_symbols: point has {} coords, manifold dim is {d}",
                x_vec.len()
            )));
        }
        if g_inv_vec.len() != d * d {
            return Err(Error::Msg(format!(
                "christoffel_symbols: metric inverse has {} entries, need {}",
                g_inv_vec.len(),
                d * d
            )));
        }

        // Compute partial derivatives dg_{ij} / dx^l
        // Shape: [l, i, j]
        let mut dg = crate::layout::try_filled(cells3, 0.0f64)?;

        for l in 0..d {
            let mut x_plus = x_vec.clone();
            let mut x_minus = x_vec.clone();
            x_plus[l] += eps;
            x_minus[l] -= eps;

            let t_plus = Tensor::from_vec(x_plus, x.shape())?;
            let t_minus = Tensor::from_vec(x_minus, x.shape())?;

            let g_plus = metric_fn(&t_plus)?.to_vec::<f64>()?;
            let g_minus = metric_fn(&t_minus)?.to_vec::<f64>()?;
            let need_sq = d * d;
            if g_plus.len() != need_sq || g_minus.len() != need_sq {
                return Err(Error::Msg(format!(
                    "christoffel_symbols: metric samples have ({}, {}) entries, need {need_sq}",
                    g_plus.len(),
                    g_minus.len()
                )));
            }

            let inv_2eps = 1.0 / (2.0 * eps);
            for i in 0..d {
                for j in 0..d {
                    let deriv = (g_plus[i * d + j] - g_minus[i * d + j]) * inv_2eps;
                    dg[l * d * d + i * d + j] = deriv;
                }
            }
        }

        // \Gamma^k_{ij} = 0.5 * \sum_l g^{kl} (dg_{il}/dx^j + dg_{jl}/dx^i - dg_{ij}/dx^l)
        let mut gamma = crate::layout::try_filled(cells3, 0.0f64)?;
        for k in 0..d {
            for i in 0..d {
                for j in 0..d {
                    let mut sum = 0.0f64;
                    for l in 0..d {
                        let g_kl = g_inv_vec[k * d + l];
                        let dg_il_dj = dg[j * d * d + i * d + l];
                        let dg_jl_di = dg[i * d * d + j * d + l];
                        let dg_ij_dl = dg[l * d * d + i * d + j];
                        sum += g_kl * (dg_il_dj + dg_jl_di - dg_ij_dl);
                    }
                    gamma[k * d * d + i * d + j] = 0.5 * sum;
                }
            }
        }

        Tensor::from_vec(gamma, &[d, d, d])
    }

    /// Computes geodesic acceleration $\ddot{x}^k = - \Gamma^k_{ij} \dot{x}^i \dot{x}^j$.
    pub fn geodesic_acceleration(&self, velocity: &Tensor, gamma: &Tensor) -> Result<Tensor> {
        let d = self.dimension;
        let v = velocity.to_vec::<f64>()?;
        let gamma_vec = gamma.to_vec::<f64>()?;
        if v.len() != d {
            return Err(Error::Msg(format!(
                "geodesic_acceleration: velocity has {} entries, need {d}",
                v.len()
            )));
        }
        let need3 = d
            .checked_mul(d)
            .and_then(|t| t.checked_mul(d))
            .ok_or_else(|| Error::Msg(format!("geodesic_acceleration: {d}^3 overflows")))?;
        if gamma_vec.len() != need3 {
            return Err(Error::Msg(format!(
                "geodesic_acceleration: gamma has {} entries, need {need3}",
                gamma_vec.len()
            )));
        }

        let mut a = vec![0.0f64; d];
        for k in 0..d {
            let mut acc = 0.0f64;
            for i in 0..d {
                for j in 0..d {
                    let g_ijk = gamma_vec[k * d * d + i * d + j];
                    acc += g_ijk * v[i] * v[j];
                }
            }
            a[k] = -acc;
        }

        Tensor::from_vec(a, velocity.shape())
    }

    /// Integrates a geodesic curve on the manifold given starting position $x_0$ and initial velocity $v_0$.
    ///
    /// Integrates $\ddot{x}^k + \Gamma^k_{ij} \dot{x}^i \dot{x}^j = 0$ over `steps` with step size `dt`.
    pub fn integrate_geodesic<F>(
        &self,
        x0: &Tensor,
        v0: &Tensor,
        metric_fn: &F,
        dt: f64,
        steps: usize,
    ) -> Result<(Tensor, Tensor)>
    where
        F: Fn(&Tensor) -> Result<Tensor>,
    {
        let d = self.dimension;
        // Each step runs two full Christoffel evaluations; uncap and hang.
        let steps = crate::limits::check_iterations("steps", steps)?;
        let cells = (steps + 1).checked_mul(d).ok_or_else(|| {
            Error::Msg(format!(
                "geodesic trajectory size ({steps} + 1) * {d} overflows"
            ))
        })?;
        let mut x = x0.to_dtype(crate::dtype::DType::F64)?;
        let mut v = v0.to_dtype(crate::dtype::DType::F64)?;

        let mut traj_x: Vec<f64> = crate::layout::try_with_capacity(cells)?;
        let mut traj_v: Vec<f64> = crate::layout::try_with_capacity(cells)?;

        traj_x.extend_from_slice(&x.to_vec::<f64>()?);
        traj_v.extend_from_slice(&v.to_vec::<f64>()?);

        for _ in 0..steps {
            let gamma = self.christoffel_symbols(&x, metric_fn, 1e-4)?;
            let a = self.geodesic_acceleration(&v, &gamma)?;

            // Leapfrog / Velocity-Verlet update
            let half_v = v.add(&a.mul_scalar(0.5 * dt)?)?;
            x = x.add(&half_v.mul_scalar(dt)?)?;

            let next_gamma = self.christoffel_symbols(&x, metric_fn, 1e-4)?;
            let next_a = self.geodesic_acceleration(&half_v, &next_gamma)?;
            v = half_v.add(&next_a.mul_scalar(0.5 * dt)?)?;

            traj_x.extend_from_slice(&x.to_vec::<f64>()?);
            traj_v.extend_from_slice(&v.to_vec::<f64>()?);
        }

        let x_tensor = Tensor::from_vec(traj_x, &[steps + 1, d])?;
        let v_tensor = Tensor::from_vec(traj_v, &[steps + 1, d])?;
        Ok((x_tensor, v_tensor))
    }

    /// Parallel transports a tangent vector $w \in T_x \mathcal{M}$ along velocity direction $v$:
    /// $$\dot{w}^k = - \Gamma^k_{ij} v^i w^j$$
    pub fn parallel_transport_step(
        &self,
        w: &Tensor,
        v: &Tensor,
        gamma: &Tensor,
        dt: f64,
    ) -> Result<Tensor> {
        let d = self.dimension;
        let w_vec = w.to_vec::<f64>()?;
        let v_vec = v.to_vec::<f64>()?;
        let gamma_vec = gamma.to_vec::<f64>()?;
        if w_vec.len() != d || v_vec.len() != d {
            return Err(Error::Msg(format!(
                "parallel_transport_step: vectors have ({}, {}) entries, need {d}",
                w_vec.len(),
                v_vec.len()
            )));
        }
        let need3 = d
            .checked_mul(d)
            .and_then(|t| t.checked_mul(d))
            .ok_or_else(|| Error::Msg(format!("parallel_transport_step: {d}^3 overflows")))?;
        if gamma_vec.len() != need3 {
            return Err(Error::Msg(format!(
                "parallel_transport_step: gamma has {} entries, need {need3}",
                gamma_vec.len()
            )));
        }

        let mut dw = vec![0.0f64; d];
        for k in 0..d {
            let mut acc = 0.0f64;
            for i in 0..d {
                for j in 0..d {
                    let g_ijk = gamma_vec[k * d * d + i * d + j];
                    acc += g_ijk * v_vec[i] * w_vec[j];
                }
            }
            dw[k] = -acc;
        }

        let delta_w = Tensor::from_vec(dw, w.shape())?.mul_scalar(dt)?;
        w.add(&delta_w)
    }
}
