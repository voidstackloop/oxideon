//! Poincaré Ball Hyperbolic Manifold Engine for Hierarchical Knowledge Embeddings.
//!
//! Provides Riemannian operations on the Poincaré ball model $\mathbb{D}^n = \{ x \in \mathbb{R}^n : \|x\| < 1 \}$:
//! - Hyperbolic geodesic distance $d_{\mathbb{D}}(u, v) = \operatorname{arcosh}\left(1 + 2 \frac{\|u - v\|^2}{(1 - \|u\|^2)(1 - \|v\|^2)}\right)$
//! - Möbius vector addition $u \oplus v$
//! - Exponential and logarithmic maps ($\exp_0, \log_0$)

use crate::error::Result;
use crate::tensor::Tensor;

/// Poincaré ball model of hyperbolic geometry.
#[derive(Debug, Clone, Copy, Default)]
pub struct PoincareManifold;

impl PoincareManifold {
    pub const EPS: f64 = 1e-5;

    /// Projects points into the open unit ball $\|x\| \le 1 - \epsilon$.
    pub fn project(x: &Tensor) -> Result<Tensor> {
        let norm = x.norm_p(2.0, &[-1], true)?;
        let max_norm = 1.0 - Self::EPS;
        let denom = norm.clamp(max_norm, f64::INFINITY)?;
        let scale = denom.recip()?.mul_scalar(max_norm)?;
        x.mul(&scale)
    }

    /// Computes pairwise hyperbolic distance between points `u` and `v` of shape `[N, D]`:
    /// $$d_{\mathbb{D}}(u, v) = \operatorname{arcosh}\left(1 + 2 \frac{\|u - v\|^2}{(1 - \|u\|^2)(1 - \|v\|^2)}\right)$$
    pub fn distance(u: &Tensor, v: &Tensor) -> Result<Tensor> {
        let u_proj = Self::project(u)?;
        let v_proj = Self::project(v)?;

        let diff = u_proj.sub(&v_proj)?;
        let diff_sq = diff.powf(2.0)?.sum_axis(-1, true)?;

        let u_sq = u_proj.powf(2.0)?.sum_axis(-1, true)?;
        let v_sq = v_proj.powf(2.0)?.sum_axis(-1, true)?;

        let denom_u = u_sq.neg()?.add_scalar(1.0)?.clamp(Self::EPS, 1.0)?;
        let denom_v = v_sq.neg()?.add_scalar(1.0)?.clamp(Self::EPS, 1.0)?;
        let denom = denom_u.mul(&denom_v)?;

        let delta = diff_sq.mul_scalar(2.0)?.div(&denom)?;
        let arg = delta.add_scalar(1.0 + Self::EPS)?;

        // arcosh(x) = ln(x + sqrt(x^2 - 1))
        let arg_sq_minus_1 = arg.powf(2.0)?.sub_scalar(1.0)?.clamp(0.0, f64::INFINITY)?;
        let sqrt_part = arg_sq_minus_1.sqrt()?;
        arg.add(&sqrt_part)?.ln()
    }

    /// Computes Möbius addition of vectors $u \oplus v$:
    /// $$u \oplus v = \frac{(1 + 2\langle u, v\rangle + \|v\|^2) u + (1 - \|u\|^2) v}{1 + 2\langle u, v\rangle + \|u\|^2 \|v\|^2}$$
    pub fn mobius_add(u: &Tensor, v: &Tensor) -> Result<Tensor> {
        let u_proj = Self::project(u)?;
        let v_proj = Self::project(v)?;

        let u_sq = u_proj.powf(2.0)?.sum_axis(-1, true)?;
        let v_sq = v_proj.powf(2.0)?.sum_axis(-1, true)?;
        let uv = u_proj.mul(&v_proj)?.sum_axis(-1, true)?;

        // Numerator: (1 + 2 <u, v> + ||v||^2) u + (1 - ||u||^2) v
        let coeff_u = uv.mul_scalar(2.0)?.add_scalar(1.0)?.add(&v_sq)?;
        let coeff_v = u_sq.neg()?.add_scalar(1.0)?;
        let num = u_proj.mul(&coeff_u)?.add(&v_proj.mul(&coeff_v)?)?;

        // Denominator: 1 + 2 <u, v> + ||u||^2 ||v||^2
        let denom = uv
            .mul_scalar(2.0)?
            .add_scalar(1.0)?
            .add(&u_sq.mul(&v_sq)?)?;

        let res = num.div(&denom.clamp(Self::EPS, f64::INFINITY)?)?;
        Self::project(&res)
    }

    /// Exponential map at origin $\exp_0(v) = \tanh(\|v\|) \frac{v}{\|v\|}$.
    pub fn exp_map_zero(v: &Tensor) -> Result<Tensor> {
        let norm = v
            .norm_p(2.0, &[-1], true)?
            .clamp(Self::EPS, f64::INFINITY)?;
        let tanh_norm = norm.tanh()?;
        let scale = tanh_norm.div(&norm)?;
        let res = v.mul(&scale)?;
        Self::project(&res)
    }

    /// Logarithmic map at origin $\log_0(y) = \operatorname{artanh}(\|y\|) \frac{y}{\|y\|}$.
    pub fn log_map_zero(y: &Tensor) -> Result<Tensor> {
        let y_proj = Self::project(y)?;
        let norm = y_proj
            .norm_p(2.0, &[-1], true)?
            .clamp(Self::EPS, 1.0 - Self::EPS)?;
        // artanh(x) = 0.5 * ln((1 + x) / (1 - x))
        let one_plus = norm.add_scalar(1.0)?;
        let one_minus = norm.neg()?.add_scalar(1.0)?.clamp(Self::EPS, 2.0)?;
        let artanh = one_plus.div(&one_minus)?.ln()?.mul_scalar(0.5)?;
        let scale = artanh.div(&norm)?;
        y_proj.mul(&scale)
    }
}
