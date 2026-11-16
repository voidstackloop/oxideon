//! Lorentz (Hyperboloid) Model of Hyperbolic Geometry & Riemannian Operations.
//!
//! Features:
//! - Stable, non-singular representation on the forward hyperboloid sheet $\mathbb{L}^n = \{ x \in \mathbb{R}^{n+1} : \langle x, x \rangle_{\mathbb{L}} = -1/c, x_0 > 0 \}$.
//! - Minkowski bilinear inner product: $\langle u, v \rangle_{\mathbb{L}} = -u_0 v_0 + \sum_{i=1}^n u_i v_i$.
//! - Exact hyperbolic geodesic distance: $d_{\mathbb{L}}(u, v) = \frac{1}{\sqrt{c}} \operatorname{arcosh}(-c \langle u, v \rangle_{\mathbb{L}})$.
//! - Manifold projection, exponential and logarithmic maps at the origin $\mu = (\frac{1}{\sqrt{c}}, 0, \dots, 0)$.

use crate::error::{Error, Result};
use crate::tensor::Tensor;

/// Lorentz (Hyperboloid) manifold model of hyperbolic geometry.
#[derive(Debug, Clone, Copy, Default)]
pub struct LorentzManifold;

impl LorentzManifold {
    pub const EPS: f64 = 1e-6;

    fn check_last_dim(op: &'static str, x: &Tensor) -> Result<usize> {
        match x.shape().last() {
            Some(&d) if d >= 1 => Ok(d - 1),
            _ => Err(Error::Msg(format!(
                "{op}: last dim must have size >= 1, got {:?}",
                x.shape()
            ))),
        }
    }

    /// Computes the Minkowski inner product $\langle u, v \rangle_{\mathbb{L}} = -u_0 v_0 + \sum_{i=1}^n u_i v_i$.
    pub fn minkowski_inner_product(u: &Tensor, v: &Tensor) -> Result<Tensor> {
        let space_dim = Self::check_last_dim("minkowski_inner_product", u)?;
        let v_space = Self::check_last_dim("minkowski_inner_product", v)?;
        if v_space != space_dim {
            return Err(Error::ShapeMismatch {
                op: "minkowski_inner_product",
                expected: u.shape().to_vec(),
                got: v.shape().to_vec(),
            });
        }
        let u0 = u.narrow(-1, 0, 1)?;
        let v0 = v.narrow(-1, 0, 1)?;
        let time_term = u0.mul(&v0)?.neg()?;

        if space_dim > 0 {
            let u_space = u.narrow(-1, 1, space_dim)?;
            let v_space = v.narrow(-1, 1, space_dim)?;
            let space_term = u_space.mul(&v_space)?.sum_axis(-1, true)?;
            time_term.add(&space_term)
        } else {
            Ok(time_term)
        }
    }

    /// Projects arbitrary vectors onto the forward Lorentz hyperboloid sheet:
    /// $$x_0 = \sqrt{\frac{1}{c} + \|x_{1:n}\|^2}$$
    pub fn project(x: &Tensor, c: f64) -> Result<Tensor> {
        let c = c.max(1e-5);
        let space_dim = Self::check_last_dim("project", x)?;
        let x_space = x.narrow(-1, 1, space_dim)?;
        let space_norm_sq = x_space.powf(2.0)?.sum_axis(-1, true)?;

        let x0 = space_norm_sq.add_scalar(1.0 / c)?.sqrt()?;
        Tensor::cat(&[x0, x_space], -1)
    }

    /// Computes the geodesic distance in the Lorentz model:
    /// $$d_{\mathbb{L}}(u, v) = \frac{1}{\sqrt{c}} \operatorname{arcosh}(-c \langle u, v \rangle_{\mathbb{L}})$$
    pub fn distance(u: &Tensor, v: &Tensor, c: f64) -> Result<Tensor> {
        let c = c.max(1e-5);
        let u_proj = Self::project(u, c)?;
        let v_proj = Self::project(v, c)?;

        let inner = Self::minkowski_inner_product(&u_proj, &v_proj)?;
        let scaled = inner.mul_scalar(-c)?;
        let clamped = scaled.clamp(1.0, f64::INFINITY)?;
        let acosh = Self::tensor_arcosh(&clamped)?;

        acosh.mul_scalar(1.0 / c.sqrt())
    }

    /// Exponential map from the tangent space at the origin $\mu = (\frac{1}{\sqrt{c}}, 0, \dots, 0)$
    /// to the Lorentz manifold.
    ///
    /// Input `v` is a Euclidean spatial vector in $\mathbb{R}^n$.
    pub fn exp_map_zero(v: &Tensor, c: f64) -> Result<Tensor> {
        let c = c.max(1e-5);
        let sqrt_c = c.sqrt();
        let v_norm = v
            .norm_p(2.0, &[-1], true)?
            .clamp(Self::EPS, f64::INFINITY)?;

        // x0 = cosh(sqrt(c) * ||v||) / sqrt(c)
        let scaled_norm = v_norm.mul_scalar(sqrt_c)?;
        let cosh_term = Self::tensor_cosh(&scaled_norm)?;
        let x0 = cosh_term.mul_scalar(1.0 / sqrt_c)?;

        // x_space = sinh(sqrt(c) * ||v||) / (sqrt(c) * ||v||) * v
        let sinh_term = Self::tensor_sinh(&scaled_norm)?;
        let scale = sinh_term.div(&v_norm)?.mul_scalar(1.0 / sqrt_c)?;
        let x_space = v.mul(&scale)?;

        Tensor::cat(&[x0, x_space], -1)
    }

    /// Logarithmic map from the Lorentz manifold back to the tangent space at the origin.
    ///
    /// Returns the spatial Euclidean vector in $\mathbb{R}^n$.
    pub fn log_map_zero(y: &Tensor, c: f64) -> Result<Tensor> {
        let c = c.max(1e-5);
        let sqrt_c = c.sqrt();
        let space_dim = Self::check_last_dim("log_map_zero", y)?;
        let y0 = y.narrow(-1, 0, 1)?;
        let y_space = y.narrow(-1, 1, space_dim)?;

        // dist = arcosh(sqrt(c) * y0)
        let scaled_y0 = y0.mul_scalar(sqrt_c)?.clamp(1.0, f64::INFINITY)?;
        let dist = Self::tensor_arcosh(&scaled_y0)?;

        let space_norm = y_space
            .norm_p(2.0, &[-1], true)?
            .clamp(Self::EPS, f64::INFINITY)?;
        let scale = dist.div(&space_norm)?.mul_scalar(1.0 / sqrt_c)?;
        y_space.mul(&scale)
    }

    fn tensor_arcosh(x: &Tensor) -> Result<Tensor> {
        let x_sq_minus_1 = x.powf(2.0)?.sub_scalar(1.0)?.clamp(0.0, f64::INFINITY)?;
        let sqrt_part = x_sq_minus_1.sqrt()?;
        x.add(&sqrt_part)?.ln()
    }

    fn tensor_cosh(x: &Tensor) -> Result<Tensor> {
        let exp_x = x.exp()?;
        let exp_neg_x = x.neg()?.exp()?;
        exp_x.add(&exp_neg_x)?.mul_scalar(0.5)
    }

    fn tensor_sinh(x: &Tensor) -> Result<Tensor> {
        let exp_x = x.exp()?;
        let exp_neg_x = x.neg()?.exp()?;
        exp_x.sub(&exp_neg_x)?.mul_scalar(0.5)
    }
}
