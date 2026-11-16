//! Caffarelli Regularity of Optimal Transport Maps & Brenier-McCann Displacement Interpolation (Brenier, 1991; Caffarelli, 1992; McCann, 1997).
//!
//! ### 1. Monge-Ampère Jacobian Equation $\det(D^2 \varphi(x)) = f(x) / g(\nabla \varphi(x))$
//! By Brenier's theorem, the unique $W_2$ optimal transport map $T = \nabla \varphi$ is the gradient of a convex potential $\varphi$.
//! Under McCann's displacement interpolation $\mu_t = ((1 - t) \mathrm{Id} + t \nabla \varphi)_\# \mu_0$, the Jacobian is $\det((1 - t) I + t D^2 \varphi)$.

use crate::error::{Error, Result};

/// Caffarelli Monge-Ampère Regularity & McCann Displacement Interpolation Engine.
#[derive(Debug, Clone, Copy, Default)]
pub struct CaffarelliMongeAmpereTransport;

impl CaffarelliMongeAmpereTransport {
    /// Creates a new Caffarelli Monge-Ampère Transport engine.
    pub fn new() -> Self {
        Self
    }

    /// Evaluates the McCann displacement Jacobian $\det((1 - t) I + t D^2 \varphi) = \prod_i ((1 - t) + t \lambda_i)$ and verifies McCann displacement concavity $\det^{1/n}((1-t)I + t H) \ge (1-t) + t \det(H)^{1/n}$.
    pub fn mccann_displacement_jacobian_and_concavity(
        hessian_eigenvalues: &[f64],
        t: f64,
    ) -> Result<(f64, f64, bool)> {
        if hessian_eigenvalues.is_empty()
            || hessian_eigenvalues.iter().any(|&l| l <= 0.0)
            || !(0.0..=1.0).contains(&t)
        {
            return Err(Error::Msg(
                "Positive Hessian eigenvalues and t in [0, 1] required".to_string(),
            ));
        }
        let n = hessian_eigenvalues.len() as f64;
        let det_interp: f64 = hessian_eigenvalues
            .iter()
            .map(|&l| (1.0 - t) + t * l)
            .product();
        let det_h: f64 = hessian_eigenvalues.iter().product();
        let lhs = det_interp.powf(1.0 / n);
        let rhs = (1.0 - t) + t * det_h.powf(1.0 / n);
        Ok((det_interp, lhs - rhs, lhs >= rhs - 1e-12))
    }
}
