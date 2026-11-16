//! Alan Turing's Chemical Basis of Morphogenesis & Diffusion-Driven Pattern Instability (Turing, 1952).
//!
//! ### 1. Turing's Four Conditions for Diffusion-Driven Symmetry Breaking
//! For an activator-inhibitor reaction-diffusion system $\partial_t u = d_u \nabla^2 u + f(u,v)$, $\partial_t v = d_v \nabla^2 v + g(u,v)$
//! with homogeneous Jacobian $J = \begin{pmatrix} f_u & f_v \\ g_u & g_v \end{pmatrix}$, a spatially uniform steady state is
//! stable to homogeneous perturbations ($k = 0$) yet unstable to spatial wavenumber $k_c > 0$ iff:
//! 1. $\operatorname{Tr}(J) = f_u + g_v < 0$
//! 2. $\det(J) = f_u g_v - f_v g_u > 0$
//! 3. $d_v f_u + d_u g_v > 2 \sqrt{d_u d_v \det(J)}$

use crate::error::{Error, Result};

/// Alan Turing Reaction-Diffusion Morphogenesis & Critical Wavenumber Engine.
#[derive(Debug, Clone, Copy)]
pub struct TuringReactionDiffusionEngine {
    /// Activator diffusion coefficient $d_u > 0$.
    pub du: f64,
    /// Inhibitor diffusion coefficient $d_v > d_u$.
    pub dv: f64,
    /// Reaction Jacobian $[f_u, f_v, g_u, g_v]$.
    pub jacobian: [f64; 4],
}

impl TuringReactionDiffusionEngine {
    /// Creates a new Turing reaction-diffusion morphogenesis engine.
    pub fn new(du: f64, dv: f64, jacobian: [f64; 4]) -> Result<Self> {
        if du <= 0.0 || dv <= 0.0 {
            return Err(Error::Msg(
                "Diffusion coefficients du and dv must be > 0".to_string(),
            ));
        }
        Ok(Self { du, dv, jacobian })
    }

    /// Checks whether Turing's exact 3 instability criteria hold and returns `(is_turing_unstable, critical_k_sq)`.
    pub fn check_turing_instability(&self) -> (bool, f64) {
        let [fu, fv, gu, gv] = self.jacobian;
        let tr = fu + gv;
        let det = fu * gv - fv * gu;
        let cross = self.dv * fu + self.du * gv;
        let is_turing = tr < 0.0 && det > 0.0 && cross > 2.0 * (self.du * self.dv * det).sqrt();
        let kc_sq = (cross / (2.0 * self.du * self.dv)).max(0.0);
        (is_turing, kc_sq)
    }
}
