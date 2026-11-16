//! Ames Control Barrier Function (CBF) Safety Filter & CLF-CBF Quadratic Program (Ames, Grizzle & Tabuada, 2014; Ames et al., 2019).
//!
//! ### 1. Forward Invariance of Safe Set $\mathcal{C} = \{x : h(x) \ge 0\}$ via Nagumo Condition $\dot{h}(x, u) + \alpha(h(x)) \ge 0$
//! For a nominal control $u_{\mathrm{nom}}$ and affine dynamics $\dot{x} = f(x) + g(x) u$, the minimum-norm safety-filtering QP:
//! $$u^* = \arg\min_u \frac{1}{2}\|u - u_{\mathrm{nom}}\|^2 \quad \text{s.t.} \quad L_f h(x) + L_g h(x) u + \gamma h(x) \ge 0$$
//! has closed-form KKT solution $u^* = u_{\mathrm{nom}} - \frac{\min(0, \Psi(x, u_{\mathrm{nom}}))}{\|L_g h(x)\|^2} L_g h(x)^\top$ where $\Psi = L_f h + L_g h \, u_{\mathrm{nom}} + \gamma h$.

use crate::error::{Error, Result};

/// Ames Control Barrier Function (CBF) Safety Filter QP Engine.
#[derive(Debug, Clone, Copy)]
pub struct AmesControlBarrierFunctionQp {
    /// Class-$\mathcal{K}$ barrier decay rate $\gamma > 0$.
    pub barrier_gamma: f64,
}

impl Default for AmesControlBarrierFunctionQp {
    fn default() -> Self {
        Self { barrier_gamma: 1.0 }
    }
}

impl AmesControlBarrierFunctionQp {
    /// Creates a new Ames Control Barrier Function QP engine.
    pub fn new(barrier_gamma: f64) -> Result<Self> {
        if barrier_gamma <= 0.0 || !barrier_gamma.is_finite() {
            return Err(Error::Msg("barrier_gamma must be > 0".to_string()));
        }
        Ok(Self { barrier_gamma })
    }

    /// Solves the 1D CBF-QP safety filter for $u^*$ and returns $(u^*, \dot{h}(x, u^*) + \gamma h(x))$.
    pub fn filter_nominal_control(
        &self,
        h_val: f64,
        lf_h: f64,
        lg_h: f64,
        u_nom: f64,
    ) -> Result<(f64, f64)> {
        if lg_h.abs() < 1e-12 {
            return Err(Error::Msg(
                "lg_h must be non-zero for relative-degree-1 CBF".to_string(),
            ));
        }
        let psi = lf_h + lg_h * u_nom + self.barrier_gamma * h_val;
        if psi >= 0.0 {
            Ok((u_nom, psi))
        } else {
            let u_safe = (-lf_h - self.barrier_gamma * h_val) / lg_h;
            Ok((u_safe, 0.0))
        }
    }
}
