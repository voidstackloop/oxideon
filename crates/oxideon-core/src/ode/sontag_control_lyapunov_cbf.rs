//! Sontag Universal Control Lyapunov Formula & Ames Zeroing Control Barrier Functions (CBF) (Sontag, 1989; Ames et al., 2017).
//!
//! ### 1. Sontag's Universal Stabilizing Formula & Zeroing CBF Safety Filter
//! For $\dot{x} = f(x) + g(x) u$ with Control Lyapunov Function $V(x)$ ($a(x) = L_f V$, $b(x) = L_g V$), Sontag's universal
//! stabilizing feedback law is:
//! $$u_{\mathrm{Sontag}}(a, b) = -\frac{a + \sqrt{a^2 + b^4}}{b} \quad (b \ne 0)$$
//! And a Zeroing Control Barrier Function $h(x) \ge 0$ enforces forward invariance of the safe set $\mathcal{C} = \{x : h(x) \ge 0\}$
//! via the minimally invasive QP safety filter $L_f h(x) + L_g h(x) u + \alpha h(x) \ge 0$.

/// Sontag Universal Control Lyapunov (CLF) & Ames Control Barrier Function (CBF) Engine.
#[derive(Debug, Clone, Copy, Default)]
pub struct ControlBarrierLyapunovEngine;

impl ControlBarrierLyapunovEngine {
    /// Creates a new CLF-CBF engine.
    pub fn new() -> Self {
        Self
    }

    /// Evaluates Sontag's universal stabilizing control $u(a, b)$ given Lie derivatives $a = L_f V(x)$ and $b = L_g V(x)$.
    pub fn sontag_universal_control(lf_v: f64, lg_v: f64) -> f64 {
        if lg_v.abs() < 1e-12 {
            0.0
        } else {
            -(lf_v + (lf_v * lf_v + lg_v.powi(4)).sqrt()) / lg_v
        }
    }

    /// Projects a nominal control $u_{\mathrm{nom}}$ onto the CBF safety half-line $L_f h + L_g h \, u + \alpha h \ge 0$ (for scalar $u$).
    pub fn cbf_safety_filter(u_nom: f64, lf_h: f64, lg_h: f64, alpha_h: f64) -> f64 {
        if lf_h + lg_h * u_nom + alpha_h >= 0.0 || lg_h.abs() < 1e-12 {
            u_nom
        } else {
            -(lf_h + alpha_h) / lg_h
        }
    }
}
