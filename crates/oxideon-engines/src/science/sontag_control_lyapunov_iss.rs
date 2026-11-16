//! Sontag Universal Control Lyapunov Function (CLF) Formula & Input-to-State Stability (ISS) (Artstein, 1983; Sontag, 1989).
//!
//! ### 1. Sontag Universal Stabilizing Feedback Law $u_S(x)$ for $\dot{x} = f(x) + g(x) u$
//! Given a smooth Control Lyapunov Function $V(x)$ with Lie derivatives $a(x) = L_f V(x)$ and $b(x) = L_g V(x)$ satisfying $b(x) = 0 \implies a(x) < 0$ ($x \neq 0$), Sontag's universal formula is:
//! $$u_S(x) = -\frac{a(x) + \sqrt{a(x)^2 + |b(x)|^4}}{|b(x)|^2} b(x)^\top \quad (b(x) \neq 0)$$
//! yielding $\dot{V}(x) = -\sqrt{a(x)^2 + |b(x)|^4} < 0$.

use crate::error::{Error, Result};

/// Sontag Universal Control Lyapunov Function (CLF) Engine.
#[derive(Debug, Clone, Copy, Default)]
pub struct SontagControlLyapunovEngine;

impl SontagControlLyapunovEngine {
    /// Creates a new Sontag CLF engine.
    pub fn new() -> Self {
        Self
    }

    /// Evaluates Sontag's scalar feedback control $u_S(a, b)$ and closed-loop Lyapunov decay rate $\dot{V} = a + b u_S = -\sqrt{a^2 + b^4}$.
    pub fn sontag_control_and_lyapunov_rate(lie_f_a: f64, lie_g_b: f64) -> Result<(f64, f64)> {
        if !lie_f_a.is_finite() || !lie_g_b.is_finite() {
            return Err(Error::Msg("lie_f_a and lie_g_b must be finite".to_string()));
        }
        let b2 = lie_g_b * lie_g_b;
        let root = (lie_f_a * lie_f_a + b2 * b2).sqrt();
        if b2 < 1e-15 {
            if lie_f_a >= 0.0 {
                return Err(Error::Msg(
                    "CLF condition violated: b=0 requires a<0".to_string(),
                ));
            }
            return Ok((0.0, lie_f_a));
        }
        let u_s = -(lie_f_a + root) / lie_g_b;
        Ok((u_s, -root))
    }
}
