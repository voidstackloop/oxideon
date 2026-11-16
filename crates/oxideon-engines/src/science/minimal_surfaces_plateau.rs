//! Plateau's Problem, Weierstrass-Enneper Minimal Surface Parameterization & Willmore Conjecture (Marques & Neves, 2014).
//!
//! ### 1. Weierstrass-Enneper Minimal Surface Immersion ($H \equiv 0$)
//! Every minimal surface in $\mathbb{R}^3$ (mean curvature $H = \frac{1}{2}(\kappa_1 + \kappa_2) = 0$) admits a local
//! conformal parameterization with Gaussian curvature:
//! $$K(z) = -\left(\frac{4 |g'(z)|}{|f(z)| (1 + |g(z)|^2)^2}\right)^2 \le 0$$
//! For Enneper's minimal surface ($f(z) = 1, g(z) = z$), $K(r) = -\frac{16}{(1 + r^2)^4}$ and $H \equiv 0$.
//!
//! ### 2. Marques-Neves Theorem on the Willmore Conjecture (2014)
//! For any immersed torus $\Sigma \subset \mathbb{R}^3$, the Willmore bending energy $\mathcal{W}(\Sigma) = \int_\Sigma H^2 dA$
//! satisfies $\mathcal{W}(\Sigma) \ge 2 \pi^2$, saturated by the Clifford Torus (radii ratio $R / r = \sqrt{2}$).

use std::f64::consts::PI;

/// Weierstrass-Enneper Minimal Surface & Marques-Neves Willmore Energy Engine.
#[derive(Debug, Clone, Copy, Default)]
pub struct WeierstrassMinimalSurfaceEngine;

impl WeierstrassMinimalSurfaceEngine {
    /// Creates a new Weierstrass Minimal Surface engine.
    pub fn new() -> Self {
        Self
    }

    /// Evaluates coordinates $(x, y, z)$ and Gaussian curvature $K \le 0$ of Enneper's minimal surface at $(u, v)$.
    pub fn enneper_surface_point_and_curvature(u: f64, v: f64) -> ([f64; 3], f64) {
        let x = u - u.powi(3) / 3.0 + u * v * v;
        let y = v - v.powi(3) / 3.0 + v * u * u;
        let z = u * u - v * v;
        let r2 = u * u + v * v;
        let gauss_k = -16.0 / (1.0 + r2).powi(4);
        ([x, y, z], gauss_k)
    }

    /// Computes the Willmore energy $\mathcal{W}(R, r) = \frac{\pi^2 R^2}{\sqrt{R^2 - r^2}}$ of a torus of revolution ($R > r > 0$),
    /// which achieves its global minimum $2 \pi^2$ at the Clifford Torus $R = \sqrt{2} r$.
    pub fn torus_willmore_energy(major_r: f64, minor_r: f64) -> f64 {
        PI * PI * major_r * major_r / (major_r * major_r - minor_r * minor_r).sqrt()
    }
}
