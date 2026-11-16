//! Non-Adiabatic Conical Intersections, Yarkony $(g, h)$ Branching Space & Longuet-Higgins $\pi$ Berry Phase (Longuet-Higgins, 1958; Yarkony, 1996).
//!
//! ### 1. Linear Vibronic Coupling Model in the 2D $(g, h)$ Branching Plane
//! Near a conical intersection seam of codimension 2, the two adiabatic potential energy surfaces $E_\pm(x, y)$
//! split linearly along the gradient-difference vector $\mathbf{g}$ and derivative-coupling vector $\mathbf{h}$:
//! $$E_\pm(x, y) = S_x x + S_y y \pm \sqrt{(g x)^2 + (h y)^2}$$
//! Encircling the degeneracy point $(0, 0)$ in nuclear coordinate space induces the topological **Longuet-Higgins Sign Flip**
//! (Berry phase $\gamma_B = \pi$, $\exp(i \gamma_B) = -1$).

/// Yarkony Conical Intersection $(g, h)$ Branching Space & Longuet-Higgins Phase Engine.
#[derive(Debug, Clone, Copy)]
pub struct YarkonyConicalIntersectionEngine {
    /// Gradient difference slope $g > 0$.
    pub slope_g: f64,
    /// Non-adiabatic coupling slope $h > 0$.
    pub slope_h: f64,
}

impl YarkonyConicalIntersectionEngine {
    /// Creates a new Yarkony Conical Intersection engine.
    pub fn new(slope_g: f64, slope_h: f64) -> Self {
        Self { slope_g, slope_h }
    }

    /// Evaluates the lower and upper adiabatic potential surfaces $(E_-(x, y), E_+(x, y))$ and the Jahn-Teller Mixing Angle $\theta(x, y) = \frac{1}{2} \operatorname{atan2}(h y, g x)$.
    pub fn adiabatic_surfaces_and_mixing_angle(&self, x: f64, y: f64) -> (f64, f64, f64) {
        let gx = self.slope_g * x;
        let hy = self.slope_h * y;
        let split = (gx * gx + hy * hy).sqrt();
        let theta = 0.5 * hy.atan2(gx);
        (-split, split, theta)
    }
}
