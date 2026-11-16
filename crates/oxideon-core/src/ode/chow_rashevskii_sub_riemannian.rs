//! Sub-Riemannian Geometry, Chow-Rashevskii Theorem, Brockett Nonholonomic Integrator & Heisenberg Carnot-Carathéodory Metric (Chow, 1939).
//!
//! ### 1. Brockett's Nonholonomic Heisenberg Integrator $\dot{x} = u_1, \dot{y} = u_2, \dot{z} = \frac{1}{2}(x u_2 - y u_1)$
//! Although the horizontal distribution $\mathcal{D} = \operatorname{span}(X_1, X_2)$ has rank 2 in $\mathbb{R}^3$,
//! the Lie bracket $[X_1, X_2] = \partial_z$ spans the vertical missing direction (step-2 Carnot group).
//! Sub-Riemannian geodesics from $(0,0,0)$ to $(0, 0, z)$ are circular helices enclosing signed area $A = 2 z$ with exact
//! Carnot-Carathéodory distance:
//! $$d_{\mathrm{CC}}\left((0,0,0), (0, 0, z)\right) = \sqrt{4 \pi |z|}$$

use std::f64::consts::PI;

/// Sub-Riemannian Heisenberg Carnot Group & Chow-Rashevskii Geodesic Engine.
#[derive(Debug, Clone, Copy, Default)]
pub struct SubRiemannianHeisenbergEngine;

impl SubRiemannianHeisenbergEngine {
    /// Creates a new Sub-Riemannian Heisenberg engine.
    pub fn new() -> Self {
        Self
    }

    /// Evaluates the exact vertical Carnot-Carathéodory sub-Riemannian distance $d_{\mathrm{CC}}((0,0,0), (0,0,z)) = \sqrt{4 \pi |z|}$
    /// along the Dido isoperimetric optimal circle.
    pub fn vertical_carnot_distance(z_height: f64) -> f64 {
        (4.0 * PI * z_height.abs()).sqrt()
    }
}
