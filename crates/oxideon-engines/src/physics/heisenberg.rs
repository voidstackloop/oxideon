//! Sub-Riemannian Geometry, Carnot Groups & Hypoelliptic Diffusion on the Heisenberg Group $\mathbb{H}^1$.
//!
//! # Mathematical Foundations
//!
//! The first Heisenberg group $\mathbb{H}^1 \cong \mathbb{R}^3$ is the prototypical sub-Riemannian manifold
//! and step-2 nilpotent Carnot group:
//!
//! 1. **Group Law**:
//!    $$(x_1, y_1, z_1) \star (x_2, y_2, z_2) = \left(x_1 + x_2, \, y_1 + y_2, \, z_1 + z_2 + \frac{1}{2}(x_1 y_2 - y_1 x_2)\right)$$
//!    with identity $e = (0, 0, 0)$ and inverse $(x, y, z)^{-1} = (-x, -y, -z)$.
//!
//! 2. **Non-Holonomic Frame & Lie Algebra $\mathfrak{h}_1$**:
//!    Left-invariant vector fields:
//!    $$X = \frac{\partial}{\partial x} - \frac{y}{2}\frac{\partial}{\partial z}, \quad Y = \frac{\partial}{\partial y} + \frac{x}{2}\frac{\partial}{\partial z}, \quad Z = \frac{\partial}{\partial z}$$
//!    The horizontal distribution $\Delta = \operatorname{span}\{X, Y\}$ is bracket-generating:
//!    $$[X, Y] = Z, \quad [X, Z] = 0, \quad [Y, Z] = 0$$
//!
//! 3. **Sub-Laplacian & Hörmander Hypoellipticity**:
//!    $$\Delta_H = X^2 + Y^2 = \partial_{xx} + \partial_{yy} + (x \partial_{yz} - y \partial_{xz}) + \frac{x^2 + y^2}{4} \partial_{zz}$$
//!    By Hörmander's theorem, although $\Delta_H$ is degenerate (rank 2 in a 3D tangent bundle),
//!    it is strictly hypoelliptic: smooth solutions emerge from non-smooth initial data along all directions,
//!    including the non-holonomic vertical direction $Z$.
//!
//! 4. **Carnot-Carathéodory (CC) Geodesic Distance**:
//!    The intrinsic sub-Riemannian shortest path length between points. For relative vector $(x, y, z)$ with
//!    $r = \sqrt{x^2 + y^2} > 0$, the characteristic parameter $\theta \in (-\pi, \pi)$ solves:
//!    $$\mu(\theta) \equiv \frac{\theta - \sin\theta\cos\theta}{\sin^2\theta} = \frac{4 z}{r^2}$$
//!    and the exact sub-Riemannian distance is:
//!    $$d_{CC} = \frac{r |\theta|}{\sin|\theta|}$$
//!    For $r = 0$, $d_{CC} = \sqrt{4 \pi |z|}$.
//!
//! 5. **Korányi Gauge Metric & Anisotropic Dilation**:
//!    $$N(x, y, z) = \left((x^2 + y^2)^2 + 16 z^2\right)^{1/4}$$
//!    Satisfies the anisotropic Carnot dilation $\delta_\lambda(x, y, z) = (\lambda x, \lambda y, \lambda^2 z)$
//!    with homogeneous dimension $Q = 1 + 1 + 2 = 4$.

use crate::error::{Error, Result};
use std::f64::consts::PI;

/// A point in the 3-dimensional Heisenberg group $\mathbb{H}^1 = (\mathbb{R}^3, \star)$.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct HeisenbergPoint {
    /// Horizontal coordinate $x$.
    pub x: f64,
    /// Horizontal coordinate $y$.
    pub y: f64,
    /// Vertical / central coordinate $z$.
    pub z: f64,
}

impl HeisenbergPoint {
    /// Constructs a point $(x, y, z) \in \mathbb{H}^1$.
    pub const fn new(x: f64, y: f64, z: f64) -> Self {
        Self { x, y, z }
    }

    /// Identity element $e = (0, 0, 0)$.
    pub const fn identity() -> Self {
        Self {
            x: 0.0,
            y: 0.0,
            z: 0.0,
        }
    }

    /// Heisenberg group multiplication:
    ///
    /// $$(x_1, y_1, z_1) \star (x_2, y_2, z_2) = \left(x_1 + x_2, \, y_1 + y_2, \, z_1 + z_2 + \frac{1}{2}(x_1 y_2 - y_1 x_2)\right)$$
    pub fn multiply(&self, other: &Self) -> Self {
        Self {
            x: self.x + other.x,
            y: self.y + other.y,
            z: self.z + other.z + 0.5 * (self.x * other.y - self.y * other.x),
        }
    }

    /// Group inverse: $(x, y, z)^{-1} = (-x, -y, -z)$.
    pub fn inverse(&self) -> Self {
        Self {
            x: -self.x,
            y: -self.y,
            z: -self.z,
        }
    }

    /// Horizontal projection $(x, y)$ and horizontal radius $r = \sqrt{x^2 + y^2}$.
    pub fn horizontal_radius(&self) -> f64 {
        (self.x * self.x + self.y * self.y).sqrt()
    }

    /// Korányi gauge norm:
    ///
    /// $$N(x, y, z) = \left((x^2 + y^2)^2 + 16 z^2\right)^{1/4}$$
    pub fn koranyi_norm(&self) -> f64 {
        let r2 = self.x * self.x + self.y * self.y;
        (r2 * r2 + 16.0 * self.z * self.z).powf(0.25)
    }

    /// Left-invariant Korányi distance $d_K(p, q) = N(p^{-1} \star q)$.
    pub fn koranyi_distance(&self, other: &Self) -> f64 {
        let diff = self.inverse().multiply(other);
        diff.koranyi_norm()
    }

    /// Exact Carnot-Carathéodory (CC) sub-Riemannian geodesic distance $d_{CC}(p, q) = d_{CC}(e, p^{-1} \star q)$.
    pub fn carnot_caratheodory_distance(&self, other: &Self) -> f64 {
        let rel = self.inverse().multiply(other);
        let r = rel.horizontal_radius();
        let abs_z = rel.z.abs();

        if r < 1e-12 {
            // Pure vertical displacement along central axis
            return (4.0 * PI * abs_z).sqrt();
        }

        if abs_z < 1e-14 {
            // Pure horizontal geodesic line segment
            return r;
        }

        let target = 4.0 * abs_z / (r * r);

        // Solve mu(theta) = target for theta in (0, pi) using bracketed bisection / Newton
        let theta = solve_cc_theta(target);
        r * theta / theta.sin()
    }

    /// Anisotropic Carnot dilation $\delta_\lambda(x, y, z) = (\lambda x, \lambda y, \lambda^2 z)$.
    pub fn dilate(&self, lambda: f64) -> Self {
        Self {
            x: lambda * self.x,
            y: lambda * self.y,
            z: lambda * lambda * self.z,
        }
    }
}

/// Solves $\mu(\theta) = \frac{\theta - \sin\theta\cos\theta}{\sin^2\theta} = \text{target}$
/// for $\theta \in (0, \pi)$.
fn solve_cc_theta(target: f64) -> f64 {
    let mut low = 1e-7;
    let mut high = PI - 1e-7;

    let mu_fn = |th: f64| -> f64 {
        let s = th.sin();
        let c = th.cos();
        (th - s * c) / (s * s)
    };

    // Bisection for robust global convergence
    for _ in 0..60 {
        let mid = 0.5 * (low + high);
        let val = mu_fn(mid);
        if val < target {
            low = mid;
        } else {
            high = mid;
        }
    }
    0.5 * (low + high)
}

/// Element of the Heisenberg Lie algebra $\mathfrak{h}_1 = \operatorname{span}\{X, Y, Z\}$.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct HeisenbergLieAlgebra {
    /// Coefficient of $X$.
    pub vx: f64,
    /// Coefficient of $Y$.
    pub vy: f64,
    /// Coefficient of $Z$.
    pub vz: f64,
}

impl HeisenbergLieAlgebra {
    /// Creates a Lie algebra element $v = v_x X + v_y Y + v_z Z$.
    pub const fn new(vx: f64, vy: f64, vz: f64) -> Self {
        Self { vx, vy, vz }
    }

    /// Lie bracket $[v_1, v_2] = (v_{1x} v_{2y} - v_{1y} v_{2x}) Z$.
    pub fn bracket(&self, other: &Self) -> Self {
        Self {
            vx: 0.0,
            vy: 0.0,
            vz: self.vx * other.vy - self.vy * other.vx,
        }
    }

    /// Lie group exponential map $\exp: \mathfrak{h}_1 \to \mathbb{H}^1$.
    pub fn exp(&self) -> HeisenbergPoint {
        HeisenbergPoint::new(self.vx, self.vy, self.vz)
    }

    /// Lie group logarithmic map $\log: \mathbb{H}^1 \to \mathfrak{h}_1$.
    pub fn log(point: &HeisenbergPoint) -> Self {
        Self::new(point.x, point.y, point.z)
    }
}

/// 3D Hypoelliptic Sub-Laplacian Diffusion Solver on $\mathbb{H}^1$.
///
/// Discretizes $\partial_t u = \Delta_H u = X^2 u + Y^2 u$ on a uniform grid $[-L, L]^3$.
#[derive(Debug, Clone)]
pub struct HypoellipticDiffusion3D {
    /// Grid resolution along each axis $N$.
    pub n: usize,
    /// Domain half-length $L$ (domain is $[-L, L]^3$).
    pub l: f64,
    /// Grid spacing $h = 2 L / (N - 1)$.
    pub h: f64,
    /// Density / scalar field values flattened $[N_x, N_y, N_z]$.
    pub grid: Vec<f64>,
}

impl HypoellipticDiffusion3D {
    /// Creates a 3D grid with resolution $N \times N \times N$ over $[-L, L]^3$.
    pub fn new(n: usize, l: f64) -> Result<Self> {
        if n < 5 {
            return Err(Error::Msg(format!(
                "Grid resolution n must be >= 5, got {n}"
            )));
        }
        if l <= 0.0 {
            return Err(Error::Msg(format!(
                "Domain half-length l must be > 0, got {l}"
            )));
        }
        let h = (2.0 * l) / ((n - 1) as f64);
        let cells = crate::limits::check_elements("diffusion grid", &[n, n, n])?;
        let grid = crate::layout::try_filled(cells, 0.0)?;
        Ok(Self { n, l, h, grid })
    }

    /// Linear index from 3D coordinates $(i, j, k)$.
    #[inline]
    pub fn idx(&self, i: usize, j: usize, k: usize) -> usize {
        (i * self.n + j) * self.n + k
    }

    /// Real spatial coordinate $(x, y, z)$ from grid index $(i, j, k)$.
    pub fn coord(&self, i: usize, j: usize, k: usize) -> (f64, f64, f64) {
        let x = -self.l + (i as f64) * self.h;
        let y = -self.l + (j as f64) * self.h;
        let z = -self.l + (k as f64) * self.h;
        (x, y, z)
    }

    /// Initializes grid with a normalized Gaussian bump centered at origin.
    pub fn init_gaussian(&mut self, sigma: f64) -> Result<()> {
        if !sigma.is_finite() || sigma <= 0.0 {
            return Err(Error::Msg(format!(
                "init_gaussian requires finite sigma > 0, got {sigma}"
            )));
        }
        let s2 = 2.0 * sigma * sigma;
        let mut total = 0.0;
        for i in 0..self.n {
            for j in 0..self.n {
                for k in 0..self.n {
                    let (x, y, z) = self.coord(i, j, k);
                    let val = (-((x * x + y * y + z * z) / s2)).exp();
                    let id = self.idx(i, j, k);
                    self.grid[id] = val;
                    total += val;
                }
            }
        }
        if total > 0.0 {
            let norm = self.h.powi(3) * total;
            for val in &mut self.grid {
                *val /= norm;
            }
        }
        Ok(())
    }

    /// Computes the exact discrete sub-Laplacian $\Delta_H u$ at interior node $(i, j, k)$:
    ///
    /// $$\Delta_H u = u_{xx} + u_{yy} + (x u_{yz} - y u_{xz}) + \frac{x^2 + y^2}{4} u_{zz}$$
    pub fn eval_sub_laplacian(&self, i: usize, j: usize, k: usize) -> Result<f64> {
        if self.n < 3
            || i == 0
            || j == 0
            || k == 0
            || i >= self.n - 1
            || j >= self.n - 1
            || k >= self.n - 1
        {
            return Err(Error::Msg(format!(
                "eval_sub_laplacian: ({i}, {j}, {k}) is not an interior node of an n={} grid",
                self.n
            )));
        }
        let need = self
            .n
            .checked_mul(self.n)
            .and_then(|v| v.checked_mul(self.n))
            .ok_or_else(|| Error::Msg("eval_sub_laplacian: grid size overflows".into()))?;
        if self.grid.len() < need {
            return Err(Error::Msg(format!(
                "eval_sub_laplacian: grid holds {} values, need {need}",
                self.grid.len()
            )));
        }
        let (x, y, _) = self.coord(i, j, k);
        let inv_h2 = 1.0 / (self.h * self.h);
        let inv_4h2 = 1.0 / (4.0 * self.h * self.h);

        let u_c = self.grid[self.idx(i, j, k)];

        // u_xx = (u_{i+1, j, k} - 2 u_c + u_{i-1, j, k}) / h^2
        let u_xx = (self.grid[self.idx(i + 1, j, k)] - 2.0 * u_c
            + self.grid[self.idx(i - 1, j, k)])
            * inv_h2;

        // u_yy = (u_{i, j+1, k} - 2 u_c + u_{i, j-1, k}) / h^2
        let u_yy = (self.grid[self.idx(i, j + 1, k)] - 2.0 * u_c
            + self.grid[self.idx(i, j - 1, k)])
            * inv_h2;

        // u_zz = (u_{i, j, k+1} - 2 u_c + u_{i, j, k-1}) / h^2
        let u_zz = (self.grid[self.idx(i, j, k + 1)] - 2.0 * u_c
            + self.grid[self.idx(i, j, k - 1)])
            * inv_h2;

        // Mixed partial: u_xz = (u_{i+1, j, k+1} - u_{i+1, j, k-1} - u_{i-1, j, k+1} + u_{i-1, j, k-1}) / (4 h^2)
        let u_xz = (self.grid[self.idx(i + 1, j, k + 1)]
            - self.grid[self.idx(i + 1, j, k - 1)]
            - self.grid[self.idx(i - 1, j, k + 1)]
            + self.grid[self.idx(i - 1, j, k - 1)])
            * inv_4h2;

        // Mixed partial: u_yz = (u_{i, j+1, k+1} - u_{i, j+1, k-1} - u_{i, j-1, k+1} + u_{i, j-1, k-1}) / (4 h^2)
        let u_yz = (self.grid[self.idx(i, j + 1, k + 1)]
            - self.grid[self.idx(i, j + 1, k - 1)]
            - self.grid[self.idx(i, j - 1, k + 1)]
            + self.grid[self.idx(i, j - 1, k - 1)])
            * inv_4h2;

        Ok(u_xx + u_yy + (x * u_yz - y * u_xz) + 0.25 * (x * x + y * y) * u_zz)
    }

    /// Advances the hypoelliptic diffusion by one time step $\Delta t$ using forward Euler.
    pub fn step(&mut self, dt: f64) {
        if self.n < 3 {
            return;
        }
        let mut new_grid = self.grid.clone();
        for i in 1..(self.n - 1) {
            for j in 1..(self.n - 1) {
                for k in 1..(self.n - 1) {
                    let Ok(lap_h) = self.eval_sub_laplacian(i, j, k) else {
                        continue;
                    };
                    let id = self.idx(i, j, k);
                    new_grid[id] = self.grid[id] + dt * lap_h;
                }
            }
        }
        self.grid = new_grid;
    }

    /// Computes total discrete mass $\int u \, dV = \sum u \, h^3$.
    pub fn total_mass(&self) -> f64 {
        let sum: f64 = self.grid.iter().sum();
        sum * self.h.powi(3)
    }

    /// Computes the horizontal second moment $\langle x^2 + y^2 \rangle$.
    pub fn horizontal_variance(&self) -> f64 {
        let mut sum = 0.0;
        let mut mass = 0.0;
        for i in 0..self.n {
            for j in 0..self.n {
                for k in 0..self.n {
                    let (x, y, _) = self.coord(i, j, k);
                    let u = self.grid[self.idx(i, j, k)];
                    sum += (x * x + y * y) * u;
                    mass += u;
                }
            }
        }
        if mass > 0.0 { sum / mass } else { 0.0 }
    }

    /// Computes the vertical second moment $\langle z^2 \rangle$.
    pub fn vertical_variance(&self) -> f64 {
        let mut sum = 0.0;
        let mut mass = 0.0;
        for i in 0..self.n {
            for j in 0..self.n {
                for k in 0..self.n {
                    let (_, _, z) = self.coord(i, j, k);
                    let u = self.grid[self.idx(i, j, k)];
                    sum += z * z * u;
                    mass += u;
                }
            }
        }
        if mass > 0.0 { sum / mass } else { 0.0 }
    }
}
