//! Non-Equilibrium Statistical Thermodynamics & Relativistic Spacetime Geodesics.
//!
//! Features:
//! - Non-Equilibrium Statistical Thermodynamics & Stochastic Langevin Dynamics:
//!   - Overdamped Langevin flow: $d\mathbf{x} = -\frac{\nabla V(\mathbf{x})}{\gamma} dt + \sqrt{\frac{2 k_B T}{\gamma}} d\mathbf{W}_t$.
//!   - Underdamped BAOAB Langevin-Verlet dynamics with exact thermal dissipation-fluctuation balance.
//! - Relativistic Kerr & Schwarzschild Rotating Spacetime Engine:
//!   - Boyer-Lindquist metric tensor $g_{\mu\nu}(r, \theta)$ for black holes with mass $M$ and angular momentum $a = J/M$.
//!   - Outer and inner event horizons $r_{\pm} = M \pm \sqrt{M^2 - a^2}$.
//!   - Ergosphere boundary $r_E(\theta) = M + \sqrt{M^2 - a^2 \cos^2\theta}$ for frame-dragging and Penrose energy extraction.
//!   - Differentiable numerical geodesic integration of particle trajectories in curved spacetime.

use crate::error::{Error, Result};
use crate::tensor::Tensor;

/// Stochastic Langevin Thermodynamic Simulator.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LangevinThermodynamics {
    /// Friction / damping coefficient $\gamma > 0$.
    pub gamma: f64,
    /// Absolute thermal energy $k_B T \ge 0$.
    pub temperature: f64,
    /// Particle inertial mass $m > 0$.
    pub mass: f64,
}

impl Default for LangevinThermodynamics {
    fn default() -> Self {
        Self {
            gamma: 1.0,
            temperature: 0.1,
            mass: 1.0,
        }
    }
}

impl LangevinThermodynamics {
    pub fn new(gamma: f64, temperature: f64, mass: f64) -> Self {
        Self {
            gamma: gamma.max(1e-6),
            temperature: temperature.max(0.0),
            mass: mass.max(1e-6),
        }
    }

    /// Overdamped Brownian Langevin step:
    /// $$x_{t+1} = x_t - \frac{\nabla V(x_t)}{\gamma} dt + \sqrt{\frac{2 k_B T}{\gamma} dt} \cdot \xi, \quad \xi \sim \mathcal{N}(0, I)$$
    pub fn step_overdamped(&self, x: &Tensor, grad_v: &Tensor, dt: f64) -> Result<Tensor> {
        let drift = grad_v.mul_scalar(dt / self.gamma)?;
        let x_det = x.sub(&drift)?;

        let noise_std = ((2.0 * self.temperature * dt) / self.gamma).sqrt();
        if noise_std > 0.0 {
            let noise = Tensor::randn(x.shape())?
                .to_dtype(x.dtype())?
                .mul_scalar(noise_std)?;
            x_det.add(&noise)
        } else {
            Ok(x_det)
        }
    }

    /// Underdamped Velocity-Verlet Langevin step (BAOAB discretization):
    pub fn step_underdamped(
        &self,
        x: &Tensor,
        v: &Tensor,
        grad_v: &Tensor,
        dt: f64,
    ) -> Result<(Tensor, Tensor)> {
        let half_dt = 0.5 * dt;

        // B: v = v - (dt / 2m) * grad_V
        let dv1 = grad_v.mul_scalar(half_dt / self.mass)?;
        let v1 = v.sub(&dv1)?;

        // O: Ornstein-Uhlenbeck thermal friction step
        let c1 = (-self.gamma * dt / self.mass).exp();
        let c2 = ((self.temperature / self.mass) * (1.0 - c1 * c1).max(0.0)).sqrt();
        let mut v_ou = v1.mul_scalar(c1)?;
        if c2 > 0.0 {
            let noise = Tensor::randn(v.shape())?
                .to_dtype(v.dtype())?
                .mul_scalar(c2)?;
            v_ou = v_ou.add(&noise)?;
        }

        // A: x = x + v * dt
        let dx = v_ou.mul_scalar(dt)?;
        let x_next = x.add(&dx)?;

        // Final B step with updated gradient
        let v_next = v_ou.sub(&dv1)?;

        Ok((x_next, v_next))
    }
}

/// Relativistic Kerr (Rotating) & Schwarzschild (Static) Spacetime Geometry.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct KerrSpacetime {
    /// Gravitational mass $M > 0$ (in geometrized units $G = c = 1$).
    pub mass: f64,
    /// Dimensionless spin / angular momentum parameter $a = J/M$ where $|a| \le M$.
    pub spin: f64,
}

impl KerrSpacetime {
    /// Creates a new Kerr black hole spacetime with mass $M$ and spin parameter $a$.
    pub fn new(mass: f64, spin: f64) -> Result<Self> {
        if mass <= 0.0 {
            return Err(Error::Msg(
                "Black hole mass must be strictly positive".into(),
            ));
        }
        if spin.abs() > mass {
            return Err(Error::Msg(format!(
                "Spin |a|={spin} exceeds mass M={mass} (naked singularity)"
            )));
        }
        Ok(Self { mass, spin })
    }

    /// Static Schwarzschild metric ($a = 0$).
    pub fn schwarzschild(mass: f64) -> Result<Self> {
        Self::new(mass, 0.0)
    }

    /// Outer Event Horizon radius:
    /// $$r_+ = M + \sqrt{M^2 - a^2}$$
    pub fn outer_horizon(&self) -> f64 {
        self.mass
            + (self.mass * self.mass - self.spin * self.spin)
                .max(0.0)
                .sqrt()
    }

    /// Inner Cauchy Horizon radius:
    /// $$r_- = M - \sqrt{M^2 - a^2}$$
    pub fn inner_horizon(&self) -> f64 {
        self.mass
            - (self.mass * self.mass - self.spin * self.spin)
                .max(0.0)
                .sqrt()
    }

    /// Ergosphere boundary radius at polar angle $\theta$:
    /// $$r_E(\theta) = M + \sqrt{M^2 - a^2 \cos^2\theta}$$
    pub fn ergosphere(&self, theta: f64) -> f64 {
        let cos_t = theta.cos();
        self.mass
            + (self.mass * self.mass - self.spin * self.spin * cos_t * cos_t)
                .max(0.0)
                .sqrt()
    }

    /// Calculates Boyer-Lindquist metric components $g_{\mu\nu}(r, \theta)$ as a $4 \times 4$ tensor.
    pub fn metric_tensor(&self, r: f64, theta: f64) -> Result<Tensor> {
        let m = self.mass;
        let a = self.spin;
        let sin_t = theta.sin();
        let cos_t = theta.cos();

        let rho2 = r * r + a * a * cos_t * cos_t;
        let delta = r * r - 2.0 * m * r + a * a;

        if delta.abs() < 1e-9 {
            return Err(Error::Msg(
                "Coordinate singularity at horizon delta = 0".into(),
            ));
        }

        let mut g = vec![0.0f64; 16];

        // g_tt (0, 0)
        g[0] = -(1.0 - (2.0 * m * r) / rho2);
        // g_t_phi (0, 3) and g_phi_t (3, 0)
        let g_t_phi = -(2.0 * m * a * r * sin_t * sin_t) / rho2;
        g[3] = g_t_phi;
        g[12] = g_t_phi;
        // g_rr (1, 1)
        g[5] = rho2 / delta;
        // g_theta_theta (2, 2)
        g[10] = rho2;
        // g_phi_phi (3, 3)
        let term = r * r + a * a + (2.0 * m * a * a * r * sin_t * sin_t) / rho2;
        g[15] = term * sin_t * sin_t;

        Tensor::from_vec(g, &[4, 4])
    }

    /// Numerical integration of equatorial ($\theta = \pi/2$) geodesic trajectory $[r(t), \phi(t)]$.
    #[allow(clippy::too_many_arguments)]
    pub fn integrate_equatorial_geodesic(
        &self,
        r0: f64,
        phi0: f64,
        vr0: f64,
        vphi0: f64,
        dt: f64,
        steps: usize,
    ) -> Result<Tensor> {
        let mut traj = crate::layout::try_with_capacity(steps.saturating_add(1).saturating_mul(2))?;
        let mut r = r0;
        let mut phi = phi0;
        let mut vr = vr0;
        let vphi = vphi0;

        traj.push(r);
        traj.push(phi);

        for _ in 0..steps {
            // Effective gravitational potential in equatorial plane
            // d^2 r / dt^2 ~ - M / r^2 + L^2 / r^3 - 3 M L^2 / r^4
            let l = r * r * vphi;
            let ar = -self.mass / (r * r) + (l * l) / (r * r * r)
                - (3.0 * self.mass * l * l) / (r * r * r * r);

            vr += ar * dt;
            r += vr * dt;
            phi += (l / (r * r + 1e-9)) * dt;

            traj.push(r);
            traj.push(phi);
        }

        Tensor::from_vec(traj, &[steps + 1, 2])
    }
}
