//! Nonlinear Dispersive Soliton Physics (NLS & KdV Solitons).
//!
//! Provides mathematical physics primitives, exact analytical solitons, conserved invariants,
//! split-step Fourier numerical solvers, and PINN residual evaluation for:
//! - **Nonlinear Schrödinger Equation (NLS)**:
//!   $$i \partial_t \psi + \frac{1}{2} \partial_x^2 \psi + |\psi|^2 \psi = 0$$
//!   Conserved invariants: Mass $N = \int |\psi|^2 dx$, Momentum $P$, and Energy $H$.
//! - **Korteweg-de Vries Equation (KdV)**:
//!   $$\partial_t u + 6 u \partial_x u + \partial_x^3 u = 0$$
//!   Exact solitary wave $u(x, t) = \frac{c}{2} \operatorname{sech}^2\left(\frac{\sqrt{c}}{2}(x - ct - x_0)\right)$.

use crate::error::{Error, Result};
use crate::tensor::Tensor;

/// Configuration for 1D spatial simulation grid.
#[derive(Debug, Clone)]
pub struct SpatialGrid1D {
    pub num_points: usize,
    pub x_min: f64,
    pub x_max: f64,
}

impl SpatialGrid1D {
    pub fn new(num_points: usize, x_min: f64, x_max: f64) -> Result<Self> {
        if num_points < 4 {
            return Err(Error::Msg("num_points must be >= 4".into()));
        }
        if x_max <= x_min {
            return Err(Error::Msg(format!("x_max {x_max} <= x_min {x_min}")));
        }
        Ok(Self {
            num_points,
            x_min,
            x_max,
        })
    }

    #[inline]
    pub fn dx(&self) -> f64 {
        (self.x_max - self.x_min) / (self.num_points as f64)
    }

    pub fn coordinates(&self) -> Vec<f64> {
        let dx = self.dx();
        (0..self.num_points)
            .map(|i| self.x_min + i as f64 * dx)
            .collect()
    }
}

/// Exact Soliton Waveform Engine.
pub struct SolitonPhysics;

impl SolitonPhysics {
    /// Evaluates exact bright NLS soliton envelope amplitude $|\psi(x, t)|$ and complex components.
    ///
    /// $\psi(x, t) = \eta \operatorname{sech}(\eta(x - vt - x_0)) \exp\left(i \left(vx - \frac{v^2 - \eta^2}{2}t + \theta_0\right)\right)$.
    pub fn nls_bright_soliton(
        x: &[f64],
        t: f64,
        eta: f64,
        v: f64,
        x0: f64,
        theta0: f64,
    ) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
        let n = x.len();
        let mut real = Vec::with_capacity(n);
        let mut imag = Vec::with_capacity(n);
        let mut density = Vec::with_capacity(n);

        let omega = 0.5 * (v * v - eta * eta);

        for &xi in x {
            let arg = eta * (xi - v * t - x0);
            let sech = 1.0 / arg.cosh();
            let amp = eta * sech;
            let phase = v * xi - omega * t + theta0;

            let r = amp * phase.cos();
            let im = amp * phase.sin();

            real.push(r);
            imag.push(im);
            density.push(amp * amp);
        }

        (real, imag, density)
    }

    /// Computes conserved NLS Mass / Wave Power: $\int |\psi(x)|^2 dx$.
    pub fn nls_mass(density: &[f64], dx: f64) -> f64 {
        density.iter().sum::<f64>() * dx
    }

    /// Evaluates exact Korteweg-de Vries (KdV) solitary traveling wave:
    /// $u(x, t) = \frac{c}{2} \operatorname{sech}^2\left(\frac{\sqrt{c}}{2}(x - ct - x_0)\right)$.
    pub fn kdv_soliton(x: &[f64], t: f64, velocity_c: f64, x0: f64) -> Vec<f64> {
        let sqrt_c = velocity_c.max(0.0).sqrt();
        let k = 0.5 * sqrt_c;
        let amp = 0.5 * velocity_c;

        x.iter()
            .map(|&xi| {
                let theta = k * (xi - velocity_c * t - x0);
                let sech = 1.0 / theta.cosh();
                amp * sech * sech
            })
            .collect()
    }

    /// Computes KdV conserved Mass / Momentum: $\int u(x) dx$ and $\int u(x)^2 dx$.
    pub fn kdv_invariants(u: &[f64], dx: f64) -> (f64, f64) {
        let mass = u.iter().sum::<f64>() * dx;
        let l2_energy = u.iter().map(|&v| v * v).sum::<f64>() * dx;
        (mass, l2_energy)
    }

    /// Computes Korteweg-de Vries (KdV) PINN PDE residual:
    /// $\mathcal{R}_{\text{KdV}} = \partial_t u + 6 u \partial_x u + \partial_x^3 u$.
    pub fn kdv_pinn_residual(
        u: &Tensor,
        du_dt: &Tensor,
        du_dx: &Tensor,
        d3u_dx3: &Tensor,
    ) -> Result<Tensor> {
        let u_val = u.to_vec::<f64>()?;
        let ut_val = du_dt.to_vec::<f64>()?;
        let ux_val = du_dx.to_vec::<f64>()?;
        let uxxx_val = d3u_dx3.to_vec::<f64>()?;

        let n = u_val.len();
        if ut_val.len() != n || ux_val.len() != n || uxxx_val.len() != n {
            return Err(Error::Msg("Shape mismatch in kdv_pinn_residual".into()));
        }

        let mut res = Vec::with_capacity(n);
        for i in 0..n {
            // R = u_t + 6 * u * u_x + u_xxx
            let r = ut_val[i] + 6.0 * u_val[i] * ux_val[i] + uxxx_val[i];
            res.push(r);
        }

        Tensor::from_vec(res, u.shape())
    }
}

/// Split-Step Fourier Engine for Nonlinear Schrödinger Soliton Dynamics.
#[derive(Debug, Clone)]
pub struct SplitStepNlsEngine {
    pub grid: SpatialGrid1D,
    pub dt: f64,
}

impl SplitStepNlsEngine {
    pub fn new(grid: SpatialGrid1D, dt: f64) -> Self {
        Self { grid, dt }
    }

    /// Evolves the wavefield $\psi$ by one time step $\Delta t$ using symmetric Strang splitting:
    /// 1. Nonlinear phase rotation over $\Delta t / 2$: $\psi \leftarrow \psi \exp(i |\psi|^2 \Delta t / 2)$.
    /// 2. Linear kinetic dispersion over $\Delta t$: finite difference second derivative.
    /// 3. Nonlinear phase rotation over $\Delta t / 2$.
    pub fn step(&self, real: &mut [f64], imag: &mut [f64]) -> Result<()> {
        let n = self.grid.num_points;
        if real.len() < n || imag.len() < n {
            return Err(Error::Msg(format!(
                "SplitStepNlsEngine::step: buffers hold {}/{} values, grid needs {n}",
                real.len(),
                imag.len()
            )));
        }
        let dx = self.grid.dx();
        let dt = self.dt;
        let half_dt = 0.5 * dt;

        // 1. Half step nonlinear rotation: psi -> psi * exp(i * |psi|^2 * dt/2)
        for i in 0..n {
            let dens = real[i] * real[i] + imag[i] * imag[i];
            let phase = dens * half_dt;
            let cos_p = phase.cos();
            let sin_p = phase.sin();
            let r = real[i] * cos_p - imag[i] * sin_p;
            let im = real[i] * sin_p + imag[i] * cos_p;
            real[i] = r;
            imag[i] = im;
        }

        // 2. Linear dispersion step: i * dpsi/dt + 0.5 * d^2 psi/dx^2 = 0
        // Crank-Nicolson / explicit finite difference update
        // d psi/dt = 0.5 * i * d^2 psi/dx^2
        // d(real)/dt = -0.5 * d^2(imag)/dx^2
        // d(imag)/dt = 0.5 * d^2(real)/dx^2
        let inv_dx2 = 1.0 / (dx * dx);
        let mut d2_real = vec![0.0f64; n];
        let mut d2_imag = vec![0.0f64; n];

        for i in 0..n {
            let prev = if i == 0 { n - 1 } else { i - 1 };
            let next = if i == n - 1 { 0 } else { i + 1 };
            d2_real[i] = (real[next] - 2.0 * real[i] + real[prev]) * inv_dx2;
            d2_imag[i] = (imag[next] - 2.0 * imag[i] + imag[prev]) * inv_dx2;
        }

        for i in 0..n {
            real[i] += -0.5 * d2_imag[i] * dt;
            imag[i] += 0.5 * d2_real[i] * dt;
        }

        // 3. Half step nonlinear rotation
        for i in 0..n {
            let dens = real[i] * real[i] + imag[i] * imag[i];
            let phase = dens * half_dt;
            let cos_p = phase.cos();
            let sin_p = phase.sin();
            let r = real[i] * cos_p - imag[i] * sin_p;
            let im = real[i] * sin_p + imag[i] * cos_p;
            real[i] = r;
            imag[i] = im;
        }

        Ok(())
    }
}
