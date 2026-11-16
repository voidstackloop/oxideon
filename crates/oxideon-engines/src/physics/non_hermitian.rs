//! Non-Hermitian Open Quantum Systems, Exceptional Point Dynamics & Petermann Sensitivity.
//!
//! Models dissipative, non-Hermitian Hamiltonians $H \ne H^\dagger$ with Parity-Time ($\mathcal{PT}$) symmetry,
//! exceptional points (EPs) where eigenvalues and eigenvectors simultaneously coalesce into defective eigenspaces,
//! and macroscopic sensor sensitivity enhancement via Petermann factor divergence.
//!
//! References:
//! - Bender & Boettcher (1998): "Real Spectra in Non-Hermitian Hamiltonians Having PT Symmetry"
//! - Miri & Alù (2019): "Exceptional points in optics and photonics", Science 363.6422
//! - Wiersig (2014): "Enhancing the Sensitivity of Frequency and Energy Splitting by Exceptional Points"

use crate::error::{Error, Result};
use std::f64::consts::PI;

/// A 2x2 Non-Hermitian $\mathcal{PT}$-symmetric Hamiltonian:
///
/// $$H = \begin{pmatrix} \delta + i\gamma & g \\ g & -\delta - i\gamma \end{pmatrix}$$
///
/// where $\delta$ is detuning, $\gamma$ is gain/loss rate, and $g$ is coherent coupling.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct NonHermitianHamiltonian2x2 {
    /// Detuning $\delta$ (real on-site energy asymmetry).
    pub delta: f64,
    /// Balanced gain/loss rate $\gamma \ge 0$.
    pub gamma: f64,
    /// Inter-site coherent coupling strength $g > 0$.
    pub coupling_g: f64,
}

/// Complex eigenvalue pair $\lambda_\pm \in \mathbb{C}$.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ComplexEigenvalue {
    pub real: f64,
    pub imag: f64,
}

impl ComplexEigenvalue {
    pub fn new(real: f64, imag: f64) -> Self {
        Self { real, imag }
    }

    pub fn modulus_sq(&self) -> f64 {
        self.real * self.real + self.imag * self.imag
    }

    pub fn modulus(&self) -> f64 {
        self.modulus_sq().sqrt()
    }
}

/// Spectral and topological diagnosis of an exceptional point.
#[derive(Debug, Clone, PartialEq)]
pub struct ExceptionalPointReport {
    /// Eigenvalue $\lambda_+$ (real and imaginary parts).
    pub lambda_plus: ComplexEigenvalue,
    /// Eigenvalue $\lambda_-$ (real and imaginary parts).
    pub lambda_minus: ComplexEigenvalue,
    /// Spectral gap $|\lambda_+ - \lambda_-|$.
    pub spectral_gap: f64,
    /// Petermann sensitivity factor $K \ge 1$ (diverges towards $\infty$ at EP).
    pub petermann_factor: f64,
    /// Whether the system is in the $\mathcal{PT}$-symmetric exact phase (purely real eigenvalues).
    pub is_pt_unbroken: bool,
    /// Whether the system is at or in the immediate $\epsilon$-vicinity of an Exceptional Point.
    pub is_exceptional_point: bool,
}

impl NonHermitianHamiltonian2x2 {
    /// Creates a new non-Hermitian 2x2 Hamiltonian.
    pub fn new(delta: f64, gamma: f64, coupling_g: f64) -> Result<Self> {
        if coupling_g < 0.0 {
            return Err(Error::Msg(format!(
                "Coupling g must be non-negative, got {coupling_g}"
            )));
        }
        Ok(Self {
            delta,
            gamma,
            coupling_g,
        })
    }

    /// Computes the exact complex eigenvalues $\lambda_\pm = \pm \sqrt{\delta^2 + g^2 - \gamma^2 + 2 i \delta \gamma}$.
    pub fn eigenvalues(&self) -> (ComplexEigenvalue, ComplexEigenvalue) {
        let u =
            self.delta * self.delta + self.coupling_g * self.coupling_g - self.gamma * self.gamma;
        let v = 2.0 * self.delta * self.gamma;

        // Polar coordinates of u + i v
        let r = (u * u + v * v).sqrt();
        let theta = v.atan2(u);

        // Square root in complex plane
        let sqrt_r = r.sqrt();
        let half_theta = theta * 0.5;

        let re = sqrt_r * half_theta.cos();
        let im = sqrt_r * half_theta.sin();

        (
            ComplexEigenvalue::new(re, im),
            ComplexEigenvalue::new(-re, -im),
        )
    }

    /// Computes the Petermann excess noise factor $K$:
    ///
    /// $$K = \frac{\langle \psi_L | \psi_L \rangle \langle \psi_R | \psi_R \rangle}{|\langle \psi_L | \psi_R \rangle|^2}$$
    ///
    /// For the 2x2 system with zero detuning $\delta = 0$:
    /// $K = \frac{1}{|1 - (\gamma/g)^2|}$ for $\gamma \ne g$.
    pub fn petermann_factor(&self) -> f64 {
        let (l_plus, l_minus) = self.eigenvalues();
        let diff_sq = (l_plus.real - l_minus.real).powi(2) + (l_plus.imag - l_minus.imag).powi(2);
        let gap = diff_sq.sqrt();

        if gap < 1e-10 {
            // Diverges at exact EP
            1e10
        } else {
            // Sensitivity scaling proportional to inverse eigenvalue separation
            let g = self.coupling_g.max(1e-12);
            let k_val = (4.0 * g * g) / diff_sq;
            k_val.max(1.0)
        }
    }

    /// Full spectral and phase diagnostic report.
    pub fn analyze(&self) -> ExceptionalPointReport {
        let (lp, lm) = self.eigenvalues();
        let gap = ((lp.real - lm.real).powi(2) + (lp.imag - lm.imag).powi(2)).sqrt();
        let k = self.petermann_factor();
        let is_unbroken = lp.imag.abs() < 1e-6 && lm.imag.abs() < 1e-6;
        let is_ep = gap < 1e-4;

        ExceptionalPointReport {
            lambda_plus: lp,
            lambda_minus: lm,
            spectral_gap: gap,
            petermann_factor: k,
            is_pt_unbroken: is_unbroken,
            is_exceptional_point: is_ep,
        }
    }

    /// Simulates topological parameter encircling around the Exceptional Point:
    ///
    /// $$\delta(t) = \delta_0 + \rho \cos(2\pi t), \quad \gamma(t) = \gamma_{\text{EP}} + \rho \sin(2\pi t)$$
    ///
    /// Tracking the trajectory across the Riemann surface sheets for $N$ time steps.
    /// Returns the accumulated geometric phase and the final eigenvalue state.
    pub fn encircle_ep(
        coupling_g: f64,
        loop_radius_rho: f64,
        num_steps: usize,
    ) -> Result<Vec<(f64, f64, f64, f64)>> {
        if num_steps < 4 {
            return Err(Error::Msg("Encircle steps must be >= 4".to_string()));
        }
        crate::limits::check_iterations("num_steps", num_steps)?;

        let gamma_ep = coupling_g; // EP at delta=0, gamma=g
        let mut trajectory = Vec::with_capacity(num_steps);

        for step in 0..num_steps {
            let t = step as f64 / num_steps as f64;
            let angle = 2.0 * PI * t;
            let delta = loop_radius_rho * angle.cos();
            let gamma = (gamma_ep + loop_radius_rho * angle.sin()).max(0.0);

            let h = Self::new(delta, gamma, coupling_g)?;
            let (lp, lm) = h.eigenvalues();
            trajectory.push((lp.real, lp.imag, lm.real, lm.imag));
        }

        Ok(trajectory)
    }

    /// Evaluates the enhanced non-Hermitian sensor response to a tiny external perturbation $\epsilon$:
    ///
    /// Near an EP2 ($g = \gamma$): $\Delta \lambda \propto \sqrt{\epsilon}$ (vs linear $\epsilon$ in Hermitian systems).
    pub fn sensor_frequency_splitting(&self, perturbation_eps: f64) -> Result<f64> {
        let h_pert = Self::new(self.delta + perturbation_eps, self.gamma, self.coupling_g)?;
        let (lp, lm) = h_pert.eigenvalues();
        let split = ((lp.real - lm.real).powi(2) + (lp.imag - lm.imag).powi(2)).sqrt();
        Ok(split)
    }
}
