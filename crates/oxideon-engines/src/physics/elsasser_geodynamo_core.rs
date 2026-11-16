//! Elsasser Magnetohydrodynamic Geodynamo Theory, Elsasser Number $\Lambda$ & Magnetic Reynolds Threshold (Elsasser, 1946).
//!
//! ### 1. Elsasser Number & Magnetostrophic Force Balance
//! In a rapidly rotating planetary liquid core of density $\rho$, rotation rate $\Omega$, electrical conductivity $\sigma$, and magnetic diffusivity $\eta = (\mu_0 \sigma)^{-1}$,
//! the ratio of Lorentz to Coriolis forces is the Elsasser number:
//! $$\Lambda = \frac{\sigma B^2}{2 \rho \Omega} = \frac{B^2}{2 \rho \mu_0 \eta \Omega}, \qquad R_m = \frac{U L}{\eta}$$

use crate::error::{Error, Result};

/// Elsasser Planetary Core Geodynamo & Magnetostrophic Balance Engine.
#[derive(Debug, Clone, Copy)]
pub struct ElsasserGeodynamoCoreEngine {
    /// Core fluid density $\rho > 0$ ($\mathrm{kg/m^3}$).
    pub density_rho: f64,
    /// Planetary rotation rate $\Omega > 0$ ($\mathrm{rad/s}$).
    pub omega_rot: f64,
    /// Electrical conductivity $\sigma > 0$ ($\mathrm{S/m}$).
    pub conductivity_sigma: f64,
}

impl Default for ElsasserGeodynamoCoreEngine {
    fn default() -> Self {
        Self {
            density_rho: 1.1e4,
            omega_rot: 7.2921e-5,
            conductivity_sigma: 5.0e5,
        }
    }
}

impl ElsasserGeodynamoCoreEngine {
    /// Creates a new Elsasser Geodynamo Core engine.
    pub fn new(density_rho: f64, omega_rot: f64, conductivity_sigma: f64) -> Result<Self> {
        if density_rho <= 0.0 || omega_rot <= 0.0 || conductivity_sigma <= 0.0 {
            return Err(Error::Msg(
                "density_rho, omega_rot, and conductivity_sigma must be > 0".to_string(),
            ));
        }
        Ok(Self {
            density_rho,
            omega_rot,
            conductivity_sigma,
        })
    }

    /// Evaluates the Elsasser number $\Lambda = \sigma B^2 / (2 \rho \Omega)$ and checks the magnetostrophic dynamo regime ($0.1 \le \Lambda \le 100$).
    pub fn elsasser_number(&self, magnetic_field_tesla: f64) -> Result<(f64, bool)> {
        if magnetic_field_tesla < 0.0 {
            return Err(Error::Msg("magnetic_field_tesla must be >= 0".to_string()));
        }
        let lambda = self.conductivity_sigma * magnetic_field_tesla.powi(2)
            / (2.0 * self.density_rho * self.omega_rot);
        Ok((lambda, (0.1..=100.0).contains(&lambda)))
    }
}
