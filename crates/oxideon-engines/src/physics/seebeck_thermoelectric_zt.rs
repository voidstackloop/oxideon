//! Seebeck-Peltier Thermoelectric Figure of Merit $Z T$ & Maximum Conversion Efficiency (Ioffe, 1957; Goldsmid, 1964).
//!
//! ### 1. Dimensionless Figure of Merit $Z T$ & Reduced Carnot Factor
//! Given Seebeck coefficient $S$, electrical conductivity $\sigma$, thermal conductivity $\kappa = \kappa_e + \kappa_L$, and mean temperature $T$:
//! $$Z T = \frac{S^2 \sigma T}{\kappa}, \qquad \eta_{\mathrm{max}} = \left(1 - \frac{T_C}{T_H}\right) \frac{\sqrt{1 + Z T} - 1}{\sqrt{1 + Z T} + T_C / T_H}$$

use crate::error::{Error, Result};

/// Seebeck-Peltier Thermoelectric $Z T$ & Conversion Efficiency Engine.
#[derive(Debug, Clone, Copy)]
pub struct ThermoelectricFigureOfMeritZT {
    /// Seebeck coefficient $S$ ($\mathrm{V/K}$).
    pub seebeck_s: f64,
    /// Electrical conductivity $\sigma > 0$ ($\mathrm{S/m}$).
    pub sigma_elec: f64,
    /// Total thermal conductivity $\kappa > 0$ ($\mathrm{W/(m \cdot K)}$).
    pub kappa_therm: f64,
}

impl Default for ThermoelectricFigureOfMeritZT {
    fn default() -> Self {
        Self {
            seebeck_s: 2.0e-4,
            sigma_elec: 1.0e5,
            kappa_therm: 1.5,
        }
    }
}

impl ThermoelectricFigureOfMeritZT {
    /// Creates a new Thermoelectric Figure of Merit $Z T$ engine.
    pub fn new(seebeck_s: f64, sigma_elec: f64, kappa_therm: f64) -> Result<Self> {
        if seebeck_s == 0.0 || sigma_elec <= 0.0 || kappa_therm <= 0.0 {
            return Err(Error::Msg(
                "seebeck_s != 0 and sigma_elec, kappa_therm > 0 required".to_string(),
            ));
        }
        Ok(Self {
            seebeck_s,
            sigma_elec,
            kappa_therm,
        })
    }

    /// Evaluates $Z T_{\mathrm{mean}}$ and maximum thermoelectric generator efficiency $\eta_{\mathrm{max}}$ between cold reservoir $T_C$ and hot reservoir $T_H > T_C$.
    pub fn zt_and_max_efficiency(&self, t_cold: f64, t_hot: f64) -> Result<(f64, f64)> {
        if t_cold <= 0.0 || t_hot <= t_cold {
            return Err(Error::Msg("0 < t_cold < t_hot required".to_string()));
        }
        let t_mean = 0.5 * (t_cold + t_hot);
        let zt = self.seebeck_s.powi(2) * self.sigma_elec * t_mean / self.kappa_therm;
        let carnot = 1.0 - t_cold / t_hot;
        let m = (1.0 + zt).sqrt();
        let eta_max = carnot * (m - 1.0) / (m + t_cold / t_hot);
        Ok((zt, eta_max))
    }
}
