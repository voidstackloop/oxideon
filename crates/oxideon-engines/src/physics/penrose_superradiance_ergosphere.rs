//! Penrose Ergosphere Process, Christodoulou-Ruffini Irreducible Mass $M_{\mathrm{irr}}$ & Blandford-Znajek Jet Power (Penrose, 1969; Blandford & Znajek, 1977).
//!
//! ### 1. Christodoulou Irreducible Mass & Maximum Extractable Rotational Energy
//! By Hawking's Area Theorem ($A = 16 \pi M_{\mathrm{irr}}^2$ is non-decreasing), the irreducible mass of a Kerr black hole is:
//! $$M_{\mathrm{irr}} = M \sqrt{\frac{1 + \sqrt{1 - \chi^2}}{2}} \ge \frac{M}{\sqrt{2}}$$
//! allowing up to $1 - \frac{1}{\sqrt{2}} \approx 29.289\%$ of the total gravitational mass $M$ to be extracted via the Penrose process!

use crate::error::{Error, Result};

/// Penrose Ergosphere Energy Extraction & Blandford-Znajek Relativistic Jet Engine.
#[derive(Debug, Clone, Copy)]
pub struct BlandfordZnajekPenroseEngine {
    /// Total Kerr mass $M > 0$.
    pub mass_m: f64,
    /// Dimensionless spin $\chi \in [0, 1]$.
    pub spin_chi: f64,
}

impl BlandfordZnajekPenroseEngine {
    /// Creates a new Penrose-Blandford-Znajek engine.
    pub fn new(mass_m: f64, spin_chi: f64) -> Result<Self> {
        if mass_m <= 0.0 || !(0.0..=1.0).contains(&spin_chi) {
            return Err(Error::Msg(
                "mass_m > 0 and spin_chi in [0, 1] required".to_string(),
            ));
        }
        Ok(Self { mass_m, spin_chi })
    }

    /// Computes Christodoulou's irreducible mass $M_{\mathrm{irr}}$ and maximum extractable rotational energy $E_{\mathrm{rot}} = M - M_{\mathrm{irr}}$.
    pub fn irreducible_mass_and_rotational_energy(&self) -> (f64, f64) {
        let m_irr =
            self.mass_m * (0.5 * (1.0 + (1.0 - self.spin_chi * self.spin_chi).sqrt())).sqrt();
        (m_irr, self.mass_m - m_irr)
    }
}
