//! Matsui-Satz $J/\psi$ Quarkonium Color-Debye Screening & Sequential Melting in QGP (Matsui & Satz, 1986; Karsch, Mehr & Satz, 1988).
//!
//! ### 1. Color-Screened Cornell Potential & Dissociation Debye Mass $m_D(T)$
//! In a deconfined QGP at temperature $T$, the heavy-quark $Q\bar{Q}$ potential is screened at Debye mass $m_D(T) = g T \sqrt{N_c / 3 + N_f / 6}$:
//! $$V(r, T) = -\frac{\alpha_{\mathrm{eff}}}{r} e^{-m_D r} + \frac{\sigma}{m_D} (1 - e^{-m_D r})$$
//! A quarkonium state of Bohr radius $r_B$ dissociates (melts) when the color screening length $r_D = m_D^{-1} \lesssim r_B$.

use crate::error::{Error, Result};

/// Matsui-Satz Quarkonium Color-Debye Screening & Melting Engine.
#[derive(Debug, Clone, Copy)]
pub struct MatsuiSatzQuarkoniumMelting {
    /// Number of colors $N_c \ge 2$.
    pub num_colors_nc: usize,
    /// Number of light quark flavors $N_f$.
    pub num_flavors_nf: usize,
}

impl Default for MatsuiSatzQuarkoniumMelting {
    fn default() -> Self {
        Self {
            num_colors_nc: 3,
            num_flavors_nf: 3,
        }
    }
}

impl MatsuiSatzQuarkoniumMelting {
    /// Creates a new Matsui-Satz Quarkonium Melting engine.
    pub fn new(num_colors_nc: usize, num_flavors_nf: usize) -> Result<Self> {
        if num_colors_nc < 2 {
            return Err(Error::Msg("num_colors_nc must be >= 2".to_string()));
        }
        Ok(Self {
            num_colors_nc,
            num_flavors_nf,
        })
    }

    /// Evaluates the perturbative QCD Debye mass $m_D(T)$ (in GeV), screening radius $r_D = 0.197327 / m_D$ (in fm), and checks if quarkonium of radius $r_Q$ (fm) melts ($r_D < r_Q$).
    pub fn debye_mass_radius_and_melting(
        &self,
        coupling_g: f64,
        temp_gev: f64,
        quarkonium_radius_fm: f64,
    ) -> Result<(f64, f64, bool)> {
        if coupling_g <= 0.0 || temp_gev <= 0.0 || quarkonium_radius_fm <= 0.0 {
            return Err(Error::Msg(
                "coupling_g, temp_gev, and quarkonium_radius_fm must be > 0".to_string(),
            ));
        }
        let factor =
            ((self.num_colors_nc as f64) / 3.0 + (self.num_flavors_nf as f64) / 6.0).sqrt();
        let m_d_gev = coupling_g * temp_gev * factor;
        let r_d_fm = 0.19732698 / m_d_gev;
        Ok((m_d_gev, r_d_fm, r_d_fm < quarkonium_radius_fm))
    }
}
