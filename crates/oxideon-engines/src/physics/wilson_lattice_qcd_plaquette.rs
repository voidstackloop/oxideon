//! Wilson Lattice Gauge Theory Plaquette Action & Strong-Coupling Area-Law String Tension (Wilson, 1974).
//!
//! ### 1. Wilson Action & Strong-Coupling String Tension $\sigma a^2$
//! For $SU(N_c)$ lattice gauge theory at inverse coupling $\beta = 2 N_c / g^2$, the strong-coupling expansion ($eta \ll 2 N_c$)
//! yields the confining Wilson loop area-law string tension:
//! $$\sigma a^2 = -\ln\!\left(\frac{\beta}{2 N_c^2}\right)$$

use crate::error::{Error, Result};

/// Wilson Lattice QCD Plaquette & Strong-Coupling Confinement Engine.
#[derive(Debug, Clone, Copy)]
pub struct WilsonLatticeGaugeQCD {
    /// Number of colors $N_c \ge 2$.
    pub num_colors_nc: usize,
    /// Inverse lattice gauge coupling $\beta = 2 N_c / g^2 > 0$.
    pub beta: f64,
}

impl Default for WilsonLatticeGaugeQCD {
    fn default() -> Self {
        Self {
            num_colors_nc: 3,
            beta: 1.5,
        }
    }
}

impl WilsonLatticeGaugeQCD {
    /// Creates a new Wilson Lattice Gauge QCD engine.
    pub fn new(num_colors_nc: usize, beta: f64) -> Result<Self> {
        if num_colors_nc < 2 || beta <= 0.0 || !beta.is_finite() {
            return Err(Error::Msg(
                "num_colors_nc >= 2 and beta > 0 required".to_string(),
            ));
        }
        Ok(Self {
            num_colors_nc,
            beta,
        })
    }

    /// Evaluates the strong-coupling string tension $\sigma a^2 = -\ln(\beta / (2 N_c^2))$ and $R \times T$ Wilson loop expectation $\langle W(R, T)\rangle = e^{-\sigma a^2 R T}$.
    pub fn strong_coupling_string_tension_and_wilson_loop(
        &self,
        r_plaquettes: usize,
        t_plaquettes: usize,
    ) -> Result<(f64, f64)> {
        let nc = self.num_colors_nc as f64;
        let ratio = self.beta / (2.0 * nc * nc);
        if ratio >= 1.0 {
            return Err(Error::Msg(
                "Strong-coupling regime requires beta < 2 N_c^2".to_string(),
            ));
        }
        let sigma_a2 = -ratio.ln();
        let area = (r_plaquettes * t_plaquettes) as f64;
        Ok((sigma_a2, (-sigma_a2 * area).exp()))
    }
}
