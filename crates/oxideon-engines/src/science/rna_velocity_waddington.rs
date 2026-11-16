//! Single-Cell RNA Velocity Splicing Kinetics & Waddington Epigenetic Bifurcation Landscape (Waddington, 1957; La Manno et al., 2018).
//!
//! ### 1. Unspliced-to-Spliced Transcriptional Velocity
//! Given transcription rate $\alpha$, splicing rate $\beta > 0$, and mRNA degradation rate $\gamma > 0$,
//! unspliced $(u)$ and spliced $(s)$ abundances satisfy $\dot{u} = \alpha - \beta u$ and RNA velocity:
//! $$v_s = \frac{ds}{dt} = \beta u - \gamma s, \qquad \left(\frac{u}{s}\right)_{\mathrm{ss}} = \frac{\gamma}{\beta}$$

use crate::error::{Error, Result};

/// Single-Cell RNA Velocity & Waddington Epigenetic Landscape Engine.
#[derive(Debug, Clone, Copy)]
pub struct RnaVelocityWaddingtonLandscape {
    /// Splicing rate constant $\beta > 0$.
    pub splicing_beta: f64,
    /// Spliced mRNA degradation rate $\gamma > 0$.
    pub degradation_gamma: f64,
}

impl Default for RnaVelocityWaddingtonLandscape {
    fn default() -> Self {
        Self {
            splicing_beta: 1.0,
            degradation_gamma: 0.5,
        }
    }
}

impl RnaVelocityWaddingtonLandscape {
    /// Creates a new RNA Velocity & Waddington Landscape engine.
    pub fn new(splicing_beta: f64, degradation_gamma: f64) -> Result<Self> {
        if splicing_beta <= 0.0 || degradation_gamma <= 0.0 {
            return Err(Error::Msg(
                "splicing_beta and degradation_gamma must be > 0".to_string(),
            ));
        }
        Ok(Self {
            splicing_beta,
            degradation_gamma,
        })
    }

    /// Evaluates spliced RNA velocity $v_s = \beta u - \gamma s$ and transcriptional induction flag ($v_s > 0$) for unspliced count $u$ and spliced count $s$.
    pub fn rna_velocity_and_induction_state(
        &self,
        unspliced_u: f64,
        spliced_s: f64,
    ) -> Result<(f64, bool)> {
        if unspliced_u < 0.0 || spliced_s < 0.0 {
            return Err(Error::Msg(
                "Transcript counts u and s must be >= 0".to_string(),
            ));
        }
        let v_s = self.splicing_beta * unspliced_u - self.degradation_gamma * spliced_s;
        Ok((v_s, v_s > 0.0))
    }
}
