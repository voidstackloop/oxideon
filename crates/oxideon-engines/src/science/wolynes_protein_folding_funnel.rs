//! Bryngelson-Onuchic-Wolynes Protein Folding Energy Landscape Funnel & Frustration Ratio $T_f / T_g$ (Bryngelson & Wolynes, 1987; Onuchic, Luthey-Schulten & Wolynes, 1997).
//!
//! ### 1. Random Energy Model Glass Transition $T_g$ & Foldability Criterion $T_f / T_g > 1$
//! For a polypeptide with native energy stability gap $\Delta E$, ruggedness variance $\Delta E_r^2$, and configurational entropy $S_0 = N s_0$:
//! $$k_B T_g = \sqrt{\frac{\Delta E_r^2}{2 S_0}}, \qquad \frac{T_f}{T_g} \approx \frac{\Delta E}{\Delta E_r \sqrt{2 S_0}} = \Lambda_{\mathrm{funnel}}$$

use crate::error::{Error, Result};

/// Bryngelson-Onuchic-Wolynes Protein Folding Funnel & Minimal Frustration Engine.
#[derive(Debug, Clone, Copy)]
pub struct WolynesProteinFoldingFunnel {
    /// Native stability energy gap $\Delta E > 0$.
    pub stability_gap_de: f64,
    /// Non-native ruggedness standard deviation $\Delta E_r > 0$.
    pub ruggedness_der: f64,
}

impl Default for WolynesProteinFoldingFunnel {
    fn default() -> Self {
        Self {
            stability_gap_de: 20.0,
            ruggedness_der: 2.0,
        }
    }
}

impl WolynesProteinFoldingFunnel {
    /// Creates a new Wolynes Protein Folding Funnel engine.
    pub fn new(stability_gap_de: f64, ruggedness_der: f64) -> Result<Self> {
        if stability_gap_de <= 0.0 || ruggedness_der <= 0.0 {
            return Err(Error::Msg(
                "stability_gap_de and ruggedness_der must be > 0".to_string(),
            ));
        }
        Ok(Self {
            stability_gap_de,
            ruggedness_der,
        })
    }

    /// Evaluates the glass temperature $T_g$, folding temperature $T_f = \Delta E / S_0$, and foldability ratio $T_f / T_g$ given configurational entropy $S_0 > 0$.
    pub fn glass_temp_and_foldability_ratio(
        &self,
        config_entropy_s0: f64,
    ) -> Result<(f64, f64, f64, bool)> {
        if config_entropy_s0 <= 0.0 {
            return Err(Error::Msg("config_entropy_s0 must be > 0".to_string()));
        }
        let t_g = self.ruggedness_der / (2.0 * config_entropy_s0).sqrt();
        let t_f = self.stability_gap_de / config_entropy_s0;
        let ratio = t_f / t_g;
        Ok((t_g, t_f, ratio, ratio > 1.5))
    }
}
