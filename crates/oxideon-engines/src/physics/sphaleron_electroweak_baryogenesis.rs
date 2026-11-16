//! Sakharov Baryogenesis Conditions, Electroweak Sphaleron Rate & $B - L$ Conversion (Sakharov, 1967; Kuzmin, Rubakov & Shaposhnikov, 1985).
//!
//! ### 1. Equilibrium Sphaleron Conversion Ratio
//! High-temperature electroweak sphalerons violate $B + L$ while conserving $B - L$. For $N_g$ fermion generations and $N_H$ Higgs doublets,
//! chemical equilibrium converts a primordial $B - L$ asymmetry into the final baryon asymmetry:
//! $$Y_B = \frac{8 N_g + 4 N_H}{22 N_g + 13 N_H} Y_{B - L}$$

use crate::error::{Error, Result};

/// Sakharov Electroweak Sphaleron & Leptogenesis-to-Baryogenesis Conversion Engine.
#[derive(Debug, Clone, Copy)]
pub struct ElectroweakSphaleronBaryogenesis {
    /// Number of fermion generations $N_g \ge 1$.
    pub num_generations_ng: usize,
    /// Number of Higgs doublets $N_H \ge 1$.
    pub num_higgs_doublets_nh: usize,
}

impl Default for ElectroweakSphaleronBaryogenesis {
    fn default() -> Self {
        Self {
            num_generations_ng: 3,
            num_higgs_doublets_nh: 1,
        }
    }
}

impl ElectroweakSphaleronBaryogenesis {
    /// Creates a new Electroweak Sphaleron Baryogenesis engine.
    pub fn new(num_generations_ng: usize, num_higgs_doublets_nh: usize) -> Result<Self> {
        if num_generations_ng == 0 || num_higgs_doublets_nh == 0 {
            return Err(Error::Msg("N_g >= 1 and N_H >= 1 required".to_string()));
        }
        Ok(Self {
            num_generations_ng,
            num_higgs_doublets_nh,
        })
    }

    /// Evaluates the Harvey-Turner sphaleron conversion coefficient $c_s = (8 N_g + 4 N_H)/(22 N_g + 13 N_H)$ (equals $28/79$ in the Standard Model) and final baryon yield $Y_B$.
    pub fn sphaleron_conversion_and_baryon_yield(&self, y_b_minus_l: f64) -> (f64, f64) {
        let ng = self.num_generations_ng as f64;
        let nh = self.num_higgs_doublets_nh as f64;
        let cs = (8.0 * ng + 4.0 * nh) / (22.0 * ng + 13.0 * nh);
        (cs, cs * y_b_minus_l)
    }
}
