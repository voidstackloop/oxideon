//! Belavin-Polyakov-Schwartz-Tyupkin (BPST) $SU(2)$ Yang-Mills Instantons & Donaldson Bogomolny Bounds (Belavin et al., 1975).
//!
//! ### 1. Self-Dual Yang-Mills Equations $F_{\mu\nu} = \star F_{\mu\nu}$ & Bogomolny Action Bound
//! On Euclidean $\mathbb{R}^4 \cup \{\infty\} \cong S^4$, the Yang-Mills action for an $SU(2)$ gauge field with
//! topological Pontryagin instanton number $k \in \mathbb{Z}$ satisfies the Bogomolny bound:
//! $$S_{\text{YM}} = \frac{1}{4 g^2} \int \operatorname{Tr}(F_{\mu\nu} F^{\mu\nu}) d^4x \ge \frac{8 \pi^2}{g^2} |k|$$
//! Saturated by the **BPST Instanton** ($k = 1$) of scale size $\rho > 0$ with field strength density:
//! $$\operatorname{Tr}(F_{\mu\nu} F^{\mu\nu})(r) = \frac{96 \rho^4}{(r^2 + \rho^2)^4}$$

use crate::error::{Error, Result};
use std::f64::consts::PI;

/// BPST $SU(2)$ Self-Dual Yang-Mills Instanton & Bogomolny Bound Engine.
#[derive(Debug, Clone, Copy)]
pub struct BpstYangMillsInstanton {
    /// Instanton scale size $\rho > 0$.
    pub scale_rho: f64,
    /// Yang-Mills coupling $g > 0$.
    pub coupling_g: f64,
}

impl BpstYangMillsInstanton {
    /// Creates a new BPST Yang-Mills instanton engine.
    pub fn new(scale_rho: f64, coupling_g: f64) -> Result<Self> {
        if scale_rho <= 0.0 || coupling_g <= 0.0 {
            return Err(Error::Msg(
                "scale_rho and coupling_g must be > 0".to_string(),
            ));
        }
        Ok(Self {
            scale_rho,
            coupling_g,
        })
    }

    /// Evaluates the Bogomolny minimal Yang-Mills action $S_k = \frac{8 \pi^2}{g^2} |k|$ for Pontryagin charge $k$.
    pub fn bogomolny_instanton_action(&self, pontryagin_k: i64) -> f64 {
        8.0 * PI * PI * (pontryagin_k.abs() as f64) / (self.coupling_g * self.coupling_g)
    }

    /// Evaluates the radial BPST Pontryagin topological density $q(r) = \frac{6 \rho^4}{\pi^2 (r^2 + \rho^2)^4}$.
    pub fn bpst_topological_density(&self, radius_r: f64) -> f64 {
        let r2_rho2 = radius_r * radius_r + self.scale_rho * self.scale_rho;
        6.0 * self.scale_rho.powi(4) / (PI * PI * r2_rho2.powi(4))
    }
}
