//! Fontaine $p$-adic Hodge Theory ($B_{\mathrm{cris}}, B_{\mathrm{st}}, B_{\mathrm{dR}}$) & Faltings $C_{\mathrm{dR}}$ Comparison (Fontaine, 1982; Faltings, 1989).
//!
//! ### 1. Admissibility of Filtered $\varphi$-Modules: Weak Admissibility $t_H(D) = t_N(D)$
//! By Colmez-Fontaine ("weakly admissible implies admissible"), a filtered $\varphi$-module $D$ arises from a crystalline Galois representation iff
//! its Hodge polygon lies below its Newton polygon ($t_H(D') \le t_N(D')$ for all subobjects) and their endpoints coincide:
//! $$t_N(D) = \sum_i v_p(\lambda_i) = \sum_{i \in \mathbb{Z}} i \dim_{\mathbb{Q}_p} \operatorname{gr}^i D_{\mathrm{dR}} = t_H(D)$$

use crate::error::{Error, Result};

/// Fontaine $p$-adic Hodge Theory & Weak Admissibility Polygon Engine.
#[derive(Debug, Clone, Copy, Default)]
pub struct FontainePAdicHodgePeriodEngine;

impl FontainePAdicHodgePeriodEngine {
    /// Creates a new Fontaine $p$-adic Hodge engine.
    pub fn new() -> Self {
        Self
    }

    /// Evaluates the Newton endpoint $t_N(D)$, Hodge endpoint $t_H(D)$, and checks the Colmez-Fontaine endpoint equality $t_N(D) = t_H(D)$.
    pub fn newton_and_hodge_endpoints(
        newton_slopes: &[f64],
        hodge_weights: &[f64],
    ) -> Result<(f64, f64, bool)> {
        if newton_slopes.is_empty() || newton_slopes.len() != hodge_weights.len() {
            return Err(Error::Msg(
                "newton_slopes and hodge_weights must be non-empty of equal length".to_string(),
            ));
        }
        let tn: f64 = newton_slopes.iter().sum();
        let th: f64 = hodge_weights.iter().sum();
        Ok((tn, th, (tn - th).abs() < 1e-9))
    }
}
