//! Hoffmann-Chinchilla & Kaplan Neural Scaling Laws $L(N, D) = E + A N^{-\alpha} + B D^{-\beta}$ (Kaplan et al., 2020; Hoffmann et al., 2022).
//!
//! ### 1. Chinchilla Compute-Optimal Frontier under FLOPs Constraint $C = 6 N D$
//! Minimizing pretraining cross-entropy loss $L(N, D) = E + \frac{A}{N^\alpha} + \frac{B}{D^\beta}$ subject to FLOPs budget
//! $C = 6 N D$ via Lagrange multipliers yields $\frac{\alpha A}{N^\alpha} = \frac{\beta B}{D^\beta}$ and equal scaling when $\alpha \approx \beta$:
//! $$N_{\mathrm{opt}}(C) = G \left(\frac{C}{6}\right)^{\frac{\beta}{\alpha + \beta}}, \qquad D_{\mathrm{opt}}(C) = G^{-1} \left(\frac{C}{6}\right)^{\frac{\alpha}{\alpha + \beta}}, \qquad G = \left(\frac{\alpha A}{\beta B}\right)^{\frac{1}{\alpha + \beta}}$$

use crate::error::{Error, Result};

/// Hoffmann-Chinchilla Compute-Optimal Neural Scaling Law Engine.
#[derive(Debug, Clone, Copy)]
pub struct ComputeOptimalChinchillaEngine {
    /// Irreducible entropy of natural text $E \approx 1.69$.
    pub irreducible_e: f64,
    /// Parameter scaling prefactor $A \approx 406.4$ and exponent $\alpha \approx 0.34$.
    pub a_and_alpha: (f64, f64),
    /// Data scaling prefactor $B \approx 410.7$ and exponent $\beta \approx 0.28$.
    pub b_and_beta: (f64, f64),
}

impl Default for ComputeOptimalChinchillaEngine {
    fn default() -> Self {
        Self {
            irreducible_e: 1.69,
            a_and_alpha: (406.4, 0.34),
            b_and_beta: (410.7, 0.28),
        }
    }
}

impl ComputeOptimalChinchillaEngine {
    /// Creates a new Chinchilla Compute-Optimal Scaling Law engine.
    pub fn new(irreducible_e: f64, a: f64, alpha: f64, b: f64, beta: f64) -> Result<Self> {
        if irreducible_e < 0.0 || a <= 0.0 || alpha <= 0.0 || b <= 0.0 || beta <= 0.0 {
            return Err(Error::Msg(
                "All Chinchilla scaling parameters must be > 0".to_string(),
            ));
        }
        Ok(Self {
            irreducible_e,
            a_and_alpha: (a, alpha),
            b_and_beta: (b, beta),
        })
    }

    /// Evaluates the parametric loss $L(N, D) = E + A N^{-\alpha} + B D^{-\beta}$ and compute-optimal allocation $(N_{\mathrm{opt}}, D_{\mathrm{opt}})$ for FLOPs $C = 6 N D$.
    pub fn loss_and_optimal_allocation(
        &self,
        params_n: f64,
        tokens_d: f64,
    ) -> Result<(f64, f64, f64)> {
        if params_n <= 0.0 || tokens_d <= 0.0 {
            return Err(Error::Msg("params_n and tokens_d must be > 0".to_string()));
        }
        let (a, alpha) = self.a_and_alpha;
        let (b, beta) = self.b_and_beta;
        let loss = self.irreducible_e + a * params_n.powf(-alpha) + b * tokens_d.powf(-beta);
        let c_div_6 = params_n * tokens_d;
        let g = ((alpha * a) / (beta * b)).powf(1.0 / (alpha + beta));
        let n_opt = g * c_div_6.powf(beta / (alpha + beta));
        let d_opt = c_div_6 / n_opt;
        Ok((loss, n_opt, d_opt))
    }
}
