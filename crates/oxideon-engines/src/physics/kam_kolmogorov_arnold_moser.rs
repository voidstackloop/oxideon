//! Kolmogorov-Arnold-Moser (KAM) Invariant Tori, Diophantine Small Divisors & Noble Golden-Mean Winding (Kolmogorov, 1954; Arnold, 1963; Moser, 1962).
//!
//! ### 1. Diophantine Non-Resonance Condition & Small-Divisor Norm
//! Under a small symplectic perturbation $H(I, \theta) = H_0(I) + \epsilon H_1(I, \theta)$, an invariant torus with
//! frequency vector $\boldsymbol{\omega} = \nabla H_0(I)$ survives if it satisfies the $(\gamma, \tau)$-Diophantine condition:
//! $$|\mathbf{k} \cdot \boldsymbol{\omega}| \ge \frac{\gamma}{\|\mathbf{k}\|_1^\tau} \qquad \forall \, \mathbf{k} \in \mathbb{Z}^n \setminus \{0\}$$
//! The most robust 2D KAM torus has winding ratio equal to the noble **Golden Mean** $\omega = \frac{\sqrt{5} - 1}{2} \approx 0.6180339887$.

use crate::error::{Error, Result};

/// Kolmogorov-Arnold-Moser (KAM) Diophantine Small-Divisor Engine.
#[derive(Debug, Clone, Copy)]
pub struct KamDiophantineTorusEngine {
    /// Diophantine constant $\gamma > 0$.
    pub gamma: f64,
    /// Diophantine exponent $\tau \ge 1$.
    pub tau: f64,
}

impl KamDiophantineTorusEngine {
    /// Creates a new KAM Diophantine Torus engine.
    pub fn new(gamma: f64, tau: f64) -> Result<Self> {
        if gamma <= 0.0 || tau < 1.0 {
            return Err(Error::Msg("gamma > 0 and tau >= 1.0 required".to_string()));
        }
        Ok(Self { gamma, tau })
    }

    /// Computes the minimum Diophantine margin $\min_{1 \le |k_1| + |k_2| \le K} |k_1 \omega_1 + k_2 \omega_2| \cdot (|k_1| + |k_2|)^\tau$ up to resonance order $K$.
    pub fn minimum_diophantine_constant(
        &self,
        omega1: f64,
        omega2: f64,
        max_order_k: i32,
    ) -> Result<(f64, bool)> {
        if max_order_k < 1 {
            return Err(Error::Msg("max_order_k must be >= 1".to_string()));
        }
        // The double loop below costs (2K+1)^2 iterations: budget it up front
        // so absurd K fails fast instead of hanging.
        let side = (max_order_k as usize).saturating_mul(2).saturating_add(1);
        let iters = side
            .checked_mul(side)
            .ok_or_else(|| Error::Msg("max_order_k lattice overflows usize".to_string()))?;
        crate::limits::check_iterations("diophantine lattice", iters)?;
        let mut min_gamma = f64::INFINITY;
        for k1 in -max_order_k..=max_order_k {
            for k2 in -max_order_k..=max_order_k {
                let norm = (k1.abs() + k2.abs()) as f64;
                if norm >= 1.0 && norm <= (max_order_k as f64) {
                    let div = ((k1 as f64) * omega1 + (k2 as f64) * omega2).abs();
                    let g_eff = div * norm.powf(self.tau);
                    if g_eff < min_gamma {
                        min_gamma = g_eff;
                    }
                }
            }
        }
        Ok((min_gamma, min_gamma >= self.gamma))
    }
}
