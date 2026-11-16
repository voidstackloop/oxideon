//! Étale Cohomology, Grothendieck-Lefschetz Trace Formula & Weil Zeta Functions over $\mathbb{F}_{p^r}$.
//!
//! ### 1. Grothendieck-Lefschetz Trace Formula in $\ell$-Adic Étale Cohomology
//! For a smooth projective curve $C / \mathbb{F}_q$ of genus $g$, the number of $\mathbb{F}_{q^r}$-rational points
//! is governed by the Frobenius eigenvalues $\alpha_1, \dots, \alpha_{2g}$ acting on $H^1_{\text{ét}}(\bar{C}, \mathbb{Q}_\ell)$:
//! $$\#C(\mathbb{F}_{q^r}) = 1 + q^r - \sum_{i=1}^{2g} \alpha_i^r$$
//! By Deligne's Weil Conjectures (Riemann Hypothesis for varieties over finite fields), $|\alpha_i| = q^{1/2}$
//! and the Hasse-Weil bound holds for all $r \ge 1$:
//! $$\left|\#C(\mathbb{F}_{q^r}) - (q^r + 1)\right| \le 2 g \, q^{r/2}$$

use crate::error::{Error, Result};

/// Étale Cohomology Frobenius Trace & Hasse-Weil Zeta Engine.
#[derive(Debug, Clone)]
pub struct EtaleWeilZetaEngine {
    /// Prime power field size $q = p^k$.
    pub field_q: u64,
    /// Trace of Frobenius $a_1 = q + 1 - \#E(\mathbb{F}_q)$ on an elliptic curve ($g = 1$).
    pub frobenius_trace: i64,
}

impl EtaleWeilZetaEngine {
    /// Constructs an Étale Weil Zeta engine for an elliptic curve over $\mathbb{F}_q$ and verifies the Hasse bound $|a_1| \le 2\sqrt{q}$.
    pub fn new(field_q: u64, point_count_f_q: u64) -> Result<Self> {
        if field_q < 2 {
            return Err(Error::Msg("field_q must be >= 2".to_string()));
        }
        let a1 = (field_q as i64) + 1 - (point_count_f_q as i64);
        let bound = 2.0 * (field_q as f64).sqrt();
        if (a1.abs() as f64) > bound + 1e-9 {
            return Err(Error::Msg(format!(
                "Point count violates Hasse-Weil Riemann Hypothesis bound: |{a1}| > {bound}"
            )));
        }
        Ok(Self {
            field_q,
            frobenius_trace: a1,
        })
    }

    /// Computes $\#E(\mathbb{F}_{q^r}) = 1 + q^r - (\alpha^r + \bar{\alpha}^r)$ via Chebyshev-Frobenius recurrence
    /// $t_r = a_1 t_{r-1} - q t_{r-2}$ with $t_0 = 2, t_1 = a_1$.
    pub fn rational_point_count(&self, degree_r: usize) -> Result<u64> {
        if !(1..=20).contains(&degree_r) {
            return Err(Error::Msg(
                "Extension degree r must be in 1..=20".to_string(),
            ));
        }
        let q = self.field_q as i64;
        let mut t_prev2 = 2_i64;
        let mut t_prev1 = self.frobenius_trace;
        for _ in 2..=degree_r {
            let t_curr = self.frobenius_trace * t_prev1 - q * t_prev2;
            t_prev2 = t_prev1;
            t_prev1 = t_curr;
        }
        let tr = if degree_r == 1 {
            self.frobenius_trace
        } else {
            t_prev1
        };
        let qr = (self.field_q as i64).pow(degree_r as u32);
        Ok((1 + qr - tr) as u64)
    }

    /// Evaluates the rational Weil Zeta function $Z(E / \mathbb{F}_q, T) = \frac{1 - a_1 T + q T^2}{(1 - T)(1 - q T)}$.
    pub fn evaluate_zeta(&self, t: f64) -> Result<f64> {
        let q = self.field_q as f64;
        let denom = (1.0 - t) * (1.0 - q * t);
        if denom.abs() < 1e-15 {
            return Err(Error::Msg(
                "Pole of Weil Zeta function at T=1 or T=1/q".to_string(),
            ));
        }
        let num = 1.0 - (self.frobenius_trace as f64) * t + q * t * t;
        Ok(num / denom)
    }
}
