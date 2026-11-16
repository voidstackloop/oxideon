//! Discrete Morse-Smale-Witten Theory, Forman Gradient Vector Fields & Morse Inequalities (Forman, 1998).
//!
//! ### 1. Discrete Morse Functions & Critical Cells
//! Given a finite simplicial complex $K$ with critical cell counts $(c_0, c_1, c_2, \dots, c_d)$ induced by
//! a discrete Forman gradient vector field $V$ (an acyclic partial matching of face-coface pairs $\sigma^{(p)} < \tau^{(p+1)}$),
//! the Morse-Smale-Witten chain complex $C_p^{\text{MSW}}$ is generated exclusively by the unmatched **critical $p$-simplices**.
//!
//! ### 2. Weak & Strong Morse Inequalities
//! For any discrete Morse function on $K$ with Betti numbers $(b_0, b_1, \dots, b_d)$:
//! • **Weak Morse Inequality**: $c_p \ge b_p$ for every dimension $p$, and Euler-Poincaré identity:
//! $$\chi(K) = \sum_{p=0}^d (-1)^p c_p = \sum_{p=0}^d (-1)^p b_p$$
//! • **Strong Morse Inequality**: For every $k = 0, \dots, d$:
//! $$\sum_{p=0}^k (-1)^{k-p} c_p \ge \sum_{p=0}^k (-1)^{k-p} b_p$$

use crate::error::{Error, Result};

/// Discrete Morse-Smale-Witten Cohomology & Forman Gradient Engine.
#[derive(Debug, Clone)]
pub struct DiscreteMorseEngine {
    /// Critical simplex counts $c_p$ per dimension $p = 0, \dots, d$.
    pub critical_counts: Vec<usize>,
    /// Topological Betti numbers $b_p$ per dimension $p = 0, \dots, d$.
    pub betti_numbers: Vec<usize>,
}

impl DiscreteMorseEngine {
    /// Constructs a discrete Morse engine and validates dimensions.
    pub fn new(critical_counts: Vec<usize>, betti_numbers: Vec<usize>) -> Result<Self> {
        if critical_counts.is_empty() || critical_counts.len() != betti_numbers.len() {
            return Err(Error::Msg(
                "critical_counts and betti_numbers must have equal non-zero length".to_string(),
            ));
        }
        Ok(Self {
            critical_counts,
            betti_numbers,
        })
    }

    /// Computes the Euler characteristic $\chi = \sum_p (-1)^p c_p$ from critical simplex counts.
    pub fn morse_euler_characteristic(&self) -> i64 {
        let mut chi = 0_i64;
        for (p, &c_p) in self.critical_counts.iter().enumerate() {
            let sign = if p % 2 == 0 { 1_i64 } else { -1_i64 };
            chi += sign * (c_p as i64);
        }
        chi
    }

    /// Computes the homological Euler characteristic $\chi = \sum_p (-1)^p b_p$ from Betti numbers.
    pub fn betti_euler_characteristic(&self) -> i64 {
        let mut chi = 0_i64;
        for (p, &b_p) in self.betti_numbers.iter().enumerate() {
            let sign = if p % 2 == 0 { 1_i64 } else { -1_i64 };
            chi += sign * (b_p as i64);
        }
        chi
    }

    /// Verifies both the Weak and Strong Morse Inequalities and exact Euler-Poincaré identity.
    pub fn verify_strong_morse_inequalities(&self) -> bool {
        if self.critical_counts.len() != self.betti_numbers.len() {
            return false;
        }
        if self.morse_euler_characteristic() != self.betti_euler_characteristic() {
            return false;
        }
        let d = self.critical_counts.len();
        for k in 0..d {
            if self.critical_counts[k] < self.betti_numbers[k] {
                return false;
            }
            let mut alt_c = 0_i64;
            let mut alt_b = 0_i64;
            for p in 0..=k {
                let sign = if (k - p) % 2 == 0 { 1_i64 } else { -1_i64 };
                alt_c += sign * (self.critical_counts[p] as i64);
                alt_b += sign * (self.betti_numbers[p] as i64);
            }
            if alt_c < alt_b {
                return false;
            }
        }
        true
    }

    /// Evaluates the Morse polynomial $M(t) = \sum_{p=0}^d c_p t^p$ and Poincaré polynomial $P(t) = \sum_{p=0}^d b_p t^p$.
    pub fn evaluate_morse_and_poincare_polynomials(&self, t: f64) -> (f64, f64) {
        let mut m_t = 0.0;
        let mut p_t = 0.0;
        for (p, (&c_p, &b_p)) in self
            .critical_counts
            .iter()
            .zip(self.betti_numbers.iter())
            .enumerate()
        {
            let tp = t.powi(p as i32);
            m_t += (c_p as f64) * tp;
            p_t += (b_p as f64) * tp;
        }
        (m_t, p_t)
    }
}
