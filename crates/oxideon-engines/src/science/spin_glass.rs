//! Spin Glass Statistical Physics: Sherrington-Kirkpatrick (SK) Model,
//! Parisi Replica Symmetry Breaking (RSB) & Thouless-Anderson-Palmer (TAP) Cavity Equations.
//!
//! # Theoretical Framework
//!
//! 1. **Sherrington-Kirkpatrick (SK) Hamiltonian**:
//!    For $N$ discrete Ising spins $\sigma_i \in \{-1, +1\}$:
//!    $$H(\sigma) = -\sum_{1 \le i < j \le N} J_{ij} \sigma_i \sigma_j - \sum_{i=1}^N h_i \sigma_i$$
//!    where quenched couplings are Gaussian $J_{ij} \sim \mathcal{N}\left(0, \frac{J^2}{N}\right)$.
//!
//! 2. **Thouless-Anderson-Palmer (TAP) Equations & Onsager Cavity Reaction**:
//!    The local magnetizations $m_i = \langle \sigma_i \rangle \in [-1, 1]$ at inverse temperature $\beta = 1 / T$
//!    satisfy self-consistent cavity equations with the Onsager reaction term:
//!    $$m_i = \tanh\left( \beta \sum_{j \neq i} J_{ij} m_j + \beta h_i - \beta^2 \sum_{j \neq i} J_{ij}^2 (1 - m_j^2) m_i \right)$$
//!    For the SK model where $\sum_j J_{ij}^2 \approx J^2$:
//!    $$m_i = \tanh\left( \beta \sum_{j \neq i} J_{ij} m_j + \beta h_i - \beta^2 J^2 (1 - q) m_i \right)$$
//!    where $q = \frac{1}{N} \sum_j m_j^2$ is the Edwards-Anderson order parameter.
//!
//! 3. **TAP Free Energy**:
//!    $$F_{\text{TAP}}(m) = -\sum_{i < j} J_{ij} m_i m_j - \sum_i h_i m_i - \frac{\beta}{2} \sum_{i < j} J_{ij}^2 (1 - m_i^2)(1 - m_j^2) - \frac{1}{\beta} \sum_i S(m_i)$$
//!    with single-spin entropy $S(m) = -\frac{1+m}{2} \ln \frac{1+m}{2} - \frac{1-m}{2} \ln \frac{1-m}{2}$.
//!
//! 4. **de Almeida-Thouless (AT) Instability & Replica Symmetry Breaking**:
//!    At high temperature, the replica-symmetric (RS) solution is stable. The AT eigenvalue:
//!    $$\lambda_{\text{AT}} = 1 - \beta^2 J^2 \frac{1}{N} \sum_{i=1}^N (1 - m_i^2)^2$$
//!    When $\lambda_{\text{AT}} \le 0$, replica symmetry spontaneously breaks into Parisi's ultrametric hierarchy.

use crate::error::{Error, Result};
use std::f64::consts::PI;

/// Result of solving the Thouless-Anderson-Palmer (TAP) cavity equations.
#[derive(Debug, Clone, PartialEq)]
pub struct TapSolution {
    /// Local spin magnetizations $m_i \in [-1, 1]$.
    pub magnetizations: Vec<f64>,
    /// Edwards-Anderson spin-glass order parameter $q_{EA} = \frac{1}{N} \sum_{i=1}^N m_i^2$.
    pub edwards_anderson_q: f64,
    /// Total TAP free energy $F_{\text{TAP}}(m)$.
    pub tap_free_energy: f64,
    /// de Almeida-Thouless (AT) stability eigenvalue $\lambda_{\text{AT}}$.
    pub at_eigenvalue: f64,
    /// Whether replica symmetry is broken ($\lambda_{\text{AT}} \le 0$).
    pub is_replica_symmetry_broken: bool,
    /// Number of fixed-point iterations performed.
    pub iterations: usize,
    /// Whether the solver converged within tolerance.
    pub converged: bool,
}

/// Sherrington-Kirkpatrick (SK) infinite-range spin glass model.
#[derive(Debug, Clone)]
pub struct SherringtonKirkpatrickModel {
    /// Number of spins $N$.
    pub num_spins: usize,
    /// Coupling matrix $J_{ij}$ of shape $[N, N]$ (symmetric, zero diagonal).
    pub couplings: Vec<f64>,
    /// External local magnetic fields $h_i$ of shape $\[N\]$.
    pub external_fields: Vec<f64>,
    /// Coupling variance scale $J$.
    pub j_scale: f64,
}

impl SherringtonKirkpatrickModel {
    /// Creates an SK model with specified couplings and fields.
    pub fn new(num_spins: usize, couplings: Vec<f64>, external_fields: Vec<f64>) -> Result<Self> {
        if num_spins == 0 {
            return Err(Error::Msg("num_spins must be > 0".to_string()));
        }
        let cells = crate::layout::checked_numel(&[num_spins, num_spins])?;
        if couplings.len() != cells {
            return Err(Error::Msg(format!(
                "Couplings length {} does not match N^2 = {cells}",
                couplings.len()
            )));
        }
        if external_fields.len() != num_spins {
            return Err(Error::Msg(format!(
                "External fields length {} does not match N = {}",
                external_fields.len(),
                num_spins
            )));
        }

        // Estimate j_scale: sqrt(N * mean(J_ij^2))
        let mut sum_sq = 0.0;
        let mut count = 0;
        for i in 0..num_spins {
            for j in (i + 1)..num_spins {
                let j_val = couplings[i * num_spins + j];
                sum_sq += j_val * j_val;
                count += 1;
            }
        }
        let j_scale = if count > 0 {
            ((num_spins as f64) * sum_sq / (count as f64)).sqrt()
        } else {
            1.0
        };

        Ok(Self {
            num_spins,
            couplings,
            external_fields,
            j_scale,
        })
    }

    /// Generates a randomized SK model with $J_{ij} \sim \mathcal{N}(0, J^2 / N)$ and $h_i = 0$.
    pub fn random_gaussian(num_spins: usize, j_scale: f64, seed: u64) -> Result<Self> {
        if num_spins == 0 {
            return Err(Error::Msg("num_spins must be > 0".to_string()));
        }

        let mut couplings =
            crate::layout::try_filled(crate::layout::checked_numel(&[num_spins, num_spins])?, 0.0)?;
        let std_dev = j_scale / (num_spins as f64).sqrt();

        // Deterministic pseudo-random Gaussian via Box-Muller
        let mut state = seed.wrapping_add(1);
        let mut next_rand = || -> f64 {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            ((state >> 11) as f64) / ((1u64 << 53) as f64)
        };

        for i in 0..num_spins {
            for j in (i + 1)..num_spins {
                let u1 = next_rand().max(1e-12);
                let u2 = next_rand();
                let g = (-2.0 * u1.ln()).sqrt() * (2.0 * PI * u2).cos();
                let val = g * std_dev;
                couplings[i * num_spins + j] = val;
                couplings[j * num_spins + i] = val;
            }
        }

        let external_fields = crate::layout::try_filled(num_spins, 0.0)?;
        Ok(Self {
            num_spins,
            couplings,
            external_fields,
            j_scale,
        })
    }

    /// Evaluates the exact Hamiltonian energy for a discrete spin configuration $\sigma_i \in \{-1, +1\}$:
    ///
    /// $$H(\sigma) = -\sum_{i < j} J_{ij} \sigma_i \sigma_j - \sum_i h_i \sigma_i$$
    pub fn energy(&self, spins: &[i8]) -> Result<f64> {
        let n = self.num_spins;
        if spins.len() != n {
            return Err(Error::Msg(format!(
                "Spins length {} != N {}",
                spins.len(),
                n
            )));
        }

        let mut e_int = 0.0;
        for i in 0..n {
            for j in (i + 1)..n {
                let s_i = spins[i] as f64;
                let s_j = spins[j] as f64;
                e_int -= self.couplings[i * n + j] * s_i * s_j;
            }
        }

        let mut e_ext = 0.0;
        for (i, &s) in spins.iter().enumerate().take(n) {
            e_ext -= self.external_fields[i] * (s as f64);
        }

        Ok(e_int + e_ext)
    }

    /// Solves the Thouless-Anderson-Palmer (TAP) cavity equations via damped fixed-point iteration.
    pub fn solve_tap(
        &self,
        beta: f64,
        max_iters: usize,
        tol: f64,
        damping: f64,
    ) -> Result<TapSolution> {
        if beta <= 0.0 {
            return Err(Error::Msg(
                "Inverse temperature beta must be > 0".to_string(),
            ));
        }

        let n = self.num_spins;
        let damp = damping.clamp(0.01, 0.99);

        // Initialize magnetizations using local fields + symmetry-breaking perturbation
        let mut m = Vec::with_capacity(n);
        for i in 0..n {
            let pert = ((i % 7) as f64 - 3.0) * 0.08 + beta * self.external_fields[i];
            m.push(pert.tanh());
        }

        let mut converged = false;
        let mut iters = 0;

        for step in 0..max_iters {
            iters = step + 1;
            let mut m_next = Vec::with_capacity(n);
            let mut max_diff = 0.0;

            for i in 0..n {
                // Cavity field: sum_{j != i} J_ij m_j
                let mut cavity_field = 0.0;
                let mut sum_j2_1m2 = 0.0;

                for (j, &m_j) in m.iter().enumerate().take(n) {
                    if i != j {
                        let j_ij = self.couplings[i * n + j];
                        cavity_field += j_ij * m_j;
                        sum_j2_1m2 += j_ij * j_ij * (1.0 - m_j * m_j);
                    }
                }

                // Onsager reaction term with Plefka susceptibility bound:
                // - min(beta^2 * sum_j J_ij^2 (1 - m_j^2), beta * J) * m_i
                let onsager_coeff = (beta * beta * sum_j2_1m2).min(beta * self.j_scale);
                let onsager = onsager_coeff * m[i];
                let eff_field = beta * (cavity_field + self.external_fields[i]) - onsager;
                let m_target = eff_field.tanh();

                // Damped update
                let m_new = (1.0 - damp) * m[i] + damp * m_target;
                let diff = (m_new - m[i]).abs();
                if diff > max_diff {
                    max_diff = diff;
                }
                m_next.push(m_new);
            }

            m = m_next;
            if max_diff < tol {
                converged = true;
                break;
            }
        }

        // Compute final Edwards-Anderson order parameter q_EA
        let q_ea = m.iter().map(|&val| val * val).sum::<f64>() / (n as f64);

        // Compute de Almeida-Thouless (AT) stability eigenvalue:
        // lambda_AT = 1 - beta^2 sum_{i < j} J_ij^2 (1 - m_i^2)(1 - m_j^2) / N
        let mut sum_at = 0.0;
        for i in 0..n {
            for j in (i + 1)..n {
                let j_ij = self.couplings[i * n + j];
                sum_at += j_ij * j_ij * (1.0 - m[i] * m[i]) * (1.0 - m[j] * m[j]);
            }
        }
        let at_eigenvalue = 1.0 - (beta * beta * sum_at / (n as f64));
        let is_replica_symmetry_broken = at_eigenvalue <= 0.0;

        // Compute TAP Free Energy
        let mut e_int = 0.0;
        let mut onsager_energy = 0.0;
        for i in 0..n {
            for j in (i + 1)..n {
                let j_ij = self.couplings[i * n + j];
                e_int -= j_ij * m[i] * m[j];
                onsager_energy -=
                    0.5 * beta * j_ij * j_ij * (1.0 - m[i] * m[i]) * (1.0 - m[j] * m[j]);
            }
        }

        let mut e_ext = 0.0;
        let mut entropy_sum = 0.0;
        for (i, &m_i) in m.iter().enumerate().take(n) {
            e_ext -= self.external_fields[i] * m_i;
            let p_plus = (0.5 * (1.0 + m_i)).clamp(1e-12, 1.0 - 1e-12);
            let p_minus = (0.5 * (1.0 - m_i)).clamp(1e-12, 1.0 - 1e-12);
            let s_i = -(p_plus * p_plus.ln() + p_minus * p_minus.ln());
            entropy_sum += s_i;
        }

        let tap_free_energy = e_int + e_ext + onsager_energy - (1.0 / beta) * entropy_sum;

        Ok(TapSolution {
            magnetizations: m,
            edwards_anderson_q: q_ea,
            tap_free_energy,
            at_eigenvalue,
            is_replica_symmetry_broken,
            iterations: iters,
            converged,
        })
    }
}
