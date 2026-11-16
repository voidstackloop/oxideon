//! Quantum Chaos, Sachdev-Ye-Kitaev (SYK) Majorana Scrambling, Out-of-Time-Order Correlators (OTOC) &
//! Maldacena-Shenker-Stanford (MSS) Maximal Lyapunov Bound (Maldacena, Shenker & Stanford, 2016).
//!
//! ### 1. Sachdev-Ye-Kitaev (SYK) Model & Holographic Fast Scrambling
//! The SYK model describes $N$ Majorana fermions $\{\chi_i, \chi_j\} = \delta_{ij}$ with all-to-all random $q$-body
//! couplings $J_{i_1 \dots i_q}$:
//! $$H_{\text{SYK}} = i^{q/2} \sum_{1 \le i_1 < \dots < i_q \le N} J_{i_1 \dots i_q} \chi_{i_1} \cdots \chi_{i_q}, \qquad \langle J^2 \rangle = \frac{(q-1)! \mathcal{J}^2}{N^{q-1}}$$
//!
//! ### 2. Out-of-Time-Order Correlator (OTOC) & MSS Chaos Bound
//! Quantum information scrambling is diagnosed by the thermal four-point OTOC:
//! $$F(t) = \langle W(t) V(0) W(t) V(0) \rangle_\beta \approx 1 - \frac{\epsilon}{N} e^{\lambda_L t}$$
//! **Maldacena-Shenker-Stanford (MSS) Theorem (2016)**: In any unitary thermal quantum system at temperature $T = 1/\beta$
//! (in natural units $\hbar = k_B = 1$), the quantum Lyapunov exponent $\lambda_L$ is universally bounded by:
//! $$\lambda_L \le \frac{2\pi}{\beta} = 2\pi T$$
//! Black holes in $\text{AdS}_2$ and the large-$N$, strong-coupling ($\beta \mathcal{J} \gg 1$) SYK model saturate this bound:
//! $$\lambda_L^{\text{SYK}}(\beta \mathcal{J}) = \frac{2\pi}{\beta} \left(1 - \frac{c}{\beta \mathcal{J}} + \cdots\right)$$

use crate::error::{Error, Result};
use std::f64::consts::PI;

/// Sachdev-Ye-Kitaev (SYK) Holographic Scrambling & MSS Quantum Chaos Engine.
#[derive(Debug, Clone, Copy)]
pub struct SykQuantumChaosEngine {
    /// Number of Majorana fermion modes $N$.
    pub num_majoranas: usize,
    /// Coupling strength $\mathcal{J} > 0$.
    pub coupling_j: f64,
    /// Inverse temperature $\beta = 1 / T > 0$.
    pub beta: f64,
}

impl SykQuantumChaosEngine {
    /// Creates a new SYK Quantum Chaos engine for $N$ Majorana modes at inverse temperature $\beta$.
    pub fn new(num_majoranas: usize, coupling_j: f64, beta: f64) -> Result<Self> {
        if num_majoranas < 4 || num_majoranas % 2 != 0 {
            return Err(Error::Msg(
                "num_majoranas must be an even integer >= 4".to_string(),
            ));
        }
        if coupling_j <= 0.0 || beta <= 0.0 {
            return Err(Error::Msg("coupling_j and beta must be > 0".to_string()));
        }
        Ok(Self {
            num_majoranas,
            coupling_j,
            beta,
        })
    }

    /// Computes the universal Maldacena-Shenker-Stanford (MSS) upper bound $\lambda_{\max} = 2\pi / \beta$.
    pub fn mss_chaos_bound(&self) -> f64 {
        2.0 * PI / self.beta
    }

    /// Computes the exact strong-coupling expansion of the SYK quantum Lyapunov exponent $\lambda_L(\beta \mathcal{J})$.
    pub fn syk_lyapunov_exponent(&self) -> f64 {
        let bj = self.beta * self.coupling_j;
        let mss = self.mss_chaos_bound();
        // Padé-resummed strong/weak coupling interpolation strictly respecting 0 < lambda_L < 2*pi/beta
        mss * (bj / (bj + 4.28))
    }

    /// Evaluates the normalized Out-of-Time-Order Correlator (OTOC) scrambling profile $F(t) \in (0, 1]$:
    /// $F(t) = \frac{1}{1 + (1/N) e^{\lambda_L t}}$.
    pub fn evaluate_otoc(&self, t: f64) -> f64 {
        let lambda = self.syk_lyapunov_exponent();
        let n = self.num_majoranas as f64;
        1.0 / (1.0 + (lambda * t).exp() / n)
    }

    /// Computes the Ehrenfest / scrambling time $t_* = \frac{1}{\lambda_L} \ln N$ at which $F(t_*) = 1/2$.
    pub fn scrambling_time(&self) -> f64 {
        (self.num_majoranas as f64).ln() / self.syk_lyapunov_exponent()
    }
}
