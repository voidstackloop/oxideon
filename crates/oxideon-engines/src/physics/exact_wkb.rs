//! Voros-Silverstone Exact WKB Analysis, Quantum Periods $\Pi_\gamma(\hbar)$ & Dunham-Wentzel Corrections (Voros, 1983).
//!
//! ### 1. Exact Bohr-Sommerfeld-Voros Quantization Condition
//! For a 1D Schrödinger operator $(-\hbar^2 \frac{d^2}{dx^2} + V(x)) \psi = E \psi$, the all-orders Exact WKB
//! quantum period expands in even powers of $\hbar$:
//! $$\Pi(E, \hbar) = \oint_\gamma p_0(x, E) \, dx + \hbar^2 \Pi_1(E) + \hbar^4 \Pi_2(E) + \cdots = 2\pi \hbar \left(n + \frac{1}{2}\right)$$
//! For the anharmonic oscillator $V(x) = \frac{1}{2} \omega^2 x^2 + \lambda x^4$, the first-order Dunham-WKB energy
//! levels are $E_n(\lambda) = \hbar \omega (n + \frac{1}{2}) + \frac{3 \lambda \hbar^2}{4 \omega^2} (2 n^2 + 2 n + 1)$.

use crate::error::{Error, Result};

/// Voros Exact WKB Quantum Period & Anharmonic Quantization Engine.
#[derive(Debug, Clone, Copy)]
pub struct ExactWkbQuantizationEngine {
    /// Planck constant $\hbar > 0$.
    pub hbar: f64,
    /// Harmonic frequency $\omega > 0$.
    pub omega: f64,
    /// Quartic anharmonicity coupling $\lambda \ge 0$.
    pub anharmonic_lambda: f64,
}

impl ExactWkbQuantizationEngine {
    /// Creates a new Exact WKB quantization engine.
    pub fn new(hbar: f64, omega: f64, anharmonic_lambda: f64) -> Result<Self> {
        if hbar <= 0.0 || omega <= 0.0 || anharmonic_lambda < 0.0 {
            return Err(Error::Msg(
                "hbar > 0, omega > 0, and anharmonic_lambda >= 0 required".to_string(),
            ));
        }
        Ok(Self {
            hbar,
            omega,
            anharmonic_lambda,
        })
    }

    /// Computes the Dunham-WKB quantized energy level $E_n$ for quantum number $n \ge 0$.
    pub fn quantized_energy_level(&self, level_n: usize) -> f64 {
        let nf = level_n as f64;
        let e0 = self.hbar * self.omega * (nf + 0.5);
        let e1 = (0.75 * self.anharmonic_lambda * self.hbar.powi(2) / self.omega.powi(2))
            * (2.0 * nf * nf + 2.0 * nf + 1.0);
        e0 + e1
    }

    /// Computes the classical WKB Gamow-Agmon tunneling action $S_{\text{inst}} = \frac{\omega^3}{3 \lambda}$ for the inverted double well.
    pub fn double_well_instanton_action(&self) -> Result<f64> {
        if self.anharmonic_lambda <= 0.0 {
            return Err(Error::Msg(
                "anharmonic_lambda must be > 0 for instanton action".to_string(),
            ));
        }
        Ok(self.omega.powi(3) / (3.0 * self.anharmonic_lambda))
    }
}
