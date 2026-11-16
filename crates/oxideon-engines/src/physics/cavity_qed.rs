//! Differentiable Cavity Quantum Electrodynamics (Cavity QED), Jaynes-Cummings Hamiltonian & Vacuum Rabi Splitting.
//!
//! ### 1. Jaynes-Cummings Hamiltonian (Jaynes & Cummings, 1963)
//! A two-level quantum emitter (qubit $|g\rangle, |e\rangle$ of frequency $\omega_q$) coupled to a quantized single-mode
//! optical cavity ($a^\dagger, a$ of frequency $\omega_c$) in the Rotating-Wave Approximation (RWA) obeys:
//! $$\hat{H}_{\text{JC}} = \hbar \omega_c \hat{a}^\dagger \hat{a} + \frac{\hbar \omega_q}{2} \hat{\sigma}_z + \hbar g \left(\hat{a}^\dagger \hat{\sigma}_- + \hat{a} \hat{\sigma}_+\right)$$
//!
//! ### 2. Dressed-State Polaritons & Vacuum Rabi Oscillations
//! Since total excitation number $\hat{N} = \hat{a}^\dagger \hat{a} + |e\rangle\langle e|$ is strictly conserved,
//! the $n$-excitation subspace $\operatorname{span}\{|e, n-1\rangle, |g, n\rangle\}$ is a $2 \times 2$ block with
//! generalized Rabi frequency:
//! $$\Omega_n = \sqrt{\Delta^2 + 4 g^2 n}, \qquad \Delta = \omega_q - \omega_c$$
//! Starting in $|e, n-1\rangle$ at $t = 0$, the atomic excited-state population oscillates exactly as:
//! $$P_e(t) = 1 - \frac{4 g^2 n}{\Omega_n^2} \sin^2\left(\frac{\Omega_n t}{2}\right)$$

use crate::error::{Error, Result};

/// Jaynes-Cummings Cavity QED & Vacuum Rabi Polariton Engine.
#[derive(Debug, Clone, Copy)]
pub struct JaynesCummingsCavityQed {
    /// Cavity resonance frequency $\omega_c$.
    pub omega_c: f64,
    /// Qubit transition frequency $\omega_q$.
    pub omega_q: f64,
    /// Single-photon dipole coupling strength $g > 0$.
    pub coupling_g: f64,
    /// Cavity photon decay rate $\kappa \ge 0$.
    pub kappa: f64,
}

impl JaynesCummingsCavityQed {
    /// Creates a new Jaynes-Cummings Cavity QED solver.
    pub fn new(omega_c: f64, omega_q: f64, coupling_g: f64, kappa: f64) -> Result<Self> {
        if coupling_g <= 0.0 || kappa < 0.0 {
            return Err(Error::Msg(
                "coupling_g must be > 0 and kappa >= 0".to_string(),
            ));
        }
        Ok(Self {
            omega_c,
            omega_q,
            coupling_g,
            kappa,
        })
    }

    /// Computes the generalized $n$-photon Rabi frequency $\Omega_n = \sqrt{(\omega_q - \omega_c)^2 + 4 g^2 n}$.
    pub fn generalized_rabi_frequency(&self, n_excitations: usize) -> f64 {
        let delta = self.omega_q - self.omega_c;
        (delta * delta + 4.0 * self.coupling_g * self.coupling_g * (n_excitations as f64)).sqrt()
    }

    /// Computes the dressed polariton eigenenergies $(E_{n, -}, E_{n, +})$ in the $n$-excitation manifold ($n \ge 1$).
    pub fn polariton_eigenenergies(&self, n_excitations: usize) -> Result<(f64, f64)> {
        if n_excitations == 0 {
            return Err(Error::Msg(
                "Polariton doublet requires n_excitations >= 1".to_string(),
            ));
        }
        let center = self.omega_c * ((n_excitations as f64) - 0.5);
        let half_split = 0.5 * self.generalized_rabi_frequency(n_excitations);
        Ok((center - half_split, center + half_split))
    }

    /// Evaluates the damped excited-state population $P_e(t)$ and cavity photon population $P_c(t)$
    /// starting from initial state $|e, n-1\rangle$ at $t = 0$.
    pub fn rabi_oscillation_populations(&self, n_excitations: usize, t: f64) -> Result<(f64, f64)> {
        if n_excitations == 0 || t < 0.0 {
            return Err(Error::Msg(
                "n_excitations must be >= 1 and t >= 0".to_string(),
            ));
        }
        let omega_n = self.generalized_rabi_frequency(n_excitations);
        let sin_sq = (0.5 * omega_n * t).sin().powi(2);
        let transfer = (4.0 * self.coupling_g * self.coupling_g * (n_excitations as f64)
            / (omega_n * omega_n))
            * sin_sq;
        let envelope = (-0.5 * self.kappa * t).exp();
        let p_e = envelope * (1.0 - transfer);
        let p_c = envelope * transfer;
        Ok((p_e, p_c))
    }
}
