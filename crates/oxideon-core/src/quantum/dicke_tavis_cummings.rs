//! Dicke Collective Superradiance & Hepp-Lieb Superradiant Quantum Phase Transition (Dicke, 1954; Hepp & Lieb, 1973).
//!
//! ### 1. Dicke Collective Emission Rate & Hepp-Lieb Critical Coupling
//! For $N$ two-level atoms in collective Dicke state $|J = N/2, M\rangle$, the spontaneous emission rate is
//! $\Gamma_M = \gamma_0 (J + M)(J - M + 1)$ (peaking at $\Gamma_{M=0} \approx \gamma_0 N^2 / 4$), and the Dicke model critical coupling is $g_c = \frac{1}{2}\sqrt{\omega_c \omega_0}$.

use crate::error::{Error, Result};

/// Dicke Superradiance & Hepp-Lieb Quantum Phase Transition Engine.
#[derive(Debug, Clone, Copy)]
pub struct DickeTavisCummingsSuperradiance {
    /// Number of two-level emitters $N \ge 2$.
    pub num_atoms_n: usize,
    /// Single-atom spontaneous decay rate $\gamma_0 > 0$.
    pub single_decay_gamma0: f64,
}

impl Default for DickeTavisCummingsSuperradiance {
    fn default() -> Self {
        Self {
            num_atoms_n: 100,
            single_decay_gamma0: 1.0,
        }
    }
}

impl DickeTavisCummingsSuperradiance {
    /// Creates a new Dicke Superradiance engine.
    pub fn new(num_atoms_n: usize, single_decay_gamma0: f64) -> Result<Self> {
        if num_atoms_n < 2 || single_decay_gamma0 <= 0.0 {
            return Err(Error::Msg(
                "num_atoms_n >= 2 and single_decay_gamma0 > 0 required".to_string(),
            ));
        }
        Ok(Self {
            num_atoms_n,
            single_decay_gamma0,
        })
    }

    /// Evaluates the peak superradiant burst rate $\Gamma_{\mathrm{max}} = \gamma_0 (N/2)(N/2 + 1)$ and Hepp-Lieb critical coupling $g_c = \frac{1}{2}\sqrt{\omega_c \omega_0}$.
    pub fn peak_burst_rate_and_critical_coupling(
        &self,
        omega_c: f64,
        omega_0: f64,
    ) -> Result<(f64, f64)> {
        if omega_c <= 0.0 || omega_0 <= 0.0 {
            return Err(Error::Msg("omega_c and omega_0 must be > 0".to_string()));
        }
        let j = 0.5 * (self.num_atoms_n as f64);
        let gamma_max = self.single_decay_gamma0 * j * (j + 1.0);
        let g_crit = 0.5 * (omega_c * omega_0).sqrt();
        Ok((gamma_max, g_crit))
    }
}
