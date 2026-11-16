//! Kitaev 1D Topological Superconductor Chain, Majorana Zero Modes & $\mathbb{Z}_2$ Pfaffian Invariant (Kitaev, 2001).
//!
//! ### 1. Kitaev $p$-Wave Superconducting Chain & Topological Phase Transition
//! The 1D spinless $p$-wave superconductor with chemical potential $\mu$, hopping $t > 0$, and pairing $\Delta > 0$
//! has Bogoliubov-de Gennes quasiparticle dispersion:
//! $$E(k) = \sqrt{(2 t \cos k + \mu)^2 + 4 \Delta^2 \sin^2 k}$$
//! The $\mathbb{Z}_2$ Pfaffian topological invariant $\mathcal{Q} = \operatorname{sgn}\left(\mathrm{Pf}(A(0)) \mathrm{Pf}(A(\pi))\right) = \operatorname{sgn}(\mu^2 - 4 t^2)$
//! equals $-1$ in the **Topological Majorana Phase** ($|\mu| < 2 t$, hosting unpaired edge Majorana zero modes $\gamma_L, \gamma_R$)
//! and $+1$ in the trivial phase ($|\mu| > 2 t$).

use crate::error::{Error, Result};

/// Kitaev 1D Majorana Topological Superconductor & Pfaffian Invariant Engine.
#[derive(Debug, Clone, Copy)]
pub struct KitaevMajoranaChainEngine {
    /// Chemical potential $\mu$.
    pub mu: f64,
    /// Nearest-neighbor hopping $t > 0$.
    pub hopping_t: f64,
    /// $p$-wave superconducting pairing amplitude $\Delta > 0$.
    pub pairing_delta: f64,
}

impl KitaevMajoranaChainEngine {
    /// Creates a new Kitaev Majorana chain engine.
    pub fn new(mu: f64, hopping_t: f64, pairing_delta: f64) -> Result<Self> {
        if hopping_t <= 0.0 || pairing_delta <= 0.0 {
            return Err(Error::Msg(
                "hopping_t and pairing_delta must be > 0".to_string(),
            ));
        }
        Ok(Self {
            mu,
            hopping_t,
            pairing_delta,
        })
    }

    /// Computes the $\mathbb{Z}_2$ Majorana Pfaffian topological invariant $\mathcal{Q} \in \{-1, +1\}$ (`-1` = topological phase with Majorana zero modes).
    pub fn z2_pfaffian_invariant(&self) -> i32 {
        if self.mu.abs() < 2.0 * self.hopping_t {
            -1
        } else {
            1
        }
    }

    /// Evaluates the Bogoliubov-de Gennes quasiparticle energy $E(k)$ at crystal momentum $k \in [-\pi, \pi]$.
    pub fn bdg_quasiparticle_energy(&self, k: f64) -> f64 {
        let eps_k = 2.0 * self.hopping_t * k.cos() + self.mu;
        let gap_k = 2.0 * self.pairing_delta * k.sin();
        (eps_k * eps_k + gap_k * gap_k).sqrt()
    }
}
