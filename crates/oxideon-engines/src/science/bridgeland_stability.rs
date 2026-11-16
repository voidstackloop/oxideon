//! Bridgeland Stability Conditions on Triangulated Categories, Central Charge $Z(E) \in \mathbb{H}$ &
//! Harder-Narasimhan Filtrations (Bridgeland, 2007).
//!
//! ### 1. Bridgeland Central Charge & Phase $\phi(E) \in (0, 1]$
//! Given a stability condition $\sigma = (Z, \mathcal{P})$ on a derived category $D^b(\mathrm{Coh}(X))$, the central charge
//! $Z: K_0(\mathcal{D}) \to \mathbb{C}$ maps non-zero objects $E \in \mathcal{A}$ into the upper half-plane
//! $\mathbb{H} \cup \mathbb{R}_{<0} = \{r e^{i \pi \phi} : r > 0, \; \phi \in (0, 1]\}$:
//! $$Z(r, d) = -d + i \, r, \qquad \mu(E) = \frac{d}{r}, \qquad \phi(E) = \frac{1}{\pi} \operatorname{Arg}(Z(E)) \in (0, 1]$$
//!
//! ### 2. Harder-Narasimhan Polygon & Mass $m_\sigma(E) = \sum_i |Z(A_i)|$
//! Every object $E$ admits a unique Harder-Narasimhan filtration $0 = E_0 \subset E_1 \subset \cdots \subset E_m = E$
//! with semistable quotients $A_i = E_i / E_{i-1}$ of strictly decreasing phase $\phi(A_1) > \phi(A_2) > \cdots > \phi(A_m)$.

use crate::error::{Error, Result};
use std::f64::consts::PI;

/// Bridgeland Derived Category Stability Condition & Harder-Narasimhan Engine.
#[derive(Debug, Clone, Copy)]
pub struct BridgelandStabilityEngine {
    /// Kähler volume parameter $\omega > 0$ scaling the imaginary part of $Z(r, d) = -d + i \omega r$.
    pub kahler_omega: f64,
}

impl BridgelandStabilityEngine {
    /// Creates a new Bridgeland stability engine with Kähler parameter $\omega > 0$.
    pub fn new(kahler_omega: f64) -> Result<Self> {
        if kahler_omega <= 0.0 {
            return Err(Error::Msg("kahler_omega must be > 0".to_string()));
        }
        Ok(Self { kahler_omega })
    }

    /// Evaluates the complex central charge $Z(r, d) = (-d, \omega r)$ for rank $r \ge 0$ and degree $d$ (with $(r, d) \neq (0,0)$).
    pub fn central_charge(&self, rank: f64, degree: f64) -> Result<(f64, f64)> {
        if rank < 0.0 || (rank == 0.0 && degree <= 0.0) {
            return Err(Error::Msg(
                "Object must lie in the heart: rank > 0 or (rank == 0 and degree > 0)".to_string(),
            ));
        }
        Ok((-degree, self.kahler_omega * rank))
    }

    /// Computes the normalized Bridgeland phase $\phi(r, d) \in (0, 1]$ and mass $|Z(r, d)|$.
    pub fn phase_and_mass(&self, rank: f64, degree: f64) -> Result<(f64, f64)> {
        let (re, im) = self.central_charge(rank, degree)?;
        let mass = (re * re + im * im).sqrt();
        let angle = im.atan2(re); // in (0, pi] since im >= 0 and if im == 0, re < 0 => angle = pi
        Ok((angle / PI, mass))
    }

    /// Verifies the Harder-Narasimhan strict phase ordering $\phi(A_1) > \phi(A_2) > \cdots > \phi(A_m)$
    /// and support property $|Z(A_i)| > 0$ for a sequence of HN factors `(rank, degree)`.
    pub fn verify_harder_narasimhan_filtration(
        &self,
        factors: &[(f64, f64)],
    ) -> Result<(bool, f64)> {
        if factors.is_empty() {
            return Err(Error::Msg(
                "HN filtration requires at least one factor".to_string(),
            ));
        }
        let mut prev_phi = f64::INFINITY;
        let mut total_mass = 0.0;
        for &(r, d) in factors {
            let (phi, m) = self.phase_and_mass(r, d)?;
            if phi >= prev_phi {
                return Ok((false, total_mass));
            }
            prev_phi = phi;
            total_mass += m;
        }
        Ok((true, total_mass))
    }
}
