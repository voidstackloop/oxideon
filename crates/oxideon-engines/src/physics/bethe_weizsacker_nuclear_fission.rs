//! Bethe-Weizsäcker Liquid-Drop Nuclear Binding Energy & Bohr-Wheeler Fissility Parameter (Weizsäcker, 1935; Bohr & Wheeler, 1939).
//!
//! ### 1. Semi-Empirical Binding Energy $B(A, Z)$ & Bohr-Wheeler Spontaneous Fission Limit
//! $$B(A, Z) = a_V A - a_S A^{2/3} - a_C \frac{Z(Z - 1)}{A^{1/3}} - a_A \frac{(A - 2Z)^2}{A} + \delta(A, Z)$$
//! The liquid drop becomes unstable against quadrupole deformation when the Bohr-Wheeler fissility $x = \frac{E_C}{2 E_S} = \frac{a_C Z^2}{2 a_S A} \ge 1$.

use crate::error::{Error, Result};

/// Bethe-Weizsäcker Nuclear Binding Energy & Bohr-Wheeler Fissility Engine.
#[derive(Debug, Clone, Copy)]
pub struct BetheWeizsackerNuclearMass {
    /// Volume coefficient $a_V$ (MeV).
    pub a_v: f64,
    /// Surface coefficient $a_S$ (MeV).
    pub a_s: f64,
    /// Coulomb coefficient $a_C$ (MeV).
    pub a_c: f64,
    /// Asymmetry coefficient $a_A$ (MeV).
    pub a_a: f64,
}

impl Default for BetheWeizsackerNuclearMass {
    fn default() -> Self {
        Self {
            a_v: 15.8,
            a_s: 18.3,
            a_c: 0.714,
            a_a: 23.2,
        }
    }
}

impl BetheWeizsackerNuclearMass {
    /// Creates a new Bethe-Weizsäcker Nuclear Mass engine.
    pub fn new(a_v: f64, a_s: f64, a_c: f64, a_a: f64) -> Result<Self> {
        if a_v <= 0.0 || a_s <= 0.0 || a_c <= 0.0 || a_a <= 0.0 {
            return Err(Error::Msg(
                "All liquid-drop coefficients must be > 0".to_string(),
            ));
        }
        Ok(Self { a_v, a_s, a_c, a_a })
    }

    /// Evaluates binding energy per nucleon $B(A, Z)/A$ (in MeV) and the Bohr-Wheeler fissility parameter $x = a_C Z(Z-1) / (2 a_S A)$.
    pub fn binding_per_nucleon_and_fissility(
        &self,
        mass_a: u32,
        atomic_z: u32,
    ) -> Result<(f64, f64)> {
        if mass_a < 2 || atomic_z == 0 || atomic_z > mass_a {
            return Err(Error::Msg(
                "1 <= atomic_z <= mass_a and mass_a >= 2 required".to_string(),
            ));
        }
        let a = mass_a as f64;
        let z = atomic_z as f64;
        let a13 = a.cbrt();
        let a23 = a13 * a13;
        let b_tot = self.a_v * a
            - self.a_s * a23
            - self.a_c * z * (z - 1.0) / a13
            - self.a_a * (a - 2.0 * z).powi(2) / a;
        let fissility_x = (self.a_c * z * (z - 1.0)) / (2.0 * self.a_s * a);
        Ok((b_tot / a, fissility_x))
    }
}
