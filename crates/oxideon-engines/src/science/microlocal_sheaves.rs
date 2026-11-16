//! Hörmander Microlocal Analysis, Wavefront Sets $WF(u) \subset T^*M \setminus 0$ & Bicharakteristic Hamiltonian Flow (Hörmander, 1971).
//!
//! ### 1. Propagation of Singularities Theorem
//! For a differential operator $P(x, D)$ with real principal symbol $p(x, \xi)$ on $T^*M \setminus 0$,
//! Hörmander's **Propagation of Singularities Theorem** states that the wavefront set $WF(u)$ of a solution
//! $P u = 0$ is contained in the characteristic variety $\mathrm{Char}(P) = \{(x, \xi) : p(x, \xi) = 0\}$ and is
//! invariant under the symplectic Hamiltonian flow of $p$:
//! $$\dot{x}_i = \frac{\partial p}{\partial \xi_i}, \qquad \dot{\xi}_i = -\frac{\partial p}{\partial x_i}$$

use crate::error::{Error, Result};

/// Hörmander Microlocal Wavefront Set & Bicharakteristic Ray Engine.
#[derive(Debug, Clone, Copy)]
pub struct MicrolocalWavefrontEngine {
    /// Acoustic/optical phase speed $c > 0$ in principal symbol $p(x, \xi, t, \tau) = \tau^2 - c^2 |\xi|^2$.
    pub wave_speed_c: f64,
}

impl MicrolocalWavefrontEngine {
    /// Creates a new Microlocal Wavefront engine.
    pub fn new(wave_speed_c: f64) -> Result<Self> {
        if wave_speed_c <= 0.0 {
            return Err(Error::Msg("wave_speed_c must be > 0".to_string()));
        }
        Ok(Self { wave_speed_c })
    }

    /// Evaluates the wave operator principal symbol $p(\xi, \tau) = \tau^2 - c^2 |\xi|^2$ (vanishes on the characteristic lightcone).
    pub fn principal_symbol(&self, xi: [f64; 2], tau: f64) -> f64 {
        let xi_sq = xi[0] * xi[0] + xi[1] * xi[1];
        tau * tau - self.wave_speed_c * self.wave_speed_c * xi_sq
    }

    /// Propagates a covector singularity $(x_0, y_0, \xi_0, \eta_0)$ along the null bicharacteristic strip for time $t$.
    pub fn propagate_bicharacteristic(
        &self,
        x0: [f64; 2],
        xi: [f64; 2],
        t: f64,
    ) -> Result<[f64; 2]> {
        let norm_xi = (xi[0] * xi[0] + xi[1] * xi[1]).sqrt();
        if norm_xi < 1e-15 {
            return Err(Error::Msg(
                "Zero section covector is excluded from T^*M minus 0".to_string(),
            ));
        }
        Ok([
            x0[0] + self.wave_speed_c * (xi[0] / norm_xi) * t,
            x0[1] + self.wave_speed_c * (xi[1] / norm_xi) * t,
        ])
    }
}
