//! Allard Varifold Regularity Theorem & Monotonicity of Density Ratios (Allard, 1972).
//!
//! ### 1. Stationary Varifold Density Ratio $\Theta_k(\mu, x_0, r)$ & Allard $\varepsilon$-Regularity
//! For a stationary $k$-rectifiable varifold $V$ in $\mathbb{R}^n$, the normalized mass ratio $\Theta_k(x_0, r) = \frac{\mu(B_r(x_0))}{\omega_k r^k}$
//! is monotone non-decreasing in $r$. If $\Theta_k(x_0, r) < 1 + \varepsilon_{\mathrm{Allard}}$ and the tilt-excess is small, $V$ is a smooth $C^{1,\alpha}$ minimal graph in $B_{r/2}(x_0)$.

use super::unit_ball_volume;
use crate::error::{Error, Result};

/// Allard Varifold Monotonicity & Density-One Regularity Engine.
#[derive(Debug, Clone, Copy)]
pub struct AllardVarifoldMonotonicityEngine {
    /// Varifold dimension $k \ge 1$.
    pub dim_k: usize,
    /// Allard excess regularity threshold $\varepsilon_{\mathrm{Allard}} > 0$.
    pub epsilon_allard: f64,
}

impl Default for AllardVarifoldMonotonicityEngine {
    fn default() -> Self {
        Self {
            dim_k: 2,
            epsilon_allard: 0.05,
        }
    }
}

impl AllardVarifoldMonotonicityEngine {
    /// Creates a new Allard Varifold Monotonicity engine.
    pub fn new(dim_k: usize, epsilon_allard: f64) -> Result<Self> {
        if dim_k == 0 || epsilon_allard <= 0.0 {
            return Err(Error::Msg(
                "dim_k >= 1 and epsilon_allard > 0 required".to_string(),
            ));
        }
        Ok(Self {
            dim_k,
            epsilon_allard,
        })
    }

    /// Evaluates the $k$-dimensional density ratio $\Theta_k(r) = \mu(B_r) / (\omega_k r^k)$ (with the exact unit ball volume $\omega_k = \pi^{k/2} / \Gamma(k/2 + 1)$) and checks Allard regularity ($1 \le \Theta_k < 1 + \varepsilon$).
    pub fn density_ratio_and_regularity_check(
        &self,
        radius_r: f64,
        varifold_mass_br: f64,
    ) -> Result<(f64, bool)> {
        if radius_r <= 0.0 || varifold_mass_br <= 0.0 {
            return Err(Error::Msg(
                "radius_r and varifold_mass_br must be > 0".to_string(),
            ));
        }
        let omega_k = unit_ball_volume(self.dim_k);
        let theta = varifold_mass_br / (omega_k * radius_r.powf(self.dim_k as f64));
        let is_reg = theta >= 1.0 - 1e-12 && theta < 1.0 + self.epsilon_allard;
        Ok((theta, is_reg))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    /// A mass equal to the volume of the radius-1.5 ball has density ratio 1, also for k >= 5 (omega_5 = 8 pi^2 / 15).
    #[test]
    fn ball_mass_has_density_one() {
        let r: f64 = 1.5;
        let balls = [
            (1, 2.0 * r),
            (2, PI * r.powi(2)),
            (3, 4.0 * PI / 3.0 * r.powi(3)),
            (4, PI * PI / 2.0 * r.powi(4)),
            (5, 8.0 * PI * PI / 15.0 * r.powi(5)),
        ];
        for (k, mass) in balls {
            let (theta, ok) = AllardVarifoldMonotonicityEngine::new(k, 0.05)
                .unwrap()
                .density_ratio_and_regularity_check(r, mass)
                .unwrap();
            assert!((theta - 1.0).abs() < 1e-12 && ok, "k={k}: theta={theta}");
        }
    }

    /// An absurd dimension must answer immediately (it used to loop k/2 times).
    #[test]
    fn huge_dimension_does_not_hang() {
        let e = AllardVarifoldMonotonicityEngine {
            dim_k: usize::MAX / 2,
            epsilon_allard: 0.05,
        };
        assert!(e.density_ratio_and_regularity_check(1.0, 1.0).is_ok());
    }
}
