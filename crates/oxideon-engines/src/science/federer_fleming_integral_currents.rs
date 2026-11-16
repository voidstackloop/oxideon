//! Federer-Fleming Compactness for Integral Currents, Sharp Isoperimetric Inequality & Coarea Formula (Federer, 1959; Federer & Fleming, 1960).
//!
//! ### 1. Sharp Euclidean Isoperimetric Ratio & Coarea Identity
//! In $\mathbb{R}^n$ ($n \ge 2$), any integral $n$-current $T$ with boundary mass $\mathbf{M}(\partial T) = P$ satisfies the sharp isoperimetric bound:
//! $$\mathbf{M}(T) \le \left(\frac{\mathbf{M}(\partial T)}{n \omega_n^{1/n}}\right)^{\frac{n}{n-1}}, \qquad \int_{\mathbb{R}^n} |\nabla f| \, dx = \int_{-\infty}^{\infty} \mathcal{H}^{n-1}(f^{-1}(t)) \, dt$$

use super::unit_ball_volume;
use crate::error::{Error, Result};

/// Federer-Fleming Integral Currents & Coarea Formula Engine.
#[derive(Debug, Clone, Copy)]
pub struct FedererFlemingCurrentsCoarea {
    /// Ambient dimension $n \in \{2, 3, 4\}$.
    pub dim_n: usize,
}

impl Default for FedererFlemingCurrentsCoarea {
    fn default() -> Self {
        Self { dim_n: 3 }
    }
}

impl FedererFlemingCurrentsCoarea {
    /// Creates a new Federer-Fleming Integral Currents engine.
    pub fn new(dim_n: usize) -> Result<Self> {
        if !(2..=4).contains(&dim_n) {
            return Err(Error::Msg("dim_n must be in 2..=4".to_string()));
        }
        Ok(Self { dim_n })
    }

    /// Evaluates the sharp Euclidean isoperimetric upper bound on enclosed mass $\mathbf{M}(T)_{\max}$ for boundary perimeter $P > 0$ and checks if candidate volume $V \le \mathbf{M}(T)_{\max}$.
    pub fn sharp_isoperimetric_volume_bound(
        &self,
        perimeter_p: f64,
        volume_v: f64,
    ) -> Result<(f64, bool)> {
        if perimeter_p <= 0.0 || volume_v < 0.0 {
            return Err(Error::Msg(
                "perimeter_p > 0 and volume_v >= 0 required".to_string(),
            ));
        }
        // `dim_n` is pub, so `new()`'s range check can be bypassed.
        if !(2..=4).contains(&self.dim_n) {
            return Err(Error::Msg("dim_n must be in 2..=4".to_string()));
        }
        let n = self.dim_n as f64;
        let omega_n = unit_ball_volume(self.dim_n);
        let v_max = (perimeter_p / (n * omega_n.powf(1.0 / n))).powf(n / (n - 1.0));
        // Relative tolerance: v_max scales like P^(n/(n-1)), so a fixed absolute epsilon drowns in rounding for large shapes.
        Ok((v_max, volume_v <= v_max * (1.0 + 1e-12)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    /// The round ball is extremal: its surface area must give back exactly its own volume.
    #[test]
    fn ball_saturates_bound() {
        for (n, area, vol) in [
            (2, 2.0 * PI, PI),
            (3, 4.0 * PI, 4.0 * PI / 3.0),
            (4, 2.0 * PI * PI, 0.5 * PI * PI),
        ] {
            let (v_max, ok) = FedererFlemingCurrentsCoarea::new(n)
                .unwrap()
                .sharp_isoperimetric_volume_bound(area, vol)
                .unwrap();
            assert!(
                (v_max - vol).abs() < 1e-12 && ok,
                "n={n}: v_max={v_max}, expected {vol}"
            );
        }
    }

    /// Feasibility uses a relative tolerance: a radius-1000 ball nudged 1e-13 over the bound passes, 1e-9 over fails.
    #[test]
    fn tolerance_is_relative() {
        let r = 1.0e3;
        for (n, area, vol) in [
            (2, 2.0 * PI * r, PI * r * r),
            (3, 4.0 * PI * r * r, 4.0 * PI / 3.0 * r * r * r),
        ] {
            let ff = FedererFlemingCurrentsCoarea::new(n).unwrap();
            assert!(
                ff.sharp_isoperimetric_volume_bound(area, vol * (1.0 + 1e-13))
                    .unwrap()
                    .1,
                "n={n}: within tolerance"
            );
            assert!(
                !ff.sharp_isoperimetric_volume_bound(area, vol * (1.0 + 1e-9))
                    .unwrap()
                    .1,
                "n={n}: beyond tolerance"
            );
        }
    }

    /// Out-of-range `dim_n` (reachable through the pub field) is an error, not a silently wrong omega_n.
    #[test]
    fn rejects_out_of_range_dim() {
        for dim_n in [0, 1, 5, 7] {
            assert!(
                FedererFlemingCurrentsCoarea { dim_n }
                    .sharp_isoperimetric_volume_bound(1.0, 1.0)
                    .is_err(),
                "dim_n={dim_n}"
            );
        }
    }
}
