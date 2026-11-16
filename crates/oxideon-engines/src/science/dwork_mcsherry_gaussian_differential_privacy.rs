//! Dwork-McSherry-Nissim-Smith $(\varepsilon, \delta)$-Differential Privacy & Gaussian Mechanism (Dwork, McSherry, Nissim & Smith, 2006; Dwork & Roth, 2014).
//!
//! ### 1. Classical Gaussian Mechanism Calibration $\sigma \ge \frac{\Delta_2 f}{\varepsilon} \sqrt{2 \ln(1.25 / \delta)}$
//! For a query $f$ with $\ell_2$-sensitivity $\Delta_2 f = \sup_{D \sim D'} \|f(D) - f(D')\|_2$, adding $\mathcal{N}(0, \sigma^2 I)$ achieves $(\varepsilon, \delta)$-DP for $\varepsilon \in (0, 1)$ when:
//! $$\sigma_{\min}(\varepsilon, \delta) = \frac{\Delta_2 f}{\varepsilon} \sqrt{2 \ln\!\left(\frac{1.25}{\delta}\right)}$$

use crate::error::{Error, Result};

/// Dwork $(\varepsilon, \delta)$-Differential Privacy Gaussian Mechanism Engine.
#[derive(Debug, Clone, Copy)]
pub struct DworkGaussianDifferentialPrivacy {
    /// $\ell_2$-sensitivity $\Delta_2 f > 0$.
    pub l2_sensitivity: f64,
}

impl Default for DworkGaussianDifferentialPrivacy {
    fn default() -> Self {
        Self {
            l2_sensitivity: 1.0,
        }
    }
}

impl DworkGaussianDifferentialPrivacy {
    /// Creates a new Dwork Gaussian Differential Privacy engine.
    pub fn new(l2_sensitivity: f64) -> Result<Self> {
        if l2_sensitivity <= 0.0 || !l2_sensitivity.is_finite() {
            return Err(Error::Msg("l2_sensitivity must be > 0".to_string()));
        }
        Ok(Self { l2_sensitivity })
    }

    /// Evaluates the calibrated Gaussian noise standard deviation $\sigma_{\min}(\varepsilon, \delta)$ and advanced composition $\varepsilon_{\mathrm{tot}} \approx \sqrt{2 k \ln(1/\delta')} \varepsilon + k \varepsilon(e^\varepsilon - 1)$.
    pub fn gaussian_sigma_and_composed_epsilon(
        &self,
        epsilon: f64,
        delta: f64,
        steps_k: usize,
    ) -> Result<(f64, f64)> {
        if !(0.0..=1.0).contains(&epsilon)
            || epsilon == 0.0
            || !(0.0..1.0).contains(&delta)
            || delta == 0.0
            || steps_k == 0
        {
            return Err(Error::Msg(
                "epsilon in (0, 1], delta in (0, 1), steps_k >= 1 required".to_string(),
            ));
        }
        let sigma = (self.l2_sensitivity / epsilon) * (2.0 * (1.25 / delta).ln()).sqrt();
        let kf = steps_k as f64;
        let eps_tot =
            (2.0 * kf * (1.0 / delta).ln()).sqrt() * epsilon + kf * epsilon * (epsilon.exp() - 1.0);
        Ok((sigma, eps_tot))
    }
}
