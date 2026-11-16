//! Gross Logarithmic Sobolev Inequality (LSI), Bakry-Émery $\mathrm{CD}(\rho, \infty)$ Curvature & Herbst Sub-Gaussian Concentration (Gross, 1975; Talagrand, 1996).
//!
//! ### 1. Bakry-Émery Criterion & Herbst Concentration Argument
//! A probability measure $d\mu(x) = Z^{-1} e^{-V(x)} dx$ with strongly convex potential $\nabla^2 V(x) \succeq \rho I_d$ ($\rho > 0$)
//! satisfies the **Logarithmic Sobolev Inequality** $\operatorname{Ent}_\mu(f^2) \le \frac{2}{\rho} \int \|\nabla f\|_2^2 d\mu$
//! and Talagrand's quadratic transport inequality $W_2^2(\nu, \mu) \le \frac{2}{\rho} \mathrm{KL}(\nu \| \mu)$.
//! By Herbst's argument, any $L$-Lipschitz observable $f$ obeys sub-Gaussian tail concentration:
//! $$\mathbb{P}_\mu\left(f(X) - \mathbb{E}_\mu\[f\] \ge t\right) \le \exp\left(-\frac{\rho \, t^2}{2 L^2}\right)$$

use crate::error::{Error, Result};

/// Gross Log-Sobolev, Bakry-Émery Curvature & Talagrand Concentration Engine.
#[derive(Debug, Clone, Copy)]
pub struct LogSobolevConcentrationEngine {
    /// Bakry-Émery strong convexity / curvature parameter $\rho > 0$ (so $c_{\mathrm{LSI}} = 1 / \rho$).
    pub curvature_rho: f64,
}

impl LogSobolevConcentrationEngine {
    /// Creates a new Log-Sobolev Concentration engine.
    pub fn new(curvature_rho: f64) -> Result<Self> {
        if curvature_rho <= 0.0 {
            return Err(Error::Msg("curvature_rho must be > 0".to_string()));
        }
        Ok(Self { curvature_rho })
    }

    /// Evaluates the Herbst sub-Gaussian tail probability bound $\exp(-\rho t^2 / (2 L^2))$ and Talagrand $W_2$ transport bound $\sqrt{2 \mathrm{KL} / \rho}$.
    pub fn herbst_tail_and_talagrand_w2(
        &self,
        deviation_t: f64,
        lipschitz_l: f64,
        kl_div: f64,
    ) -> Result<(f64, f64)> {
        if deviation_t < 0.0 || lipschitz_l <= 0.0 || kl_div < 0.0 {
            return Err(Error::Msg(
                "Requires deviation_t >= 0, lipschitz_l > 0, and kl_div >= 0".to_string(),
            ));
        }
        let tail = (-0.5 * self.curvature_rho * deviation_t * deviation_t
            / (lipschitz_l * lipschitz_l))
            .exp();
        let w2_bound = (2.0 * kl_div / self.curvature_rho).sqrt();
        Ok((tail, w2_bound))
    }
}
