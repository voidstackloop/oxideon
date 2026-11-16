//! Information Geometry, Fisher-Rao Riemannian Metric, $\alpha$-Connections &
//! Amari Dual-Flat Bregman Manifolds (Amari, 2016).
//!
//! ### 1. Fisher-Rao Metric on Univariate Normal Manifold $\mathcal{N}(\mu, \sigma^2)$
//! On the statistical manifold $\mathcal{M} = \{(\mu, \sigma) \in \mathbb{R} \times \mathbb{R}_+\}$,
//! the unique Markov-invariant Chentsov-Fisher-Rao Riemannian metric is:
//! $$ds^2 = \frac{d\mu^2 + 2 d\sigma^2}{\sigma^2}$$
//! Identifying $x = \mu / \sqrt{2}$ and $y = \sigma$, this is isomorphic to the Poincaré upper half-plane
//! of constant negative Gaussian curvature $K = -1/2$, with exact Fisher-Rao geodesic distance:
//! $$d_{\text{FR}}\Big((\mu_1, \sigma_1), (\mu_2, \sigma_2)\Big) = \sqrt{2} \, \operatorname{arccosh}\left(1 + \frac{(\mu_1 - \mu_2)^2 + 2(\sigma_1 - \sigma_2)^2}{4 \sigma_1 \sigma_2}\right)$$
//!
//! ### 2. Amari Dual $(\theta, \eta)$ Coordinates & Generalized Pythagorean Theorem
//! For the discrete simplex $\Delta^{n-1}$, the canonical divergence is the Kullback-Leibler divergence:
//! $$D_{\text{KL}}(p \parallel q) = \sum_{i=1}^n p_i \ln\left(\frac{p_i}{q_i}\right)$$
//! With natural ($e$-flat) coordinates $\theta_i = \ln(p_i / p_n)$ and expectation ($m$-flat) coordinates $\eta_i = p_i$,
//! any triple of distributions $(P, Q, R)$ satisfying the dual orthogonality condition
//! $\langle \eta_P - \eta_Q, \theta_Q - \theta_R \rangle = \sum_i (p_i - q_i)(\ln q_i - \ln r_i) = 0$
//! obeys Amari's **Generalized Pythagorean Theorem**:
//! $$D_{\text{KL}}(P \parallel R) = D_{\text{KL}}(P \parallel Q) + D_{\text{KL}}(Q \parallel R)$$

use crate::error::{Error, Result};

/// Information Geometry, Fisher-Rao Geodesic & Dual-Flat Bregman Manifold Engine.
#[derive(Debug, Clone, Copy, Default)]
pub struct InformationGeometryEngine;

impl InformationGeometryEngine {
    /// Creates a new information geometry engine.
    pub fn new() -> Self {
        Self
    }

    /// Computes the exact $2 \times 2$ Fisher Information Matrix $G(\mu, \sigma) = \operatorname{diag}(1/\sigma^2, 2/\sigma^2)$
    /// for a Gaussian distribution $\mathcal{N}(\mu, \sigma^2)$.
    pub fn gaussian_fisher_matrix(_mu: f64, sigma: f64) -> Result<[[f64; 2]; 2]> {
        if sigma <= 0.0 {
            return Err(Error::Msg("sigma must be > 0".to_string()));
        }
        let inv_s2 = 1.0 / (sigma * sigma);
        Ok([[inv_s2, 0.0], [0.0, 2.0 * inv_s2]])
    }

    /// Computes the exact closed-form Fisher-Rao geodesic distance between $\mathcal{N}(\mu_1, \sigma_1^2)$
    /// and $\mathcal{N}(\mu_2, \sigma_2^2)$.
    pub fn gaussian_fisher_rao_distance(
        mu1: f64,
        sigma1: f64,
        mu2: f64,
        sigma2: f64,
    ) -> Result<f64> {
        if sigma1 <= 0.0 || sigma2 <= 0.0 {
            return Err(Error::Msg("Standard deviations must be > 0".to_string()));
        }
        let dmu = mu1 - mu2;
        let dsig = sigma1 - sigma2;
        let arg = 1.0 + (dmu * dmu + 2.0 * dsig * dsig) / (4.0 * sigma1 * sigma2);
        Ok(std::f64::consts::SQRT_2 * arg.max(1.0).acosh())
    }

    /// Computes the exact Fisher-Rao spherical geodesic (Bhattacharyya angle) distance on the probability simplex:
    ///
    /// $$d_{\text{FR}}(p, q) = 2 \arccos\left( \sum_{i=1}^n \sqrt{p_i q_i} \right)$$
    pub fn categorical_fisher_rao_distance(p: &[f64], q: &[f64]) -> Result<f64> {
        if p.is_empty() || p.len() != q.len() {
            return Err(Error::Msg(
                "Probability vectors must have equal non-zero length".to_string(),
            ));
        }
        let mut bc = 0.0;
        for (&pi, &qi) in p.iter().zip(q.iter()) {
            if pi < 0.0 || qi < 0.0 {
                return Err(Error::Msg("Probabilities must be non-negative".to_string()));
            }
            bc += (pi * qi).sqrt();
        }
        Ok(2.0 * bc.clamp(-1.0, 1.0).acos())
    }

    /// Computes the $\alpha$-geodesic interpolation $p^{(\alpha)}(t)$ ($t \in [0, 1]$) between two categorical distributions:
    /// • For $\alpha = -1$ ($m$-connection): linear mixture $p(t) = (1 - t) p + t q$
    /// • For $\alpha = +1$ ($e$-connection): exponential (log-linear) mixture $p_i(t) \propto p_i^{1-t} q_i^t$
    /// • For general $\alpha \neq \pm 1$: normalized power mean $\left((1-t) p_i^{(1-\alpha)/2} + t q_i^{(1-\alpha)/2}\right)^{2/(1-\alpha)}$
    pub fn alpha_geodesic_interpolate(
        p: &[f64],
        q: &[f64],
        alpha: f64,
        t: f64,
    ) -> Result<Vec<f64>> {
        if p.is_empty() || p.len() != q.len() {
            return Err(Error::Msg("Distribution length mismatch".to_string()));
        }
        let n = p.len();
        let mut out = vec![0.0; n];

        if (alpha - 1.0).abs() < 1e-9 {
            // e-geodesic (exponential family mixture)
            for (i, out_i) in out.iter_mut().enumerate().take(n) {
                *out_i = p[i].max(1e-15).powf(1.0 - t) * q[i].max(1e-15).powf(t);
            }
        } else {
            let power = 0.5 * (1.0 - alpha);
            let inv_power = 1.0 / power;
            for (i, out_i) in out.iter_mut().enumerate().take(n) {
                let term =
                    (1.0 - t) * p[i].max(1e-15).powf(power) + t * q[i].max(1e-15).powf(power);
                *out_i = term.max(1e-15).powf(inv_power);
            }
        }

        let sum: f64 = out.iter().sum();
        if sum > 0.0 {
            for val in &mut out {
                *val /= sum;
            }
        }
        Ok(out)
    }

    /// Evaluates Amari's Generalized Pythagorean residual:
    /// $|D_{\text{KL}}(P \parallel R) - (D_{\text{KL}}(P \parallel Q) + D_{\text{KL}}(Q \parallel R)) - \langle \eta_P - \eta_Q, \theta_Q - \theta_R \rangle|$
    /// which is identically zero for all probability triples $(P, Q, R)$ on the simplex!
    pub fn pythagorean_duality_residual(p: &[f64], q: &[f64], r: &[f64]) -> Result<f64> {
        if p.is_empty() || p.len() != q.len() || p.len() != r.len() {
            return Err(Error::Msg("Distribution dimension mismatch".to_string()));
        }
        let kl = |a: &[f64], b: &[f64]| -> f64 {
            a.iter()
                .zip(b.iter())
                .map(|(&ai, &bi)| {
                    if ai > 1e-15 {
                        ai * (ai / bi.max(1e-15)).ln()
                    } else {
                        0.0
                    }
                })
                .sum()
        };

        let d_pr = kl(p, r);
        let d_pq = kl(p, q);
        let d_qr = kl(q, r);

        let mut dual_inner = 0.0;
        for (i, &p_i) in p.iter().enumerate() {
            let d_eta = p_i - q[i];
            let d_theta = q[i].max(1e-15).ln() - r[i].max(1e-15).ln();
            dual_inner += d_eta * d_theta;
        }

        Ok((d_pr - (d_pq + d_qr + dual_inner)).abs())
    }
}
