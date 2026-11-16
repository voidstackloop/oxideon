//! Continuous-Variable (CV) Optical Quantum Computing & Gaussian Boson Sampling (GBS).
//!
//! # Mathematical & Physical Foundations
//!
//! 1. **Symplectic Phase Space & Gaussian States**:
//!    An $M$-mode bosonic quantum system is described by $2M$ canonical quadrature operators
//!    $\hat{r} = (\hat{q}_1, \hat{p}_1, \dots, \hat{q}_M, \hat{p}_M)^T$ with $[\hat{q}_j, \hat{p}_k] = 2i \delta_{jk}$ ($\hbar = 2$ convention).
//!    A Gaussian state is completely characterized by:
//!    - First-moment displacement vector $\bar{r} = \langle \hat{r} \rangle \in \mathbb{R}^{2M}$
//!    - Second-moment $2M \times 2M$ Wigner covariance matrix $V_{ij} = \frac{1}{2}\langle \{\Delta \hat{r}_i, \Delta \hat{r}_j\} \rangle$
//!
//!    For the vacuum state $|0\rangle^{\otimes M}$, $\bar{r} = \mathbf{0}$ and $V = I_{2M}$.
//!
//! 2. **Gaussian Unitary Evolution ($Sp(2M, \mathbb{R})$)**:
//!    Gaussian unitaries correspond to symplectic matrices $S \in Sp(2M, \mathbb{R})$ preserving the symplectic form
//!    $S \Omega S^T = \Omega$, transforming moments via:
//!    $$\bar{r} \mapsto S \bar{r} + d, \qquad V \mapsto S V S^T$$
//!    - **Single-Mode Squeezing $\hat{S}(z)$**: $S_{\text{sq}}(r) = \operatorname{diag}(e^{-r}, e^{r})$
//!    - **Phase-Space Rotation $\hat{R}(\phi)$**: $2\times 2$ orthogonal rotation on $(q_m, p_m)$
//!    - **Two-Mode Beamsplitter $\hat{B}(\theta)$**: mixes modes $(m_1, m_2)$ with transmissivity $\tau = \cos^2\theta$
//!
//! 3. **Gaussian Boson Sampling (GBS) & Matrix Hafnian**:
//!    - **Vacuum Probability**: $P(\mathbf{0}) = \frac{1}{\sqrt{\det(V_Q)}}$ (for zero displacement), where
//!      $V_Q = \frac{1}{2}(V + I_{2M})$ is the Husimi $Q$-function covariance matrix.
//!    - **Matrix Hafnian**: For a symmetric $2K \times 2K$ matrix $A$, the Hafnian sums products over all
//!      $(2K - 1)!!$ perfect matchings $\mathcal{M}_{2K}$:
//!      $$\operatorname{haf}(A) = \sum_{\mu \in \mathcal{M}_{2K}} \prod_{(i, j) \in \mu} A_{ij}$$

use crate::error::{Error, Result};

/// Continuous-Variable Gaussian quantum state simulator over $M$ optical qumodes.
#[derive(Debug, Clone)]
pub struct GaussianCvCircuit {
    /// Number of optical qumodes $M$.
    pub num_modes: usize,
    /// Mean quadrature displacement vector $\bar{r} = (\bar{q}_1, \bar{p}_1, \dots, \bar{q}_M, \bar{p}_M)$ of length $2M$.
    pub displacement: Vec<f64>,
    /// Symplectic Wigner covariance matrix $V$ of shape $[2M, 2M]$ (flattened row-major).
    pub covariance: Vec<f64>,
}

impl GaussianCvCircuit {
    /// Initializes an $M$-mode optical register in the multi-mode vacuum state $|0\rangle^{\otimes M}$
    /// ($\bar{r} = 0$, $V = I_{2M}$).
    pub fn new(num_modes: usize) -> Result<Self> {
        if num_modes == 0 || num_modes > 32 {
            return Err(Error::Msg(format!(
                "Invalid num_modes {num_modes} (must be in 1..=32)"
            )));
        }
        let dim = 2 * num_modes;
        let displacement = vec![0.0; dim];
        let mut covariance = vec![0.0; dim * dim];
        for i in 0..dim {
            covariance[i * dim + i] = 1.0;
        }
        Ok(Self {
            num_modes,
            displacement,
            covariance,
        })
    }

    /// Verifies the quadrature buffers agree with the (public, tamperable)
    /// mode count before any indexing.
    fn check_layout(&self) -> Result<usize> {
        let dim = self
            .num_modes
            .checked_mul(2)
            .ok_or_else(|| Error::Msg(format!("CV num_modes {} overflows 2*M", self.num_modes)))?;
        if self.displacement.len() != dim {
            return Err(Error::Msg(format!(
                "CV displacement length {} disagrees with 2*{}",
                self.displacement.len(),
                self.num_modes
            )));
        }
        let cells = dim
            .checked_mul(dim)
            .ok_or_else(|| Error::Msg(format!("CV covariance size {dim} * {dim} overflows")))?;
        if self.covariance.len() != cells {
            return Err(Error::Msg(format!(
                "CV covariance length {} disagrees with {dim}*{dim}",
                self.covariance.len()
            )));
        }
        Ok(dim)
    }

    fn check_mode(&self, mode: usize) -> Result<()> {
        self.check_layout()?;
        if mode >= self.num_modes {
            return Err(Error::Msg(format!(
                "Mode index {mode} out of range for {} modes",
                self.num_modes
            )));
        }
        Ok(())
    }

    /// Applies a single-mode quadrature displacement gate $\hat{D}_m(\delta_q, \delta_p)$.
    pub fn displace(&mut self, mode: usize, delta_q: f64, delta_p: f64) -> Result<()> {
        self.check_mode(mode)?;
        self.displacement[2 * mode] += delta_q;
        self.displacement[2 * mode + 1] += delta_p;
        Ok(())
    }

    /// Applies a single-mode squeezing gate $\hat{S}_m(r)$ along the $(q_m, p_m)$ axes:
    /// $q_m \mapsto e^{-r} q_m$, $p_m \mapsto e^{r} p_m$.
    pub fn squeeze(&mut self, mode: usize, r: f64) -> Result<()> {
        self.check_mode(mode)?;
        let dim = 2 * self.num_modes;
        let mut s = vec![0.0; dim * dim];
        for i in 0..dim {
            s[i * dim + i] = 1.0;
        }
        let iq = 2 * mode;
        let ip = 2 * mode + 1;
        s[iq * dim + iq] = (-r).exp();
        s[ip * dim + ip] = r.exp();
        self.apply_symplectic(&s);
        Ok(())
    }

    /// Applies a single-mode phase rotation gate $\hat{R}_m(\phi)$.
    pub fn phase_shift(&mut self, mode: usize, phi: f64) -> Result<()> {
        self.check_mode(mode)?;
        let dim = 2 * self.num_modes;
        let mut s = vec![0.0; dim * dim];
        for i in 0..dim {
            s[i * dim + i] = 1.0;
        }
        let iq = 2 * mode;
        let ip = 2 * mode + 1;
        let c = phi.cos();
        let sn = phi.sin();
        s[iq * dim + iq] = c;
        s[iq * dim + ip] = sn;
        s[ip * dim + iq] = -sn;
        s[ip * dim + ip] = c;
        self.apply_symplectic(&s);
        Ok(())
    }

    /// Applies a two-mode beamsplitter $\hat{B}_{m_1, m_2}(\theta)$ with transmissivity $\tau = \cos^2\theta$.
    pub fn beamsplitter(&mut self, mode1: usize, mode2: usize, theta: f64) -> Result<()> {
        self.check_mode(mode1)?;
        self.check_mode(mode2)?;
        if mode1 == mode2 {
            return Err(Error::Msg(
                "Beamsplitter requires two distinct modes".to_string(),
            ));
        }

        let dim = 2 * self.num_modes;
        let mut s = vec![0.0; dim * dim];
        for i in 0..dim {
            s[i * dim + i] = 1.0;
        }
        let c = theta.cos();
        let sn = theta.sin();

        let q1 = 2 * mode1;
        let p1 = 2 * mode1 + 1;
        let q2 = 2 * mode2;
        let p2 = 2 * mode2 + 1;

        s[q1 * dim + q1] = c;
        s[q1 * dim + q2] = sn;
        s[q2 * dim + q1] = -sn;
        s[q2 * dim + q2] = c;

        s[p1 * dim + p1] = c;
        s[p1 * dim + p2] = sn;
        s[p2 * dim + p1] = -sn;
        s[p2 * dim + p2] = c;

        self.apply_symplectic(&s);
        Ok(())
    }

    /// Applies a $2M \times 2M$ symplectic transformation $S$:
    /// $\bar{r} \leftarrow S \bar{r}$ and $V \leftarrow S V S^T$.
    fn apply_symplectic(&mut self, s: &[f64]) {
        let dim = 2 * self.num_modes;

        // 1. Update displacement: r_new = S * r
        let mut new_disp = vec![0.0; dim];
        for i in 0..dim {
            let mut sum = 0.0;
            for j in 0..dim {
                sum += s[i * dim + j] * self.displacement[j];
            }
            new_disp[i] = sum;
        }
        self.displacement = new_disp;

        // 2. Update covariance: temp = S * V, then V_new = temp * S^T
        let mut temp = vec![0.0; dim * dim];
        for i in 0..dim {
            for j in 0..dim {
                let mut sum = 0.0;
                for k in 0..dim {
                    sum += s[i * dim + k] * self.covariance[k * dim + j];
                }
                temp[i * dim + j] = sum;
            }
        }

        let mut new_cov = vec![0.0; dim * dim];
        for i in 0..dim {
            for j in 0..dim {
                let mut sum = 0.0;
                for k in 0..dim {
                    sum += temp[i * dim + k] * s[j * dim + k];
                }
                new_cov[i * dim + j] = sum;
            }
        }
        self.covariance = new_cov;
    }

    /// Computes the exact expected photon number $\langle \hat{n}_m \rangle$ in `mode`:
    ///
    /// $$\langle \hat{n}_m \rangle = \frac{1}{4}\left(V_{q_m, q_m} + V_{p_m, p_m} + \bar{q}_m^2 + \bar{p}_m^2\right) - \frac{1}{2}$$
    pub fn mean_photon_number(&self, mode: usize) -> Result<f64> {
        self.check_mode(mode)?;
        let dim = 2 * self.num_modes;
        let iq = 2 * mode;
        let ip = 2 * mode + 1;

        let v_qq = self.covariance[iq * dim + iq];
        let v_pp = self.covariance[ip * dim + ip];
        let q_bar = self.displacement[iq];
        let p_bar = self.displacement[ip];

        Ok(0.25 * (v_qq + v_pp + q_bar * q_bar + p_bar * p_bar) - 0.5)
    }

    /// Computes total mean photon number $\sum_{m=0}^{M-1} \langle \hat{n}_m \rangle$ across all modes.
    pub fn total_mean_photon_number(&self) -> Result<f64> {
        let mut total = 0.0;
        for m in 0..self.num_modes {
            total += self.mean_photon_number(m)?;
        }
        Ok(total)
    }

    /// Computes the purity $\mu = \operatorname{Tr}(\hat{\rho}^2) = \frac{1}{\sqrt{\det(V)}} \in (0, 1]$.
    /// For any pure Gaussian state evolved by symplectic unitaries, $\det(V) = 1 \implies \mu = 1$.
    pub fn purity(&self) -> Result<f64> {
        let dim = self.check_layout()?;
        let det_v = determinant(&self.covariance, dim)?;
        if det_v <= 0.0 {
            return Err(Error::Msg(format!(
                "Non-positive covariance determinant: {det_v}"
            )));
        }
        Ok(1.0 / det_v.sqrt())
    }

    /// Computes the all-zero photon-click probability $P(0, 0, \dots, 0)$ (vacuum overlap) for
    /// zero-displacement Gaussian Boson Sampling:
    ///
    /// $$P(\mathbf{0}) = \frac{1}{\sqrt{\det(V_Q)}}, \qquad V_Q = \frac{1}{2}(V + I_{2M})$$
    pub fn vacuum_probability(&self) -> Result<f64> {
        let dim = self.check_layout()?;
        let mut v_q = vec![0.0; dim * dim];
        for i in 0..dim {
            for j in 0..dim {
                let diag = if i == j { 1.0 } else { 0.0 };
                v_q[i * dim + j] = 0.5 * (self.covariance[i * dim + j] + diag);
            }
        }
        let det_vq = determinant(&v_q, dim)?;
        if det_vq <= 0.0 {
            return Err(Error::Msg(format!(
                "Invalid Husimi covariance determinant: {det_vq}"
            )));
        }
        Ok(1.0 / det_vq.sqrt())
    }
}

/// Computes the determinant of an $N \times N$ matrix via Gaussian elimination with partial pivoting.
fn determinant(matrix: &[f64], n: usize) -> Result<f64> {
    let need = n
        .checked_mul(n)
        .ok_or_else(|| Error::Msg(format!("determinant: n = {n} overflows n * n")))?;
    if matrix.len() != need {
        return Err(Error::Msg(
            "Matrix size mismatch in determinant".to_string(),
        ));
    }
    let mut a = matrix.to_vec();
    let mut det = 1.0;

    for i in 0..n {
        let mut pivot = i;
        let mut max_val = a[i * n + i].abs();
        for r in (i + 1)..n {
            let val = a[r * n + i].abs();
            if val > max_val {
                max_val = val;
                pivot = r;
            }
        }
        if max_val < 1e-15 {
            return Ok(0.0);
        }
        if pivot != i {
            for c in 0..n {
                a.swap(i * n + c, pivot * n + c);
            }
            det = -det;
        }
        let diag = a[i * n + i];
        det *= diag;
        for r in (i + 1)..n {
            let factor = a[r * n + i] / diag;
            for c in (i + 1)..n {
                a[r * n + c] -= factor * a[i * n + c];
            }
        }
    }
    Ok(det)
}

/// Computes the exact matrix Hafnian $\operatorname{haf}(A)$ of an even-dimensional $N \times N$ symmetric matrix.
///
/// The Hafnian sums products of edge weights over all $(N - 1)!!$ perfect matchings of the complete graph $K_N$:
///
/// $$\operatorname{haf}(A) = \sum_{\mu \in \mathcal{M}_N} \prod_{(u, v) \in \mu} A_{uv}$$
pub fn matrix_hafnian(matrix: &[f64], n: usize) -> Result<f64> {
    let need = n
        .checked_mul(n)
        .ok_or_else(|| Error::Msg(format!("matrix_hafnian: n = {n} overflows n * n")))?;
    if matrix.len() != need {
        return Err(Error::Msg(format!(
            "Matrix length {} does not match n^2 = {}",
            matrix.len(),
            need
        )));
    }
    if n == 0 {
        return Ok(1.0);
    }
    if n % 2 != 0 {
        return Ok(0.0);
    }
    if n > 16 {
        return Err(Error::Msg(
            "Exact matrix_hafnian supports n <= 16".to_string(),
        ));
    }

    fn recurse_matchings(matrix: &[f64], n: usize, mask: u32) -> f64 {
        let full_mask = (1u32 << n) - 1;
        if mask == full_mask {
            return 1.0;
        }
        // Find first unmatched vertex u
        let u = (!mask).trailing_zeros() as usize;
        let mask_with_u = mask | (1u32 << u);

        let mut sum = 0.0;
        for v in (u + 1)..n {
            if (mask_with_u & (1u32 << v)) == 0 {
                let w = matrix[u * n + v];
                if w.abs() > 0.0 {
                    sum += w * recurse_matchings(matrix, n, mask_with_u | (1u32 << v));
                }
            }
        }
        sum
    }

    Ok(recurse_matchings(matrix, n, 0))
}
