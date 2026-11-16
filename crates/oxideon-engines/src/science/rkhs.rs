//! Infinite-Dimensional Reproducing Kernel Hilbert Spaces (RKHS) & Maximum Mean Discrepancy (MMD).
//!
//! Provides non-parametric functional estimation in Hilbert spaces:
//! - Mercer Kernels: Gaussian RBF, Matérn 3/2, Matérn 5/2, Polynomial.
//! - Random Fourier Features (RFF) for Mercer spectral approximations $\phi(x)^T \phi(y) \approx k(x, y)$.
//! - Kernel Ridge Regression (KRR) with analytical regularization $(K + \lambda I)^{-1} y$.
//! - Two-Sample Maximum Mean Discrepancy (MMD) test statistic for distribution alignment.

use crate::error::{Error, Result};
use crate::tensor::Tensor;

/// Mercer Kernel families supported in RKHS.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum MercerKernel {
    /// Radial Basis Function (Gaussian): $k(x, y) = \exp(-\gamma \|x - y\|^2)$.
    GaussianRbf { gamma: f64 },
    /// Matérn 3/2: $k(x, y) = (1 + \sqrt{3} r / \ell) \exp(-\sqrt{3} r / \ell)$.
    Matern32 { length_scale: f64 },
    /// Matérn 5/2: $k(x, y) = (1 + \sqrt{5} r / \ell + \frac{5}{3} r^2 / \ell^2) \exp(-\sqrt{5} r / \ell)$.
    Matern52 { length_scale: f64 },
    /// Linear Kernel: $k(x, y) = \langle x, y \rangle$.
    Linear,
    /// Polynomial Kernel: $k(x, y) = (\gamma \langle x, y \rangle + c)^d$.
    Polynomial { gamma: f64, c: f64, degree: usize },
}

impl MercerKernel {
    /// Computes pairwise kernel evaluation $k(x, y)$.
    pub fn eval(&self, x: &[f64], y: &[f64]) -> Result<f64> {
        if x.len() != y.len() {
            return Err(Error::ShapeMismatch {
                op: "MercerKernel::eval",
                expected: vec![x.len()],
                got: vec![y.len()],
            });
        }

        match self {
            Self::GaussianRbf { gamma } => {
                let dist_sq: f64 = x
                    .iter()
                    .zip(y.iter())
                    .map(|(&a, &b)| (a - b) * (a - b))
                    .sum();
                Ok((-gamma * dist_sq).exp())
            }
            Self::Matern32 { length_scale } => {
                let dist = x
                    .iter()
                    .zip(y.iter())
                    .map(|(&a, &b)| (a - b) * (a - b))
                    .sum::<f64>()
                    .sqrt();
                let scaled = 3.0_f64.sqrt() * dist / length_scale.max(1e-8);
                Ok((1.0 + scaled) * (-scaled).exp())
            }
            Self::Matern52 { length_scale } => {
                let dist = x
                    .iter()
                    .zip(y.iter())
                    .map(|(&a, &b)| (a - b) * (a - b))
                    .sum::<f64>()
                    .sqrt();
                let scaled = 5.0_f64.sqrt() * dist / length_scale.max(1e-8);
                Ok((1.0 + scaled + scaled * scaled / 3.0) * (-scaled).exp())
            }
            Self::Linear => {
                let dot: f64 = x.iter().zip(y.iter()).map(|(&a, &b)| a * b).sum();
                Ok(dot)
            }
            Self::Polynomial { gamma, c, degree } => {
                let dot: f64 = x.iter().zip(y.iter()).map(|(&a, &b)| a * b).sum();
                Ok((gamma * dot + c).powi(*degree as i32))
            }
        }
    }

    /// Evaluates the Gram matrix $K \in \mathbb{R}^{n \times m}$ between samples $X \in \mathbb{R}^{n \times d}$ and $Y \in \mathbb{R}^{m \times d}$.
    pub fn gram_matrix(&self, x: &[Vec<f64>], y: &[Vec<f64>]) -> Result<Vec<f64>> {
        let n = x.len();
        let m = y.len();
        let mut k = vec![0.0; n * m];

        for i in 0..n {
            for j in 0..m {
                k[i * m + j] = self.eval(&x[i], &y[j])?;
            }
        }

        Ok(k)
    }
}

/// Random Fourier Features (RFF) spectral approximation of shift-invariant kernels.
#[derive(Debug, Clone)]
pub struct RandomFourierFeatures {
    /// Dimension of input data $d$.
    pub in_dim: usize,
    /// Number of Fourier features $D$.
    pub num_features: usize,
    /// Random Gaussian frequency matrix $W \in \mathbb{R}^{D \times d}$.
    pub weights: Vec<f64>,
    /// Uniform phase offsets $b \in \mathbb{R}^D \sim [0, 2\pi]$.
    pub biases: Vec<f64>,
}

impl RandomFourierFeatures {
    /// Constructs Random Fourier Features for Gaussian RBF kernel with bandwidth $\gamma$.
    pub fn new_gaussian_rbf(
        in_dim: usize,
        num_features: usize,
        gamma: f64,
        seed: Option<u64>,
    ) -> Self {
        let mut rng_state = seed.unwrap_or(42).max(1);
        let mut next_f64 = || -> f64 {
            let mut x = rng_state;
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            rng_state = x;
            (x as f64) / (u64::MAX as f64)
        };

        // Standard deviation for RBF with parameter gamma (exp(-gamma ||x-y||^2)):
        // sigma = sqrt(2 * gamma)
        let sigma = (2.0 * gamma).sqrt();
        let mut weights = vec![0.0; num_features * in_dim];
        for w in &mut weights {
            // Box-Muller standard normal
            let u1 = next_f64().max(1e-9);
            let u2 = next_f64();
            let z0 = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
            *w = z0 * sigma;
        }

        let mut biases = vec![0.0; num_features];
        for b in &mut biases {
            *b = next_f64() * 2.0 * std::f64::consts::PI;
        }

        Self {
            in_dim,
            num_features,
            weights,
            biases,
        }
    }

    /// Embeds an input vector $x \in \mathbb{R}^d$ into RKHS feature space $\phi(x) \in \mathbb{R}^D$:
    /// $$\phi(x) = \sqrt{\frac{2}{D}} \cos(W x + b)$$
    pub fn transform(&self, x: &[f64]) -> Result<Vec<f64>> {
        if x.len() != self.in_dim {
            return Err(Error::ShapeMismatch {
                op: "RandomFourierFeatures::transform",
                expected: vec![self.in_dim],
                got: vec![x.len()],
            });
        }
        let need_w = self.num_features.checked_mul(self.in_dim).ok_or_else(|| {
            Error::Msg("RandomFourierFeatures::transform: weight size overflows".into())
        })?;
        if self.biases.len() != self.num_features || self.weights.len() != need_w {
            return Err(Error::Msg(format!(
                "RandomFourierFeatures::transform: need {need_w} weights and {} biases, got {}/{}",
                self.num_features,
                self.weights.len(),
                self.biases.len()
            )));
        }

        let d = self.num_features;
        let scale = (2.0 / d as f64).sqrt();
        let mut phi = vec![0.0; d];

        for (i, phi_val) in phi.iter_mut().enumerate().take(d) {
            let mut dot = self.biases[i];
            for (j, &xj) in x.iter().enumerate().take(self.in_dim) {
                dot += self.weights[i * self.in_dim + j] * xj;
            }
            *phi_val = scale * dot.cos();
        }

        Ok(phi)
    }

    /// Embeds a batch tensor `[N, d]` into feature matrix `[N, D]`.
    pub fn transform_tensor(&self, x: &Tensor) -> Result<Tensor> {
        let shape = x.shape();
        if shape.len() != 2 || shape[1] != self.in_dim {
            return Err(Error::ShapeMismatch {
                op: "transform_tensor",
                expected: vec![0, self.in_dim],
                got: shape.to_vec(),
            });
        }
        let n = shape[0];
        let data = x.to_vec::<f64>()?;
        let mut out = Vec::with_capacity(n * self.num_features);
        for i in 0..n {
            let row = &data[i * self.in_dim..(i + 1) * self.in_dim];
            let phi = self.transform(row)?;
            out.extend_from_slice(&phi);
        }
        Tensor::from_slice(&out, &[n, self.num_features])
    }
}

/// Kernel Ridge Regression (KRR) in RKHS.
#[derive(Debug, Clone)]
pub struct KernelRidgeRegressor {
    /// Chosen Mercer kernel.
    pub kernel: MercerKernel,
    /// L2 regularization parameter $\lambda > 0$.
    pub lambda: f64,
    /// Training points $X_{\text{train}}$.
    pub train_x: Vec<Vec<f64>>,
    /// Dual solution coefficients $\alpha \in \mathbb{R}^n$.
    pub dual_coeffs: Vec<f64>,
}

impl KernelRidgeRegressor {
    /// Creates a new Kernel Ridge Regressor.
    pub fn new(kernel: MercerKernel, lambda: f64) -> Self {
        Self {
            kernel,
            lambda,
            train_x: Vec::new(),
            dual_coeffs: Vec::new(),
        }
    }

    /// Fits KRR model by solving $(K + \lambda I) \alpha = y$ via Gaussian elimination.
    pub fn fit(&mut self, x: &[Vec<f64>], y: &[f64]) -> Result<()> {
        let n = x.len();
        if n == 0 || n != y.len() {
            return Err(Error::Msg(format!(
                "Invalid data sizes for KRR fit: {n} points, {} targets",
                y.len()
            )));
        }

        let k = self.kernel.gram_matrix(x, x)?;

        // Regularized matrix A = K + lambda * I
        let mut a = vec![0.0; n * n];
        for i in 0..n {
            for j in 0..n {
                a[i * n + j] = k[i * n + j];
                if i == j {
                    a[i * n + j] += self.lambda;
                }
            }
        }

        // Solve A * alpha = y via Gaussian elimination with partial pivoting
        let mut aug = vec![0.0; n * (n + 1)];
        for i in 0..n {
            for j in 0..n {
                aug[i * (n + 1) + j] = a[i * n + j];
            }
            aug[i * (n + 1) + n] = y[i];
        }

        for i in 0..n {
            // Find pivot
            let mut max_row = i;
            let mut max_val = aug[i * (n + 1) + i].abs();
            for r in (i + 1)..n {
                let val = aug[r * (n + 1) + i].abs();
                if val > max_val {
                    max_val = val;
                    max_row = r;
                }
            }
            if max_val < 1e-12 {
                return Err(Error::Msg("Singular Gram matrix in KRR fit".into()));
            }

            // Swap rows
            if max_row != i {
                for c in 0..=(n) {
                    aug.swap(i * (n + 1) + c, max_row * (n + 1) + c);
                }
            }

            // Eliminate
            let pivot = aug[i * (n + 1) + i];
            for r in (i + 1)..n {
                let factor = aug[r * (n + 1) + i] / pivot;
                for c in i..=(n) {
                    aug[r * (n + 1) + c] -= factor * aug[i * (n + 1) + c];
                }
            }
        }

        // Back substitution
        let mut alpha = vec![0.0; n];
        for i in (0..n).rev() {
            let mut sum = aug[i * (n + 1) + n];
            for j in (i + 1)..n {
                sum -= aug[i * (n + 1) + j] * alpha[j];
            }
            alpha[i] = sum / aug[i * (n + 1) + i];
        }

        self.train_x = x.to_vec();
        self.dual_coeffs = alpha;

        Ok(())
    }

    /// Predicts target value for a query point $x^*$:
    /// $$\hat{y} = \sum_{i=1}^n \alpha_i k(x_i, x^*)$$
    pub fn predict(&self, query: &[f64]) -> Result<f64> {
        if self.train_x.is_empty() {
            return Err(Error::Msg("KRR regressor is not fitted yet".into()));
        }

        let mut pred = 0.0;
        for (i, train_pt) in self.train_x.iter().enumerate() {
            let k_val = self.kernel.eval(train_pt, query)?;
            pred += self.dual_coeffs[i] * k_val;
        }

        Ok(pred)
    }

    /// Predicts target values for query tensor of shape `[N, D]`.
    pub fn predict_tensor(&self, query: &Tensor) -> Result<Tensor> {
        let shape = query.shape();
        if shape.len() != 2 {
            return Err(Error::ShapeMismatch {
                op: "predict_tensor",
                expected: vec![0, self.train_x.first().map(|p| p.len()).unwrap_or(0)],
                got: shape.to_vec(),
            });
        }
        let n = shape[0];
        let d = shape[1];
        let data = query.to_vec::<f64>()?;
        let mut preds = Vec::with_capacity(n);
        for i in 0..n {
            let row = &data[i * d..(i + 1) * d];
            preds.push(self.predict(row)?);
        }
        Tensor::from_slice(&preds, &[n])
    }
}

/// Computes Maximum Mean Discrepancy (MMD) squared between sample sets $X \sim P$ and $Y \sim Q$:
///
/// $$\operatorname{MMD}^2(X, Y) = \frac{1}{n(n-1)} \sum_{i \ne j} k(x_i, x_j) + \frac{1}{m(m-1)} \sum_{i \ne j} k(y_i, y_j) - \frac{2}{nm} \sum_{i,j} k(x_i, y_j)$$
pub fn maximum_mean_discrepancy(
    x: &[Vec<f64>],
    y: &[Vec<f64>],
    kernel: MercerKernel,
    unbiased: bool,
) -> Result<f64> {
    let n = x.len();
    let m = y.len();
    if n < 2 || m < 2 {
        return Err(Error::Msg("MMD requires at least 2 samples per set".into()));
    }

    let k_xx = kernel.gram_matrix(x, x)?;
    let k_yy = kernel.gram_matrix(y, y)?;
    let k_xy = kernel.gram_matrix(x, y)?;

    if unbiased {
        let mut sum_xx = 0.0;
        for i in 0..n {
            for j in 0..n {
                if i != j {
                    sum_xx += k_xx[i * n + j];
                }
            }
        }
        let mean_xx = sum_xx / (n * (n - 1)) as f64;

        let mut sum_yy = 0.0;
        for i in 0..m {
            for j in 0..m {
                if i != j {
                    sum_yy += k_yy[i * m + j];
                }
            }
        }
        let mean_yy = sum_yy / (m * (m - 1)) as f64;

        let mean_xy: f64 = k_xy.iter().sum::<f64>() / (n * m) as f64;

        Ok(mean_xx + mean_yy - 2.0 * mean_xy)
    } else {
        let mean_xx: f64 = k_xx.iter().sum::<f64>() / (n * n) as f64;
        let mean_yy: f64 = k_yy.iter().sum::<f64>() / (m * m) as f64;
        let mean_xy: f64 = k_xy.iter().sum::<f64>() / (n * m) as f64;

        Ok(mean_xx + mean_yy - 2.0 * mean_xy)
    }
}
