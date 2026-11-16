//! Spatio-Temporal Graph Wavelet Neural Networks (GWNN).
//!
//! Provides multiscale spectral graph wavelets via Chebyshev polynomial
//! approximations of graph heat kernels, localized spatial graph diffusion,
//! and spatio-temporal recurrence over time-varying graph signals.

use crate::error::{Error, Result};
use crate::tensor::Tensor;

/// Type of Graph Laplacian operator.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LaplacianType {
    /// Unnormalized combinatorial Laplacian: $L = D - A$.
    Combinatorial,
    /// Symmetric normalized Laplacian: $L_{\text{sym}} = I - D^{-1/2} A D^{-1/2}$.
    SymmetricNormalized,
    /// Random walk normalized Laplacian: $L_{\text{rw}} = I - D^{-1} A$.
    RandomWalk,
}

/// Representation and spectral utilities for a Graph Laplacian.
#[derive(Debug, Clone)]
pub struct GraphLaplacian {
    /// Number of nodes $N$.
    pub num_nodes: usize,
    /// Adjacency matrix $A \in \mathbb{R}^{N \times N}$.
    pub adj: Vec<f64>,
    /// Degree vector $D \in \mathbb{R}^N$.
    pub degrees: Vec<f64>,
    /// Laplacian matrix $L \in \mathbb{R}^{N \times N}$.
    pub laplacian: Vec<f64>,
    /// Type of Laplacian.
    pub laplacian_type: LaplacianType,
    /// Estimated largest eigenvalue $\lambda_{\max}$.
    pub lambda_max: f64,
}

impl GraphLaplacian {
    /// Constructs a graph Laplacian from an adjacency matrix.
    pub fn new(adj: &[f64], num_nodes: usize, laplacian_type: LaplacianType) -> Result<Self> {
        // Checked: an overflowing num_nodes^2 can never match a real slice length.
        let need = num_nodes.checked_mul(num_nodes);
        if need != Some(adj.len()) {
            return Err(Error::ShapeMismatch {
                op: "GraphLaplacian::new",
                expected: vec![num_nodes, num_nodes],
                got: vec![adj.len()],
            });
        }

        // Degree calculation
        let mut degrees = vec![0.0; num_nodes];
        for i in 0..num_nodes {
            let mut deg = 0.0;
            for j in 0..num_nodes {
                deg += adj[i * num_nodes + j];
            }
            degrees[i] = deg;
        }

        let mut laplacian = vec![0.0; num_nodes * num_nodes];
        match laplacian_type {
            LaplacianType::Combinatorial => {
                for i in 0..num_nodes {
                    for j in 0..num_nodes {
                        let a_ij = adj[i * num_nodes + j];
                        laplacian[i * num_nodes + j] =
                            if i == j { degrees[i] - a_ij } else { -a_ij };
                    }
                }
            }
            LaplacianType::SymmetricNormalized => {
                let inv_sqrt_deg: Vec<f64> = degrees
                    .iter()
                    .map(|&d| if d > 1e-12 { 1.0 / d.sqrt() } else { 0.0 })
                    .collect();

                for i in 0..num_nodes {
                    for j in 0..num_nodes {
                        let a_ij = adj[i * num_nodes + j];
                        let norm_a = inv_sqrt_deg[i] * a_ij * inv_sqrt_deg[j];
                        laplacian[i * num_nodes + j] = if i == j { 1.0 - norm_a } else { -norm_a };
                    }
                }
            }
            LaplacianType::RandomWalk => {
                let inv_deg: Vec<f64> = degrees
                    .iter()
                    .map(|&d| if d > 1e-12 { 1.0 / d } else { 0.0 })
                    .collect();

                for i in 0..num_nodes {
                    for j in 0..num_nodes {
                        let a_ij = adj[i * num_nodes + j];
                        let rw_a = inv_deg[i] * a_ij;
                        laplacian[i * num_nodes + j] = if i == j { 1.0 - rw_a } else { -rw_a };
                    }
                }
            }
        }

        let lambda_max = Self::estimate_largest_eigenvalue(&laplacian, num_nodes, 40);

        Ok(Self {
            num_nodes,
            adj: adj.to_vec(),
            degrees,
            laplacian,
            laplacian_type,
            lambda_max: lambda_max.max(1.0),
        })
    }

    /// Estimates the largest eigenvalue $\lambda_{\max}$ using power iteration.
    fn estimate_largest_eigenvalue(mat: &[f64], n: usize, max_iter: usize) -> f64 {
        let mut v = vec![1.0 / (n as f64).sqrt(); n];
        let mut lambda = 1.0;

        for _ in 0..max_iter {
            // w = mat * v
            let mut w = vec![0.0; n];
            for i in 0..n {
                let mut sum = 0.0;
                for j in 0..n {
                    sum += mat[i * n + j] * v[j];
                }
                w[i] = sum;
            }

            // Rayleigh quotient: v^T w
            let mut rayleigh = 0.0;
            for i in 0..n {
                rayleigh += v[i] * w[i];
            }
            lambda = rayleigh;

            // Normalize w
            let norm: f64 = w.iter().map(|&x| x * x).sum::<f64>().sqrt();
            if norm < 1e-12 {
                break;
            }
            for i in 0..n {
                v[i] = w[i] / norm;
            }
        }

        lambda.abs()
    }

    /// Matrix-vector product: $y = L x$.
    pub fn matvec(&self, x: &[f64]) -> Result<Vec<f64>> {
        let n = self.num_nodes;
        if x.len() != n {
            return Err(Error::Msg(format!(
                "GraphLaplacian::matvec: signal has {} entries, need {n}",
                x.len()
            )));
        }
        let need = n
            .checked_mul(n)
            .ok_or_else(|| Error::Msg(format!("GraphLaplacian::matvec: {n} * {n} overflows")))?;
        if self.laplacian.len() != need {
            return Err(Error::Msg(format!(
                "GraphLaplacian::matvec: laplacian has {} entries, need {need}",
                self.laplacian.len()
            )));
        }
        let mut y = vec![0.0; n];
        for (i, y_val) in y.iter_mut().enumerate() {
            let mut sum = 0.0;
            for (j, &xj) in x.iter().enumerate() {
                sum += self.laplacian[i * n + j] * xj;
            }
            *y_val = sum;
        }
        Ok(y)
    }
}

/// Chebyshev polynomial filter approximating continuous spectral graph wavelets:
/// $\psi_s(L) = \exp(-s L)$.
#[derive(Debug, Clone)]
pub struct ChebyshevWaveletFilter {
    /// Degree of polynomial approximation $K$.
    pub order: usize,
    /// Diffusion scale $s > 0$.
    pub scale: f64,
    /// Largest eigenvalue of the Laplacian.
    pub lambda_max: f64,
    /// Chebyshev expansion coefficients $c_0, c_1, \dots, c_K$.
    pub coeffs: Vec<f64>,
}

impl ChebyshevWaveletFilter {
    /// Computes Chebyshev coefficients for the heat diffusion kernel $g(\lambda) = \exp(-s \lambda)$.
    ///
    /// The spectrum $\lambda \in [0, \lambda_{\max}]$ is mapped to $[-1, 1]$ via:
    /// $\tilde{\lambda} = \frac{2 \lambda}{\lambda_{\max}} - 1 \iff \lambda = \frac{\lambda_{\max}}{2}(\tilde{\lambda} + 1)$.
    pub fn new(scale: f64, order: usize, lambda_max: f64) -> Result<Self> {
        // Quadrature work is O(order^2); an uncapped order hangs or panics.
        let order = crate::limits::check_repeats("order", order)?;
        let k_nodes = (order + 1).max(10);
        let mut coeffs = vec![0.0; order + 1];

        // Gauss-Chebyshev quadrature nodes: x_m = cos(pi * (m + 0.5) / M)
        let m_pts = k_nodes * 2;
        for (j, coeff) in coeffs.iter_mut().enumerate().take(order + 1) {
            let mut sum = 0.0;
            for m in 0..m_pts {
                let theta = std::f64::consts::PI * (m as f64 + 0.5) / (m_pts as f64);
                let x_m = theta.cos(); // in [-1, 1]
                // Map back to lambda in [0, lambda_max]
                let lambda_m = (lambda_max / 2.0) * (x_m + 1.0);
                let g_lambda = (-scale * lambda_m).exp();
                let cheb_j = (j as f64 * theta).cos();
                sum += g_lambda * cheb_j;
            }
            let factor = if j == 0 {
                1.0 / (m_pts as f64)
            } else {
                2.0 / (m_pts as f64)
            };
            *coeff = factor * sum;
        }

        Ok(Self {
            order,
            scale,
            lambda_max,
            coeffs,
        })
    }

    /// Evaluates the wavelet filter on a graph signal $x \in \mathbb{R}^N$ using Clenshaw/Chebyshev recurrence.
    ///
    /// Recurrence on scaled Laplacian $\tilde{L} = \frac{2}{\lambda_{\max}} L - I$:
    /// $T_0(\tilde{L}) x = x$
    /// $T_1(\tilde{L}) x = \tilde{L} x$
    /// $T_k(\tilde{L}) x = 2 \tilde{L} T_{k-1} x - T_{k-2} x$
    pub fn filter(&self, laplacian: &GraphLaplacian, x: &[f64]) -> Result<Vec<f64>> {
        let n = laplacian.num_nodes;
        if x.len() != n {
            return Err(Error::ShapeMismatch {
                op: "ChebyshevWaveletFilter::filter",
                expected: vec![n],
                got: vec![x.len()],
            });
        }

        // Scaled operator: L_tilde * v = (2.0 / lambda_max) * L * v - v
        let scale_factor = 2.0 / self.lambda_max;
        let apply_scaled_l = |v: &[f64]| -> Result<Vec<f64>> {
            let lv = laplacian.matvec(v)?;
            let mut res = vec![0.0; n];
            for i in 0..n {
                res[i] = scale_factor * lv[i] - v[i];
            }
            Ok(res)
        };

        // Coeffs are public: truncated/extended buffers index OOB below.
        let need_coeffs = self
            .order
            .checked_add(1)
            .ok_or_else(|| Error::Msg(format!("Chebyshev order {} overflows", self.order)))?;
        if self.coeffs.len() != need_coeffs {
            return Err(Error::Msg(format!(
                "Chebyshev coeffs length {} disagrees with order {}",
                self.coeffs.len(),
                self.order
            )));
        }
        if self.order == 0 {
            let mut y = vec![0.0; n];
            for i in 0..n {
                y[i] = self.coeffs[0] * x[i];
            }
            return Ok(y);
        }

        // T_0 = x
        let mut t_prev2 = x.to_vec();
        // T_1 = L_tilde * x
        let mut t_prev1 = apply_scaled_l(x)?;

        // Result accumulator: y = c_0 * T_0 + c_1 * T_1 + ...
        let mut y = vec![0.0; n];
        for i in 0..n {
            y[i] = self.coeffs[0] * t_prev2[i] + self.coeffs[1] * t_prev1[i];
        }

        for k in 2..=self.order {
            let l_t_prev1 = apply_scaled_l(&t_prev1)?;
            let mut t_curr = vec![0.0; n];
            for i in 0..n {
                t_curr[i] = 2.0 * l_t_prev1[i] - t_prev2[i];
                y[i] += self.coeffs[k] * t_curr[i];
            }
            t_prev2 = t_prev1;
            t_prev1 = t_curr;
        }

        Ok(y)
    }
}

/// Multiscale Graph Wavelet Transform across multiple diffusion scales.
#[derive(Debug, Clone)]
pub struct MultiscaleGraphWaveletTransform {
    /// Number of nodes $N$.
    pub num_nodes: usize,
    /// Diffusion scales $[s_1, s_2, \dots, s_M]$.
    pub scales: Vec<f64>,
    /// Chebyshev order $K$.
    pub order: usize,
    /// Filters for each scale.
    pub filters: Vec<ChebyshevWaveletFilter>,
    /// Laplacian representation.
    pub laplacian: GraphLaplacian,
}

impl MultiscaleGraphWaveletTransform {
    /// Initializes a multiscale graph wavelet transform.
    pub fn new(
        adj: &[f64],
        num_nodes: usize,
        scales: &[f64],
        order: usize,
        lap_type: LaplacianType,
    ) -> Result<Self> {
        crate::limits::check_repeats("order", order)?;
        let laplacian = GraphLaplacian::new(adj, num_nodes, lap_type)?;
        let filters: Vec<ChebyshevWaveletFilter> = scales
            .iter()
            .map(|&s| ChebyshevWaveletFilter::new(s, order, laplacian.lambda_max))
            .collect::<Result<Vec<_>>>()?;

        Ok(Self {
            num_nodes,
            scales: scales.to_vec(),
            order,
            filters,
            laplacian,
        })
    }

    /// Decomposes an input signal $X \in \mathbb{R}^{N \times F}$ across all wavelet scales.
    ///
    /// Returns a multiscale coefficient tensor of shape $[M, N, F]$, where $M$ is the number of scales.
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let shape = x.shape();
        if shape.len() != 2 || shape[0] != self.num_nodes {
            return Err(Error::ShapeMismatch {
                op: "MultiscaleGraphWaveletTransform::forward",
                expected: vec![self.num_nodes, 0],
                got: shape.to_vec(),
            });
        }

        let f = shape[1];
        let num_scales = self.scales.len();
        if self.filters.len() != num_scales {
            return Err(Error::Msg(format!(
                "multiscale filters length {} disagrees with {} scales",
                self.filters.len(),
                num_scales
            )));
        }
        let cells =
            crate::limits::check_elements("multiscale output", &[num_scales, self.num_nodes, f])?;
        let mut out = crate::layout::try_filled(cells, 0.0)?;

        let x_data = x.to_vec::<f64>()?;

        for (s_idx, filter) in self.filters.iter().enumerate() {
            for col in 0..f {
                // Extract column signal across all nodes
                let mut col_signal = vec![0.0; self.num_nodes];
                for i in 0..self.num_nodes {
                    col_signal[i] = x_data[i * f + col];
                }

                // Filter column signal
                let filtered = filter.filter(&self.laplacian, &col_signal)?;

                for i in 0..self.num_nodes {
                    out[s_idx * (self.num_nodes * f) + i * f + col] = filtered[i];
                }
            }
        }

        Tensor::from_slice(&out, &[num_scales, self.num_nodes, f])
    }

    /// Reconstructs the original signal by inverse synthesis across scales:
    /// $\hat{X} = \frac{1}{M} \sum_{m=1}^M \Psi_{s_m} * W_m$.
    pub fn inverse(&self, wavelets: &Tensor) -> Result<Tensor> {
        let shape = wavelets.shape();
        let num_scales = self.scales.len();
        if shape.len() != 3 || shape[0] != num_scales || shape[1] != self.num_nodes {
            return Err(Error::ShapeMismatch {
                op: "MultiscaleGraphWaveletTransform::inverse",
                expected: vec![num_scales, self.num_nodes, 0],
                got: shape.to_vec(),
            });
        }

        let f = shape[2];
        if self.filters.len() != num_scales {
            return Err(Error::Msg(format!(
                "multiscale filters length {} disagrees with {} scales",
                self.filters.len(),
                num_scales
            )));
        }
        let w_data = wavelets.to_vec::<f64>()?;
        let mut reconstructed = vec![0.0; self.num_nodes * f];

        for (s_idx, filter) in self.filters.iter().enumerate() {
            for col in 0..f {
                let mut col_w = vec![0.0; self.num_nodes];
                for i in 0..self.num_nodes {
                    col_w[i] = w_data[s_idx * (self.num_nodes * f) + i * f + col];
                }

                let synth = filter.filter(&self.laplacian, &col_w)?;
                for i in 0..self.num_nodes {
                    reconstructed[i * f + col] += synth[i] / (num_scales as f64);
                }
            }
        }

        Tensor::from_slice(&reconstructed, &[self.num_nodes, f])
    }
}

/// Spatio-Temporal Graph Wavelet Recurrent Layer (GW-RNN).
///
/// Combines multiscale spatial wavelet diffusion with temporal gating
/// for sequence-of-graphs modeling.
#[derive(Debug, Clone)]
pub struct SpatioTemporalWaveletLayer {
    /// Multiscale wavelet transform engine.
    pub wavelet_transform: MultiscaleGraphWaveletTransform,
    /// Input feature dimension $F_{\text{in}}$.
    pub in_features: usize,
    /// Hidden state dimension $F_{\text{hidden}}$.
    pub hidden_features: usize,
    /// Spatial convolution weights $W_s \in \mathbb{R}^{(M \cdot F_{\text{in}}) \times F_{\text{hidden}}}$.
    pub weight_spatial: Vec<f64>,
    /// Temporal recurrence weights $W_h \in \mathbb{R}^{F_{\text{hidden}} \times F_{\text{hidden}}}$.
    pub weight_recurrent: Vec<f64>,
    /// Bias vector $b \in \mathbb{R}^{F_{\text{hidden}}}$.
    pub bias: Vec<f64>,
}

impl SpatioTemporalWaveletLayer {
    /// Creates a new Spatio-Temporal Graph Wavelet Layer.
    pub fn new(
        adj: &[f64],
        num_nodes: usize,
        scales: &[f64],
        chebyshev_order: usize,
        in_features: usize,
        hidden_features: usize,
    ) -> Result<Self> {
        let wavelet_transform = MultiscaleGraphWaveletTransform::new(
            adj,
            num_nodes,
            scales,
            chebyshev_order,
            LaplacianType::SymmetricNormalized,
        )?;

        let num_scales = scales.len();
        let spatial_in =
            crate::limits::check_elements("wavelet input width", &[num_scales, in_features])?;
        let n_spatial = crate::limits::check_elements(
            "wavelet spatial weights",
            &[spatial_in, hidden_features],
        )?;

        // Deterministic Xavier initialization
        let bound_s = (6.0 / (spatial_in as f64 + hidden_features as f64)).sqrt();
        let mut weight_spatial: Vec<f64> = crate::layout::try_with_capacity(n_spatial)?;
        weight_spatial.extend((0..n_spatial).map(|i| {
            let frac = ((i * 17 + 13) % 1000) as f64 / 1000.0;
            (frac * 2.0 - 1.0) * bound_s
        }));

        let n_recurrent = crate::limits::check_elements(
            "wavelet recurrent weights",
            &[hidden_features, hidden_features],
        )?;
        let bound_h = (6.0 / (2 * hidden_features) as f64).sqrt();
        let mut weight_recurrent: Vec<f64> = crate::layout::try_with_capacity(n_recurrent)?;
        weight_recurrent.extend((0..n_recurrent).map(|i| {
            let frac = ((i * 31 + 7) % 1000) as f64 / 1000.0;
            (frac * 2.0 - 1.0) * bound_h
        }));

        let bias = crate::layout::try_filled(hidden_features, 0.0)?;

        Ok(Self {
            wavelet_transform,
            in_features,
            hidden_features,
            weight_spatial,
            weight_recurrent,
            bias,
        })
    }

    /// Performs one forward step in time given input $X_t \in \mathbb{R}^{N \times F_{\text{in}}}$
    /// and previous hidden state $H_{t-1} \in \mathbb{R}^{N \times F_{\text{hidden}}}$.
    ///
    /// Computes:
    /// 1. Multiscale wavelet spatial transform: $\Psi(X_t) \in \mathbb{R}^{M \times N \times F_{\text{in}}}$.
    /// 2. Concatenates scales per node: $Z_t \in \mathbb{R}^{N \times (M \cdot F_{\text{in}})}$.
    /// 3. Spatio-temporal gated activation:
    ///    $$H_t = \tanh(Z_t W_s + H_{t-1} W_h + b)$$
    pub fn step(&self, x_t: &Tensor, h_prev: Option<&Tensor>) -> Result<Tensor> {
        let n = self.wavelet_transform.num_nodes;
        let num_scales = self.wavelet_transform.scales.len();
        let spatial_in = num_scales.checked_mul(self.in_features).ok_or_else(|| {
            Error::Msg(format!(
                "wavelet spatial width {num_scales} * {} overflows",
                self.in_features
            ))
        })?;
        let x_shape = x_t.shape();
        if x_shape.len() != 2 || x_shape[0] != n || x_shape[1] != self.in_features {
            return Err(Error::ShapeMismatch {
                op: "SpatioTemporalWaveletLayer::step",
                expected: vec![n, self.in_features],
                got: x_shape.to_vec(),
            });
        }

        // 1. Multiscale wavelet transform: [M, N, F_in]
        let wavelets = self.wavelet_transform.forward(x_t)?;
        let w_data = wavelets.to_vec::<f64>()?;

        // 2. Reshape to node-concatenated spatial features [N, M * F_in]
        let mut z = vec![0.0; n * spatial_in];
        for i in 0..n {
            for m in 0..num_scales {
                for f_in in 0..self.in_features {
                    let src_idx = m * (n * self.in_features) + i * self.in_features + f_in;
                    let dst_idx = i * spatial_in + m * self.in_features + f_in;
                    z[dst_idx] = w_data[src_idx];
                }
            }
        }

        // Weight agreement (pub buffers vs tamperable dims).
        let need_spatial = spatial_in
            .checked_mul(self.hidden_features)
            .ok_or_else(|| {
                Error::Msg(format!(
                    "wavelet spatial weights {spatial_in} * {} overflows",
                    self.hidden_features
                ))
            })?;
        if self.weight_spatial.len() != need_spatial {
            return Err(Error::ShapeMismatch {
                op: "SpatioTemporalWaveletLayer::step",
                expected: vec![spatial_in, self.hidden_features],
                got: vec![self.weight_spatial.len()],
            });
        }
        let need_rec = self
            .hidden_features
            .checked_mul(self.hidden_features)
            .ok_or_else(|| {
                Error::Msg(format!(
                    "wavelet recurrent weights {} * {} overflows",
                    self.hidden_features, self.hidden_features
                ))
            })?;
        if self.weight_recurrent.len() != need_rec {
            return Err(Error::ShapeMismatch {
                op: "SpatioTemporalWaveletLayer::step",
                expected: vec![self.hidden_features, self.hidden_features],
                got: vec![self.weight_recurrent.len()],
            });
        }
        if self.bias.len() != self.hidden_features {
            return Err(Error::ShapeMismatch {
                op: "SpatioTemporalWaveletLayer::step",
                expected: vec![self.hidden_features],
                got: vec![self.bias.len()],
            });
        }
        // 3. Spatio-temporal linear combination: Z_t * W_s
        let mut h_next = vec![0.0; n * self.hidden_features];
        for i in 0..n {
            for h_col in 0..self.hidden_features {
                let mut sum = self.bias[h_col];
                for s_col in 0..spatial_in {
                    sum += z[i * spatial_in + s_col]
                        * self.weight_spatial[s_col * self.hidden_features + h_col];
                }
                h_next[i * self.hidden_features + h_col] = sum;
            }
        }

        // 4. Add recurrent contribution H_{t-1} * W_h if present
        if let Some(h_p) = h_prev {
            let hp_shape = h_p.shape();
            if hp_shape.len() != 2 || hp_shape[0] != n || hp_shape[1] != self.hidden_features {
                return Err(Error::ShapeMismatch {
                    op: "SpatioTemporalWaveletLayer::step",
                    expected: vec![n, self.hidden_features],
                    got: hp_shape.to_vec(),
                });
            }
            let hp_data = h_p.to_vec::<f64>()?;
            for i in 0..n {
                for h_col in 0..self.hidden_features {
                    let mut rec_sum = 0.0;
                    for h_in in 0..self.hidden_features {
                        rec_sum += hp_data[i * self.hidden_features + h_in]
                            * self.weight_recurrent[h_in * self.hidden_features + h_col];
                    }
                    h_next[i * self.hidden_features + h_col] += rec_sum;
                }
            }
        }

        // 5. Nonlinear activation (tanh)
        for val in &mut h_next {
            *val = val.tanh();
        }

        Tensor::from_slice(&h_next, &[n, self.hidden_features])
    }
}
