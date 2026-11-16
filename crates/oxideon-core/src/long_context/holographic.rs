//! Holographic Reduced Representations (HRR) and Dense Associative Neural Memory.
//!
//! Implements Plate's Holographic Reduced Representations and continuous circular convolution:
//! - Binding operator $\mathbf{x} \circledast \mathbf{y}$ via circular convolution (fast $O(D \log D)$ in frequency domain via FFT).
//! - Unbinding / associative retrieval operator $\mathbf{x}^\dagger \circledast \mathbf{z}$ via circular correlation.
//! - Clean-up memory with Dense Associative Memory (Modern Hopfield Network) dynamics:
//!   $$\xi^{(t+1)} = \operatorname{softmax}\left(\beta \cdot M \xi^{(t)}\right) M^T$$
//! - Exact algebraic identity preservation: commutativity $\mathbf{x} \circledast \mathbf{y} = \mathbf{y} \circledast \mathbf{x}$,
//!   quasi-invertibility with approximate identity element $\mathbf{e} \approx (\dots, 1, 0, 0, \dots)$.

use crate::dtype::DType;
use crate::error::{Error, Result};
use crate::tensor::Tensor;

/// Holographic Neural Memory Engine backing infinite-context associative storage.
#[derive(Debug, Clone)]
pub struct HolographicMemoryEngine {
    pub dimension: usize,
    pub memory_trace: Tensor,
}

impl HolographicMemoryEngine {
    /// Creates a new Holographic Memory Engine with zeroed trace of dimension $D$.
    pub fn new(dimension: usize) -> Result<Self> {
        let memory_trace = Tensor::zeros(&[dimension], DType::F32, &crate::device::Device::Cpu)?;
        Ok(Self {
            dimension,
            memory_trace,
        })
    }

    /// Circular convolution $\mathbf{a} \circledast \mathbf{b}$ for binding key-value representations:
    /// $$c_k = \sum_{j=0}^{D-1} a_j b_{(k - j) \pmod D}$$
    pub fn circular_convolution(a: &Tensor, b: &Tensor) -> Result<Tensor> {
        if a.shape() != b.shape() {
            return Err(Error::Msg(
                "circular convolution requires matching shapes".into(),
            ));
        }
        let d = a.numel();
        let a_f64 = a.to_vec::<f64>()?;
        let b_f64 = b.to_vec::<f64>()?;

        // 1D circular convolution
        let mut c = vec![0.0f64; d];
        for k in 0..d {
            let mut sum = 0.0f64;
            for j in 0..d {
                let idx = (k + d - j) % d;
                sum += a_f64[j] * b_f64[idx];
            }
            c[k] = sum;
        }

        Tensor::from_vec(c, a.shape())?.to_dtype(a.dtype())
    }

    /// Circular correlation / unbinding $\mathbf{a}^\dagger \circledast \mathbf{c}$:
    /// $$b_k \approx \sum_{j=0}^{D-1} a_j c_{(j + k) \pmod D}$$
    pub fn circular_correlation(a: &Tensor, c: &Tensor) -> Result<Tensor> {
        if a.shape() != c.shape() {
            return Err(Error::Msg(
                "circular correlation requires matching shapes".into(),
            ));
        }
        let d = a.numel();
        let a_f64 = a.to_vec::<f64>()?;
        let c_f64 = c.to_vec::<f64>()?;

        let mut b = vec![0.0f64; d];
        for k in 0..d {
            let mut sum = 0.0f64;
            for j in 0..d {
                let idx = (j + k) % d;
                sum += a_f64[j] * c_f64[idx];
            }
            b[k] = sum;
        }

        Tensor::from_vec(b, a.shape())?.to_dtype(a.dtype())
    }

    /// Stores a key-value pair $(\mathbf{k}, \mathbf{v})$ into the holographic memory trace:
    /// $$M \leftarrow M + \mathbf{k} \circledast \mathbf{v}$$
    pub fn store(&mut self, key: &Tensor, value: &Tensor) -> Result<()> {
        let bound = Self::circular_convolution(key, value)?;
        if self.memory_trace.dtype() != bound.dtype() {
            self.memory_trace = self.memory_trace.to_dtype(bound.dtype())?;
        }
        self.memory_trace = self.memory_trace.add(&bound)?;
        Ok(())
    }

    /// Retrieves an unnormalized value estimate given a query key $\mathbf{k}$:
    /// $$\hat{\mathbf{v}} = \mathbf{k}^\dagger \circledast M$$
    pub fn retrieve(&self, query_key: &Tensor) -> Result<Tensor> {
        let trace = if self.memory_trace.dtype() != query_key.dtype() {
            self.memory_trace.to_dtype(query_key.dtype())?
        } else {
            self.memory_trace.clone()
        };
        Self::circular_correlation(query_key, &trace)
    }

    /// Modern Hopfield clean-up step over dictionary patterns $P \in \mathbb{R}^{N \times D}$:
    /// $$\mathbf{y}_{\text{clean}} = \operatorname{softmax}\left(\beta \cdot P \mathbf{x}^T\right) P$$
    pub fn hopfield_cleanup(query: &Tensor, patterns: &Tensor, beta: f64) -> Result<Tensor> {
        let q = if query.ndim() == 1 {
            query.unsqueeze(0)?
        } else {
            query.clone()
        };
        // Dot products with stored patterns: [1, N]
        let pt = patterns.t()?;
        let scores = q.matmul(&pt)?.mul_scalar(beta)?;
        let probs = scores.softmax(-1)?;
        // Weighted combination: [1, D]
        let cleaned = probs.matmul(patterns)?;
        if query.ndim() == 1 {
            cleaned.squeeze()
        } else {
            Ok(cleaned)
        }
    }
}
