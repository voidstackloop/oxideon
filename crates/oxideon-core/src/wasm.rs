//! WebAssembly and WebGPU in-browser deployment interface for Oxideon.
//!
//! Enables zero-install client-side inference directly inside modern web browsers
//! utilizing WebGPU hardware acceleration via `wgpu`.

use crate::error::Result;
use crate::tensor::Tensor;

/// Lightweight WebAssembly tensor wrapper exposing typed arrays to JavaScript.
#[derive(Clone)]
pub struct WasmTensor {
    inner: Tensor,
}

impl WasmTensor {
    /// Creates a tensor from flat 32-bit floating point array and shape dimensions.
    pub fn from_f32_array(data: &[f32], shape: &[usize]) -> Result<Self> {
        let inner = Tensor::from_slice(data, shape)?;
        Ok(Self { inner })
    }

    /// Exports tensor buffer as a flat `Vec<f32>` (convertible to Float32Array in JS).
    pub fn to_f32_vec(&self) -> Result<Vec<f32>> {
        self.inner.to_vec::<f32>()
    }

    /// Shape dimensions of the tensor.
    pub fn shape(&self) -> Vec<usize> {
        self.inner.shape().to_vec()
    }

    /// Total number of elements.
    pub fn numel(&self) -> usize {
        self.inner.numel()
    }

    /// Matrix multiplication.
    pub fn matmul(&self, other: &WasmTensor) -> Result<WasmTensor> {
        let out = self.inner.matmul(&other.inner)?;
        Ok(WasmTensor { inner: out })
    }

    /// Element-wise addition.
    pub fn add(&self, other: &WasmTensor) -> Result<WasmTensor> {
        let out = self.inner.add(&other.inner)?;
        Ok(WasmTensor { inner: out })
    }

    /// SiLU activation.
    pub fn silu(&self) -> Result<WasmTensor> {
        let out = self.inner.silu()?;
        Ok(WasmTensor { inner: out })
    }

    /// Softmax along the specified axis.
    pub fn softmax(&self, axis: isize) -> Result<WasmTensor> {
        let out = self.inner.softmax(axis)?;
        Ok(WasmTensor { inner: out })
    }
}

/// In-browser client inference session.
pub struct WasmInferenceSession {
    pub is_webgpu_active: bool,
}

impl Default for WasmInferenceSession {
    fn default() -> Self {
        Self::new()
    }
}

impl WasmInferenceSession {
    pub fn new() -> Self {
        Self {
            is_webgpu_active: cfg!(feature = "wgpu"),
        }
    }

    /// Returns whether hardware-accelerated WebGPU is active.
    pub fn has_webgpu(&self) -> bool {
        self.is_webgpu_active
    }
}
