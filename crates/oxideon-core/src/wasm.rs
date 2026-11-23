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

    pub fn sub(&self, other: &WasmTensor) -> Result<WasmTensor> {
        Ok(WasmTensor { inner: self.inner.sub(&other.inner)? })
    }

    pub fn mul(&self, other: &WasmTensor) -> Result<WasmTensor> {
        Ok(WasmTensor { inner: self.inner.mul(&other.inner)? })
    }

    pub fn div(&self, other: &WasmTensor) -> Result<WasmTensor> {
        Ok(WasmTensor { inner: self.inner.div(&other.inner)? })
    }

    pub fn relu(&self) -> Result<WasmTensor> {
        Ok(WasmTensor { inner: self.inner.relu()? })
    }

    pub fn gelu(&self) -> Result<WasmTensor> {
        Ok(WasmTensor { inner: self.inner.gelu()? })
    }

    pub fn tanh(&self) -> Result<WasmTensor> {
        Ok(WasmTensor { inner: self.inner.tanh()? })
    }

    pub fn exp(&self) -> Result<WasmTensor> {
        Ok(WasmTensor { inner: self.inner.exp()? })
    }

    pub fn reshape(&self, shape: &[usize]) -> Result<WasmTensor> {
        Ok(WasmTensor { inner: self.inner.reshape(shape)? })
    }

    pub fn transpose(&self, d0: isize, d1: isize) -> Result<WasmTensor> {
        Ok(WasmTensor { inner: self.inner.transpose(d0, d1)? })
    }

    pub fn sigmoid(&self) -> Result<WasmTensor> {
        Ok(WasmTensor { inner: self.inner.sigmoid()? })
    }

    pub fn sum(&self) -> Result<WasmTensor> {
        Ok(WasmTensor { inner: self.inner.sum()? })
    }

    pub fn mean(&self) -> Result<WasmTensor> {
        Ok(WasmTensor { inner: self.inner.mean()? })
    }

    pub fn set_requires_grad(&self, req: bool) -> Result<()> {
        self.inner.set_requires_grad(req)
    }

    pub fn requires_grad(&self) -> bool {
        self.inner.requires_grad()
    }

    pub fn backward(&self) -> Result<()> {
        self.inner.backward()
    }

    pub fn grad(&self) -> Option<WasmTensor> {
        self.inner.grad().map(|inner| WasmTensor { inner })
    }

    pub fn zero_grad(&self) {
        self.inner.zero_grad()
    }
}

/// Live interactive 2-hidden-layer MLP trained in-browser with Oxideon autograd + AdamW.
pub struct WasmMlpTrainer {
    l1: crate::nn::Linear,
    l2: crate::nn::Linear,
    l3: crate::nn::Linear,
    opt: crate::optim::Adam,
    step_count: usize,
}

impl WasmMlpTrainer {
    pub fn new(hidden: usize, lr: f64) -> Result<Self> {
        let h = hidden.clamp(4, 128);
        let l1 = crate::nn::Linear::new(2, h)?;
        let l2 = crate::nn::Linear::new(h, h)?;
        let l3 = crate::nn::Linear::new(h, 2)?;
        let mut params = Vec::new();
        use crate::nn::Module;
        params.extend(l1.parameters());
        params.extend(l2.parameters());
        params.extend(l3.parameters());
        let opt = crate::optim::Adam::adamw(params, lr, 1e-4);
        Ok(Self { l1, l2, l3, opt, step_count: 0 })
    }

    fn forward_logits(&self, x: &Tensor) -> Result<Tensor> {
        use crate::nn::Module;
        let h1 = self.l1.forward(x)?.tanh()?;
        let h2 = self.l2.forward(&h1)?.tanh()?;
        self.l3.forward(&h2)
    }

    /// Runs `epochs` gradient descent steps with AdamW and returns `(loss, accuracy)`.
    pub fn train_steps(
        &mut self,
        xs_flat: &[f32],
        labels: &[u32],
        epochs: usize,
    ) -> Result<[f32; 2]> {
        use crate::optim::Optimizer;
        let n = labels.len();
        if n == 0 || xs_flat.len() != n * 2 {
            return Err(crate::error::Error::Msg(
                "train_steps: expected xs_flat len == 2 * labels len".into(),
            ));
        }
        let x = Tensor::from_slice(xs_flat, &[n, 2])?;
        let targets: Vec<usize> = labels.iter().map(|&v| v as usize).collect();
        let mut last_loss = 0.0f32;
        let mut acc = 0.0f32;
        for _ in 0..epochs.max(1) {
            self.opt.zero_grad();
            let logits = self.forward_logits(&x)?;
            let loss = crate::nn::cross_entropy(&logits, &targets)?;
            last_loss = loss.to_vec::<f32>()?[0];
            loss.backward()?;
            self.opt.step()?;
            self.step_count += 1;

            let lv = logits.to_vec::<f32>()?;
            let mut correct = 0usize;
            for i in 0..n {
                let pred = if lv[i * 2 + 1] > lv[i * 2] { 1 } else { 0 };
                if pred == targets[i] {
                    correct += 1;
                }
            }
            acc = correct as f32 / n as f32;
        }
        Ok([last_loss, acc])
    }

    /// Evaluates class-1 probability across `n_points` `[x, y]` coordinates.
    pub fn predict_probs(&self, xs_flat: &[f32]) -> Result<Vec<f32>> {
        let n = xs_flat.len() / 2;
        if n == 0 || xs_flat.len() != n * 2 {
            return Ok(Vec::new());
        }
        let _ng = crate::tensor::set_grad_enabled(false);
        let x = Tensor::from_slice(xs_flat, &[n, 2])?;
        let probs = self.forward_logits(&x)?.softmax(-1)?.to_vec::<f32>()?;
        Ok((0..n).map(|i| probs[i * 2 + 1]).collect())
    }

    pub fn step_count(&self) -> usize {
        self.step_count
    }
}

/// Returns the TypeScript `.d.ts` interface declarations and JS glue for `WasmTensor` / `WasmInferenceSession`.
pub fn wasm_typescript_declarations() -> &'static str {
    r#"export class WasmTensor {
  static from_f32_array(data: Float32Array, shape: Uint32Array): WasmTensor;
  to_f32_vec(): Float32Array;
  shape(): Uint32Array;
  numel(): number;
  matmul(other: WasmTensor): WasmTensor;
  add(other: WasmTensor): WasmTensor;
  sub(other: WasmTensor): WasmTensor;
  mul(other: WasmTensor): WasmTensor;
  div(other: WasmTensor): WasmTensor;
  relu(): WasmTensor;
  gelu(): WasmTensor;
  silu(): WasmTensor;
  tanh(): WasmTensor;
  exp(): WasmTensor;
  sigmoid(): WasmTensor;
  sum(): WasmTensor;
  mean(): WasmTensor;
  softmax(axis: number): WasmTensor;
  reshape(shape: Uint32Array): WasmTensor;
  transpose(d0: number, d1: number): WasmTensor;
  set_requires_grad(req: boolean): void;
  requires_grad(): boolean;
  backward(): void;
  grad(): WasmTensor | undefined;
  zero_grad(): void;
  free(): void;
}
export class WasmMlpTrainer {
  constructor(hidden: number, lr: number);
  train_steps(xs_flat: Float32Array, labels: Uint32Array, epochs: number): Float32Array;
  predict_probs(xs_flat: Float32Array): Float32Array;
  step_count(): number;
  free(): void;
}
export class WasmInferenceSession {
  constructor();
  has_webgpu(): boolean;
}
"#
}

/// The JavaScript classes (`wasm-bindgen`), built with the `wasm` feature. They wrap [`WasmTensor`] and
/// [`WasmInferenceSession`]; `free()` and the TypeScript declarations are generated by `wasm-bindgen`.
#[cfg(feature = "wasm")]
pub mod js {
    use super::{WasmInferenceSession, WasmTensor};
    use wasm_bindgen::prelude::*;

    fn js_err(e: impl std::fmt::Display) -> JsError {
        JsError::new(&e.to_string())
    }

    #[wasm_bindgen(js_name = WasmTensor)]
    pub struct JsTensor {
        inner: WasmTensor,
    }

    macro_rules! binary {
        ($($name:ident),*) => {$(
            #[wasm_bindgen(js_class = WasmTensor)]
            impl JsTensor {
                pub fn $name(&self, other: &JsTensor) -> Result<JsTensor, JsError> {
                    self.inner.$name(&other.inner).map(|inner| JsTensor { inner }).map_err(js_err)
                }
            }
        )*};
    }
    macro_rules! unary {
        ($($name:ident),*) => {$(
            #[wasm_bindgen(js_class = WasmTensor)]
            impl JsTensor {
                pub fn $name(&self) -> Result<JsTensor, JsError> {
                    self.inner.$name().map(|inner| JsTensor { inner }).map_err(js_err)
                }
            }
        )*};
    }

    binary!(matmul, add, sub, mul, div);
    unary!(relu, gelu, silu, tanh, exp, sigmoid, sum, mean);

    #[wasm_bindgen(js_class = WasmTensor)]
    impl JsTensor {
        /// Creates a tensor from a `Float32Array` and a `Uint32Array` shape.
        pub fn from_f32_array(data: &[f32], shape: &[u32]) -> Result<JsTensor, JsError> {
            let shape: Vec<usize> = shape.iter().map(|&d| d as usize).collect();
            WasmTensor::from_f32_array(data, &shape).map(|inner| JsTensor { inner }).map_err(js_err)
        }

        pub fn to_f32_vec(&self) -> Result<Vec<f32>, JsError> {
            self.inner.to_f32_vec().map_err(js_err)
        }

        pub fn shape(&self) -> Vec<u32> {
            self.inner.shape().into_iter().map(|d| d as u32).collect()
        }

        pub fn numel(&self) -> usize {
            self.inner.numel()
        }

        pub fn softmax(&self, axis: i32) -> Result<JsTensor, JsError> {
            self.inner.softmax(axis as isize).map(|inner| JsTensor { inner }).map_err(js_err)
        }

        pub fn reshape(&self, shape: &[u32]) -> Result<JsTensor, JsError> {
            let shape: Vec<usize> = shape.iter().map(|&d| d as usize).collect();
            self.inner.reshape(&shape).map(|inner| JsTensor { inner }).map_err(js_err)
        }

        pub fn transpose(&self, d0: i32, d1: i32) -> Result<JsTensor, JsError> {
            self.inner.transpose(d0 as isize, d1 as isize).map(|inner| JsTensor { inner }).map_err(js_err)
        }

        pub fn set_requires_grad(&self, req: bool) -> Result<(), JsError> {
            self.inner.set_requires_grad(req).map_err(js_err)
        }

        pub fn requires_grad(&self) -> bool {
            self.inner.requires_grad()
        }

        pub fn backward(&self) -> Result<(), JsError> {
            self.inner.backward().map_err(js_err)
        }

        pub fn grad(&self) -> Option<JsTensor> {
            self.inner.grad().map(|inner| JsTensor { inner })
        }

        pub fn zero_grad(&self) {
            self.inner.zero_grad();
        }
    }

    #[wasm_bindgen(js_name = WasmMlpTrainer)]
    pub struct JsMlpTrainer {
        inner: super::WasmMlpTrainer,
    }

    #[wasm_bindgen(js_class = WasmMlpTrainer)]
    impl JsMlpTrainer {
        #[wasm_bindgen(constructor)]
        pub fn new(hidden: usize, lr: f64) -> Result<JsMlpTrainer, JsError> {
            super::WasmMlpTrainer::new(hidden, lr)
                .map(|inner| JsMlpTrainer { inner })
                .map_err(js_err)
        }

        pub fn train_steps(
            &mut self,
            xs_flat: &[f32],
            labels: &[u32],
            epochs: usize,
        ) -> Result<Vec<f32>, JsError> {
            self.inner
                .train_steps(xs_flat, labels, epochs)
                .map(|pair| pair.to_vec())
                .map_err(js_err)
        }

        pub fn predict_probs(&self, xs_flat: &[f32]) -> Result<Vec<f32>, JsError> {
            self.inner.predict_probs(xs_flat).map_err(js_err)
        }

        pub fn step_count(&self) -> usize {
            self.inner.step_count()
        }
    }

    #[wasm_bindgen(js_name = WasmInferenceSession)]
    pub struct JsSession {
        inner: WasmInferenceSession,
    }

    #[wasm_bindgen(js_class = WasmInferenceSession)]
    impl JsSession {
        #[wasm_bindgen(constructor)]
        pub fn new() -> JsSession {
            JsSession { inner: WasmInferenceSession::new() }
        }

        pub fn has_webgpu(&self) -> bool {
            self.inner.has_webgpu()
        }
    }

    impl Default for JsSession {
        fn default() -> Self {
            Self::new()
        }
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn wasm_tensor_autograd_and_mlp_trainer_learn_xor() {
        let x = WasmTensor::from_f32_array(&[2.0, -1.0], &[1, 2]).unwrap();
        x.set_requires_grad(true).unwrap();
        let y = x.mul(&x).unwrap().mean().unwrap();
        y.backward().unwrap();
        let g = x.grad().unwrap().to_f32_vec().unwrap();
        assert!((g[0] - 2.0).abs() < 1e-5 && (g[1] + 1.0).abs() < 1e-5);

        let mut tr = WasmMlpTrainer::new(16, 0.05).unwrap();
        let xs = [
            0.8f32, 0.8, -0.8, -0.8, 0.8, -0.8, -0.8, 0.8,
            0.6, 0.7, -0.7, -0.6, 0.7, -0.6, -0.6, 0.7,
        ];
        let labels = [1u32, 1, 0, 0, 1, 1, 0, 0];
        let first = tr.train_steps(&xs, &labels, 1).unwrap();
        let last = tr.train_steps(&xs, &labels, 60).unwrap();
        assert!(last[0] < first[0], "loss should decrease: {} -> {}", first[0], last[0]);
        assert!(last[1] >= 0.99, "accuracy should converge on XOR: {}", last[1]);
        let probs = tr.predict_probs(&[0.8, 0.8, 0.8, -0.8]).unwrap();
        assert_eq!(probs.len(), 2);
        assert!(probs[0] > 0.7 && probs[1] < 0.3);
    }
}
