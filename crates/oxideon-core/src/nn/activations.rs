//! Complete suite of neural network activations, spatial/stochastic-depth regularizers,
//! bilinear/cosine layers, and weight parameterizations (`WeightNormLinear`, `SpectralNormLinear`).

use std::sync::RwLock;
use std::sync::atomic::{AtomicBool, Ordering};

use crate::error::{Error, Result};
use crate::nn::{Module, init};
use crate::rng;
use crate::tensor::{Tensor, no_grad};

// ---------------------------------------------------------------------------
// 1. Advanced Non-Linear Activations (`Mish`, `Hardswish`, `Hardsigmoid`,
//    `Softplus`, `Softsign`, `SELU`, `CELU`, `PReLU`, `GLU`)
// ---------------------------------------------------------------------------

impl Tensor {
    /// Softplus activation with scaling parameter $\beta$: $\frac{1}{\beta} \ln(1 + \exp(\beta x))$ (`torch.nn.functional.softplus`).
    pub fn softplus_beta(&self, beta: f64) -> Result<Tensor> {
        if (beta - 1.0).abs() < 1e-7 {
            self.softplus()
        } else {
            let bx = self.mul_scalar(beta)?;
            bx.softplus()?.div_scalar(beta)
        }
    }

    /// Mish activation $x \cdot \tanh(\operatorname{softplus}(x))$ (Misra 2019, YOLOv4+).
    pub fn mish(&self) -> Result<Tensor> {
        let sp = self.softplus()?;
        self.mul(&sp.tanh()?)
    }

    /// Hardsigmoid activation $\operatorname{clamp}(x / 6 + 0.5, 0, 1)$ (MobileNetV3).
    pub fn hardsigmoid(&self) -> Result<Tensor> {
        self.div_scalar(6.0)?.add_scalar(0.5)?.clamp_open(0.0, 1.0)
    }

    /// Hardswish activation $x \cdot \operatorname{hardsigmoid}(x)$ (MobileNetV3).
    pub fn hardswish(&self) -> Result<Tensor> {
        self.mul(&self.hardsigmoid()?)
    }

    /// Softsign activation $\frac{x}{1 + |x|}$ (`torch.nn.functional.softsign`).
    pub fn softsign(&self) -> Result<Tensor> {
        self.div(&self.abs()?.add_scalar(1.0)?)
    }

    /// Self-Normalizing Exponential Linear Unit (`SELU`, Klambauer et al., NeurIPS 2017):
    /// $\lambda (\max(0, x) + \min(0, \alpha (e^x - 1)))$ with $\alpha \approx 1.673263, \lambda \approx 1.050701$.
    pub fn selu(&self) -> Result<Tensor> {
        const ALPHA: f64 = 1.673_263_242_354_377_2;
        const SCALE: f64 = 1.050_700_987_355_480_5;
        self.elu(ALPHA)?.mul_scalar(SCALE)
    }

    /// Continuously Differentiable Exponential Linear Unit (`CELU`, Barron 2017):
    /// $\max(0, x) + \min(0, \alpha (\exp(x / \alpha) - 1))$.
    pub fn celu(&self, alpha: f64) -> Result<Tensor> {
        if alpha == 0.0 || alpha.is_nan() {
            return Err(crate::error::Error::Msg("celu: alpha must be a nonzero number".into()));
        }
        if alpha > 0.0 {
            // celu(x) = alpha * elu(x / alpha, 1): same value, and the gradient at exactly 0 is 1 as in PyTorch
            return self.div_scalar(alpha)?.elu(1.0)?.mul_scalar(alpha);
        }
        // negative alpha: the identity above does not hold, use the definition
        let neg_part = self.div_scalar(alpha)?.exp()?.add_scalar(-1.0)?.mul_scalar(alpha)?.clamp(f64::NEG_INFINITY, 0.0)?;
        self.relu()?.add(&neg_part)
    }

    /// ReLU6 activation $\min(\max(x, 0), 6)$ (`torch.nn.ReLU6`).
    pub fn relu6(&self) -> Result<Tensor> {
        self.clamp_open(0.0, 6.0)
    }

    /// Hardtanh activation $\max(\min(x, \text{max\_val}), \text{min\_val})$ (`torch.nn.Hardtanh`).
    pub fn hardtanh(&self, min_val: f64, max_val: f64) -> Result<Tensor> {
        self.clamp_open(min_val, max_val)
    }

    /// LogSigmoid activation $\ln(\sigma(x)) = -\operatorname{softplus}(-x)$ (`torch.nn.LogSigmoid`).
    pub fn log_sigmoid(&self) -> Result<Tensor> {
        self.neg()?.softplus()?.neg()
    }

    /// Softshrink activation $x - \lambda$ if $x > \lambda$, $x + \lambda$ if $x < -\lambda$, else $0$.
    pub fn softshrink(&self, lambd: f64) -> Result<Tensor> {
        let abs = self.abs()?;
        let excess = abs.sub_scalar(lambd)?.relu()?;
        let sign = self.sign()?;
        excess.mul(&sign)
    }

    /// Hardshrink activation $x$ if $|x| > \lambda$, else $0$.
    pub fn hardshrink(&self, lambd: f64) -> Result<Tensor> {
        let abs = self.abs()?;
        let thresh = abs.full_like(lambd)?;
        let mask = abs.gt(&thresh)?;
        let zero = self.zeros_like()?;
        Tensor::where_cond(&mask, self, &zero)
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub struct Mish;
impl Module for Mish {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        input.mish()
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub struct Hardswish;
impl Module for Hardswish {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        input.hardswish()
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub struct Hardsigmoid;
impl Module for Hardsigmoid {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        input.hardsigmoid()
    }
}

#[derive(Clone, Copy, Debug)]
pub struct Softplus {
    pub beta: f64,
}
impl Default for Softplus {
    fn default() -> Self {
        Self { beta: 1.0 }
    }
}
impl Softplus {
    pub fn new(beta: f64) -> Self {
        Self { beta }
    }
}
impl Module for Softplus {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        input.softplus_beta(self.beta)
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub struct Softsign;
impl Module for Softsign {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        input.softsign()
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub struct SELU;
impl Module for SELU {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        input.selu()
    }
}

#[derive(Clone, Copy, Debug)]
pub struct CELU {
    pub alpha: f64,
}
impl Default for CELU {
    fn default() -> Self {
        Self { alpha: 1.0 }
    }
}
impl CELU {
    pub fn new(alpha: f64) -> Self {
        Self { alpha }
    }
}
impl Module for CELU {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        input.celu(self.alpha)
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub struct ReLU6;
impl Module for ReLU6 {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        input.relu6()
    }
}

#[derive(Clone, Copy, Debug)]
pub struct Hardtanh {
    pub min_val: f64,
    pub max_val: f64,
}
impl Default for Hardtanh {
    fn default() -> Self {
        Self {
            min_val: -1.0,
            max_val: 1.0,
        }
    }
}
impl Hardtanh {
    pub fn new(min_val: f64, max_val: f64) -> Self {
        Self { min_val, max_val }
    }
}
impl Module for Hardtanh {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        input.hardtanh(self.min_val, self.max_val)
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub struct LogSigmoid;
impl Module for LogSigmoid {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        input.log_sigmoid()
    }
}

#[derive(Clone, Copy, Debug)]
pub struct Softshrink {
    pub lambd: f64,
}
impl Default for Softshrink {
    fn default() -> Self {
        Self { lambd: 0.5 }
    }
}
impl Softshrink {
    pub fn new(lambd: f64) -> Self {
        Self { lambd }
    }
}
impl Module for Softshrink {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        input.softshrink(self.lambd)
    }
}

#[derive(Clone, Copy, Debug)]
pub struct Hardshrink {
    pub lambd: f64,
}
impl Default for Hardshrink {
    fn default() -> Self {
        Self { lambd: 0.5 }
    }
}
impl Hardshrink {
    pub fn new(lambd: f64) -> Self {
        Self { lambd }
    }
}
impl Module for Hardshrink {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        input.hardshrink(self.lambd)
    }
}

/// Parametric ReLU (`torch.nn.PReLU`): $\max(0, x) + a \cdot \min(0, x)$ with learnable slope `weight`.
pub struct PReLU {
    pub weight: Tensor,
}

impl PReLU {
    pub fn new(num_parameters: usize, init_slope: f64) -> Result<Self> {
        let weight = Tensor::full(
            &[num_parameters.max(1)],
            init_slope,
            crate::dtype::DType::F32,
            &crate::backend::Device::Cpu,
        )?
        .requires_grad_(true);
        Ok(Self { weight })
    }
}

impl Module for PReLU {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        let pos = input.relu()?;
        let neg = input.neg()?.relu()?.neg()?;
        let w = if self.weight.numel() == 1 || input.ndim() < 2 {
            self.weight
                .to_device(&input.device())?
                .to_dtype(input.dtype())?
        } else {
            let mut view_shape = vec![1usize; input.ndim()];
            view_shape[1] = self.weight.numel();
            self.weight
                .to_device(&input.device())?
                .to_dtype(input.dtype())?
                .reshape(&view_shape)?
        };
        pos.add(&neg.mul(&w)?)
    }

    fn named_parameters(&self) -> Vec<(String, Tensor)> {
        vec![("weight".into(), self.weight.clone())]
    }
}

/// Gated Linear Unit (`torch.nn.GLU`): splits `axis` in half `(a, b)` and computes $a \odot \sigma(b)$.
pub struct GLU {
    pub axis: isize,
}

impl Default for GLU {
    fn default() -> Self {
        Self { axis: -1 }
    }
}

impl GLU {
    pub fn new(axis: isize) -> Self {
        Self { axis }
    }
}

impl Module for GLU {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        let ax = crate::layout::normalize_axis("GLU", self.axis, input.ndim())?;
        let dim = input.shape()[ax];
        if dim < 2 || dim % 2 != 0 {
            return Err(crate::error::Error::Msg(format!(
                "GLU: gated axis must have even size >= 2, got {dim}"
            )));
        }
        let chunks = input.chunk(2, self.axis)?;
        chunks[0].mul(&chunks[1].sigmoid()?)
    }
}

// ---------------------------------------------------------------------------
// 2. Spatial & Stochastic-Depth Regularizers (`Dropout2d`, `AlphaDropout`, `DropPath`)
// ---------------------------------------------------------------------------

/// Spatial 2D channel-wise dropout (`torch.nn.Dropout2d`): drops entire `[N, C, H, W]` feature map
/// channels with probability `p` during training and scales surviving channels by $1 / (1 - p)$.
pub struct Dropout2d {
    pub p: f64,
    training: AtomicBool,
}

impl Dropout2d {
    pub fn new(p: f64) -> Self {
        Self {
            // Clamp into [0, 1] (torch parity: p == 1.0 zeroes everything).
            p: p.clamp(0.0, 1.0),
            training: AtomicBool::new(true),
        }
    }
}

impl Module for Dropout2d {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        if !self.training.load(Ordering::Relaxed) || self.p <= 0.0 {
            return Ok(input.clone());
        }
        if input.ndim() != 4 {
            return Err(Error::Msg("Dropout2d expects 4D input [N, C, H, W]".into()));
        }
        if self.p >= 1.0 {
            return Tensor::zeros(input.shape(), input.dtype(), &input.device());
        }
        let (n, c) = (input.shape()[0], input.shape()[1]);
        let keep = 1.0 - self.p;
        let scale = (1.0 / keep) as f32;
        let mask_data: Vec<f32> = rng::with_rng(|r| {
            (0..n * c)
                .map(|_| {
                    if (r.next_f32() as f64) < keep {
                        scale
                    } else {
                        0.0
                    }
                })
                .collect()
        });
        let mask = Tensor::from_slice_on(&mask_data, &[n, c, 1, 1], &input.device())?
            .to_dtype(input.dtype())?
            .broadcast_to(input.shape())?;
        input.mul(&mask)
    }

    fn train(&self, mode: bool) {
        self.training.store(mode, Ordering::Relaxed);
    }
}

/// Self-Normalizing Alpha Dropout (`torch.nn.AlphaDropout`, Klambauer et al. 2017):
/// Maintains zero mean and unit variance in `SELU` networks by setting dropped activations
/// to the negative saturation value $\alpha' = -\lambda \alpha \approx -1.7581$ and applying
/// an affine transformation $a x + b$.
pub struct AlphaDropout {
    pub p: f64,
    training: AtomicBool,
}

impl AlphaDropout {
    pub fn new(p: f64) -> Self {
        Self {
            // Clamp into [0, 1] (torch parity: p == 1.0 zeroes everything).
            p: p.clamp(0.0, 1.0),
            training: AtomicBool::new(true),
        }
    }
}

impl Module for AlphaDropout {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        if !self.training.load(Ordering::Relaxed) || self.p <= 0.0 {
            return Ok(input.clone());
        }
        // Torch parity: at p == 1 the affine normalization degenerates, and
        // torch returns zeros rather than NaN.
        if self.p >= 1.0 {
            return input.mul_scalar(0.0);
        }
        const ALPHA_PRIME: f64 = -1.758_099_340_847_376_6;
        let q = 1.0 - self.p;
        let a = (q + ALPHA_PRIME * ALPHA_PRIME * q * self.p).powf(-0.5);
        let b = -a * self.p * ALPHA_PRIME;

        let numel = input.numel();
        let mask_vec: Vec<f32> = rng::with_rng(|r| {
            (0..numel)
                .map(|_| if (r.next_f32() as f64) < q { 1.0 } else { 0.0 })
                .collect()
        });
        let mask = Tensor::from_slice_on(&mask_vec, input.shape(), &input.device())?
            .to_dtype(input.dtype())?;
        let inv_mask = mask.neg()?.add_scalar(1.0)?;
        let dropped = input.mul(&mask)?.add(&inv_mask.mul_scalar(ALPHA_PRIME)?)?;
        dropped.mul_scalar(a)?.add_scalar(b)
    }

    fn train(&self, mode: bool) {
        self.training.store(mode, Ordering::Relaxed);
    }
}

/// Stochastic Depth / DropPath (`timm.layers.DropPath`, Huang et al. / ConvNeXt / ViT):
/// Drops entire residual sample paths per batch index (`[B, 1, ...]`) with probability `drop_prob`.
pub struct DropPath {
    pub drop_prob: f64,
    training: AtomicBool,
}

impl DropPath {
    pub fn new(drop_prob: f64) -> Self {
        Self {
            // Clamp into [0, 1] (p == 1.0 drops every path: all zeros).
            drop_prob: drop_prob.clamp(0.0, 1.0),
            training: AtomicBool::new(true),
        }
    }
}

impl Module for DropPath {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        if !self.training.load(Ordering::Relaxed) || self.drop_prob <= 0.0 {
            return Ok(input.clone());
        }
        if input.ndim() < 1 {
            return Err(Error::Msg("DropPath expects input of rank >= 1".into()));
        }
        if self.drop_prob >= 1.0 {
            return input.mul_scalar(0.0);
        }
        let b = input.shape()[0];
        let keep = 1.0 - self.drop_prob;
        let scale = (1.0 / keep) as f32;
        let mask_b: Vec<f32> = rng::with_rng(|r| {
            (0..b)
                .map(|_| {
                    if (r.next_f32() as f64) < keep {
                        scale
                    } else {
                        0.0
                    }
                })
                .collect()
        });
        let mut mask_shape = vec![1usize; input.ndim()];
        mask_shape[0] = b;
        let mask = Tensor::from_slice_on(&mask_b, &mask_shape, &input.device())?
            .to_dtype(input.dtype())?
            .broadcast_to(input.shape())?;
        input.mul(&mask)
    }

    fn train(&self, mode: bool) {
        self.training.store(mode, Ordering::Relaxed);
    }
}

// ---------------------------------------------------------------------------
// 3. Bilinear, CosineLinear & Weight-Parameterized Layers
//    (`WeightNormLinear`, `SpectralNormLinear`)
// ---------------------------------------------------------------------------

/// Bilinear transformation (`torch.nn.Bilinear`):
/// Computes $y_k = x_1^T W_k x_2 + b_k$ for inputs `x1: [..., in1]` and `x2: [..., in2]`.
pub struct Bilinear {
    pub weight: Tensor, // [out_features, in1_features, in2_features]
    pub bias: Option<Tensor>,
    pub in1_features: usize,
    pub in2_features: usize,
    pub out_features: usize,
}

impl Bilinear {
    pub fn new(
        in1_features: usize,
        in2_features: usize,
        out_features: usize,
        use_bias: bool,
    ) -> Result<Self> {
        let bound = 1.0 / (in1_features.max(1) as f32).sqrt();
        let weight = init::uniform(&[out_features, in1_features, in2_features], -bound, bound)?
            .requires_grad_(true);
        let bias = if use_bias {
            Some(init::uniform(&[out_features], -bound, bound)?.requires_grad_(true))
        } else {
            None
        };
        Ok(Self {
            weight,
            bias,
            in1_features,
            in2_features,
            out_features,
        })
    }

    pub fn forward_pair(&self, x1: &Tensor, x2: &Tensor) -> Result<Tensor> {
        if x1.ndim() < 1 || x2.ndim() < 1 {
            return Err(crate::error::Error::Msg(format!(
                "Bilinear::forward_pair expects rank >= 1 inputs, got {:?}/{:?}",
                x1.shape(),
                x2.shape()
            )));
        }
        let (d1, d2) = (x1.shape()[x1.ndim() - 1], x2.shape()[x2.ndim() - 1]);
        if d1 != self.in1_features || d2 != self.in2_features {
            return Err(crate::error::Error::Msg(format!(
                "Bilinear::forward_pair feature mismatch: got [{d1}, {d2}], expected [{}, {}]",
                self.in1_features, self.in2_features
            )));
        }
        // Broadcast the leading batch dims (torch allows [..., in] inputs),
        // then flatten to one row per batch element.
        let batch = crate::layout::broadcast_shapes(
            "Bilinear::forward_pair",
            &x1.shape()[..x1.ndim() - 1],
            &x2.shape()[..x2.ndim() - 1],
        )?;
        let mut s1 = batch.clone();
        s1.push(d1);
        let mut s2 = batch.clone();
        s2.push(d2);
        let b: usize = batch.iter().product();
        let x1_2d = x1.broadcast_to(&s1)?.reshape(&[b, self.in1_features])?;
        let x2_2d = x2.broadcast_to(&s2)?.reshape(&[b, self.in2_features])?;
        // Contract: (x1 @ W_k) . x2 for each output feature k
        // Reshape W as [out * in2, in1]^T -> x1 [b, in1] @ W_perm [in1, out * in2] -> [b, out, in2]
        // `out_features`/`in2_features` are pub: check the product (the d1/d2
        // checks above pin `in2` to a real dim, but `out` is unconstrained).
        let w_cols = self
            .out_features
            .checked_mul(self.in2_features)
            .ok_or_else(|| {
                crate::error::Error::Msg(format!(
                    "Bilinear::forward_pair: out_features * in2_features overflows ({} * {})",
                    self.out_features, self.in2_features
                ))
            })?;
        let w_perm = self
            .weight
            .permute(&[1, 0, 2])?
            .contiguous()?
            .reshape(&[self.in1_features, w_cols])?;
        let mid = x1_2d
            .matmul(&w_perm)?
            .reshape(&[b, self.out_features, self.in2_features])?;
        let prod = mid
            .mul(
                &x2_2d
                    .unsqueeze(1)?
                    .broadcast_to(&[b, self.out_features, self.in2_features])?,
            )?
            .sum_axis(2, false)?;
        let biased = match &self.bias {
            Some(bias) => prod.add(bias)?,
            None => prod,
        };
        let mut out_shape = batch;
        out_shape.push(self.out_features);
        biased.reshape(&out_shape)
    }

    pub fn named_parameters(&self) -> Vec<(String, Tensor)> {
        let mut out = vec![("weight".into(), self.weight.clone())];
        if let Some(b) = &self.bias {
            out.push(("bias".into(), b.clone()));
        }
        out
    }
}

/// Cosine Similarity Linear Layer (`CosineLinear`):
/// Computes $\tau \cdot \frac{x}{\|x\|_2} \cdot \left(\frac{W}{\|W\|_2}\right)^T$ with a learnable
/// logit scale parameter $\tau$.
pub struct CosineLinear {
    pub weight: Tensor,
    pub scale: Tensor,
    pub in_features: usize,
    pub out_features: usize,
}

impl CosineLinear {
    pub fn new(in_features: usize, out_features: usize, init_scale: f64) -> Result<Self> {
        let weight =
            init::kaiming_uniform(&[out_features, in_features], in_features)?.requires_grad_(true);
        let scale = Tensor::full(
            &[1],
            init_scale,
            crate::dtype::DType::F32,
            &crate::backend::Device::Cpu,
        )?
        .requires_grad_(true);
        Ok(Self {
            weight,
            scale,
            in_features,
            out_features,
        })
    }
}

impl Module for CosineLinear {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        let x_norm = input.normalize(2.0, -1, 1e-12)?;
        let w_norm = self.weight.normalize(2.0, -1, 1e-12)?;
        let cos_sim = x_norm.matmul(&w_norm.t()?)?;
        cos_sim.mul(
            &self
                .scale
                .to_device(&input.device())?
                .to_dtype(input.dtype())?,
        )
    }

    fn named_parameters(&self) -> Vec<(String, Tensor)> {
        vec![
            ("weight".into(), self.weight.clone()),
            ("scale".into(), self.scale.clone()),
        ]
    }
}

/// **Weight Normalization Linear Layer** (`torch.nn.utils.weight_norm`, Salimans & Kingma, NeurIPS 2016):
/// Reparameterizes each row of weight matrix $W$ into a learnable direction vector $v$ and scalar
/// magnitude $g$:
/// $$W = g \cdot \frac{v}{\|v\|_2}$$
pub struct WeightNormLinear {
    pub weight_v: Tensor, // [out_features, in_features]
    pub weight_g: Tensor, // [out_features, 1]
    pub bias: Option<Tensor>,
    pub in_features: usize,
    pub out_features: usize,
}

impl WeightNormLinear {
    pub fn new(in_features: usize, out_features: usize, use_bias: bool) -> Result<Self> {
        let bound = 1.0 / (in_features.max(1) as f32).sqrt();
        let weight_v = init::uniform(&[out_features, in_features], -bound, bound)?;
        let weight_g = no_grad(|| weight_v.norm_p(2.0, &[1], true))?.requires_grad_(true);
        let weight_v = weight_v.requires_grad_(true);
        let bias = if use_bias {
            Some(init::uniform(&[out_features], -bound, bound)?.requires_grad_(true))
        } else {
            None
        };
        Ok(Self {
            weight_v,
            weight_g,
            bias,
            in_features,
            out_features,
        })
    }

    /// Computes the effective normalized weight matrix $W = g \cdot \frac{v}{\|v\|_2}$.
    pub fn effective_weight(&self) -> Result<Tensor> {
        let v_norm = self.weight_v.normalize(2.0, 1, 1e-12)?;
        v_norm.mul(&self.weight_g.broadcast_to(self.weight_v.shape())?)
    }
}

impl Module for WeightNormLinear {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        let w = self.effective_weight()?;
        let y = input.matmul(&w.t()?)?;
        match &self.bias {
            Some(b) => y.add(b),
            None => Ok(y),
        }
    }

    fn named_parameters(&self) -> Vec<(String, Tensor)> {
        let mut out = vec![
            ("weight_v".into(), self.weight_v.clone()),
            ("weight_g".into(), self.weight_g.clone()),
        ];
        if let Some(b) = &self.bias {
            out.push(("bias".into(), b.clone()));
        }
        out
    }
}

/// **Spectral Normalization Linear Layer** (`torch.nn.utils.spectral_norm`, Miyato et al., ICLR 2018):
/// Constrains the Lipschitz constant of the linear layer to $1$ by dividing $W$ by its largest
/// singular value $\sigma(W) = u^T W v$, updated via `n_power_iterations` of power iteration.
pub struct SpectralNormLinear {
    pub weight: Tensor, // [out_features, in_features]
    pub bias: Option<Tensor>,
    pub u_vec: RwLock<Tensor>, // [out_features, 1]
    pub v_vec: RwLock<Tensor>, // [in_features, 1]
    pub n_power_iterations: usize,
    training: AtomicBool,
}

impl SpectralNormLinear {
    pub fn new(
        in_features: usize,
        out_features: usize,
        use_bias: bool,
        n_power_iterations: usize,
    ) -> Result<Self> {
        let bound = 1.0 / (in_features.max(1) as f32).sqrt();
        let weight =
            init::uniform(&[out_features, in_features], -bound, bound)?.requires_grad_(true);
        let bias = if use_bias {
            Some(init::uniform(&[out_features], -bound, bound)?.requires_grad_(true))
        } else {
            None
        };
        let u = Tensor::randn(&[out_features, 1])?.normalize(2.0, 0, 1e-12)?;
        let v = Tensor::randn(&[in_features, 1])?.normalize(2.0, 0, 1e-12)?;
        Ok(Self {
            weight,
            bias,
            u_vec: RwLock::new(u),
            v_vec: RwLock::new(v),
            n_power_iterations: n_power_iterations.max(1),
            training: AtomicBool::new(true),
        })
    }

    /// Computes spectrally normalized weight $W / \sigma(W)$ where $\sigma(W) = u^T W v$.
    pub fn spectral_weight(&self) -> Result<Tensor> {
        // `n_power_iterations` is pub with only a constructor floor: re-check the
        // ceiling here so post-construction tampering yields Err, not a hang.
        crate::limits::check_iterations("n_power_iterations", self.n_power_iterations)?;
        // The vec locks are pub: recover from poisoning instead of panicking.
        let rd = |m: &RwLock<Tensor>| m.read().unwrap_or_else(|p| p.into_inner()).clone();
        if self.training.load(Ordering::Relaxed) {
            no_grad(|| -> Result<()> {
                let mut u = rd(&self.u_vec);
                let mut v = rd(&self.v_vec);
                let wt = self.weight.detach().t()?;
                let w = self.weight.detach();
                for _ in 0..self.n_power_iterations {
                    v = wt.matmul(&u)?.normalize(2.0, 0, 1e-12)?;
                    u = w.matmul(&v)?.normalize(2.0, 0, 1e-12)?;
                }
                *self.u_vec.write().unwrap_or_else(|p| p.into_inner()) = u;
                *self.v_vec.write().unwrap_or_else(|p| p.into_inner()) = v;
                Ok(())
            })?;
        }
        let u = rd(&self.u_vec);
        let v = rd(&self.v_vec);
        // sigma = u^T @ W @ v (differentiable w.r.t. W!)
        let sigma = u.t()?.matmul(&self.weight)?.matmul(&v)?.add_scalar(1e-12)?;
        self.weight.div(&sigma.broadcast_to(self.weight.shape())?)
    }
}

impl Module for SpectralNormLinear {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        let w_sn = self.spectral_weight()?;
        let y = input.matmul(&w_sn.t()?)?;
        match &self.bias {
            Some(b) => y.add(b),
            None => Ok(y),
        }
    }

    fn named_parameters(&self) -> Vec<(String, Tensor)> {
        let mut out = vec![("weight".into(), self.weight.clone())];
        if let Some(b) = &self.bias {
            out.push(("bias".into(), b.clone()));
        }
        out
    }

    fn train(&self, mode: bool) {
        self.training.store(mode, Ordering::Relaxed);
    }
}
