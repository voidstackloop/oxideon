//! Block-wise Weight Quantization (`Q8_0` 8-bit and `Q4_0` 4-bit) and [`QuantizedLinear`].
//!
//! Modelled after GGML / llama.cpp block quantization:
//! - [`QuantScheme::Q8_0`]: Each block of `B` elements stores a single `f16` scale $d = \frac{\max |x_i|}{127}$
//!   and `B` signed 8-bit integers $q_i = \text{round}(x_i / d) \in [-127, 127]$.
//! - [`QuantScheme::Q4_0`]: Each block of `B` elements (even `B`) stores a single `f16` scale
//!   $d = \frac{\max |x_i|}{7}$ and `B / 2` packed bytes in GGML split-half nibble order
//!   (byte `j` packs `x[j]` low + `x[j + B/2]` high, each offset by `+8` into `[0, 15]`),
//!   interchangeable with GGUF `Q4_0` blocks. A trailing partial block of length `L`
//!   pairs `x[j]` with `x[j + \lceil L/2 \rceil]` (GGML has no partial blocks; the rule
//!   coincides with GGML on full even blocks).

use crate::dtype::{DType, f16};
use crate::error::{Error, Result};
use crate::nn::{Linear, Module};
use crate::tensor::Tensor;

/// Block quantization scheme.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QuantScheme {
    /// 8-bit symmetric block quantization (default block_size = 32).
    Q8_0 { block_size: usize },
    /// 4-bit symmetric packed block quantization (default block_size = 32, must be even).
    Q4_0 { block_size: usize },
}

impl QuantScheme {
    pub const DEFAULT_Q8_0: Self = Self::Q8_0 { block_size: 32 };
    pub const DEFAULT_Q4_0: Self = Self::Q4_0 { block_size: 32 };

    pub fn block_size(self) -> usize {
        match self {
            Self::Q8_0 { block_size } | Self::Q4_0 { block_size } => block_size,
        }
    }
}

/// Compressed block-quantized tensor storage.
#[derive(Debug, Clone)]
pub struct QuantizedTensor {
    pub scheme: QuantScheme,
    pub shape: Vec<usize>,
    pub scales: Vec<f16>,
    /// For `Q8_0`: 1 byte (`i8` as `u8`) per element.
    /// For `Q4_0`: 1 byte per 2 elements in GGML split-half order (byte `j` of a
    /// chunk packs `x[j]` low + `x[j + half]` high, each `+ 8`).
    pub data: Vec<u8>,
}

impl QuantizedTensor {
    /// Quantizes a tensor into `scheme`.
    pub fn quantize(tensor: &Tensor, scheme: QuantScheme) -> Result<Self> {
        let bsize = scheme.block_size();
        if bsize == 0 || (matches!(scheme, QuantScheme::Q4_0 { .. }) && bsize % 2 != 0) {
            return Err(Error::Msg(format!(
                "invalid quantization block_size {bsize}"
            )));
        }
        let src = tensor.to_vec::<f32>()?;
        let n = src.len();
        let num_blocks = n.div_ceil(bsize);
        let mut scales = Vec::with_capacity(num_blocks);

        match scheme {
            QuantScheme::Q8_0 { .. } => {
                let mut data = Vec::with_capacity(n);
                for chunk in src.chunks(bsize) {
                    let max_abs = chunk.iter().fold(0.0f32, |m, &v| m.max(v.abs()));
                    let scale = if max_abs > 0.0 { max_abs / 127.0 } else { 0.0 };
                    let s_f16 = f16::from_f32(scale);
                    let eff_scale = s_f16.to_f32();
                    scales.push(s_f16);
                    let inv = if eff_scale > 0.0 {
                        1.0 / eff_scale
                    } else {
                        0.0
                    };
                    for &v in chunk {
                        let q = (v * inv).round().clamp(-127.0, 127.0) as i8;
                        data.push(q as u8);
                    }
                }
                Ok(Self {
                    scheme,
                    shape: tensor.shape().to_vec(),
                    scales,
                    data,
                })
            }
            QuantScheme::Q4_0 { .. } => {
                let mut data = Vec::with_capacity(n.div_ceil(2));
                for chunk in src.chunks(bsize) {
                    let max_abs = chunk.iter().fold(0.0f32, |m, &v| m.max(v.abs()));
                    let scale = if max_abs > 0.0 { max_abs / 7.0 } else { 0.0 };
                    let s_f16 = f16::from_f32(scale);
                    let eff_scale = s_f16.to_f32();
                    scales.push(s_f16);
                    let inv = if eff_scale > 0.0 {
                        1.0 / eff_scale
                    } else {
                        0.0
                    };

                    // GGML split-half nibble order: byte j packs x[j] (low) and
                    // x[j + half] (high), matching GGUF Q4_0 blocks (an adjacent
                    // pairing here produced bytes no GGML tool could read back
                    // under the same Q4_0 name).
                    let half = chunk.len().div_ceil(2);
                    for j in 0..half {
                        let q0 = ((chunk[j] * inv).round().clamp(-8.0, 7.0) as i8 + 8) as u8;
                        let q1 = if j + half < chunk.len() {
                            ((chunk[j + half] * inv).round().clamp(-8.0, 7.0) as i8 + 8) as u8
                        } else {
                            8u8
                        };
                        data.push((q0 & 0x0F) | ((q1 & 0x0F) << 4));
                    }
                }
                Ok(Self {
                    scheme,
                    shape: tensor.shape().to_vec(),
                    scales,
                    data,
                })
            }
        }
    }

    /// Dequantizes into an `f32` vector.
    /// Validates scales and data against shape under the block scheme,
    /// returning (block_size, numel). Manual struct construction could
    /// otherwise make dequantize chunk by zero or index out of bounds.
    fn check_layout(&self) -> Result<(usize, usize)> {
        let bsize = self.scheme.block_size();
        if bsize == 0 {
            return Err(Error::Msg(
                "QuantizedTensor: block_size must be positive".into(),
            ));
        }
        let n = crate::layout::checked_numel(&self.shape)?;
        let num_blocks = n.div_ceil(bsize);
        // Exact counts: trailing bytes/scales are a corrupt layout, and extra
        // data chunks would index `scales` out of bounds in dequantize_vec.
        if self.scales.len() != num_blocks {
            return Err(Error::Msg(format!(
                "QuantizedTensor: {} blocks need exactly {} scales, got {}",
                num_blocks,
                num_blocks,
                self.scales.len()
            )));
        }
        let need_data = match self.scheme {
            QuantScheme::Q8_0 { .. } => n,
            QuantScheme::Q4_0 { .. } => {
                if bsize % 2 != 0 {
                    return Err(Error::Msg(format!(
                        "QuantizedTensor: Q4_0 block_size must be even, got {}",
                        bsize
                    )));
                }
                // Full blocks hold bsize/2 bytes each; a trailing partial block
                // of L elems holds ceil(L/2). With even bsize this sums to
                // ceil(n/2) exactly (a full-block multiple would reject
                // quantize()'s own partial-block output).
                n.div_ceil(2)
            }
        };
        if self.data.len() != need_data {
            return Err(Error::Msg(format!(
                "QuantizedTensor: layout needs exactly {} data bytes, got {}",
                need_data,
                self.data.len()
            )));
        }
        Ok((bsize, n))
    }

    pub fn dequantize_vec(&self) -> Vec<f32> {
        // Infallible for backward compatibility: inconsistent layouts
        // panic with a clear message here, so prefer dequantize().
        let (bsize, n) = self
            .check_layout()
            .expect("QuantizedTensor has inconsistent layout");
        let mut out = Vec::with_capacity(n);

        match self.scheme {
            QuantScheme::Q8_0 { .. } => {
                for (blk_idx, chunk) in self.data.chunks(bsize).enumerate() {
                    let d = self.scales[blk_idx].to_f32();
                    for &b in chunk {
                        out.push((b as i8 as f32) * d);
                    }
                }
            }
            QuantScheme::Q4_0 { .. } => {
                // Split-half nibble order (see quantize): byte j of a block
                // holds x[j] (low) and x[j + half] (high).
                let mut data_off = 0usize;
                let mut elem_off = 0usize;
                for scale in &self.scales {
                    let blen = (n - elem_off).min(bsize);
                    let half = blen.div_ceil(2);
                    let d = scale.to_f32();
                    let base = out.len();
                    out.resize(base + blen, 0.0);
                    for j in 0..half {
                        let byte = self.data[data_off + j];
                        out[base + j] = (((byte & 0x0F) as i8) - 8) as f32 * d;
                        if j + half < blen {
                            out[base + j + half] = ((((byte >> 4) & 0x0F) as i8) - 8) as f32 * d;
                        }
                    }
                    data_off += half;
                    elem_off += blen;
                }
            }
        }
        out.truncate(n);
        out
    }

    /// Dequantizes back into a [`Tensor`] of `dtype`.
    pub fn dequantize(&self, dtype: DType) -> Result<Tensor> {
        self.check_layout()?;
        let t = Tensor::from_vec(self.dequantize_vec(), &self.shape)?;
        t.to_dtype(dtype)
    }

    /// Total memory footprint of the compressed representation in bytes.
    pub fn size_in_bytes(&self) -> usize {
        self.scales.len() * 2 + self.data.len()
    }

    /// Compression ratio relative to uncompressed `f32` storage.
    pub fn compression_ratio_f32(&self) -> f64 {
        // `shape` is public: checked product, saturating to MAX on overflow
        // (the ratio stays finite; f64 holds 4 * usize::MAX exactly enough).
        let numel = crate::layout::checked_numel(&self.shape).unwrap_or(usize::MAX);
        (numel as f64 * 4.0) / (self.size_in_bytes().max(1) as f64)
    }
}

/// Validates the trailing feature dim of a linear-layer input, returning
/// (original shape, last-dim k, flattened leading rows). Scalar (0-dim)
/// inputs and last-dim mismatches become Err, never an unwrap panic.
fn checked_linear_last_dim(
    input: &Tensor,
    in_features: usize,
    op: &'static str,
) -> Result<(Vec<usize>, usize, usize)> {
    let orig = input.shape().to_vec();
    let Some(k) = orig.last().copied() else {
        return Err(Error::Msg(format!(
            "{}: scalar input has no feature dim (in_features = {})",
            op, in_features
        )));
    };
    if k != in_features {
        return Err(Error::ShapeMismatch {
            op,
            expected: vec![in_features],
            got: vec![k],
        });
    }
    if k == 0 {
        return Err(Error::Msg(format!(
            "{}: zero-sized feature dim with in_features = 0",
            op
        )));
    }
    let rows = input.numel() / k;
    Ok((orig, k, rows))
}

/// A quantized linear layer storing weights in block-quantized format.
pub struct QuantizedLinear {
    pub qweight: QuantizedTensor,
    pub bias: Option<Tensor>,
    pub in_features: usize,
    pub out_features: usize,
}

impl QuantizedLinear {
    /// Quantizes an existing [`Linear`] layer into `scheme`.
    pub fn from_linear(linear: &Linear, scheme: QuantScheme) -> Result<Self> {
        let qweight = QuantizedTensor::quantize(&linear.weight, scheme)?;
        Ok(Self {
            qweight,
            bias: linear.bias.as_ref().map(Tensor::detach),
            in_features: linear.in_features(),
            out_features: linear.out_features(),
        })
    }

    /// Compressed size of the weight matrix in bytes.
    pub fn weight_bytes(&self) -> usize {
        self.qweight.size_in_bytes()
    }

    /// Dequantizes the weights and returns a standard [`Linear`] layer.
    pub fn to_linear(&self, dtype: DType) -> Result<Linear> {
        let w = self.qweight.dequantize(dtype)?;
        let b = match &self.bias {
            Some(bias) => Some(bias.to_dtype(dtype)?),
            None => None,
        };
        Linear::from_tensors(w, b)
    }
}

impl Module for QuantizedLinear {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        let w = self
            .qweight
            .dequantize(input.dtype())?
            .to_device(&input.device())?;
        let wt = w.t()?;
        let y = if input.ndim() == 2 {
            input.matmul(&wt)?
        } else {
            let (orig, k, rows) =
                checked_linear_last_dim(input, self.in_features, "QuantizedLinear")?;
            let flat = input.reshape(&[rows, k])?.matmul(&wt)?;
            let mut out_shape = orig[..orig.len() - 1].to_vec();
            out_shape.push(self.out_features);
            flat.reshape(&out_shape)?
        };
        match &self.bias {
            Some(b) => y.add(&b.to_device(&input.device())?.to_dtype(input.dtype())?),
            None => Ok(y),
        }
    }
}

// ---------------------------------------------------------------------------
// FP8 (`E4M3` & `E5M2`, Hopper / Blackwell / DeepSeek-V3 standard)
// ---------------------------------------------------------------------------

/// 8-bit floating point `E4M3` (1 sign bit, 4 exponent bits, bias = 7, 3 mantissa bits, max = 448.0).
#[allow(non_camel_case_types)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[repr(transparent)]
pub struct fp8_e4m3(pub u8);

impl fp8_e4m3 {
    pub const MAX_VAL: f32 = 448.0;

    pub fn from_f32(v: f32) -> Self {
        if v.is_nan() {
            return Self(0x7F);
        }
        let sign = if v.is_sign_negative() { 0x80u8 } else { 0u8 };
        let abs_v = v.abs().min(Self::MAX_VAL);
        if abs_v == 0.0 {
            return Self(sign);
        }
        // Bias = 7, subnormal < 2^-6
        let min_norm = 2.0f32.powi(-6);
        if abs_v < min_norm {
            // OCP round-to-nearest-even (plain round() breaks ties away from zero).
            let mant = ((abs_v / 2.0f32.powi(-9)).round_ties_even() as u8).min(7);
            return Self(sign | mant);
        }
        let exp_unbiased = abs_v.log2().floor() as i32;
        let exp_biased = (exp_unbiased + 7).clamp(1, 15);
        let scale = 2.0f32.powi(exp_biased - 7);
        let frac = ((abs_v / scale - 1.0) * 8.0).round_ties_even() as i32;
        let (eb, mb) = if frac >= 8 {
            if exp_biased >= 15 {
                (15, 6)
            } else {
                (exp_biased + 1, 0)
            }
        } else {
            (exp_biased, frac.clamp(0, 7))
        };
        if eb == 15 && mb == 7 {
            Self(sign | ((15u8 << 3) | 6u8))
        } else {
            Self(sign | (((eb as u8) << 3) | (mb as u8)))
        }
    }

    pub fn to_f32(self) -> f32 {
        let sign = if (self.0 & 0x80) != 0 {
            -1.0f32
        } else {
            1.0f32
        };
        let exp = (self.0 >> 3) & 0x0F;
        let mant = self.0 & 0x07;
        if exp == 0 {
            if mant == 0 {
                return 0.0 * sign;
            }
            return sign * (mant as f32) * 2.0f32.powi(-9);
        }
        if exp == 15 && mant == 7 {
            return f32::NAN;
        }
        sign * (1.0 + (mant as f32) / 8.0) * 2.0f32.powi((exp as i32) - 7)
    }
}

/// 8-bit floating point `E5M2` (1 sign bit, 5 exponent bits, bias = 15, 2 mantissa bits, max = 57344.0).
#[allow(non_camel_case_types)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[repr(transparent)]
pub struct fp8_e5m2(pub u8);

impl fp8_e5m2 {
    pub const MAX_VAL: f32 = 57344.0;

    pub fn from_f32(v: f32) -> Self {
        if v.is_nan() {
            return Self(0x7F);
        }
        let sign = if v.is_sign_negative() { 0x80u8 } else { 0u8 };
        if v.is_infinite() {
            return Self(sign | 0x7C);
        }
        let abs_v = v.abs().min(Self::MAX_VAL);
        if abs_v == 0.0 {
            return Self(sign);
        }
        let min_norm = 2.0f32.powi(-14);
        if abs_v < min_norm {
            // OCP round-to-nearest-even (plain round() breaks ties away from zero).
            let mant = ((abs_v / 2.0f32.powi(-16)).round_ties_even() as u8).min(3);
            return Self(sign | mant);
        }
        let exp_unbiased = abs_v.log2().floor() as i32;
        let exp_biased = (exp_unbiased + 15).clamp(1, 30);
        let scale = 2.0f32.powi(exp_biased - 15);
        let frac = ((abs_v / scale - 1.0) * 4.0).round_ties_even() as i32;
        let (eb, mb) = if frac >= 4 {
            if exp_biased >= 30 {
                (30, 3)
            } else {
                (exp_biased + 1, 0)
            }
        } else {
            (exp_biased, frac.clamp(0, 3))
        };
        Self(sign | (((eb as u8) << 2) | (mb as u8)))
    }

    pub fn to_f32(self) -> f32 {
        let sign = if (self.0 & 0x80) != 0 {
            -1.0f32
        } else {
            1.0f32
        };
        let exp = (self.0 >> 2) & 0x1F;
        let mant = self.0 & 0x03;
        if exp == 0 {
            if mant == 0 {
                return 0.0 * sign;
            }
            return sign * (mant as f32) * 2.0f32.powi(-16);
        }
        if exp == 31 {
            return if mant == 0 {
                sign * f32::INFINITY
            } else {
                f32::NAN
            };
        }
        sign * (1.0 + (mant as f32) / 4.0) * 2.0f32.powi((exp as i32) - 15)
    }
}

/// Activation-Aware Weight Quantization (`AWQ`, Lin et al., MLSys 2024):
/// Protects salient input channels by calibrating per-channel scales $s_j = (\text{mean}|X_{:,j}|)^\alpha$
/// on a representative activation sample, quantizing $W \cdot \operatorname{diag}(s)$, and applying
/// $X \cdot \operatorname{diag}(s)^{-1}$ at inference time.
pub struct AWQLinear {
    pub qweight: QuantizedTensor,
    pub channel_scales: Tensor,
    pub bias: Option<Tensor>,
    pub in_features: usize,
    pub out_features: usize,
}

impl AWQLinear {
    pub fn from_linear_calibrated(
        linear: &Linear,
        calib_x: &Tensor,
        scheme: QuantScheme,
        alpha: f64,
    ) -> Result<Self> {
        let in_f = linear.in_features();
        let out_f = linear.out_features();
        if in_f == 0 {
            return Err(Error::Msg("AWQ requires in_features > 0".into()));
        }
        if calib_x.ndim() < 1 || calib_x.shape()[calib_x.ndim() - 1] != in_f {
            return Err(Error::Msg(format!(
                "AWQ calibration last dim must be {}, got {:?}",
                in_f,
                calib_x.shape()
            )));
        }
        if calib_x.numel() % in_f != 0 {
            return Err(Error::Msg(format!(
                "AWQ calibration numel {} is not a multiple of in_features {}",
                calib_x.numel(),
                in_f
            )));
        }
        let rows = calib_x.numel() / in_f;
        let x_flat = calib_x.reshape(&[rows, in_f])?;
        // Per-channel activation magnitude s_j = (mean_i |X_{i,j}| + eps)^alpha
        let s = x_flat
            .abs()?
            .mean_axis(0, false)?
            .add_scalar(1e-6)?
            .powf(alpha)?;
        let w_scaled = linear.weight.mul(&s.unsqueeze(0)?)?;
        let qweight = QuantizedTensor::quantize(&w_scaled, scheme)?;
        Ok(Self {
            qweight,
            channel_scales: s.detach(),
            bias: linear.bias.as_ref().map(Tensor::detach),
            in_features: in_f,
            out_features: out_f,
        })
    }

    /// Dequantizes the weights and returns an equivalent unquantized [`Linear`] layer.
    pub fn to_linear(&self, dtype: DType) -> Result<Linear> {
        let w_scaled = self.qweight.dequantize(dtype)?;
        let s = self.channel_scales.to_dtype(dtype)?.unsqueeze(0)?;
        let w = w_scaled.div(&s)?;
        let b = match &self.bias {
            Some(bias) => Some(bias.to_dtype(dtype)?),
            None => None,
        };
        Linear::from_tensors(w, b)
    }
}

impl Module for AWQLinear {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        // Validate before the channel-scale division, which would otherwise
        // broadcast a scalar input into a vector and hide the shape mistake.
        checked_linear_last_dim(input, self.in_features, "AWQLinear")?;
        let s = self
            .channel_scales
            .to_device(&input.device())?
            .to_dtype(input.dtype())?;
        let x_div = input.div(&s)?;
        let w = self
            .qweight
            .dequantize(input.dtype())?
            .to_device(&input.device())?;
        let wt = w.t()?;
        let y = if x_div.ndim() == 2 {
            x_div.matmul(&wt)?
        } else {
            let (orig, k, rows) = checked_linear_last_dim(&x_div, self.in_features, "AWQLinear")?;
            let flat = x_div.reshape(&[rows, k])?.matmul(&wt)?;
            let mut out_shape = orig[..orig.len() - 1].to_vec();
            out_shape.push(self.out_features);
            flat.reshape(&out_shape)?
        };
        match &self.bias {
            Some(b) => y.add(&b.to_device(&input.device())?.to_dtype(input.dtype())?),
            None => Ok(y),
        }
    }
}

// ---------------------------------------------------------------------------
// BitNet 1.58b: 1.58-bit Ternary Quantization (Wang et al. 2024)
// ---------------------------------------------------------------------------

/// 1.58-bit Ternary Linear Layer (`BitLinear`, Wang et al. 2024 / Ma et al. 2024).
///
/// Weights are quantized to ternary values $\{-1, 0, +1\}$ scaled by average magnitude:
/// $$\gamma = \frac{1}{d_{\text{out}} d_{\text{in}}} \sum |W_{ij}|, \quad \widetilde{W} = \operatorname{clip}(\operatorname{round}(W / \gamma), -1, 1)$$
/// Activations are dynamically scaled to 8-bit integers:
/// $$\eta = \frac{127.0}{\max |X| + \epsilon}, \quad \widetilde{X} = \operatorname{clip}(\operatorname{round}(X \cdot \eta), -128, 127)$$
/// Output is evaluated with 2-bit packed ternary weights:
/// $$Y = (\widetilde{X} \cdot \widetilde{W}^T) \cdot \frac{\gamma}{\eta} + \text{bias}$$
#[derive(Debug, Clone)]
pub struct BitLinear {
    /// 2 bits per ternary weight packed into `u8` (4 weights per byte).
    /// Encoding: `0b00` -> 0, `0b01` -> +1, `0b10` -> -1.
    pub packed_weights: Vec<u8>,
    pub weight_scale: f32,
    pub bias: Option<Tensor>,
    pub in_features: usize,
    pub out_features: usize,
}

impl BitLinear {
    /// Quantizes a standard [`Linear`] layer into a 1.58-bit ternary [`BitLinear`].
    pub fn from_linear(linear: &Linear) -> Result<Self> {
        let in_f = linear.in_features();
        let out_f = linear.out_features();
        let w_vec = linear.weight.to_vec::<f32>()?;
        let total_elems = w_vec.len();

        // gamma = mean(|W|)
        let sum_abs: f32 = w_vec.iter().map(|v| v.abs()).sum();
        let gamma = (sum_abs / (total_elems.max(1) as f32)).max(1e-7);
        let inv_gamma = 1.0 / gamma;

        // Pack ternary weights: 4 weights per byte
        let packed_len = total_elems.div_ceil(4);
        let mut packed = vec![0u8; packed_len];

        for (idx, &w) in w_vec.iter().enumerate() {
            let ternary = (w * inv_gamma).round().clamp(-1.0, 1.0) as i8;
            let code = match ternary {
                1 => 0b01u8,
                -1 => 0b10u8,
                _ => 0b00u8,
            };
            let byte_idx = idx / 4;
            let shift = (idx % 4) * 2;
            packed[byte_idx] |= code << shift;
        }

        Ok(Self {
            packed_weights: packed,
            weight_scale: gamma,
            bias: linear.bias.as_ref().map(Tensor::detach),
            in_features: in_f,
            out_features: out_f,
        })
    }

    /// Dequantizes the packed ternary weights into an `[out_features, in_features]` tensor.
    pub fn dequantize_weight(&self, dtype: DType) -> Result<Tensor> {
        let total = self
            .out_features
            .checked_mul(self.in_features)
            .ok_or_else(|| Error::Msg("BitLinear: feature dims overflow".into()))?;
        if self.packed_weights.len() < total.div_ceil(4) {
            return Err(Error::Msg(format!(
                "BitLinear: packed weights hold {} bytes, need {}",
                self.packed_weights.len(),
                total.div_ceil(4)
            )));
        }
        let mut out = Vec::with_capacity(total);
        for i in 0..total {
            let byte_idx = i / 4;
            let shift = (i % 4) * 2;
            let code = (self.packed_weights[byte_idx] >> shift) & 0b11;
            let val = match code {
                0b01 => self.weight_scale,
                0b10 => -self.weight_scale,
                _ => 0.0f32,
            };
            out.push(val);
        }
        let t = Tensor::from_vec(out, &[self.out_features, self.in_features])?;
        t.to_dtype(dtype)
    }

    /// Total memory size of the packed weight matrix in bytes.
    pub fn weight_bytes(&self) -> usize {
        self.packed_weights.len() + std::mem::size_of::<f32>()
    }

    /// Dequantizes the weights and returns an unquantized [`Linear`] layer.
    pub fn to_linear(&self, dtype: DType) -> Result<Linear> {
        let w = self.dequantize_weight(dtype)?;
        let b = match &self.bias {
            Some(bias) => Some(bias.to_dtype(dtype)?),
            None => None,
        };
        Linear::from_tensors(w, b)
    }
}

impl Module for BitLinear {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        // Dynamic activation quantization scale eta = 127 / max(|X|)
        let in_f32 = input.to_dtype(DType::F32)?;
        let in_vec = in_f32.to_vec::<f32>()?;
        let max_abs = in_vec.iter().fold(0.0f32, |m, &x| m.max(x.abs())).max(1e-7);
        let eta = 127.0f32 / max_abs;
        let inv_eta = 1.0f32 / eta;

        // Quantize activations to i8 [-128, 127]
        let mut q_act = Vec::with_capacity(in_vec.len());
        for &x in &in_vec {
            let q = (x * eta).round().clamp(-128.0, 127.0) as i8;
            q_act.push(q);
        }

        // Compute matmul using integer addition/subtraction scaled by (gamma * inv_eta)
        let orig_shape = input.shape().to_vec();
        let Some(last) = orig_shape.last().copied() else {
            return Err(Error::Msg(format!(
                "BitLinear: scalar input has no feature dim (in_features = {})",
                self.in_features
            )));
        };
        if last != self.in_features {
            return Err(Error::ShapeMismatch {
                op: "BitLinear",
                expected: vec![self.in_features],
                got: vec![last],
            });
        }
        let k = self.in_features;
        let n = self.out_features;
        if k == 0 || n == 0 {
            return Err(Error::Msg(
                "BitLinear: in_features and out_features must be positive".into(),
            ));
        }
        let need_packed = k
            .checked_mul(n)
            .map(|t| t.div_ceil(4))
            .ok_or_else(|| Error::Msg("BitLinear: feature dims overflow".into()))?;
        if self.packed_weights.len() < need_packed {
            return Err(Error::Msg(format!(
                "BitLinear: packed weights hold {} bytes, need {}",
                self.packed_weights.len(),
                need_packed
            )));
        }
        let m = in_vec.len() / k;
        let scale_factor = self.weight_scale * inv_eta;

        let out_len = m
            .checked_mul(n)
            .ok_or_else(|| Error::Msg("BitLinear: output size overflows".into()))?;
        let mut out_f32 = crate::layout::try_filled(out_len, 0.0f32)?;

        for row in 0..m {
            let act_slice = &q_act[row * k..(row + 1) * k];
            for col in 0..n {
                let mut sum_i64 = 0i64;
                let w_offset = col * k;
                for (p, &act_val) in act_slice.iter().enumerate().take(k) {
                    let w_idx = w_offset + p;
                    let byte = self.packed_weights[w_idx / 4];
                    let code = (byte >> ((w_idx % 4) * 2)) & 0b11;
                    match code {
                        0b01 => sum_i64 += act_val as i64,
                        0b10 => sum_i64 -= act_val as i64,
                        _ => {}
                    }
                }
                out_f32[row * n + col] = (sum_i64 as f32) * scale_factor;
            }
        }

        let mut out_shape = orig_shape[..orig_shape.len() - 1].to_vec();
        out_shape.push(n);
        let out_t = Tensor::from_vec(out_f32, &out_shape)?
            .to_device(&input.device())?
            .to_dtype(input.dtype())?;

        match &self.bias {
            Some(b) => out_t.add(&b.to_device(&input.device())?.to_dtype(input.dtype())?),
            None => Ok(out_t),
        }
    }
}

// ---------------------------------------------------------------------------
// Native FP8 Tensor & FP8 Linear
// ---------------------------------------------------------------------------

/// 8-bit floating point format.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FP8Format {
    /// E4M3 (1 sign, 4 exponent, 3 mantissa; recommended for forward activations & weights).
    E4M3,
    /// E5M2 (1 sign, 5 exponent, 2 mantissa; recommended for gradients and wide dynamic range).
    E5M2,
}

/// A packed 8-bit floating point tensor with per-tensor dynamic scale.
#[derive(Debug, Clone)]
pub struct FP8Tensor {
    pub shape: Vec<usize>,
    pub format: FP8Format,
    pub scale: f32,
    pub data: Vec<u8>,
}

impl FP8Tensor {
    /// Quantizes a [`Tensor`] into packed [`FP8Tensor`].
    pub fn from_tensor(tensor: &Tensor, format: FP8Format) -> Result<Self> {
        let v = tensor.to_vec::<f32>()?;
        let max_abs = v.iter().fold(0.0f32, |m, &x| m.max(x.abs()));
        let max_rep = match format {
            FP8Format::E4M3 => fp8_e4m3::MAX_VAL,
            FP8Format::E5M2 => fp8_e5m2::MAX_VAL,
        };
        let scale = if max_abs > 0.0 {
            max_abs / max_rep
        } else {
            1.0
        };
        let inv_scale = 1.0 / scale;

        let mut data = Vec::with_capacity(v.len());
        match format {
            FP8Format::E4M3 => {
                for &x in &v {
                    data.push(fp8_e4m3::from_f32(x * inv_scale).0);
                }
            }
            FP8Format::E5M2 => {
                for &x in &v {
                    data.push(fp8_e5m2::from_f32(x * inv_scale).0);
                }
            }
        }

        Ok(Self {
            shape: tensor.shape().to_vec(),
            format,
            scale,
            data,
        })
    }

    /// Dequantizes back into a [`Tensor`].
    pub fn to_tensor(&self, target_dtype: DType) -> Result<Tensor> {
        let mut out = Vec::with_capacity(self.data.len());
        match self.format {
            FP8Format::E4M3 => {
                for &b in &self.data {
                    out.push(fp8_e4m3(b).to_f32() * self.scale);
                }
            }
            FP8Format::E5M2 => {
                for &b in &self.data {
                    out.push(fp8_e5m2(b).to_f32() * self.scale);
                }
            }
        }
        let t = Tensor::from_vec(out, &self.shape)?;
        t.to_dtype(target_dtype)
    }

    /// Memory footprint in bytes.
    pub fn size_in_bytes(&self) -> usize {
        self.data.len() + std::mem::size_of::<f32>()
    }
}

/// A linear layer with weights stored in native 8-bit floating point format.
pub struct FP8Linear {
    pub qweight: FP8Tensor,
    pub bias: Option<Tensor>,
    pub in_features: usize,
    pub out_features: usize,
}

impl FP8Linear {
    pub fn from_linear(linear: &Linear, format: FP8Format) -> Result<Self> {
        let qweight = FP8Tensor::from_tensor(&linear.weight, format)?;
        Ok(Self {
            qweight,
            bias: linear.bias.as_ref().map(Tensor::detach),
            in_features: linear.in_features(),
            out_features: linear.out_features(),
        })
    }

    /// Dequantizes the weights and returns an unquantized [`Linear`] layer.
    pub fn to_linear(&self, dtype: DType) -> Result<Linear> {
        let w = self.qweight.to_tensor(dtype)?;
        let b = match &self.bias {
            Some(bias) => Some(bias.to_dtype(dtype)?),
            None => None,
        };
        Linear::from_tensors(w, b)
    }
}

impl Module for FP8Linear {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        let w = self
            .qweight
            .to_tensor(input.dtype())?
            .to_device(&input.device())?;
        let wt = w.t()?;
        let y = if input.ndim() == 2 {
            input.matmul(&wt)?
        } else {
            let (orig, k, rows) = checked_linear_last_dim(input, self.in_features, "FP8Linear")?;
            let flat = input.reshape(&[rows, k])?.matmul(&wt)?;
            let mut out_shape = orig[..orig.len() - 1].to_vec();
            out_shape.push(self.out_features);
            flat.reshape(&out_shape)?
        };
        match &self.bias {
            Some(b) => y.add(&b.to_device(&input.device())?.to_dtype(input.dtype())?),
            None => Ok(y),
        }
    }
}

// ---------------------------------------------------------------------------
// GPTQ: Generalized Post-Training Quantization (Frantar et al. 2022)
// ---------------------------------------------------------------------------

/// Generalized Post-Training Quantization (`GPTQ`, Frantar et al. 2022):
/// Performs second-order inverse-Hessian error compensation column-by-column across
/// weight matrix columns using representative calibration activations.
pub struct GPTQLinear {
    pub qweight: QuantizedTensor,
    pub bias: Option<Tensor>,
    pub in_features: usize,
    pub out_features: usize,
}

impl GPTQLinear {
    /// Calibrates and quantizes a [`Linear`] layer using GPTQ with damping.
    pub fn from_linear_calibrated(
        linear: &Linear,
        calib_x: &Tensor,
        scheme: QuantScheme,
        damp_percent: f64,
    ) -> Result<Self> {
        let in_f = linear.in_features();
        let out_f = linear.out_features();
        if in_f == 0 {
            return Err(Error::Msg("GPTQ requires in_features > 0".into()));
        }
        if calib_x.ndim() < 1 || calib_x.shape()[calib_x.ndim() - 1] != in_f {
            return Err(Error::Msg(format!(
                "GPTQ calibration last dim must be {in_f}, got {:?}",
                calib_x.shape()
            )));
        }
        if calib_x.numel() % in_f != 0 {
            return Err(Error::Msg(format!(
                "GPTQ calibration numel {} is not a multiple of in_features {in_f}",
                calib_x.numel()
            )));
        }
        let n = calib_x.numel() / in_f;
        let x_flat = calib_x.reshape(&[n, in_f])?;

        // Empirical Hessian H = (2 / N) * X^T * X
        let xt = x_flat.t()?;
        let h = xt.matmul(&x_flat)?.mul_scalar(2.0 / (n.max(1) as f64))?;

        // Damping: lambda = damp_percent * mean(diag(H))
        let mut h_data = h.to_vec::<f64>()?;
        let mut diag_sum = 0.0f64;
        for i in 0..in_f {
            diag_sum += h_data[i * in_f + i];
        }
        let damp = damp_percent * (diag_sum / (in_f.max(1) as f64)).max(1e-6);
        for i in 0..in_f {
            h_data[i * in_f + i] += damp;
        }

        // Invert damped Hessian H
        let h_damped = Tensor::from_vec(h_data, &[in_f, in_f])?;
        let h_inv = h_damped.inv()?;
        let h_inv_data = h_inv.to_vec::<f64>()?;

        // Iterate through columns of W, quantize, and apply second-order compensation
        let mut w_mat = linear.weight.to_vec::<f32>()?; // [out_f, in_f]

        let max_range = match scheme {
            QuantScheme::Q8_0 { .. } => 127.0f32,
            QuantScheme::Q4_0 { .. } => 7.0f32,
        };
        for col in 0..in_f {
            let h_inv_jj = h_inv_data[col * in_f + col];
            let inv_h_jj = if h_inv_jj.abs() > 1e-12 {
                1.0 / h_inv_jj
            } else {
                0.0
            };
            // One scale per column, shared by all rows (per-scalar scales
            // would map every weight to itself and defeat quantization).
            let mut col_max = 0.0f32;
            for row in 0..out_f {
                col_max = col_max.max(w_mat[row * in_f + col].abs());
            }
            let q_scale = (col_max / max_range).max(1e-7);

            for row in 0..out_f {
                let w_orig = w_mat[row * in_f + col];
                let w_q = (w_orig / q_scale).round().clamp(-max_range, max_range) * q_scale;
                let err = (w_orig - w_q) as f64;
                w_mat[row * in_f + col] = w_q;

                // Error compensation for subsequent columns
                if err.abs() > 1e-12 {
                    for next_col in (col + 1)..in_f {
                        let h_inv_jk = h_inv_data[col * in_f + next_col];
                        let delta = err * (h_inv_jk * inv_h_jj);
                        w_mat[row * in_f + next_col] -= delta as f32;
                    }
                }
            }
        }

        let compensated_tensor = Tensor::from_vec(w_mat, &[out_f, in_f])?;
        let qweight = QuantizedTensor::quantize(&compensated_tensor, scheme)?;

        Ok(Self {
            qweight,
            bias: linear.bias.as_ref().map(Tensor::detach),
            in_features: in_f,
            out_features: out_f,
        })
    }

    /// Dequantizes the weights and returns an unquantized [`Linear`] layer.
    pub fn to_linear(&self, dtype: DType) -> Result<Linear> {
        let w = self.qweight.dequantize(dtype)?;
        let b = match &self.bias {
            Some(bias) => Some(bias.to_dtype(dtype)?),
            None => None,
        };
        Linear::from_tensors(w, b)
    }
}

impl Module for GPTQLinear {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        let w = self
            .qweight
            .dequantize(input.dtype())?
            .to_device(&input.device())?;
        let wt = w.t()?;
        let y = if input.ndim() == 2 {
            input.matmul(&wt)?
        } else {
            let (orig, k, rows) = checked_linear_last_dim(input, self.in_features, "GPTQLinear")?;
            let flat = input.reshape(&[rows, k])?.matmul(&wt)?;
            let mut out_shape = orig[..orig.len() - 1].to_vec();
            out_shape.push(self.out_features);
            flat.reshape(&out_shape)?
        };
        match &self.bias {
            Some(b) => y.add(&b.to_device(&input.device())?.to_dtype(input.dtype())?),
            None => Ok(y),
        }
    }
}

/// BitNet b1.58 Radix-3 Ternary Packed Storage (5 ternary weights per byte = 1.6 bits/weight).
#[derive(Debug, Clone)]
pub struct TernaryPackedTensor {
    pub shape: Vec<usize>,
    pub scale: f32,
    pub packed_data: Vec<u8>,
    pub num_elements: usize,
}

impl TernaryPackedTensor {
    /// Packs a float tensor into 1.58-bit base-3 ternary format ({-1, 0, +1} -> {0, 1, 2}).
    pub fn pack(tensor: &Tensor, scale: f32) -> Result<Self> {
        if !scale.is_finite() {
            return Err(Error::Msg(format!(
                "TernaryPackedTensor::pack needs a finite scale, got {}",
                scale
            )));
        }
        let shape = tensor.shape().to_vec();
        let num_elements = tensor.numel();
        let data = tensor.to_vec::<f32>()?;

        // 5 ternary values per byte (3^5 = 243 <= 256)
        let num_bytes = num_elements.div_ceil(5);
        let mut packed_data = vec![0u8; num_bytes];

        for (chunk_idx, slot) in packed_data.iter_mut().enumerate().take(num_bytes) {
            let start = chunk_idx * 5;
            let mut byte_val = 0u8;
            let mut power = 1u8;

            for i in 0..5 {
                let elem_idx = start + i;
                let t_val: i8 = if elem_idx < num_elements {
                    let v = data[elem_idx] / scale.max(1e-8);
                    if v > 0.5 {
                        1
                    } else if v < -0.5 {
                        -1
                    } else {
                        0
                    }
                } else {
                    0
                };

                let u_val = (t_val + 1) as u8; // 0, 1, 2
                byte_val += u_val * power;
                if i < 4 {
                    power *= 3;
                }
            }

            *slot = byte_val;
        }

        Ok(Self {
            shape,
            scale,
            packed_data,
            num_elements,
        })
    }

    /// Unpacks ternary packed bytes back into a full-precision float tensor.
    pub fn unpack(&self) -> Result<Tensor> {
        // `num_elements` is public: agree with the shape before reserving.
        let expect = crate::layout::checked_numel(&self.shape)?;
        if self.num_elements != expect {
            return Err(Error::Msg(format!(
                "TernaryPackedTensor: num_elements {} disagrees with shape {:?} ({} elements)",
                self.num_elements, self.shape, expect
            )));
        }
        let mut out = Vec::with_capacity(self.num_elements);

        for &byte in &self.packed_data {
            let mut b = byte;
            for _ in 0..5 {
                if out.len() == self.num_elements {
                    break;
                }
                let u = b % 3;
                b /= 3;
                let t = (u as i8) - 1;
                out.push((t as f32) * self.scale);
            }
        }

        Tensor::from_slice(&out, &self.shape)
    }

    /// Computes `Y = X * W^T` via unpack + dense matmul in the input's dtype
    /// (matching the other quantized linears, which compute in the input dtype
    /// rather than erroring on mixed dtypes).
    pub fn ternary_matmul(&self, input: &Tensor) -> Result<Tensor> {
        if self.shape.len() != 2 {
            return Err(Error::Msg(
                "TernaryPackedTensor matmul expects 2D weight matrix".into(),
            ));
        }

        let in_f = self.shape[1];
        let in_shape = input.shape();
        let in_last = *in_shape.last().unwrap_or(&0);

        if in_last != in_f {
            return Err(Error::ShapeMismatch {
                op: "ternary_matmul",
                expected: vec![in_f],
                got: vec![in_last],
            });
        }

        let w_unpacked = self
            .unpack()?
            .to_dtype(input.dtype())?
            .to_device(&input.device())?;
        let wt = w_unpacked.t()?;
        input.matmul(&wt)
    }
}

/// Linear Layer backed by BitNet b1.58 Radix-3 Ternary Packed Storage.
pub struct PackedBitLinear {
    pub weight: TernaryPackedTensor,
    pub bias: Option<Tensor>,
    pub in_features: usize,
    pub out_features: usize,
}

impl PackedBitLinear {
    pub fn from_linear(linear: &Linear, scale: f32) -> Result<Self> {
        let w = linear.weight.detach();
        let weight = TernaryPackedTensor::pack(&w, scale)?;
        Ok(Self {
            weight,
            bias: linear.bias.as_ref().map(Tensor::detach),
            in_features: linear.in_features(),
            out_features: linear.out_features(),
        })
    }

    /// Dequantizes the weights and returns an unquantized [`Linear`] layer.
    pub fn to_linear(&self, dtype: DType) -> Result<Linear> {
        let w = self.weight.unpack()?.to_dtype(dtype)?;
        let b = match &self.bias {
            Some(bias) => Some(bias.to_dtype(dtype)?),
            None => None,
        };
        Linear::from_tensors(w, b)
    }
}

impl Module for PackedBitLinear {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        let y = self.weight.ternary_matmul(input)?;
        match &self.bias {
            Some(b) => y.add(&b.to_device(&input.device())?.to_dtype(input.dtype())?),
            None => Ok(y),
        }
    }
}

pub mod autonomous;

pub use autonomous::{
    AdaptiveQuantLinear, DynamicLlmQuantizer, DynamicPrecisionLevel, DynamicQuantPolicy,
    DynamicTelemetry, DynamicTransitionPlan,
};

#[cfg(test)]
mod tests {
    use super::*;

    fn t32(data: &[f32], shape: &[usize]) -> Tensor {
        Tensor::from_slice(data, shape).unwrap()
    }

    /// GGML interchange: a 32-wide Q4_0 block uses split-half nibble order
    /// (byte j packs x[j] low + x[j+16] high), NOT adjacent pairs. Hand-derived:
    /// x = [0, 7, 0, ...] has max 7 so d = 1.0; quants are 8 everywhere except
    /// x[1] = 15, landing in byte 1's low nibble. The old adjacent packing put
    /// 0xF8 in byte 0 instead of 0x88.
    #[test]
    fn q4_packing_matches_ggml_split_half_layout() {
        let mut vals = vec![0.0f32; 32];
        vals[1] = 7.0;
        let q = QuantizedTensor::quantize(&t32(&vals, &[32]), QuantScheme::Q4_0 { block_size: 32 })
            .unwrap();
        assert_eq!(q.scales.len(), 1);
        assert!((q.scales[0].to_f32() - 1.0).abs() < 1e-6);
        let mut want = vec![0x88u8; 16];
        want[1] = 0x8F;
        assert_eq!(q.data, want);
        // ...and it decodes back to the same values.
        let back = q.dequantize_vec();
        assert!((back[1] - 7.0).abs() < 0.01);
        assert!(back.iter().enumerate().all(|(i, &v)| i == 1 || v == 0.0));
    }

    /// Odd-sized trailing partial blocks stay self-consistent (33 = 32 + 1).
    #[test]
    fn q4_partial_block_roundtrips() {
        let vals = vec![3.0f32; 33];
        let q = QuantizedTensor::quantize(&t32(&vals, &[33]), QuantScheme::DEFAULT_Q4_0).unwrap();
        assert_eq!(q.scales.len(), 2);
        assert_eq!(q.data.len(), 16 + 1);
        for v in q.dequantize_vec() {
            assert!((v - 3.0).abs() / 3.0 < 0.01, "value {v}");
        }
    }

    /// Q8_0 exact bytes: max 127 gives d = 1.0, so small integers survive
    /// bit-exact (and the all-zero block yields scale 0 + zero bytes).
    #[test]
    fn q8_exact_bytes_and_zero_block() {
        let q = QuantizedTensor::quantize(
            &t32(&[0.0, 127.0, -127.0, 64.0], &[4]),
            QuantScheme::DEFAULT_Q8_0,
        )
        .unwrap();
        assert!((q.scales[0].to_f32() - 1.0).abs() < 1e-6);
        assert_eq!(q.data, vec![0u8, 127, 129, 64]);
        assert_eq!(q.dequantize_vec(), vec![0.0f32, 127.0, -127.0, 64.0]);
        let z =
            QuantizedTensor::quantize(&t32(&[0.0f32; 8], &[8]), QuantScheme::DEFAULT_Q8_0).unwrap();
        assert_eq!(z.scales[0].to_f32(), 0.0);
        assert!(z.data.iter().all(|&b| b == 0));
        assert!(z.dequantize_vec().iter().all(|&v| v == 0.0));
    }

    /// OCP FP8 bit patterns (hand-derived from the spec): zeros, ones, max,
    /// saturation, NaN/Inf, and subnormals in both formats.
    #[test]
    fn fp8_bit_patterns_match_spec() {
        assert_eq!(fp8_e4m3::from_f32(0.0).0, 0x00);
        assert_eq!(fp8_e4m3::from_f32(-0.0).0, 0x80);
        assert_eq!(fp8_e4m3::from_f32(1.0).0, 0x38);
        assert_eq!(fp8_e4m3::from_f32(-2.0).0, 0xC0);
        assert_eq!(fp8_e4m3::from_f32(448.0).0, 0x7E);
        assert_eq!(fp8_e4m3::from_f32(500.0).0, 0x7E); // saturate (no Inf)
        assert_eq!(fp8_e4m3::from_f32(f32::INFINITY).0, 0x7E);
        assert_eq!(fp8_e4m3::from_f32(f32::NAN).0, 0x7F);
        assert_eq!(fp8_e4m3::from_f32(2.0f32.powi(-9)).0, 0x01); // subnormal
        assert_eq!(fp8_e4m3(0x38).to_f32(), 1.0);
        assert!(fp8_e4m3(0x7F).to_f32().is_nan());

        assert_eq!(fp8_e5m2::from_f32(0.0).0, 0x00);
        assert_eq!(fp8_e5m2::from_f32(1.0).0, 0x3C);
        assert_eq!(fp8_e5m2::from_f32(f32::INFINITY).0, 0x7C);
        assert_eq!(fp8_e5m2::from_f32(f32::NEG_INFINITY).0, 0xFC);
        assert_eq!(fp8_e5m2::from_f32(f32::NAN).0, 0x7F);
        assert_eq!(fp8_e5m2::from_f32(57344.0).0, 0x7B);
        assert_eq!(fp8_e5m2::from_f32(1.0e6).0, 0x7B); // saturate
        assert_eq!(fp8_e5m2::from_f32(2.0f32.powi(-16)).0, 0x01); // subnormal
        assert_eq!(fp8_e5m2(0x7C).to_f32(), f32::INFINITY);
        assert!(fp8_e5m2(0x7F).to_f32().is_nan());
    }

    /// OCP mandates round-to-nearest-even (not half-away): 1.0625 sits exactly
    /// between E4M3 1.0 (0x38) and 1.125 (0x39), so it must round to the even
    /// mantissa 0x38; likewise 1.125 in E5M2 rounds to 0x3C, not 0x3D.
    #[test]
    fn fp8_rounds_ties_to_even() {
        assert_eq!(fp8_e4m3::from_f32(1.0625).0, 0x38);
        assert_eq!(fp8_e5m2::from_f32(1.125).0, 0x3C);
    }

    /// FP8Tensor round-trips within half-ULP relative error and never panics
    /// on non-finite input (NaN survives; Inf saturates).
    #[test]
    fn fp8_tensor_roundtrip_and_nonfinite() {
        let vals: Vec<f32> = (0..64).map(|i| 1.0 + (i as f32) * 0.01).collect();
        // Per-format half-ULP bounds: E4M3 ~6.3%, E5M2 (2 mantissa bits) 12.5%.
        for (fmt, tol) in [(FP8Format::E4M3, 0.1f32), (FP8Format::E5M2, 0.15f32)] {
            let ft = FP8Tensor::from_tensor(&t32(&vals, &[64]), fmt).unwrap();
            let back = ft.to_tensor(DType::F32).unwrap().to_vec::<f32>().unwrap();
            for (b, v) in back.iter().zip(vals.iter()) {
                assert!((b - v).abs() / v < tol, "fmt {fmt:?}: {b} vs {v}");
            }
        }
        let nan = FP8Tensor::from_tensor(&t32(&[1.0, f32::NAN], &[2]), FP8Format::E4M3).unwrap();
        let back = nan.to_tensor(DType::F32).unwrap().to_vec::<f32>().unwrap();
        assert!(back[1].is_nan());
        let inf =
            FP8Tensor::from_tensor(&t32(&[1.0, f32::INFINITY], &[2]), FP8Format::E4M3).unwrap();
        let back = inf.to_tensor(DType::F32).unwrap().to_vec::<f32>().unwrap();
        assert!(back.iter().all(|v| v.is_finite() || v.is_nan()));
    }

    /// Radix-3 packing is exact by hand: [1,-1,1] with scale 1.0 packs to
    /// 2 + 0*3 + 2*9 + 1*27 + 1*81 = 128 (padding slots encode 0 as digit 1).
    #[test]
    fn ternary_pack_unpack_exact() {
        let p = TernaryPackedTensor::pack(&t32(&[1.0, -2.0, 3.0], &[3]), 1.0).unwrap();
        assert_eq!(p.packed_data, vec![128u8]);
        assert_eq!(
            p.unpack().unwrap().to_vec::<f32>().unwrap(),
            vec![1.0f32, -1.0, 1.0]
        );
    }

    /// `ternary_matmul` matches dense matmul and honors the input dtype (it
    /// used to matmul F32 weights against any input, erroring on F64).
    #[test]
    fn ternary_matmul_matches_dense_and_dtype() {
        let w = t32(&[1.0, -1.0, 0.0, 2.0], &[2, 2]);
        let p = TernaryPackedTensor::pack(&w, 1.0).unwrap();
        let x = t32(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
        let got = p.ternary_matmul(&x).unwrap().to_vec::<f32>().unwrap();
        // unpacked W = [[1,-1],[0,1]]; X @ W^T = [[-1,2],[-1,4]]
        assert_eq!(got, vec![-1.0f32, 2.0, -1.0, 4.0]);
        let x64 = Tensor::from_slice(&[1.0f64, 2.0, 3.0, 4.0], &[2, 2]).unwrap();
        let got64 = p.ternary_matmul(&x64).unwrap();
        assert_eq!(got64.dtype(), DType::F64);
        assert_eq!(
            got64.to_vec::<f64>().unwrap(),
            vec![-1.0f64, 2.0, -1.0, 4.0]
        );
    }

    /// BitLinear: gamma is exact mean|W|, packing is exact, and forward matches
    /// a fully hand-computed integer path (W=[[1,1]], X=[[2,4]]: eta=127/4,
    /// q=[64,127], out = 191/31.75).
    #[test]
    fn bitlinear_packs_and_runs() {
        let lin = Linear::from_tensors(t32(&[1.0, 1.0], &[1, 2]), None).unwrap();
        let bl = BitLinear::from_linear(&lin).unwrap();
        assert!((bl.weight_scale - 1.0).abs() < 1e-6);
        assert_eq!(bl.packed_weights, vec![0b0101u8]);
        assert_eq!(
            bl.dequantize_weight(DType::F32)
                .unwrap()
                .to_vec::<f32>()
                .unwrap(),
            vec![1.0f32, 1.0]
        );
        let out = bl
            .forward(&t32(&[2.0, 4.0], &[1, 2]))
            .unwrap()
            .to_vec::<f32>()
            .unwrap();
        assert!((out[0] - 191.0f32 / 31.75).abs() < 1e-3, "out {}", out[0]);
    }

    /// QuantizedLinear/AWQ/GPTQ/FP8Linear forward smoke: shapes right, finite,
    /// and close to the dense layer (Q8-level tolerance).
    #[test]
    fn quantized_linears_track_dense() {
        let w = t32(&[0.5, -1.5, 2.0, 1.0, -0.5, 0.25, 1.25, -2.0], &[2, 4]);
        let b = t32(&[0.1, -0.2], &[2]);
        let lin = Linear::from_tensors(w, Some(b)).unwrap();
        let x = t32(&[1.0, 0.5, -1.0, 2.0, 0.0, 1.0, 1.0, -0.5], &[2, 4]);
        let want = lin.forward(&x).unwrap().to_vec::<f32>().unwrap();

        let q = QuantizedLinear::from_linear(&lin, QuantScheme::DEFAULT_Q8_0).unwrap();
        let got = q.forward(&x).unwrap().to_vec::<f32>().unwrap();
        for (g, v) in got.iter().zip(want.iter()) {
            assert!((g - v).abs() < 0.15, "q8 {g} vs {v}");
        }

        let calib = t32(
            &[
                1.0, 2.0, 3.0, 4.0, 2.0, 1.0, 0.0, 1.0, 1.0, 1.0, 2.0, 0.0, 0.0, 3.0, 1.0, 2.0,
            ],
            &[4, 4],
        );
        let awq = AWQLinear::from_linear_calibrated(&lin, &calib, QuantScheme::DEFAULT_Q8_0, 0.5)
            .unwrap();
        let got = awq.forward(&x).unwrap().to_vec::<f32>().unwrap();
        for (g, v) in got.iter().zip(want.iter()) {
            assert!((g - v).abs() < 0.3, "awq {g} vs {v}");
        }

        // E4M3 ≈ 3-bit mantissa: per-weight ~6% error accumulates over the row.
        let f8 = FP8Linear::from_linear(&lin, FP8Format::E4M3).unwrap();
        let got = f8.forward(&x).unwrap().to_vec::<f32>().unwrap();
        for (g, v) in got.iter().zip(want.iter()) {
            assert!((g - v).abs() < 0.5, "fp8 {g} vs {v}");
        }

        let gptq =
            GPTQLinear::from_linear_calibrated(&lin, &calib, QuantScheme::DEFAULT_Q8_0, 0.01)
                .unwrap();
        let got = gptq.forward(&x).unwrap().to_vec::<f32>().unwrap();
        for (g, v) in got.iter().zip(want.iter()) {
            assert!((g - v).abs() < 0.3, "gptq {g} vs {v}");
        }
    }

    /// PackedBitLinear forward honors non-F32 input dtypes like the other linears.
    #[test]
    fn packed_bitlinear_forward_matches_dense_and_dtype() {
        let w = t32(&[1.0, -1.0, 0.0, 2.0], &[2, 2]);
        let lin = Linear::from_tensors(w, None).unwrap();
        let pb = PackedBitLinear::from_linear(&lin, 1.0).unwrap();
        let x = t32(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
        let got = pb.forward(&x).unwrap().to_vec::<f32>().unwrap();
        assert_eq!(got, vec![-1.0f32, 2.0, -1.0, 4.0]);
        let x64 = Tensor::from_slice(&[1.0f64, 2.0, 3.0, 4.0], &[2, 2]).unwrap();
        let got64 = pb.forward(&x64).unwrap();
        assert_eq!(got64.dtype(), DType::F64);
    }
}
