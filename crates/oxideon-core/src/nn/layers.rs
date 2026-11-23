//! Neural network layers: Linear, Conv1d, Conv2d, Pooling, BatchNorm, LayerNorm,
//! RMSNorm, Embedding, Dropout, MultiHeadAttention, and Activations.

use std::sync::atomic::{AtomicBool, Ordering};

use super::{Module, init};
use crate::dtype::DType;
use crate::error::{Error, Result};
use crate::rng;
use crate::tensor::{Tensor, no_grad};

// ---------------------------------------------------------------------------
// Linear (`weight: [out_features, in_features]` like PyTorch/safetensors)
// ---------------------------------------------------------------------------
#[derive(Clone)]
pub struct Linear {
    pub weight: Tensor,
    pub bias: Option<Tensor>,
    in_features: usize,
    out_features: usize,
}

impl Linear {
    pub fn new(in_features: usize, out_features: usize) -> Result<Self> {
        Self::with_bias(in_features, out_features, true)
    }

    pub fn with_bias(in_features: usize, out_features: usize, bias: bool) -> Result<Self> {
        // NOTE: in_features == 0 stays allowed: the weight is then empty and
        // zero-feature layers are exercised by quantization calibration tests.
        let bound = 1.0 / (in_features as f32).sqrt();
        let weight =
            init::uniform(&[out_features, in_features], -bound, bound)?.requires_grad_(true);
        let bias = if bias {
            Some(init::uniform(&[out_features], -bound, bound)?.requires_grad_(true))
        } else {
            None
        };
        Ok(Self {
            weight,
            bias,
            in_features,
            out_features,
        })
    }

    pub fn from_tensors(weight: Tensor, bias: Option<Tensor>) -> Result<Self> {
        if weight.ndim() != 2 {
            return Err(Error::Msg("Linear weight must be 2D [out, in]".into()));
        }
        let (out_features, in_features) = (weight.shape()[0], weight.shape()[1]);
        if let Some(b) = &bias {
            if b.shape() != [out_features] {
                return Err(Error::ShapeMismatch {
                    op: "Linear::from_tensors",
                    expected: vec![out_features],
                    got: b.shape().to_vec(),
                });
            }
        }
        Ok(Self {
            weight: weight.requires_grad_(true),
            bias: bias.map(|b| b.requires_grad_(true)),
            in_features,
            out_features,
        })
    }

    pub fn in_features(&self) -> usize {
        self.in_features
    }

    pub fn out_features(&self) -> usize {
        self.out_features
    }
}

impl Module for Linear {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        let orig = input.shape().to_vec();
        let k = *orig
            .last()
            .ok_or_else(|| Error::Msg("Linear requires rank >= 1".into()))?;
        let rows = match input.numel().checked_div(k) {
            Some(r) => r,
            None => orig[..orig.len() - 1]
                .iter()
                .try_fold(1usize, |acc, &d| acc.checked_mul(d))
                .ok_or_else(|| Error::Msg("Linear: leading dimensions overflow".into()))?,
        };
        let can_fuse = rows > 0
            && k == self.in_features()
            && self.out_features > 0
            && input.dtype() == crate::dtype::DType::F32
            && self.weight.dtype() == crate::dtype::DType::F32
            && input.device().is_cpu()
            && self.weight.device().is_cpu()
            && self.bias.as_ref().is_none_or(|b| b.dtype() == crate::dtype::DType::F32 && b.device().is_cpu())
            && !crate::amp::is_autocast_enabled();

        if can_fuse {
            let x2 = if input.ndim() == 2 { input.clone() } else { input.reshape(&[rows, k])? };
            let mut y2 = crate::tensor::no_grad(|| -> Result<Tensor> {
                let wt = self.weight.t()?;
                let mm = x2.matmul(&wt)?;
                if let Some(bias) = &self.bias {
                    if mm.storage().try_add_bias_inplace_f32(bias.storage(), bias.layout(), self.out_features) {
                        Ok(mm)
                    } else {
                        mm.add(bias)
                    }
                } else {
                    Ok(mm)
                }
            })?;
            if y2.ndim() != orig.len() {
                let mut out_shape = orig[..orig.len() - 1].to_vec();
                out_shape.push(self.out_features);
                y2 = y2.reshape(&out_shape)?;
            }
            let needs_grad = crate::tensor::is_grad_enabled()
                && (input.requires_grad()
                    || self.weight.requires_grad()
                    || self.bias.as_ref().is_some_and(Tensor::requires_grad));
            if !needs_grad {
                return Ok(y2);
            }
            let x_saved = input.clone();
            let w_saved = self.weight.clone();
            let b_saved = self.bias.clone();
            let out_f = self.out_features;
            let mut inputs = vec![x_saved.clone(), w_saved.clone()];
            if let Some(b) = &b_saved {
                inputs.push(b.clone());
            }
            let backward: crate::tensor::BackwardFn = Box::new(move |g: &Tensor| {
                let g2 = if g.ndim() == 2 { g.clone() } else { g.reshape(&[rows, out_f])? };
                let x_in = if x_saved.ndim() == 2 { x_saved.clone() } else { x_saved.reshape(&[rows, k])? };
                let gx = if x_saved.requires_grad() {
                    let dx2 = g2.matmul(&w_saved)?;
                    Some(if orig.len() == 2 { dx2 } else { dx2.reshape(&orig)? })
                } else {
                    None
                };
                let gw = if w_saved.requires_grad() {
                    Some(g2.t()?.matmul(&x_in)?)
                } else {
                    None
                };
                let mut grads = vec![gx, gw];
                if let Some(b) = &b_saved {
                    let gb = if b.requires_grad() {
                        Some(g2.sum_axis(0, false)?)
                    } else {
                        None
                    };
                    grads.push(gb);
                }
                Ok(grads)
            });
            return Ok(Tensor::from_parts(
                y2.storage().clone(),
                y2.layout().clone(),
                true,
                Some(crate::tensor::Node {
                    op: "linear",
                    inputs,
                    backward,
                }),
            ));
        }

        // Because `self.weight.t()` is a zero-copy view and our GEMM accepts arbitrary
        // strides directly, `input.matmul(&self.weight.t()?)` never copies `weight`!
        let wt = self.weight.t()?;
        let y = if input.ndim() == 2 {
            input.matmul(&wt)?
        } else {
            let flat = input.reshape(&[rows, k])?.matmul(&wt)?;
            let mut out_shape = orig[..orig.len() - 1].to_vec();
            out_shape.push(self.out_features);
            flat.reshape(&out_shape)?
        };
        match &self.bias {
            Some(b) => y.add(&crate::amp::autocast_cast_like(b, &y)?),
            None => Ok(y),
        }
    }

    fn named_parameters(&self) -> Vec<(String, Tensor)> {
        let mut p = vec![("weight".to_string(), self.weight.clone())];
        if let Some(b) = &self.bias {
            p.push(("bias".to_string(), b.clone()));
        }
        p
    }
}

// ---------------------------------------------------------------------------
// Conv2d & Conv1d
// ---------------------------------------------------------------------------
pub struct Conv2d {
    pub weight: Tensor,
    pub bias: Option<Tensor>,
    pub stride: (usize, usize),
    pub padding: (usize, usize),
    pub dilation: (usize, usize),
    pub groups: usize,
}

impl Conv2d {
    pub fn new(
        in_channels: usize,
        out_channels: usize,
        kernel_size: (usize, usize),
        stride: (usize, usize),
        padding: (usize, usize),
    ) -> Result<Self> {
        Self::with_groups(in_channels, out_channels, kernel_size, stride, padding, 1)
    }

    pub fn with_groups(
        in_channels: usize,
        out_channels: usize,
        kernel_size: (usize, usize),
        stride: (usize, usize),
        padding: (usize, usize),
        groups: usize,
    ) -> Result<Self> {
        if groups == 0 || in_channels % groups != 0 || out_channels % groups != 0 {
            return Err(crate::error::Error::Msg(format!(
                "in_channels ({in_channels}) and out_channels ({out_channels}) must be divisible by a positive groups ({groups})"
            )));
        }
        let fan_in = (in_channels / groups)
            .checked_mul(kernel_size.0)
            .and_then(|v| v.checked_mul(kernel_size.1))
            .ok_or_else(|| {
                crate::error::Error::Msg(format!(
                    "Conv2d: fan_in dimensions overflow (in_channels {in_channels}, kernel {kernel_size:?})"
                ))
            })?;
        if fan_in == 0 {
            return Err(crate::error::Error::Msg(
                "Conv2d: fan_in must be positive".into(),
            ));
        }
        let bound = 1.0 / (fan_in as f32).sqrt();
        let weight = init::uniform(
            &[
                out_channels,
                in_channels / groups,
                kernel_size.0,
                kernel_size.1,
            ],
            -bound,
            bound,
        )?
        .requires_grad_(true);
        let bias = Some(init::uniform(&[out_channels], -bound, bound)?.requires_grad_(true));
        Ok(Self {
            weight,
            bias,
            stride,
            padding,
            dilation: (1, 1),
            groups,
        })
    }

    pub fn with_dilation(mut self, dilation: (usize, usize)) -> Self {
        self.dilation = dilation;
        self
    }
}

impl Module for Conv2d {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        input.conv2d_grouped(
            &self.weight,
            self.bias.as_ref(),
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )
    }

    fn named_parameters(&self) -> Vec<(String, Tensor)> {
        let mut p = vec![("weight".to_string(), self.weight.clone())];
        if let Some(b) = &self.bias {
            p.push(("bias".to_string(), b.clone()));
        }
        p
    }
}

pub struct ConvTranspose2d {
    pub weight: Tensor,
    pub bias: Option<Tensor>,
    pub stride: (usize, usize),
    pub padding: (usize, usize),
    pub output_padding: (usize, usize),
    pub dilation: (usize, usize),
}

impl ConvTranspose2d {
    pub fn new(
        in_channels: usize,
        out_channels: usize,
        kernel_size: (usize, usize),
        stride: (usize, usize),
        padding: (usize, usize),
    ) -> Result<Self> {
        let fan_in = in_channels
            .checked_mul(kernel_size.0)
            .and_then(|v| v.checked_mul(kernel_size.1))
            .ok_or_else(|| {
                crate::error::Error::Msg(format!(
                    "ConvTranspose2d: fan_in dimensions overflow (in_channels {in_channels}, kernel {kernel_size:?})"
                ))
            })?;
        if fan_in == 0 {
            return Err(crate::error::Error::Msg(
                "ConvTranspose2d: fan_in must be positive".into(),
            ));
        }
        let bound = 1.0 / (fan_in as f32).sqrt();
        let weight = init::uniform(
            &[in_channels, out_channels, kernel_size.0, kernel_size.1],
            -bound,
            bound,
        )?
        .requires_grad_(true);
        let bias = Some(init::uniform(&[out_channels], -bound, bound)?.requires_grad_(true));
        Ok(Self {
            weight,
            bias,
            stride,
            padding,
            output_padding: (0, 0),
            dilation: (1, 1),
        })
    }

    pub fn with_output_padding(mut self, output_padding: (usize, usize)) -> Self {
        self.output_padding = output_padding;
        self
    }
}

impl Module for ConvTranspose2d {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        input.conv_transpose2d(
            &self.weight,
            self.bias.as_ref(),
            self.stride,
            self.padding,
            self.output_padding,
            self.dilation,
        )
    }

    fn named_parameters(&self) -> Vec<(String, Tensor)> {
        let mut p = vec![("weight".to_string(), self.weight.clone())];
        if let Some(b) = &self.bias {
            p.push(("bias".to_string(), b.clone()));
        }
        p
    }
}

pub struct ConvTranspose1d {
    pub weight: Tensor,
    pub bias: Option<Tensor>,
    pub stride: usize,
    pub padding: usize,
    pub output_padding: usize,
    pub dilation: usize,
}

impl ConvTranspose1d {
    pub fn new(
        in_channels: usize,
        out_channels: usize,
        kernel_size: usize,
        stride: usize,
        padding: usize,
    ) -> Result<Self> {
        let fan_in = in_channels.checked_mul(kernel_size).ok_or_else(|| {
            crate::error::Error::Msg(format!(
                "ConvTranspose1d: fan_in dimensions overflow (in_channels {in_channels}, kernel {kernel_size})"
            ))
        })?;
        if fan_in == 0 {
            return Err(crate::error::Error::Msg(
                "ConvTranspose1d: fan_in must be positive".into(),
            ));
        }
        let bound = 1.0 / (fan_in as f32).sqrt();
        let weight = init::uniform(&[in_channels, out_channels, kernel_size], -bound, bound)?
            .requires_grad_(true);
        let bias = Some(init::uniform(&[out_channels], -bound, bound)?.requires_grad_(true));
        Ok(Self {
            weight,
            bias,
            stride,
            padding,
            output_padding: 0,
            dilation: 1,
        })
    }
}

impl Module for ConvTranspose1d {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        input.conv_transpose1d(
            &self.weight,
            self.bias.as_ref(),
            self.stride,
            self.padding,
            self.output_padding,
            self.dilation,
        )
    }

    fn named_parameters(&self) -> Vec<(String, Tensor)> {
        let mut p = vec![("weight".to_string(), self.weight.clone())];
        if let Some(b) = &self.bias {
            p.push(("bias".to_string(), b.clone()));
        }
        p
    }
}

pub struct Conv1d {
    pub weight: Tensor,
    pub bias: Option<Tensor>,
    pub stride: usize,
    pub padding: usize,
    pub dilation: usize,
}

impl Conv1d {
    pub fn new(
        in_channels: usize,
        out_channels: usize,
        kernel_size: usize,
        stride: usize,
        padding: usize,
    ) -> Result<Self> {
        let fan_in = in_channels.checked_mul(kernel_size).ok_or_else(|| {
            crate::error::Error::Msg(format!(
                "Conv1d: fan_in dimensions overflow (in_channels {in_channels}, kernel {kernel_size})"
            ))
        })?;
        if fan_in == 0 {
            return Err(crate::error::Error::Msg(
                "Conv1d: fan_in must be positive".into(),
            ));
        }
        let bound = 1.0 / (fan_in as f32).sqrt();
        let weight = init::uniform(&[out_channels, in_channels, kernel_size], -bound, bound)?
            .requires_grad_(true);
        let bias = Some(init::uniform(&[out_channels], -bound, bound)?.requires_grad_(true));
        Ok(Self {
            weight,
            bias,
            stride,
            padding,
            dilation: 1,
        })
    }
}

impl Module for Conv1d {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        input.conv1d(
            &self.weight,
            self.bias.as_ref(),
            self.stride,
            self.padding,
            self.dilation,
        )
    }

    fn named_parameters(&self) -> Vec<(String, Tensor)> {
        let mut p = vec![("weight".to_string(), self.weight.clone())];
        if let Some(b) = &self.bias {
            p.push(("bias".to_string(), b.clone()));
        }
        p
    }
}

// ---------------------------------------------------------------------------
// Pooling & Flatten
// ---------------------------------------------------------------------------
pub struct MaxPool1d {
    pub kernel_size: usize,
    pub stride: usize,
    pub padding: usize,
}

impl MaxPool1d {
    pub fn new(kernel_size: usize, stride: usize) -> Self {
        Self {
            kernel_size,
            stride,
            padding: 0,
        }
    }

    pub fn with_padding(mut self, padding: usize) -> Self {
        self.padding = padding;
        self
    }
}

impl Module for MaxPool1d {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        input.max_pool1d(self.kernel_size, self.stride, self.padding)
    }
}

pub struct AvgPool1d {
    pub kernel_size: usize,
    pub stride: usize,
    pub padding: usize,
}

impl AvgPool1d {
    pub fn new(kernel_size: usize, stride: usize) -> Self {
        Self {
            kernel_size,
            stride,
            padding: 0,
        }
    }

    pub fn with_padding(mut self, padding: usize) -> Self {
        self.padding = padding;
        self
    }
}

impl Module for AvgPool1d {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        input.avg_pool1d(self.kernel_size, self.stride, self.padding)
    }
}

#[derive(Clone, Copy, Debug)]
pub struct AdaptiveAvgPool1d(pub usize);

impl AdaptiveAvgPool1d {
    pub fn new(output_size: usize) -> Self {
        Self(output_size)
    }
}

impl Module for AdaptiveAvgPool1d {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        input.adaptive_avg_pool1d(self.0)
    }
}

#[derive(Clone, Copy, Debug)]
pub struct AdaptiveMaxPool1d(pub usize);

impl AdaptiveMaxPool1d {
    pub fn new(output_size: usize) -> Self {
        Self(output_size)
    }
}

impl Module for AdaptiveMaxPool1d {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        input.adaptive_max_pool1d(self.0)
    }
}

pub struct LPPool1d {
    pub norm_type: f64,
    pub kernel_size: usize,
    pub stride: usize,
}

impl LPPool1d {
    pub fn new(norm_type: f64, kernel_size: usize, stride: usize) -> Self {
        Self {
            norm_type,
            kernel_size,
            stride,
        }
    }
}

impl Module for LPPool1d {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        input.lp_pool1d(self.norm_type, self.kernel_size, self.stride)
    }
}

pub struct LPPool2d {
    pub norm_type: f64,
    pub kernel_size: (usize, usize),
    pub stride: (usize, usize),
}

impl LPPool2d {
    pub fn new(norm_type: f64, kernel_size: (usize, usize), stride: (usize, usize)) -> Self {
        Self {
            norm_type,
            kernel_size,
            stride,
        }
    }
}

impl Module for LPPool2d {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        input.lp_pool2d(self.norm_type, self.kernel_size, self.stride)
    }
}

pub struct MaxPool2d {
    pub kernel_size: (usize, usize),
    pub stride: (usize, usize),
    pub padding: (usize, usize),
}

impl MaxPool2d {
    pub fn new(kernel_size: (usize, usize), stride: (usize, usize)) -> Self {
        Self {
            kernel_size,
            stride,
            padding: (0, 0),
        }
    }
}

impl Module for MaxPool2d {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        input.max_pool2d(self.kernel_size, self.stride, self.padding)
    }
}

pub struct AvgPool2d {
    pub kernel_size: (usize, usize),
    pub stride: (usize, usize),
    pub padding: (usize, usize),
}

impl AvgPool2d {
    pub fn new(kernel_size: (usize, usize), stride: (usize, usize)) -> Self {
        Self {
            kernel_size,
            stride,
            padding: (0, 0),
        }
    }
}

impl Module for AvgPool2d {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        input.avg_pool2d(self.kernel_size, self.stride, self.padding)
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub struct Flatten;

impl Module for Flatten {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        input.flatten()
    }
}

#[derive(Clone, Copy, Debug)]
pub struct AdaptiveAvgPool2d(pub (usize, usize));

impl AdaptiveAvgPool2d {
    pub fn new(output_size: (usize, usize)) -> Self {
        Self(output_size)
    }
}

impl Module for AdaptiveAvgPool2d {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        input.adaptive_avg_pool2d(self.0)
    }
}

#[derive(Clone, Copy, Debug)]
pub struct AdaptiveMaxPool2d(pub (usize, usize));

impl AdaptiveMaxPool2d {
    pub fn new(output_size: (usize, usize)) -> Self {
        Self(output_size)
    }
}

#[derive(Clone, Copy, Debug)]
pub enum UpsampleMode {
    Nearest,
    Bilinear { align_corners: bool },
}

#[derive(Clone, Copy, Debug)]
pub struct Upsample2d {
    pub output_size: (usize, usize),
    pub mode: UpsampleMode,
}

impl Upsample2d {
    pub fn nearest(output_size: (usize, usize)) -> Self {
        Self {
            output_size,
            mode: UpsampleMode::Nearest,
        }
    }

    pub fn bilinear(output_size: (usize, usize), align_corners: bool) -> Self {
        Self {
            output_size,
            mode: UpsampleMode::Bilinear { align_corners },
        }
    }
}

impl Module for Upsample2d {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        match self.mode {
            UpsampleMode::Nearest => input.upsample_nearest2d(self.output_size),
            UpsampleMode::Bilinear { align_corners } => {
                input.upsample_bilinear2d(self.output_size, align_corners)
            }
        }
    }
}

/// Group Normalization (`[N, C, ...]`) dividing `num_channels` into `num_groups`.
pub struct GroupNorm {
    pub weight: Tensor,
    pub bias: Tensor,
    pub num_groups: usize,
    pub num_channels: usize,
    pub eps: f64,
}

impl GroupNorm {
    pub fn new(num_groups: usize, num_channels: usize, eps: f64) -> Result<Self> {
        if num_groups == 0 || num_channels % num_groups != 0 {
            return Err(crate::error::Error::Msg(format!(
                "num_channels ({num_channels}) must be divisible by a positive num_groups ({num_groups})"
            )));
        }
        Ok(Self {
            weight: Tensor::ones_f32(&[num_channels])?.requires_grad_(true),
            bias: Tensor::zeros_f32(&[num_channels])?.requires_grad_(true),
            num_groups,
            num_channels,
            eps,
        })
    }
}

impl Module for GroupNorm {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        let orig_shape = input.shape().to_vec();
        if orig_shape.len() < 2 {
            return Err(crate::error::Error::Msg(format!(
                "GroupNorm expects rank >= 2 [N, C, ...], got {orig_shape:?}"
            )));
        }
        let n = orig_shape[0];
        let c = orig_shape[1];
        if c != self.num_channels {
            return Err(crate::error::Error::Msg(format!(
                "GroupNorm channel mismatch: got {c}, expected {}",
                self.num_channels
            )));
        }
        let spatial: usize = orig_shape[2..].iter().product();
        let g = self.num_groups;

        let grouped = input.reshape(&[n, g, (c / g) * spatial])?;
        let (centered, _) = grouped.centered_by_mean(&[2])?;
        let var = centered.square()?.mean_axis(2, true)?;
        let normed = centered
            .mul(&var.add_scalar(self.eps)?.rsqrt()?)?
            .reshape(&orig_shape)?;

        let mut param_shape = vec![1usize; orig_shape.len()];
        param_shape[1] = c;
        let w = self.weight.reshape(&param_shape)?;
        let b = self.bias.reshape(&param_shape)?;
        normed.mul(&w)?.add(&b)
    }

    fn named_parameters(&self) -> Vec<(String, Tensor)> {
        vec![
            ("weight".to_string(), self.weight.clone()),
            ("bias".to_string(), self.bias.clone()),
        ]
    }
}

/// 1D Instance Normalization (`torch.nn.InstanceNorm1d`).
pub struct InstanceNorm1d(pub GroupNorm);

impl InstanceNorm1d {
    pub fn new(num_channels: usize, eps: f64) -> Result<Self> {
        Ok(Self(GroupNorm::new(num_channels, num_channels, eps)?))
    }
}

impl Module for InstanceNorm1d {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        self.0.forward(input)
    }

    fn named_parameters(&self) -> Vec<(String, Tensor)> {
        self.0.named_parameters()
    }
}

/// 2D Instance Normalization (`GroupNorm` where `num_groups == num_channels`).
pub struct InstanceNorm2d(pub GroupNorm);

impl InstanceNorm2d {
    pub fn new(num_channels: usize, eps: f64) -> Result<Self> {
        Ok(Self(GroupNorm::new(num_channels, num_channels, eps)?))
    }
}

impl Module for InstanceNorm2d {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        self.0.forward(input)
    }

    fn named_parameters(&self) -> Vec<(String, Tensor)> {
        self.0.named_parameters()
    }
}

/// 3D Instance Normalization (`torch.nn.InstanceNorm3d`).
pub struct InstanceNorm3d(pub GroupNorm);

impl InstanceNorm3d {
    pub fn new(num_channels: usize, eps: f64) -> Result<Self> {
        Ok(Self(GroupNorm::new(num_channels, num_channels, eps)?))
    }
}

impl Module for InstanceNorm3d {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        self.0.forward(input)
    }

    fn named_parameters(&self) -> Vec<(String, Tensor)> {
        self.0.named_parameters()
    }
}

/// ResNet Basic Residual Block (`Conv3x3 -> BN -> ReLU -> Conv3x3 -> BN + shortcut -> ReLU`).
pub struct ResidualBlock {
    pub conv1: Conv2d,
    pub bn1: BatchNorm2d,
    pub conv2: Conv2d,
    pub bn2: BatchNorm2d,
    pub downsample: Option<(Conv2d, BatchNorm2d)>,
}

impl ResidualBlock {
    pub fn new(in_channels: usize, out_channels: usize, stride: usize) -> Result<Self> {
        let conv1 = Conv2d::new(in_channels, out_channels, (3, 3), (stride, stride), (1, 1))?;
        let bn1 = BatchNorm2d::new(out_channels)?;
        let conv2 = Conv2d::new(out_channels, out_channels, (3, 3), (1, 1), (1, 1))?;
        let bn2 = BatchNorm2d::new(out_channels)?;
        let downsample = if stride != 1 || in_channels != out_channels {
            Some((
                Conv2d::new(in_channels, out_channels, (1, 1), (stride, stride), (0, 0))?,
                BatchNorm2d::new(out_channels)?,
            ))
        } else {
            None
        };
        Ok(Self {
            conv1,
            bn1,
            conv2,
            bn2,
            downsample,
        })
    }
}

impl Module for ResidualBlock {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let identity = match &self.downsample {
            Some((c, bn)) => bn.forward(&c.forward(x)?)?,
            None => x.clone(),
        };
        let out = self.bn1.forward(&self.conv1.forward(x)?)?.relu()?;
        let out = self.bn2.forward(&self.conv2.forward(&out)?)?;
        out.add(&identity)?.relu()
    }

    fn named_parameters(&self) -> Vec<(String, Tensor)> {
        let mut out = Vec::new();
        for (p, m) in [("conv1", &self.conv1), ("conv2", &self.conv2)] {
            for (k, v) in m.named_parameters() {
                out.push((format!("{p}.{k}"), v));
            }
        }
        for (p, m) in [("bn1", &self.bn1), ("bn2", &self.bn2)] {
            for (k, v) in m.named_parameters() {
                out.push((format!("{p}.{k}"), v));
            }
        }
        if let Some((dc, dbn)) = &self.downsample {
            for (k, v) in dc.named_parameters() {
                out.push((format!("downsample.0.{k}"), v));
            }
            for (k, v) in dbn.named_parameters() {
                out.push((format!("downsample.1.{k}"), v));
            }
        }
        out
    }

    fn named_buffers(&self) -> Vec<(String, Tensor)> {
        let mut out = Vec::new();
        for (p, m) in [("bn1", &self.bn1), ("bn2", &self.bn2)] {
            for (k, v) in m.named_buffers() {
                out.push((format!("{p}.{k}"), v));
            }
        }
        if let Some((_, dbn)) = &self.downsample {
            for (k, v) in dbn.named_buffers() {
                out.push((format!("downsample.1.{k}"), v));
            }
        }
        out
    }

    fn train(&self, mode: bool) {
        self.bn1.train(mode);
        self.bn2.train(mode);
        if let Some((_, dbn)) = &self.downsample {
            dbn.train(mode);
        }
    }
}

// ---------------------------------------------------------------------------
// Normalization: LayerNorm, RMSNorm, BatchNorm1d, BatchNorm2d
// ---------------------------------------------------------------------------
pub struct LayerNorm {
    pub weight: Tensor,
    pub bias: Tensor,
    pub eps: f64,
    pub normalized_shape: Vec<usize>,
}

impl LayerNorm {
    pub fn new(normalized_dim: usize, eps: f64) -> Result<Self> {
        Self::with_shape(&[normalized_dim], eps)
    }

    pub fn with_shape(normalized_shape: &[usize], eps: f64) -> Result<Self> {
        if normalized_shape.is_empty() {
            return Err(Error::Msg(
                "LayerNorm normalized_shape cannot be empty".to_string(),
            ));
        }
        for &d in normalized_shape {
            if d == 0 {
                return Err(Error::Msg(
                    "LayerNorm dimensions must be positive non-zero".to_string(),
                ));
            }
        }
        Ok(Self {
            weight: Tensor::ones_f32(normalized_shape)?.requires_grad_(true),
            bias: Tensor::zeros_f32(normalized_shape)?.requires_grad_(true),
            eps,
            normalized_shape: normalized_shape.to_vec(),
        })
    }
}

impl Module for LayerNorm {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        let k = self.normalized_shape.len();
        if input.ndim() < k {
            return Err(Error::Msg(format!(
                "LayerNorm input ndim {} must be >= normalized_shape len {}",
                input.ndim(),
                k
            )));
        }
        let in_shape = input.shape();
        let trailing = &in_shape[in_shape.len() - k..];
        if trailing != self.normalized_shape.as_slice() {
            return Err(Error::Msg(format!(
                "LayerNorm input trailing shape {:?} does not match normalized_shape {:?}",
                trailing, self.normalized_shape
            )));
        }

        if k == 1
            && let Some(out_storage) = input.storage().try_layer_norm_last_axis(
                input.layout(),
                self.weight.storage(),
                self.weight.layout(),
                self.bias.storage(),
                self.bias.layout(),
                self.eps,
            )
        {
            let out_layout = crate::layout::Layout::contiguous(in_shape);
            let needs_grad = crate::tensor::is_grad_enabled()
                && (input.requires_grad() || self.weight.requires_grad() || self.bias.requires_grad());
            if !needs_grad {
                return Ok(Tensor::from_parts(out_storage, out_layout, false, None));
            }
            let x_saved = input.clone();
            let w_saved = self.weight.clone();
            let b_saved = self.bias.clone();
            let eps = self.eps;
            let backward: crate::tensor::BackwardFn = Box::new(move |g: &Tensor| {
                if !crate::tensor::is_grad_enabled()
                    && let Some((dx_s, dw_s, db_s)) = x_saved.storage().try_layer_norm_bwd_f32(
                        x_saved.layout(),
                        g.storage(),
                        g.layout(),
                        w_saved.storage(),
                        w_saved.layout(),
                        eps,
                    )
                {
                    let gx = x_saved
                        .requires_grad()
                        .then(|| Tensor::from_parts(dx_s, crate::layout::Layout::contiguous(x_saved.shape().to_vec()), false, None));
                    let gw = w_saved
                        .requires_grad()
                        .then(|| Tensor::from_parts(dw_s, crate::layout::Layout::contiguous(w_saved.shape().to_vec()), false, None));
                    let gb = b_saved
                        .requires_grad()
                        .then(|| Tensor::from_parts(db_s, crate::layout::Layout::contiguous(b_saved.shape().to_vec()), false, None));
                    return Ok(vec![gx, gw, gb]);
                }
                let (centered, _) = x_saved.centered_by_mean(&[-1])?;
                let var = centered.square()?.mean_axes(&[-1], true)?;
                let inv_std = var.add_scalar(eps)?.rsqrt()?;
                let x_hat = centered.mul(&inv_std)?;

                let gx = if x_saved.requires_grad() {
                    let g_xhat = g.mul(&w_saved)?;
                    let mean_g_xhat = g_xhat.mean_axes(&[-1], true)?;
                    let mean_g_xhat_xhat = g_xhat.mul(&x_hat)?.mean_axes(&[-1], true)?;
                    let dx = g_xhat.sub(&mean_g_xhat)?.sub(&x_hat.mul(&mean_g_xhat_xhat)?)?.mul(&inv_std)?;
                    Some(dx)
                } else {
                    None
                };
                let sum_axes: Vec<isize> = (0..x_saved.ndim().saturating_sub(1) as isize).collect();
                let gw = if w_saved.requires_grad() {
                    let prod = g.mul(&x_hat)?;
                    Some(if sum_axes.is_empty() { prod } else { prod.sum_axes(&sum_axes, false)? })
                } else {
                    None
                };
                let gb = if b_saved.requires_grad() {
                    Some(if sum_axes.is_empty() { g.clone() } else { g.sum_axes(&sum_axes, false)? })
                } else {
                    None
                };
                Ok(vec![gx, gw, gb])
            });
            return Ok(Tensor::from_parts(
                out_storage,
                out_layout,
                true,
                Some(crate::tensor::Node {
                    op: "layer_norm",
                    inputs: vec![input.clone(), self.weight.clone(), self.bias.clone()],
                    backward,
                }),
            ));
        }

        let axes: Vec<isize> = (-(k as isize)..0).collect();
        let (centered, _) = input.centered_by_mean(&axes)?;
        let var = centered.square()?.mean_axes(&axes, true)?;
        let inv_std = var.add_scalar(self.eps)?.rsqrt()?;
        centered.mul(&inv_std)?.mul(&self.weight)?.add(&self.bias)
    }

    fn named_parameters(&self) -> Vec<(String, Tensor)> {
        vec![
            ("weight".to_string(), self.weight.clone()),
            ("bias".to_string(), self.bias.clone()),
        ]
    }
}

#[derive(Clone)]
pub struct RMSNorm {
    pub weight: Tensor,
    pub eps: f64,
}

impl RMSNorm {
    pub fn new(dim: usize, eps: f64) -> Result<Self> {
        Ok(Self {
            weight: Tensor::ones_f32(&[dim])?.requires_grad_(true),
            eps,
        })
    }
}

impl Module for RMSNorm {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        let rms_inv = input
            .square()?
            .mean_axis(-1, true)?
            .add_scalar(self.eps)?
            .rsqrt()?;
        input.mul(&rms_inv)?.mul(&self.weight)
    }

    fn named_parameters(&self) -> Vec<(String, Tensor)> {
        vec![("weight".to_string(), self.weight.clone())]
    }
}

pub struct BatchNorm1d {
    pub weight: Tensor,
    pub bias: Tensor,
    pub running_mean: Tensor,
    pub running_var: Tensor,
    pub momentum: f64,
    pub eps: f64,
    training: AtomicBool,
}

impl BatchNorm1d {
    pub fn new(num_features: usize) -> Result<Self> {
        Ok(Self {
            weight: Tensor::ones_f32(&[1, num_features])?.requires_grad_(true),
            bias: Tensor::zeros_f32(&[1, num_features])?.requires_grad_(true),
            running_mean: Tensor::zeros_f32(&[1, num_features])?,
            running_var: Tensor::ones_f32(&[1, num_features])?,
            momentum: 0.1,
            eps: 1e-5,
            training: AtomicBool::new(true),
        })
    }
}

impl Module for BatchNorm1d {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        if input.ndim() != 2 && input.ndim() != 3 {
            return Err(Error::Msg(format!(
                "BatchNorm1d expects [N, C] or [N, C, L] input, got shape {:?}",
                input.shape()
            )));
        }
        let mut batch_centered: Option<Tensor> = None;
        let (mu, var) = if self.training.load(Ordering::Relaxed) {
            // [N, C] reduces over the batch; [N, C, L] over batch and length.
            let reduce: &[isize] = if input.ndim() == 3 { &[0, 2] } else { &[0] };
            // Torch parity: training needs more than 1 value per channel, and
            // the running variance tracks the Bessel-corrected (unbiased)
            // batch variance while the output normalizes with the biased one.
            let n: usize = reduce.iter().map(|&a| input.shape()[a as usize]).product();
            if n <= 1 {
                return Err(Error::Msg(format!(
                    "BatchNorm1d training needs more than 1 value per channel, got shape {:?}",
                    input.shape()
                )));
            }
            let (centered, mu) = input.centered_by_mean(reduce)?;
            let var = centered.square()?.mean_axes(reduce, true)?;
            batch_centered = Some(centered);
            let unbiased = var.mul_scalar(n as f64 / (n - 1) as f64)?;
            no_grad(|| -> Result<()> {
                let m = self.momentum;
                // Squeeze the batch statistics onto the running-stat shape: for 3-D input the
                // keepdim mean is [1, C, 1] and must not broadcast against [1, C] here.
                let stat_shape = self.running_mean.shape().to_vec();
                let new_mu = self
                    .running_mean
                    .mul_scalar(1.0 - m)?
                    .add(&mu.detach().mul_scalar(m)?.reshape(&stat_shape)?)?;
                let new_var = self
                    .running_var
                    .mul_scalar(1.0 - m)?
                    .add(&unbiased.detach().mul_scalar(m)?.reshape(&stat_shape)?)?;
                self.running_mean.assign(&new_mu)?;
                self.running_var.assign(&new_var)
            })?;
            (mu, var)
        } else if input.ndim() == 3 {
            // View the running stats in keepdim form so 3-D input broadcasts
            // per channel; raw [1, C] would align C against L instead.
            let c = self.running_mean.shape()[1];
            (
                self.running_mean.reshape(&[1, c, 1])?,
                self.running_var.reshape(&[1, c, 1])?,
            )
        } else {
            (self.running_mean.clone(), self.running_var.clone())
        };
        let centered = match batch_centered {
            Some(c) => c,
            None => input.sub(&mu)?,
        };
        let norm = centered.mul(&var.add_scalar(self.eps)?.rsqrt()?)?;
        if input.ndim() == 3 {
            // Same keepdim view for the affine parameters (a square C == L
            // input would otherwise broadcast weight along L silently).
            let c = self.weight.shape()[1];
            norm.mul(&self.weight.reshape(&[1, c, 1])?)?
                .add(&self.bias.reshape(&[1, c, 1])?)
        } else {
            norm.mul(&self.weight)?.add(&self.bias)
        }
    }

    fn named_parameters(&self) -> Vec<(String, Tensor)> {
        vec![
            ("weight".to_string(), self.weight.clone()),
            ("bias".to_string(), self.bias.clone()),
        ]
    }

    fn named_buffers(&self) -> Vec<(String, Tensor)> {
        vec![
            ("running_mean".to_string(), self.running_mean.clone()),
            ("running_var".to_string(), self.running_var.clone()),
        ]
    }

    fn train(&self, mode: bool) {
        self.training.store(mode, Ordering::Relaxed);
    }
}

pub struct BatchNorm2d {
    pub weight: Tensor,
    pub bias: Tensor,
    pub running_mean: Tensor,
    pub running_var: Tensor,
    pub momentum: f64,
    pub eps: f64,
    training: AtomicBool,
}

impl BatchNorm2d {
    pub fn new(num_features: usize) -> Result<Self> {
        Ok(Self {
            weight: Tensor::ones_f32(&[1, num_features, 1, 1])?.requires_grad_(true),
            bias: Tensor::zeros_f32(&[1, num_features, 1, 1])?.requires_grad_(true),
            running_mean: Tensor::zeros_f32(&[1, num_features, 1, 1])?,
            running_var: Tensor::ones_f32(&[1, num_features, 1, 1])?,
            momentum: 0.1,
            eps: 1e-5,
            training: AtomicBool::new(true),
        })
    }
}

impl Module for BatchNorm2d {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        if input.ndim() != 4 {
            return Err(Error::Msg(format!(
                "BatchNorm2d expects [N, C, H, W] input, got shape {:?}",
                input.shape()
            )));
        }
        let mut batch_centered: Option<Tensor> = None;
        let (mu, var) = if self.training.load(Ordering::Relaxed) {
            // Torch parity: training needs more than 1 value per channel, and
            // the running variance tracks the Bessel-corrected batch variance.
            let shape = input.shape();
            let n = shape[0] * shape[2] * shape[3];
            if n <= 1 {
                return Err(Error::Msg(format!(
                    "BatchNorm2d training needs more than 1 value per channel, got shape {:?}",
                    input.shape()
                )));
            }
            let (centered, mu) = input.centered_by_mean(&[0, 2, 3])?;
            let var = centered.square()?.mean_axes(&[0, 2, 3], true)?;
            batch_centered = Some(centered);
            let unbiased = var.mul_scalar(n as f64 / (n - 1) as f64)?;
            no_grad(|| -> Result<()> {
                let m = self.momentum;
                let new_mu = self
                    .running_mean
                    .mul_scalar(1.0 - m)?
                    .add(&mu.detach().mul_scalar(m)?)?;
                let new_var = self
                    .running_var
                    .mul_scalar(1.0 - m)?
                    .add(&unbiased.detach().mul_scalar(m)?)?;
                self.running_mean.assign(&new_mu)?;
                self.running_var.assign(&new_var)
            })?;
            (mu, var)
        } else {
            (self.running_mean.clone(), self.running_var.clone())
        };
        let centered = match batch_centered {
            Some(c) => c,
            None => input.sub(&mu)?,
        };
        let norm = centered.mul(&var.add_scalar(self.eps)?.rsqrt()?)?;
        norm.mul(&self.weight)?.add(&self.bias)
    }

    fn named_parameters(&self) -> Vec<(String, Tensor)> {
        vec![
            ("weight".to_string(), self.weight.clone()),
            ("bias".to_string(), self.bias.clone()),
        ]
    }

    fn named_buffers(&self) -> Vec<(String, Tensor)> {
        vec![
            ("running_mean".to_string(), self.running_mean.clone()),
            ("running_var".to_string(), self.running_var.clone()),
        ]
    }

    fn train(&self, mode: bool) {
        self.training.store(mode, Ordering::Relaxed);
    }
}

// ---------------------------------------------------------------------------
// Embedding, Dropout & MultiHeadAttention
// ---------------------------------------------------------------------------
#[derive(Clone)]
pub struct Embedding {
    pub weight: Tensor,
    num_embeddings: usize,
    embedding_dim: usize,
}

impl Embedding {
    pub fn new(num_embeddings: usize, embedding_dim: usize) -> Result<Self> {
        let weight = Tensor::randn(&[num_embeddings, embedding_dim])?.requires_grad_(true);
        Ok(Self {
            weight,
            num_embeddings,
            embedding_dim,
        })
    }

    pub fn num_embeddings(&self) -> usize {
        self.num_embeddings
    }

    pub fn embedding_dim(&self) -> usize {
        self.embedding_dim
    }
}

impl Module for Embedding {
    fn forward(&self, indices: &Tensor) -> Result<Tensor> {
        let orig_shape = indices.shape().to_vec();
        let flat = indices.reshape(&[indices.numel()])?;
        let selected = self.weight.index_select(0, &flat)?;
        let mut out_shape = orig_shape;
        out_shape.push(self.embedding_dim);
        selected.reshape(&out_shape)
    }

    fn named_parameters(&self) -> Vec<(String, Tensor)> {
        vec![("weight".to_string(), self.weight.clone())]
    }
}

/// Reduction mode for [`EmbeddingBag`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EmbeddingBagMode {
    Sum,
    Mean,
    Max,
}

/// EmbeddingBag module (`torch.nn.EmbeddingBag`) computing sums, means, or maxes of "bags" of embeddings.
pub struct EmbeddingBag {
    pub weight: Tensor,
    num_embeddings: usize,
    embedding_dim: usize,
    pub mode: EmbeddingBagMode,
}

impl EmbeddingBag {
    pub fn new(
        num_embeddings: usize,
        embedding_dim: usize,
        mode: EmbeddingBagMode,
    ) -> Result<Self> {
        let weight = Tensor::randn(&[num_embeddings, embedding_dim])?.requires_grad_(true);
        Ok(Self {
            weight,
            num_embeddings,
            embedding_dim,
            mode,
        })
    }

    pub fn num_embeddings(&self) -> usize {
        self.num_embeddings
    }

    pub fn embedding_dim(&self) -> usize {
        self.embedding_dim
    }

    /// Forward pass with explicit offset indices for 1D jagged input tensors.
    pub fn forward_offsets(&self, indices: &Tensor, offsets: &[usize]) -> Result<Tensor> {
        if indices.ndim() != 1 {
            return Err(Error::Msg("forward_offsets expects 1D indices".into()));
        }
        let total = indices.numel();
        let num_bags = offsets.len();
        let mut bag_outs = Vec::with_capacity(num_bags);
        for i in 0..num_bags {
            let start = offsets[i];
            let end = if i + 1 < num_bags {
                offsets[i + 1]
            } else {
                total
            };
            if start > end {
                // Decreasing offsets: torch silently computes garbage here,
                // fail closed instead (deliberate divergence).
                return Err(Error::Msg(format!("invalid offset range [{start}, {end})")));
            }
            if start == end {
                // Torch parity: empty bags pool to zeros in every mode.
                let z = Tensor::zeros(
                    &[1, self.embedding_dim],
                    self.weight.dtype(),
                    &self.weight.device(),
                )?;
                z.set_requires_grad(self.weight.requires_grad())?;
                bag_outs.push(z);
                continue;
            }
            let len = end - start;
            let slice = indices.narrow(0, start, len)?;
            let emb = self.weight.index_select(0, &slice)?;
            let reduced = match self.mode {
                EmbeddingBagMode::Sum => emb.sum_axis(0, true)?,
                EmbeddingBagMode::Mean => emb.mean_axis(0, true)?,
                EmbeddingBagMode::Max => emb.max_axis(0, true)?,
            };
            bag_outs.push(reduced);
        }
        Tensor::cat(&bag_outs, 0)
    }
}

impl Module for EmbeddingBag {
    fn forward(&self, indices: &Tensor) -> Result<Tensor> {
        if indices.ndim() == 2 {
            let (b, n) = (indices.shape()[0], indices.shape()[1]);
            let flat = indices.reshape(&[indices.numel()])?;
            let selected = self.weight.index_select(0, &flat)?;
            let emb = selected.reshape(&[b, n, self.embedding_dim])?;
            match self.mode {
                EmbeddingBagMode::Sum => emb.sum_axis(1, false),
                EmbeddingBagMode::Mean => emb.mean_axis(1, false),
                EmbeddingBagMode::Max => emb.max_axis(1, false),
            }
        } else if indices.ndim() == 1 {
            let selected = self.weight.index_select(0, indices)?;
            match self.mode {
                EmbeddingBagMode::Sum => selected.sum_axis(0, true),
                EmbeddingBagMode::Mean => selected.mean_axis(0, true),
                EmbeddingBagMode::Max => selected.max_axis(0, true),
            }
        } else {
            Err(Error::Msg(
                "EmbeddingBag expects 1D or 2D input indices".into(),
            ))
        }
    }

    fn named_parameters(&self) -> Vec<(String, Tensor)> {
        vec![("weight".to_string(), self.weight.clone())]
    }
}

pub struct Dropout {
    p: f32,
    training: AtomicBool,
}

impl Dropout {
    pub fn new(p: f32) -> Self {
        // Clamp into [0, 1] like Dropout2d (torch parity: p == 1.0 zeroes
        // everything). NaN is not a probability; keep the loud failure.
        assert!(!p.is_nan(), "dropout probability must not be NaN");
        Self {
            p: p.clamp(0.0, 1.0),
            training: AtomicBool::new(true),
        }
    }
}

impl Module for Dropout {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        if !self.training.load(Ordering::Relaxed) || self.p == 0.0 {
            return Ok(input.clone());
        }
        if self.p == 1.0 {
            return input.zeros_like();
        }
        let keep = 1.0 - self.p;
        let n = input.numel();
        // CUDA: the mask comes from a kernel, so the activations never leave the device
        if matches!(input.device(), crate::device::Device::Cuda(_)) {
            let key = rng::with_rng(|r| r.next_u64());
            if let Some(mask) = crate::backend::Storage::cuda_bernoulli_mask(&input.device(), input.dtype(), n, key, f64::from(keep), 1.0 / f64::from(keep))? {
                return input.mul(&Tensor::from_storage(mask, crate::layout::Layout::contiguous(input.shape().to_vec())));
            }
        }
        let mask_data: Vec<f32> = rng::with_rng(|r| {
            (0..n)
                .map(|_| if r.next_f32() < keep { 1.0 / keep } else { 0.0 })
                .collect()
        });
        let mask = Tensor::from_slice_on(&mask_data, input.shape(), &input.device())?
            .to_dtype(input.dtype())?;
        input.mul(&mask)
    }

    fn train(&self, mode: bool) {
        self.training.store(mode, Ordering::Relaxed);
    }
}

/// Scaled dot-product Multi-Head Attention (`[batch, seq_len, embed_dim]`).
pub struct MultiHeadAttention {
    pub q_proj: Linear,
    pub k_proj: Linear,
    pub v_proj: Linear,
    pub out_proj: Linear,
    pub embed_dim: usize,
    pub num_heads: usize,
    pub head_dim: usize,
}

impl MultiHeadAttention {
    pub fn new(embed_dim: usize, num_heads: usize) -> Result<Self> {
        if num_heads == 0 || embed_dim % num_heads != 0 {
            return Err(crate::error::Error::Msg(format!(
                "embed_dim ({embed_dim}) must be divisible by a positive num_heads ({num_heads})"
            )));
        }
        Ok(Self {
            q_proj: Linear::new(embed_dim, embed_dim)?,
            k_proj: Linear::new(embed_dim, embed_dim)?,
            v_proj: Linear::new(embed_dim, embed_dim)?,
            out_proj: Linear::new(embed_dim, embed_dim)?,
            embed_dim,
            num_heads,
            head_dim: embed_dim / num_heads,
        })
    }

    pub fn forward_qkv(
        &self,
        q_in: &Tensor,
        k_in: &Tensor,
        v_in: &Tensor,
        causal_mask: bool,
    ) -> Result<Tensor> {
        for (name, t) in [("q", q_in), ("k", k_in), ("v", v_in)] {
            if t.ndim() != 3 {
                return Err(Error::Msg(format!(
                    "MultiHeadAttention::forward_qkv: expected [B, L, D] {name}, got {:?}",
                    t.shape()
                )));
            }
        }
        let (b, lq, _) = (q_in.shape()[0], q_in.shape()[1], q_in.shape()[2]);
        let lk = k_in.shape()[1];
        let h = self.num_heads;
        let d = self.head_dim;

        let q = self
            .q_proj
            .forward(q_in)?
            .reshape(&[b, lq, h, d])?
            .transpose(1, 2)?; // [b, h, lq, d]
        let k = self
            .k_proj
            .forward(k_in)?
            .reshape(&[b, lk, h, d])?
            .transpose(1, 2)?; // [b, h, lk, d]
        let v = self
            .v_proj
            .forward(v_in)?
            .reshape(&[b, lk, h, d])?
            .transpose(1, 2)?; // [b, h, lk, d]

        let scale = 1.0 / (d as f64).sqrt();
        let mut scores = q.matmul(&k.t()?)?.mul_scalar(scale)?; // [b, h, lq, lk]

        if causal_mask {
            let ones = Tensor::ones(&[lq, lk], DType::Bool, &scores.device())?;
            let upper = ones.triu(1)?;
            scores = scores.masked_fill(&upper, -1e9)?;
        }

        let attn = scores.softmax(-1)?;
        let ctx = attn
            .matmul(&v)?
            .transpose(1, 2)?
            .reshape(&[b, lq, self.embed_dim])?;
        self.out_proj.forward(&ctx)
    }
}

impl Module for MultiHeadAttention {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        self.forward_qkv(input, input, input, false)
    }

    fn named_parameters(&self) -> Vec<(String, Tensor)> {
        let mut out = Vec::new();
        for (prefix, m) in [
            ("q_proj", &self.q_proj),
            ("k_proj", &self.k_proj),
            ("v_proj", &self.v_proj),
            ("out_proj", &self.out_proj),
        ] {
            for (k, v) in m.named_parameters() {
                out.push((format!("{prefix}.{k}"), v));
            }
        }
        out
    }
}

// ---------------------------------------------------------------------------
// Activation Modules
// ---------------------------------------------------------------------------
macro_rules! activation_mod {
    ($name:ident, |$x:ident| $body:expr) => {
        #[derive(Clone, Copy, Debug, Default)]
        pub struct $name;

        impl Module for $name {
            fn forward(&self, $x: &Tensor) -> Result<Tensor> {
                $body
            }
        }
    };
}

activation_mod!(ReLU, |x| x.relu());
activation_mod!(GELU, |x| x.gelu());
activation_mod!(SiLU, |x| x.silu());
activation_mod!(Sigmoid, |x| x.sigmoid());
activation_mod!(Tanh, |x| x.tanh());

#[derive(Clone, Copy, Debug)]
pub struct LeakyReLU(pub f64);

impl Default for LeakyReLU {
    fn default() -> Self {
        Self(0.01)
    }
}

impl Module for LeakyReLU {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        input.leaky_relu(self.0)
    }
}

#[derive(Clone, Copy, Debug)]
pub struct ELU(pub f64);

impl Default for ELU {
    fn default() -> Self {
        Self(1.0)
    }
}

impl Module for ELU {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        input.elu(self.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn batchnorm1d_training_accepts_3d_ncl() -> Result<()> {
        let bn = BatchNorm1d::new(2)?;
        let x = Tensor::from_slice(&[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], &[2, 2, 2])?;
        let y = bn.forward(&x)?; // training mode normalizes over (N, L) per channel
        assert_eq!(y.shape(), &[2, 2, 2]);
        let mu = y.mean_axes(&[0, 2], false)?.to_vec::<f32>()?;
        assert!(mu.iter().all(|m| m.abs() < 1e-4), "{mu:?}");
        Ok(())
    }

    fn assert_close_f32(got: &[f32], want: &[f32], tol: f32, what: &str) {
        assert_eq!(got.len(), want.len(), "{what}: length");
        for (i, (&g, &w)) in got.iter().zip(want.iter()).enumerate() {
            assert!((g - w).abs() <= tol, "{what} element {i}: {g} vs {w}");
        }
    }

    #[test]
    fn batchnorm_running_var_uses_unbiased_estimate() -> Result<()> {
        // torch updates running_var with the Bessel-corrected batch variance
        // while normalizing with the biased one; all values torch-verified.
        let bn = BatchNorm1d::new(2)?;
        let x = Tensor::from_slice(&[1.0f32, 2.0, 3.0, 4.0], &[2, 2])?;
        let y = bn.forward(&x)?.to_vec::<f32>()?;
        assert_close_f32(
            &y,
            &[-0.999995, -0.999995, 0.999995, 0.999995],
            1e-5,
            "bn1d out",
        );
        assert_close_f32(
            &bn.running_mean.to_vec::<f32>()?,
            &[0.2, 0.3],
            1e-6,
            "bn1d mean",
        );
        assert_close_f32(
            &bn.running_var.to_vec::<f32>()?,
            &[1.1, 1.1],
            1e-6,
            "bn1d var",
        );
        // 3-D input corrects over N * L = 6 values per channel
        let bn3 = BatchNorm1d::new(2)?;
        let x3 = Tensor::from_slice(
            &[
                0.0f32, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0,
            ],
            &[2, 2, 3],
        )?;
        bn3.forward(&x3)?;
        assert_close_f32(
            &bn3.running_mean.to_vec::<f32>()?,
            &[0.4, 0.7],
            1e-6,
            "bn1d 3d mean",
        );
        assert_close_f32(
            &bn3.running_var.to_vec::<f32>()?,
            &[2.06, 2.06],
            1e-5,
            "bn1d 3d var",
        );
        // BatchNorm2d corrects over N * H * W = 4 values per channel
        let bn2 = BatchNorm2d::new(1)?;
        let x2 = Tensor::from_slice(&[1.0f32, 2.0, 3.0, 4.0], &[2, 1, 1, 2])?;
        bn2.forward(&x2)?;
        assert_close_f32(
            &bn2.running_var.to_vec::<f32>()?,
            &[1.0666666],
            1e-6,
            "bn2d var",
        );
        Ok(())
    }

    #[test]
    fn batchnorm_training_rejects_single_value_per_channel() -> Result<()> {
        // torch raises ("Expected more than 1 value per channel when training")
        let bn = BatchNorm1d::new(2)?;
        let x = Tensor::from_slice(&[1.0f32, 2.0], &[1, 2])?;
        assert!(bn.forward(&x).is_err());
        let bn2 = BatchNorm2d::new(1)?;
        let x2 = Tensor::from_slice(&[1.0f32], &[1, 1, 1, 1])?;
        assert!(bn2.forward(&x2).is_err());
        // N = 1 with L = 2 still has 2 values per channel: fine (torch-verified)
        let ok = Tensor::from_slice(&[1.0f32, 3.0, 2.0, 4.0], &[1, 2, 2])?;
        bn.forward(&ok)?;
        assert_close_f32(
            &bn.running_var.to_vec::<f32>()?,
            &[1.1, 1.1],
            1e-6,
            "bn1d n=1 var",
        );
        Ok(())
    }

    #[test]
    fn batchnorm1d_eval_broadcasts_running_stats_per_channel() -> Result<()> {
        // eval mode must align running stats per channel on 3-D input too
        // (values torch-verified with running_mean = [1, 2], running_var = [4, 9]).
        let bn = BatchNorm1d::new(2)?;
        bn.running_mean
            .assign(&Tensor::from_slice(&[1.0f32, 2.0], &[1, 2])?)?;
        bn.running_var
            .assign(&Tensor::from_slice(&[4.0f32, 9.0], &[1, 2])?)?;
        bn.train(false);
        let x = Tensor::from_slice(
            &[
                0.0f32, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0,
            ],
            &[2, 2, 3],
        )?;
        let y = bn.forward(&x)?.to_vec::<f32>()?;
        assert_close_f32(
            &y,
            &[
                -0.4999994, 0.0, 0.4999994, 0.3333331, 0.6666663, 0.9999994, 2.4999971, 2.9999964,
                3.4999957, 2.333_332, 2.666_665, 2.999_998,
            ],
            1e-5,
            "bn1d eval 3d",
        );
        // non-default affine locks per-channel alignment of weight/bias too
        // (torch-verified with weight = [2, 0.5], bias = [1, -1]).
        bn.weight
            .assign(&Tensor::from_slice(&[2.0f32, 0.5], &[1, 2])?)?;
        bn.bias
            .assign(&Tensor::from_slice(&[1.0f32, -1.0], &[1, 2])?)?;
        let ya = bn.forward(&x)?.to_vec::<f32>()?;
        assert_close_f32(
            &ya,
            &[
                0.0, 1.0, 1.9999988, -0.8333334, -0.6666669, -0.5000003, 5.9999943, 6.999_993,
                7.9999914, 0.166_666, 0.3333325, 0.4999991,
            ],
            1e-5,
            "bn1d eval 3d affine",
        );
        Ok(())
    }

    #[test]
    fn batchnorm_rejects_wrong_rank() -> Result<()> {
        // torch's BatchNorm1d/2d only accept their documented ranks
        let bn = BatchNorm1d::new(2)?;
        let bad4 = Tensor::from_slice(&[1.0f32; 8], &[1, 2, 2, 2])?;
        assert!(bn.forward(&bad4).is_err());
        bn.train(false);
        assert!(bn.forward(&bad4).is_err());
        let bn2 = BatchNorm2d::new(2)?;
        let bad3 = Tensor::from_slice(&[1.0f32; 8], &[1, 2, 4])?;
        assert!(bn2.forward(&bad3).is_err());
        let bad5 = Tensor::from_slice(&[1.0f32; 16], &[1, 2, 2, 2, 2])?;
        assert!(bn2.forward(&bad5).is_err());
        Ok(())
    }

    #[test]
    fn layernorm_groupnorm_rmsnorm_match_torch() -> Result<()> {
        // all forward values torch-verified, including the all-zero row (eps path)
        let ln = LayerNorm::new(3, 1e-5)?;
        let xl = Tensor::from_slice(&[1.0f32, 2.0, 3.0, 0.0, 0.0, 0.0], &[2, 3])?;
        let yl = ln.forward(&xl)?.to_vec::<f32>()?;
        assert_close_f32(
            &yl,
            &[-1.2247356, 0.0, 1.2247356, 0.0, 0.0, 0.0],
            1e-5,
            "ln",
        );
        let gn = GroupNorm::new(2, 4, 1e-5)?;
        let xg = Tensor::from_slice(&[0.0f32, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0], &[1, 4, 2])?;
        let yg = gn.forward(&xg)?.to_vec::<f32>()?;
        assert_close_f32(
            &yg,
            &[
                -1.3416355, -0.4472119, 0.4472117, 1.3416353, -1.3416352, -0.4472116, 0.447212,
                1.3416356,
            ],
            1e-5,
            "gn",
        );
        let rms = RMSNorm::new(3, 1e-6)?;
        let xr = Tensor::from_slice(&[1.0f32, 2.0, 3.0], &[1, 3])?;
        let yr = rms.forward(&xr)?.to_vec::<f32>()?;
        assert_close_f32(&yr, &[0.46291, 0.92582, 1.38873], 1e-5, "rms");
        // normalization backward stays finite even on the constant row
        let xg2 =
            Tensor::from_slice(&[1.0f32, 2.0, 3.0, 0.0, 0.0, 0.0], &[2, 3])?.requires_grad_(true);
        ln.forward(&xg2)?.sum()?.backward()?;
        assert!(
            xg2.grad()
                .unwrap()
                .to_vec::<f32>()?
                .iter()
                .all(|v| v.is_finite())
        );
        Ok(())
    }

    fn hand_embedding() -> Result<Embedding> {
        let emb = Embedding::new(4, 3)?;
        emb.weight.assign(&Tensor::from_slice(
            &[
                0.0f32, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0,
            ],
            &[4, 3],
        )?)?;
        Ok(emb)
    }

    #[test]
    fn embedding_forward_selects_rows_and_validates() -> Result<()> {
        let emb = hand_embedding()?;
        let idx = Tensor::from_slice(&[2i64, 0, 2], &[3])?;
        let y = emb.forward(&idx)?.to_vec::<f32>()?;
        assert_close_f32(
            &y,
            &[6.0, 7.0, 8.0, 0.0, 1.0, 2.0, 6.0, 7.0, 8.0],
            0.0,
            "emb",
        );
        let idx2 = Tensor::from_slice(&[1i64, 3], &[1, 2])?;
        assert_eq!(emb.forward(&idx2)?.shape(), &[1, 2, 3]);
        let oob = Tensor::from_slice(&[4i64], &[1])?;
        assert!(emb.forward(&oob).is_err());
        // gradients reach selected rows only (row 1 untouched)
        let idxg = Tensor::from_slice(&[2i64, 0, 2], &[3])?;
        emb.forward(&idxg)?.sum()?.backward()?;
        let g = emb.weight.grad().unwrap().to_vec::<f32>()?;
        assert_close_f32(
            &g,
            &[1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 2.0, 2.0, 2.0, 0.0, 0.0, 0.0],
            0.0,
            "emb grad",
        );
        Ok(())
    }

    #[test]
    fn embedding_bag_empty_bags_are_zeros() -> Result<()> {
        // torch pools empty bags to zeros in every mode (torch-verified).
        for (mode, want) in [
            (
                EmbeddingBagMode::Sum,
                vec![0.0, 0.0, 0.0, 3.0, 5.0, 7.0, 6.0, 7.0, 8.0],
            ),
            (
                EmbeddingBagMode::Mean,
                vec![0.0, 0.0, 0.0, 1.5, 2.5, 3.5, 6.0, 7.0, 8.0],
            ),
            (
                EmbeddingBagMode::Max,
                vec![0.0, 0.0, 0.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
            ),
        ] {
            let eb = EmbeddingBag::new(4, 3, mode)?;
            eb.weight.assign(&hand_embedding()?.weight)?;
            let idx = Tensor::from_slice(&[0i64, 1, 2], &[3])?;
            let y = eb.forward_offsets(&idx, &[0, 0, 2])?.to_vec::<f32>()?;
            assert_close_f32(&y, &want, 1e-6, "ebag empty");
        }
        // 2-D forward reduces each row-bag; decreasing offsets fail closed
        // (torch silently computes garbage there: deliberate divergence).
        let eb = EmbeddingBag::new(4, 3, EmbeddingBagMode::Sum)?;
        eb.weight.assign(&hand_embedding()?.weight)?;
        let idx2 = Tensor::from_slice(&[0i64, 1, 2, 3], &[2, 2])?;
        let y = eb.forward(&idx2)?.to_vec::<f32>()?;
        assert_close_f32(&y, &[3.0, 5.0, 7.0, 15.0, 17.0, 19.0], 0.0, "ebag 2d");
        let idx = Tensor::from_slice(&[0i64, 1, 2], &[3])?;
        assert!(eb.forward_offsets(&idx, &[0, 2, 1]).is_err());
        Ok(())
    }

    #[test]
    fn dropout_scales_and_gates_modes() -> Result<()> {
        // p = 0 and eval mode are the identity; p = 1 zeroes everything
        let x = Tensor::from_slice(&[1.0f32, 2.0, 3.0, 4.0], &[4])?;
        assert_close_f32(
            &Dropout::new(0.0).forward(&x)?.to_vec::<f32>()?,
            &[1.0, 2.0, 3.0, 4.0],
            0.0,
            "p0",
        );
        let d = Dropout::new(0.5);
        d.train(false);
        assert_close_f32(
            &d.forward(&x)?.to_vec::<f32>()?,
            &[1.0, 2.0, 3.0, 4.0],
            0.0,
            "eval",
        );
        assert_close_f32(
            &Dropout::new(1.0).forward(&x)?.to_vec::<f32>()?,
            &[0.0, 0.0, 0.0, 0.0],
            0.0,
            "p1",
        );
        // training p = 0.5: inverted scaling {0, 2x}, reproducible under seed
        crate::rng::manual_seed(7);
        let d = Dropout::new(0.5);
        let y1 = d.forward(&x)?.to_vec::<f32>()?;
        assert!(
            y1.iter()
                .zip([1.0, 2.0, 3.0, 4.0])
                .all(|(&g, v)| g == 0.0 || g == 2.0 * v)
        );
        assert!(y1.contains(&0.0) && y1.iter().any(|&g| g != 0.0));
        crate::rng::manual_seed(7);
        let y2 = d.forward(&x)?.to_vec::<f32>()?;
        assert_eq!(y1, y2);
        // backward uses the same mask as forward
        let xg = Tensor::from_slice(&[1.0f32, 1.0, 1.0, 1.0], &[4])?.requires_grad_(true);
        crate::rng::manual_seed(11);
        let yg = d.forward(&xg)?;
        yg.sum()?.backward()?;
        assert_eq!(xg.grad().unwrap().to_vec::<f32>()?, yg.to_vec::<f32>()?);
        Ok(())
    }

    #[test]
    #[should_panic(expected = "must not be NaN")]
    fn dropout_nan_probability_panics() {
        let _ = Dropout::new(f32::NAN);
    }
}
