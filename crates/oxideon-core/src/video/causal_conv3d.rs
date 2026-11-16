//! Causal 3D Volumetric Convolutions for Video Modeling.
//!
//! Enforces strict temporal causality: a frame at time step $t$ can only attend to
//! and receive receptive field activations from time steps $\le t$, with zero future leakage.

use crate::error::{Error, Result};
use crate::nn::{Module, init};
use crate::tensor::Tensor;

/// Causal 3D convolution layer over 5D video tensors `[N, C_in, T, H, W]`.
///
/// Applies causal asymmetric padding along the temporal dimension ($T_{\text{pad\_left}} = k_t - 1$,
/// $T_{\text{pad\_right}} = 0$) and symmetric padding along the spatial dimensions ($H, W$).
pub struct CausalConv3d {
    pub weight: Tensor,
    pub bias: Option<Tensor>,
    pub in_channels: usize,
    pub out_channels: usize,
    pub kernel_size: (usize, usize, usize),
    pub stride: (usize, usize, usize),
}

impl CausalConv3d {
    pub fn new(
        in_channels: usize,
        out_channels: usize,
        kernel_size: (usize, usize, usize),
        stride: (usize, usize, usize),
    ) -> Result<Self> {
        let (kt, kh, kw) = kernel_size;
        let fan = in_channels
            .checked_mul(kt)
            .and_then(|v| v.checked_mul(kh))
            .and_then(|v| v.checked_mul(kw))
            .ok_or_else(|| {
                Error::Msg(format!(
                    "CausalConv3d: fan (in_channels {in_channels} * kernel {kt}x{kh}x{kw}) overflows"
                ))
            })?;
        if fan == 0 {
            return Err(Error::Msg(
                "CausalConv3d: fan (in_channels * kernel) must be positive".into(),
            ));
        }
        let k = (1.0 / (fan as f64).sqrt()) as f32;
        let weight = init::uniform(&[out_channels, in_channels, kt, kh, kw], -k, k)?;
        let bias = Some(init::uniform(&[out_channels], -k, k)?);

        Ok(Self {
            weight,
            bias,
            in_channels,
            out_channels,
            kernel_size,
            stride,
        })
    }

    /// Temporal causal pad: adds `kt - 1` frames of zeros to the beginning of the time axis.
    fn pad_temporal_causal(&self, input: &Tensor) -> Result<Tensor> {
        let (kt, _, _) = self.kernel_size;
        let pad_t = kt.saturating_sub(1);
        if pad_t == 0 {
            return Ok(input.clone());
        }

        let shape = input.shape();
        let (n, c, pad_t, h, w) = (shape[0], shape[1], pad_t, shape[3], shape[4]);
        let pad_zeros = Tensor::zeros(&[n, c, pad_t, h, w], input.dtype(), &input.device())?;

        Tensor::cat(&[pad_zeros, input.clone()], 2)
    }
}

impl Module for CausalConv3d {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        if input.ndim() != 5 {
            return Err(Error::Msg(format!(
                "CausalConv3d expected 5D tensor [N, C, T, H, W], got ndim {}",
                input.ndim()
            )));
        }

        let padded_input = self.pad_temporal_causal(input)?;
        let (_, kh, kw) = self.kernel_size;
        let spatial_pad = (kh / 2, kw / 2);

        // Apply 3D convolution with zero temporal padding (already causally padded)
        // and symmetric spatial padding
        padded_input.conv3d(
            &self.weight,
            self.bias.as_ref(),
            self.stride,
            (0, spatial_pad.0, spatial_pad.1),
        )
    }
}

impl CausalConv3d {
    pub fn named_parameters(&self) -> Vec<(String, Tensor)> {
        let mut out = vec![("weight".to_string(), self.weight.clone())];
        if let Some(b) = &self.bias {
            out.push(("bias".to_string(), b.clone()));
        }
        out
    }
}
