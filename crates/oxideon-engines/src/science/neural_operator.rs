//! Continuous Neural Operators (Fourier Neural Operator / FNO) for Infinite-Dimensional PDEs.
//!
//! Implements resolution-invariant operator learning between Banach spaces:
//! - [`SpectralConv1d`]: Fourier space integral kernel $\mathcal{K}(v)(x) = \mathcal{F}^{-1}(R \cdot \mathcal{F}(v))(x)$
//!   with parameterized complex weights over truncated Fourier modes.
//! - [`FourierNeuralOperator1d`]: End-to-end continuous operator network with lifting,
//!   iterated spectral convolution + local skip layers, and target projection.
//! - Discretization-invariance: trained on coarse spatial grids and zero-shot evaluated
//!   at arbitrarily fine resolutions without changing weights.

use crate::error::{Error, Result};
use crate::nn::{Linear, Module};
use crate::tensor::Tensor;

/// 1D Spectral Convolution layer operating in Fourier domain.
#[derive(Debug, Clone)]
pub struct SpectralConv1d {
    pub in_channels: usize,
    pub out_channels: usize,
    pub modes: usize,
    pub weights_real: Tensor,
    pub weights_imag: Tensor,
}

impl SpectralConv1d {
    /// Creates a new `SpectralConv1d` with truncated mode count `modes`.
    pub fn new(in_channels: usize, out_channels: usize, modes: usize) -> Result<Self> {
        crate::limits::check_elements(
            "SpectralConv1d weights",
            &[in_channels, out_channels, modes],
        )?;
        let scale = (2.0 / (in_channels as f64 + out_channels as f64)).sqrt();

        let wr = Tensor::randn(&[in_channels, out_channels, modes])?.mul_scalar(scale)?;
        let wi = Tensor::randn(&[in_channels, out_channels, modes])?.mul_scalar(scale)?;

        Ok(Self {
            in_channels,
            out_channels,
            modes,
            weights_real: wr,
            weights_imag: wi,
        })
    }

    /// Forward pass:
    /// Input `x` of shape `[Batch, in_channels, N]`.
    /// Returns `[Batch, out_channels, N]`.
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        if x.ndim() != 3 {
            return Err(Error::Msg(format!(
                "SpectralConv1d expects 3D tensor [Batch, InChannels, N], got {:?}",
                x.shape()
            )));
        }
        let batch = x.shape()[0];
        let in_c = x.shape()[1];
        let n = x.shape()[2];

        if in_c != self.in_channels {
            return Err(Error::Msg(format!(
                "Input channels {} != layer in_channels {}",
                in_c, self.in_channels
            )));
        }

        // 1. FFT along spatial dimension: [Batch, InChannels, N] -> [Batch, InChannels, N, 2]
        let x_fft = x.rfft()?;
        let x_fft_vec = x_fft.to_vec::<f64>()?;

        let wr_vec = self.weights_real.to_vec::<f64>()?;
        let wi_vec = self.weights_imag.to_vec::<f64>()?;

        // 2. Frequency-domain complex multiplication over lowest `modes`
        let modes_to_keep = self.modes.min(n);
        let mut out_fft = vec![0.0f64; batch * self.out_channels * n * 2];

        for b in 0..batch {
            for m in 0..modes_to_keep {
                for oc in 0..self.out_channels {
                    let mut sum_r = 0.0f64;
                    let mut sum_i = 0.0f64;

                    for ic in 0..self.in_channels {
                        let in_idx = ((b * in_c + ic) * n + m) * 2;
                        let r_in = x_fft_vec[in_idx];
                        let i_in = x_fft_vec[in_idx + 1];

                        let w_idx = (ic * self.out_channels + oc) * self.modes + m;
                        let r_w = wr_vec[w_idx];
                        let i_w = wi_vec[w_idx];

                        // (r_in + i * i_in) * (r_w + i * i_w)
                        sum_r += r_in * r_w - i_in * i_w;
                        sum_i += r_in * i_w + i_in * r_w;
                    }

                    let out_idx = ((b * self.out_channels + oc) * n + m) * 2;
                    out_fft[out_idx] = sum_r;
                    out_fft[out_idx + 1] = sum_i;
                }
            }
        }

        let out_fft_tensor =
            Tensor::from_vec(out_fft, &[batch, self.out_channels, n, 2])?.to_dtype(x.dtype())?;

        // 3. Inverse FFT back to spatial domain
        out_fft_tensor.irfft()
    }
}

/// Continuous 1D Fourier Neural Operator (FNO) network.
#[derive(Clone)]
pub struct FourierNeuralOperator1d {
    pub in_channels: usize,
    pub out_channels: usize,
    pub hidden_channels: usize,
    pub modes: usize,
    pub lifting: Linear,
    pub spectral_convs: Vec<SpectralConv1d>,
    pub local_skips: Vec<Linear>,
    pub projection: Linear,
}

impl FourierNeuralOperator1d {
    /// Creates a 1D FNO with `num_layers` spectral convolution layers.
    pub fn new(
        in_channels: usize,
        out_channels: usize,
        hidden_channels: usize,
        modes: usize,
        num_layers: usize,
    ) -> Result<Self> {
        crate::limits::check_repeats("num_layers", num_layers)?;
        let lifting = Linear::new(in_channels, hidden_channels)?;
        let mut spectral_convs = Vec::with_capacity(num_layers);
        let mut local_skips = Vec::with_capacity(num_layers);

        for _ in 0..num_layers {
            spectral_convs.push(SpectralConv1d::new(
                hidden_channels,
                hidden_channels,
                modes,
            )?);
            local_skips.push(Linear::new(hidden_channels, hidden_channels)?);
        }

        let projection = Linear::new(hidden_channels, out_channels)?;

        Ok(Self {
            in_channels,
            out_channels,
            hidden_channels,
            modes,
            lifting,
            spectral_convs,
            local_skips,
            projection,
        })
    }

    /// Discretization-invariant continuous forward pass:
    /// Input `x`: `[Batch, in_channels, N]`.
    /// Returns: `[Batch, out_channels, N]`.
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        if x.ndim() != 3 {
            return Err(Error::Msg(format!(
                "FNO expects 3D tensor [Batch, InChannels, N], got {:?}",
                x.shape()
            )));
        }

        // 1. Lifting: [Batch, N, InChannels] -> [Batch, N, HiddenChannels]
        let x_perm = x.permute(&[0, 2, 1])?; // [B, N, C_in]
        let x_lifted = self.lifting.forward(&x_perm)?; // [B, N, C_hidden]
        let mut h = x_lifted.permute(&[0, 2, 1])?; // [B, C_hidden, N]

        // 2. Iterated Spectral Convolution + Local Skip + Non-linearity
        for (conv, skip) in self.spectral_convs.iter().zip(self.local_skips.iter()) {
            let h_spectral = conv.forward(&h)?;

            let h_perm = h.permute(&[0, 2, 1])?;
            let h_skip_perm = skip.forward(&h_perm)?;
            let h_skip = h_skip_perm.permute(&[0, 2, 1])?;

            h = h_spectral.add(&h_skip)?.gelu()?;
        }

        // 3. Projection: [B, C_hidden, N] -> [B, N, C_out] -> [B, C_out, N]
        let h_perm = h.permute(&[0, 2, 1])?;
        let out_perm = self.projection.forward(&h_perm)?;
        out_perm.permute(&[0, 2, 1])
    }
}
