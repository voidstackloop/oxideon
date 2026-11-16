//! Coordinate-network and contrastive pieces: [`Siren`] (sinusoidal representation network), [`FourierFeatures`]
//! (NeRF-style positional encoding and random Fourier features), [`LayerScale`], and the SimCLR [`nt_xent_loss`].

use super::Module;
use crate::error::{Error, Result};
use crate::tensor::Tensor;

fn uniform_param(shape: &[usize], bound: f64) -> Result<Tensor> {
    Ok(Tensor::rand_uniform(shape, -bound as f32, bound as f32)?.requires_grad_(true))
}

/// SIREN: `sin(w0 * (W x + b))` hidden layers with the initialization of Sitzmann et al. and a linear output.
pub struct Siren {
    weights: Vec<Tensor>,
    biases: Vec<Tensor>,
    pub w0: f64,
}

impl Siren {
    /// `dims` = `[in, hidden..., out]` (at least two entries).
    pub fn new(dims: &[usize], w0: f64) -> Result<Self> {
        if dims.len() < 2 || dims.contains(&0) || w0 <= 0.0 {
            return Err(Error::Msg("Siren: need dims [in, ..., out] with positive sizes and w0 > 0".into()));
        }
        let (mut weights, mut biases) = (Vec::new(), Vec::new());
        for (i, w) in dims.windows(2).enumerate() {
            let bound = if i == 0 { 1.0 / w[0] as f64 } else { (6.0 / w[0] as f64).sqrt() / w0 };
            weights.push(uniform_param(&[w[0], w[1]], bound)?);
            biases.push(uniform_param(&[w[1]], 1.0 / (w[0] as f64).sqrt())?);
        }
        Ok(Self { weights, biases, w0 })
    }
}

impl Module for Siren {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        let last = self.weights.len() - 1;
        let mut x = input.clone();
        for (i, (w, b)) in self.weights.iter().zip(&self.biases).enumerate() {
            x = x.matmul(w)?.add(b)?;
            if i < last {
                x = x.mul_scalar(self.w0)?.sin()?;
            }
        }
        Ok(x)
    }

    fn named_parameters(&self) -> Vec<(String, Tensor)> {
        self.weights.iter().zip(&self.biases).enumerate().flat_map(|(i, (w, b))| [(format!("layers.{i}.weight"), w.clone()), (format!("layers.{i}.bias"), b.clone())]).collect()
    }
}

/// Maps `[N, D]` coordinates to sinusoidal features. `positional(L)` gives `[x, sin(2^k pi x), cos(2^k pi x)]` for
/// `k < L`; `gaussian(D, m, sigma)` gives random Fourier features `[sin(2 pi B x), cos(2 pi B x)]` with a fixed
/// Gaussian matrix `B` (`D x m`, scale `sigma`).
pub struct FourierFeatures {
    freqs: Tensor, // [D, F]
    include_input: bool,
}

impl FourierFeatures {
    pub fn positional(dim: usize, levels: usize) -> Result<Self> {
        if dim == 0 || levels == 0 {
            return Err(Error::Msg("FourierFeatures: dim and levels must be positive".into()));
        }
        // frequency columns 2^k * pi for every input dim, laid out so column d*levels + k belongs to input d
        let mut data = vec![0.0f32; dim * dim * levels];
        for d in 0..dim {
            for k in 0..levels {
                data[d * (dim * levels) + d * levels + k] = (2f64.powi(k as i32) * std::f64::consts::PI) as f32;
            }
        }
        Ok(Self { freqs: Tensor::from_slice(&data, &[dim, dim * levels])?, include_input: true })
    }

    pub fn gaussian(dim: usize, features: usize, sigma: f64) -> Result<Self> {
        if dim == 0 || features == 0 || sigma <= 0.0 {
            return Err(Error::Msg("FourierFeatures: dim, features and sigma must be positive".into()));
        }
        Ok(Self { freqs: Tensor::randn(&[dim, features])?.mul_scalar(2.0 * std::f64::consts::PI * sigma)?, include_input: false })
    }

    pub fn output_dim(&self, dim: usize) -> usize {
        2 * self.freqs.shape()[1] + if self.include_input { dim } else { 0 }
    }
}

impl Module for FourierFeatures {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        let proj = input.matmul(&self.freqs.to_dtype(input.dtype())?)?;
        let mut parts = Vec::new();
        if self.include_input {
            parts.push(input.clone());
        }
        parts.push(proj.sin()?);
        parts.push(proj.cos()?);
        Tensor::cat(&parts, -1)
    }
}

/// Learnable per-channel scale (CaiT / ConvNeXt `LayerScale`), initialized to a small constant.
pub struct LayerScale {
    pub gamma: Tensor,
}

impl LayerScale {
    pub fn new(channels: usize, init: f64) -> Result<Self> {
        Ok(Self { gamma: Tensor::full(&[channels], init, crate::dtype::DType::F32, &crate::backend::Device::Cpu)?.requires_grad_(true) })
    }
}

impl Module for LayerScale {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        input.mul(&self.gamma.to_dtype(input.dtype())?)
    }
    fn named_parameters(&self) -> Vec<(String, Tensor)> {
        vec![("gamma".to_string(), self.gamma.clone())]
    }
}

/// SimCLR NT-Xent loss for two augmented views `z1`, `z2` (`[N, D]`): each embedding must pick its partner out of the
/// other `2N - 2` embeddings.
pub fn nt_xent_loss(z1: &Tensor, z2: &Tensor, temperature: f64) -> Result<Tensor> {
    if z1.shape() != z2.shape() || z1.ndim() != 2 || z1.shape()[0] < 2 || temperature <= 0.0 {
        return Err(Error::Msg("nt_xent_loss: need two [N >= 2, D] views and a positive temperature".into()));
    }
    let n = z1.shape()[0];
    let z = Tensor::cat(&[z1.clone(), z2.clone()], 0)?;
    let norm = z.square()?.sum_axis(1, true)?.sqrt()?.add_scalar(1e-12)?;
    let z = z.div(&norm)?;
    let sim = z.matmul(&z.t()?)?.div_scalar(temperature)?;
    // mask the diagonal (self-similarity) with a large negative value
    let eye = Tensor::eye(2 * n, crate::dtype::DType::F32, &z.device())?.gt(&Tensor::zeros(&[2 * n, 2 * n], crate::dtype::DType::F32, &z.device())?)?;
    let logits = sim.masked_fill(&eye, -1e9)?;
    let targets: Vec<usize> = (0..2 * n).map(|i| (i + n) % (2 * n)).collect();
    super::loss::cross_entropy(&logits, &targets)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn siren_fits_a_high_frequency_signal() {
        let net = Siren::new(&[1, 32, 32, 1], 30.0).unwrap();
        let xs: Vec<f32> = (0..64).map(|i| i as f32 / 63.0 * 2.0 - 1.0).collect();
        let x = Tensor::from_slice(&xs, &[64, 1]).unwrap();
        let y = Tensor::from_slice(&xs.iter().map(|v| (12.0 * v).sin()).collect::<Vec<_>>(), &[64, 1]).unwrap();
        let mut opt = crate::optim::Adam::new(net.parameters(), 1e-3);
        use crate::optim::Optimizer;
        let first = net.forward(&x).unwrap().sub(&y).unwrap().square().unwrap().mean().unwrap().item::<f32>().unwrap();
        for _ in 0..300 {
            opt.zero_grad();
            net.forward(&x).unwrap().sub(&y).unwrap().square().unwrap().mean().unwrap().backward().unwrap();
            opt.step().unwrap();
        }
        let last = net.forward(&x).unwrap().sub(&y).unwrap().square().unwrap().mean().unwrap().item::<f32>().unwrap();
        assert!(last < first * 0.2 && last < 0.1, "{first} -> {last}");
        assert_eq!(net.named_parameters().len(), 6);
        assert!(Siren::new(&[3], 30.0).is_err());
    }

    #[test]
    fn fourier_features_shapes_and_values() {
        let pe = FourierFeatures::positional(2, 3).unwrap();
        let x = Tensor::from_slice(&[0.25f32, 0.5], &[1, 2]).unwrap();
        let y = pe.forward(&x).unwrap();
        assert_eq!(y.shape(), [1, pe.output_dim(2)]);
        assert_eq!(pe.output_dim(2), 2 + 2 * 6);
        let v = y.to_vec::<f32>().unwrap();
        assert_eq!(&v[..2], &[0.25, 0.5]);
        // first sine feature is sin(pi * x0)
        assert!((v[2] - (std::f32::consts::PI * 0.25).sin()).abs() < 1e-5, "{v:?}");
        let rff = FourierFeatures::gaussian(2, 16, 1.0).unwrap();
        assert_eq!(rff.forward(&x).unwrap().shape(), [1, 32]);
        assert!(FourierFeatures::positional(0, 3).is_err());
    }

    #[test]
    fn layer_scale_and_nt_xent() {
        let ls = LayerScale::new(3, 0.1).unwrap();
        let out = ls.forward(&Tensor::from_slice(&[1.0f32, 2.0, 3.0], &[1, 3]).unwrap()).unwrap().to_vec::<f32>().unwrap();
        assert!(out.iter().zip([0.1, 0.2, 0.3]).all(|(a, b)| (a - b).abs() < 1e-6));
        assert_eq!(ls.num_parameters(), 3);
        // aligned views give a lower loss than unrelated ones, and gradients flow
        let a = Tensor::randn(&[8, 16]).unwrap().requires_grad_(true);
        let noise = Tensor::randn(&[8, 16]).unwrap().mul_scalar(0.05).unwrap();
        let aligned = nt_xent_loss(&a, &a.add(&noise).unwrap(), 0.5).unwrap();
        let random = nt_xent_loss(&a, &Tensor::randn(&[8, 16]).unwrap(), 0.5).unwrap();
        assert!(aligned.item::<f32>().unwrap() < random.item::<f32>().unwrap());
        aligned.backward().unwrap();
        assert!(a.grad().is_some());
        assert!(nt_xent_loss(&a, &Tensor::randn(&[4, 16]).unwrap(), 0.5).is_err());
    }
}
