//! Small layers that were missing: [`PadLayer`] (constant / reflection / replication / circular padding of the last
//! dims), [`GaussianNoise`] (training-time additive noise), [`ParameterList`], and [`ConvLstmCell`].

use super::functional_ext::{pad_mode, PadMode};
use super::{Conv2d, Module};
use crate::error::{Error, Result};
use crate::tensor::Tensor;
use std::sync::atomic::{AtomicBool, Ordering};

/// Pads the last `pads.len()` dimensions (`(before, after)` pairs starting from the last dimension).
pub struct PadLayer {
    pub pads: Vec<(usize, usize)>,
    pub mode: PadMode,
}

impl PadLayer {
    pub fn new(pads: Vec<(usize, usize)>, mode: PadMode) -> Self {
        Self { pads, mode }
    }
    /// Same padding `p` on all four sides of an NCHW tensor.
    pub fn square2d(p: usize, mode: PadMode) -> Self {
        Self::new(vec![(p, p), (p, p)], mode)
    }
}

impl Module for PadLayer {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        pad_mode(input, &self.pads, self.mode)
    }
}

/// Adds `N(0, std^2)` noise while training; identity in eval mode.
pub struct GaussianNoise {
    pub std: f64,
    training: AtomicBool,
}

impl GaussianNoise {
    pub fn new(std: f64) -> Self {
        Self { std, training: AtomicBool::new(true) }
    }
}

impl Module for GaussianNoise {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        if !self.training.load(Ordering::Relaxed) || self.std == 0.0 {
            return Ok(input.clone());
        }
        let noise = Tensor::randn_with(input.shape(), input.dtype(), &input.device())?.mul_scalar(self.std)?;
        input.add(&noise)
    }
    fn train(&self, mode: bool) {
        self.training.store(mode, Ordering::Relaxed);
    }
}

/// An indexable list of parameters that registers all of them (`torch.nn.ParameterList`).
#[derive(Default)]
pub struct ParameterList {
    items: Vec<Tensor>,
}

impl ParameterList {
    pub fn new(items: Vec<Tensor>) -> Self {
        Self { items: items.into_iter().map(|t| t.requires_grad_(true)).collect() }
    }
    pub fn push(&mut self, t: Tensor) {
        self.items.push(t.requires_grad_(true));
    }
    pub fn get(&self, i: usize) -> Option<&Tensor> {
        self.items.get(i)
    }
    pub fn len(&self) -> usize {
        self.items.len()
    }
    pub fn is_empty(&self) -> bool {
        self.items.is_empty()
    }
}

impl Module for ParameterList {
    /// Not a computation: returns the input unchanged.
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        Ok(input.clone())
    }
    fn named_parameters(&self) -> Vec<(String, Tensor)> {
        self.items.iter().enumerate().map(|(i, t)| (i.to_string(), t.clone())).collect()
    }
}

/// Convolutional LSTM cell: the gates are 2-D convolutions over `[x, h]` (`same` padding for odd kernels).
pub struct ConvLstmCell {
    pub conv: Conv2d,
    pub hidden: usize,
}

impl ConvLstmCell {
    pub fn new(in_channels: usize, hidden: usize, kernel: usize) -> Result<Self> {
        if kernel % 2 == 0 {
            return Err(Error::Msg("ConvLstmCell: kernel size must be odd".into()));
        }
        Ok(Self { conv: Conv2d::new(in_channels + hidden, 4 * hidden, (kernel, kernel), (1, 1), (kernel / 2, kernel / 2))?, hidden })
    }

    /// `x: [B, C, H, W]`, state `(h, c)` each `[B, hidden, H, W]` (zeros when `None`). Returns the new `(h, c)`.
    pub fn step(&self, x: &Tensor, state: Option<(&Tensor, &Tensor)>) -> Result<(Tensor, Tensor)> {
        let (b, h, w) = (x.shape()[0], x.shape()[2], x.shape()[3]);
        let zeros = Tensor::zeros(&[b, self.hidden, h, w], x.dtype(), &x.device())?;
        let (h0, c0) = state.unwrap_or((&zeros, &zeros));
        let gates = self.conv.forward(&Tensor::cat(&[x.clone(), h0.clone()], 1)?)?;
        let g = gates.chunk(4, 1)?;
        let (i, f, o, c_hat) = (g[0].sigmoid()?, g[1].sigmoid()?, g[2].sigmoid()?, g[3].tanh()?);
        let c = f.mul(c0)?.add(&i.mul(&c_hat)?)?;
        let hn = o.mul(&c.tanh()?)?;
        Ok((hn, c))
    }

    /// Runs `x: [B, T, C, H, W]` and returns all hidden states `[B, T, hidden, H, W]`.
    pub fn forward_sequence(&self, x: &Tensor) -> Result<Tensor> {
        let t = x.shape()[1];
        let mut state: Option<(Tensor, Tensor)> = None;
        let mut outs = Vec::with_capacity(t);
        for i in 0..t {
            let xi = x.narrow(1, i, 1)?.squeeze_axis(1)?;
            let (h, c) = self.step(&xi, state.as_ref().map(|(h, c)| (h, c)))?;
            outs.push(h.unsqueeze(1)?);
            state = Some((h, c));
        }
        Tensor::cat(&outs, 1)
    }

    pub fn named_parameters(&self) -> Vec<(String, Tensor)> {
        self.conv.named_parameters().into_iter().map(|(k, v)| (format!("conv.{k}"), v)).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pad_noise_paramlist() {
        let x = Tensor::from_slice(&[1.0f32, 2.0, 3.0, 4.0], &[1, 1, 2, 2]).unwrap();
        let p = PadLayer::square2d(1, PadMode::Replicate).forward(&x).unwrap();
        assert_eq!(p.shape(), [1, 1, 4, 4]);
        assert_eq!(p.to_vec::<f32>().unwrap()[..4], [1.0, 1.0, 2.0, 2.0]);
        let z = PadLayer::new(vec![(1, 0)], PadMode::Constant(9.0)).forward(&x).unwrap();
        assert_eq!(z.to_vec::<f32>().unwrap()[..3], [9.0, 1.0, 2.0]);
        let n = GaussianNoise::new(0.5);
        assert_ne!(n.forward(&x).unwrap().to_vec::<f32>().unwrap(), x.to_vec::<f32>().unwrap());
        n.eval();
        assert_eq!(n.forward(&x).unwrap().to_vec::<f32>().unwrap(), x.to_vec::<f32>().unwrap());
        let mut pl = ParameterList::new(vec![Tensor::randn(&[2]).unwrap()]);
        pl.push(Tensor::randn(&[3]).unwrap());
        assert_eq!((pl.len(), pl.num_parameters()), (2, 5));
        assert!(pl.parameters().iter().all(|p| p.requires_grad()));
    }

    #[test]
    fn conv_lstm_shapes_and_grads() {
        let cell = ConvLstmCell::new(2, 3, 3).unwrap();
        let x = Tensor::randn(&[2, 4, 2, 5, 5]).unwrap();
        let y = cell.forward_sequence(&x).unwrap();
        assert_eq!(y.shape(), [2, 4, 3, 5, 5]);
        assert!(y.to_vec::<f32>().unwrap().iter().all(|v| v.abs() < 1.0)); // h = o * tanh(c)
        y.sum().unwrap().backward().unwrap();
        assert!(cell.named_parameters().iter().all(|(_, p)| p.grad().is_some()));
        assert!(ConvLstmCell::new(2, 3, 2).is_err());
    }
}
