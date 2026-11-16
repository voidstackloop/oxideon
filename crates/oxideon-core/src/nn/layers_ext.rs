//! Layer utilities from `torch.nn` that were missing: [`embedding`] with `padding_idx` / `max_norm`,
//! [`fuse_conv_bn`] (folding an inference-mode batch norm into the preceding convolution), and [`ConvPadded`]
//! (convolutions with reflect / replicate / circular padding, `padding_mode` in PyTorch).

use super::functional_ext::{PadMode, pad_mode};
use super::{BatchNorm2d, Conv2d, Module};
use crate::error::{Error, Result};
use crate::tensor::{Tensor, no_grad};

/// Embedding lookup `[..] -> [.., dim]`. Rows addressed by `padding_idx` are looked up normally but receive no
/// gradient. With `max_norm = Some((max, p))` the referenced rows whose `p`-norm exceeds `max` are rescaled in
/// place first, as `torch.nn.functional.embedding` does.
pub fn embedding(weight: &Tensor, indices: &Tensor, padding_idx: Option<i64>, max_norm: Option<(f64, f64)>) -> Result<Tensor> {
    let rows = weight.shape()[0] as i64;
    let flat_idx = indices.to_vec::<i64>()?;
    if let Some(&bad) = flat_idx.iter().find(|&&i| i < 0 || i >= rows) {
        return Err(Error::Msg(format!("embedding: index {bad} is out of range for {rows} rows")));
    }
    if let Some((max, p)) = max_norm {
        no_grad(|| -> Result<()> {
            let mut uniq = flat_idx.clone();
            uniq.sort_unstable();
            uniq.dedup();
            let ids = Tensor::from_slice_on(&uniq, &[uniq.len()], &weight.device())?;
            let picked = weight.detach().index_select(0, &ids)?;
            let norms = picked.norm_p(p, &[1], true)?;
            let scale = Tensor::where_cond(
                &norms.gt(&norms.full_like(max)?)?,
                &norms.add_scalar(1e-7)?.recip()?.mul_scalar(max)?,
                &norms.ones_like()?,
            )?;
            let renormed = picked.mul(&scale)?;
            let new_w = weight.detach().index_copy(0, &ids, &renormed)?;
            weight.assign(&new_w)
        })?;
    }
    let ids = Tensor::from_slice_on(&flat_idx, &[flat_idx.len()], &weight.device())?;
    let looked = weight.index_select(0, &ids)?;
    let looked = match padding_idx {
        Some(pad) => {
            let mask: Vec<bool> = flat_idx.iter().map(|&i| i == pad).collect();
            let mask = Tensor::from_slice_on(&mask, &[mask.len(), 1], &weight.device())?.broadcast_to(looked.shape())?;
            Tensor::where_cond(&mask, &looked.detach(), &looked)?
        }
        None => looked,
    };
    let mut shape = indices.shape().to_vec();
    shape.push(weight.shape()[1]);
    looked.reshape(&shape)
}

/// Folds `bn` (using its running statistics) into `conv`, returning a convolution that computes `bn(conv(x))` in
/// inference mode.
pub fn fuse_conv_bn(conv: &Conv2d, bn: &BatchNorm2d) -> Result<Conv2d> {
    let out_c = conv.weight.shape()[0];
    let flat = |t: &Tensor| t.reshape(&[t.numel()]);
    let (gamma, beta) = (flat(&bn.weight)?, flat(&bn.bias)?);
    let (mean, var) = (flat(&bn.running_mean)?, flat(&bn.running_var)?);
    if gamma.numel() != out_c {
        return Err(Error::Msg(format!("fuse_conv_bn: the conv has {out_c} output channels, the batch norm {}", gamma.numel())));
    }
    let scale = gamma.div(&var.add_scalar(bn.eps)?.sqrt()?)?; // [out]
    let weight = conv.weight.detach().mul(&scale.reshape(&[out_c, 1, 1, 1])?)?;
    let base_bias = match &conv.bias {
        Some(b) => b.detach(),
        None => scale.zeros_like()?,
    };
    let bias = base_bias.sub(&mean)?.mul(&scale)?.add(&beta)?;
    Ok(Conv2d {
        weight: weight.requires_grad_(true),
        bias: Some(bias.requires_grad_(true)),
        stride: conv.stride,
        padding: conv.padding,
        dilation: conv.dilation,
        groups: conv.groups,
    })
}

/// A [`Conv2d`] whose zero padding is replaced by `mode` (reflect, replicate or circular).
pub struct ConvPadded {
    pub conv: Conv2d,
    pub mode: PadMode,
    pad: (usize, usize),
}

impl ConvPadded {
    /// `conv.padding` is moved into the explicit pre-padding and the inner convolution runs unpadded.
    pub fn new(mut conv: Conv2d, mode: PadMode) -> Self {
        let pad = conv.padding;
        conv.padding = (0, 0);
        Self { conv, mode, pad }
    }
}

impl Module for ConvPadded {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        let padded = pad_mode(input, &[(self.pad.1, self.pad.1), (self.pad.0, self.pad.0)], self.mode)?;
        self.conv.forward(&padded)
    }
    fn named_parameters(&self) -> Vec<(String, Tensor)> {
        self.conv.named_parameters()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn vals(t: &Tensor) -> Vec<f64> {
        t.to_vec::<f64>().unwrap()
    }

    #[test]
    fn embedding_padding_and_max_norm() {
        let w = Tensor::from_slice(&[3.0f64, 4.0, 0.3, 0.4, 1.0, 1.0], &[3, 2]).unwrap().requires_grad_(true);
        let ids = Tensor::from_slice(&[0i64, 1, 1, 2], &[2, 2]).unwrap();
        let out = embedding(&w, &ids, Some(1), None).unwrap();
        assert_eq!(out.shape(), [2, 2, 2]);
        out.sum().unwrap().backward().unwrap();
        // padding row 1 gets no gradient; rows 0 and 2 are used once
        assert_eq!(vals(&w.grad().unwrap()), vec![1.0, 1.0, 0.0, 0.0, 1.0, 1.0]);
        assert!(embedding(&w, &Tensor::from_slice(&[3i64], &[1]).unwrap(), None, None).is_err());
        // max_norm rescales only the over-long referenced row (row 0, norm 5 -> 1)
        let w2 = Tensor::from_slice(&[3.0f64, 4.0, 0.3, 0.4, 6.0, 8.0], &[3, 2]).unwrap();
        let o = embedding(&w2, &Tensor::from_slice(&[0i64, 1], &[2]).unwrap(), None, Some((1.0, 2.0))).unwrap();
        let o = vals(&o);
        assert!((o[0] - 0.6).abs() < 1e-6 && (o[1] - 0.8).abs() < 1e-6 && (o[2] - 0.3).abs() < 1e-9);
        assert_eq!(vals(&w2)[4], 6.0); // row 2 was not referenced
        assert!((vals(&w2)[0] - 0.6).abs() < 1e-6);
    }

    #[test]
    fn batch_norm_folds_into_the_convolution() {
        let conv = Conv2d::new(2, 4, (3, 3), (1, 1), (1, 1)).unwrap();
        let bn = BatchNorm2d::new(4).unwrap();
        bn.running_mean.assign(&Tensor::randn(&[1, 4, 1, 1]).unwrap()).unwrap();
        bn.running_var.assign(&Tensor::rand_uniform(&[1, 4, 1, 1], 0.5, 2.0).unwrap()).unwrap();
        bn.weight.assign(&Tensor::rand_uniform(&[1, 4, 1, 1], 0.5, 1.5).unwrap()).unwrap();
        bn.bias.assign(&Tensor::randn(&[1, 4, 1, 1]).unwrap()).unwrap();
        bn.eval();
        let x = Tensor::randn(&[2, 2, 6, 6]).unwrap();
        let want = bn.forward(&conv.forward(&x).unwrap()).unwrap();
        let fused = fuse_conv_bn(&conv, &bn).unwrap();
        let got = fused.forward(&x).unwrap();
        let (a, b) = (want.to_vec::<f32>().unwrap(), got.to_vec::<f32>().unwrap());
        assert!(a.iter().zip(&b).all(|(p, q)| (p - q).abs() < 1e-4));
    }

    #[test]
    fn padded_convolutions() {
        let weight = Tensor::from_slice(&[1.0f32 / 3.0; 3], &[1, 1, 1, 3]).unwrap();
        let make = || Conv2d { weight: weight.clone(), bias: None, stride: (1, 1), padding: (0, 1), dilation: (1, 1), groups: 1 };
        let x = Tensor::from_slice(&[1.0f32, 2.0, 3.0, 4.0], &[1, 1, 1, 4]).unwrap();
        let y = ConvPadded::new(make(), PadMode::Reflect).forward(&x).unwrap().to_vec::<f32>().unwrap();
        // padded row: 2 1 2 3 4 3
        assert!((y[0] - 5.0 / 3.0).abs() < 1e-6 && (y[3] - 10.0 / 3.0).abs() < 1e-6);
        let yc = ConvPadded::new(make(), PadMode::Circular).forward(&x).unwrap().to_vec::<f32>().unwrap();
        // padded row: 4 1 2 3 4 1
        assert!((yc[0] - 7.0 / 3.0).abs() < 1e-6 && (yc[3] - 8.0 / 3.0).abs() < 1e-6);
    }
}
