//! Image and set-valued losses and metrics: [`dice_loss`], [`tversky_loss`], [`iou_loss`], [`ssim`], [`psnr`],
//! [`total_variation`], [`chamfer_distance`] and the 1-D [`wasserstein1d`] distance. All are differentiable
//! compositions of tensor ops. Image tensors are `[N, C, H, W]`.

use crate::error::{Error, Result};
use crate::nn::functional_ext::{PadMode, pad_mode};
use crate::tensor::Tensor;

fn same_shape(a: &Tensor, b: &Tensor, op: &str) -> Result<()> {
    if a.shape() != b.shape() {
        return Err(Error::Msg(format!("{op}: shapes {:?} and {:?} differ", a.shape(), b.shape())));
    }
    Ok(())
}

/// Sums over every dimension except the first (the batch).
fn per_sample(t: &Tensor) -> Result<Tensor> {
    let n = t.shape()[0];
    t.reshape(&[n, t.numel() / n.max(1)])?.sum_axis(1, false)
}

/// Soft Dice loss `1 - (2 |P ∩ T| + smooth) / (|P| + |T| + smooth)` averaged over the batch; `probs` in `[0, 1]`.
pub fn dice_loss(probs: &Tensor, target: &Tensor, smooth: f64) -> Result<Tensor> {
    same_shape(probs, target, "dice_loss")?;
    let inter = per_sample(&probs.mul(target)?)?;
    let denom = per_sample(probs)?.add(&per_sample(target)?)?;
    inter.mul_scalar(2.0)?.add_scalar(smooth)?.div(&denom.add_scalar(smooth)?)?.neg()?.add_scalar(1.0)?.mean()
}

/// Tversky loss: Dice generalized with weights `alpha` (false positives) and `beta` (false negatives).
pub fn tversky_loss(probs: &Tensor, target: &Tensor, alpha: f64, beta: f64, smooth: f64) -> Result<Tensor> {
    same_shape(probs, target, "tversky_loss")?;
    let tp = per_sample(&probs.mul(target)?)?;
    let fp = per_sample(&probs.mul(&target.neg()?.add_scalar(1.0)?)?)?;
    let fn_ = per_sample(&probs.neg()?.add_scalar(1.0)?.mul(target)?)?;
    let denom = tp.add(&fp.mul_scalar(alpha)?)?.add(&fn_.mul_scalar(beta)?)?.add_scalar(smooth)?;
    tp.add_scalar(smooth)?.div(&denom)?.neg()?.add_scalar(1.0)?.mean()
}

/// Soft intersection-over-union loss `1 - |P ∩ T| / |P ∪ T|`.
pub fn iou_loss(probs: &Tensor, target: &Tensor, smooth: f64) -> Result<Tensor> {
    same_shape(probs, target, "iou_loss")?;
    let inter = per_sample(&probs.mul(target)?)?;
    let union = per_sample(probs)?.add(&per_sample(target)?)?.sub(&inter)?;
    inter.add_scalar(smooth)?.div(&union.add_scalar(smooth)?)?.neg()?.add_scalar(1.0)?.mean()
}

fn gaussian_window(size: usize, sigma: f64, channels: usize, like: &Tensor) -> Result<Tensor> {
    let r = (size / 2) as isize;
    let g: Vec<f64> = (-r..=r).map(|i| (-(i * i) as f64 / (2.0 * sigma * sigma)).exp()).collect();
    let s: f64 = g.iter().sum();
    let mut w = vec![0.0; channels * size * size];
    for c in 0..channels {
        for i in 0..size {
            for j in 0..size {
                w[(c * size + i) * size + j] = g[i] * g[j] / (s * s);
            }
        }
    }
    Tensor::from_slice_on(&w, &[channels, 1, size, size], &like.device())?.to_dtype(like.dtype())
}

/// Structural similarity index (mean over the image) with a Gaussian window (`size` odd, `sigma`) and dynamic
/// range `data_range`; returns the batch mean.
pub fn ssim(x: &Tensor, y: &Tensor, data_range: f64, size: usize, sigma: f64) -> Result<Tensor> {
    same_shape(x, y, "ssim")?;
    if x.ndim() != 4 || size % 2 == 0 {
        return Err(Error::Msg("ssim: expects [N, C, H, W] images and an odd window size".into()));
    }
    let c = x.shape()[1];
    let (h, w) = (x.shape()[2], x.shape()[3]);
    if h < size || w < size {
        return Err(Error::Msg("ssim: the image is smaller than the window".into()));
    }
    let win = gaussian_window(size, sigma, c, x)?;
    let blur = |t: &Tensor| t.conv2d_grouped(&win, None, (1, 1), (0, 0), (1, 1), c);
    let (mu_x, mu_y) = (blur(x)?, blur(y)?);
    let (mu_xx, mu_yy, mu_xy) = (mu_x.square()?, mu_y.square()?, mu_x.mul(&mu_y)?);
    let sigma_x = blur(&x.square()?)?.sub(&mu_xx)?;
    let sigma_y = blur(&y.square()?)?.sub(&mu_yy)?;
    let sigma_xy = blur(&x.mul(y)?)?.sub(&mu_xy)?;
    let (c1, c2) = ((0.01 * data_range).powi(2), (0.03 * data_range).powi(2));
    let num = mu_xy.mul_scalar(2.0)?.add_scalar(c1)?.mul(&sigma_xy.mul_scalar(2.0)?.add_scalar(c2)?)?;
    let den = mu_xx.add(&mu_yy)?.add_scalar(c1)?.mul(&sigma_x.add(&sigma_y)?.add_scalar(c2)?)?;
    num.div(&den)?.mean()
}

/// Peak signal-to-noise ratio in dB: `10 log10(range^2 / MSE)`.
pub fn psnr(x: &Tensor, y: &Tensor, data_range: f64) -> Result<Tensor> {
    same_shape(x, y, "psnr")?;
    let mse = x.sub(y)?.square()?.mean()?.clamp(1e-12, f64::INFINITY)?;
    mse.recip()?.mul_scalar(data_range * data_range)?.ln()?.mul_scalar(10.0 / std::f64::consts::LN_10)
}

/// Anisotropic total variation of `[N, C, H, W]` images (sum of absolute neighbour differences, batch mean).
pub fn total_variation(x: &Tensor) -> Result<Tensor> {
    if x.ndim() != 4 {
        return Err(Error::Msg("total_variation: expects [N, C, H, W]".into()));
    }
    let (h, w) = (x.shape()[2], x.shape()[3]);
    let dv = x.narrow(2, 1, h.saturating_sub(1))?.sub(&x.narrow(2, 0, h.saturating_sub(1))?)?.abs()?;
    let dh = x.narrow(3, 1, w.saturating_sub(1))?.sub(&x.narrow(3, 0, w.saturating_sub(1))?)?.abs()?;
    per_sample(&dv)?.add(&per_sample(&dh)?)?.mean()
}

/// Chamfer distance between point sets `[N, P, D]` and `[N, Q, D]`: the mean squared distance from each point to its
/// nearest neighbour in the other set, in both directions.
pub fn chamfer_distance(a: &Tensor, b: &Tensor) -> Result<Tensor> {
    if a.ndim() != 3 || b.ndim() != 3 || a.shape()[0] != b.shape()[0] || a.shape()[2] != b.shape()[2] {
        return Err(Error::Msg("chamfer_distance: expects [N, P, D] and [N, Q, D]".into()));
    }
    let d = a.unsqueeze(2)?.sub(&b.unsqueeze(1)?)?.square()?.sum_axis(3, false)?; // [N, P, Q]
    d.min_axis(2, false)?.mean()?.add(&d.min_axis(1, false)?.mean()?)
}

/// 1-Wasserstein distance between two equally weighted empirical distributions of `n` samples each (the mean absolute
/// difference of the sorted samples), per row of `[B, n]` inputs, averaged over the batch.
pub fn wasserstein1d(a: &Tensor, b: &Tensor) -> Result<Tensor> {
    same_shape(a, b, "wasserstein1d")?;
    let axis = a.ndim() as isize - 1;
    let (sa, _) = a.sort(axis, false)?;
    let (sb, _) = b.sort(axis, false)?;
    sa.sub(&sb)?.abs()?.mean()
}

/// Reflect-pads an image batch by `p` and applies [`ssim`]-style smoothing helpers elsewhere; exposed for reuse.
pub fn reflect_pad2d(x: &Tensor, p: usize) -> Result<Tensor> {
    pad_mode(x, &[(p, p), (p, p)], PadMode::Reflect)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn v(t: &Tensor) -> f64 {
        t.item::<f64>().unwrap()
    }
    fn img(n: usize, h: usize, w: usize, seed: f64) -> Tensor {
        let d: Vec<f64> = (0..n * h * w).map(|i| (i as f64 * 0.37 + seed).sin() * 0.5 + 0.5).collect();
        Tensor::from_slice(&d, &[n, 1, h, w]).unwrap()
    }

    #[test]
    fn overlap_losses() {
        let t = Tensor::from_slice(&[1.0f64, 1.0, 0.0, 0.0], &[1, 4]).unwrap();
        assert!(v(&dice_loss(&t, &t, 0.0).unwrap()).abs() < 1e-12);
        assert!(v(&iou_loss(&t, &t, 0.0).unwrap()).abs() < 1e-12);
        let inv = t.neg().unwrap().add_scalar(1.0).unwrap();
        assert!((v(&dice_loss(&inv, &t, 0.0).unwrap()) - 1.0).abs() < 1e-12);
        // half overlap: P = (1, 0, 0, 0): dice = 1 - 2*1/(1+2) = 1/3, IoU = 1 - 1/2
        let p = Tensor::from_slice(&[1.0f64, 0.0, 0.0, 0.0], &[1, 4]).unwrap();
        assert!((v(&dice_loss(&p, &t, 0.0).unwrap()) - 1.0 / 3.0).abs() < 1e-12);
        assert!((v(&iou_loss(&p, &t, 0.0).unwrap()) - 0.5).abs() < 1e-12);
        // tversky with alpha = beta = 0.5 equals dice
        assert!((v(&tversky_loss(&p, &t, 0.5, 0.5, 0.0).unwrap()) - v(&dice_loss(&p, &t, 0.0).unwrap())).abs() < 1e-12);
        // penalizing false negatives more makes the same prediction worse
        assert!(v(&tversky_loss(&p, &t, 0.3, 0.7, 0.0).unwrap()) > v(&tversky_loss(&p, &t, 0.7, 0.3, 0.0).unwrap()));
        assert!(dice_loss(&p, &Tensor::zeros(&[1, 3], crate::dtype::DType::F64, &crate::backend::Device::Cpu).unwrap(), 1.0).is_err());
    }

    #[test]
    fn image_quality_metrics() {
        let x = img(2, 16, 16, 0.0);
        assert!((v(&ssim(&x, &x, 1.0, 7, 1.5).unwrap()) - 1.0).abs() < 1e-9);
        let noisy = x.add(&img(2, 16, 16, 3.0).sub_scalar(0.5).unwrap().mul_scalar(0.4).unwrap()).unwrap();
        let s = v(&ssim(&x, &noisy, 1.0, 7, 1.5).unwrap());
        assert!(s < 0.99 && s > -1.0);
        // more noise -> lower SSIM and PSNR
        let noisier = x.add(&img(2, 16, 16, 3.0).sub_scalar(0.5).unwrap().mul_scalar(0.9).unwrap()).unwrap();
        assert!(v(&ssim(&x, &noisier, 1.0, 7, 1.5).unwrap()) < s);
        assert!(v(&psnr(&x, &noisier, 1.0).unwrap()) < v(&psnr(&x, &noisy, 1.0).unwrap()));
        // PSNR of a uniform error of 0.1 at range 1 is 20 dB
        let shifted = x.add_scalar(0.1).unwrap();
        assert!((v(&psnr(&x, &shifted, 1.0).unwrap()) - 20.0).abs() < 1e-9);
        // SSIM is differentiable
        let p = noisy.requires_grad_(true);
        ssim(&x, &p, 1.0, 7, 1.5).unwrap().backward().unwrap();
        assert!(p.grad().unwrap().abs().unwrap().sum().unwrap().item::<f64>().unwrap() > 0.0);
        assert!(ssim(&x, &x, 1.0, 4, 1.5).is_err());
    }

    #[test]
    fn variation_and_distances() {
        let flat = Tensor::full(&[1, 1, 4, 4], 0.3, crate::dtype::DType::F64, &crate::backend::Device::Cpu).unwrap();
        assert_eq!(v(&total_variation(&flat).unwrap()), 0.0);
        let ramp = Tensor::from_slice(&(0..16).map(|i| (i % 4) as f64).collect::<Vec<_>>(), &[1, 1, 4, 4]).unwrap();
        assert_eq!(v(&total_variation(&ramp).unwrap()), 12.0); // 4 rows x 3 unit steps horizontally
        let a = Tensor::from_slice(&[0.0f64, 0.0, 1.0, 0.0], &[1, 2, 2]).unwrap();
        let b = Tensor::from_slice(&[0.0f64, 0.0, 3.0, 0.0], &[1, 2, 2]).unwrap();
        // a: (0,0),(1,0); b: (0,0),(3,0): a->b squared distances 0 and 1 (mean 0.5), b->a 0 and 4 (mean 2)
        assert!((v(&chamfer_distance(&a, &b).unwrap()) - 2.5).abs() < 1e-12);
        assert!(v(&chamfer_distance(&a, &a).unwrap()).abs() < 1e-12);
        let x = Tensor::from_slice(&[0.0f64, 1.0, 2.0], &[1, 3]).unwrap();
        let y = Tensor::from_slice(&[2.0f64, 3.0, 4.0], &[1, 3]).unwrap();
        assert!((v(&wasserstein1d(&x, &y).unwrap()) - 2.0).abs() < 1e-12);
        let shuffled = Tensor::from_slice(&[2.0f64, 0.0, 1.0], &[1, 3]).unwrap();
        assert!(v(&wasserstein1d(&x, &shuffled).unwrap()).abs() < 1e-12);
        assert_eq!(reflect_pad2d(&ramp, 1).unwrap().shape(), [1, 1, 6, 6]);
    }
}
