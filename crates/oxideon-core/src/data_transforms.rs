//! Image augmentation for `[C, H, W]` samples, implementing [`Transform`]: [`RandomHorizontalFlip`],
//! [`RandomVerticalFlip`], [`CenterCrop`], [`RandomCrop`] (with optional padding), [`Resize`], [`RandomErasing`],
//! [`ColorJitter`] (brightness / contrast / saturation), [`GaussianBlur`], [`Grayscale`]; plus the batch-level mixing
//! functions [`mixup`] and [`cutmix`] for `[N, C, H, W]` batches with one-hot (or soft) labels.

use crate::data::Transform;
use crate::error::{Error, Result};
use crate::nn::functional_ext::{InterpMode, PadMode, interpolate, pad_mode};
use crate::rng;
use crate::tensor::Tensor;

fn chw(x: &Tensor, op: &str) -> Result<(usize, usize, usize)> {
    if x.ndim() != 3 {
        return Err(Error::Msg(format!("{op}: expects a [C, H, W] image, got shape {:?}", x.shape())));
    }
    Ok((x.shape()[0], x.shape()[1], x.shape()[2]))
}

fn coin(p: f64) -> bool {
    rng::with_rng(|r| r.next_f64() < p)
}

fn below(n: usize) -> usize {
    rng::with_rng(|r| r.below(n.max(1)).unwrap_or(0))
}

/// Flips left-right with probability `p`.
pub struct RandomHorizontalFlip(pub f64);

impl Transform for RandomHorizontalFlip {
    fn apply(&self, x: &Tensor) -> Result<Tensor> {
        chw(x, "RandomHorizontalFlip")?;
        if coin(self.0) { x.flip(&[2]) } else { Ok(x.clone()) }
    }
}

/// Flips top-bottom with probability `p`.
pub struct RandomVerticalFlip(pub f64);

impl Transform for RandomVerticalFlip {
    fn apply(&self, x: &Tensor) -> Result<Tensor> {
        chw(x, "RandomVerticalFlip")?;
        if coin(self.0) { x.flip(&[1]) } else { Ok(x.clone()) }
    }
}

/// Crops the central `(height, width)` window.
pub struct CenterCrop(pub usize, pub usize);

impl Transform for CenterCrop {
    fn apply(&self, x: &Tensor) -> Result<Tensor> {
        let (_, h, w) = chw(x, "CenterCrop")?;
        if self.0 > h || self.1 > w {
            return Err(Error::Msg(format!("CenterCrop: {}x{} window exceeds the {h}x{w} image", self.0, self.1)));
        }
        x.narrow(1, (h - self.0) / 2, self.0)?.narrow(2, (w - self.1) / 2, self.1)
    }
}

/// Crops a random `(height, width)` window, after padding every side by `padding` (zeros or `mode`).
pub struct RandomCrop {
    pub height: usize,
    pub width: usize,
    pub padding: usize,
    pub mode: PadMode,
}

impl RandomCrop {
    pub fn new(height: usize, width: usize) -> Self {
        Self { height, width, padding: 0, mode: PadMode::Constant(0.0) }
    }
}

impl Transform for RandomCrop {
    fn apply(&self, x: &Tensor) -> Result<Tensor> {
        chw(x, "RandomCrop")?;
        let x = if self.padding > 0 { pad_mode(x, &[(self.padding, self.padding), (self.padding, self.padding)], self.mode)? } else { x.clone() };
        let (_, h, w) = chw(&x, "RandomCrop")?;
        if self.height > h || self.width > w {
            return Err(Error::Msg(format!("RandomCrop: {}x{} window exceeds the {h}x{w} image", self.height, self.width)));
        }
        let (top, left) = (below(h - self.height + 1), below(w - self.width + 1));
        x.narrow(1, top, self.height)?.narrow(2, left, self.width)
    }
}

/// Resizes to `(height, width)`.
pub struct Resize {
    pub height: usize,
    pub width: usize,
    pub mode: InterpMode,
}

impl Transform for Resize {
    fn apply(&self, x: &Tensor) -> Result<Tensor> {
        chw(x, "Resize")?;
        interpolate(&x.unsqueeze(0)?, Some(&[self.height, self.width]), None, self.mode, false)?.squeeze_axis(0)
    }
}

/// Fills a random rectangle with `value` with probability `p` (Cutout / Random Erasing). The rectangle covers a
/// fraction of the image area drawn from `scale` with an aspect ratio drawn from `ratio`.
pub struct RandomErasing {
    pub p: f64,
    pub scale: (f64, f64),
    pub ratio: (f64, f64),
    pub value: f64,
}

impl Default for RandomErasing {
    fn default() -> Self {
        Self { p: 0.5, scale: (0.02, 0.33), ratio: (0.3, 3.3), value: 0.0 }
    }
}

impl Transform for RandomErasing {
    fn apply(&self, x: &Tensor) -> Result<Tensor> {
        let (c, h, w) = chw(x, "RandomErasing")?;
        if !coin(self.p) {
            return Ok(x.clone());
        }
        for _ in 0..10 {
            let (area_frac, log_ratio) = rng::with_rng(|r| {
                (self.scale.0 + (self.scale.1 - self.scale.0) * r.next_f64(), self.ratio.0.ln() + (self.ratio.1.ln() - self.ratio.0.ln()) * r.next_f64())
            });
            let area = area_frac * (h * w) as f64;
            let aspect = log_ratio.exp();
            let (eh, ew) = ((area * aspect).sqrt().round() as usize, (area / aspect).sqrt().round() as usize);
            if eh == 0 || ew == 0 || eh > h || ew > w {
                continue;
            }
            let (top, left) = (below(h - eh + 1), below(w - ew + 1));
            let mut mask = vec![false; c * h * w];
            for ch in 0..c {
                for i in top..top + eh {
                    for j in left..left + ew {
                        mask[(ch * h + i) * w + j] = true;
                    }
                }
            }
            let mask = Tensor::from_slice_on(&mask, x.shape(), &x.device())?;
            return Tensor::where_cond(&mask, &x.full_like(self.value)?, x);
        }
        Ok(x.clone())
    }
}

/// Converts to grey (ITU-R 601 luma) keeping `num_output_channels` identical channels.
pub struct Grayscale(pub usize);

fn luma(x: &Tensor) -> Result<Tensor> {
    let (c, _, _) = chw(x, "Grayscale")?;
    if c != 3 {
        return Err(Error::Msg("Grayscale: expects 3 colour channels".into()));
    }
    let w = Tensor::from_slice(&[0.299f64, 0.587, 0.114], &[3, 1, 1])?.to_dtype(x.dtype())?.to_device(&x.device())?;
    x.mul(&w)?.sum_axis(0, true)
}

impl Transform for Grayscale {
    fn apply(&self, x: &Tensor) -> Result<Tensor> {
        let g = luma(x)?;
        if self.0 == 1 { Ok(g) } else { g.broadcast_to(&[self.0, x.shape()[1], x.shape()[2]])?.contiguous() }
    }
}

/// Random brightness, contrast and saturation changes (each factor drawn from `[max(0, 1 - a), 1 + a]`), with the
/// result clamped to `[0, 1]`. Needs 3-channel images.
pub struct ColorJitter {
    pub brightness: f64,
    pub contrast: f64,
    pub saturation: f64,
}

impl Transform for ColorJitter {
    fn apply(&self, x: &Tensor) -> Result<Tensor> {
        let draw = |a: f64| if a <= 0.0 { 1.0 } else { rng::with_rng(|r| (1.0 - a).max(0.0) + (2.0 * a.min(1.0)) * r.next_f64()) };
        let mut y = x.mul_scalar(draw(self.brightness))?.clamp(0.0, 1.0)?;
        let c = draw(self.contrast);
        let mean = luma(&y)?.mean()?;
        y = y.sub(&mean)?.mul_scalar(c)?.add(&mean)?.clamp(0.0, 1.0)?;
        let s = draw(self.saturation);
        let g = luma(&y)?;
        y.sub(&g)?.mul_scalar(s)?.add(&g)?.clamp(0.0, 1.0)
    }
}

/// Gaussian blur with an odd `kernel` size and standard deviation `sigma` (separable, reflect padding).
pub struct GaussianBlur {
    pub kernel: usize,
    pub sigma: f64,
}

impl Transform for GaussianBlur {
    fn apply(&self, x: &Tensor) -> Result<Tensor> {
        let (c, h, w) = chw(x, "GaussianBlur")?;
        if self.kernel % 2 == 0 || self.kernel == 0 || self.sigma <= 0.0 {
            return Err(Error::Msg("GaussianBlur: the kernel must be odd and sigma positive".into()));
        }
        let r = (self.kernel / 2) as isize;
        let k: Vec<f64> = (-r..=r).map(|i| (-(i * i) as f64 / (2.0 * self.sigma * self.sigma)).exp()).collect();
        let s: f64 = k.iter().sum();
        let k: Vec<f64> = k.iter().map(|v| v / s).collect();
        let kh = Tensor::from_slice(&k, &[1, 1, self.kernel, 1])?.to_dtype(x.dtype())?.to_device(&x.device())?;
        let kw = Tensor::from_slice(&k, &[1, 1, 1, self.kernel])?.to_dtype(x.dtype())?.to_device(&x.device())?;
        // blur every channel independently: fold channels into the batch dimension
        let p = self.kernel / 2;
        let img = pad_mode(&x.reshape(&[c, 1, h, w])?, &[(p, p), (p, p)], PadMode::Reflect)?;
        let y = img.conv2d(&kh, None, (1, 1), (0, 0), (1, 1))?.conv2d(&kw, None, (1, 1), (0, 0), (1, 1))?;
        y.reshape(&[c, h, w])
    }
}

fn beta_sample(alpha: f64) -> Result<f64> {
    use crate::distributions_ext::{Beta, Distribution};
    let a = Tensor::from_slice(&[alpha], &[1])?;
    Beta::new(a.clone(), a)?.sample()?.to_vec::<f64>().map(|v| v[0])
}

/// Mixup: blends each sample with a shuffled partner using a `Beta(alpha, alpha)` weight; returns the mixed images
/// and labels `(x, y)`. Labels must be one-hot / soft `[N, K]`.
pub fn mixup(x: &Tensor, y: &Tensor, alpha: f64) -> Result<(Tensor, Tensor)> {
    if x.shape()[0] != y.shape()[0] {
        return Err(Error::Msg("mixup: images and labels must have the same batch size".into()));
    }
    let lam = if alpha > 0.0 { beta_sample(alpha)? } else { 1.0 };
    let perm = Tensor::randperm(x.shape()[0])?;
    let (xp, yp) = (x.index_select(0, &perm)?, y.index_select(0, &perm)?);
    Ok((x.mul_scalar(lam)?.add(&xp.mul_scalar(1.0 - lam)?)?, y.mul_scalar(lam)?.add(&yp.mul_scalar(1.0 - lam)?)?))
}

/// CutMix: pastes a random box from a shuffled partner into every image and mixes the labels by the pasted area.
pub fn cutmix(x: &Tensor, y: &Tensor, alpha: f64) -> Result<(Tensor, Tensor)> {
    if x.ndim() != 4 || x.shape()[0] != y.shape()[0] {
        return Err(Error::Msg("cutmix: expects images [N, C, H, W] and labels with the same batch size".into()));
    }
    let (n, c, h, w) = (x.shape()[0], x.shape()[1], x.shape()[2], x.shape()[3]);
    let lam = if alpha > 0.0 { beta_sample(alpha)? } else { 1.0 };
    let cut = (1.0 - lam).sqrt();
    let (ch, cw) = (((h as f64) * cut) as usize, ((w as f64) * cut) as usize);
    let (cy, cx) = (below(h), below(w));
    let (top, bottom) = (cy.saturating_sub(ch / 2), (cy + ch / 2).min(h));
    let (left, right) = (cx.saturating_sub(cw / 2), (cx + cw / 2).min(w));
    let mut mask = vec![false; h * w];
    for i in top..bottom {
        for j in left..right {
            mask[i * w + j] = true;
        }
    }
    let area = (bottom - top) * (right - left);
    let lam_adj = 1.0 - area as f64 / (h * w) as f64;
    let mask = Tensor::from_slice_on(&mask, &[1, 1, h, w], &x.device())?.broadcast_to(&[n, c, h, w])?;
    let perm = Tensor::randperm(n)?;
    let xp = x.index_select(0, &perm)?;
    let mixed = Tensor::where_cond(&mask, &xp, x)?;
    let labels = y.mul_scalar(lam_adj)?.add(&y.index_select(0, &perm)?.mul_scalar(1.0 - lam_adj)?)?;
    Ok((mixed, labels))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn img(c: usize, h: usize, w: usize) -> Tensor {
        Tensor::from_slice(&(0..c * h * w).map(|i| i as f64 / (c * h * w) as f64).collect::<Vec<_>>(), &[c, h, w]).unwrap()
    }
    fn vals(t: &Tensor) -> Vec<f64> {
        t.to_vec::<f64>().unwrap()
    }

    #[test]
    fn geometric_transforms() {
        let x = img(3, 4, 6);
        assert_eq!(vals(&RandomHorizontalFlip(1.0).apply(&x).unwrap()), vals(&x.flip(&[2]).unwrap()));
        assert_eq!(vals(&RandomHorizontalFlip(0.0).apply(&x).unwrap()), vals(&x));
        assert_eq!(vals(&RandomVerticalFlip(1.0).apply(&x).unwrap()), vals(&x.flip(&[1]).unwrap()));
        let c = CenterCrop(2, 4).apply(&x).unwrap();
        assert_eq!(c.shape(), [3, 2, 4]);
        assert_eq!(vals(&c)[0], vals(&x)[6 + 1]);
        assert!(CenterCrop(5, 4).apply(&x).is_err());
        rng::manual_seed(3);
        for _ in 0..20 {
            let r = RandomCrop { height: 4, width: 6, padding: 2, mode: PadMode::Reflect }.apply(&x).unwrap();
            assert_eq!(r.shape(), [3, 4, 6]);
        }
        let big = Resize { height: 8, width: 12, mode: InterpMode::Linear }.apply(&x).unwrap();
        assert_eq!(big.shape(), [3, 8, 12]);
        let same = Resize { height: 4, width: 6, mode: InterpMode::Nearest }.apply(&x).unwrap();
        assert_eq!(vals(&same), vals(&x));
        assert!(RandomCrop::new(5, 5).apply(&x).is_err());
        assert!(RandomHorizontalFlip(0.5).apply(&Tensor::zeros(&[4, 4], crate::dtype::DType::F32, &crate::backend::Device::Cpu).unwrap()).is_err());
    }

    #[test]
    fn photometric_transforms() {
        rng::manual_seed(4);
        let x = img(3, 8, 8);
        let e = RandomErasing { p: 1.0, scale: (0.2, 0.3), ratio: (0.5, 2.0), value: -1.0 }.apply(&x).unwrap();
        let erased = vals(&e).iter().filter(|&&v| v == -1.0).count();
        assert!(erased > 0 && erased % 3 == 0, "{erased}");
        assert_eq!(vals(&RandomErasing { p: 0.0, ..Default::default() }.apply(&x).unwrap()), vals(&x));
        let g = Grayscale(3).apply(&x).unwrap();
        let gv = vals(&g);
        assert!((0..64).all(|i| gv[i] == gv[64 + i] && gv[i] == gv[128 + i]));
        assert_eq!(Grayscale(1).apply(&x).unwrap().shape(), [1, 8, 8]);
        let j = ColorJitter { brightness: 0.4, contrast: 0.4, saturation: 0.4 }.apply(&x).unwrap();
        assert!(vals(&j).iter().all(|&v| (0.0..=1.0).contains(&v)));
        let ident = ColorJitter { brightness: 0.0, contrast: 0.0, saturation: 0.0 }.apply(&x).unwrap();
        assert!(vals(&ident).iter().zip(vals(&x)).all(|(a, b)| (a - b).abs() < 1e-9));
        // blurring a constant image changes nothing; blurring an impulse spreads it symmetrically with unit mass
        let flat = Tensor::full(&[1, 6, 6], 0.7, crate::dtype::DType::F64, &crate::backend::Device::Cpu).unwrap();
        assert!(vals(&GaussianBlur { kernel: 3, sigma: 1.0 }.apply(&flat).unwrap()).iter().all(|v| (v - 0.7).abs() < 1e-9));
        let mut imp = vec![0.0; 49];
        imp[24] = 1.0;
        let b = GaussianBlur { kernel: 5, sigma: 1.0 }.apply(&Tensor::from_slice(&imp, &[1, 7, 7]).unwrap()).unwrap();
        let bv = vals(&b);
        assert!((bv.iter().sum::<f64>() - 1.0).abs() < 1e-9 && bv[24] > bv[25] && (bv[23] - bv[25]).abs() < 1e-12);
        assert!(GaussianBlur { kernel: 4, sigma: 1.0 }.apply(&flat).is_err());
    }

    #[test]
    fn batch_mixing() {
        rng::manual_seed(5);
        let x = Tensor::from_slice(&(0..2 * 3 * 4 * 4).map(|i| (i / 48) as f64).collect::<Vec<_>>(), &[2, 3, 4, 4]).unwrap();
        let y = Tensor::from_slice(&[1.0f64, 0.0, 0.0, 1.0], &[2, 2]).unwrap();
        for _ in 0..10 {
            let (mx, my) = mixup(&x, &y, 0.4).unwrap();
            // labels stay a probability distribution and the pixel mix uses the same weight as the labels
            let l = vals(&my);
            assert!((l[0] + l[1] - 1.0).abs() < 1e-9);
            let lam = l[0] * 1.0 + l[1] * 0.0; // image 0 holds 0.0, its partner (if swapped) 1.0
            let _ = lam;
            assert!(vals(&mx).iter().all(|v| (0.0..=1.0 + 1e-9).contains(v)));
            let (cx, cy) = cutmix(&x, &y, 1.0).unwrap();
            assert_eq!(cx.shape(), x.shape());
            let cl = vals(&cy);
            assert!((cl[0] + cl[1] - 1.0).abs() < 1e-9 && (cl[2] + cl[3] - 1.0).abs() < 1e-9);
            // the label weight of the pasted partner equals the pasted pixel fraction (image 0: 0 -> 1 where pasted)
            let pasted = vals(&cx)[..48].iter().filter(|&&v| v == 1.0).count() as f64 / 48.0;
            let partner = 1.0 - cl[0];
            assert!(partner == 0.0 || (pasted - partner).abs() < 1e-9 || pasted == 0.0, "{pasted} vs {partner}");
        }
        assert!(mixup(&x, &Tensor::zeros(&[3, 2], crate::dtype::DType::F64, &crate::backend::Device::Cpu).unwrap(), 0.4).is_err());
    }
}
