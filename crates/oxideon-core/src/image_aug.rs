//! More image augmentations for `[C, H, W]` images in `[0, 1]`: [`RandomAffine`] (rotation / translation / scale /
//! shear via `affine_grid` + `grid_sample`), [`RandomResizedCrop`], [`Invert`], [`Posterize`], [`Solarize`],
//! [`AutoContrast`], [`RandomApply`], and [`RandAugment`].

use crate::data::Transform;
use crate::error::{Error, Result};
use crate::nn::functional_ext::{interpolate, InterpMode};
use crate::rng;
use crate::tensor::Tensor;
use crate::vision_ops::{affine_grid, grid_sample};

fn uniform(lo: f64, hi: f64) -> f64 {
    lo + (hi - lo) * rng::with_rng(|r| r.next_f64())
}

fn chw(x: &Tensor, op: &str) -> Result<(usize, usize, usize)> {
    if x.ndim() != 3 {
        return Err(Error::Msg(format!("{op}: expects a [C, H, W] image, got shape {:?}", x.shape())));
    }
    Ok((x.shape()[0], x.shape()[1], x.shape()[2]))
}

/// Warps `x` by the 2x3 normalized-coordinate matrix `m` (the sampling map from output to input coordinates).
pub fn warp_affine(x: &Tensor, m: [f64; 6]) -> Result<Tensor> {
    let (c, h, w) = chw(x, "warp_affine")?;
    let theta = Tensor::from_slice(&m, &[1, 2, 3])?.to_dtype(x.dtype())?.to_device(&x.device())?;
    let grid = affine_grid(&theta, [1, c, h, w], false)?;
    grid_sample(&x.unsqueeze(0)?, &grid, false)?.squeeze_axis(0)
}

/// Random rotation (degrees), translation (fraction of size), isotropic scale and shear (degrees about x).
pub struct RandomAffine {
    pub degrees: f64,
    pub translate: f64,
    pub scale: (f64, f64),
    pub shear: f64,
}

impl Transform for RandomAffine {
    fn apply(&self, x: &Tensor) -> Result<Tensor> {
        let a = uniform(-self.degrees, self.degrees).to_radians();
        let sh = uniform(-self.shear, self.shear).to_radians();
        let s = uniform(self.scale.0, self.scale.1).max(1e-3);
        let (tx, ty) = (uniform(-self.translate, self.translate) * 2.0, uniform(-self.translate, self.translate) * 2.0);
        // sampling map = inverse of the forward transform (rotate, shear, scale, then translate)
        let (c, sn) = (a.cos() / s, a.sin() / s);
        let m = [c, -sn + sh.tan() * c, -tx, sn, c + sh.tan() * sn, -ty];
        warp_affine(x, m)
    }
}

/// Crops a random area (`scale` of the image) with a random aspect ratio and resizes it to `(height, width)`.
pub struct RandomResizedCrop {
    pub height: usize,
    pub width: usize,
    pub scale: (f64, f64),
    pub ratio: (f64, f64),
}

impl Transform for RandomResizedCrop {
    fn apply(&self, x: &Tensor) -> Result<Tensor> {
        let (_, h, w) = chw(x, "RandomResizedCrop")?;
        let (mut ch, mut cw) = (h, w);
        for _ in 0..10 {
            let area = (h * w) as f64 * uniform(self.scale.0, self.scale.1);
            let r = uniform(self.ratio.0.ln(), self.ratio.1.ln()).exp();
            let (tw, th) = ((area * r).sqrt().round() as usize, (area / r).sqrt().round() as usize);
            if tw >= 1 && th >= 1 && tw <= w && th <= h {
                (ch, cw) = (th, tw);
                break;
            }
        }
        let top = rng::with_rng(|r| r.below(h - ch + 1).unwrap_or(0));
        let left = rng::with_rng(|r| r.below(w - cw + 1).unwrap_or(0));
        let patch = x.narrow(1, top, ch)?.narrow(2, left, cw)?;
        interpolate(&patch.unsqueeze(0)?, Some(&[self.height, self.width]), None, InterpMode::Linear, false)?.squeeze_axis(0)
    }
}

/// `1 - x`.
pub struct Invert;

impl Transform for Invert {
    fn apply(&self, x: &Tensor) -> Result<Tensor> {
        x.neg()?.add_scalar(1.0)
    }
}

/// Keeps only the top `bits` bits of every 8-bit level.
pub struct Posterize(pub u32);

impl Transform for Posterize {
    fn apply(&self, x: &Tensor) -> Result<Tensor> {
        let levels = 2f64.powi(self.0.clamp(1, 8) as i32);
        x.mul_scalar(levels)?.floor()?.clamp(0.0, levels - 1.0)?.div_scalar(levels)
    }
}

/// Inverts every pixel at or above `threshold`.
pub struct Solarize(pub f64);

impl Transform for Solarize {
    fn apply(&self, x: &Tensor) -> Result<Tensor> {
        let mask = x.ge(&Tensor::full(x.shape(), self.0, x.dtype(), &x.device())?)?;
        Tensor::where_cond(&mask, &x.neg()?.add_scalar(1.0)?, x)
    }
}

/// Stretches each channel to span `[0, 1]`; constant channels are left untouched.
pub struct AutoContrast;

impl Transform for AutoContrast {
    fn apply(&self, x: &Tensor) -> Result<Tensor> {
        let (c, h, w) = chw(x, "AutoContrast")?;
        let flat = x.reshape(&[c, h * w])?;
        let (lo, hi) = (flat.min_axis(1, true)?, flat.max_axis(1, true)?);
        let range = hi.sub(&lo)?.to_vec::<f64>()?;
        let on = |f: &dyn Fn(f64) -> f64| -> Result<Tensor> {
            Tensor::from_slice(&range.iter().map(|&r| f(r)).collect::<Vec<_>>(), &[c, 1])?.to_dtype(x.dtype())?.to_device(&x.device())
        };
        let divisor = on(&|r| if r > 1e-12 { r } else { 1.0 })?;
        let keep_lo = on(&|r| if r > 1e-12 { 1.0 } else { 0.0 })?;
        flat.sub(&lo.mul(&keep_lo)?)?.div(&divisor)?.reshape(&[c, h, w])
    }
}

/// Applies the inner transform with probability `p`.
pub struct RandomApply {
    pub inner: Box<dyn Transform + Send + Sync>,
    pub p: f64,
}

impl Transform for RandomApply {
    fn apply(&self, x: &Tensor) -> Result<Tensor> {
        if rng::with_rng(|r| r.next_f64()) < self.p { self.inner.apply(x) } else { Ok(x.clone()) }
    }
}

/// RandAugment (Cubuk et al. 2020): applies `n` operations drawn uniformly from a fixed pool, all at strength
/// `magnitude` in `[0, 1]`.
pub struct RandAugment {
    pub n: usize,
    pub magnitude: f64,
}

impl Transform for RandAugment {
    fn apply(&self, x: &Tensor) -> Result<Tensor> {
        let m = self.magnitude.clamp(0.0, 1.0);
        let sign = |v: f64| if rng::with_rng(|r| r.next_f64()) < 0.5 { -v } else { v };
        let mut y = x.clone();
        for _ in 0..self.n {
            y = match rng::with_rng(|r| r.below(8).unwrap_or(0)) {
                0 => y,
                1 => RandomAffine { degrees: 30.0 * m, translate: 0.0, scale: (1.0, 1.0), shear: 0.0 }.apply(&y)?,
                2 => RandomAffine { degrees: 0.0, translate: 0.33 * m, scale: (1.0, 1.0), shear: 0.0 }.apply(&y)?,
                3 => RandomAffine { degrees: 0.0, translate: 0.0, scale: (1.0, 1.0), shear: 20.0 * m }.apply(&y)?,
                4 => y.mul_scalar(1.0 + sign(0.9 * m))?.clamp(0.0, 1.0)?,
                5 => Posterize(8 - (4.0 * m).round() as u32).apply(&y)?,
                6 => Solarize(1.0 - m).apply(&y)?,
                _ => AutoContrast.apply(&y)?,
            };
        }
        Ok(y)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn img() -> Tensor {
        Tensor::from_slice(&(0..48).map(|i| i as f32 / 47.0).collect::<Vec<_>>(), &[3, 4, 4]).unwrap()
    }
    fn v(t: &Tensor) -> Vec<f32> {
        t.to_vec::<f32>().unwrap()
    }

    #[test]
    fn point_ops() {
        let x = img();
        assert!(v(&Invert.apply(&x).unwrap()).iter().zip(v(&x)).all(|(a, b)| (a + b - 1.0).abs() < 1e-6));
        let p = v(&Posterize(1).apply(&x).unwrap());
        assert!(p.iter().all(|&a| a == 0.0 || a == 0.5));
        let s = v(&Solarize(0.5).apply(&x).unwrap());
        assert!(s[0].abs() < 1e-6 && s[47].abs() < 1e-6 && (s[40] - (1.0 - 40.0 / 47.0)).abs() < 1e-5);
        let squeezed = x.mul_scalar(0.25).unwrap().add_scalar(0.25).unwrap();
        let a = v(&AutoContrast.apply(&squeezed).unwrap());
        let ch0 = &a[..16];
        assert!(ch0.iter().cloned().fold(f32::INFINITY, f32::min).abs() < 1e-5);
        assert!((ch0.iter().cloned().fold(0.0, f32::max) - 1.0).abs() < 1e-5);
        let flat = Tensor::full(&[1, 2, 2], 0.3, crate::dtype::DType::F32, &crate::backend::Device::Cpu).unwrap();
        assert!(v(&AutoContrast.apply(&flat).unwrap()).iter().all(|&a| (a - 0.3).abs() < 1e-6));
    }

    #[test]
    fn geometric_ops() {
        let x = img();
        let id = warp_affine(&x, [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]).unwrap();
        assert!(v(&id).iter().zip(v(&x)).all(|(a, b)| (a - b).abs() < 1e-5));
        let t = RandomAffine { degrees: 30.0, translate: 0.1, scale: (0.9, 1.1), shear: 10.0 };
        assert_eq!(t.apply(&x).unwrap().shape(), [3, 4, 4]);
        let rrc = RandomResizedCrop { height: 8, width: 6, scale: (0.3, 1.0), ratio: (0.75, 1.33) };
        assert_eq!(rrc.apply(&x).unwrap().shape(), [3, 8, 6]);
        let ra = RandAugment { n: 3, magnitude: 0.5 };
        for _ in 0..20 {
            let y = ra.apply(&x).unwrap();
            assert_eq!(y.shape(), [3, 4, 4]);
            assert!(v(&y).iter().all(|a| a.is_finite()));
        }
        let never = RandomApply { inner: Box::new(Invert), p: 0.0 };
        assert_eq!(v(&never.apply(&x).unwrap()), v(&x));
        assert!(AutoContrast.apply(&Tensor::zeros_f32(&[4]).unwrap()).is_err());
    }
}
