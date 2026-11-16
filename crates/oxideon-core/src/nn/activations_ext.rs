//! Activation modules that the layer set lacked: [`Softmax`], [`LogSoftmax`], [`Softmin`], [`Softmax2d`],
//! [`Tanhshrink`], [`Threshold`], [`RReLU`] and the tanh-approximated [`GeluTanh`].

use super::Module;
use crate::error::Result;
use crate::tensor::Tensor;
use std::sync::atomic::{AtomicBool, Ordering};

/// `softmax` along `dim`.
pub struct Softmax(pub isize);

impl Module for Softmax {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        input.softmax(self.0)
    }
}

/// `log_softmax` along `dim`.
pub struct LogSoftmax(pub isize);

impl Module for LogSoftmax {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        input.log_softmax(self.0)
    }
}

/// `softmax(-x)` along `dim`.
pub struct Softmin(pub isize);

impl Module for Softmin {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        input.softmin(self.0)
    }
}

/// Softmax over the channel dimension of `[N, C, H, W]` (or `[C, H, W]`).
pub struct Softmax2d;

impl Module for Softmax2d {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        if !(3..=4).contains(&input.ndim()) {
            return Err(crate::error::Error::Msg("Softmax2d: expects [N, C, H, W] or [C, H, W]".into()));
        }
        input.softmax(-3)
    }
}

/// `x - tanh(x)`.
pub struct Tanhshrink;

impl Module for Tanhshrink {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        input.tanhshrink()
    }
}

/// `x` where `x > threshold`, else `value`.
pub struct Threshold {
    pub threshold: f64,
    pub value: f64,
}

impl Module for Threshold {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        let keep = input.gt(&input.full_like(self.threshold)?)?;
        Tensor::where_cond(&keep, input, &input.full_like(self.value)?)
    }
}

/// Randomized leaky ReLU: the negative slope is drawn uniformly from `[lower, upper]` while training and fixed at the
/// midpoint in evaluation mode.
pub struct RReLU {
    pub lower: f64,
    pub upper: f64,
    training: AtomicBool,
}

impl RReLU {
    pub fn new(lower: f64, upper: f64) -> Self {
        Self { lower, upper, training: AtomicBool::new(true) }
    }
}

impl Default for RReLU {
    fn default() -> Self {
        Self::new(1.0 / 8.0, 1.0 / 3.0)
    }
}

impl Module for RReLU {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        let slope = if self.training.load(Ordering::Relaxed) {
            Tensor::rand_uniform(input.shape(), self.lower as f32, self.upper as f32)?
                .to_dtype(input.dtype())?
                .to_device(&input.device())?
        } else {
            input.full_like((self.lower + self.upper) / 2.0)?
        };
        let pos = input.ge(&input.zeros_like()?)?;
        Tensor::where_cond(&pos, input, &input.mul(&slope)?)
    }
    fn train(&self, mode: bool) {
        self.training.store(mode, Ordering::Relaxed);
    }
}

/// GELU with the tanh approximation (`GELU(approximate='tanh')`).
pub struct GeluTanh;

impl Module for GeluTanh {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        let c = (2.0 / std::f64::consts::PI).sqrt();
        let inner = input.add(&input.powf(3.0)?.mul_scalar(0.044_715)?)?.mul_scalar(c)?;
        input.mul(&inner.tanh()?.add_scalar(1.0)?)?.mul_scalar(0.5)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn f(v: &[f64], shape: &[usize]) -> Tensor {
        Tensor::from_slice(v, shape).unwrap()
    }
    fn vals(t: &Tensor) -> Vec<f64> {
        t.to_vec::<f64>().unwrap()
    }

    #[test]
    fn activations() {
        let x = f(&[0.0, 0.0], &[1, 2]);
        assert_eq!(vals(&Softmax(1).forward(&x).unwrap()), vec![0.5, 0.5]);
        assert!((vals(&LogSoftmax(1).forward(&x).unwrap())[0] + std::f64::consts::LN_2).abs() < 1e-12);
        let y = f(&[1.0, 3.0], &[2]);
        let sm = vals(&Softmin(0).forward(&y).unwrap());
        assert!(sm[0] > sm[1] && (sm[0] + sm[1] - 1.0).abs() < 1e-12);
        let img = f(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], &[1, 2, 2, 2]);
        let s2 = Softmax2d.forward(&img).unwrap();
        let v = vals(&s2);
        for i in 0..4 {
            assert!((v[i] + v[4 + i] - 1.0).abs() < 1e-12);
        }
        assert!(Softmax2d.forward(&f(&[1.0, 2.0], &[2])).is_err());
        assert!((vals(&Tanhshrink.forward(&f(&[1.0], &[1])).unwrap())[0] - (1.0 - 1f64.tanh())).abs() < 1e-12);
        assert_eq!(vals(&Threshold { threshold: 0.5, value: -1.0 }.forward(&f(&[0.2, 0.5, 0.9], &[3])).unwrap()), vec![-1.0, -1.0, 0.9]);
        let r = RReLU::new(0.2, 0.4);
        r.eval();
        let ev = vals(&r.forward(&f(&[-10.0, 2.0], &[2])).unwrap());
        assert!((ev[0] + 3.0).abs() < 1e-9 && ev[1] == 2.0, "{ev:?}");
        r.train(true);
        let slope = vals(&r.forward(&f(&[-10.0; 200], &[200])).unwrap());
        assert!(slope.iter().all(|v| (-4.0..=-2.0).contains(v)) && slope.windows(2).any(|w| w[0] != w[1]));
        // tanh-GELU against the closed form at x = 1
        let g = vals(&GeluTanh.forward(&f(&[1.0], &[1])).unwrap())[0];
        assert!((g - 0.841_192).abs() < 1e-5, "{g}");
    }
}
