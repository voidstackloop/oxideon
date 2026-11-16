//! Quantization-aware training: [`fake_quantize`] with a straight-through estimator, range observers
//! ([`MinMaxObserver`], [`MovingAverageMinMaxObserver`]) that turn tensor statistics into `(scale, zero_point)`,
//! the [`FakeQuantize`] module and [`QatLinear`], a linear layer whose weights (per output channel) and activations
//! (per tensor) are fake-quantized during training so the network learns to tolerate the rounding.
//!
//! Fake quantization computes `(clamp(round(x / scale) + zp, qmin, qmax) - zp) * scale`; the gradient passes through
//! `round` unchanged (straight-through) and is zero for values clipped by the range.

use crate::error::{Error, Result};
use crate::nn::{Linear, Module};
use crate::tensor::{Tensor, no_grad};
use std::sync::Mutex;

/// Integer grid of a fake quantizer.
#[derive(Clone, Copy, Debug)]
pub struct QuantRange {
    pub qmin: i64,
    pub qmax: i64,
}

impl QuantRange {
    /// Signed `bits`-bit grid, e.g. `[-128, 127]` for 8.
    pub fn signed(bits: u32) -> Self {
        Self { qmin: -(1 << (bits - 1)), qmax: (1 << (bits - 1)) - 1 }
    }
    /// Unsigned `bits`-bit grid, e.g. `[0, 255]`.
    pub fn unsigned(bits: u32) -> Self {
        Self { qmin: 0, qmax: (1 << bits) - 1 }
    }
}

/// `(scale, zero_point)` broadcastable against the tensor being quantized.
#[derive(Clone)]
pub struct QParams {
    pub scale: Tensor,
    pub zero_point: Tensor,
}

/// Fake-quantizes `x` to the grid of `range` with straight-through gradients (also to `scale` if it requires grad).
pub fn fake_quantize(x: &Tensor, q: &QParams, range: QuantRange) -> Result<Tensor> {
    let scaled = x.div(&q.scale)?;
    // straight-through rounding: forward round(scaled), backward identity
    let rounded = scaled.add(&scaled.round()?.sub(&scaled)?.detach())?;
    let shifted = rounded.add(&q.zero_point)?;
    let clamped = shifted.clamp(range.qmin as f64, range.qmax as f64)?;
    clamped.sub(&q.zero_point)?.mul(&q.scale)
}

/// Computes quantization parameters from observed ranges.
fn qparams_from_range(min: &Tensor, max: &Tensor, range: QuantRange, symmetric: bool) -> Result<QParams> {
    let zero = min.zeros_like()?;
    let min = min.minimum(&zero)?; // the range must contain zero
    let max = max.maximum(&zero)?;
    let span = (range.qmax - range.qmin) as f64;
    if symmetric {
        let amax = max.maximum(&min.neg()?)?;
        let scale = amax.mul_scalar(2.0 / span)?.clamp(1e-12, f64::INFINITY)?;
        let zp = scale.full_like(((range.qmax + range.qmin + 1) / 2) as f64)?;
        let zp = if range.qmin == 0 { zp } else { scale.zeros_like()? };
        Ok(QParams { scale, zero_point: zp })
    } else {
        let scale = max.sub(&min)?.div_scalar(span)?.clamp(1e-12, f64::INFINITY)?;
        let zp = min.neg()?.div(&scale)?.round()?.add_scalar(range.qmin as f64)?.clamp(range.qmin as f64, range.qmax as f64)?;
        Ok(QParams { scale, zero_point: zp })
    }
}

/// Tracks the running minimum and maximum of what it sees, per tensor or along `channel_axis`.
pub struct MinMaxObserver {
    pub channel_axis: Option<isize>,
    state: Mutex<Option<(Tensor, Tensor)>>,
    /// `Some(m)` keeps an exponential moving average with momentum `m` instead of the all-time extremes.
    pub averaging: Option<f64>,
}

pub type MovingAverageMinMaxObserver = MinMaxObserver;

impl MinMaxObserver {
    pub fn per_tensor() -> Self {
        Self { channel_axis: None, state: Mutex::new(None), averaging: None }
    }
    pub fn per_channel(axis: isize) -> Self {
        Self { channel_axis: Some(axis), state: Mutex::new(None), averaging: None }
    }
    pub fn moving_average(mut self, momentum: f64) -> Self {
        self.averaging = Some(momentum);
        self
    }

    /// Updates the statistics with a batch of values.
    pub fn observe(&self, x: &Tensor) -> Result<()> {
        no_grad(|| -> Result<()> {
            let x = x.detach();
            let (mn, mx) = match self.channel_axis {
                None => (x.amin(&[], false)?, x.amax(&[], false)?),
                Some(a) => {
                    let ax = crate::layout::normalize_axis("observe", a, x.ndim())?;
                    let others: Vec<isize> = (0..x.ndim() as isize).filter(|&d| d != ax as isize).collect();
                    (x.amin(&others, false)?, x.amax(&others, false)?)
                }
            };
            let mut st = self.state.lock().map_err(|_| Error::Msg("observer lock poisoned".into()))?;
            *st = Some(match (st.take(), self.averaging) {
                (None, _) => (mn, mx),
                (Some((pmn, pmx)), Some(m)) => (
                    pmn.mul_scalar(1.0 - m)?.add(&mn.mul_scalar(m)?)?,
                    pmx.mul_scalar(1.0 - m)?.add(&mx.mul_scalar(m)?)?,
                ),
                (Some((pmn, pmx)), None) => (pmn.minimum(&mn)?, pmx.maximum(&mx)?),
            });
            Ok(())
        })
    }

    /// Quantization parameters for the ranges seen so far; shaped to broadcast against a tensor of rank `ndim`.
    pub fn qparams(&self, range: QuantRange, symmetric: bool, ndim: usize) -> Result<QParams> {
        let st = self.state.lock().map_err(|_| Error::Msg("observer lock poisoned".into()))?;
        let (mn, mx) = st.as_ref().ok_or_else(|| Error::Msg("observer has not seen any data".into()))?;
        let mut q = qparams_from_range(mn, mx, range, symmetric)?;
        if let Some(a) = self.channel_axis {
            let ax = crate::layout::normalize_axis("qparams", a, ndim)?;
            let mut shape = vec![1; ndim];
            shape[ax] = q.scale.numel();
            q = QParams { scale: q.scale.reshape(&shape)?, zero_point: q.zero_point.reshape(&shape)? };
        }
        Ok(q)
    }
}

/// An observer plus a fake quantizer. While `observing` the statistics are updated on every call; with `enabled`
/// false the module is the identity.
pub struct FakeQuantize {
    pub observer: MinMaxObserver,
    pub range: QuantRange,
    pub symmetric: bool,
    pub enabled: std::sync::atomic::AtomicBool,
    pub observing: std::sync::atomic::AtomicBool,
}

impl FakeQuantize {
    pub fn new(observer: MinMaxObserver, range: QuantRange, symmetric: bool) -> Self {
        Self { observer, range, symmetric, enabled: true.into(), observing: true.into() }
    }
    /// 8-bit symmetric per-channel weight quantizer along `axis`.
    pub fn weights_int8(axis: isize) -> Self {
        Self::new(MinMaxObserver::per_channel(axis), QuantRange::signed(8), true)
    }
    /// 8-bit asymmetric per-tensor activation quantizer with a moving-average range.
    pub fn activations_uint8() -> Self {
        Self::new(MinMaxObserver::per_tensor().moving_average(0.1), QuantRange::unsigned(8), false)
    }
    pub fn qparams(&self, ndim: usize) -> Result<QParams> {
        self.observer.qparams(self.range, self.symmetric, ndim)
    }
}

impl Module for FakeQuantize {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        use std::sync::atomic::Ordering::Relaxed;
        if self.observing.load(Relaxed) {
            self.observer.observe(x)?;
        }
        if !self.enabled.load(Relaxed) {
            return Ok(x.clone());
        }
        let q = self.qparams(x.ndim())?;
        fake_quantize(x, &q, self.range)
    }
}

/// A [`Linear`] trained with fake-quantized weights and input activations.
pub struct QatLinear {
    pub linear: Linear,
    pub weight_fq: FakeQuantize,
    pub act_fq: FakeQuantize,
}

impl QatLinear {
    pub fn new(linear: Linear) -> Self {
        Self { linear, weight_fq: FakeQuantize::weights_int8(0), act_fq: FakeQuantize::activations_uint8() }
    }

    /// The int8 weights `[out, in]` and per-output-channel scales this layer would deploy with.
    pub fn quantized_weights(&self) -> Result<(Vec<i8>, Vec<f32>)> {
        self.weight_fq.observer.observe(&self.linear.weight)?;
        let q = self.weight_fq.qparams(2)?;
        let scaled = self.linear.weight.detach().div(&q.scale)?.round()?.clamp(-128.0, 127.0)?;
        let w: Vec<i8> = scaled.to_vec::<f32>()?.into_iter().map(|v| v as i8).collect();
        Ok((w, q.scale.to_vec::<f32>()?))
    }
}

impl Module for QatLinear {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let xq = self.act_fq.forward(x)?;
        self.weight_fq.observer.observe(&self.linear.weight)?;
        let q = self.weight_fq.qparams(2)?;
        let wq = fake_quantize(&self.linear.weight, &q, self.weight_fq.range)?;
        let y = xq.matmul(&wq.t()?)?;
        match &self.linear.bias {
            Some(b) => y.add(b),
            None => Ok(y),
        }
    }
    fn named_parameters(&self) -> Vec<(String, Tensor)> {
        self.linear.named_parameters()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn t(v: &[f64], shape: &[usize]) -> Tensor {
        Tensor::from_slice(v, shape).unwrap()
    }
    fn vals(x: &Tensor) -> Vec<f64> {
        x.to_vec::<f64>().unwrap()
    }

    #[test]
    fn rounding_error_is_bounded_by_half_a_step() {
        let x = t(&(0..200).map(|i| -3.0 + 0.03 * i as f64).collect::<Vec<_>>(), &[200]);
        let obs = MinMaxObserver::per_tensor();
        obs.observe(&x).unwrap();
        let q = obs.qparams(QuantRange::signed(8), false, 1).unwrap();
        let scale = vals(&q.scale)[0];
        assert!((scale - 5.97 / 255.0).abs() < 1e-9);
        let y = fake_quantize(&x, &q, QuantRange::signed(8)).unwrap();
        assert!(vals(&x).iter().zip(vals(&y)).all(|(a, b)| (a - b).abs() <= scale / 2.0 + 1e-9));
        // the grid has at most 256 distinct levels
        let mut levels = vals(&y);
        levels.sort_by(f64::total_cmp);
        levels.dedup_by(|a, b| (*a - *b).abs() < 1e-9);
        assert!(levels.len() <= 256);
    }

    #[test]
    fn observers_compute_known_parameters() {
        let o = MinMaxObserver::per_tensor();
        o.observe(&t(&[-2.0, 1.0, 6.0], &[3])).unwrap();
        let q = o.qparams(QuantRange::unsigned(8), false, 1).unwrap();
        assert!((vals(&q.scale)[0] - 8.0 / 255.0).abs() < 1e-12);
        assert_eq!(vals(&q.zero_point)[0], 64.0);
        let sym = o.qparams(QuantRange::signed(8), true, 1).unwrap();
        assert!((vals(&sym.scale)[0] - 12.0 / 255.0).abs() < 1e-12); // 6 / 127.5
        assert_eq!(vals(&sym.zero_point)[0], 0.0);
        // per channel along axis 0
        let pc = MinMaxObserver::per_channel(0);
        pc.observe(&t(&[1.0, -4.0, 2.0, 0.5], &[2, 2])).unwrap();
        let q = pc.qparams(QuantRange::signed(8), true, 2).unwrap();
        assert_eq!(q.scale.shape(), [2, 1]);
        assert!((vals(&q.scale)[0] - 8.0 / 255.0).abs() < 1e-12 && (vals(&q.scale)[1] - 4.0 / 255.0).abs() < 1e-12);
        // moving average mixes old and new ranges
        let ma = MinMaxObserver::per_tensor().moving_average(0.5);
        ma.observe(&t(&[0.0, 2.0], &[2])).unwrap();
        ma.observe(&t(&[0.0, 6.0], &[2])).unwrap();
        let q = ma.qparams(QuantRange::unsigned(8), false, 1).unwrap();
        assert!((vals(&q.scale)[0] - 4.0 / 255.0).abs() < 1e-12);
        assert!(MinMaxObserver::per_tensor().qparams(QuantRange::signed(8), true, 1).is_err());
    }

    #[test]
    fn straight_through_gradients() {
        let x = t(&[0.1, 0.45, 100.0, -100.0], &[4]).requires_grad_(true);
        let q = QParams { scale: t(&[0.1], &[1]), zero_point: t(&[0.0], &[1]) };
        fake_quantize(&x, &q, QuantRange::signed(8)).unwrap().sum().unwrap().backward().unwrap();
        // inside the range (|x| / 0.1 <= 127) the gradient is 1; clipped values get 0
        assert_eq!(vals(&x.grad().unwrap()), vec![1.0, 1.0, 0.0, 0.0]);
    }

    #[test]
    fn qat_trains_a_layer_that_survives_quantization() {
        crate::rng::manual_seed(21);
        let layer = QatLinear::new(Linear::new(8, 4).unwrap());
        let x = Tensor::randn(&[64, 8]).unwrap();
        let w_true = Tensor::randn(&[8, 4]).unwrap().mul_scalar(0.5).unwrap();
        let y = x.matmul(&w_true).unwrap();
        let mut opt = crate::optim::Adam::new(layer.parameters(), 0.05);
        use crate::optim::Optimizer;
        let mut first = 0.0;
        let mut last = 0.0;
        for step in 0..150 {
            opt.zero_grad();
            let loss = crate::nn::loss::mse_loss(&layer.forward(&x).unwrap(), &y).unwrap();
            if step == 0 {
                first = loss.item::<f32>().unwrap();
            }
            last = loss.item::<f32>().unwrap();
            loss.backward().unwrap();
            opt.step().unwrap();
        }
        assert!(last < first * 0.05, "{first} -> {last}");
        let (w8, scales) = layer.quantized_weights().unwrap();
        assert_eq!((w8.len(), scales.len()), (32, 4));
        // dequantized int8 weights stay close to the float weights
        let wf = layer.linear.weight.to_vec::<f32>().unwrap();
        let worst = (0..32).map(|i| (w8[i] as f32 * scales[i / 8] - wf[i]).abs()).fold(0.0f32, f32::max);
        assert!(worst <= scales.iter().cloned().fold(0.0, f32::max) / 2.0 + 1e-5, "{worst}");
    }
}
