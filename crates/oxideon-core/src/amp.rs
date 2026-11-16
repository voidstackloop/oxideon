//! Automatic Mixed Precision (`autocast`, `GradScaler`) and Exponential Moving Average (`EMA`).

use std::cell::Cell;

use crate::dtype::DType;
use crate::error::{Error, Result};
use crate::optim::Optimizer;
use crate::tensor::{Tensor, no_grad};

thread_local! {
    static AUTOCAST_DTYPE: Cell<Option<DType>> = const { Cell::new(None) };
}

/// Runs closure `f` inside an Automatic Mixed Precision (`AMP`) context with the given target
/// floating-point `dtype` (typically [`DType::F16`] or [`DType::BF16`]).
/// Restores the outer autocast state on exit even if `f` returns an error.
pub fn autocast<F, R>(dtype: DType, f: F) -> R
where
    F: FnOnce() -> R,
{
    struct Guard(Option<DType>);
    impl Drop for Guard {
        fn drop(&mut self) {
            AUTOCAST_DTYPE.with(|c| c.set(self.0));
        }
    }
    let prev = AUTOCAST_DTYPE.with(|c| {
        let p = c.get();
        c.set(Some(dtype));
        p
    });
    let _guard = Guard(prev);
    f()
}

/// Returns `true` if [`autocast`] is currently active on this thread.
pub fn is_autocast_enabled() -> bool {
    AUTOCAST_DTYPE.with(|c| c.get().is_some())
}

/// Returns the active [`autocast`] [`DType`], if any.
pub fn autocast_dtype() -> Option<DType> {
    AUTOCAST_DTYPE.with(|c| c.get())
}

/// Casts floating-point `tensor` to the active [`autocast`] [`DType`] if `autocast` is enabled,
/// or returns `Ok(tensor.clone())` unchanged otherwise.
pub fn autocast_cast(tensor: &Tensor) -> Result<Tensor> {
    if let Some(target_dt) = autocast_dtype()
        && matches!(
            tensor.dtype(),
            DType::F16 | DType::BF16 | DType::F32 | DType::F64
        )
        && tensor.dtype() != target_dt
    {
        return tensor.to_dtype(target_dt);
    }
    Ok(tensor.clone())
}

/// Casts `bias` to the dtype of `like` when autocast is active (a Linear bias added to a half-precision product).
pub fn autocast_cast_like(bias: &Tensor, like: &Tensor) -> Result<Tensor> {
    if is_autocast_enabled() && bias.dtype() != like.dtype() && bias.dtype().is_floating_point() && like.dtype().is_floating_point() {
        return bias.to_dtype(like.dtype());
    }
    Ok(bias.clone())
}

/// The pair cast to the autocast dtype when autocast is active and a cast is needed (`None` otherwise): the entry point of
/// operations that run in low precision (matrix products, convolutions).
pub(crate) fn lowp_pair(a: &Tensor, b: &Tensor) -> Result<Option<(Tensor, Tensor)>> {
    let Some(dt) = autocast_dtype() else { return Ok(None) };
    let is_float = |t: &Tensor| matches!(t.dtype(), DType::F16 | DType::BF16 | DType::F32 | DType::F64);
    if !is_float(a) || !is_float(b) || (a.dtype() == dt && b.dtype() == dt) {
        return Ok(None);
    }
    Ok(Some((a.to_dtype(dt)?, b.to_dtype(dt)?)))
}

/// `x` widened to float32 when autocast is active and `x` is half precision (softmax, normalization, exp / ln run in
/// float32); `None` when nothing has to change.
pub(crate) fn fp32_input(x: &Tensor) -> Result<Option<Tensor>> {
    if is_autocast_enabled() && matches!(x.dtype(), DType::F16 | DType::BF16) {
        return Ok(Some(x.to_dtype(DType::F32)?));
    }
    Ok(None)
}

/// A half-precision parameter widened to float32 for an autocast float32 operation.
pub(crate) fn fp32_value(t: &Tensor) -> Result<Tensor> {
    if matches!(t.dtype(), DType::F16 | DType::BF16) { t.to_dtype(DType::F32) } else { Ok(t.clone()) }
}

/// Dynamic loss scaler for Mixed Precision (`f16` / `bf16`) training.
///
/// Multiplies the loss by `scale` before backpropagation to prevent `f16` gradient
/// underflow, unscales parameter gradients before the optimizer step, skips the step
/// if any gradient contains `Inf` or `NaN`, and adapts `scale` over time.
pub struct GradScaler {
    scale: f64,
    growth_factor: f64,
    backoff_factor: f64,
    growth_interval: usize,
    growth_tracker: usize,
}

impl Default for GradScaler {
    fn default() -> Self {
        Self {
            scale: 65536.0,
            growth_factor: 2.0,
            backoff_factor: 0.5,
            growth_interval: 2000,
            growth_tracker: 0,
        }
    }
}

impl GradScaler {
    /// Creates a scaler with initial scale `init_scale` (must be finite and
    /// positive: a negative scale would silently flip gradient signs, and a
    /// zero/NaN scale would skip every step forever).
    pub fn new(init_scale: f64) -> Result<Self> {
        if init_scale <= 0.0 || !init_scale.is_finite() {
            return Err(Error::Msg(format!(
                "GradScaler::new requires a finite positive init_scale, got {init_scale}"
            )));
        }
        Ok(Self {
            scale: init_scale,
            growth_factor: 2.0,
            backoff_factor: 0.5,
            growth_interval: 2000,
            growth_tracker: 0,
        })
    }

    pub fn with_intervals(
        mut self,
        growth_factor: f64,
        backoff_factor: f64,
        growth_interval: usize,
    ) -> Result<Self> {
        if growth_factor <= 0.0 || !growth_factor.is_finite() {
            return Err(Error::Msg(format!(
                "GradScaler::with_intervals requires a finite positive growth_factor, got {growth_factor}"
            )));
        }
        if backoff_factor <= 0.0 || !backoff_factor.is_finite() {
            return Err(Error::Msg(format!(
                "GradScaler::with_intervals requires a finite positive backoff_factor, got {backoff_factor}"
            )));
        }
        self.growth_factor = growth_factor;
        self.backoff_factor = backoff_factor;
        self.growth_interval = growth_interval.max(1);
        Ok(self)
    }

    pub fn scale_factor(&self) -> f64 {
        self.scale
    }

    /// Multiplies `loss` by the current dynamic scale factor (half-precision losses are scaled in float32, where the
    /// scale cannot overflow).
    pub fn scale(&self, loss: &Tensor) -> Result<Tensor> {
        if matches!(loss.dtype(), DType::F16 | DType::BF16) {
            return loss.to_dtype(DType::F32)?.mul_scalar(self.scale);
        }
        loss.mul_scalar(self.scale)
    }

    /// Divides the gradients of `params` by the current scale (in float32, whatever the gradient dtype) and reports
    /// whether any of them contains an infinity or a NaN. Gradients are left untouched when one does.
    pub fn unscale_(&self, params: &[Tensor]) -> Result<bool> {
        let inv = 1.0 / self.scale;
        let mut unscaled: Vec<(&Tensor, Tensor)> = Vec::new();
        let mut found_inf = false;
        no_grad(|| -> Result<()> {
            for p in params {
                if let Some(g) = p.grad() {
                    let g32 = g.to_dtype(DType::F32)?.mul_scalar(inv)?;
                    // any inf / NaN makes the sum non-finite; the sum of finite unscaled gradients cannot overflow f64
                    let total = g32.to_dtype(DType::F64)?.abs()?.sum()?.item::<f64>()?;
                    if !total.is_finite() {
                        found_inf = true;
                        break;
                    }
                    unscaled.push((p, g32.to_dtype(g.dtype())?));
                }
            }
            Ok(())
        })?;
        if !found_inf {
            for (p, g) in unscaled {
                p.set_grad(Some(g));
            }
        }
        Ok(found_inf)
    }

    /// Adjusts the scale after a step: backs off after an overflow, grows after `growth_interval` clean steps.
    pub fn update(&mut self, found_inf: bool) {
        if found_inf {
            self.scale = (self.scale * self.backoff_factor).max(1.0);
            self.growth_tracker = 0;
        } else {
            self.growth_tracker += 1;
            if self.growth_tracker >= self.growth_interval {
                // Clamp: an unbounded product reaches +inf (1e308 * 2), after
                // which every gradient unscales to zero and training stalls
                // silently instead of erroring.
                self.scale = (self.scale * self.growth_factor).min(f64::MAX);
                self.growth_tracker = 0;
            }
        }
    }

    /// Unscales gradients of `params`, checks for `NaN`/`Inf`, steps `opt` if finite,
    /// and updates the scale factor. Returns `Ok(true)` if the optimizer step was taken,
    /// or `Ok(false)` if skipped due to overflow.
    pub fn step_and_update(&mut self, opt: &mut dyn Optimizer, params: &[Tensor]) -> Result<bool> {
        let found_inf = self.unscale_(params)?;
        if found_inf {
            opt.zero_grad();
        } else {
            opt.step()?;
        }
        self.update(found_inf);
        Ok(!found_inf)
    }
}

/// Float32 master copies of low-precision (F16 / BF16) model parameters, the standard recipe for pure half-precision
/// training: the optimizer updates the master copies (build it over [`master_params`](Self::master_params)), whose
/// float32 precision keeps small updates from vanishing, and the results are rounded back into the model.
pub struct MasterParams {
    model: Vec<Tensor>,
    master: Vec<Tensor>,
}

impl MasterParams {
    pub fn new(model_params: &[Tensor]) -> Result<Self> {
        let master = model_params.iter().map(|p| Ok(p.detach().to_dtype(DType::F32)?.requires_grad_(true))).collect::<Result<Vec<_>>>()?;
        Ok(Self { model: model_params.to_vec(), master })
    }

    /// The float32 parameters to hand to the optimizer (and to [`GradScaler::step_and_update`]).
    pub fn master_params(&self) -> &[Tensor] {
        &self.master
    }

    /// Moves the (still scaled) half-precision gradients of the model onto the master copies, in float32.
    pub fn grads_to_master(&self) -> Result<()> {
        no_grad(|| -> Result<()> {
            for (p, m) in self.model.iter().zip(&self.master) {
                m.set_grad(p.grad().map(|g| g.to_dtype(DType::F32)).transpose()?);
            }
            Ok(())
        })
    }

    /// Rounds the updated master values back into the model parameters.
    pub fn sync_to_model(&self) -> Result<()> {
        no_grad(|| -> Result<()> {
            for (p, m) in self.model.iter().zip(&self.master) {
                p.assign(&m.detach().to_dtype(p.dtype())?)?;
            }
            Ok(())
        })
    }

    /// Clears the gradients of the model parameters.
    pub fn zero_grad(&self) {
        for p in &self.model {
            p.zero_grad();
        }
    }
}

/// Exponential Moving Average (`EMA`) of model parameters:
/// $\theta_{\text{shadow}} \leftarrow \text{decay} \cdot \theta_{\text{shadow}} + (1 - \text{decay}) \cdot \theta$.
pub struct EMA {
    pub decay: f64,
    pub shadow: Vec<Tensor>,
}

impl EMA {
    pub fn new(params: &[Tensor], decay: f64) -> Result<Self> {
        let shadow = no_grad(|| {
            params
                .iter()
                .map(|p| Ok(p.add_scalar(0.0)?.detach()))
                .collect::<Result<Vec<_>>>()
        })?;
        Ok(Self { decay, shadow })
    }

    /// Updates shadow parameters from the current live `params`.
    pub fn update(&mut self, params: &[Tensor]) -> Result<()> {
        let d = self.decay;
        if !d.is_finite() || !(0.0..=1.0).contains(&d) {
            return Err(Error::Msg(format!(
                "EMA::update requires decay in [0, 1], got {d}"
            )));
        }
        if params.len() != self.shadow.len() {
            return Err(Error::ShapeMismatch {
                op: "EMA::update",
                expected: vec![self.shadow.len()],
                got: vec![params.len()],
            });
        }
        no_grad(|| {
            for (s, p) in self.shadow.iter_mut().zip(params) {
                *s = s.mul_scalar(d)?.add(&p.detach().mul_scalar(1.0 - d)?)?;
            }
            Ok(())
        })
    }

    /// Copies the EMA shadow weights into `params` for evaluation, returning a backup
    /// of the live weights that can be restored via [`EMA::restore`].
    pub fn apply_shadow(&self, params: &[Tensor]) -> Result<Vec<Tensor>> {
        if params.len() != self.shadow.len() {
            return Err(Error::ShapeMismatch {
                op: "EMA::apply_shadow",
                expected: vec![self.shadow.len()],
                got: vec![params.len()],
            });
        }
        no_grad(|| {
            let mut backup = Vec::with_capacity(params.len());
            for (p, s) in params.iter().zip(&self.shadow) {
                backup.push(p.add_scalar(0.0)?.detach());
                p.assign(s)?;
            }
            Ok(backup)
        })
    }

    /// Restores live weights saved by [`EMA::apply_shadow`].
    pub fn restore(&self, params: &[Tensor], backup: &[Tensor]) -> Result<()> {
        if params.len() != backup.len() {
            return Err(Error::ShapeMismatch {
                op: "EMA::restore",
                expected: vec![params.len()],
                got: vec![backup.len()],
            });
        }
        no_grad(|| {
            for (p, b) in params.iter().zip(backup) {
                p.assign(b)?;
            }
            Ok(())
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::nn::{Linear, Module};
    use crate::optim::{Adam, Sgd};

    #[test]
    fn autocast_runs_products_in_low_precision_and_reductions_in_float32() {
        crate::rng::manual_seed(1);
        let (a, b) = (Tensor::randn(&[4, 8]).unwrap(), Tensor::randn(&[8, 3]).unwrap());
        for dt in [DType::BF16, DType::F16] {
            let y = autocast(dt, || a.matmul(&b).unwrap());
            assert_eq!(y.dtype(), dt);
            // identical to casting by hand
            let by_hand = a.to_dtype(dt).unwrap().matmul(&b.to_dtype(dt).unwrap()).unwrap();
            assert_eq!(y.to_dtype(DType::F32).unwrap().to_vec::<f32>().unwrap(), by_hand.to_dtype(DType::F32).unwrap().to_vec::<f32>().unwrap());
            // softmax / log_softmax / layer_norm / exp of a half tensor come back as float32
            autocast(dt, || {
                assert_eq!(y.softmax(-1).unwrap().dtype(), DType::F32);
                assert_eq!(y.log_softmax(-1).unwrap().dtype(), DType::F32);
                assert_eq!(y.exp().unwrap().dtype(), DType::F32);
                assert_eq!(crate::nn::functional_ext::layer_norm(&y, &[3], None, None, 1e-5).unwrap().dtype(), DType::F32);
            });
        }
        // outside autocast nothing changes
        assert_eq!(a.matmul(&b).unwrap().dtype(), DType::F32);
        // Linear: float32 parameters, half-precision output, float32 gradients
        let l = Linear::new(8, 3).unwrap();
        let out = autocast(DType::BF16, || l.forward(&a).unwrap());
        assert_eq!(out.dtype(), DType::BF16);
        out.to_dtype(DType::F32).unwrap().sum().unwrap().backward().unwrap();
        assert_eq!(l.weight.grad().unwrap().dtype(), DType::F32);
        assert!(l.weight.grad().unwrap().to_vec::<f32>().unwrap().iter().any(|g| *g != 0.0));
    }

    #[test]
    fn grad_scaler_skips_overflow_backs_off_and_grows() {
        let p = Tensor::from_slice(&[1.0f32, 2.0], &[2]).unwrap().requires_grad_(true);
        let mut opt = Sgd::new(vec![p.clone()], 0.1);
        let mut sc = GradScaler::new(1024.0).unwrap().with_intervals(2.0, 0.5, 2).unwrap();
        // clean step: scaled gradient 1024 * [1, 1] unscales to [1, 1]
        p.set_grad(Some(Tensor::from_slice(&[1024.0f32, 1024.0], &[2]).unwrap()));
        assert!(sc.step_and_update(&mut opt, std::slice::from_ref(&p)).unwrap());
        let v = p.to_vec::<f32>().unwrap();
        assert!((v[0] - 0.9).abs() < 1e-6 && (v[1] - 1.9).abs() < 1e-6);
        // overflow: the step is skipped, parameters stay, the scale halves
        p.set_grad(Some(Tensor::from_slice(&[f32::INFINITY, 1.0], &[2]).unwrap()));
        assert!(!sc.step_and_update(&mut opt, std::slice::from_ref(&p)).unwrap());
        assert_eq!(p.to_vec::<f32>().unwrap(), v);
        assert_eq!(sc.scale_factor(), 512.0);
        // NaN is an overflow too
        p.set_grad(Some(Tensor::from_slice(&[f32::NAN, 1.0], &[2]).unwrap()));
        assert!(!sc.step_and_update(&mut opt, std::slice::from_ref(&p)).unwrap());
        assert_eq!(sc.scale_factor(), 256.0);
        // two clean steps double the scale again
        for _ in 0..2 {
            p.set_grad(Some(Tensor::from_slice(&[256.0f32, 256.0], &[2]).unwrap()));
            assert!(sc.step_and_update(&mut opt, std::slice::from_ref(&p)).unwrap());
        }
        assert_eq!(sc.scale_factor(), 512.0);
        // a half-precision loss is scaled in float32 (65536 would overflow f16)
        let loss = Tensor::from_slice(&[2.0f32], &[1]).unwrap().to_dtype(DType::F16).unwrap();
        let s = GradScaler::new(65536.0).unwrap().scale(&loss).unwrap();
        assert_eq!(s.dtype(), DType::F32);
        assert_eq!(s.to_vec::<f32>().unwrap(), vec![131072.0]);
    }

    /// Data for `y = tanh(x w)` with a small MLP to fit.
    fn problem() -> (Tensor, Tensor) {
        crate::rng::manual_seed(11);
        let x = Tensor::randn(&[64, 8]).unwrap();
        let wt = Tensor::randn(&[8, 1]).unwrap().mul_scalar(0.5).unwrap();
        (x.clone(), x.matmul(&wt).unwrap().tanh().unwrap())
    }

    fn mlp() -> (Linear, Linear) {
        crate::rng::manual_seed(5);
        (Linear::new(8, 16).unwrap(), Linear::new(16, 1).unwrap())
    }

    fn loss_of(out: &Tensor, y: &Tensor) -> Tensor {
        out.to_dtype(DType::F32).unwrap().sub(y).unwrap().square().unwrap().mean().unwrap()
    }

    fn train_fp32(steps: usize) -> f64 {
        let (x, y) = problem();
        let (l1, l2) = mlp();
        let params: Vec<Tensor> = l1.parameters().into_iter().chain(l2.parameters()).collect();
        let mut opt = Adam::new(params, 0.02);
        let mut last = 0.0;
        for _ in 0..steps {
            opt.zero_grad();
            let loss = loss_of(&l2.forward(&l1.forward(&x).unwrap().relu().unwrap()).unwrap(), &y);
            loss.backward().unwrap();
            opt.step().unwrap();
            last = loss.item::<f32>().unwrap() as f64;
        }
        last
    }

    #[test]
    fn autocast_bf16_with_float32_weights_trains_like_float32() {
        let (x, y) = problem();
        let (l1, l2) = mlp();
        let params: Vec<Tensor> = l1.parameters().into_iter().chain(l2.parameters()).collect();
        let mut opt = Adam::new(params.clone(), 0.02);
        let mut scaler = GradScaler::new(1024.0).unwrap();
        let (mut first, mut last) = (0.0, 0.0);
        for step in 0..150 {
            opt.zero_grad();
            let loss = autocast(DType::BF16, || loss_of(&l2.forward(&l1.forward(&x).unwrap().relu().unwrap()).unwrap(), &y));
            scaler.scale(&loss).unwrap().backward().unwrap();
            assert!(scaler.step_and_update(&mut opt, &params).unwrap(), "no overflow expected");
            last = loss.item::<f32>().unwrap() as f64;
            if step == 0 {
                first = last;
            }
        }
        let reference = train_fp32(150);
        assert!(last < first * 0.2, "loss {first} -> {last}");
        assert!(last < reference * 1.5 + 0.01, "autocast loss {last} vs float32 {reference}");
    }

    #[test]
    fn pure_half_precision_model_with_master_weights_trains() {
        let (x, y) = problem();
        let reference = train_fp32(150);
        for dt in [DType::BF16, DType::F16] {
            let (l1, l2) = mlp();
            let half: Vec<Tensor> = l1.parameters().into_iter().chain(l2.parameters()).map(|p| p.detach().to_dtype(dt).unwrap().requires_grad_(true)).collect();
            let (w1, b1, w2, b2) = (half[0].clone(), half[1].clone(), half[2].clone(), half[3].clone());
            let master = MasterParams::new(&half).unwrap();
            let mut opt = Adam::new(master.master_params().to_vec(), 0.02);
            let mut scaler = GradScaler::new(512.0).unwrap();
            let xh = x.to_dtype(dt).unwrap();
            let (mut first, mut last) = (0.0, 0.0);
            for step in 0..150 {
                master.zero_grad();
                let h = xh.matmul(&w1.t().unwrap()).unwrap().add(&b1).unwrap().relu().unwrap();
                let out = h.matmul(&w2.t().unwrap()).unwrap().add(&b2).unwrap();
                let loss = loss_of(&out, &y);
                scaler.scale(&loss).unwrap().backward().unwrap();
                master.grads_to_master().unwrap();
                if scaler.step_and_update(&mut opt, master.master_params()).unwrap() {
                    master.sync_to_model().unwrap();
                }
                last = loss.item::<f32>().unwrap() as f64;
                if step == 0 {
                    first = last;
                }
            }
            assert!(w1.dtype() == dt);
            assert!(last < first * 0.3, "{dt:?}: loss {first} -> {last}");
            assert!(last < reference * 2.0 + 0.02, "{dt:?}: loss {last} vs float32 {reference}");
        }
    }

    #[test]
    fn loss_scaling_rescues_gradients_that_underflow_in_float16() {
        // d/dw of 1e-8 * sum(x * w) is 1e-8 * x, below the smallest float16 subnormal (6e-8): it rounds to zero
        let x = Tensor::from_slice(&[1.0f32, 2.0, 3.0], &[3]).unwrap().to_dtype(DType::F16).unwrap();
        let grad = |scale: f64| {
            let w = Tensor::ones(&[3], DType::F16, &crate::backend::Device::Cpu).unwrap().requires_grad_(true);
            let loss = w.mul(&x).unwrap().to_dtype(DType::F32).unwrap().sum().unwrap().mul_scalar(1e-8 * scale).unwrap();
            loss.backward().unwrap();
            w.grad().unwrap().to_dtype(DType::F32).unwrap().to_vec::<f32>().unwrap()
        };
        assert_eq!(grad(1.0), vec![0.0; 3], "unscaled float16 gradients flush to zero");
        let scaled = grad(65536.0);
        for (g, t) in scaled.iter().zip([1e-8f32, 2e-8, 3e-8]) {
            let r = g / 65536.0;
            assert!((r - t).abs() / t < 0.05, "{r} vs {t}");
        }
    }
}
