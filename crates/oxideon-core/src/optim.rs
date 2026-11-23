//! Optimizers (`Sgd`, `Adam`, `AdamW`, `RmsProp`), gradient clipping, and LR schedulers.

use crate::error::Result;
use crate::tensor::{Tensor, no_grad};

pub trait Optimizer {
    fn step(&mut self) -> Result<()>;
    fn zero_grad(&mut self);
    fn learning_rate(&self) -> f64;
    fn set_learning_rate(&mut self, lr: f64);

    /// Everything needed to resume training with this optimizer: moments, velocities, step counters and the
    /// learning rate (`torch.optim.Optimizer.state_dict`). Optimizers that do not implement it return an empty state.
    fn state_dict(&self) -> OptimizerState {
        OptimizerState::default()
    }

    /// Restores a state produced by [`state_dict`](Self::state_dict) of the same optimizer type over parameters of
    /// the same shapes.
    fn load_state_dict(&mut self, _state: &OptimizerState) -> Result<()> {
        Err(crate::error::Error::Msg(
            "this optimizer does not support load_state_dict".into(),
        ))
    }
}

/// Serializable optimizer state: tensors keyed `"<param index>.<name>"` and scalar hyper-state (learning rate,
/// step counters).
#[derive(Default, Clone)]
pub struct OptimizerState {
    pub tensors: std::collections::HashMap<String, Tensor>,
    pub scalars: std::collections::HashMap<String, f64>,
}

impl OptimizerState {
    /// Stores per-parameter optional tensors under `"<i>.<name>"`.
    pub fn put_tensors(&mut self, name: &str, slots: &[Option<Tensor>]) {
        for (i, s) in slots.iter().enumerate() {
            if let Some(t) = s {
                self.tensors.insert(format!("{i}.{name}"), t.detach());
            }
        }
    }

    /// Reads back what [`put_tensors`](Self::put_tensors) stored; slots without an entry become `None`.
    pub fn take_tensors(&self, name: &str, slots: &mut [Option<Tensor>]) -> Result<()> {
        for (i, slot) in slots.iter_mut().enumerate() {
            *slot = match self.tensors.get(&format!("{i}.{name}")) {
                Some(t) => Some(t.detach().contiguous()?.mul_scalar(1.0)?),
                None => None,
            };
        }
        Ok(())
    }

    /// Stores per-parameter scalars under `"<i>.<name>"`.
    pub fn put_scalars(&mut self, name: &str, values: impl IntoIterator<Item = f64>) {
        for (i, v) in values.into_iter().enumerate() {
            self.scalars.insert(format!("{i}.{name}"), v);
        }
    }

    /// Reads per-parameter scalars; a missing entry keeps the value already in `out`.
    pub fn take_scalars(&self, name: &str, out: &mut [f64]) {
        for (i, v) in out.iter_mut().enumerate() {
            if let Some(&s) = self.scalars.get(&format!("{i}.{name}")) {
                *v = s;
            }
        }
    }

    /// Writes the state to a `.safetensors` file (scalars become 1-element `f64` tensors under `"__scalar__."`).
    pub fn save(&self, path: impl AsRef<std::path::Path>) -> Result<()> {
        let mut all = self.tensors.clone();
        for (k, &v) in &self.scalars {
            all.insert(format!("__scalar__.{k}"), Tensor::from_slice(&[v], &[1])?);
        }
        crate::serialize::save_safetensors(&all, path)
    }

    pub fn load(path: impl AsRef<std::path::Path>) -> Result<Self> {
        let all = crate::serialize::load_safetensors(path)?;
        let mut state = Self::default();
        for (k, t) in all {
            if let Some(name) = k.strip_prefix("__scalar__.") {
                state.scalars.insert(name.to_string(), t.to_vec::<f64>()?[0]);
            } else {
                state.tensors.insert(k, t);
            }
        }
        Ok(state)
    }
}

// ---------------------------------------------------------------------------
// Gradient Clipping
// ---------------------------------------------------------------------------

/// Clips the global L2 norm of all parameter gradients to `max_norm`, returning
/// the original total norm.
pub fn clip_grad_norm(params: &[Tensor], max_norm: f64) -> Result<f64> {
    if max_norm.is_nan() || max_norm < 0.0 {
        return Err(crate::error::Error::Msg(format!(
            "clip_grad_norm: max_norm must be non-negative, got {max_norm}"
        )));
    }
    no_grad(|| {
        let mut total_sq = 0.0f64;
        // device-resident gradients: sum of squares on the device (one small download each, no f64 round trip of the data)
        let mut pending: Vec<Tensor> = Vec::new();
        for p in params {
            if let Some(g) = p.grad() {
                if let Some(sq) = g.storage().cuda_sum_squares(g.layout()) {
                    total_sq += sq;
                } else {
                    pending.push(g);
                }
            }
        }
        for g in pending {
            let sq = g.square()?.sum()?.item::<f64>()?;
            total_sq += sq;
        }
        let total_norm = total_sq.sqrt();
        if total_norm > max_norm {
            let scale = max_norm / (total_norm + 1e-6);
            for p in params {
                if let Some(g) = p.grad() {
                    p.set_grad(Some(g.mul_scalar(scale)?));
                }
            }
        }
        Ok(total_norm)
    })
}

/// Clips all parameter gradient elements to `[-clip_value, clip_value]`.
pub fn clip_grad_value(params: &[Tensor], clip_value: f64) -> Result<()> {
    no_grad(|| {
        for p in params {
            if let Some(g) = p.grad() {
                p.set_grad(Some(g.clamp(-clip_value, clip_value)?));
            }
        }
        Ok(())
    })
}

// ---------------------------------------------------------------------------
// SGD (with momentum & weight decay)
// ---------------------------------------------------------------------------
pub struct Sgd {
    params: Vec<Tensor>,
    lr: f64,
    momentum: f64,
    weight_decay: f64,
    dampening: f64,
    nesterov: bool,
    maximize: bool,
    velocity: Vec<Option<Tensor>>,
}

impl Sgd {
    pub fn new(params: Vec<Tensor>, lr: f64) -> Self {
        let velocity = vec![None; params.len()];
        Self {
            params,
            lr,
            momentum: 0.0,
            weight_decay: 0.0,
            dampening: 0.0,
            nesterov: false,
            maximize: false,
            velocity,
        }
    }

    /// Dampening applied to the gradient in the momentum buffer (`torch.optim.SGD(dampening=...)`).
    pub fn dampening(mut self, dampening: f64) -> Self {
        self.dampening = dampening;
        self
    }

    /// Nesterov momentum (requires `momentum > 0` and no dampening, as in PyTorch).
    pub fn nesterov(mut self, nesterov: bool) -> Self {
        self.nesterov = nesterov;
        self
    }

    /// Maximize the objective instead of minimizing it.
    pub fn maximize(mut self, maximize: bool) -> Self {
        self.maximize = maximize;
        self
    }

    pub fn momentum(mut self, momentum: f64) -> Self {
        self.momentum = momentum;
        self
    }

    pub fn weight_decay(mut self, weight_decay: f64) -> Self {
        self.weight_decay = weight_decay;
        self
    }
}

impl Optimizer for Sgd {
    fn step(&mut self) -> Result<()> {
        if self.nesterov && (self.momentum <= 0.0 || self.dampening != 0.0) {
            return Err(crate::error::Error::Msg("Sgd: Nesterov momentum requires momentum > 0 and zero dampening".into()));
        }
        no_grad(|| {
            for (p, vel) in self.params.iter().zip(&mut self.velocity) {
                let Some(mut g) = p.grad() else { continue };
                if self.maximize {
                    g = g.neg()?;
                }
                if self.weight_decay != 0.0 {
                    g = g.add(&p.mul_scalar(self.weight_decay)?)?;
                }
                let update = if self.momentum != 0.0 {
                    let v_new = match vel.take() {
                        Some(v) => v.mul_scalar(self.momentum)?.add(&g.mul_scalar(1.0 - self.dampening)?)?,
                        None => g.clone(),
                    };
                    *vel = Some(v_new.clone());
                    if self.nesterov { g.add(&v_new.mul_scalar(self.momentum)?)? } else { v_new }
                } else {
                    g
                };
                let new_p = p.sub(&update.mul_scalar(self.lr)?)?;
                p.assign(&new_p)?;
            }
            Ok(())
        })
    }

    fn zero_grad(&mut self) {
        self.params.iter().for_each(Tensor::zero_grad);
    }

    fn learning_rate(&self) -> f64 {
        self.lr
    }

    fn set_learning_rate(&mut self, lr: f64) {
        self.lr = lr;
    }

    fn state_dict(&self) -> OptimizerState {
        let mut s = OptimizerState::default();
        s.put_tensors("momentum_buffer", &self.velocity);
        s.scalars.insert("lr".into(), self.lr);
        s
    }

    fn load_state_dict(&mut self, state: &OptimizerState) -> Result<()> {
        state.take_tensors("momentum_buffer", &mut self.velocity)?;
        if let Some(&lr) = state.scalars.get("lr") {
            self.lr = lr;
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Adam & AdamW
// ---------------------------------------------------------------------------
pub struct Adam {
    params: Vec<Tensor>,
    lr: f64,
    beta1: f64,
    beta2: f64,
    eps: f64,
    weight_decay: f64,
    decoupled: bool,
    amsgrad: bool,
    maximize: bool,
    m: Vec<Option<Tensor>>,
    v: Vec<Option<Tensor>>,
    v_max: Vec<Option<Tensor>>,
    /// Per-parameter step counts (torch parity: only params with a gradient
    /// advance, so bias correction matches each param's own history).
    t: Vec<i32>,
}

impl Adam {
    pub fn new(params: Vec<Tensor>, lr: f64) -> Self {
        let n = params.len();
        Self {
            params,
            lr,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
            weight_decay: 0.0,
            decoupled: false,
            amsgrad: false,
            maximize: false,
            m: vec![None; n],
            v: vec![None; n],
            v_max: vec![None; n],
            t: vec![0; n],
        }
    }

    /// AMSGrad variant (Reddi et al. 2018): normalizes by the running maximum of the second moment.
    pub fn amsgrad(mut self, amsgrad: bool) -> Self {
        self.amsgrad = amsgrad;
        self
    }

    /// Maximize the objective instead of minimizing it.
    pub fn maximize(mut self, maximize: bool) -> Self {
        self.maximize = maximize;
        self
    }

    pub fn adamw(params: Vec<Tensor>, lr: f64, weight_decay: f64) -> Self {
        let mut opt = Self::new(params, lr);
        opt.weight_decay = weight_decay;
        opt.decoupled = true;
        opt
    }

    pub fn betas(mut self, beta1: f64, beta2: f64) -> Self {
        self.beta1 = beta1;
        self.beta2 = beta2;
        self
    }

    pub fn eps(mut self, eps: f64) -> Self {
        self.eps = eps;
        self
    }

    pub fn weight_decay(mut self, wd: f64) -> Self {
        self.weight_decay = wd;
        self
    }
}

impl Optimizer for Adam {
    fn step(&mut self) -> Result<()> {
        if !(0.0..1.0).contains(&self.beta1) || !(0.0..1.0).contains(&self.beta2) {
            return Err(crate::error::Error::Msg(format!(
                "Adam: betas must lie in [0, 1), got ({}, {})",
                self.beta1, self.beta2
            )));
        }
        no_grad(|| {
            for ((((p, m_slot), v_slot), t_slot), vmax_slot) in self
                .params
                .iter()
                .zip(&mut self.m)
                .zip(&mut self.v)
                .zip(&mut self.t)
                .zip(&mut self.v_max)
            {
                let Some(mut g) = p.grad() else { continue };
                if self.maximize {
                    g = g.neg()?;
                }
                *t_slot += 1;
                let bc1 = 1.0 - self.beta1.powi(*t_slot);
                let bc2 = 1.0 - self.beta2.powi(*t_slot);
                let on_cuda = matches!(p.device(), crate::device::Device::Cuda(_)) && g.device() == p.device();
                if !self.amsgrad
                    && !self.maximize
                    && p.dtype() == crate::dtype::DType::F32
                    && g.dtype() == crate::dtype::DType::F32
                    && ((p.device().is_cpu() && g.device().is_cpu()) || on_cuda)
                {
                    let m_buf = match m_slot {
                        Some(m) => m,
                        None => {
                            *m_slot = Some(Tensor::zeros(p.shape(), crate::dtype::DType::F32, &p.device())?);
                            m_slot.as_mut().unwrap()
                        }
                    };
                    let v_buf = match v_slot {
                        Some(v) => v,
                        None => {
                            *v_slot = Some(Tensor::zeros(p.shape(), crate::dtype::DType::F32, &p.device())?);
                            v_slot.as_mut().unwrap()
                        }
                    };
                    if p.storage().try_adam_step_f32(
                        p.layout(),
                        g.storage(),
                        g.layout(),
                        m_buf.storage(),
                        v_buf.storage(),
                        self.lr,
                        self.beta1,
                        self.beta2,
                        self.eps,
                        self.weight_decay,
                        self.decoupled,
                        bc1,
                        bc2,
                    ) {
                        continue;
                    }
                }
                let mut cur_p = p.clone();
                if self.weight_decay != 0.0 {
                    if self.decoupled {
                        cur_p = cur_p.mul_scalar(1.0 - self.lr * self.weight_decay)?;
                    } else {
                        g = g.add(&cur_p.mul_scalar(self.weight_decay)?)?;
                    }
                }
                let m_new = match m_slot.take() {
                    Some(m_prev) => m_prev
                        .mul_scalar(self.beta1)?
                        .add(&g.mul_scalar(1.0 - self.beta1)?)?,
                    None => g.mul_scalar(1.0 - self.beta1)?,
                };
                let v_new = match v_slot.take() {
                    Some(v_prev) => v_prev
                        .mul_scalar(self.beta2)?
                        .add(&g.square()?.mul_scalar(1.0 - self.beta2)?)?,
                    None => g.square()?.mul_scalar(1.0 - self.beta2)?,
                };
                let v_used = if self.amsgrad {
                    let mx = match vmax_slot.take() {
                        Some(prev) => prev.maximum(&v_new)?,
                        None => v_new.clone(),
                    };
                    *vmax_slot = Some(mx.clone());
                    mx
                } else {
                    v_new.clone()
                };
                let m_hat = m_new.div_scalar(bc1)?;
                let v_hat = v_used.div_scalar(bc2)?;
                let step = m_hat
                    .div(&v_hat.sqrt()?.add_scalar(self.eps)?)?
                    .mul_scalar(self.lr)?;
                let updated = cur_p.sub(&step)?;
                p.assign(&updated)?;
                *m_slot = Some(m_new);
                *v_slot = Some(v_new);
            }
            Ok(())
        })
    }

    fn zero_grad(&mut self) {
        self.params.iter().for_each(Tensor::zero_grad);
    }

    fn learning_rate(&self) -> f64 {
        self.lr
    }

    fn set_learning_rate(&mut self, lr: f64) {
        self.lr = lr;
    }

    fn state_dict(&self) -> OptimizerState {
        let mut s = OptimizerState::default();
        s.put_tensors("exp_avg", &self.m);
        s.put_tensors("exp_avg_sq", &self.v);
        if self.amsgrad {
            s.put_tensors("max_exp_avg_sq", &self.v_max);
        }
        s.put_scalars("step", self.t.iter().map(|&t| f64::from(t)));
        s.scalars.insert("lr".into(), self.lr);
        s
    }

    fn load_state_dict(&mut self, state: &OptimizerState) -> Result<()> {
        state.take_tensors("exp_avg", &mut self.m)?;
        state.take_tensors("exp_avg_sq", &mut self.v)?;
        if self.amsgrad {
            state.take_tensors("max_exp_avg_sq", &mut self.v_max)?;
        }
        let mut steps: Vec<f64> = self.t.iter().map(|&t| f64::from(t)).collect();
        state.take_scalars("step", &mut steps);
        self.t = steps.into_iter().map(|t| t as i32).collect();
        if let Some(&lr) = state.scalars.get("lr") {
            self.lr = lr;
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// RMSprop
// ---------------------------------------------------------------------------
pub struct RmsProp {
    params: Vec<Tensor>,
    lr: f64,
    alpha: f64,
    eps: f64,
    weight_decay: f64,
    momentum: f64,
    centered: bool,
    sq_avg: Vec<Option<Tensor>>,
    grad_avg: Vec<Option<Tensor>>,
    buf: Vec<Option<Tensor>>,
}

impl RmsProp {
    pub fn new(params: Vec<Tensor>, lr: f64) -> Self {
        let n = params.len();
        Self {
            params,
            lr,
            alpha: 0.99,
            eps: 1e-8,
            weight_decay: 0.0,
            momentum: 0.0,
            centered: false,
            sq_avg: vec![None; n],
            grad_avg: vec![None; n],
            buf: vec![None; n],
        }
    }

    /// Smoothing constant of the squared-gradient average (default 0.99).
    pub fn alpha(mut self, alpha: f64) -> Self {
        self.alpha = alpha;
        self
    }

    pub fn eps(mut self, eps: f64) -> Self {
        self.eps = eps;
        self
    }

    pub fn weight_decay(mut self, wd: f64) -> Self {
        self.weight_decay = wd;
        self
    }

    pub fn momentum(mut self, momentum: f64) -> Self {
        self.momentum = momentum;
        self
    }

    /// Normalize by the estimated variance of the gradient instead of its raw second moment.
    pub fn centered(mut self, centered: bool) -> Self {
        self.centered = centered;
        self
    }
}

impl Optimizer for RmsProp {
    fn step(&mut self) -> Result<()> {
        no_grad(|| {
            for i in 0..self.params.len() {
                let p = &self.params[i];
                let Some(mut g) = p.grad() else { continue };
                if self.weight_decay != 0.0 {
                    g = g.add(&p.mul_scalar(self.weight_decay)?)?;
                }
                let sq = match self.sq_avg[i].take() {
                    Some(prev) => prev.mul_scalar(self.alpha)?.add(&g.square()?.mul_scalar(1.0 - self.alpha)?)?,
                    None => g.square()?.mul_scalar(1.0 - self.alpha)?,
                };
                let avg = if self.centered {
                    let ga = match self.grad_avg[i].take() {
                        Some(prev) => prev.mul_scalar(self.alpha)?.add(&g.mul_scalar(1.0 - self.alpha)?)?,
                        None => g.mul_scalar(1.0 - self.alpha)?,
                    };
                    let var = sq.sub(&ga.square()?)?;
                    self.grad_avg[i] = Some(ga);
                    var.sqrt()?.add_scalar(self.eps)?
                } else {
                    sq.sqrt()?.add_scalar(self.eps)?
                };
                let dir = g.div(&avg)?;
                let update = if self.momentum > 0.0 {
                    let b = match self.buf[i].take() {
                        Some(prev) => prev.mul_scalar(self.momentum)?.add(&dir)?,
                        None => dir,
                    };
                    self.buf[i] = Some(b.clone());
                    b
                } else {
                    dir
                };
                p.assign(&p.sub(&update.mul_scalar(self.lr)?)?)?;
                self.sq_avg[i] = Some(sq);
            }
            Ok(())
        })
    }

    fn zero_grad(&mut self) {
        self.params.iter().for_each(Tensor::zero_grad);
    }

    fn learning_rate(&self) -> f64 {
        self.lr
    }

    fn set_learning_rate(&mut self, lr: f64) {
        self.lr = lr;
    }

    fn state_dict(&self) -> OptimizerState {
        let mut s = OptimizerState::default();
        s.put_tensors("square_avg", &self.sq_avg);
        if self.centered {
            s.put_tensors("grad_avg", &self.grad_avg);
        }
        if self.momentum > 0.0 {
            s.put_tensors("momentum_buffer", &self.buf);
        }
        s.scalars.insert("lr".into(), self.lr);
        s
    }

    fn load_state_dict(&mut self, state: &OptimizerState) -> Result<()> {
        state.take_tensors("square_avg", &mut self.sq_avg)?;
        if self.centered {
            state.take_tensors("grad_avg", &mut self.grad_avg)?;
        }
        if self.momentum > 0.0 {
            state.take_tensors("momentum_buffer", &mut self.buf)?;
        }
        if let Some(&lr) = state.scalars.get("lr") {
            self.lr = lr;
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Learning Rate Schedulers
// ---------------------------------------------------------------------------
pub struct StepLr {
    base_lr: f64,
    step_size: usize,
    gamma: f64,
    epoch: usize,
}

impl StepLr {
    pub fn new(base_lr: f64, step_size: usize, gamma: f64) -> Self {
        Self {
            base_lr,
            step_size: step_size.max(1),
            gamma,
            epoch: 0,
        }
    }

    pub fn step(&mut self, opt: &mut dyn Optimizer) -> f64 {
        self.epoch += 1;
        let k = (self.epoch / self.step_size) as i32;
        let lr = self.base_lr * self.gamma.powi(k);
        opt.set_learning_rate(lr);
        lr
    }
}

pub struct CosineAnnealingLr {
    base_lr: f64,
    min_lr: f64,
    t_max: usize,
    step_num: usize,
}

impl CosineAnnealingLr {
    pub fn new(base_lr: f64, t_max: usize, min_lr: f64) -> Self {
        Self {
            base_lr,
            min_lr,
            t_max: t_max.max(1),
            step_num: 0,
        }
    }

    pub fn step(&mut self, opt: &mut dyn Optimizer) -> f64 {
        // Torch parity: progress keeps growing past T_max (the schedule oscillates
        // instead of saturating at min_lr).
        self.step_num = self.step_num.saturating_add(1);
        let progress = (self.step_num as f64) / (self.t_max as f64);
        let cosine = 0.5 * (1.0 + (std::f64::consts::PI * progress).cos());
        let lr = self.min_lr + (self.base_lr - self.min_lr) * cosine;
        opt.set_learning_rate(lr);
        lr
    }
}

// ---------------------------------------------------------------------------
// Adagrad
// ---------------------------------------------------------------------------
pub struct Adagrad {
    params: Vec<Tensor>,
    lr: f64,
    lr_decay: f64,
    eps: f64,
    weight_decay: f64,
    initial_accumulator: f64,
    sum_sq: Vec<Option<Tensor>>,
    t: Vec<i32>,
}

impl Adagrad {
    pub fn new(params: Vec<Tensor>, lr: f64) -> Self {
        let n = params.len();
        Self {
            params,
            lr,
            lr_decay: 0.0,
            eps: 1e-10,
            weight_decay: 0.0,
            initial_accumulator: 0.0,
            sum_sq: vec![None; n],
            t: vec![0; n],
        }
    }

    pub fn weight_decay(mut self, wd: f64) -> Self {
        self.weight_decay = wd;
        self
    }

    /// Learning-rate decay: the step size is `lr / (1 + (t - 1) * lr_decay)`.
    pub fn lr_decay(mut self, lr_decay: f64) -> Self {
        self.lr_decay = lr_decay;
        self
    }

    pub fn eps(mut self, eps: f64) -> Self {
        self.eps = eps;
        self
    }

    /// Starting value of the squared-gradient accumulator.
    pub fn initial_accumulator_value(mut self, v: f64) -> Self {
        self.initial_accumulator = v;
        self
    }
}

impl Optimizer for Adagrad {
    fn step(&mut self) -> Result<()> {
        no_grad(|| {
            for i in 0..self.params.len() {
                let p = &self.params[i];
                let Some(mut g) = p.grad() else { continue };
                self.t[i] += 1;
                let clr = self.lr / (1.0 + f64::from(self.t[i] - 1) * self.lr_decay);
                if self.weight_decay != 0.0 {
                    g = g.add(&p.mul_scalar(self.weight_decay)?)?;
                }
                let prev = match self.sum_sq[i].take() {
                    Some(prev) => prev,
                    None => p.zeros_like()?.add_scalar(self.initial_accumulator)?,
                };
                let acc_new = prev.add(&g.square()?)?;
                let step = g.div(&acc_new.sqrt()?.add_scalar(self.eps)?)?.mul_scalar(clr)?;
                p.assign(&p.sub(&step)?)?;
                self.sum_sq[i] = Some(acc_new);
            }
            Ok(())
        })
    }

    fn zero_grad(&mut self) {
        self.params.iter().for_each(Tensor::zero_grad);
    }

    fn learning_rate(&self) -> f64 {
        self.lr
    }

    fn set_learning_rate(&mut self, lr: f64) {
        self.lr = lr;
    }

    fn state_dict(&self) -> OptimizerState {
        let mut s = OptimizerState::default();
        s.put_tensors("sum", &self.sum_sq);
        s.put_scalars("step", self.t.iter().map(|&t| f64::from(t)));
        s.scalars.insert("lr".into(), self.lr);
        s
    }

    fn load_state_dict(&mut self, state: &OptimizerState) -> Result<()> {
        state.take_tensors("sum", &mut self.sum_sq)?;
        let mut steps: Vec<f64> = self.t.iter().map(|&t| f64::from(t)).collect();
        state.take_scalars("step", &mut steps);
        self.t = steps.into_iter().map(|t| t as i32).collect();
        if let Some(&lr) = state.scalars.get("lr") {
            self.lr = lr;
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Lion (EvoLved Sign Momentum, Chen et al. 2023)
// ---------------------------------------------------------------------------
pub struct Lion {
    params: Vec<Tensor>,
    lr: f64,
    beta1: f64,
    beta2: f64,
    weight_decay: f64,
    exp_avg: Vec<Option<Tensor>>,
}

impl Lion {
    pub fn new(params: Vec<Tensor>, lr: f64) -> Self {
        let n = params.len();
        Self {
            params,
            lr,
            beta1: 0.9,
            beta2: 0.99,
            weight_decay: 0.0,
            exp_avg: vec![None; n],
        }
    }

    pub fn weight_decay(mut self, wd: f64) -> Self {
        self.weight_decay = wd;
        self
    }
}

impl Optimizer for Lion {
    fn step(&mut self) -> Result<()> {
        no_grad(|| {
            for (p, m_slot) in self.params.iter().zip(&mut self.exp_avg) {
                let Some(g) = p.grad() else { continue };
                let mut cur_p = p.clone();
                if self.weight_decay != 0.0 {
                    cur_p = cur_p.mul_scalar(1.0 - self.lr * self.weight_decay)?;
                }
                let m_prev = match m_slot.take() {
                    Some(m) => m,
                    None => g.zeros_like()?,
                };
                // update = sign(beta1 * m + (1 - beta1) * g)
                let interp = m_prev
                    .mul_scalar(self.beta1)?
                    .add(&g.mul_scalar(1.0 - self.beta1)?)?;
                let update = interp.sign()?.mul_scalar(self.lr)?;
                p.assign(&cur_p.sub(&update)?)?;

                // m_next = beta2 * m + (1 - beta2) * g
                let m_next = m_prev
                    .mul_scalar(self.beta2)?
                    .add(&g.mul_scalar(1.0 - self.beta2)?)?;
                *m_slot = Some(m_next);
            }
            Ok(())
        })
    }

    fn zero_grad(&mut self) {
        self.params.iter().for_each(Tensor::zero_grad);
    }

    fn learning_rate(&self) -> f64 {
        self.lr
    }

    fn set_learning_rate(&mut self, lr: f64) {
        self.lr = lr;
    }
}

// ---------------------------------------------------------------------------
// Additional Schedulers: LinearWarmupCosineLr, ExponentialLr, ReduceLROnPlateau
// ---------------------------------------------------------------------------

/// Linear warmup for `warmup_steps` followed by cosine decay to `min_lr` over `total_steps`.
pub struct LinearWarmupCosineLr {
    base_lr: f64,
    min_lr: f64,
    warmup_steps: usize,
    total_steps: usize,
    step_num: usize,
}

impl LinearWarmupCosineLr {
    pub fn new(base_lr: f64, warmup_steps: usize, total_steps: usize, min_lr: f64) -> Self {
        Self {
            base_lr,
            min_lr,
            warmup_steps,
            // Saturate: `warmup_steps + 1` overflowed on giant warmups.
            total_steps: total_steps.max(warmup_steps.saturating_add(1)),
            step_num: 0,
        }
    }

    pub fn step(&mut self, opt: &mut dyn Optimizer) -> f64 {
        self.step_num += 1;
        let lr = if self.step_num <= self.warmup_steps && self.warmup_steps > 0 {
            self.base_lr * (self.step_num as f64) / (self.warmup_steps as f64)
        } else {
            let decay_steps = (self.total_steps - self.warmup_steps).max(1);
            let cur = (self.step_num - self.warmup_steps).min(decay_steps);
            let progress = (cur as f64) / (decay_steps as f64);
            let cosine = 0.5 * (1.0 + (std::f64::consts::PI * progress).cos());
            self.min_lr + (self.base_lr - self.min_lr) * cosine
        };
        opt.set_learning_rate(lr);
        lr
    }
}

pub struct ExponentialLr {
    gamma: f64,
}

impl ExponentialLr {
    pub fn new(gamma: f64) -> Self {
        Self { gamma }
    }

    pub fn step(&mut self, opt: &mut dyn Optimizer) -> f64 {
        let lr = opt.learning_rate() * self.gamma;
        opt.set_learning_rate(lr);
        lr
    }
}

pub struct ReduceLROnPlateau {
    factor: f64,
    patience: usize,
    min_lr: f64,
    maximize: bool,
    threshold: f64,
    relative: bool,
    cooldown: usize,
    eps: f64,
    best_metric: f64,
    bad_epochs: usize,
    cooldown_counter: usize,
}

impl ReduceLROnPlateau {
    /// Defaults match `torch.optim.lr_scheduler.ReduceLROnPlateau(mode="min", threshold=1e-4, threshold_mode="rel",
    /// cooldown=0, eps=1e-8)`.
    pub fn new(factor: f64, patience: usize, min_lr: f64) -> Self {
        Self {
            factor,
            patience,
            min_lr,
            maximize: false,
            threshold: 1e-4,
            relative: true,
            cooldown: 0,
            eps: 1e-8,
            best_metric: f64::INFINITY,
            bad_epochs: 0,
            cooldown_counter: 0,
        }
    }

    /// Treat larger metric values as better (`mode="max"`).
    pub fn maximize(mut self, maximize: bool) -> Self {
        self.maximize = maximize;
        self.best_metric = if maximize { f64::NEG_INFINITY } else { f64::INFINITY };
        self
    }

    /// Minimum change that counts as an improvement; `relative` selects `threshold_mode="rel"` (default) or `"abs"`.
    pub fn threshold(mut self, threshold: f64, relative: bool) -> Self {
        self.threshold = threshold;
        self.relative = relative;
        self
    }

    /// Epochs to wait after a reduction before counting bad epochs again.
    pub fn cooldown(mut self, cooldown: usize) -> Self {
        self.cooldown = cooldown;
        self
    }

    /// Reductions smaller than `eps` are ignored.
    pub fn eps(mut self, eps: f64) -> Self {
        self.eps = eps;
        self
    }

    fn is_better(&self, metric: f64) -> bool {
        match (self.maximize, self.relative) {
            (false, true) => metric < self.best_metric * (1.0 - self.threshold),
            (false, false) => metric < self.best_metric - self.threshold,
            (true, true) => metric > self.best_metric * (1.0 + self.threshold),
            (true, false) => metric > self.best_metric + self.threshold,
        }
    }

    pub fn step(&mut self, metric: f64, opt: &mut dyn Optimizer) -> f64 {
        if self.is_better(metric) {
            self.best_metric = metric;
            self.bad_epochs = 0;
        } else {
            self.bad_epochs += 1;
        }
        if self.cooldown_counter > 0 {
            self.cooldown_counter -= 1;
            self.bad_epochs = 0;
        }
        if self.bad_epochs > self.patience {
            let old = opt.learning_rate();
            let new_lr = (old * self.factor).max(self.min_lr);
            if old - new_lr > self.eps {
                opt.set_learning_rate(new_lr);
            }
            self.cooldown_counter = self.cooldown;
            self.bad_epochs = 0;
        }
        opt.learning_rate()
    }
}

// ---------------------------------------------------------------------------
// Muon (Momentum Orthogonalized by Newton-Schulz, Jordan et al. 2024)
// ---------------------------------------------------------------------------

/// `Muon` optimizer: accumulates Nesterov momentum and orthogonalizes 2D weight updates
/// to the nearest semi-orthogonal matrix ($U V^T$) via a 5-step quintic Newton–Schulz
/// iteration without any SVD call, falling back to RMS-normalized momentum for 1D/ND tensors.
pub struct Muon {
    params: Vec<Tensor>,
    lr: f64,
    momentum: f64,
    weight_decay: f64,
    ns_steps: usize,
    bufs: Vec<Option<Tensor>>,
}

impl Muon {
    pub fn new(params: Vec<Tensor>, lr: f64) -> Self {
        let n = params.len();
        Self {
            params,
            lr,
            momentum: 0.95,
            weight_decay: 0.0,
            ns_steps: 5,
            bufs: vec![None; n],
        }
    }

    pub fn momentum(mut self, momentum: f64) -> Self {
        self.momentum = momentum;
        self
    }

    pub fn weight_decay(mut self, wd: f64) -> Self {
        self.weight_decay = wd;
        self
    }

    pub fn ns_steps(mut self, steps: usize) -> Self {
        self.ns_steps = steps.max(1);
        self
    }
}

/// Quintic Newton–Schulz iteration ($X_{k+1} = a X_k + b (X_k X_k^T) X_k + c (X_k X_k^T)^2 X_k$)
/// approximating the orthogonal polar factor $U V^T$ of a 2D matrix $G$.
pub fn newton_schulz_orthogonalize(g: &Tensor, steps: usize) -> Result<Tensor> {
    if g.ndim() != 2 {
        return Err(crate::error::Error::Msg(format!(
            "newton_schulz_orthogonalize: expected 2D input, got {:?}",
            g.shape()
        )));
    }
    let (r, c) = (g.shape()[0], g.shape()[1]);
    let transposed = r > c;
    let mut x = if transposed { g.t()? } else { g.clone() };

    // Normalize Frobenius norm so spectral norm <= 1
    let frob = x.square()?.sum()?.sqrt()?.add_scalar(1e-7)?;
    x = x.div(&frob)?;

    for step in 0..steps {
        let (a, b, c_coef) = if step == 0 && steps > 1 {
            // Step 0: high-slope quintic polynomial to boost small singular values
            (3.4445, -4.7750, 2.0315)
        } else {
            // Subsequent steps: exact fixed-point quintic polynomial (15/8, -5/4, 3/8) converging to 1.0
            (1.875, -1.25, 0.375)
        };
        let a_mat = x.matmul(&x.t()?)?; // [min(r,c), min(r,c)]
        let b_mat = a_mat
            .mul_scalar(b)?
            .add(&a_mat.matmul(&a_mat)?.mul_scalar(c_coef)?)?;
        x = x.mul_scalar(a)?.add(&b_mat.matmul(&x)?)?;
    }

    if transposed { x.t() } else { Ok(x) }
}

impl Optimizer for Muon {
    fn step(&mut self) -> Result<()> {
        no_grad(|| {
            for (p, b_slot) in self.params.iter().zip(&mut self.bufs) {
                let Some(g) = p.grad() else { continue };
                let buf = match b_slot.take() {
                    Some(prev) => prev.mul_scalar(self.momentum)?.add(&g)?,
                    None => g.clone(),
                };
                // Nesterov lookahead momentum: g + momentum * buf
                let u = g.add(&buf.mul_scalar(self.momentum)?)?;
                *b_slot = Some(buf);

                let orth = if p.ndim() == 2 {
                    let scale = (p.shape()[0].max(p.shape()[1]) as f64).sqrt() * 0.2;
                    newton_schulz_orthogonalize(&u, self.ns_steps)?.mul_scalar(scale)?
                } else {
                    let rms = u.square()?.mean()?.sqrt()?.add_scalar(1e-8)?;
                    u.div(&rms)?
                };

                let mut next_p = p.clone();
                if self.weight_decay != 0.0 {
                    next_p = next_p.mul_scalar(1.0 - self.lr * self.weight_decay)?;
                }
                p.assign(&next_p.sub(&orth.mul_scalar(self.lr)?)?)?;
            }
            Ok(())
        })
    }

    fn zero_grad(&mut self) {
        self.params.iter().for_each(Tensor::zero_grad);
    }

    fn learning_rate(&self) -> f64 {
        self.lr
    }

    fn set_learning_rate(&mut self, lr: f64) {
        self.lr = lr;
    }
}

// ---------------------------------------------------------------------------
// L-BFGS (Limited-Memory BFGS Quasi-Newton Optimizer, Nocedal 1980)
// ---------------------------------------------------------------------------

/// Limited-Memory BFGS (`LBFGS`) quasi-Newton optimizer maintaining a rolling window of
/// `history_size` curvature pairs $(s_k, y_k)$ and computing the inverse-Hessian search
/// direction via Nocedal's two-loop recursion.
pub struct LBFGS {
    params: Vec<Tensor>,
    lr: f64,
    history_size: usize,
    prev_flat_param: Option<Vec<f64>>,
    prev_flat_grad: Option<Vec<f64>>,
    s_history: Vec<Vec<f64>>,
    y_history: Vec<Vec<f64>>,
    rho_history: Vec<f64>,
}

impl LBFGS {
    pub fn new(params: Vec<Tensor>, lr: f64, history_size: usize) -> Self {
        Self {
            params,
            lr,
            history_size: history_size.max(1),
            prev_flat_param: None,
            prev_flat_grad: None,
            s_history: Vec::new(),
            y_history: Vec::new(),
            rho_history: Vec::new(),
        }
    }

    fn flatten_params(&self) -> Result<Vec<f64>> {
        let mut out = Vec::new();
        for p in &self.params {
            out.extend(p.to_vec::<f64>()?);
        }
        Ok(out)
    }

    fn flatten_grads(&self) -> Result<Vec<f64>> {
        let mut out = Vec::new();
        for p in &self.params {
            match p.grad() {
                Some(g) => out.extend(g.to_vec::<f64>()?),
                None => out.extend(std::iter::repeat_n(0.0, p.numel())),
            }
        }
        Ok(out)
    }

    fn assign_flat_params(&self, flat: &[f64]) -> Result<()> {
        let mut offset = 0usize;
        for p in &self.params {
            let n = p.numel();
            let slice = &flat[offset..offset + n];
            let next = Tensor::from_slice_on(slice, p.shape(), &p.device())?.to_dtype(p.dtype())?;
            p.assign(&next)?;
            offset += n;
        }
        Ok(())
    }
}

impl Optimizer for LBFGS {
    fn step(&mut self) -> Result<()> {
        no_grad(|| {
            let x = self.flatten_params()?;
            let g = self.flatten_grads()?;
            let n = x.len();
            if n == 0 {
                return Ok(());
            }
            // torch.optim.LBFGS(tolerance_grad=1e-7): already at a stationary point
            if g.iter().fold(0.0f64, |m, v| m.max(v.abs())) <= 1e-7 {
                return Ok(());
            }
            let first_iteration = self.prev_flat_param.is_none();

            if let (Some(x_prev), Some(g_prev)) =
                (self.prev_flat_param.take(), self.prev_flat_grad.take())
            {
                let s: Vec<f64> = x.iter().zip(&x_prev).map(|(a, b)| a - b).collect();
                let y: Vec<f64> = g.iter().zip(&g_prev).map(|(a, b)| a - b).collect();
                let ys: f64 = s.iter().zip(&y).map(|(a, b)| a * b).sum();
                if ys > 1e-10 {
                    if self.s_history.len() >= self.history_size {
                        self.s_history.remove(0);
                        self.y_history.remove(0);
                        self.rho_history.remove(0);
                    }
                    self.s_history.push(s);
                    self.y_history.push(y);
                    self.rho_history.push(1.0 / ys);
                }
            }

            // Two-loop recursion to compute r = H_k * g
            let mut q = g.clone();
            let m = self.s_history.len();
            let mut alphas = vec![0.0f64; m];

            for i in (0..m).rev() {
                let dot: f64 = self.s_history[i].iter().zip(&q).map(|(a, b)| a * b).sum();
                let alpha = self.rho_history[i] * dot;
                alphas[i] = alpha;
                for (qj, &yj) in q.iter_mut().zip(&self.y_history[i]) {
                    *qj -= alpha * yj;
                }
            }

            let gamma = if m > 0 {
                // H_diag = y.s / y.y (torch applies no clamp)
                let sy = 1.0 / self.rho_history[m - 1];
                let yy: f64 = self.y_history[m - 1].iter().map(|v| v * v).sum();
                sy / yy
            } else {
                1.0
            };

            let mut r: Vec<f64> = q.into_iter().map(|v| v * gamma).collect();
            for (i, &alpha_i) in alphas.iter().enumerate() {
                let dot: f64 = self.y_history[i].iter().zip(&r).map(|(a, b)| a * b).sum();
                let beta = self.rho_history[i] * dot;
                let diff = alpha_i - beta;
                for (rj, &sj) in r.iter_mut().zip(&self.s_history[i]) {
                    *rj += sj * diff;
                }
            }

            // torch.optim.LBFGS(tolerance_change=1e-9): a vanishing directional derivative ends the iteration
            let gtd = -g.iter().zip(&r).map(|(a, b)| a * b).sum::<f64>();
            if gtd > -1e-9 {
                self.prev_flat_param = Some(x);
                self.prev_flat_grad = Some(g);
                return Ok(());
            }
            // torch: the very first step is damped to t = min(1, 1 / |g|_1) * lr
            let t = if first_iteration { (1.0 / g.iter().map(|v| v.abs()).sum::<f64>()).min(1.0) * self.lr } else { self.lr };
            let x_next: Vec<f64> = x.iter().zip(&r).map(|(xi, ri)| xi - t * ri).collect();
            self.prev_flat_param = Some(x);
            self.prev_flat_grad = Some(g);
            self.assign_flat_params(&x_next)
        })
    }

    fn zero_grad(&mut self) {
        self.params.iter().for_each(Tensor::zero_grad);
    }

    fn learning_rate(&self) -> f64 {
        self.lr
    }

    fn set_learning_rate(&mut self, lr: f64) {
        self.lr = lr;
    }
}

// ---------------------------------------------------------------------------
// Schedule-Free AdamW (Defazio et al., NeurIPS 2024) & Lookahead (Zhang et al., NeurIPS 2019)
// ---------------------------------------------------------------------------

/// **Schedule-Free `AdamW`** (Defazio et al., 2024):
/// Eliminates the need for a predetermined learning-rate schedule ($T_{\max}$) by maintaining
/// a primal sequence $z_t$ and an online Polyak-averaged sequence $x_t$, while evaluating
/// gradients at the interpolated point $y_t = (1 - \beta_1) z_t + \beta_1 x_t$.
pub struct ScheduleFreeAdamW {
    params: Vec<Tensor>,
    lr: f64,
    betas: (f64, f64),
    eps: f64,
    weight_decay: f64,
    step_count: usize,
    z_states: Vec<Tensor>,
    x_states: Vec<Tensor>,
    v_states: Vec<Tensor>,
    in_eval_mode: bool,
}

impl ScheduleFreeAdamW {
    pub fn new(params: Vec<Tensor>, lr: f64, weight_decay: f64) -> Self {
        let mut z_states = Vec::with_capacity(params.len());
        let mut x_states = Vec::with_capacity(params.len());
        let mut v_states = Vec::with_capacity(params.len());
        no_grad(|| {
            for p in &params {
                z_states.push(p.add_scalar(0.0).expect("clone z").detach());
                x_states.push(p.add_scalar(0.0).expect("clone x").detach());
                v_states.push(p.zeros_like().expect("zeros v").detach());
            }
        });
        Self {
            params,
            lr,
            betas: (0.9, 0.999),
            eps: 1e-8,
            weight_decay,
            step_count: 0,
            z_states,
            x_states,
            v_states,
            in_eval_mode: false,
        }
    }

    /// Switches parameters to the averaged sequence $x_t$ for validation/inference.
    pub fn eval_mode(&mut self) -> Result<()> {
        if !self.in_eval_mode {
            no_grad(|| {
                for (p, x) in self.params.iter().zip(&self.x_states) {
                    p.assign(x)?;
                }
                Ok(())
            })?;
            self.in_eval_mode = true;
        }
        Ok(())
    }

    /// Switches parameters back to the interpolated point $y_t = (1 - \beta_1) z_t + \beta_1 x_t$ for training.
    pub fn train_mode(&mut self) -> Result<()> {
        if self.in_eval_mode {
            let b1 = self.betas.0;
            no_grad(|| {
                for ((p, z), x) in self.params.iter().zip(&self.z_states).zip(&self.x_states) {
                    let y = z.mul_scalar(1.0 - b1)?.add(&x.mul_scalar(b1)?)?;
                    p.assign(&y)?;
                }
                Ok(())
            })?;
            self.in_eval_mode = false;
        }
        Ok(())
    }
}

impl Optimizer for ScheduleFreeAdamW {
    fn step(&mut self) -> Result<()> {
        self.train_mode()?;
        self.step_count += 1;
        let t = self.step_count;
        let (b1, b2) = self.betas;
        let bc2 = 1.0 - b2.powi(t as i32);
        let c_t = 1.0 / (t as f64);

        no_grad(|| {
            for i in 0..self.params.len() {
                let Some(g) = self.params[i].grad() else {
                    continue;
                };
                self.v_states[i] = self.v_states[i]
                    .mul_scalar(b2)?
                    .add(&g.square()?.mul_scalar(1.0 - b2)?)?;
                let v_hat = self.v_states[i].div_scalar(bc2)?;
                let denom = v_hat.sqrt()?.add_scalar(self.eps)?;
                let mut d = g.div(&denom)?;
                if self.weight_decay > 0.0 {
                    d = d.add(&self.params[i].mul_scalar(self.weight_decay)?)?;
                }

                // Primal step: z_{t+1} = z_t - lr * d_t
                self.z_states[i] = self.z_states[i].sub(&d.mul_scalar(self.lr)?)?;
                // Polyak average: x_{t+1} = (1 - c_t) x_t + c_t z_{t+1}
                self.x_states[i] = self.x_states[i]
                    .mul_scalar(1.0 - c_t)?
                    .add(&self.z_states[i].mul_scalar(c_t)?)?;
                // Next gradient evaluation point: y_{t+1} = (1 - b1) z_{t+1} + b1 x_{t+1}
                let y_next = self.z_states[i]
                    .mul_scalar(1.0 - b1)?
                    .add(&self.x_states[i].mul_scalar(b1)?)?;
                self.params[i].assign(&y_next)?;
            }
            Ok(())
        })
    }

    fn zero_grad(&mut self) {
        self.params.iter().for_each(Tensor::zero_grad);
    }

    fn learning_rate(&self) -> f64 {
        self.lr
    }

    fn set_learning_rate(&mut self, lr: f64) {
        self.lr = lr;
    }
}

/// **Lookahead** wrapper optimizer (Zhang et al., 2019):
/// Wraps any fast inner [`Optimizer`] $O$ and every $k$ steps interpolates the slow weights
/// $\phi \leftarrow \phi + \alpha (\theta - \phi)$ and resets $\theta \leftarrow \phi$.
pub struct Lookahead<O: Optimizer> {
    pub inner: O,
    pub params: Vec<Tensor>,
    pub slow_weights: Vec<Tensor>,
    pub k_steps: usize,
    pub alpha: f64,
    pub step_counter: usize,
}

impl<O: Optimizer> Lookahead<O> {
    pub fn new(inner: O, params: Vec<Tensor>, k_steps: usize, alpha: f64) -> Self {
        let slow_weights = no_grad(|| {
            params
                .iter()
                .map(|p| p.add_scalar(0.0).expect("Lookahead clone").detach())
                .collect()
        });
        Self {
            inner,
            params,
            slow_weights,
            k_steps: k_steps.max(1),
            alpha,
            step_counter: 0,
        }
    }
}

impl<O: Optimizer> Optimizer for Lookahead<O> {
    fn step(&mut self) -> Result<()> {
        self.inner.step()?;
        self.step_counter += 1;
        // `k_steps` is pub: a tampered 0 divided by zero below.
        if self.step_counter % self.k_steps.max(1) == 0 {
            let a = self.alpha;
            no_grad(|| {
                for (p, phi) in self.params.iter().zip(&mut self.slow_weights) {
                    let next_phi = phi.add(&p.sub(phi)?.mul_scalar(a)?)?.detach();
                    p.assign(&next_phi)?;
                    *phi = next_phi;
                }
                Ok(())
            })?;
        }
        Ok(())
    }

    fn zero_grad(&mut self) {
        self.inner.zero_grad();
    }

    fn learning_rate(&self) -> f64 {
        self.inner.learning_rate()
    }

    fn set_learning_rate(&mut self, lr: f64) {
        self.inner.set_learning_rate(lr);
    }
}

/// SGDR: Cosine Annealing with Warm Restarts (`torch.optim.lr_scheduler.CosineAnnealingWarmRestarts`,
/// Loshchilov & Hutter, ICLR 2017).
pub struct CosineAnnealingWarmRestarts {
    pub base_lr: f64,
    pub eta_min: f64,
    pub t_0: usize,
    pub t_mult: usize,
    pub t_cur: usize,
    pub t_i: usize,
}

impl CosineAnnealingWarmRestarts {
    pub fn new(base_lr: f64, t_0: usize, t_mult: usize, eta_min: f64) -> Self {
        let t_0 = t_0.max(1);
        Self {
            base_lr,
            eta_min,
            t_0,
            t_mult: t_mult.max(1),
            t_cur: 0,
            t_i: t_0,
        }
    }

    pub fn current_lr(&self) -> f64 {
        // Fields are pub: guard against a zeroed t_i (division would yield NaN lr).
        let t_i = (self.t_i as f64).max(1.0);
        let cos_factor = 0.5 * (1.0 + (std::f64::consts::PI * (self.t_cur as f64) / t_i).cos());
        self.eta_min + (self.base_lr - self.eta_min) * cos_factor
    }

    pub fn step<O: Optimizer>(&mut self, optimizer: &mut O) -> f64 {
        self.t_cur = self.t_cur.saturating_add(1);
        if self.t_cur >= self.t_i {
            self.t_cur = 0;
            self.t_i = self.t_i.saturating_mul(self.t_mult.max(1));
        }
        let lr = self.current_lr();
        optimizer.set_learning_rate(lr);
        lr
    }
}

/// Super-Convergence 1-Cycle Learning Rate Policy (`torch.optim.lr_scheduler.OneCycleLR`,
/// Smith & Topin, 2019). Linearly warms up from `initial_lr = max_lr / div_factor` to `max_lr`
/// over `pct_start * total_steps`, then cosine-anneals to `final_lr = initial_lr / final_div_factor`.
pub struct OneCycleLR {
    pub max_lr: f64,
    pub total_steps: usize,
    pub pct_start: f64,
    pub div_factor: f64,
    pub final_div_factor: f64,
    pub current_step: usize,
}

impl OneCycleLR {
    pub fn new(
        max_lr: f64,
        total_steps: usize,
        pct_start: f64,
        div_factor: f64,
        final_div_factor: f64,
    ) -> Self {
        Self {
            max_lr,
            total_steps: total_steps.max(2),
            pct_start: pct_start.clamp(0.01, 0.99),
            div_factor: div_factor.max(1.0),
            final_div_factor: final_div_factor.max(1.0),
            current_step: 0,
        }
    }

    pub fn current_lr(&self) -> f64 {
        // Torch parity: the ramp peak is reached after `up = pct*total - 1` steps and
        // the cosine tail spans `down = total - up - 1` steps; `step` panics past
        // total_steps, mirroring torch's ValueError.
        let initial_lr = self.max_lr / self.div_factor;
        let final_lr = initial_lr / self.final_div_factor;
        let up = ((self.total_steps as f64) * self.pct_start - 1.0).max(1.0);
        let down = (self.total_steps as f64 - up - 1.0).max(1.0);
        let step = self.current_step as f64;
        if step <= up {
            // Cosine ramp (torch default anneal_strategy='cos'): linear only
            // matches at the midpoint, quarter points discriminate.
            initial_lr
                + (self.max_lr - initial_lr) * (1.0 - (std::f64::consts::PI * step / up).cos())
                    / 2.0
        } else {
            let pct = (step - up) / down;
            let cos_factor = 0.5 * (1.0 + (std::f64::consts::PI * pct).cos());
            final_lr + (self.max_lr - final_lr) * cos_factor
        }
    }

    pub fn step<O: Optimizer>(&mut self, optimizer: &mut O) -> f64 {
        let next = self.current_step.saturating_add(1);
        assert!(
            next <= self.total_steps,
            "Tried to step {next} times. The specified number of total steps is {}",
            self.total_steps
        );
        self.current_step = next;
        let lr = self.current_lr();
        optimizer.set_learning_rate(lr);
        lr
    }
}

// ---------------------------------------------------------------------------
// Riemannian Adam on Smooth Manifolds
// ---------------------------------------------------------------------------

/// Manifold geometry for constrained parameter optimization.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ManifoldType {
    /// Unit hypersphere: $\|x\|_2 = 1$.
    Sphere,
    /// Poincaré ball hyperbolic space with curvature $c > 0$ scaled by 1000: $\|x\|_2 < 1 / \sqrt{c}$.
    PoincareBall { c_x1000: u32 },
    /// Stiefel manifold of orthonormal frames: $X^T X = I_p$.
    Stiefel { p: usize },
}

impl ManifoldType {
    /// Projects an ambient Euclidean gradient $G$ at point $x$ onto the tangent space $T_x \mathcal{M}$.
    pub fn project_tangent(&self, x: &Tensor, grad: &Tensor) -> Result<Tensor> {
        match self {
            Self::Sphere => {
                let x_vec = x.to_vec::<f64>()?;
                let g_vec = grad.to_vec::<f64>()?;
                let inner: f64 = x_vec.iter().zip(&g_vec).map(|(&a, &b)| a * b).sum();
                let proj: Vec<f64> = g_vec
                    .iter()
                    .zip(&x_vec)
                    .map(|(&g, &xx)| g - inner * xx)
                    .collect();
                Tensor::from_vec(proj, x.shape())?.to_dtype(x.dtype())
            }
            Self::PoincareBall { c_x1000 } => {
                let c = (*c_x1000 as f64) / 1000.0;
                let x_vec = x.to_vec::<f64>()?;
                let g_vec = grad.to_vec::<f64>()?;
                let norm_sq: f64 = x_vec.iter().map(|&a| a * a).sum();
                let lambda = 2.0 / (1.0 - c * norm_sq).max(1e-6);
                let inv_lambda_sq = 1.0 / (lambda * lambda);
                let r_grad: Vec<f64> = g_vec.iter().map(|&g| g * inv_lambda_sq).collect();
                Tensor::from_vec(r_grad, x.shape())?.to_dtype(x.dtype())
            }
            Self::Stiefel { p: _ } => {
                let xt = x.t()?;
                let xt_g = xt.matmul(grad)?;
                let gt_x = xt_g.t()?;
                let sym = xt_g.add(&gt_x)?.mul_scalar(0.5)?;
                let x_sym = x.matmul(&sym)?;
                grad.sub(&x_sym)
            }
        }
    }

    /// Retracts a tangent vector $\xi \in T_x \mathcal{M}$ back onto the manifold $\mathcal{M}$ from base point $x$.
    pub fn retract(&self, x: &Tensor, xi: &Tensor) -> Result<Tensor> {
        match self {
            Self::Sphere => {
                let x_vec = x.to_vec::<f64>()?;
                let xi_vec = xi.to_vec::<f64>()?;
                let mut sum: Vec<f64> = x_vec.iter().zip(&xi_vec).map(|(&a, &b)| a + b).collect();
                let norm: f64 = sum.iter().map(|&a| a * a).sum::<f64>().sqrt().max(1e-12);
                for v in &mut sum {
                    *v /= norm;
                }
                Tensor::from_vec(sum, x.shape())?.to_dtype(x.dtype())
            }
            Self::PoincareBall { c_x1000 } => {
                let c = (*c_x1000 as f64) / 1000.0;
                let x_vec = x.to_vec::<f64>()?;
                let v_vec = xi.to_vec::<f64>()?;
                let x_sq: f64 = x_vec.iter().map(|&a| a * a).sum();
                let v_sq: f64 = v_vec.iter().map(|&a| a * a).sum();
                let inner: f64 = x_vec.iter().zip(&v_vec).map(|(&a, &b)| a * b).sum();

                let denom = 1.0 + 2.0 * c * inner + c * c * x_sq * v_sq;
                let alpha = 1.0 + 2.0 * c * inner + c * v_sq;
                let beta = 1.0 - c * x_sq;

                let mut next: Vec<f64> = x_vec
                    .iter()
                    .zip(&v_vec)
                    .map(|(&xx, &vv)| (alpha * xx + beta * vv) / denom.max(1e-12))
                    .collect();

                let max_norm = (1.0 - 1e-5) / c.sqrt();
                let cur_norm: f64 = next.iter().map(|&a| a * a).sum::<f64>().sqrt();
                if cur_norm > max_norm {
                    let scale = max_norm / cur_norm;
                    for v in &mut next {
                        *v *= scale;
                    }
                }
                Tensor::from_vec(next, x.shape())?.to_dtype(x.dtype())
            }
            Self::Stiefel { p: _ } => {
                let cand = x.add(xi)?;
                if cand.ndim() != 2 {
                    return Err(crate::error::Error::Msg(format!(
                        "Stiefel retract: expected 2D point, got {:?}",
                        cand.shape()
                    )));
                }
                let shape = cand.shape();
                let (n, p) = (shape[0], shape[1]);
                if n == 0 {
                    // QR of a 0-row matrix is the empty 0xP frame (torch parity);
                    // the column loop below would index an empty column.
                    return Tensor::from_vec(Vec::<f64>::new(), &[0, p])?.to_dtype(x.dtype());
                }
                let cand_vec = cand.to_vec::<f64>()?;
                let mut q_mat = vec![0.0f64; n * p];

                for j in 0..p {
                    let mut col = vec![0.0f64; n];
                    for i in 0..n {
                        col[i] = cand_vec[i * p + j];
                    }
                    for prev in 0..j {
                        let mut dot = 0.0f64;
                        for i in 0..n {
                            dot += col[i] * q_mat[i * p + prev];
                        }
                        for i in 0..n {
                            col[i] -= dot * q_mat[i * p + prev];
                        }
                    }
                    // Deterministic QR sign convention: the largest-magnitude
                    // entry of each column is positive (no random sign flips).
                    let mut imax = 0;
                    for i in 1..n {
                        if col[i].abs() > col[imax].abs() {
                            imax = i;
                        }
                    }
                    if col[imax] < 0.0 {
                        for v in col.iter_mut() {
                            *v = -*v;
                        }
                    }
                    let norm = col.iter().map(|&a| a * a).sum::<f64>().sqrt().max(1e-12);
                    for i in 0..n {
                        q_mat[i * p + j] = col[i] / norm;
                    }
                }
                Tensor::from_vec(q_mat, &[n, p])?.to_dtype(x.dtype())
            }
        }
    }
}

/// Riemannian Adam Optimizer on smooth manifolds.
pub struct RiemannianAdam {
    pub params: Vec<Tensor>,
    pub manifolds: Vec<ManifoldType>,
    pub lr: f64,
    pub beta1: f64,
    pub beta2: f64,
    pub eps: f64,
    /// Per-parameter step counts: only params with a gradient advance, so bias
    /// correction matches each param's own history (same rule as [`Adam`]).
    pub t: Vec<i32>,
    pub m: Vec<Option<Tensor>>,
    pub v: Vec<Option<Tensor>>,
}

impl RiemannianAdam {
    pub fn new(params: Vec<Tensor>, manifolds: Vec<ManifoldType>, lr: f64) -> Result<Self> {
        if params.len() != manifolds.len() {
            return Err(crate::error::Error::Msg(format!(
                "Mismatch between params len {} and manifolds len {}",
                params.len(),
                manifolds.len()
            )));
        }
        let len = params.len();
        Ok(Self {
            params,
            manifolds,
            lr,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
            t: vec![0; len],
            m: vec![None; len],
            v: vec![None; len],
        })
    }

    pub fn learning_rate(&self) -> f64 {
        self.lr
    }

    pub fn set_learning_rate(&mut self, lr: f64) {
        self.lr = lr;
    }

    pub fn step(&mut self) -> Result<()> {
        // All vecs are pub: re-check agreement (a shortened vec indexed out
        // of bounds below) and saturate the per-param counters.
        if self.manifolds.len() != self.params.len()
            || self.m.len() != self.params.len()
            || self.v.len() != self.params.len()
            || self.t.len() != self.params.len()
        {
            return Err(crate::error::Error::Msg(format!(
                "RiemannianAdam::step: params/manifolds/m/v/t length mismatch ({} vs {} vs {} vs {} vs {})",
                self.params.len(),
                self.manifolds.len(),
                self.m.len(),
                self.v.len(),
                self.t.len()
            )));
        }

        no_grad(|| {
            for idx in 0..self.params.len() {
                let p = &self.params[idx];
                let manifold = self.manifolds[idx];
                let Some(euc_grad) = p.grad() else { continue };
                self.t[idx] = self.t[idx].saturating_add(1);
                let bc1 = 1.0 - self.beta1.powi(self.t[idx]);
                let bc2 = 1.0 - self.beta2.powi(self.t[idx]);

                // 1. Project Euclidean gradient to Riemannian gradient
                let r_grad = manifold.project_tangent(p, &euc_grad)?;

                // 2. Transport previous first moment to current tangent space
                let m_prev_transported = match &self.m[idx] {
                    Some(m_old) => Some(manifold.project_tangent(p, m_old)?),
                    None => None,
                };

                let m_new = match m_prev_transported {
                    Some(m_t) => m_t
                        .mul_scalar(self.beta1)?
                        .add(&r_grad.mul_scalar(1.0 - self.beta1)?)?,
                    None => r_grad.mul_scalar(1.0 - self.beta1)?,
                };

                let v_new = match self.v[idx].take() {
                    Some(v_old) => v_old
                        .mul_scalar(self.beta2)?
                        .add(&r_grad.square()?.mul_scalar(1.0 - self.beta2)?)?,
                    None => r_grad.square()?.mul_scalar(1.0 - self.beta2)?,
                };

                // 3. Bias correction and search direction
                let m_hat = m_new.div_scalar(bc1)?;
                let v_hat = v_new.div_scalar(bc2)?;
                let step_dir = m_hat
                    .div(&v_hat.sqrt()?.add_scalar(self.eps)?)?
                    .mul_scalar(-self.lr)?;

                // Ensure step direction is strictly tangent
                let xi = manifold.project_tangent(p, &step_dir)?;

                // 4. Retract back onto the manifold
                let p_next = manifold.retract(p, &xi)?;

                // 5. In-place parameter update
                p.assign(&p_next)?;

                // 6. Store moments
                let m_next = manifold.project_tangent(&p_next, &m_new)?;
                self.m[idx] = Some(m_next);
                self.v[idx] = Some(v_new);
            }
            Ok(())
        })
    }

    pub fn zero_grad(&mut self) {
        self.params.iter().for_each(Tensor::zero_grad);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_close(got: &[f64], want: &[f64], tol: f64, what: &str) {
        assert_eq!(got.len(), want.len(), "{what}: length");
        for (i, (&g, &w)) in got.iter().zip(want.iter()).enumerate() {
            assert!((g - w).abs() <= tol, "{what} element {i}: {g} vs {w}");
        }
    }

    fn grad_param(vals: &[f64], grads: &[f64]) -> Tensor {
        let p = Tensor::from_slice(vals, &[vals.len()])
            .unwrap()
            .requires_grad_(true);
        p.set_grad(Some(Tensor::from_slice(grads, &[grads.len()]).unwrap()));
        p
    }

    #[test]
    fn sgd_nesterov_dampening_maximize_and_adam_amsgrad() {
        // two SGD steps with constant gradient 1: buf1 = 1, buf2 = 0.9*1 + 1 = 1.9; nesterov step uses g + 0.9*buf
        let p = grad_param(&[0.0], &[1.0]);
        let mut o = Sgd::new(vec![p.clone()], 0.1).momentum(0.9).nesterov(true);
        o.step().unwrap();
        o.step().unwrap();
        assert_close(&p.to_vec::<f64>().unwrap(), &[-(0.1 * 1.9 + 0.1 * (1.0 + 0.9 * 1.9))], 1e-6, "nesterov");
        // dampening 0.5: buf2 = 0.9*1 + 0.5*1 = 1.4
        let p = grad_param(&[0.0], &[1.0]);
        let mut o = Sgd::new(vec![p.clone()], 0.1).momentum(0.9).dampening(0.5);
        o.step().unwrap();
        o.step().unwrap();
        assert_close(&p.to_vec::<f64>().unwrap(), &[-(0.1 * 1.0 + 0.1 * 1.4)], 1e-6, "dampening");
        // maximize flips the sign; Nesterov without momentum is an error
        let p = grad_param(&[0.0], &[1.0]);
        Sgd::new(vec![p.clone()], 0.1).maximize(true).step().unwrap();
        assert_close(&p.to_vec::<f64>().unwrap(), &[0.1], 1e-6, "maximize");
        assert!(Sgd::new(vec![p.clone()], 0.1).nesterov(true).step().is_err());
        // AMSGrad keeps the running maximum of the second moment: a large then small gradient shrinks vanilla Adam's
        // denominator but not AMSGrad's, so AMSGrad takes the smaller second step.
        let (pa, pb) = (grad_param(&[0.0], &[10.0]), grad_param(&[0.0], &[10.0]));
        let mut a = Adam::new(vec![pa.clone()], 0.1);
        let mut b = Adam::new(vec![pb.clone()], 0.1).amsgrad(true);
        a.step().unwrap();
        b.step().unwrap();
        let (a1, b1) = (pa.to_vec::<f64>().unwrap()[0], pb.to_vec::<f64>().unwrap()[0]);
        pa.set_grad(Some(Tensor::from_slice(&[0.1], &[1]).unwrap()));
        pb.set_grad(Some(Tensor::from_slice(&[0.1], &[1]).unwrap()));
        a.step().unwrap();
        b.step().unwrap();
        let (da, db) = ((pa.to_vec::<f64>().unwrap()[0] - a1).abs(), (pb.to_vec::<f64>().unwrap()[0] - b1).abs());
        assert!(db < da, "amsgrad step {db} should be smaller than adam step {da}");
    }

    fn dummy_opt(lr: f64) -> Sgd {
        Sgd::new(vec![Tensor::from_slice(&[0.0], &[1]).unwrap()], lr)
    }

    fn close_seq(got: &[f64], want: &[f64], what: &str) {
        assert_close(got, want, 1e-9, what);
    }

    #[test]
    fn sgd_momentum_and_decay_match_torch() {
        // lr=0.1, momentum=0.9, weight_decay=0.01, grad=[0.1,0.2] (torch-verified)
        let p = grad_param(&[1.0, 2.0], &[0.1, 0.2]);
        let mut opt = Sgd::new(vec![p.clone()], 0.1)
            .momentum(0.9)
            .weight_decay(0.01);
        opt.step().unwrap();
        assert_close(&p.to_vec::<f64>().unwrap(), &[0.989, 1.978], 1e-12, "sgd1");
        p.set_grad(Some(Tensor::from_slice(&[0.1, 0.2], &[2]).unwrap()));
        opt.step().unwrap();
        assert_close(
            &p.to_vec::<f64>().unwrap(),
            &[0.968111, 1.936222],
            1e-12,
            "sgd2",
        );
        opt.zero_grad();
        assert!(p.grad().is_none());
    }

    #[test]
    fn adam_variants_match_torch() {
        // lr=0.01, betas=(0.9,0.999), eps=1e-8, grad=[0.1,-0.2] (all torch-verified)
        let a = grad_param(&[1.0, 2.0], &[0.1, -0.2]);
        let mut opt = Adam::new(vec![a.clone()], 0.01);
        opt.step().unwrap();
        assert_close(
            &a.to_vec::<f64>().unwrap(),
            &[0.9900000009999999, 2.0099999995],
            1e-12,
            "adam1",
        );
        a.set_grad(Some(Tensor::from_slice(&[0.1, -0.2], &[2]).unwrap()));
        opt.step().unwrap();
        assert_close(
            &a.to_vec::<f64>().unwrap(),
            &[0.9800000019999998, 2.019999999],
            1e-12,
            "adam2",
        );
        let w = grad_param(&[1.0, 2.0], &[0.1, -0.2]);
        Adam::adamw(vec![w.clone()], 0.01, 0.01).step().unwrap();
        assert_close(
            &w.to_vec::<f64>().unwrap(),
            &[0.9899000009999999, 2.0097999995],
            1e-12,
            "adamw",
        );
        let l2 = grad_param(&[1.0, 2.0], &[0.1, -0.2]);
        Adam::new(vec![l2.clone()], 0.01)
            .weight_decay(0.01)
            .step()
            .unwrap();
        assert_close(
            &l2.to_vec::<f64>().unwrap(),
            &[0.9900000009090908, 2.0099999994444446],
            1e-12,
            "adam_l2",
        );
        // beta1 = 0 is legal (torch accepts it) and drops momentum
        let z = grad_param(&[1.0], &[0.5]);
        Adam::new(vec![z.clone()], 0.01)
            .betas(0.0, 0.999)
            .step()
            .unwrap();
        assert_close(&z.to_vec::<f64>().unwrap(), &[0.9900000002], 1e-12, "beta0");
        // ...while beta = 1 or negative is rejected
        assert!(
            Adam::new(vec![z.clone()], 0.01)
                .betas(1.0, 0.999)
                .step()
                .is_err()
        );
        assert!(
            Adam::new(vec![z.clone()], 0.01)
                .betas(0.9, -0.5)
                .step()
                .is_err()
        );
    }

    #[test]
    fn adam_bias_correction_is_per_param() {
        // A steps twice, B joins on the second step: B must get a t=1
        // correction like a fresh optimizer, not the shared t=2 (torch-verified).
        let a = grad_param(&[1.0], &[0.5]);
        let b = Tensor::from_slice(&[1.0], &[1])
            .unwrap()
            .requires_grad_(true);
        let mut opt = Adam::new(vec![a.clone(), b.clone()], 0.01);
        opt.step().unwrap();
        a.set_grad(Some(Tensor::from_slice(&[0.5], &[1]).unwrap()));
        b.set_grad(Some(Tensor::from_slice(&[0.5], &[1]).unwrap()));
        opt.step().unwrap();
        assert_close(
            &a.to_vec::<f64>().unwrap(),
            &[0.9800000004],
            1e-10,
            "param A",
        );
        assert_close(
            &b.to_vec::<f64>().unwrap(),
            &[0.9900000002],
            1e-10,
            "param B",
        );
    }

    #[test]
    fn rmsprop_adagrad_lion_and_clipping() {
        // lr=0.01, grad=[0.1,-0.2] (rmsprop/adagrad torch-verified, lion hand-computed)
        let r = grad_param(&[1.0, 2.0], &[0.1, -0.2]);
        RmsProp::new(vec![r.clone()], 0.01).step().unwrap();
        assert_close(
            &r.to_vec::<f64>().unwrap(),
            &[0.9000000999999, 2.099999950000025],
            1e-12,
            "rmsprop",
        );
        let g = grad_param(&[1.0, 2.0], &[0.1, -0.2]);
        Adagrad::new(vec![g.clone()], 0.01).step().unwrap();
        assert_close(
            &g.to_vec::<f64>().unwrap(),
            &[0.99000000001, 2.009999999995],
            1e-12,
            "adagrad",
        );
        let l = grad_param(&[1.0, 2.0], &[0.1, -0.2]);
        Lion::new(vec![l.clone()], 0.01).step().unwrap();
        assert_close(&l.to_vec::<f64>().unwrap(), &[0.99, 2.01], 1e-12, "lion");
        // clip_grad_norm: total 5.0 scaled to max_norm (torch's max/(total+1e-6) form)
        let p1 = grad_param(&[0.0], &[3.0]);
        let p2 = grad_param(&[0.0], &[4.0]);
        let total = clip_grad_norm(&[p1.clone(), p2.clone()], 2.0).unwrap();
        assert!((total - 5.0).abs() <= 1e-12);
        let s = 2.0 / (5.0 + 1e-6);
        assert_close(
            &p1.grad().unwrap().to_vec::<f64>().unwrap(),
            &[3.0 * s],
            1e-12,
            "clip1",
        );
        assert_close(
            &p2.grad().unwrap().to_vec::<f64>().unwrap(),
            &[4.0 * s],
            1e-12,
            "clip2",
        );
        // under the cap nothing changes; negative caps error
        let p3 = grad_param(&[0.0], &[1.0]);
        assert!((clip_grad_norm(std::slice::from_ref(&p3), 2.0).unwrap() - 1.0).abs() <= 1e-12);
        assert_close(
            &p3.grad().unwrap().to_vec::<f64>().unwrap(),
            &[1.0],
            0.0,
            "noclip",
        );
        assert!(clip_grad_norm(std::slice::from_ref(&p3), -1.0).is_err());
        clip_grad_value(std::slice::from_ref(&p1), 0.5).unwrap();
        assert_close(
            &p1.grad().unwrap().to_vec::<f64>().unwrap(),
            &[0.5],
            1e-12,
            "clipval",
        );
    }

    #[test]
    fn schedulers_match_torch_step_cosine_sgdr() {
        // StepLR(3, 0.5) from 0.1 (torch-verified)
        let mut opt = dummy_opt(0.1);
        let mut s = StepLr::new(0.1, 3, 0.5);
        let got: Vec<f64> = (0..7).map(|_| s.step(&mut opt)).collect();
        close_seq(&got, &[0.1, 0.1, 0.05, 0.05, 0.05, 0.025, 0.025], "step");
        // CosineAnnealingLR(T=4, min=0.01): oscillates past T_max like torch
        let mut opt = dummy_opt(0.1);
        let mut s = CosineAnnealingLr::new(0.1, 4, 0.01);
        let got: Vec<f64> = (0..6).map(|_| s.step(&mut opt)).collect();
        close_seq(
            &got,
            &[
                0.086819805153,
                0.055,
                0.023180194847,
                0.01,
                0.023180194847,
                0.055,
            ],
            "cos",
        );
        // SGDR(T0=2, mult=2) from 0.1 (torch-verified)
        let mut opt = dummy_opt(0.1);
        let mut s = CosineAnnealingWarmRestarts::new(0.1, 2, 2, 0.0);
        let got: Vec<f64> = (0..8).map(|_| s.step(&mut opt)).collect();
        close_seq(
            &got,
            &[
                0.05,
                0.1,
                0.085355339059,
                0.05,
                0.014644660941,
                0.1,
                0.096193976626,
                0.085355339059,
            ],
            "sgdr",
        );
    }

    #[test]
    fn onecycle_warmup_and_decay_match_torch() {
        // total=10, pct=0.3, max=0.1, div=25, final_div=1e4 (torch-verified):
        // peak after `up = pct*total - 1` steps, cosine tail to total_steps.
        let mut opt = dummy_opt(0.1);
        let mut s = OneCycleLR::new(0.1, 10, 0.3, 25.0, 10000.0);
        assert!((s.current_lr() - 0.004).abs() <= 1e-12);
        let got: Vec<f64> = (0..10).map(|_| s.step(&mut opt)).collect();
        close_seq(
            &got,
            &[
                0.052,
                0.1,
                0.095048463201,
                0.081174565395,
                0.061126202194,
                0.038874197806,
                0.018825834605,
                0.004951936799,
                0.0000004,
                0.004951936799,
            ],
            "onecycle",
        );
        // total=20 ramp quarter-points prove the cosine (not linear) warmup.
        let mut opt = dummy_opt(0.01);
        let mut s = OneCycleLR::new(0.1, 20, 0.3, 25.0, 10000.0);
        let got: Vec<f64> = (0..5).map(|_| s.step(&mut opt)).collect();
        close_seq(
            &got,
            &[
                0.01316718427,
                0.03716718427,
                0.06683281573,
                0.09083281573,
                0.1,
            ],
            "ramp",
        );
    }

    #[test]
    #[should_panic(expected = "total steps")]
    fn onecycle_panics_past_total_steps() {
        // Torch raises ValueError when stepped past total_steps.
        let mut opt = dummy_opt(0.1);
        let mut s = OneCycleLR::new(0.1, 3, 0.3, 25.0, 10000.0);
        for _ in 0..4 {
            s.step(&mut opt);
        }
    }

    #[test]
    fn exponential_compounds_current_lr() {
        // torch applies gamma to the live lr even after external edits (verified)
        let mut opt = dummy_opt(0.1);
        let mut s = ExponentialLr::new(0.9);
        assert!((s.step(&mut opt) - 0.09).abs() <= 1e-12);
        assert!((s.step(&mut opt) - 0.081).abs() <= 1e-12);
        opt.set_learning_rate(0.5);
        assert!((s.step(&mut opt) - 0.45).abs() <= 1e-12);
    }

    #[test]
    fn plateau_uses_relative_threshold() {
        // min mode, factor=0.5, patience=2 (torch-verified with 1e-4 rel threshold)
        let mut opt = dummy_opt(0.1);
        let mut s = ReduceLROnPlateau::new(0.5, 2, 0.01);
        let mut got = Vec::new();
        for m in [1.0, 0.99999, 0.99998, 0.99997, 0.5, 0.5, 0.5, 0.5] {
            got.push(s.step(m, &mut opt));
        }
        close_seq(
            &got,
            &[0.1, 0.1, 0.1, 0.05, 0.05, 0.05, 0.05, 0.025],
            "plateau",
        );
    }

    #[test]
    fn warmup_cosine_hand_values() {
        // linear warmup then cosine decay to min, clamped past total (by design)
        let mut opt = dummy_opt(0.0);
        let mut s = LinearWarmupCosineLr::new(0.1, 2, 6, 0.01);
        let got: Vec<f64> = (0..7).map(|_| s.step(&mut opt)).collect();
        close_seq(
            &got,
            &[0.05, 0.1, 0.086819805153, 0.055, 0.023180194847, 0.01, 0.01],
            "warmup",
        );
    }
}
