//! More optimizers and learning-rate schedules in the style of `torch.optim`: `Adadelta`, `Adamax`, `NAdam`,
//! `RAdam`, `ASGD`, `Rprop`, `Adafactor`, `Lars`, plus composable schedules (`ConstantLr`, `LinearLr`,
//! `MultiStepLr`, `PolynomialLr`, `LambdaLr`, `CyclicLr`, `SequentialLr`, `ChainedLr`) driven by [`LrScheduler`].
//!
//! Optimizers follow the update rules of the PyTorch implementations (single-tensor path). A parameter without a
//! gradient is skipped and its state does not advance.

use crate::error::{Error, Result};
use crate::optim::{Optimizer, OptimizerState};
use crate::tensor::{Tensor, no_grad};

fn with_weight_decay(p: &Tensor, g: Tensor, wd: f64) -> Result<Tensor> {
    if wd != 0.0 { g.add(&p.mul_scalar(wd)?) } else { Ok(g) }
}

fn check_betas(name: &str, b1: f64, b2: f64) -> Result<()> {
    if !(0.0..1.0).contains(&b1) || !(0.0..1.0).contains(&b2) {
        return Err(Error::Msg(format!("{name}: betas must lie in [0, 1), got ({b1}, {b2})")));
    }
    Ok(())
}

macro_rules! optimizer_boilerplate {
    () => {
        fn zero_grad(&mut self) {
            self.params.iter().for_each(Tensor::zero_grad);
        }
        fn learning_rate(&self) -> f64 {
            self.lr
        }
        fn set_learning_rate(&mut self, lr: f64) {
            self.lr = lr;
        }
    };
}


macro_rules! state_impl {
    (tensors: [$(($name:literal, $field:ident)),*], scalars: [$(($sname:literal, $sfield:ident)),*]) => {
        fn state_dict(&self) -> OptimizerState {
            let mut s = OptimizerState::default();
            $( s.put_tensors($name, &self.$field); )*
            $( s.put_scalars($sname, self.$sfield.iter().map(|&v| v as f64)); )*
            s.scalars.insert("lr".into(), self.lr);
            s
        }
        fn load_state_dict(&mut self, state: &OptimizerState) -> Result<()> {
            $( state.take_tensors($name, &mut self.$field)?; )*
            $(
                {
                    let mut tmp: Vec<f64> = self.$sfield.iter().map(|&v| v as f64).collect();
                    state.take_scalars($sname, &mut tmp);
                    for (d, s) in self.$sfield.iter_mut().zip(tmp) {
                        *d = s as _;
                    }
                }
            )*
            if let Some(&lr) = state.scalars.get("lr") {
                self.lr = lr;
            }
            Ok(())
        }
    };
}
// ---------------------------------------------------------------------------------------------------------------
// Adadelta
// ---------------------------------------------------------------------------------------------------------------

/// Adadelta (Zeiler 2012): `lr = 1.0`, `rho = 0.9`, `eps = 1e-6` by default.
pub struct Adadelta {
    params: Vec<Tensor>,
    lr: f64,
    rho: f64,
    eps: f64,
    weight_decay: f64,
    square_avg: Vec<Option<Tensor>>,
    acc_delta: Vec<Option<Tensor>>,
}

impl Adadelta {
    pub fn new(params: Vec<Tensor>, lr: f64) -> Self {
        let n = params.len();
        Self { params, lr, rho: 0.9, eps: 1e-6, weight_decay: 0.0, square_avg: vec![None; n], acc_delta: vec![None; n] }
    }
    pub fn rho(mut self, rho: f64) -> Self {
        self.rho = rho;
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

impl Optimizer for Adadelta {
    fn step(&mut self) -> Result<()> {
        if !(0.0..=1.0).contains(&self.rho) {
            return Err(Error::Msg(format!("Adadelta: rho must lie in [0, 1], got {}", self.rho)));
        }
        no_grad(|| {
            for (i, p) in self.params.iter().enumerate() {
                let Some(g) = p.grad() else { continue };
                let g = with_weight_decay(p, g, self.weight_decay)?;
                let sq = self.square_avg[i].take().map_or_else(|| p.zeros_like(), Ok)?;
                let acc = self.acc_delta[i].take().map_or_else(|| p.zeros_like(), Ok)?;
                let sq = sq.mul_scalar(self.rho)?.add(&g.square()?.mul_scalar(1.0 - self.rho)?)?;
                let std = sq.add_scalar(self.eps)?.sqrt()?;
                let delta = acc.add_scalar(self.eps)?.sqrt()?.div(&std)?.mul(&g)?;
                let acc = acc.mul_scalar(self.rho)?.add(&delta.square()?.mul_scalar(1.0 - self.rho)?)?;
                p.assign(&p.sub(&delta.mul_scalar(self.lr)?)?)?;
                self.square_avg[i] = Some(sq);
                self.acc_delta[i] = Some(acc);
            }
            Ok(())
        })
    }
    optimizer_boilerplate!();
    state_impl!(tensors: [("square_avg", square_avg), ("acc_delta", acc_delta)], scalars: []);
}

// ---------------------------------------------------------------------------------------------------------------
// Adamax
// ---------------------------------------------------------------------------------------------------------------

/// Adamax: Adam with the infinity norm (`lr = 2e-3`, betas `(0.9, 0.999)`, `eps = 1e-8`).
pub struct Adamax {
    params: Vec<Tensor>,
    lr: f64,
    betas: (f64, f64),
    eps: f64,
    weight_decay: f64,
    exp_avg: Vec<Option<Tensor>>,
    exp_inf: Vec<Option<Tensor>>,
    t: Vec<i32>,
}

impl Adamax {
    pub fn new(params: Vec<Tensor>, lr: f64) -> Self {
        let n = params.len();
        Self { params, lr, betas: (0.9, 0.999), eps: 1e-8, weight_decay: 0.0, exp_avg: vec![None; n], exp_inf: vec![None; n], t: vec![0; n] }
    }
    pub fn betas(mut self, b1: f64, b2: f64) -> Self {
        self.betas = (b1, b2);
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

impl Optimizer for Adamax {
    fn step(&mut self) -> Result<()> {
        check_betas("Adamax", self.betas.0, self.betas.1)?;
        no_grad(|| {
            for (i, p) in self.params.iter().enumerate() {
                let Some(g) = p.grad() else { continue };
                let g = with_weight_decay(p, g, self.weight_decay)?;
                self.t[i] += 1;
                let m = self.exp_avg[i].take().map_or_else(|| p.zeros_like(), Ok)?;
                let u = self.exp_inf[i].take().map_or_else(|| p.zeros_like(), Ok)?;
                let m = m.mul_scalar(self.betas.0)?.add(&g.mul_scalar(1.0 - self.betas.0)?)?;
                let u = u.mul_scalar(self.betas.1)?.maximum(&g.abs()?.add_scalar(self.eps)?)?;
                let clr = self.lr / (1.0 - self.betas.0.powi(self.t[i]));
                p.assign(&p.sub(&m.div(&u)?.mul_scalar(clr)?)?)?;
                self.exp_avg[i] = Some(m);
                self.exp_inf[i] = Some(u);
            }
            Ok(())
        })
    }
    optimizer_boilerplate!();
    state_impl!(tensors: [("exp_avg", exp_avg), ("exp_inf", exp_inf)], scalars: [("step", t)]);
}

// ---------------------------------------------------------------------------------------------------------------
// NAdam
// ---------------------------------------------------------------------------------------------------------------

/// NAdam: Adam with Nesterov momentum (`lr = 2e-3`, `momentum_decay = 4e-3`).
pub struct NAdam {
    params: Vec<Tensor>,
    lr: f64,
    betas: (f64, f64),
    eps: f64,
    weight_decay: f64,
    momentum_decay: f64,
    exp_avg: Vec<Option<Tensor>>,
    exp_avg_sq: Vec<Option<Tensor>>,
    mu_product: Vec<f64>,
    t: Vec<i32>,
}

impl NAdam {
    pub fn new(params: Vec<Tensor>, lr: f64) -> Self {
        let n = params.len();
        Self {
            params,
            lr,
            betas: (0.9, 0.999),
            eps: 1e-8,
            weight_decay: 0.0,
            momentum_decay: 4e-3,
            exp_avg: vec![None; n],
            exp_avg_sq: vec![None; n],
            mu_product: vec![1.0; n],
            t: vec![0; n],
        }
    }
    pub fn betas(mut self, b1: f64, b2: f64) -> Self {
        self.betas = (b1, b2);
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
    pub fn momentum_decay(mut self, d: f64) -> Self {
        self.momentum_decay = d;
        self
    }
}

impl Optimizer for NAdam {
    fn step(&mut self) -> Result<()> {
        check_betas("NAdam", self.betas.0, self.betas.1)?;
        no_grad(|| {
            for (i, p) in self.params.iter().enumerate() {
                let Some(g) = p.grad() else { continue };
                let g = with_weight_decay(p, g, self.weight_decay)?;
                self.t[i] += 1;
                let t = f64::from(self.t[i]);
                let (b1, b2) = self.betas;
                let bias2 = 1.0 - b2.powf(t);
                let mu = b1 * (1.0 - 0.5 * 0.96f64.powf(t * self.momentum_decay));
                let mu_next = b1 * (1.0 - 0.5 * 0.96f64.powf((t + 1.0) * self.momentum_decay));
                self.mu_product[i] *= mu;
                let m = self.exp_avg[i].take().map_or_else(|| p.zeros_like(), Ok)?;
                let v = self.exp_avg_sq[i].take().map_or_else(|| p.zeros_like(), Ok)?;
                let m = m.mul_scalar(b1)?.add(&g.mul_scalar(1.0 - b1)?)?;
                let v = v.mul_scalar(b2)?.add(&g.square()?.mul_scalar(1.0 - b2)?)?;
                let denom = v.div_scalar(bias2)?.sqrt()?.add_scalar(self.eps)?;
                let prod = self.mu_product[i];
                let a = g.div(&denom)?.mul_scalar(self.lr * (1.0 - mu) / (1.0 - prod))?;
                let b = m.div(&denom)?.mul_scalar(self.lr * mu_next / (1.0 - prod * mu_next))?;
                p.assign(&p.sub(&a)?.sub(&b)?)?;
                self.exp_avg[i] = Some(m);
                self.exp_avg_sq[i] = Some(v);
            }
            Ok(())
        })
    }
    optimizer_boilerplate!();
    state_impl!(tensors: [("exp_avg", exp_avg), ("exp_avg_sq", exp_avg_sq)], scalars: [("step", t), ("mu_product", mu_product)]);
}

// ---------------------------------------------------------------------------------------------------------------
// RAdam
// ---------------------------------------------------------------------------------------------------------------

/// Rectified Adam (`lr = 1e-3`): falls back to un-adapted momentum SGD while the variance estimate is unreliable.
pub struct RAdam {
    params: Vec<Tensor>,
    lr: f64,
    betas: (f64, f64),
    eps: f64,
    weight_decay: f64,
    decoupled: bool,
    exp_avg: Vec<Option<Tensor>>,
    exp_avg_sq: Vec<Option<Tensor>>,
    t: Vec<i32>,
}

impl RAdam {
    pub fn new(params: Vec<Tensor>, lr: f64) -> Self {
        let n = params.len();
        Self { params, lr, betas: (0.9, 0.999), eps: 1e-8, weight_decay: 0.0, decoupled: false, exp_avg: vec![None; n], exp_avg_sq: vec![None; n], t: vec![0; n] }
    }
    pub fn betas(mut self, b1: f64, b2: f64) -> Self {
        self.betas = (b1, b2);
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
    /// Decoupled (AdamW-style) weight decay.
    pub fn decoupled_weight_decay(mut self, on: bool) -> Self {
        self.decoupled = on;
        self
    }
}

impl Optimizer for RAdam {
    fn step(&mut self) -> Result<()> {
        check_betas("RAdam", self.betas.0, self.betas.1)?;
        no_grad(|| {
            for (i, p) in self.params.iter().enumerate() {
                let Some(mut g) = p.grad() else { continue };
                self.t[i] += 1;
                let t = f64::from(self.t[i]);
                let (b1, b2) = self.betas;
                if self.weight_decay != 0.0 {
                    if self.decoupled {
                        p.assign(&p.mul_scalar(1.0 - self.lr * self.weight_decay)?)?;
                    } else {
                        g = g.add(&p.mul_scalar(self.weight_decay)?)?;
                    }
                }
                let m = self.exp_avg[i].take().map_or_else(|| p.zeros_like(), Ok)?;
                let v = self.exp_avg_sq[i].take().map_or_else(|| p.zeros_like(), Ok)?;
                let m = m.mul_scalar(b1)?.add(&g.mul_scalar(1.0 - b1)?)?;
                let v = v.mul_scalar(b2)?.add(&g.square()?.mul_scalar(1.0 - b2)?)?;
                let bias1 = 1.0 - b1.powf(t);
                let bias2 = 1.0 - b2.powf(t);
                let m_hat = m.div_scalar(bias1)?;
                let rho_inf = 2.0 / (1.0 - b2) - 1.0;
                let rho_t = rho_inf - 2.0 * t * b2.powf(t) / bias2;
                let update = if rho_t > 5.0 {
                    let rect = ((rho_t - 4.0) * (rho_t - 2.0) * rho_inf / ((rho_inf - 4.0) * (rho_inf - 2.0) * rho_t)).sqrt();
                    let adaptive = v.sqrt()?.add_scalar(self.eps)?.recip()?.mul_scalar(bias2.sqrt())?;
                    m_hat.mul(&adaptive)?.mul_scalar(rect)?
                } else {
                    m_hat
                };
                p.assign(&p.sub(&update.mul_scalar(self.lr)?)?)?;
                self.exp_avg[i] = Some(m);
                self.exp_avg_sq[i] = Some(v);
            }
            Ok(())
        })
    }
    optimizer_boilerplate!();
    state_impl!(tensors: [("exp_avg", exp_avg), ("exp_avg_sq", exp_avg_sq)], scalars: [("step", t)]);
}

// ---------------------------------------------------------------------------------------------------------------
// ASGD
// ---------------------------------------------------------------------------------------------------------------

/// Averaged SGD (Polyak-Ruppert); [`averaged`](Self::averaged) returns the running average of a parameter.
pub struct Asgd {
    params: Vec<Tensor>,
    lr: f64,
    lambd: f64,
    alpha: f64,
    t0: f64,
    weight_decay: f64,
    eta: Vec<f64>,
    mu: Vec<f64>,
    ax: Vec<Option<Tensor>>,
    t: Vec<f64>,
}

impl Asgd {
    pub fn new(params: Vec<Tensor>, lr: f64) -> Self {
        let n = params.len();
        Self { params, lr, lambd: 1e-4, alpha: 0.75, t0: 1e6, weight_decay: 0.0, eta: vec![lr; n], mu: vec![1.0; n], ax: vec![None; n], t: vec![0.0; n] }
    }
    pub fn lambd(mut self, v: f64) -> Self {
        self.lambd = v;
        self
    }
    pub fn alpha(mut self, v: f64) -> Self {
        self.alpha = v;
        self
    }
    pub fn t0(mut self, v: f64) -> Self {
        self.t0 = v;
        self
    }
    pub fn weight_decay(mut self, wd: f64) -> Self {
        self.weight_decay = wd;
        self
    }
    /// The averaged value of parameter `index` (`None` before its first step).
    pub fn averaged(&self, index: usize) -> Option<&Tensor> {
        self.ax.get(index).and_then(Option::as_ref)
    }
}

impl Optimizer for Asgd {
    fn step(&mut self) -> Result<()> {
        no_grad(|| {
            for (i, p) in self.params.iter().enumerate() {
                let Some(g) = p.grad() else { continue };
                let g = with_weight_decay(p, g, self.weight_decay)?;
                self.t[i] += 1.0;
                let step = self.t[i];
                let eta = self.eta[i];
                let new_p = p.mul_scalar(1.0 - self.lambd * eta)?.sub(&g.mul_scalar(eta)?)?;
                p.assign(&new_p)?;
                let ax = self.ax[i].take().map_or_else(|| p.zeros_like(), Ok)?;
                let ax = if self.mu[i] != 1.0 {
                    ax.add(&p.sub(&ax)?.mul_scalar(self.mu[i])?)?
                } else {
                    p.detach()
                };
                self.ax[i] = Some(ax);
                self.eta[i] = self.lr / (1.0 + self.lambd * self.lr * step).powf(self.alpha);
                self.mu[i] = 1.0 / (step - self.t0).max(1.0);
            }
            Ok(())
        })
    }
    optimizer_boilerplate!();
    state_impl!(tensors: [("ax", ax)], scalars: [("eta", eta), ("mu", mu), ("step", t)]);
}

// ---------------------------------------------------------------------------------------------------------------
// Rprop
// ---------------------------------------------------------------------------------------------------------------

/// Resilient backpropagation: per-element step sizes adapted by the sign agreement of consecutive gradients.
pub struct Rprop {
    params: Vec<Tensor>,
    lr: f64,
    etas: (f64, f64),
    step_sizes: (f64, f64),
    prev: Vec<Option<Tensor>>,
    steps: Vec<Option<Tensor>>,
}

impl Rprop {
    pub fn new(params: Vec<Tensor>, lr: f64) -> Self {
        let n = params.len();
        Self { params, lr, etas: (0.5, 1.2), step_sizes: (1e-6, 50.0), prev: vec![None; n], steps: vec![None; n] }
    }
    pub fn etas(mut self, minus: f64, plus: f64) -> Self {
        self.etas = (minus, plus);
        self
    }
    pub fn step_sizes(mut self, min: f64, max: f64) -> Self {
        self.step_sizes = (min, max);
        self
    }
}

impl Optimizer for Rprop {
    fn step(&mut self) -> Result<()> {
        let (eta_minus, eta_plus) = self.etas;
        if !(0.0 < eta_minus && eta_minus < 1.0 && 1.0 < eta_plus) {
            return Err(Error::Msg(format!("Rprop: need 0 < eta_minus < 1 < eta_plus, got {:?}", self.etas)));
        }
        no_grad(|| {
            for (i, p) in self.params.iter().enumerate() {
                let Some(g) = p.grad() else { continue };
                let prev = self.prev[i].take().map_or_else(|| p.zeros_like(), Ok)?;
                let steps = self.steps[i].take().map_or_else(|| p.full_like(self.lr), Ok)?;
                let sign = g.mul(&prev)?.sign()?;
                let pos = sign.gt(&sign.zeros_like()?)?;
                let neg = sign.lt(&sign.zeros_like()?)?;
                let factor = Tensor::where_cond(
                    &pos,
                    &sign.full_like(eta_plus)?,
                    &Tensor::where_cond(&neg, &sign.full_like(eta_minus)?, &sign.ones_like()?)?,
                )?;
                let steps = steps.mul(&factor)?.clamp(self.step_sizes.0, self.step_sizes.1)?;
                // after a sign flip the gradient is treated as zero for this step and the next comparison
                let g = Tensor::where_cond(&neg, &g.zeros_like()?, &g)?;
                p.assign(&p.sub(&g.sign()?.mul(&steps)?)?)?;
                self.prev[i] = Some(g);
                self.steps[i] = Some(steps);
            }
            Ok(())
        })
    }
    optimizer_boilerplate!();
    state_impl!(tensors: [("prev", prev), ("step_size", steps)], scalars: []);
}

// ---------------------------------------------------------------------------------------------------------------
// Adafactor
// ---------------------------------------------------------------------------------------------------------------

enum Second {
    Factored { row: Tensor, col: Tensor },
    Full(Tensor),
}

/// Adafactor (Shazeer & Stern 2018) with factored second moments for matrices, following `torch.optim.Adafactor`.
pub struct Adafactor {
    params: Vec<Tensor>,
    lr: f64,
    beta2_decay: f64,
    eps: (Option<f64>, f64),
    d: f64,
    weight_decay: f64,
    state: Vec<Option<Second>>,
    t: Vec<f64>,
}

impl Adafactor {
    pub fn new(params: Vec<Tensor>, lr: f64) -> Self {
        let n = params.len();
        Self { params, lr, beta2_decay: -0.8, eps: (None, 1e-3), d: 1.0, weight_decay: 0.0, state: (0..n).map(|_| None).collect(), t: vec![0.0; n] }
    }
    pub fn beta2_decay(mut self, v: f64) -> Self {
        self.beta2_decay = v;
        self
    }
    /// `(eps1, eps2)`; `eps1 = None` uses the machine epsilon of the parameter dtype.
    pub fn eps(mut self, eps1: Option<f64>, eps2: f64) -> Self {
        self.eps = (eps1, eps2);
        self
    }
    pub fn clip_threshold(mut self, d: f64) -> Self {
        self.d = d;
        self
    }
    pub fn weight_decay(mut self, wd: f64) -> Self {
        self.weight_decay = wd;
        self
    }
}

impl Optimizer for Adafactor {
    fn step(&mut self) -> Result<()> {
        no_grad(|| {
            for (i, p) in self.params.iter().enumerate() {
                let Some(g) = p.grad() else { continue };
                self.t[i] += 1.0;
                let step = self.t[i];
                let eps1 = self.eps.0.unwrap_or(match p.dtype() {
                    crate::dtype::DType::F64 => f64::EPSILON,
                    crate::dtype::DType::F16 => 9.765_625e-4,
                    crate::dtype::DType::BF16 => 7.812_5e-3,
                    _ => f64::from(f32::EPSILON),
                });
                let beta2_t = 1.0 - step.powf(self.beta2_decay);
                let rho_t = self.lr.min(1.0 / step.sqrt());
                let numel = p.numel() as f64;
                let rms = p.square()?.sum()?.item::<f64>()?.sqrt() / numel.sqrt();
                let alpha = self.eps.1.max(rms) * rho_t;
                if self.weight_decay != 0.0 {
                    p.assign(&p.mul_scalar(1.0 - self.lr * self.weight_decay)?)?;
                }
                let nd = g.ndim();
                let var_estimate = if nd > 1 {
                    let row_mean = g.square()?.sum_axis(-1, true)?.div_scalar(g.shape()[nd - 1] as f64)?;
                    let col_mean = g.square()?.sum_axis(-2, true)?.div_scalar(g.shape()[nd - 2] as f64)?;
                    let (row, col) = match self.state[i].take() {
                        Some(Second::Factored { row, col }) => (row, col),
                        _ => (row_mean.zeros_like()?, col_mean.zeros_like()?),
                    };
                    let w = 1.0 - beta2_t;
                    let row = row.add(&row_mean.sub(&row)?.mul_scalar(w)?)?;
                    let col = col.add(&col_mean.sub(&col)?.mul_scalar(w)?)?;
                    let est = row.matmul(&col)?;
                    let denom = row.mean_axis(-2, true)?.clamp(eps1, f64::INFINITY)?;
                    self.state[i] = Some(Second::Factored { row, col });
                    est.div(&denom)?
                } else {
                    let var = match self.state[i].take() {
                        Some(Second::Full(v)) => v,
                        _ => g.zeros_like()?,
                    };
                    let var = var.add(&g.square()?.sub(&var)?.mul_scalar(1.0 - beta2_t)?)?;
                    self.state[i] = Some(Second::Full(var.clone()));
                    var
                };
                let update = var_estimate.clamp(eps1 * eps1, f64::INFINITY)?.rsqrt()?.mul(&g)?;
                let unorm = update.square()?.sum()?.item::<f64>()?.sqrt();
                let denom = (unorm / (update.numel() as f64).sqrt() / self.d).max(1.0);
                p.assign(&p.sub(&update.mul_scalar(alpha / denom)?)?)?;
            }
            Ok(())
        })
    }
    optimizer_boilerplate!();

    fn state_dict(&self) -> OptimizerState {
        let mut s = OptimizerState::default();
        for (i, st) in self.state.iter().enumerate() {
            match st {
                Some(Second::Factored { row, col }) => {
                    s.tensors.insert(format!("{i}.row"), row.detach());
                    s.tensors.insert(format!("{i}.col"), col.detach());
                }
                Some(Second::Full(v)) => {
                    s.tensors.insert(format!("{i}.variance"), v.detach());
                }
                None => {}
            }
        }
        s.put_scalars("step", self.t.iter().copied());
        s.scalars.insert("lr".into(), self.lr);
        s
    }

    fn load_state_dict(&mut self, state: &OptimizerState) -> Result<()> {
        for (i, slot) in self.state.iter_mut().enumerate() {
            let copy = |t: &Tensor| t.detach().contiguous().and_then(|c| c.mul_scalar(1.0));
            *slot = match (state.tensors.get(&format!("{i}.row")), state.tensors.get(&format!("{i}.col")), state.tensors.get(&format!("{i}.variance"))) {
                (Some(r), Some(c), _) => Some(Second::Factored { row: copy(r)?, col: copy(c)? }),
                (_, _, Some(v)) => Some(Second::Full(copy(v)?)),
                _ => None,
            };
        }
        state.take_scalars("step", &mut self.t);
        if let Some(&lr) = state.scalars.get("lr") {
            self.lr = lr;
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------------------------------------------
// LARS
// ---------------------------------------------------------------------------------------------------------------

/// Layer-wise adaptive rate scaling (You et al. 2017) on top of momentum SGD. Parameters with one dimension
/// (biases, norm scales) are updated without the trust ratio.
pub struct Lars {
    params: Vec<Tensor>,
    lr: f64,
    momentum: f64,
    weight_decay: f64,
    trust_coefficient: f64,
    eps: f64,
    velocity: Vec<Option<Tensor>>,
}

impl Lars {
    pub fn new(params: Vec<Tensor>, lr: f64) -> Self {
        let n = params.len();
        Self { params, lr, momentum: 0.9, weight_decay: 0.0, trust_coefficient: 0.001, eps: 1e-8, velocity: vec![None; n] }
    }
    pub fn momentum(mut self, m: f64) -> Self {
        self.momentum = m;
        self
    }
    pub fn weight_decay(mut self, wd: f64) -> Self {
        self.weight_decay = wd;
        self
    }
    pub fn trust_coefficient(mut self, c: f64) -> Self {
        self.trust_coefficient = c;
        self
    }
}

impl Optimizer for Lars {
    fn step(&mut self) -> Result<()> {
        no_grad(|| {
            for (i, p) in self.params.iter().enumerate() {
                let Some(g) = p.grad() else { continue };
                let g = with_weight_decay(p, g, if p.ndim() > 1 { self.weight_decay } else { 0.0 })?;
                let scale = if p.ndim() > 1 {
                    let (wn, gn) = (p.square()?.sum()?.item::<f64>()?.sqrt(), g.square()?.sum()?.item::<f64>()?.sqrt());
                    if wn > 0.0 && gn > 0.0 { self.trust_coefficient * wn / (gn + self.eps) } else { 1.0 }
                } else {
                    1.0
                };
                let scaled = g.mul_scalar(scale)?;
                let v = match self.velocity[i].take() {
                    Some(v) => v.mul_scalar(self.momentum)?.add(&scaled)?,
                    None => scaled,
                };
                p.assign(&p.sub(&v.mul_scalar(self.lr)?)?)?;
                self.velocity[i] = Some(v);
            }
            Ok(())
        })
    }
    optimizer_boilerplate!();
    state_impl!(tensors: [("momentum_buffer", velocity)], scalars: []);
}

// ---------------------------------------------------------------------------------------------------------------
// Learning-rate schedules
// ---------------------------------------------------------------------------------------------------------------

/// A closed-form schedule: the multiplier of the base learning rate at a given epoch.
pub trait Schedule {
    fn factor(&self, epoch: usize) -> f64;
}

/// Drives an optimizer's learning rate from a [`Schedule`]; call [`step`](Self::step) once per epoch.
pub struct LrScheduler<S: Schedule> {
    base_lr: f64,
    epoch: usize,
    schedule: S,
}

impl<S: Schedule> LrScheduler<S> {
    pub fn new(base_lr: f64, schedule: S) -> Self {
        Self { base_lr, epoch: 0, schedule }
    }
    /// The learning rate of the current epoch.
    pub fn lr(&self) -> f64 {
        self.base_lr * self.schedule.factor(self.epoch)
    }
    /// Advances one epoch and applies the new rate to `opt`, returning it.
    pub fn step(&mut self, opt: &mut dyn Optimizer) -> f64 {
        self.epoch += 1;
        let lr = self.lr();
        opt.set_learning_rate(lr);
        lr
    }
    pub fn epoch(&self) -> usize {
        self.epoch
    }
}

/// `factor` for the first `total_iters` epochs, then 1.
pub struct ConstantLr {
    pub factor: f64,
    pub total_iters: usize,
}

impl Schedule for ConstantLr {
    fn factor(&self, epoch: usize) -> f64 {
        if epoch < self.total_iters { self.factor } else { 1.0 }
    }
}

/// Linear ramp of the multiplier from `start_factor` to `end_factor` over `total_iters` epochs.
pub struct LinearLr {
    pub start_factor: f64,
    pub end_factor: f64,
    pub total_iters: usize,
}

impl Schedule for LinearLr {
    fn factor(&self, epoch: usize) -> f64 {
        let t = epoch.min(self.total_iters) as f64 / self.total_iters.max(1) as f64;
        self.start_factor + (self.end_factor - self.start_factor) * t
    }
}

/// Multiplies by `gamma` at every milestone epoch.
pub struct MultiStepLr {
    pub milestones: Vec<usize>,
    pub gamma: f64,
}

impl Schedule for MultiStepLr {
    fn factor(&self, epoch: usize) -> f64 {
        self.gamma.powi(self.milestones.iter().filter(|&&m| m <= epoch).count() as i32)
    }
}

/// `(1 - epoch / total_iters) ^ power`, held at zero afterwards.
pub struct PolynomialLr {
    pub total_iters: usize,
    pub power: f64,
}

impl Schedule for PolynomialLr {
    fn factor(&self, epoch: usize) -> f64 {
        (1.0 - epoch.min(self.total_iters) as f64 / self.total_iters.max(1) as f64).powf(self.power)
    }
}

/// An arbitrary multiplier function of the epoch.
pub struct LambdaLr(pub Box<dyn Fn(usize) -> f64 + Send + Sync>);

impl Schedule for LambdaLr {
    fn factor(&self, epoch: usize) -> f64 {
        (self.0)(epoch)
    }
}

/// How the amplitude of a [`CyclicLr`] changes between cycles.
#[derive(Clone, Copy)]
pub enum CyclicMode {
    Triangular,
    /// Halves the amplitude each cycle.
    Triangular2,
    /// Scales the amplitude by `gamma ^ epoch`.
    ExpRange(f64),
}

/// Cyclical learning rate (Smith 2017) between `base_lr` and `max_lr`; the base learning rate of the surrounding
/// [`LrScheduler`] must equal `base_lr`.
pub struct CyclicLr {
    pub base_lr: f64,
    pub max_lr: f64,
    pub step_size_up: usize,
    pub step_size_down: usize,
    pub mode: CyclicMode,
}

impl Schedule for CyclicLr {
    fn factor(&self, epoch: usize) -> f64 {
        let total = (self.step_size_up + self.step_size_down).max(1) as f64;
        let ratio = self.step_size_up as f64 / total;
        let cycle = (1.0 + epoch as f64 / total).floor();
        let x = 1.0 + epoch as f64 / total - cycle;
        let scale = if x <= ratio { x / ratio } else { (x - 1.0) / (ratio - 1.0) };
        let amp = (self.max_lr - self.base_lr) * scale;
        let amp = match self.mode {
            CyclicMode::Triangular => amp,
            CyclicMode::Triangular2 => amp / 2f64.powf(cycle - 1.0),
            CyclicMode::ExpRange(g) => amp * g.powi(epoch as i32),
        };
        (self.base_lr + amp) / self.base_lr
    }
}

/// Runs schedules back to back, switching at the given milestone epochs (the next schedule restarts its epoch at 0).
pub struct SequentialLr {
    pub schedules: Vec<Box<dyn Schedule + Send + Sync>>,
    pub milestones: Vec<usize>,
}

impl Schedule for SequentialLr {
    fn factor(&self, epoch: usize) -> f64 {
        let idx = self.milestones.iter().filter(|&&m| m <= epoch).count().min(self.schedules.len() - 1);
        let start = if idx == 0 { 0 } else { self.milestones[idx - 1] };
        self.schedules[idx].factor(epoch - start)
    }
}

/// Applies all schedules at once: the multipliers multiply.
pub struct ChainedLr(pub Vec<Box<dyn Schedule + Send + Sync>>);

impl Schedule for ChainedLr {
    fn factor(&self, epoch: usize) -> f64 {
        self.0.iter().map(|s| s.factor(epoch)).product()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn param(v: &[f64]) -> Tensor {
        Tensor::from_slice(v, &[v.len()]).unwrap().requires_grad_(true)
    }
    fn vals(t: &Tensor) -> Vec<f64> {
        t.to_vec::<f64>().unwrap()
    }
    /// Gradient of sum((x - 3)^2) (assigned manually so the test does not depend on autograd details).
    fn set_quadratic_grad(p: &Tensor) {
        let g = p.detach().sub_scalar(3.0).unwrap().mul_scalar(2.0).unwrap();
        p.set_grad(Some(g));
    }
    fn run(opt: &mut dyn Optimizer, p: &Tensor, steps: usize) {
        for _ in 0..steps {
            opt.zero_grad();
            set_quadratic_grad(p);
            opt.step().unwrap();
        }
    }

    #[test]
    fn every_optimizer_descends_a_quadratic() {
        type Make = Box<dyn Fn(Vec<Tensor>) -> Box<dyn Optimizer>>;
        let cases: Vec<(&str, Make, usize, f64)> = vec![
            ("adadelta", Box::new(|p| Box::new(Adadelta::new(p, 1.0).eps(1e-2))), 3000, 0.2),
            ("adamax", Box::new(|p| Box::new(Adamax::new(p, 0.1))), 400, 0.05),
            ("nadam", Box::new(|p| Box::new(NAdam::new(p, 0.1))), 400, 0.05),
            ("radam", Box::new(|p| Box::new(RAdam::new(p, 0.1))), 400, 0.05),
            ("asgd", Box::new(|p| Box::new(Asgd::new(p, 0.1))), 200, 1e-3),
            ("rprop", Box::new(|p| Box::new(Rprop::new(p, 0.1))), 200, 1e-3),
            ("adafactor", Box::new(|p| Box::new(Adafactor::new(p, 0.5))), 800, 0.2),
            ("lars", Box::new(|p| Box::new(Lars::new(p, 0.1).momentum(0.0))), 200, 1.0), // 1-D params skip the trust ratio
        ];
        for (name, make, steps, tol) in cases {
            let p = param(&[0.0, 10.0, -4.0]);
            let mut opt = make(vec![p.clone()]);
            run(opt.as_mut(), &p, steps);
            for v in vals(&p) {
                assert!((v - 3.0).abs() < tol, "{name}: ended at {v}");
            }
        }
    }

    #[test]
    fn first_steps_match_the_formulas() {
        // Adamax step 1: p -= lr * sign(g) (g / (|g| + eps))
        let p = param(&[1.0]);
        p.set_grad(Some(Tensor::from_slice(&[4.0], &[1]).unwrap()));
        Adamax::new(vec![p.clone()], 0.01).step().unwrap();
        assert!((vals(&p)[0] - (1.0 - 0.01 * 4.0 / (4.0 + 1e-8))).abs() < 1e-12);
        // Rprop step 1: p -= lr * sign(g); step 2 with the same sign grows the step by eta_plus
        let p = param(&[1.0]);
        let mut o = Rprop::new(vec![p.clone()], 0.1);
        p.set_grad(Some(Tensor::from_slice(&[2.0], &[1]).unwrap()));
        o.step().unwrap();
        assert!((vals(&p)[0] - 0.9).abs() < 1e-12);
        p.set_grad(Some(Tensor::from_slice(&[2.0], &[1]).unwrap()));
        o.step().unwrap();
        assert!((vals(&p)[0] - (0.9 - 0.12)).abs() < 1e-12);
        // a sign flip shrinks the step and skips the move
        p.set_grad(Some(Tensor::from_slice(&[-1.0], &[1]).unwrap()));
        o.step().unwrap();
        assert!((vals(&p)[0] - 0.78).abs() < 1e-12);
        // RAdam step 1 is plain bias-corrected momentum: p -= lr * g
        let p = param(&[1.0]);
        p.set_grad(Some(Tensor::from_slice(&[0.5], &[1]).unwrap()));
        RAdam::new(vec![p.clone()], 0.1).step().unwrap();
        assert!((vals(&p)[0] - 0.95).abs() < 1e-12);
        // NAdam step 1, computed by hand
        let p = param(&[1.0]);
        p.set_grad(Some(Tensor::from_slice(&[0.5], &[1]).unwrap()));
        NAdam::new(vec![p.clone()], 0.1).step().unwrap();
        let mu = 0.9 * (1.0 - 0.5 * 0.96f64.powf(4e-3));
        let mu_next = 0.9 * (1.0 - 0.5 * 0.96f64.powf(8e-3));
        let denom = 0.5 + 1e-8; // sqrt(v / (1 - b2)) = |g|
        let want = 1.0 - 0.1 * (1.0 - mu) / (1.0 - mu) * 0.5 / denom - 0.1 * mu_next / (1.0 - mu * mu_next) * (0.1 * 0.5) / denom;
        assert!((vals(&p)[0] - want).abs() < 1e-9, "{} vs {want}", vals(&p)[0]);
        // Adadelta step 1: delta = sqrt(eps) / sqrt(0.1 g^2 + eps) * g
        let p = param(&[1.0]);
        p.set_grad(Some(Tensor::from_slice(&[2.0], &[1]).unwrap()));
        Adadelta::new(vec![p.clone()], 1.0).step().unwrap();
        let want = 1.0 - (1e-6f64).sqrt() / (0.1 * 4.0 + 1e-6f64).sqrt() * 2.0;
        assert!((vals(&p)[0] - want).abs() < 1e-12);
    }

    #[test]
    fn adafactor_factors_matrices() {
        let p = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]).unwrap().requires_grad_(true);
        let mut opt = Adafactor::new(vec![p.clone()], 0.1);
        p.set_grad(Some(Tensor::from_slice(&[0.1, 0.2, 0.3, 0.4, 0.5, 0.6], &[2, 3]).unwrap()));
        opt.step().unwrap();
        assert!(vals(&p).iter().zip([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).all(|(a, b)| a < &b));
        assert!(matches!(opt.state[0], Some(Second::Factored { .. })));
    }

    #[test]
    fn schedules() {
        let s = |sch: &dyn Schedule, e: &[usize]| e.iter().map(|&i| sch.factor(i)).collect::<Vec<_>>();
        assert_eq!(s(&ConstantLr { factor: 0.5, total_iters: 2 }, &[0, 1, 2, 3]), vec![0.5, 0.5, 1.0, 1.0]);
        assert_eq!(s(&LinearLr { start_factor: 0.0, end_factor: 1.0, total_iters: 4 }, &[0, 2, 4, 8]), vec![0.0, 0.5, 1.0, 1.0]);
        assert_eq!(s(&MultiStepLr { milestones: vec![2, 4], gamma: 0.1 }, &[0, 2, 3, 4]), vec![1.0, 0.1, 0.1, 0.1 * 0.1]);
        assert_eq!(s(&PolynomialLr { total_iters: 4, power: 1.0 }, &[0, 2, 4, 9]), vec![1.0, 0.5, 0.0, 0.0]);
        assert_eq!(s(&LambdaLr(Box::new(|e| 0.95f64.powi(e as i32))), &[0, 2]), vec![1.0, 0.95 * 0.95]);
        let cyc = CyclicLr { base_lr: 0.1, max_lr: 0.5, step_size_up: 2, step_size_down: 2, mode: CyclicMode::Triangular };
        let lrs: Vec<f64> = (0..5).map(|e| 0.1 * cyc.factor(e)).collect();
        for (g, w) in lrs.iter().zip([0.1, 0.3, 0.5, 0.3, 0.1]) {
            assert!((g - w).abs() < 1e-12, "{lrs:?}");
        }
        let cyc2 = CyclicLr { mode: CyclicMode::Triangular2, ..cyc };
        assert!((0.1 * cyc2.factor(6) - 0.3).abs() < 1e-12); // second cycle peaks at half amplitude, epoch 6 is its midpoint
        let seq = SequentialLr {
            schedules: vec![Box::new(LinearLr { start_factor: 0.1, end_factor: 1.0, total_iters: 2 }), Box::new(MultiStepLr { milestones: vec![1], gamma: 0.5 })],
            milestones: vec![2],
        };
        let f: Vec<f64> = (0..5).map(|e| seq.factor(e)).collect();
        assert!((f[0] - 0.1).abs() < 1e-12 && (f[1] - 0.55).abs() < 1e-12 && f[2] == 1.0 && f[3] == 0.5 && f[4] == 0.5, "{f:?}");
        let chain = ChainedLr(vec![Box::new(ConstantLr { factor: 0.5, total_iters: 2 }), Box::new(MultiStepLr { milestones: vec![1], gamma: 0.1 })]);
        assert_eq!(s(&chain, &[0, 1, 2]), vec![0.5, 0.05, 0.1]);
        // driver
        let mut opt = Sgd::new(vec![param(&[0.0])], 1.0);
        let mut sched = LrScheduler::new(1.0, MultiStepLr { milestones: vec![1], gamma: 0.5 });
        assert_eq!(sched.lr(), 1.0);
        assert_eq!(sched.step(&mut opt), 0.5);
        assert_eq!(opt.learning_rate(), 0.5);
    }

    use crate::optim::Sgd;
}

#[cfg(test)]
mod state_tests {
    use super::*;
    use crate::optim::{Adam, Sgd};

    fn quad_grad(p: &Tensor) {
        p.set_grad(Some(p.detach().sub_scalar(3.0).unwrap().mul_scalar(2.0).unwrap()));
    }
    fn vals(t: &Tensor) -> Vec<f64> {
        t.to_vec::<f64>().unwrap()
    }

    /// Trains `a` for 5 steps, snapshots, trains 5 more; a fresh optimizer restored from the snapshot (on a copy of the
    /// parameters at the snapshot) must follow the same trajectory.
    fn resumes_identically(make: impl Fn(Vec<Tensor>) -> Box<dyn Optimizer>) {
        let p1 = Tensor::from_slice(&[0.0f64, 8.0, -3.0], &[3]).unwrap().requires_grad_(true);
        let mut o1 = make(vec![p1.clone()]);
        for _ in 0..5 {
            o1.zero_grad();
            quad_grad(&p1);
            o1.step().unwrap();
        }
        let snapshot_params = vals(&p1);
        let state = o1.state_dict();
        let path = std::env::temp_dir().join(format!("oxideon_opt_{}.safetensors", snapshot_params[0].to_bits()));
        state.save(&path).unwrap();
        let state = OptimizerState::load(&path).unwrap();
        let _ = std::fs::remove_file(&path);
        let p2 = Tensor::from_slice(&snapshot_params, &[3]).unwrap().requires_grad_(true);
        let mut o2 = make(vec![p2.clone()]);
        o2.load_state_dict(&state).unwrap();
        for _ in 0..5 {
            for (o, p) in [(&mut o1, &p1), (&mut o2, &p2)] {
                o.zero_grad();
                quad_grad(p);
                o.step().unwrap();
            }
        }
        for (a, b) in vals(&p1).iter().zip(vals(&p2)) {
            assert!((a - b).abs() < 1e-12, "{a} vs {b}");
        }
    }

    #[test]
    fn optimizers_resume_from_state() {
        resumes_identically(|p| Box::new(Sgd::new(p, 0.1).momentum(0.9)));
        resumes_identically(|p| Box::new(Adam::new(p, 0.1)));
        resumes_identically(|p| Box::new(Adadelta::new(p, 1.0)));
        resumes_identically(|p| Box::new(Adamax::new(p, 0.1)));
        resumes_identically(|p| Box::new(NAdam::new(p, 0.1)));
        resumes_identically(|p| Box::new(RAdam::new(p, 0.1)));
        resumes_identically(|p| Box::new(Asgd::new(p, 0.1)));
        resumes_identically(|p| Box::new(Rprop::new(p, 0.1)));
        resumes_identically(|p| Box::new(Adafactor::new(p, 0.3)));
        resumes_identically(|p| Box::new(Lars::new(p, 0.1)));
    }

    #[test]
    fn a_fresh_optimizer_without_the_state_diverges() {
        // sanity check of the test itself: dropping the state changes the trajectory for a stateful optimizer
        let p1 = Tensor::from_slice(&[0.0f64], &[1]).unwrap().requires_grad_(true);
        let mut o1 = Adam::new(vec![p1.clone()], 0.1);
        for _ in 0..5 {
            quad_grad(&p1);
            o1.step().unwrap();
        }
        let p2 = Tensor::from_slice(&vals(&p1), &[1]).unwrap().requires_grad_(true);
        let mut o2 = Adam::new(vec![p2.clone()], 0.1);
        quad_grad(&p1);
        o1.step().unwrap();
        quad_grad(&p2);
        o2.step().unwrap();
        assert!((vals(&p1)[0] - vals(&p2)[0]).abs() > 1e-6);
    }
}

// ---------------------------------------------------------------------------------------------------------------
// Parameter groups
// ---------------------------------------------------------------------------------------------------------------

/// A set of parameters with their own learning rate and weight decay (`torch.optim` param group).
#[derive(Clone)]
pub struct ParamGroup {
    pub params: Vec<Tensor>,
    pub lr: f64,
    pub weight_decay: f64,
}

impl ParamGroup {
    pub fn new(params: Vec<Tensor>, lr: f64) -> Self {
        Self { params, lr, weight_decay: 0.0 }
    }
    pub fn weight_decay(mut self, wd: f64) -> Self {
        self.weight_decay = wd;
        self
    }

    /// The usual two-group split: weights (two or more dimensions) get `weight_decay`, biases and norm scales
    /// (one dimension) get none.
    pub fn decay_split(named: Vec<(String, Tensor)>, lr: f64, weight_decay: f64) -> Vec<ParamGroup> {
        let (decay, no_decay): (Vec<_>, Vec<_>) = named.into_iter().map(|(_, t)| t).partition(|t| t.ndim() > 1);
        vec![
            ParamGroup { params: decay, lr, weight_decay },
            ParamGroup { params: no_decay, lr, weight_decay: 0.0 },
        ]
    }
}

/// Several optimizers of one kind, one per [`ParamGroup`], driven as a single [`Optimizer`]. A scheduler that calls
/// `set_learning_rate` scales every group by the same factor, keeping their ratios (as PyTorch's schedulers do).
pub struct GroupedOptimizer {
    opts: Vec<Box<dyn Optimizer>>,
    base_lr: Vec<f64>,
    scale: f64,
}

impl GroupedOptimizer {
    /// `make` builds the optimizer for a group from its parameters, learning rate and weight decay.
    pub fn new(groups: Vec<ParamGroup>, make: impl Fn(Vec<Tensor>, f64, f64) -> Box<dyn Optimizer>) -> Result<Self> {
        if groups.is_empty() {
            return Err(Error::Msg("GroupedOptimizer: at least one parameter group is required".into()));
        }
        let base_lr = groups.iter().map(|g| g.lr).collect();
        let opts = groups.into_iter().map(|g| make(g.params, g.lr, g.weight_decay)).collect();
        Ok(Self { opts, base_lr, scale: 1.0 })
    }
    pub fn num_groups(&self) -> usize {
        self.opts.len()
    }
    pub fn group_lr(&self, i: usize) -> f64 {
        self.opts[i].learning_rate()
    }
    /// Sets one group's rate directly (and makes it that group's new base for later scheduler scaling).
    pub fn set_group_lr(&mut self, i: usize, lr: f64) {
        self.opts[i].set_learning_rate(lr);
        self.base_lr[i] = lr / self.scale;
    }
}

impl Optimizer for GroupedOptimizer {
    fn step(&mut self) -> Result<()> {
        self.opts.iter_mut().try_for_each(|o| o.step())
    }
    fn zero_grad(&mut self) {
        self.opts.iter_mut().for_each(|o| o.zero_grad());
    }
    /// The first group's current rate.
    fn learning_rate(&self) -> f64 {
        self.opts[0].learning_rate()
    }
    /// Scales all groups so that the first group's rate becomes `lr`.
    fn set_learning_rate(&mut self, lr: f64) {
        let reference = self.base_lr[0];
        self.scale = if reference == 0.0 { 1.0 } else { lr / reference };
        for (o, &b) in self.opts.iter_mut().zip(&self.base_lr) {
            o.set_learning_rate(b * self.scale);
        }
    }
    fn state_dict(&self) -> OptimizerState {
        let mut s = OptimizerState::default();
        for (g, o) in self.opts.iter().enumerate() {
            let inner = o.state_dict();
            s.tensors.extend(inner.tensors.into_iter().map(|(k, v)| (format!("g{g}.{k}"), v)));
            s.scalars.extend(inner.scalars.into_iter().map(|(k, v)| (format!("g{g}.{k}"), v)));
        }
        s.scalars.insert("scale".into(), self.scale);
        for (g, b) in self.base_lr.iter().enumerate() {
            s.scalars.insert(format!("base_lr.{g}"), *b);
        }
        s
    }
    fn load_state_dict(&mut self, state: &OptimizerState) -> Result<()> {
        for (g, o) in self.opts.iter_mut().enumerate() {
            let prefix = format!("g{g}.");
            let sub = OptimizerState {
                tensors: state.tensors.iter().filter_map(|(k, v)| k.strip_prefix(&prefix).map(|r| (r.to_string(), v.clone()))).collect(),
                scalars: state.scalars.iter().filter_map(|(k, v)| k.strip_prefix(&prefix).map(|r| (r.to_string(), *v))).collect(),
            };
            o.load_state_dict(&sub)?;
        }
        if let Some(&s) = state.scalars.get("scale") {
            self.scale = s;
        }
        for (g, b) in self.base_lr.iter_mut().enumerate() {
            if let Some(&v) = state.scalars.get(&format!("base_lr.{g}")) {
                *b = v;
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod group_tests {
    use super::*;
    use crate::optim::{Adam, Sgd};

    fn p(v: f64) -> Tensor {
        Tensor::from_slice(&[v], &[1]).unwrap().requires_grad_(true)
    }
    fn val(t: &Tensor) -> f64 {
        t.to_vec::<f64>().unwrap()[0]
    }

    #[test]
    fn groups_use_their_own_rates_and_scale_together() {
        let (a, b) = (p(1.0), p(1.0));
        let groups = vec![ParamGroup::new(vec![a.clone()], 0.1), ParamGroup::new(vec![b.clone()], 0.01)];
        let mut opt = GroupedOptimizer::new(groups, |ps, lr, _| Box::new(Sgd::new(ps, lr))).unwrap();
        for t in [&a, &b] {
            t.set_grad(Some(Tensor::from_slice(&[1.0], &[1]).unwrap()));
        }
        opt.step().unwrap();
        assert!((val(&a) - 0.9).abs() < 1e-12 && (val(&b) - 0.99).abs() < 1e-12);
        // a scheduler halving the first group's rate halves the second as well
        opt.set_learning_rate(0.05);
        assert!((opt.group_lr(0) - 0.05).abs() < 1e-12 && (opt.group_lr(1) - 0.005).abs() < 1e-12);
        opt.set_group_lr(1, 0.02);
        assert_eq!(opt.group_lr(1), 0.02);
        assert!(GroupedOptimizer::new(vec![], |ps, lr, _| Box::new(Sgd::new(ps, lr))).is_err());
    }

    #[test]
    fn decay_split_and_state_round_trip() {
        let w = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], &[2, 2]).unwrap().requires_grad_(true);
        let bias = Tensor::from_slice(&[1.0, 1.0], &[2]).unwrap().requires_grad_(true);
        let groups = ParamGroup::decay_split(vec![("w".into(), w.clone()), ("b".into(), bias.clone())], 0.1, 0.5);
        assert_eq!((groups[0].params.len(), groups[1].params.len()), (1, 1));
        assert_eq!((groups[0].weight_decay, groups[1].weight_decay), (0.5, 0.0));
        let make = |ps: Vec<Tensor>, lr: f64, wd: f64| -> Box<dyn Optimizer> { Box::new(Adam::adamw(ps, lr, wd)) };
        let mut opt = GroupedOptimizer::new(groups, make).unwrap();
        for t in [&w, &bias] {
            t.set_grad(Some(t.detach().zeros_like().unwrap()));
        }
        opt.step().unwrap();
        // zero gradients: only the decayed weight shrinks (1 * (1 - lr * wd)); the bias is untouched
        assert!((w.to_vec::<f64>().unwrap()[0] - 0.95).abs() < 1e-9);
        assert_eq!(bias.to_vec::<f64>().unwrap(), vec![1.0, 1.0]);
        let state = opt.state_dict();
        assert!(state.scalars.keys().any(|k| k.starts_with("g0.")) && state.scalars.keys().any(|k| k.starts_with("g1.")));
        let mut other = GroupedOptimizer::new(ParamGroup::decay_split(vec![("w".into(), w), ("b".into(), bias)], 0.3, 0.5), make).unwrap();
        other.load_state_dict(&state).unwrap();
        assert!((other.group_lr(0) - 0.1).abs() < 1e-12);
    }
}

// ---------------------------------------------------------------------------------------------------------------
// More schedules (closed form, composable with SequentialLr / ChainedLr)
// ---------------------------------------------------------------------------------------------------------------

/// Cosine annealing from the base rate down to `eta_min_ratio * base` over `t_max` epochs, then held.
pub struct CosineSchedule {
    pub t_max: usize,
    pub eta_min_ratio: f64,
}

impl Schedule for CosineSchedule {
    fn factor(&self, epoch: usize) -> f64 {
        let t = epoch.min(self.t_max) as f64 / self.t_max.max(1) as f64;
        self.eta_min_ratio + (1.0 - self.eta_min_ratio) * 0.5 * (1.0 + (std::f64::consts::PI * t).cos())
    }
}

/// `gamma ^ epoch`.
pub struct ExponentialSchedule {
    pub gamma: f64,
}

impl Schedule for ExponentialSchedule {
    fn factor(&self, epoch: usize) -> f64 {
        self.gamma.powi(epoch as i32)
    }
}

/// Multiplies by `gamma` every `step_size` epochs.
pub struct StepSchedule {
    pub step_size: usize,
    pub gamma: f64,
}

impl Schedule for StepSchedule {
    fn factor(&self, epoch: usize) -> f64 {
        self.gamma.powi((epoch / self.step_size.max(1)) as i32)
    }
}

/// Cosine annealing with warm restarts (SGDR): cycles of length `t0`, `t0 * t_mult`, ...
pub struct WarmRestartsSchedule {
    pub t0: usize,
    pub t_mult: usize,
    pub eta_min_ratio: f64,
}

impl Schedule for WarmRestartsSchedule {
    fn factor(&self, epoch: usize) -> f64 {
        let (mut t, mut len) = (epoch, self.t0.max(1));
        while t >= len {
            t -= len;
            len *= self.t_mult.max(1);
            if self.t_mult <= 1 {
                t %= len;
                break;
            }
        }
        self.eta_min_ratio + (1.0 - self.eta_min_ratio) * 0.5 * (1.0 + (std::f64::consts::PI * t as f64 / len as f64).cos())
    }
}

/// One-cycle policy (Smith & Topin): rises from `base / div_factor` to `base` over the first `pct_start` of
/// `total_steps` (cosine), then anneals to `base / (div_factor * final_div_factor)`.
pub struct OneCycleSchedule {
    pub total_steps: usize,
    pub pct_start: f64,
    pub div_factor: f64,
    pub final_div_factor: f64,
}

impl Schedule for OneCycleSchedule {
    fn factor(&self, epoch: usize) -> f64 {
        let up = ((self.total_steps as f64 * self.pct_start).round() as usize).max(1);
        let initial = 1.0 / self.div_factor;
        let minimum = initial / self.final_div_factor;
        let cos = |from: f64, to: f64, t: f64| to + (from - to) * 0.5 * (1.0 + (std::f64::consts::PI * t.clamp(0.0, 1.0)).cos());
        if epoch <= up {
            cos(1.0, initial, 1.0 - epoch as f64 / up as f64)
        } else {
            let down = self.total_steps.saturating_sub(up).max(1);
            cos(minimum, 1.0, 1.0 - ((epoch - up) as f64 / down as f64).min(1.0))
        }
    }
}

/// The Transformer schedule: linear warmup for `warmup` steps, then `1 / sqrt(step)` decay, peaking at 1.
pub struct InverseSqrtSchedule {
    pub warmup: usize,
}

impl Schedule for InverseSqrtSchedule {
    fn factor(&self, epoch: usize) -> f64 {
        let w = self.warmup.max(1) as f64;
        let s = (epoch.max(1)) as f64;
        (s / w).min((w / s).sqrt())
    }
}

#[cfg(test)]
mod more_schedule_tests {
    use super::*;

    #[test]
    fn closed_forms() {
        let cos = CosineSchedule { t_max: 10, eta_min_ratio: 0.1 };
        assert!((cos.factor(0) - 1.0).abs() < 1e-12 && (cos.factor(5) - 0.55).abs() < 1e-12 && (cos.factor(10) - 0.1).abs() < 1e-12 && (cos.factor(99) - 0.1).abs() < 1e-12);
        assert!((ExponentialSchedule { gamma: 0.5 }.factor(3) - 0.125).abs() < 1e-12);
        assert_eq!(StepSchedule { step_size: 2, gamma: 0.1 }.factor(5), 0.1 * 0.1);
        // restarts: lengths 2, 4, 8 -> the factor returns to 1 at epochs 2 and 6
        let wr = WarmRestartsSchedule { t0: 2, t_mult: 2, eta_min_ratio: 0.0 };
        assert!((wr.factor(2) - 1.0).abs() < 1e-12 && (wr.factor(6) - 1.0).abs() < 1e-12 && wr.factor(1) < 0.6);
        assert!((WarmRestartsSchedule { t0: 4, t_mult: 1, eta_min_ratio: 0.0 }.factor(8) - 1.0).abs() < 1e-12);
        let oc = OneCycleSchedule { total_steps: 100, pct_start: 0.3, div_factor: 25.0, final_div_factor: 1e4 };
        assert!((oc.factor(0) - 1.0 / 25.0).abs() < 1e-12 && (oc.factor(30) - 1.0).abs() < 1e-12 && (oc.factor(100) - 1.0 / 25.0 / 1e4).abs() < 1e-12);
        assert!(oc.factor(15) > oc.factor(5) && oc.factor(60) < oc.factor(40));
        let inv = InverseSqrtSchedule { warmup: 100 };
        assert!((inv.factor(50) - 0.5).abs() < 1e-12 && (inv.factor(100) - 1.0).abs() < 1e-12 && (inv.factor(400) - 0.5).abs() < 1e-12);
        // warmup then cosine through the existing combinators
        let seq = SequentialLr {
            schedules: vec![Box::new(LinearLr { start_factor: 0.01, end_factor: 1.0, total_iters: 5 }), Box::new(CosineSchedule { t_max: 20, eta_min_ratio: 0.0 })],
            milestones: vec![5],
        };
        assert!((seq.factor(0) - 0.01).abs() < 1e-12 && (seq.factor(5) - 1.0).abs() < 1e-12 && seq.factor(25) < 1e-9);
    }
}

// ---------------------------------------------------------------------------------------------------------------
// AdaBelief, Yogi, SAM
// ---------------------------------------------------------------------------------------------------------------

/// AdaBelief (Zhuang et al. 2020): Adam with the second moment of the deviation of the gradient from its running mean.
pub struct AdaBelief {
    params: Vec<Tensor>,
    lr: f64,
    betas: (f64, f64),
    eps: f64,
    weight_decay: f64,
    exp_avg: Vec<Option<Tensor>>,
    exp_avg_sq: Vec<Option<Tensor>>,
    t: Vec<i32>,
}

impl AdaBelief {
    pub fn new(params: Vec<Tensor>, lr: f64) -> Self {
        let n = params.len();
        Self { params, lr, betas: (0.9, 0.999), eps: 1e-16, weight_decay: 0.0, exp_avg: vec![None; n], exp_avg_sq: vec![None; n], t: vec![0; n] }
    }
    pub fn betas(mut self, b1: f64, b2: f64) -> Self {
        self.betas = (b1, b2);
        self
    }
    pub fn eps(mut self, eps: f64) -> Self {
        self.eps = eps;
        self
    }
    /// Decoupled (AdamW-style) weight decay.
    pub fn weight_decay(mut self, wd: f64) -> Self {
        self.weight_decay = wd;
        self
    }
}

impl Optimizer for AdaBelief {
    fn step(&mut self) -> Result<()> {
        check_betas("AdaBelief", self.betas.0, self.betas.1)?;
        no_grad(|| {
            for (i, p) in self.params.iter().enumerate() {
                let Some(g) = p.grad() else { continue };
                self.t[i] += 1;
                let t = self.t[i];
                let (b1, b2) = self.betas;
                if self.weight_decay != 0.0 {
                    p.assign(&p.mul_scalar(1.0 - self.lr * self.weight_decay)?)?;
                }
                let m = self.exp_avg[i].take().map_or_else(|| p.zeros_like(), Ok)?;
                let s = self.exp_avg_sq[i].take().map_or_else(|| p.zeros_like(), Ok)?;
                let m = m.mul_scalar(b1)?.add(&g.mul_scalar(1.0 - b1)?)?;
                let diff = g.sub(&m)?;
                let s = s.mul_scalar(b2)?.add(&diff.square()?.mul_scalar(1.0 - b2)?)?.add_scalar(self.eps)?;
                let (bc1, bc2) = (1.0 - b1.powi(t), 1.0 - b2.powi(t));
                let denom = s.div_scalar(bc2)?.sqrt()?.add_scalar(self.eps)?;
                p.assign(&p.sub(&m.div_scalar(bc1)?.div(&denom)?.mul_scalar(self.lr)?)?)?;
                self.exp_avg[i] = Some(m);
                self.exp_avg_sq[i] = Some(s);
            }
            Ok(())
        })
    }
    optimizer_boilerplate!();
    state_impl!(tensors: [("exp_avg", exp_avg), ("exp_avg_sq", exp_avg_sq)], scalars: [("step", t)]);
}

// ---------------------------------------------------------------------------------------------------------------
// LAMB
// ---------------------------------------------------------------------------------------------------------------

/// LAMB (You et al. 2020): Adam moments with a per-tensor trust ratio `||w|| / ||update||`, for large-batch training.
pub struct Lamb {
    params: Vec<Tensor>,
    lr: f64,
    betas: (f64, f64),
    eps: f64,
    weight_decay: f64,
    exp_avg: Vec<Option<Tensor>>,
    exp_avg_sq: Vec<Option<Tensor>>,
    t: Vec<i32>,
}

impl Lamb {
    pub fn new(params: Vec<Tensor>, lr: f64) -> Self {
        let n = params.len();
        Self { params, lr, betas: (0.9, 0.999), eps: 1e-6, weight_decay: 0.0, exp_avg: vec![None; n], exp_avg_sq: vec![None; n], t: vec![0; n] }
    }
    pub fn betas(mut self, b1: f64, b2: f64) -> Self {
        self.betas = (b1, b2);
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

impl Optimizer for Lamb {
    fn step(&mut self) -> Result<()> {
        check_betas("Lamb", self.betas.0, self.betas.1)?;
        no_grad(|| {
            for (i, p) in self.params.iter().enumerate() {
                let Some(g) = p.grad() else { continue };
                self.t[i] += 1;
                let t = self.t[i];
                let (b1, b2) = self.betas;
                let m = self.exp_avg[i].take().map_or_else(|| p.zeros_like(), Ok)?;
                let v = self.exp_avg_sq[i].take().map_or_else(|| p.zeros_like(), Ok)?;
                let m = m.mul_scalar(b1)?.add(&g.mul_scalar(1.0 - b1)?)?;
                let v = v.mul_scalar(b2)?.add(&g.square()?.mul_scalar(1.0 - b2)?)?;
                let (bc1, bc2) = (1.0 - b1.powi(t), 1.0 - b2.powi(t));
                let mut update = m.div_scalar(bc1)?.div(&v.div_scalar(bc2)?.sqrt()?.add_scalar(self.eps)?)?;
                if self.weight_decay != 0.0 {
                    update = update.add(&p.mul_scalar(self.weight_decay)?)?;
                }
                let (wn, un) = (p.square()?.sum()?.item::<f64>()?.sqrt(), update.square()?.sum()?.item::<f64>()?.sqrt());
                let trust = if wn > 0.0 && un > 0.0 { wn / un } else { 1.0 };
                p.assign(&p.sub(&update.mul_scalar(self.lr * trust)?)?)?;
                self.exp_avg[i] = Some(m);
                self.exp_avg_sq[i] = Some(v);
            }
            Ok(())
        })
    }
    optimizer_boilerplate!();
    state_impl!(tensors: [("exp_avg", exp_avg), ("exp_avg_sq", exp_avg_sq)], scalars: [("step", t)]);
}

/// Yogi (Zaheer et al. 2018): Adam whose second moment grows or shrinks additively, which tames exploding gradients.
pub struct Yogi {
    params: Vec<Tensor>,
    lr: f64,
    betas: (f64, f64),
    eps: f64,
    exp_avg: Vec<Option<Tensor>>,
    exp_avg_sq: Vec<Option<Tensor>>,
    t: Vec<i32>,
}

impl Yogi {
    pub fn new(params: Vec<Tensor>, lr: f64) -> Self {
        let n = params.len();
        Self { params, lr, betas: (0.9, 0.999), eps: 1e-3, exp_avg: vec![None; n], exp_avg_sq: vec![None; n], t: vec![0; n] }
    }
    pub fn betas(mut self, b1: f64, b2: f64) -> Self {
        self.betas = (b1, b2);
        self
    }
    pub fn eps(mut self, eps: f64) -> Self {
        self.eps = eps;
        self
    }
}

impl Optimizer for Yogi {
    fn step(&mut self) -> Result<()> {
        check_betas("Yogi", self.betas.0, self.betas.1)?;
        no_grad(|| {
            for (i, p) in self.params.iter().enumerate() {
                let Some(g) = p.grad() else { continue };
                self.t[i] += 1;
                let t = self.t[i];
                let (b1, b2) = self.betas;
                let m = self.exp_avg[i].take().map_or_else(|| p.zeros_like(), Ok)?;
                let v = self.exp_avg_sq[i].take().map_or_else(|| p.full_like(1e-6), Ok)?;
                let m = m.mul_scalar(b1)?.add(&g.mul_scalar(1.0 - b1)?)?;
                let g2 = g.square()?;
                let v = v.sub(&v.sub(&g2)?.sign()?.mul(&g2)?.mul_scalar(1.0 - b2)?)?;
                let (bc1, bc2) = (1.0 - b1.powi(t), 1.0 - b2.powi(t));
                let denom = v.div_scalar(bc2)?.sqrt()?.add_scalar(self.eps)?;
                p.assign(&p.sub(&m.div_scalar(bc1)?.div(&denom)?.mul_scalar(self.lr)?)?)?;
                self.exp_avg[i] = Some(m);
                self.exp_avg_sq[i] = Some(v);
            }
            Ok(())
        })
    }
    optimizer_boilerplate!();
    state_impl!(tensors: [("exp_avg", exp_avg), ("exp_avg_sq", exp_avg_sq)], scalars: [("step", t)]);
}

/// Sharpness-aware minimization (Foret et al. 2021) around any base optimizer. Per iteration: compute gradients,
/// [`first_step`](Self::first_step) (climb to the worst point within radius `rho`), recompute gradients at the
/// perturbed weights, then [`second_step`](Self::second_step) (restore the weights and let the base optimizer move).
pub struct Sam {
    base: Box<dyn Optimizer>,
    params: Vec<Tensor>,
    pub rho: f64,
    perturbation: Vec<Option<Tensor>>,
}

impl Sam {
    pub fn new(params: Vec<Tensor>, base: Box<dyn Optimizer>, rho: f64) -> Self {
        let n = params.len();
        Self { base, params, rho, perturbation: vec![None; n] }
    }

    pub fn first_step(&mut self) -> Result<()> {
        no_grad(|| {
            let grads: Vec<Option<Tensor>> = self.params.iter().map(Tensor::grad).collect();
            let mut sq = 0.0;
            for g in grads.iter().flatten() {
                sq += g.square()?.sum()?.item::<f64>()?;
            }
            let scale = self.rho / (sq.sqrt() + 1e-12);
            for (i, p) in self.params.iter().enumerate() {
                if let Some(g) = &grads[i] {
                    let e = g.mul_scalar(scale)?;
                    p.assign(&p.add(&e)?)?;
                    self.perturbation[i] = Some(e);
                }
            }
            Ok(())
        })
    }

    pub fn second_step(&mut self) -> Result<()> {
        no_grad(|| -> Result<()> {
            for (p, e) in self.params.iter().zip(&mut self.perturbation) {
                if let Some(e) = e.take() {
                    p.assign(&p.sub(&e)?)?;
                }
            }
            Ok(())
        })?;
        self.base.step()
    }
}

impl Optimizer for Sam {
    /// A plain base-optimizer step (no perturbation); use `first_step` / `second_step` for SAM.
    fn step(&mut self) -> Result<()> {
        self.base.step()
    }
    fn zero_grad(&mut self) {
        self.base.zero_grad();
    }
    fn learning_rate(&self) -> f64 {
        self.base.learning_rate()
    }
    fn set_learning_rate(&mut self, lr: f64) {
        self.base.set_learning_rate(lr);
    }
    fn state_dict(&self) -> OptimizerState {
        self.base.state_dict()
    }
    fn load_state_dict(&mut self, state: &OptimizerState) -> Result<()> {
        self.base.load_state_dict(state)
    }
}

#[cfg(test)]
mod extra_optimizer_tests {
    use super::*;
    use crate::optim::Sgd;

    fn quad_grad(p: &Tensor) {
        p.set_grad(Some(p.detach().sub_scalar(3.0).unwrap().mul_scalar(2.0).unwrap()));
    }
    fn vals(t: &Tensor) -> Vec<f64> {
        t.to_vec::<f64>().unwrap()
    }

    type Maker = Box<dyn Fn(Vec<Tensor>) -> Box<dyn Optimizer>>;

    #[test]
    fn adabelief_and_yogi_converge_and_resume() {
        let makers: Vec<Maker> = vec![
            Box::new(|p| Box::new(AdaBelief::new(p, 0.1))),
            Box::new(|p| Box::new(Yogi::new(p, 0.1))),
        ];
        for make in makers {
            let p = Tensor::from_slice(&[0.0f64, 9.0, -2.0], &[3]).unwrap().requires_grad_(true);
            let mut opt = make(vec![p.clone()]);
            for _ in 0..600 {
                quad_grad(&p);
                opt.step().unwrap();
            }
            assert!(vals(&p).iter().all(|v| (v - 3.0).abs() < 0.05), "{:?}", vals(&p));
            let state = opt.state_dict();
            let q = Tensor::from_slice(&vals(&p), &[3]).unwrap().requires_grad_(true);
            let mut other = make(vec![q.clone()]);
            other.load_state_dict(&state).unwrap();
            quad_grad(&p);
            quad_grad(&q);
            opt.step().unwrap();
            other.step().unwrap();
            assert_eq!(vals(&p), vals(&q));
        }
    }

    #[test]
    fn sam_climbs_then_descends_from_the_original_point() {
        let p = Tensor::from_slice(&[1.0f64, 0.0], &[2]).unwrap().requires_grad_(true);
        let base = Box::new(Sgd::new(vec![p.clone()], 0.1));
        let mut sam = Sam::new(vec![p.clone()], base, 0.5);
        p.set_grad(Some(Tensor::from_slice(&[3.0f64, 4.0], &[2]).unwrap()));
        sam.first_step().unwrap();
        // moved by rho along the normalized gradient (0.6, 0.8)
        assert!((vals(&p)[0] - 1.3).abs() < 1e-9 && (vals(&p)[1] - 0.4).abs() < 1e-9);
        p.set_grad(Some(Tensor::from_slice(&[1.0f64, 1.0], &[2]).unwrap()));
        sam.second_step().unwrap();
        // restored to (1, 0) and stepped with the gradient taken at the perturbed point
        assert!((vals(&p)[0] - 0.9).abs() < 1e-9 && (vals(&p)[1] + 0.1).abs() < 1e-9);
    }
}

/// One step of `torch.optim.SparseAdam` on a dense gradient: only the rows (slices along dimension 0; elements of a
/// 1-D tensor) whose gradient is non-zero have their moments and values updated, so untouched embedding rows keep
/// both. Returns the new parameter values and moments in the parameter's dtype; `step` counts this step (1, 2, ...).
#[allow(clippy::too_many_arguments)]
pub fn sparse_adam_step(
    param: &Tensor,
    grad: &Tensor,
    exp_avg: &Tensor,
    exp_avg_sq: &Tensor,
    step: u64,
    lr: f64,
    beta1: f64,
    beta2: f64,
    eps: f64,
    maximize: bool,
) -> Result<(Tensor, Tensor, Tensor)> {
    let shape = param.shape().to_vec();
    if grad.shape() != shape.as_slice() || exp_avg.shape() != shape.as_slice() || exp_avg_sq.shape() != shape.as_slice() {
        return Err(Error::Msg("sparse_adam_step: param, grad and the moments must have the same shape".into()));
    }
    if step == 0 {
        return Err(Error::Msg("sparse_adam_step: step counts from 1".into()));
    }
    let mut p = param.to_vec::<f64>()?;
    let g = grad.to_vec::<f64>()?;
    let mut m = exp_avg.to_vec::<f64>()?;
    let mut v = exp_avg_sq.to_vec::<f64>()?;
    let rows = if shape.is_empty() { 1 } else { shape[0] };
    let row_len = p.len().checked_div(rows).unwrap_or(0);
    let bc1 = 1.0 - beta1.powi(step as i32);
    let bc2 = 1.0 - beta2.powi(step as i32);
    let step_size = lr * bc2.sqrt() / bc1;
    let sign = if maximize { -1.0 } else { 1.0 };
    for r in 0..rows {
        let span = r * row_len..(r + 1) * row_len;
        if g[span.clone()].iter().all(|&x| x == 0.0) {
            continue;
        }
        for i in span {
            let gi = sign * g[i];
            m[i] += (gi - m[i]) * (1.0 - beta1);
            v[i] += (gi * gi - v[i]) * (1.0 - beta2);
            p[i] -= step_size * m[i] / (v[i].sqrt() + eps);
        }
    }
    let dev = param.device();
    let dt = param.dtype();
    Ok((
        Tensor::from_slice_on(&p, &shape, &dev)?.to_dtype(dt)?,
        Tensor::from_slice_on(&m, &shape, &dev)?.to_dtype(dt)?,
        Tensor::from_slice_on(&v, &shape, &dev)?.to_dtype(dt)?,
    ))
}


#[cfg(test)]
mod lamb_tests {
    use super::*;

    fn quad_grad(p: &Tensor) {
        p.set_grad(Some(p.detach().sub_scalar(3.0).unwrap().mul_scalar(2.0).unwrap()));
    }
    fn vals(t: &Tensor) -> Vec<f64> {
        t.to_vec::<f64>().unwrap()
    }

    #[test]
    fn lamb_first_step_uses_the_trust_ratio_and_resumes() {
        let p = Tensor::from_slice(&[3.0f64, 4.0], &[2]).unwrap().requires_grad_(true);
        let mut opt = Lamb::new(vec![p.clone()], 0.1);
        p.set_grad(Some(Tensor::from_slice(&[0.5f64, -2.0], &[2]).unwrap()));
        opt.step().unwrap();
        // first Adam step is sign(g) (bias-corrected), norm sqrt(2); trust = ||w|| / ||u|| = 5 / sqrt(2)
        let step = 0.1 * 5.0 / 2f64.sqrt();
        assert!((vals(&p)[0] - (3.0 - step)).abs() < 1e-5 && (vals(&p)[1] - (4.0 + step)).abs() < 1e-5, "{:?}", vals(&p));
        // converges (with jitter proportional to lr * ||w||) and a resumed copy matches
        let p = Tensor::from_slice(&[0.5f64, 9.0, -2.0], &[3]).unwrap().requires_grad_(true);
        let mut opt = Lamb::new(vec![p.clone()], 0.01);
        for _ in 0..3000 {
            quad_grad(&p);
            opt.step().unwrap();
        }
        assert!(vals(&p).iter().all(|v| (v - 3.0).abs() < 0.3), "{:?}", vals(&p));
        let q = Tensor::from_slice(&vals(&p), &[3]).unwrap().requires_grad_(true);
        let mut other = Lamb::new(vec![q.clone()], 0.01);
        other.load_state_dict(&opt.state_dict()).unwrap();
        quad_grad(&p);
        quad_grad(&q);
        opt.step().unwrap();
        other.step().unwrap();
        assert_eq!(vals(&p), vals(&q));
    }
}

#[cfg(test)]
mod sparse_adam_tests {
    use super::*;

    #[test]
    fn untouched_rows_keep_values_and_moments() {
        let p = Tensor::from_slice(&[1.0f32; 6], &[3, 2]).unwrap();
        let g = Tensor::from_slice(&[0.0f32, 0.0, 1.0, 0.0, 0.0, 0.0], &[3, 2]).unwrap();
        let z = p.zeros_like().unwrap();
        let (np, nm, _) = sparse_adam_step(&p, &g, &z, &z, 1, 0.1, 0.9, 0.999, 1e-8, false).unwrap();
        let v = np.to_vec::<f32>().unwrap();
        assert_eq!(v[0], 1.0);
        assert_eq!(v[4], 1.0);
        assert!((v[2] - 0.9).abs() < 1e-6); // the first Adam step moves by lr
        assert_eq!(v[3], 1.0); // touched row, zero entry: moves by 0
        assert_eq!(nm.to_vec::<f32>().unwrap()[0], 0.0);
    }
}
