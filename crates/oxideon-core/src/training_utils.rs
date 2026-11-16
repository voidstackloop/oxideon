//! Training-loop helpers: [`EarlyStopping`], a best-value [`BestTracker`], [`GradAccumulator`] (gradient
//! accumulation over micro-batches), the learning-rate range test [`lr_find`], and a small generic [`fit`] loop that
//! ties a model, optimizer, scheduler and these helpers together.

use crate::error::{Error, Result};
use crate::optim::Optimizer;
use crate::tensor::Tensor;

/// Whether smaller or larger metric values are better.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Goal {
    Minimize,
    Maximize,
}

/// Stops training when a monitored metric has not improved by at least `min_delta` for `patience` evaluations.
pub struct EarlyStopping {
    pub patience: usize,
    pub min_delta: f64,
    pub goal: Goal,
    best: Option<f64>,
    bad: usize,
}

impl EarlyStopping {
    pub fn new(patience: usize, min_delta: f64, goal: Goal) -> Self {
        Self { patience, min_delta, goal, best: None, bad: 0 }
    }

    /// Records a metric value; returns `true` when training should stop.
    pub fn update(&mut self, value: f64) -> bool {
        let improved = match (self.best, self.goal) {
            (None, _) => true,
            (Some(b), Goal::Minimize) => value < b - self.min_delta,
            (Some(b), Goal::Maximize) => value > b + self.min_delta,
        };
        if improved {
            self.best = Some(value);
            self.bad = 0;
        } else {
            self.bad += 1;
        }
        self.bad >= self.patience
    }

    pub fn best(&self) -> Option<f64> {
        self.best
    }
}

/// Remembers the best metric seen and a snapshot of the parameters at that point.
pub struct BestTracker {
    pub goal: Goal,
    best: Option<f64>,
    snapshot: Vec<Tensor>,
    pub best_epoch: usize,
}

impl BestTracker {
    pub fn new(goal: Goal) -> Self {
        Self { goal, best: None, snapshot: Vec::new(), best_epoch: 0 }
    }

    /// Offers a new value; on improvement copies `params` and returns `true`.
    pub fn update(&mut self, epoch: usize, value: f64, params: &[Tensor]) -> Result<bool> {
        let better = match (self.best, self.goal) {
            (None, _) => true,
            (Some(b), Goal::Minimize) => value < b,
            (Some(b), Goal::Maximize) => value > b,
        };
        if better {
            self.best = Some(value);
            self.best_epoch = epoch;
            self.snapshot = params.iter().map(|p| p.detach().mul_scalar(1.0)).collect::<Result<_>>()?;
        }
        Ok(better)
    }

    pub fn best(&self) -> Option<f64> {
        self.best
    }

    /// Copies the saved parameters back into `params`.
    pub fn restore(&self, params: &[Tensor]) -> Result<()> {
        if self.snapshot.len() != params.len() {
            return Err(Error::Msg("BestTracker::restore: no snapshot for these parameters".into()));
        }
        crate::tensor::no_grad(|| params.iter().zip(&self.snapshot).try_for_each(|(p, s)| p.assign(s)))
    }
}

/// Accumulates gradients over `steps` micro-batches before each optimizer step. Scale every micro-batch loss with
/// [`scale`](Self::scale) before `backward`, then call [`step`](Self::step).
pub struct GradAccumulator {
    steps: usize,
    seen: usize,
}

impl GradAccumulator {
    pub fn new(steps: usize) -> Self {
        Self { steps: steps.max(1), seen: 0 }
    }

    /// Divides a micro-batch loss by the number of accumulation steps so the summed gradient is the mean.
    pub fn scale(&self, loss: &Tensor) -> Result<Tensor> {
        loss.div_scalar(self.steps as f64)
    }

    /// Call after each micro-batch `backward`; steps and clears gradients every `steps` calls. Returns whether the
    /// optimizer stepped.
    pub fn step(&mut self, opt: &mut dyn Optimizer) -> Result<bool> {
        self.seen += 1;
        if self.seen % self.steps == 0 {
            opt.step()?;
            opt.zero_grad();
            return Ok(true);
        }
        Ok(false)
    }
}

/// Result of [`lr_find`].
pub struct LrFindResult {
    pub lrs: Vec<f64>,
    pub losses: Vec<f64>,
}

impl LrFindResult {
    /// The learning rate at the steepest descent of the smoothed loss curve (a common starting point is a bit below).
    pub fn suggestion(&self) -> Option<f64> {
        if self.losses.len() < 3 {
            return None;
        }
        let slopes: Vec<f64> = self.losses.windows(2).map(|w| w[1] - w[0]).collect();
        let (i, _) = slopes.iter().enumerate().min_by(|a, b| a.1.total_cmp(b.1))?;
        Some(self.lrs[i + 1])
    }
}

/// Learning-rate range test: sweeps the rate geometrically from `start` to `end` over `steps` calls of `step`
/// (which must run one training step at the given rate and return the loss). Stops early once the loss exceeds
/// `diverge_factor` times the best loss seen. Losses are exponentially smoothed with `smoothing`.
pub fn lr_find(mut step: impl FnMut(f64) -> Result<f64>, start: f64, end: f64, steps: usize, smoothing: f64, diverge_factor: f64) -> Result<LrFindResult> {
    if !(start > 0.0 && end > start) || steps < 2 {
        return Err(Error::Msg("lr_find: need 0 < start < end and at least 2 steps".into()));
    }
    let ratio = (end / start).powf(1.0 / (steps - 1) as f64);
    let (mut lrs, mut losses) = (Vec::new(), Vec::new());
    let (mut smooth, mut best) = (None::<f64>, f64::INFINITY);
    let mut lr = start;
    for i in 0..steps {
        let raw = step(lr)?;
        let s = match smooth {
            None => raw,
            Some(prev) => smoothing * prev + (1.0 - smoothing) * raw,
        };
        smooth = Some(s);
        lrs.push(lr);
        losses.push(s);
        best = best.min(s);
        if !s.is_finite() || (i > 0 && s > diverge_factor * best) {
            break;
        }
        lr *= ratio;
    }
    Ok(LrFindResult { lrs, losses })
}

/// Per-epoch record produced by [`fit`].
#[derive(Clone, Debug)]
pub struct EpochLog {
    pub epoch: usize,
    pub train_loss: f64,
    pub val_loss: Option<f64>,
    pub lr: f64,
}

/// Generic training loop. `train_epoch` runs one epoch (stepping `opt`) and returns the mean training loss;
/// `validate`, if given, returns the validation loss used for early stopping and best-model tracking.
/// The scheduler closure maps the epoch to a learning rate.
#[allow(clippy::too_many_arguments)]
pub fn fit(
    epochs: usize,
    params: &[Tensor],
    opt: &mut dyn Optimizer,
    mut train_epoch: impl FnMut(usize, &mut dyn Optimizer) -> Result<f64>,
    mut validate: Option<&mut dyn FnMut() -> Result<f64>>,
    schedule: Option<&dyn Fn(usize) -> f64>,
    mut early_stopping: Option<EarlyStopping>,
    restore_best: bool,
) -> Result<Vec<EpochLog>> {
    let mut logs = Vec::new();
    let mut best = BestTracker::new(Goal::Minimize);
    for epoch in 0..epochs {
        if let Some(s) = schedule {
            opt.set_learning_rate(s(epoch));
        }
        let train_loss = train_epoch(epoch, opt)?;
        let val_loss = match validate.as_mut() {
            Some(v) => Some(v()?),
            None => None,
        };
        logs.push(EpochLog { epoch, train_loss, val_loss, lr: opt.learning_rate() });
        let monitored = val_loss.unwrap_or(train_loss);
        best.update(epoch, monitored, params)?;
        if let Some(es) = early_stopping.as_mut() {
            if es.update(monitored) {
                break;
            }
        }
    }
    if restore_best && best.best().is_some() {
        best.restore(params)?;
    }
    Ok(logs)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::optim::Sgd;

    #[test]
    fn early_stopping_and_best_tracking() {
        let mut es = EarlyStopping::new(2, 0.01, Goal::Minimize);
        assert!(!es.update(1.0) && !es.update(0.5) && !es.update(0.499) && es.update(0.6));
        assert_eq!(es.best(), Some(0.5));
        let mut up = EarlyStopping::new(1, 0.0, Goal::Maximize);
        assert!(!up.update(0.5) && !up.update(0.6) && up.update(0.55));
        let p = Tensor::from_slice(&[1.0f64], &[1]).unwrap().requires_grad_(true);
        let mut bt = BestTracker::new(Goal::Minimize);
        assert!(bt.update(0, 2.0, std::slice::from_ref(&p)).unwrap());
        p.assign(&Tensor::from_slice(&[9.0f64], &[1]).unwrap()).unwrap();
        assert!(!bt.update(1, 3.0, std::slice::from_ref(&p)).unwrap());
        bt.restore(std::slice::from_ref(&p)).unwrap();
        assert_eq!(p.to_vec::<f64>().unwrap(), vec![1.0]);
        assert!(bt.restore(&[]).is_err());
        assert_eq!(bt.best_epoch, 0);
    }

    #[test]
    fn accumulation_matches_one_large_batch() {
        let make = || Tensor::from_slice(&[1.0f64], &[1]).unwrap().requires_grad_(true);
        let (a, b) = (make(), make());
        let data = [1.0, 2.0, 3.0, 4.0];
        // one batch of four
        let mut opt_a = Sgd::new(vec![a.clone()], 0.1);
        let loss: Tensor = data.iter().map(|&d| a.sub_scalar(d).unwrap().square().unwrap()).fold(None, |acc: Option<Tensor>, l| Some(match acc { Some(s) => s.add(&l).unwrap(), None => l })).unwrap().div_scalar(4.0).unwrap();
        loss.backward().unwrap();
        opt_a.step().unwrap();
        // four micro-batches of one
        let mut opt_b = Sgd::new(vec![b.clone()], 0.1);
        let mut acc = GradAccumulator::new(4);
        let mut stepped = Vec::new();
        for &d in &data {
            let l = acc.scale(&b.sub_scalar(d).unwrap().square().unwrap().sum().unwrap()).unwrap();
            l.backward().unwrap();
            stepped.push(acc.step(&mut opt_b).unwrap());
        }
        assert_eq!(stepped, vec![false, false, false, true]);
        assert!((a.to_vec::<f64>().unwrap()[0] - b.to_vec::<f64>().unwrap()[0]).abs() < 1e-12);
    }

    #[test]
    fn lr_range_test_finds_the_descent_region() {
        // loss falls until lr = 0.1 and diverges after
        let r = lr_find(|lr| Ok(if lr < 0.1 { 2.0 - 10.0 * lr } else { 1.0 + (lr - 0.1) * 1000.0 }), 1e-4, 1.0, 40, 0.0, 4.0).unwrap();
        assert!(r.lrs.len() < 40, "diverging runs stop early");
        let s = r.suggestion().unwrap();
        assert!(s > 1e-3 && s < 0.5, "{s}");
        assert!(lr_find(|_| Ok(0.0), 1.0, 0.5, 10, 0.0, 4.0).is_err());
    }

    #[test]
    fn fit_loop_with_schedule_and_early_stopping() {
        let w = Tensor::from_slice(&[5.0f64], &[1]).unwrap().requires_grad_(true);
        let mut opt = Sgd::new(vec![w.clone()], 0.4);
        let target = 1.0;
        let mut val_calls = 0;
        let mut val = || -> Result<f64> {
            val_calls += 1;
            // validation loss gets worse after the 4th call (overfitting)
            Ok(if val_calls <= 4 { (w.to_vec::<f64>()?[0] - target).powi(2) } else { 10.0 + val_calls as f64 })
        };
        let sched = |e: usize| 0.4 * 0.9f64.powi(e as i32);
        let logs = fit(
            50,
            std::slice::from_ref(&w),
            &mut opt,
            |_, o| {
                o.zero_grad();
                let loss = w.sub_scalar(target)?.square()?.sum()?;
                loss.backward()?;
                o.step()?;
                loss.item::<f64>()
            },
            Some(&mut val),
            Some(&sched),
            Some(EarlyStopping::new(2, 0.0, Goal::Minimize)),
            true,
        )
        .unwrap();
        assert_eq!(logs.len(), 6); // 4 improving epochs + 2 without improvement
        assert!((logs[1].lr - 0.4 * 0.9).abs() < 1e-12);
        // the parameters were restored to the best validation epoch (epoch 3)
        let expected = (0..4).fold(5.0f64, |x, e| x - 0.4 * 0.9f64.powi(e) * 2.0 * (x - 1.0));
        assert!((w.to_vec::<f64>().unwrap()[0] - expected).abs() < 1e-9);
    }
}
