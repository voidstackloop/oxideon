//! A step-based training loop on top of [`PrefetchLoader`] and [`checkpoint`](crate::checkpoint): learning-rate schedule
//! function, gradient accumulation and clipping, EMA weights (used for evaluation), periodic evaluation, callbacks, and
//! automatic resume from the latest checkpoint at the exact epoch / batch where the previous run stopped.
//!
//! The older epoch-based [`Trainer`](crate::trainer::Trainer) is unchanged.

use crate::checkpoint::{CheckpointManager, TrainState};
use crate::data::Dataset;
use crate::error::Result;
use crate::loader::PrefetchLoader;
use crate::nn::Module;
use crate::optim::{Optimizer, clip_grad_norm};
use crate::summary_writer::SummaryWriter;
use crate::tensor::{Tensor, no_grad};
use std::collections::HashMap;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Control {
    Continue,
    Stop,
}

/// Passed to [`Callback::on_step`] after every optimizer step.
#[derive(Debug, Clone)]
pub struct StepInfo {
    pub epoch: u64,
    pub step: u64,
    /// Mean loss of the micro-batches that made up this step.
    pub loss: f64,
    pub lr: f64,
    pub grad_norm: Option<f64>,
}

pub trait Callback {
    fn on_step(&mut self, _info: &StepInfo, _model: &dyn Module) -> Result<Control> {
        Ok(Control::Continue)
    }
    /// `val_loss` is the evaluation loss of the epoch (with EMA weights when enabled), if there is a validation loader.
    fn on_epoch_end(&mut self, _epoch: u64, _val_loss: Option<f64>, _model: &dyn Module) -> Result<Control> {
        Ok(Control::Continue)
    }
}

/// Writes loss / learning rate / grad-norm / validation loss to a TensorBoard event file.
pub struct TensorBoardLogger {
    pub writer: SummaryWriter,
    pub every: u64,
}

impl Callback for TensorBoardLogger {
    fn on_step(&mut self, i: &StepInfo, _m: &dyn Module) -> Result<Control> {
        if i.step % self.every.max(1) == 0 {
            self.writer.add_scalar("train/loss", i.loss, i.step as i64)?;
            self.writer.add_scalar("train/lr", i.lr, i.step as i64)?;
            if let Some(g) = i.grad_norm {
                self.writer.add_scalar("train/grad_norm", g, i.step as i64)?;
            }
        }
        Ok(Control::Continue)
    }
    fn on_epoch_end(&mut self, epoch: u64, val: Option<f64>, _m: &dyn Module) -> Result<Control> {
        if let Some(v) = val {
            self.writer.add_scalar("val/loss", v, epoch as i64)?;
        }
        self.writer.flush()?;
        Ok(Control::Continue)
    }
}

/// Exponential moving average of the parameters.
#[derive(Clone)]
pub struct Ema {
    pub decay: f64,
    pub shadow: HashMap<String, Tensor>,
    backup: HashMap<String, Tensor>,
}

impl Ema {
    pub fn new(model: &dyn Module, decay: f64) -> Result<Self> {
        let shadow = model.named_parameters().into_iter().map(|(k, p)| Ok((k, p.detach().contiguous()?.mul_scalar(1.0)?))).collect::<Result<_>>()?;
        Ok(Self { decay, shadow, backup: HashMap::new() })
    }

    pub fn update(&mut self, model: &dyn Module) -> Result<()> {
        no_grad(|| {
            for (k, p) in model.named_parameters() {
                if let Some(s) = self.shadow.get_mut(&k) {
                    *s = s.mul_scalar(self.decay)?.add(&p.detach().mul_scalar(1.0 - self.decay)?)?;
                }
            }
            Ok(())
        })
    }

    /// Replaces the model's parameters by the averages (call [`restore`](Self::restore) afterwards).
    pub fn swap_in(&mut self, model: &dyn Module) -> Result<()> {
        self.backup.clear();
        for (k, p) in model.named_parameters() {
            if let Some(s) = self.shadow.get(&k) {
                self.backup.insert(k, p.detach().contiguous()?.mul_scalar(1.0)?);
                p.assign(s)?;
            }
        }
        Ok(())
    }

    pub fn restore(&mut self, model: &dyn Module) -> Result<()> {
        for (k, p) in model.named_parameters() {
            if let Some(b) = self.backup.get(&k) {
                p.assign(b)?;
            }
        }
        self.backup.clear();
        Ok(())
    }
}

const EMA_PREFIX: &str = "__ema__.";

pub struct LoopConfig {
    pub epochs: u64,
    pub max_steps: Option<u64>,
    pub grad_accum: usize,
    pub max_grad_norm: Option<f64>,
    /// Learning rate for optimizer step `n` (0-based); `None` keeps the optimizer's own rate.
    pub lr_schedule: Option<Box<dyn Fn(u64) -> f64>>,
    pub ema_decay: Option<f64>,
    /// Save a checkpoint every this many optimizer steps (needs a checkpoint manager).
    pub checkpoint_every: u64,
}

impl Default for LoopConfig {
    fn default() -> Self {
        Self { epochs: 1, max_steps: None, grad_accum: 1, max_grad_norm: None, lr_schedule: None, ema_decay: None, checkpoint_every: 0 }
    }
}

#[derive(Debug, Clone, Default)]
pub struct FitReport {
    pub steps: u64,
    pub epochs_done: u64,
    pub step_losses: Vec<f64>,
    pub val_losses: Vec<f64>,
    pub stopped_by_callback: bool,
    pub resumed_from_step: Option<u64>,
}

pub struct TrainLoop {
    pub config: LoopConfig,
    pub callbacks: Vec<Box<dyn Callback>>,
    pub checkpoints: Option<CheckpointManager>,
    pub ema: Option<Ema>,
}

impl TrainLoop {
    pub fn new(config: LoopConfig) -> Self {
        Self { config, callbacks: Vec::new(), checkpoints: None, ema: None }
    }

    fn evaluate<M: Module, D: Dataset + Send + Sync + 'static>(&mut self, model: &M, loader: &PrefetchLoader<D>, loss_fn: &dyn Fn(&Tensor, &Tensor) -> Result<Tensor>) -> Result<Option<f64>> {
        if let Some(e) = &mut self.ema {
            e.swap_in(model)?;
        }
        model.eval();
        let r = no_grad(|| -> Result<(f64, usize)> {
            let (mut sum, mut n) = (0.0, 0usize);
            for b in loader.epoch(0) {
                let (x, y) = b?;
                sum += loss_fn(&model.forward(&x)?, &y)?.item::<f64>()?;
                n += 1;
            }
            Ok((sum, n))
        });
        if let Some(e) = &mut self.ema {
            e.restore(model)?;
        }
        model.train(true);
        let (sum, n) = r?;
        Ok((n > 0).then(|| sum / n as f64))
    }

    fn snapshot<M: Module, O: Optimizer>(&self, model: &M, opt: &O, epoch: u64, step: u64, batch: u64) -> Result<TrainState> {
        let mut st = TrainState::capture(model, Some(opt), epoch, step, batch)?;
        if let Some(e) = &self.ema {
            for (k, t) in &e.shadow {
                st.model.insert(format!("{EMA_PREFIX}{k}"), t.detach().contiguous()?.mul_scalar(1.0)?);
            }
        }
        Ok(st)
    }

    pub fn fit<M, O, D, L>(&mut self, model: &M, opt: &mut O, train: &PrefetchLoader<D>, val: Option<&PrefetchLoader<D>>, loss_fn: L) -> Result<FitReport>
    where
        M: Module,
        O: Optimizer,
        D: Dataset + Send + Sync + 'static,
        L: Fn(&Tensor, &Tensor) -> Result<Tensor>,
    {
        let accum = self.config.grad_accum.max(1);
        let params = model.parameters();
        let mut report = FitReport::default();
        let (mut epoch, mut step, mut skip) = (0u64, 0u64, 0u64);
        if let Some(d) = self.config.ema_decay {
            self.ema = Some(Ema::new(model, d)?);
        }
        if let Some(mgr) = &mut self.checkpoints
            && let Some(st) = mgr.latest()?
        {
            st.restore(model, Some(opt))?;
            if let Some(e) = &mut self.ema {
                for (k, t) in &st.model {
                    if let Some(name) = k.strip_prefix(EMA_PREFIX) {
                        e.shadow.insert(name.to_string(), t.clone());
                    }
                }
            }
            (epoch, step, skip) = (st.epoch, st.step, st.batch_in_epoch);
            report.resumed_from_step = Some(step);
        }
        let per_epoch = train.num_batches() as u64;
        if per_epoch == 0 {
            return Err(crate::error::Error::Msg("TrainLoop::fit: the training loader yields no batches".into()));
        }
        model.train(true);
        let done = |step: u64, cfg: &LoopConfig| cfg.max_steps.is_some_and(|m| step >= m);
        'epochs: while epoch < self.config.epochs && !done(step, &self.config) {
            let (mut micro, mut in_step, mut loss_acc) = (skip, 0usize, 0.0f64);
            opt.zero_grad();
            for b in train.epoch_from(epoch, skip as usize) {
                let (x, y) = b?;
                let loss = loss_fn(&model.forward(&x)?, &y)?;
                loss_acc += loss.item::<f64>()?;
                (if accum > 1 { loss.div_scalar(accum as f64)? } else { loss }).backward()?;
                micro += 1;
                in_step += 1;
                if in_step == accum || micro == per_epoch {
                    let lr = match &self.config.lr_schedule {
                        Some(f) => {
                            let lr = f(step);
                            opt.set_learning_rate(lr);
                            lr
                        }
                        None => opt.learning_rate(),
                    };
                    let grad_norm = self.config.max_grad_norm.map(|m| clip_grad_norm(&params, m)).transpose()?;
                    opt.step()?;
                    opt.zero_grad();
                    step += 1;
                    if let Some(e) = &mut self.ema {
                        e.update(model)?;
                    }
                    let info = StepInfo { epoch, step, loss: loss_acc / in_step as f64, lr, grad_norm };
                    report.step_losses.push(info.loss);
                    (in_step, loss_acc) = (0, 0.0);
                    let epoch_over = micro == per_epoch;
                    if self.config.checkpoint_every > 0 && step % self.config.checkpoint_every == 0 && self.checkpoints.is_some() {
                        let (e, m) = if epoch_over { (epoch + 1, 0) } else { (epoch, micro) };
                        let st = self.snapshot(model, opt, e, step, m)?;
                        self.checkpoints.as_mut().unwrap().save(&st, false)?;
                    }
                    let mut stop = done(step, &self.config);
                    for cb in &mut self.callbacks {
                        stop |= cb.on_step(&info, model)? == Control::Stop;
                        report.stopped_by_callback |= stop && !done(step, &self.config);
                    }
                    if stop {
                        break 'epochs;
                    }
                }
            }
            epoch += 1;
            skip = 0;
            report.epochs_done += 1;
            let v = match val {
                Some(l) => self.evaluate(model, l, &loss_fn)?,
                None => None,
            };
            if let Some(v) = v {
                report.val_losses.push(v);
            }
            let mut stop = false;
            for cb in &mut self.callbacks {
                stop |= cb.on_epoch_end(epoch - 1, v, model)? == Control::Stop;
            }
            if stop {
                report.stopped_by_callback = true;
                break;
            }
        }
        report.steps = step;
        if let Some(m) = &mut self.checkpoints {
            m.wait()?;
        }
        Ok(report)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::TensorDataset;
    use crate::nn::Linear;
    use crate::optim::Adam;
    use std::sync::Arc;

    fn data(n: usize) -> Arc<TensorDataset> {
        let x: Vec<f32> = (0..n * 2).map(|i| ((i * 7 % 11) as f32 - 5.0) / 5.0).collect();
        let y: Vec<f32> = x.chunks(2).map(|c| 0.7 * c[0] - 0.3 * c[1]).collect();
        Arc::new(TensorDataset::new(Tensor::from_slice(&x, &[n, 2]).unwrap(), Tensor::from_slice(&y, &[n, 1]).unwrap()))
    }

    fn setup() -> (Linear, Adam) {
        crate::rng::manual_seed(5);
        let m = Linear::new(2, 1).unwrap();
        let o = Adam::new(m.parameters(), 0.1);
        (m, o)
    }

    fn mse(p: &Tensor, t: &Tensor) -> Result<Tensor> {
        p.sub(t)?.square()?.mean()
    }

    fn cfg(max_steps: u64) -> LoopConfig {
        LoopConfig { epochs: 100, max_steps: Some(max_steps), grad_accum: 2, max_grad_norm: Some(1.0), lr_schedule: Some(Box::new(|s| 0.1 / (1.0 + s as f64 * 0.1))), ema_decay: Some(0.9), checkpoint_every: 1 }
    }

    #[test]
    fn interrupted_and_resumed_run_equals_an_uninterrupted_one() {
        let ds = data(20);
        let loader = PrefetchLoader::new(ds, 4, true, 1).workers(3).drop_last(false); // 5 micro-batches per epoch, accum 2

        let (m, mut o) = setup();
        let mut full = TrainLoop::new(cfg(12));
        let r_full = full.fit(&m, &mut o, &loader, None, mse).unwrap();
        assert_eq!(r_full.steps, 12);
        assert!(r_full.step_losses.last() < r_full.step_losses.first());
        let w_full = m.weight.to_vec::<f32>().unwrap();
        let ema_full = full.ema.as_ref().unwrap().shadow["weight"].to_vec::<f32>().unwrap();

        // run 1 stops after 5 steps ("crash"); run 2 is a fresh model/optimizer that resumes and finishes
        let dir = std::env::temp_dir().join("oxideon_trainloop_ckpt");
        let _ = std::fs::remove_dir_all(&dir);
        let (m1, mut o1) = setup();
        let mut a = TrainLoop::new(cfg(5));
        a.checkpoints = Some(CheckpointManager::new(&dir, 2).unwrap());
        let r1 = a.fit(&m1, &mut o1, &loader, None, mse).unwrap();
        assert_eq!(r1.steps, 5);
        crate::rng::manual_seed(77);
        let m2 = Linear::new(2, 1).unwrap();
        let mut o2 = Adam::new(m2.parameters(), 0.5);
        let mut b = TrainLoop::new(cfg(12));
        b.checkpoints = Some(CheckpointManager::new(&dir, 2).unwrap());
        let r2 = b.fit(&m2, &mut o2, &loader, None, mse).unwrap();
        assert_eq!(r2.resumed_from_step, Some(5));
        assert_eq!([r1.step_losses.clone(), r2.step_losses.clone()].concat(), r_full.step_losses);
        assert_eq!(m2.weight.to_vec::<f32>().unwrap(), w_full);
        assert_eq!(b.ema.as_ref().unwrap().shadow["weight"].to_vec::<f32>().unwrap(), ema_full);
        let _ = std::fs::remove_dir_all(&dir);
    }

    struct StopAt(u64);
    impl Callback for StopAt {
        fn on_step(&mut self, i: &StepInfo, _m: &dyn Module) -> Result<Control> {
            Ok(if i.step >= self.0 { Control::Stop } else { Control::Continue })
        }
    }

    #[test]
    fn callbacks_stop_training_evaluation_uses_ema_and_weights_are_restored() {
        let ds = data(16);
        let train = PrefetchLoader::new(ds.clone(), 4, false, 0);
        let val = PrefetchLoader::new(ds, 8, false, 0);
        let (m, mut o) = setup();
        let mut l = TrainLoop::new(LoopConfig { epochs: 3, ema_decay: Some(0.5), ..LoopConfig::default() });
        let tb_dir = std::env::temp_dir().join("oxideon_trainloop_tb");
        let _ = std::fs::remove_dir_all(&tb_dir);
        l.callbacks.push(Box::new(TensorBoardLogger { writer: SummaryWriter::new(&tb_dir).unwrap(), every: 1 }));
        let r = l.fit(&m, &mut o, &train, Some(&val), mse).unwrap();
        assert_eq!((r.steps, r.epochs_done, r.val_losses.len()), (12, 3, 3));
        // evaluation swapped the EMA in and restored the live weights afterwards
        let live = m.weight.to_vec::<f32>().unwrap();
        let ema = l.ema.as_ref().unwrap().shadow["weight"].to_vec::<f32>().unwrap();
        assert_ne!(live, ema);
        l.ema.as_mut().unwrap().swap_in(&m).unwrap();
        assert_eq!(m.weight.to_vec::<f32>().unwrap(), ema);
        l.ema.as_mut().unwrap().restore(&m).unwrap();
        assert_eq!(m.weight.to_vec::<f32>().unwrap(), live);
        let _ = std::fs::remove_dir_all(&tb_dir);

        let (m, mut o) = setup();
        let mut s = TrainLoop::new(LoopConfig { epochs: 10, ..LoopConfig::default() });
        s.callbacks.push(Box::new(StopAt(3)));
        let r = s.fit(&m, &mut o, &train, None, mse).unwrap();
        assert_eq!((r.steps, r.stopped_by_callback), (3, true));
    }
}
