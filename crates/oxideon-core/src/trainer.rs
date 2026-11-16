//! Declarative [`Trainer`] engine with gradient accumulation, gradient norm clipping,
//! validation evaluation, [`EarlyStopping`], and in-memory `.safetensors` best-checkpoint restoration.

use crate::data::{DataLoader, Dataset};
use crate::error::Result;
use crate::nn::Module;
use crate::optim::{Optimizer, clip_grad_norm};
use crate::serialize::{deserialize_safetensors, serialize_safetensors};
use crate::tensor::{Tensor, no_grad};

/// Configuration for [`Trainer`].
#[derive(Debug, Clone)]
pub struct TrainerConfig {
    pub epochs: usize,
    pub grad_accum_steps: usize,
    pub max_grad_norm: Option<f64>,
    pub early_stopping_patience: Option<usize>,
    pub min_delta: f64,
    pub restore_best_weights: bool,
}

impl Default for TrainerConfig {
    fn default() -> Self {
        Self {
            epochs: 10,
            grad_accum_steps: 1,
            max_grad_norm: None,
            early_stopping_patience: None,
            min_delta: 1e-5,
            restore_best_weights: true,
        }
    }
}

impl TrainerConfig {
    pub fn new(epochs: usize) -> Self {
        Self {
            epochs,
            ..Self::default()
        }
    }

    pub fn grad_accum_steps(mut self, steps: usize) -> Self {
        self.grad_accum_steps = steps.max(1);
        self
    }

    pub fn max_grad_norm(mut self, max_norm: f64) -> Self {
        self.max_grad_norm = Some(max_norm);
        self
    }

    pub fn early_stopping(mut self, patience: usize, min_delta: f64) -> Self {
        self.early_stopping_patience = Some(patience);
        self.min_delta = min_delta;
        self
    }
}

/// Per-epoch metrics recorded during [`Trainer::fit`].
#[derive(Debug, Clone)]
pub struct EpochMetrics {
    pub epoch: usize,
    pub train_loss: f64,
    pub val_loss: Option<f64>,
    pub learning_rate: f64,
}

/// Complete history returned by [`Trainer::fit`].
#[derive(Debug, Clone)]
pub struct TrainingHistory {
    pub metrics: Vec<EpochMetrics>,
    pub best_loss: f64,
    pub best_epoch: usize,
    pub stopped_early: bool,
    pub best_checkpoint_bytes: Option<Vec<u8>>,
}

/// Standalone early-stopping monitor.
#[derive(Debug, Clone)]
pub struct EarlyStopping {
    pub patience: usize,
    pub min_delta: f64,
    pub best_loss: f64,
    pub wait_count: usize,
}

impl EarlyStopping {
    pub fn new(patience: usize, min_delta: f64) -> Self {
        Self {
            patience,
            min_delta,
            best_loss: f64::INFINITY,
            wait_count: 0,
        }
    }

    /// Updates state with `current_loss` and returns `(is_new_best, should_stop)`.
    pub fn step(&mut self, current_loss: f64) -> (bool, bool) {
        if current_loss < self.best_loss - self.min_delta {
            self.best_loss = current_loss;
            self.wait_count = 0;
            (true, false)
        } else {
            self.wait_count += 1;
            (false, self.wait_count > self.patience)
        }
    }
}

/// High-level training engine orchestrating training/validation loops, gradient accumulation,
/// gradient clipping, early stopping, and `.safetensors` best-model checkpointing.
pub struct Trainer {
    pub config: TrainerConfig,
}

impl Trainer {
    pub fn new(config: TrainerConfig) -> Self {
        Self { config }
    }

    pub fn fit<M, O, D, L>(
        &self,
        model: &M,
        optimizer: &mut O,
        train_loader: &DataLoader<'_, D>,
        val_loader: Option<&DataLoader<'_, D>>,
        loss_fn: L,
    ) -> Result<TrainingHistory>
    where
        M: Module,
        O: Optimizer,
        D: Dataset,
        L: Fn(&Tensor, &Tensor) -> Result<Tensor>,
    {
        let accum = self.config.grad_accum_steps.max(1);
        let epochs = crate::limits::check_iterations("Trainer::fit epochs", self.config.epochs)?;
        let params = model.parameters();
        // Reserve boundedly: `epochs` can be large-but-legal (up to 1e8).
        let mut metrics = Vec::with_capacity(epochs.min(1024));
        let mut early_stop = self
            .config
            .early_stopping_patience
            .map(|p| EarlyStopping::new(p, self.config.min_delta));
        let mut best_loss = f64::INFINITY;
        let mut best_epoch = 0usize;
        let mut best_checkpoint_bytes: Option<Vec<u8>> = None;
        let mut stopped_early = false;

        for epoch in 0..epochs {
            model.train(true);
            optimizer.zero_grad();

            let mut total_train_loss = 0.0f64;
            let mut num_batches = 0usize;

            for (step_idx, batch_res) in train_loader.iter().enumerate() {
                let (bx, by) = batch_res?;
                let preds = model.forward(&bx)?;
                let raw_loss = loss_fn(&preds, &by)?;
                total_train_loss += raw_loss.item::<f64>()?;
                num_batches += 1;

                let scaled_loss = if accum > 1 {
                    raw_loss.div_scalar(accum as f64)?
                } else {
                    raw_loss
                };
                scaled_loss.backward()?;

                if (step_idx + 1) % accum == 0 || (step_idx + 1) == train_loader.num_batches() {
                    if let Some(max_norm) = self.config.max_grad_norm {
                        clip_grad_norm(&params, max_norm)?;
                    }
                    optimizer.step()?;
                    optimizer.zero_grad();
                }
            }

            if num_batches == 0 {
                return Err(crate::error::Error::Msg(
                    "Trainer::fit: train loader yielded no batches (empty dataset?)".into(),
                ));
            }
            let train_loss = total_train_loss / (num_batches as f64);

            let val_loss = if let Some(vl) = val_loader {
                model.eval();
                let mut v_sum = 0.0f64;
                let mut v_batches = 0usize;
                no_grad(|| -> Result<()> {
                    for batch_res in vl.iter() {
                        let (vx, vy) = batch_res?;
                        let v_preds = model.forward(&vx)?;
                        let v_l = loss_fn(&v_preds, &vy)?;
                        v_sum += v_l.item::<f64>()?;
                        v_batches += 1;
                    }
                    Ok(())
                })?;
                // An empty val loader means "no validation signal" (None), not
                // a perfect 0.0 loss.
                if v_batches == 0 {
                    None
                } else {
                    Some(v_sum / (v_batches as f64))
                }
            } else {
                None
            };

            let monitored = val_loss.unwrap_or(train_loss);
            if monitored < best_loss - self.config.min_delta {
                best_loss = monitored;
                best_epoch = epoch;
                if self.config.restore_best_weights {
                    best_checkpoint_bytes = Some(serialize_safetensors(&model.state_dict())?);
                }
            }

            metrics.push(EpochMetrics {
                epoch,
                train_loss,
                val_loss,
                learning_rate: optimizer.learning_rate(),
            });

            if let Some(es) = &mut early_stop {
                let (_is_best, should_stop) = es.step(monitored);
                if should_stop {
                    stopped_early = true;
                    break;
                }
            }
        }

        if self.config.restore_best_weights
            && let Some(bytes) = &best_checkpoint_bytes
        {
            let best_state = deserialize_safetensors(bytes)?;
            model.load_state_dict(&best_state)?;
        }

        Ok(TrainingHistory {
            metrics,
            best_loss,
            best_epoch,
            stopped_early,
            best_checkpoint_bytes,
        })
    }
}
