//! Distributed training & model parallelism primitives:
//! - **Collective Operations**: [`ReduceOp`], [`all_reduce_tensors`], [`all_gather_tensors`], [`reduce_scatter_tensors`]
//! - **Distributed Data-Parallel (`DDP`)**: [`sync_gradients_ddp`] across model replicas
//! - **Megatron-LM 1D Tensor Parallelism**: [`ColumnParallelLinear`] & [`RowParallelLinear`]
//! - **ZeRO Stage-1 Sharded Optimizer**: [`ZeROAdamW`] (partitions optimizer momentum/variance states across ranks)

pub mod dual_pipe;
pub mod federated;
pub mod parallel_3d;
pub mod sharded_tensor;

pub use dual_pipe::{DualPipeDirection, DualPipeOpKind, DualPipeSchedule, DualPipeStep};
pub use federated::{ByzantineAggregator, DpSgdConfig};
pub use parallel_3d::{Parallel3DConfig, Parallel3DRuntime, RankCoords3D};
pub use sharded_tensor::{AutoParallelPlan, AutoParallelPlanner, Placement, ShardedTensor};

use crate::error::{Error, Result};
use crate::nn::{Linear, Module};
use crate::tensor::{Tensor, no_grad};

/// Collective reduction operator for [`all_reduce_tensors`] and [`reduce_scatter_tensors`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReduceOp {
    Sum,
    Mean,
    Max,
    Min,
}

/// Performs an across-rank collective `AllReduce` over `tensors` (one per rank) using `op`,
/// returning a new `Tensor` with the reduced result.
pub fn all_reduce_tensors(tensors: &[Tensor], op: ReduceOp) -> Result<Tensor> {
    if tensors.is_empty() {
        return Err(Error::Msg(
            "all_reduce_tensors requires at least 1 rank tensor".into(),
        ));
    }
    let shape = tensors[0].shape();
    for t in &tensors[1..] {
        if t.shape() != shape {
            return Err(Error::ShapeMismatch {
                op: "all_reduce_tensors",
                expected: shape.to_vec(),
                got: t.shape().to_vec(),
            });
        }
    }
    match op {
        ReduceOp::Sum => {
            let mut acc = tensors[0].clone();
            for t in &tensors[1..] {
                acc = acc.add(t)?;
            }
            Ok(acc)
        }
        ReduceOp::Mean => {
            let sum = all_reduce_tensors(tensors, ReduceOp::Sum)?;
            sum.div_scalar(tensors.len() as f64)
        }
        ReduceOp::Max => {
            let stacked = Tensor::stack(tensors, 0)?;
            stacked.max_axis(0, false)
        }
        ReduceOp::Min => {
            let stacked = Tensor::stack(tensors, 0)?;
            stacked.min_axis(0, false)
        }
    }
}

/// Performs an across-rank collective `AllGather` by concatenating `tensors` along `dim`.
pub fn all_gather_tensors(tensors: &[Tensor], dim: isize) -> Result<Tensor> {
    if tensors.is_empty() {
        return Err(Error::Msg(
            "all_gather_tensors requires at least 1 rank tensor".into(),
        ));
    }
    Tensor::cat(tensors, dim)
}

/// Performs a collective `ReduceScatter`: reduces `tensors` across ranks using `op`,
/// then splits the reduced tensor into `tensors.len()` equal shards along `dim`.
pub fn reduce_scatter_tensors(tensors: &[Tensor], op: ReduceOp, dim: isize) -> Result<Vec<Tensor>> {
    let reduced = all_reduce_tensors(tensors, op)?;
    reduced.chunk(tensors.len(), dim)
}

/// Synchronizes (averages) gradients across `replicas` in Distributed Data-Parallel (`DDP`)
/// training so every replica's parameter `.grad()` holds the exact all-reduced mean gradient.
pub fn sync_gradients_ddp(replicas: &[Vec<Tensor>]) -> Result<()> {
    if replicas.len() <= 1 {
        return Ok(());
    }
    let num_params = replicas[0].len();
    for r in &replicas[1..] {
        if r.len() != num_params {
            return Err(Error::Msg(
                "sync_gradients_ddp: replica parameter count mismatch".into(),
            ));
        }
    }

    no_grad(|| {
        for p_idx in 0..num_params {
            let mut grads = Vec::with_capacity(replicas.len());
            for r in replicas {
                if let Some(g) = r[p_idx].grad() {
                    grads.push(g);
                }
            }
            if grads.is_empty() {
                // Frozen or unused parameter on every replica: nothing to sync.
                continue;
            }
            if grads.len() != replicas.len() {
                return Err(Error::Msg(format!(
                    "sync_gradients_ddp: param {} has grads on only {} of {} replicas; refusing to leave replicas diverged",
                    p_idx,
                    grads.len(),
                    replicas.len()
                )));
            }
            let mean_g = all_reduce_tensors(&grads, ReduceOp::Mean)?.detach();
            for r in replicas {
                r[p_idx].set_grad(Some(mean_g.clone()));
            }
        }
        Ok(())
    })
}

/// Megatron-LM style 1D **Column-Parallel Linear** layer:
/// Partitions the output feature dimension `out_features` evenly across `world_size` shards:
/// $$W = [W_1, W_2, \dots, W_P], \quad Y_p = X W_p^T + b_p$$
/// Supports both local sharded output (`forward_sharded`) and full gathered output (`forward`).
#[derive(Clone)]
pub struct ColumnParallelLinear {
    pub shards: Vec<Linear>,
    pub in_features: usize,
    pub out_features: usize,
    pub world_size: usize,
}

impl ColumnParallelLinear {
    pub fn new(in_features: usize, out_features: usize, world_size: usize) -> Result<Self> {
        if world_size == 0 || out_features % world_size != 0 {
            return Err(Error::Msg(format!(
                "ColumnParallelLinear: out_features ({out_features}) must be divisible by world_size ({world_size})"
            )));
        }
        // The shard reserve + build loop below fan out by world_size (aborted on giant counts).
        crate::limits::check_repeats("world_size", world_size)?;
        let shard_out = out_features / world_size;
        let mut shards = Vec::with_capacity(world_size);
        for _ in 0..world_size {
            shards.push(Linear::new(in_features, shard_out)?);
        }
        Ok(Self {
            shards,
            in_features,
            out_features,
            world_size,
        })
    }

    /// Partitions an existing dense [`Linear`] layer into `world_size` column-parallel shards
    /// whose concatenated output is mathematically identical to `linear.forward(x)`.
    pub fn from_linear(linear: &Linear, world_size: usize) -> Result<Self> {
        let out_features = linear.weight.shape()[0];
        let in_features = linear.weight.shape()[1];
        if world_size == 0 || out_features % world_size != 0 {
            return Err(Error::Msg(format!(
                "ColumnParallelLinear::from_linear: out_features ({out_features}) not divisible by world_size ({world_size})"
            )));
        }
        let w_chunks = linear.weight.chunk(world_size, 0)?;
        let b_chunks = match &linear.bias {
            Some(b) => Some(b.chunk(world_size, 0)?),
            None => None,
        };
        let mut shards = Vec::with_capacity(world_size);
        for i in 0..world_size {
            let w_i = w_chunks[i].contiguous()?.detach().requires_grad_(true);
            let b_i = match &b_chunks {
                Some(bc) => Some(bc[i].contiguous()?.detach().requires_grad_(true)),
                None => None,
            };
            shards.push(Linear::from_tensors(w_i, b_i)?);
        }
        Ok(Self {
            shards,
            in_features,
            out_features,
            world_size,
        })
    }

    /// Computes per-rank sharded outputs $[Y_1, \dots, Y_P]$ without an `AllGather` (ideal for feeding directly into [`RowParallelLinear`]).
    pub fn forward_sharded(&self, x: &Tensor) -> Result<Vec<Tensor>> {
        if self.shards.len() != self.world_size {
            return Err(Error::Msg(format!(
                "ColumnParallelLinear: {} shards vs world_size {}",
                self.shards.len(),
                self.world_size
            )));
        }
        let mut outs = Vec::with_capacity(self.world_size);
        for s in &self.shards {
            outs.push(s.forward(x)?);
        }
        Ok(outs)
    }

    /// Computes per-rank sharded outputs and performs an `AllGather` along the last axis.
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let outs = self.forward_sharded(x)?;
        all_gather_tensors(&outs, -1)
    }

    pub fn parameters(&self) -> Vec<Tensor> {
        let mut params = Vec::new();
        for s in &self.shards {
            params.push(s.weight.clone());
            if let Some(b) = &s.bias {
                params.push(b.clone());
            }
        }
        params
    }
}

/// Megatron-LM style 1D **Row-Parallel Linear** layer:
/// Partitions the input feature dimension `in_features` evenly across `world_size` shards:
/// $$Y = \sum_{p=1}^{P} X_p W_p^T + b$$
/// where the reduction across ranks is a single `AllReduce(Sum)`.
#[derive(Clone)]
pub struct RowParallelLinear {
    pub shards: Vec<Linear>,
    pub bias: Option<Tensor>,
    pub in_features: usize,
    pub out_features: usize,
    pub world_size: usize,
}

impl RowParallelLinear {
    pub fn from_linear(linear: &Linear, world_size: usize) -> Result<Self> {
        let out_features = linear.weight.shape()[0];
        let in_features = linear.weight.shape()[1];
        if world_size == 0 || in_features % world_size != 0 {
            return Err(Error::Msg(format!(
                "RowParallelLinear::from_linear: in_features ({in_features}) not divisible by world_size ({world_size})"
            )));
        }
        let w_chunks = linear.weight.chunk(world_size, 1)?;
        let mut shards = Vec::with_capacity(world_size);
        for wc in w_chunks {
            let w_i = wc.contiguous()?.detach().requires_grad_(true);
            shards.push(Linear::from_tensors(w_i, None)?);
        }
        let bias = linear
            .bias
            .as_ref()
            .map(|b| b.clone().detach().requires_grad_(true));
        Ok(Self {
            shards,
            bias,
            in_features,
            out_features,
            world_size,
        })
    }

    /// Takes per-rank sharded activations `x_shards` (e.g. from [`ColumnParallelLinear::forward_sharded`]),
    /// computes local matmuls $X_p W_p^T$, performs `AllReduce(Sum)`, and adds `bias`.
    pub fn forward_from_shards(&self, x_shards: &[Tensor]) -> Result<Tensor> {
        if x_shards.len() != self.world_size {
            return Err(Error::Msg(format!(
                "RowParallelLinear expected {} input shards, got {}",
                self.world_size,
                x_shards.len()
            )));
        }
        if self.shards.len() != self.world_size {
            return Err(Error::Msg(format!(
                "RowParallelLinear: {} shards vs world_size {}",
                self.shards.len(),
                self.world_size
            )));
        }
        let mut partial = Vec::with_capacity(self.world_size);
        for (s, x_p) in self.shards.iter().zip(x_shards) {
            partial.push(s.forward(x_p)?);
        }
        let reduced = all_reduce_tensors(&partial, ReduceOp::Sum)?;
        if let Some(b) = &self.bias {
            reduced.add(b)
        } else {
            Ok(reduced)
        }
    }

    /// Splits a full tensor `x` along the last dimension into `world_size` shards and runs [`Self::forward_from_shards`].
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x_shards = x.chunk(self.world_size, -1)?;
        self.forward_from_shards(&x_shards)
    }

    pub fn parameters(&self) -> Vec<Tensor> {
        let mut params = Vec::new();
        for s in &self.shards {
            params.push(s.weight.clone());
        }
        if let Some(b) = &self.bias {
            params.push(b.clone());
        }
        params
    }
}

/// **ZeRO Stage-1** Sharded `AdamW` Optimizer (Rajbhandari et al. 2020):
/// Partitions the first- and second-moment optimizer states $(m, v)$ across `world_size` logical
/// ranks so each rank only stores and updates its assigned slice of parameters, reducing per-rank
/// optimizer state memory by $1 / \text{world\_size}$ while producing the exact `AdamW` trajectory.
pub struct ZeROAdamW {
    pub params: Vec<Tensor>,
    pub lr: f64,
    pub betas: (f64, f64),
    pub eps: f64,
    pub weight_decay: f64,
    pub world_size: usize,
    pub step_count: usize,
    /// Per-rank sharded moment buffers: `rank_states[rank]` holds `(param_idx, m_tensor, v_tensor)`.
    pub rank_states: Vec<Vec<(usize, Tensor, Tensor)>>,
}

impl ZeROAdamW {
    pub fn new(params: Vec<Tensor>, lr: f64, weight_decay: f64, world_size: usize) -> Result<Self> {
        if world_size == 0 {
            return Err(Error::Msg("ZeROAdamW requires world_size > 0".into()));
        }
        // The per-rank state table below is dense in world_size (aborted on giant counts).
        crate::limits::check_repeats("world_size", world_size)?;
        let ws = world_size;
        let mut rank_states: Vec<Vec<(usize, Tensor, Tensor)>> = vec![Vec::new(); ws];
        for (p_idx, p) in params.iter().enumerate() {
            let owner_rank = p_idx % ws;
            let m = p.zeros_like()?.detach();
            let v = p.zeros_like()?.detach();
            rank_states[owner_rank].push((p_idx, m, v));
        }
        Ok(Self {
            params,
            lr,
            betas: (0.9, 0.999),
            eps: 1e-8,
            weight_decay,
            world_size: ws,
            step_count: 0,
            rank_states,
        })
    }

    /// Returns the number of state elements stored on `rank` vs the unsharded total.
    pub fn rank_state_elements(&self, rank: usize) -> usize {
        self.rank_states
            .get(rank)
            .map(|entries| entries.iter().map(|(_, m, v)| m.numel() + v.numel()).sum())
            .unwrap_or(0)
    }

    pub fn zero_grad(&self) {
        for p in &self.params {
            p.zero_grad();
        }
    }

    pub fn step(&mut self) -> Result<()> {
        self.step_count += 1;
        let t = self.step_count as i32;
        let (beta1, beta2) = self.betas;
        let bc1 = 1.0 - beta1.powi(t);
        let bc2 = 1.0 - beta2.powi(t);

        no_grad(|| {
            for rank_entries in &mut self.rank_states {
                for (p_idx, m, v) in rank_entries.iter_mut() {
                    // `rank_states` is pub: a tampered owner index must Err, not panic.
                    let p = self.params.get(*p_idx).ok_or_else(|| {
                        Error::Msg(format!(
                            "ZeROAdamW::step: owner param index {p_idx} out of bounds ({} params)",
                            self.params.len()
                        ))
                    })?;
                    let Some(g) = p.grad() else {
                        continue;
                    };
                    *m = m.mul_scalar(beta1)?.add(&g.mul_scalar(1.0 - beta1)?)?;
                    *v = v
                        .mul_scalar(beta2)?
                        .add(&g.square()?.mul_scalar(1.0 - beta2)?)?;

                    let m_hat = m.div_scalar(bc1)?;
                    let v_hat = v.div_scalar(bc2)?;
                    let denom = v_hat.sqrt()?.add_scalar(self.eps)?;
                    let step_dir = m_hat.div(&denom)?;

                    let decayed = if self.weight_decay > 0.0 {
                        p.mul_scalar(1.0 - self.lr * self.weight_decay)?
                    } else {
                        p.clone()
                    };
                    let updated = decayed.sub(&step_dir.mul_scalar(self.lr)?)?;
                    p.assign(&updated)?;
                }
            }
            Ok(())
        })
    }
}

// ---------------------------------------------------------------------------
// ZeRO-3 Full Parameter & Gradient Sharding (Rajbhandari et al., SC 2020)
// ---------------------------------------------------------------------------

/// Parameter partitioned across all distributed ranks under ZeRO Stage-3.
///
/// In ZeRO-3, model parameters are partitioned $1 / N$ across the $N$ ranks.
/// The full parameter tensor is materialized via collective AllGather exclusively
/// during the active layer's forward/backward compute step and discarded immediately
/// afterward, reducing per-device parameter memory by a factor of $N$.
pub struct ZeRO3Parameter {
    pub original_shape: Vec<usize>,
    pub world_size: usize,
    pub rank_shards: Vec<Tensor>,
    gathered: Option<Tensor>,
}

impl ZeRO3Parameter {
    /// Partitions `param` across `world_size` ranks along its primary axis.
    pub fn from_parameter(param: &Tensor, world_size: usize) -> Result<Self> {
        if world_size == 0 {
            return Err(Error::Msg("ZeRO3Parameter requires world_size > 0".into()));
        }
        // The shard reserve + build loop below fan out by world_size (aborted on giant counts).
        crate::limits::check_repeats("world_size", world_size)?;
        let original_shape = param.shape().to_vec();
        let total_elem = param.numel();
        let shard_size = total_elem.div_ceil(world_size);
        let flat = param.reshape(&[total_elem])?;

        let mut shards = Vec::with_capacity(world_size);
        for r in 0..world_size {
            let start = r * shard_size;
            let len = shard_size.min(total_elem.saturating_sub(start));
            let s = if len > 0 {
                flat.narrow(0, start, len)?
            } else {
                Tensor::zeros(&[0], param.dtype(), &param.device())?
            };
            shards.push(s);
        }

        Ok(Self {
            original_shape,
            world_size,
            rank_shards: shards,
            gathered: None,
        })
    }

    /// Just-in-Time AllGather reconstructing the full parameter for layer compute.
    pub fn gather(&mut self) -> Result<Tensor> {
        if let Some(t) = &self.gathered {
            return Ok(t.clone());
        }
        let full_flat = Tensor::cat(&self.rank_shards, 0)?;
        // `original_shape` is pub: a tampered shape must Err, not overflow the product.
        let total = self
            .original_shape
            .iter()
            .try_fold(1usize, |acc, &d| acc.checked_mul(d))
            .ok_or(Error::Msg(
                "ZeRO3Parameter::gather: original_shape element count overflows".into(),
            ))?;
        let trimmed = full_flat.narrow(0, 0, total)?;
        let full = trimmed.reshape(&self.original_shape)?;
        self.gathered = Some(full.clone());
        Ok(full)
    }

    /// Drops the full reconstructed parameter immediately after layer execution,
    /// freeing memory back down to the $1 / N$ partitioned shard.
    pub fn release(&mut self) {
        self.gathered = None;
    }

    /// Returns the memory reduction factor achieved by ZeRO-3 parameter sharding.
    pub fn memory_savings_ratio(&self) -> f64 {
        1.0 - (1.0 / self.world_size as f64)
    }
}

// ---------------------------------------------------------------------------
// Megatron-LM 1F1B (1-Forward-1-Backward) Pipeline Parallel Scheduler
// ---------------------------------------------------------------------------

/// Execution phase for 1F1B pipeline scheduling.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PipelinePhase {
    Warmup,
    Steady1F1B,
    Cooldown,
}

/// Pipeline Parallel execution schedule coordinating forward and backward passes
/// across micro-batches with minimal pipeline bubble (Narayanan et al., ICML 2021).
pub struct PipelineSchedule1F1B {
    pub num_stages: usize,
    pub num_microbatches: usize,
    pub stage_idx: usize,
    warmup_steps: usize,
    current_step: usize,
    pub forward_mb_executed: usize,
    pub backward_mb_executed: usize,
}

impl PipelineSchedule1F1B {
    pub fn new(num_stages: usize, num_microbatches: usize, stage_idx: usize) -> Result<Self> {
        if num_stages == 0 || stage_idx >= num_stages {
            return Err(Error::Msg(format!(
                "stage_idx {stage_idx} must be below num_stages {num_stages} (and num_stages must be positive)"
            )));
        }
        let warmup = (num_stages - 1 - stage_idx).min(num_microbatches);
        Ok(Self {
            num_stages,
            num_microbatches,
            stage_idx,
            warmup_steps: warmup,
            current_step: 0,
            forward_mb_executed: 0,
            backward_mb_executed: 0,
        })
    }

    /// Determines the action for the current step in the 1F1B schedule.
    /// Returns `(phase, is_forward, microbatch_id)`.
    pub fn next_step(&mut self) -> Option<(PipelinePhase, bool, usize)> {
        if self.backward_mb_executed >= self.num_microbatches {
            return None;
        }

        self.current_step += 1;

        // 1. Warm-up phase: run forward micro-batches to fill the pipeline
        if self.forward_mb_executed < self.warmup_steps {
            let mb_id = self.forward_mb_executed;
            self.forward_mb_executed += 1;
            return Some((PipelinePhase::Warmup, true, mb_id));
        }

        // 2. Steady state: alternate 1 Forward followed by 1 Backward
        if self.forward_mb_executed < self.num_microbatches {
            let is_forward = self.forward_mb_executed == self.backward_mb_executed;
            if is_forward {
                let mb_id = self.forward_mb_executed;
                self.forward_mb_executed += 1;
                Some((PipelinePhase::Steady1F1B, true, mb_id))
            } else {
                let mb_id = self.backward_mb_executed;
                self.backward_mb_executed += 1;
                Some((PipelinePhase::Steady1F1B, false, mb_id))
            }
        } else {
            // 3. Cool-down phase: drain remaining backward passes
            let mb_id = self.backward_mb_executed;
            self.backward_mb_executed += 1;
            Some((PipelinePhase::Cooldown, false, mb_id))
        }
    }

    /// Theoretical pipeline bubble ratio $F_{\text{bubble}} = \frac{P - 1}{M}$.
    pub fn bubble_fraction(&self) -> f64 {
        (self.num_stages.saturating_sub(1)) as f64 / self.num_microbatches.max(1) as f64
    }
}
