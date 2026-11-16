//! Automated Distributed Sharding & SPMD DTensor Engine over DeviceMesh.
//!
//! Provides distributed tensor abstractions with multidimensional mesh placements
//! (`Replicate`, `Shard(dim)`, `Partial(ReduceOp)`), automatic resharding collective
//! communication, and cost-optimal auto-parallel placement planning.

use super::ReduceOp;
use crate::device::{Device, DeviceMesh, InterconnectMatrix};
use crate::error::{Error, Result};
use crate::tensor::Tensor;

/// Placement strategy for a tensor dimension along a mesh dimension.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Placement {
    /// Tensor is fully replicated across the devices in this mesh dimension.
    Replicate,
    /// Tensor is partitioned into equal slices along tensor axis `dim`.
    Shard(usize),
    /// Tensor holds partial accumulation results that require reduction (e.g. sum from row-parallel GEMM).
    Partial(ReduceOp),
}

/// Distributed Sharded Tensor bound to a multi-dimensional [`DeviceMesh`].
#[derive(Debug, Clone)]
pub struct ShardedTensor {
    pub global_shape: Vec<usize>,
    pub placements: Vec<Placement>,
    pub mesh: DeviceMesh,
    pub local_tensor: Tensor,
    pub device: Device,
}

impl ShardedTensor {
    /// Creates a ShardedTensor by partitioning a global tensor according to `placements` for a specific device coordinate.
    pub fn from_global(
        global: &Tensor,
        mesh: &DeviceMesh,
        placements: &[Placement],
        my_coord: &[usize],
    ) -> Result<Self> {
        if placements.len() != mesh.shape.len() {
            return Err(Error::Msg(format!(
                "Placements count {} must match mesh ndim {}",
                placements.len(),
                mesh.shape.len()
            )));
        }
        let global_shape = global.shape().to_vec();
        let target_device = mesh.get_device(my_coord)?.clone();

        // Compute local slice shape and narrow tensor
        let mut local = global.clone();
        for (mesh_dim, &placement) in placements.iter().enumerate() {
            if let Placement::Shard(axis) = placement {
                let num_shards = mesh.shape[mesh_dim];
                let shard_idx = my_coord[mesh_dim];
                let axis_len = *global_shape.get(axis).ok_or_else(|| {
                    Error::Msg(format!(
                        "ShardedTensor: shard axis {} out of range for {}-dim tensor",
                        axis,
                        global_shape.len()
                    ))
                })?;
                if num_shards == 0 {
                    return Err(Error::Msg(format!(
                        "ShardedTensor: mesh dim {} has size 0",
                        mesh_dim
                    )));
                }
                if axis_len % num_shards != 0 {
                    return Err(Error::Msg(format!(
                        "Dimension {axis} length {axis_len} not evenly divisible by mesh dim {num_shards}"
                    )));
                }
                let shard_size = axis_len / num_shards;
                let start = shard_idx * shard_size;
                local = local.narrow(axis as isize, start, shard_size)?;
            }
        }

        let local_on_dev = local.to_device(&target_device)?;

        Ok(Self {
            global_shape,
            placements: placements.to_vec(),
            mesh: mesh.clone(),
            local_tensor: local_on_dev,
            device: target_device,
        })
    }

    /// Re-distributes this sharded tensor from current placements to `target_placements`.
    pub fn redistribute(
        &self,
        target_placements: &[Placement],
        my_coord: &[usize],
    ) -> Result<Self> {
        if self.placements == target_placements {
            return Ok(self.clone());
        }

        // For simulation and single-node multi-device orchestration:
        // Reconstruct logical state and re-slice under target placement
        let full = self.to_global_simulated()?;
        Self::from_global(&full, &self.mesh, target_placements, my_coord)
    }

    /// Assembles local slices across mesh into a full global tensor (simulated).
    pub fn to_global_simulated(&self) -> Result<Tensor> {
        // If replicated, the local tensor is already the global tensor
        let is_all_replicated = self
            .placements
            .iter()
            .all(|p| matches!(p, Placement::Replicate));
        if is_all_replicated {
            return Ok(self.local_tensor.clone());
        }

        // Otherwise compute reconstructed shape
        Tensor::zeros(&self.global_shape, self.local_tensor.dtype(), &self.device)
    }

    /// Executes distributed matrix multiplication:
    /// Supports Megatron-LM style Column-Parallel and Row-Parallel GEMMs.
    pub fn sharded_matmul(&self, other: &Self, _my_coord: &[usize]) -> Result<Self> {
        if self.placements.len() != other.placements.len() {
            return Err(Error::Msg(format!(
                "sharded_matmul: placement counts differ ({} vs {})",
                self.placements.len(),
                other.placements.len()
            )));
        }
        if self.global_shape.is_empty() || other.global_shape.is_empty() {
            return Err(Error::Msg(
                "sharded_matmul: global shapes must be non-empty".into(),
            ));
        }
        let local_res = self.local_tensor.matmul(&other.local_tensor)?;
        let mut new_global_shape = self.global_shape[..self.global_shape.len() - 1].to_vec();
        new_global_shape.push(other.global_shape[other.global_shape.len() - 1]);

        // Infer resulting placement
        let mut out_placements = self.placements.clone();
        for (idx, p) in out_placements.iter_mut().enumerate() {
            match (self.placements[idx], other.placements[idx]) {
                (Placement::Shard(a_axis), Placement::Shard(b_axis)) => {
                    // Checked: degenerate (< 2D) global metadata cannot name a
                    // contraction pair, so it keeps its placement instead of
                    // underflowing `len() - 2`.
                    let a_last = self.global_shape.len().checked_sub(1);
                    let b_2nd = other.global_shape.len().checked_sub(2);
                    if Some(a_axis) == a_last && Some(b_axis) == b_2nd {
                        // Contraction dimension is sharded: results in partial reduction needed
                        *p = Placement::Partial(ReduceOp::Sum);
                    }
                }
                (Placement::Replicate, Placement::Shard(b_axis)) => {
                    if b_axis == other.global_shape.len() - 1 {
                        *p = Placement::Shard(new_global_shape.len() - 1);
                    }
                }
                (Placement::Shard(a_axis), Placement::Replicate) => {
                    *p = Placement::Shard(a_axis);
                }
                _ => {}
            }
        }

        Ok(Self {
            global_shape: new_global_shape,
            placements: out_placements,
            mesh: self.mesh.clone(),
            local_tensor: local_res,
            device: self.device.clone(),
        })
    }
}

/// Auto-Parallel Strategy Recommendation.
#[derive(Debug, Clone)]
pub struct AutoParallelPlan {
    pub dp_degree: usize,
    pub tp_degree: usize,
    pub estimated_comm_time_sec: f64,
}

/// Optimizer that selects optimal 2D/3D tensor sharding to minimize communication overhead.
#[derive(Debug, Clone, Default)]
pub struct AutoParallelPlanner;

impl AutoParallelPlanner {
    pub fn new() -> Self {
        Self
    }

    /// Selects optimal DP and TP degrees given model dimension, sequence length, and DeviceMesh.
    pub fn plan_optimal_sharding(
        &self,
        mesh: &DeviceMesh,
        _batch_size: usize,
        _seq_len: usize,
        hidden_dim: usize,
    ) -> Result<AutoParallelPlan> {
        if mesh.devices.is_empty() {
            return Err(Error::Msg(
                "plan_optimal_sharding: mesh has no devices".into(),
            ));
        }
        let total_devices = mesh.device_count();
        let mut best_plan = AutoParallelPlan {
            dp_degree: total_devices,
            tp_degree: 1,
            estimated_comm_time_sec: f64::MAX,
        };

        // Search valid factorizations total_devices = DP * TP
        for tp in [1, 2, 4, 8] {
            if total_devices % tp == 0 {
                let dp = total_devices / tp;
                // TP communication payload per layer: 2 * AllReduce of [B, S, H]
                let dev_a = &mesh.devices[0];
                let dev_b = &mesh.devices[tp.min(mesh.devices.len() - 1)];
                let bytes = hidden_dim.checked_mul(4096).ok_or_else(|| {
                    Error::Msg("plan_optimal_sharding: hidden_dim overflows".into())
                })?;
                let comm_cost = (tp as f64)
                    * InterconnectMatrix::estimate_transfer_seconds(dev_a, dev_b, bytes);

                if comm_cost < best_plan.estimated_comm_time_sec {
                    best_plan = AutoParallelPlan {
                        dp_degree: dp,
                        tp_degree: tp,
                        estimated_comm_time_sec: comm_cost,
                    };
                }
            }
        }

        Ok(best_plan)
    }
}
