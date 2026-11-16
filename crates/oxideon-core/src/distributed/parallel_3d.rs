//! Heterogeneous Multi-Chip 3D Parallelism Runtime (DP x TP x PP).
//!
//! Coordinates Distributed Data Parallelism (DP), Megatron-style Tensor Parallelism (TP),
//! and 1F1B / DualPipe Pipeline Parallelism (PP) over a unified 3D rank grid.
//! Implements bandwidth-optimal ring all-reduce and reduce-scatter collective communications.

use crate::device::device_type::Device;
use crate::error::{Error, Result};
use crate::tensor::Tensor;

/// 3D Parallelism Grid Configuration.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Parallel3DConfig {
    pub dp_size: usize,
    pub tp_size: usize,
    pub pp_size: usize,
}

impl Parallel3DConfig {
    pub fn new(dp_size: usize, tp_size: usize, pp_size: usize) -> Result<Self> {
        if dp_size == 0 || tp_size == 0 || pp_size == 0 {
            return Err(Error::Msg("Parallel dimensions must be >= 1".into()));
        }
        dp_size
            .checked_mul(tp_size)
            .and_then(|v| v.checked_mul(pp_size))
            .ok_or(Error::Msg("Parallel world size overflows".into()))?;
        Ok(Self {
            dp_size,
            tp_size,
            pp_size,
        })
    }

    /// Total world size = DP * TP * PP, saturating on overflow.
    /// Prefer checked_world_size when exactness matters.
    pub fn world_size(&self) -> usize {
        self.dp_size
            .saturating_mul(self.tp_size)
            .saturating_mul(self.pp_size)
    }

    /// Exact total world size, or Err when DP x TP x PP overflows usize.
    pub fn checked_world_size(&self) -> Result<usize> {
        self.dp_size
            .checked_mul(self.tp_size)
            .and_then(|v| v.checked_mul(self.pp_size))
            .ok_or(Error::Msg("Parallel3DConfig: world size overflows".into()))
    }
}

/// 3D Parallel Rank Coordinates: (Data Parallel, Tensor Parallel, Pipeline Parallel).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct RankCoords3D {
    pub dp_rank: usize,
    pub tp_rank: usize,
    pub pp_rank: usize,
    pub global_rank: usize,
}

/// Runtime coordinator managing 3D parallelism subgroups and collective communications.
#[derive(Debug, Clone)]
pub struct Parallel3DRuntime {
    pub config: Parallel3DConfig,
    pub devices: Vec<Device>,
}

impl Parallel3DRuntime {
    /// Initializes a 3D runtime with the given configuration and hardware device assignments.
    pub fn new(config: Parallel3DConfig, devices: Vec<Device>) -> Result<Self> {
        let expected = config.checked_world_size()?;
        if devices.len() != expected {
            return Err(Error::Msg(format!(
                "Parallel3DRuntime expects {expected} devices, got {}",
                devices.len()
            )));
        }
        Ok(Self { config, devices })
    }

    /// Converts a global rank to 3D grid coordinates `(dp, tp, pp)`.
    /// Ordering: global = pp * (dp * tp) + dp * tp + tp.
    pub fn get_coords(&self, global_rank: usize) -> Result<RankCoords3D> {
        let world = self.config.checked_world_size()?;
        if global_rank >= world {
            return Err(Error::Msg(format!(
                "Global rank {global_rank} exceeds world size {}",
                world
            )));
        }

        let tp_stride = 1;
        let dp_stride = self.config.tp_size;
        let pp_stride = self
            .config
            .dp_size
            .checked_mul(self.config.tp_size)
            .ok_or(Error::Msg("Parallel3D: pp stride overflows".into()))?;

        let pp_rank = global_rank / pp_stride;
        let rem = global_rank % pp_stride;
        let dp_rank = rem / dp_stride;
        let tp_rank = (rem % dp_stride) / tp_stride;

        Ok(RankCoords3D {
            dp_rank,
            tp_rank,
            pp_rank,
            global_rank,
        })
    }

    /// Converts 3D grid coordinates `(dp, tp, pp)` to global linear rank.
    pub fn coords_to_rank(&self, dp: usize, tp: usize, pp: usize) -> Result<usize> {
        if dp >= self.config.dp_size || tp >= self.config.tp_size || pp >= self.config.pp_size {
            return Err(Error::Msg(format!(
                "Coordinates ({dp}, {tp}, {pp}) out of bounds for config {:?}",
                self.config
            )));
        }
        let dp_tp = self
            .config
            .dp_size
            .checked_mul(self.config.tp_size)
            .ok_or(Error::Msg("Parallel3D: rank math overflows".into()))?;
        let base = dp
            .checked_mul(self.config.tp_size)
            .and_then(|v| v.checked_add(tp))
            .ok_or(Error::Msg("Parallel3D: rank math overflows".into()))?;
        let rank = pp
            .checked_mul(dp_tp)
            .and_then(|v| v.checked_add(base))
            .ok_or(Error::Msg("Parallel3D: rank math overflows".into()))?;
        Ok(rank)
    }

    /// Returns the Data Parallel communication subgroup for a given rank.
    /// (All ranks sharing the same TP and PP coordinates).
    pub fn get_dp_group(&self, global_rank: usize) -> Result<Vec<usize>> {
        // Config dims are pub: re-cap the looped dim (tampered values aborted the
        // reserve below and hung the group loop).
        crate::limits::check_repeats("dp_size", self.config.dp_size)?;
        let coords = self.get_coords(global_rank)?;
        let mut group = Vec::with_capacity(self.config.dp_size);
        for dp in 0..self.config.dp_size {
            group.push(self.coords_to_rank(dp, coords.tp_rank, coords.pp_rank)?);
        }
        Ok(group)
    }

    /// Returns the Tensor Parallel communication subgroup for a given rank.
    /// (All ranks sharing the same DP and PP coordinates).
    pub fn get_tp_group(&self, global_rank: usize) -> Result<Vec<usize>> {
        crate::limits::check_repeats("tp_size", self.config.tp_size)?;
        let coords = self.get_coords(global_rank)?;
        let mut group = Vec::with_capacity(self.config.tp_size);
        for tp in 0..self.config.tp_size {
            group.push(self.coords_to_rank(coords.dp_rank, tp, coords.pp_rank)?);
        }
        Ok(group)
    }

    /// Returns the Pipeline Parallel communication subgroup for a given rank.
    /// (All ranks sharing the same DP and TP coordinates).
    pub fn get_pp_group(&self, global_rank: usize) -> Result<Vec<usize>> {
        crate::limits::check_repeats("pp_size", self.config.pp_size)?;
        let coords = self.get_coords(global_rank)?;
        let mut group = Vec::with_capacity(self.config.pp_size);
        for pp in 0..self.config.pp_size {
            group.push(self.coords_to_rank(coords.dp_rank, coords.tp_rank, pp)?);
        }
        Ok(group)
    }

    /// Executes an exact, ring-based collective AllReduce over a slice of tensors
    /// (one local tensor provided per rank in `group`).
    ///
    /// Algorithmic complexity:
    /// - Data transferred per rank: $2 \frac{N-1}{N} S$ where $N = |\text{group}|$
    /// - Exact mathematical equivalence to full all-reduce with optimal ring bandwidth.
    pub fn ring_all_reduce(&self, group: &[usize], rank_tensors: &[Tensor]) -> Result<Vec<Tensor>> {
        let n = group.len();
        if n == 0 {
            return Err(Error::Msg("Ring AllReduce group cannot be empty".into()));
        }
        if rank_tensors.len() != n {
            return Err(Error::Msg(format!(
                "Ring AllReduce expects {n} rank tensors, got {}",
                rank_tensors.len()
            )));
        }
        if n == 1 {
            return Ok(rank_tensors.to_vec());
        }

        let first_shape = rank_tensors[0].shape();
        for t in &rank_tensors[1..] {
            if t.shape() != first_shape {
                return Err(Error::ShapeMismatch {
                    op: "ring_all_reduce",
                    expected: first_shape.to_vec(),
                    got: t.shape().to_vec(),
                });
            }
        }

        // Each rank's tensor is split into N equal segments along dimension 0
        let mut rank_chunks: Vec<Vec<Tensor>> = Vec::with_capacity(n);
        for t in rank_tensors {
            rank_chunks.push(t.chunk(n, 0)?);
        }
        // Short/empty dim0 yields fewer than N pieces, which the ring phases
        // below would index out of bounds; fall back to an exact (non-ring)
        // summed broadcast, which is mathematically identical.
        if rank_chunks.iter().any(|c| c.len() != n) {
            let mut acc = rank_tensors[0].clone();
            for t in &rank_tensors[1..] {
                acc = acc.add(t)?;
            }
            return Ok(vec![acc; n]);
        }

        // 1. Scatter-Reduce phase: (N - 1) rounds
        for step in 0..(n - 1) {
            let mut next_chunks = rank_chunks.clone();
            for (i, current_chunks) in rank_chunks.iter().enumerate() {
                // In step, rank i sends chunk send_idx to (i + 1) % n
                // and receives chunk recv_idx from (i - 1 + n) % n
                let send_idx = (i + n - step) % n;
                let recv_rank = (i + 1) % n;
                let chunk_to_send = &current_chunks[send_idx];

                // Receiver accumulates the chunk
                next_chunks[recv_rank][send_idx] =
                    next_chunks[recv_rank][send_idx].add(chunk_to_send)?;
            }
            rank_chunks = next_chunks;
        }

        // 2. All-Gather phase: (N - 1) rounds
        for step in 0..(n - 1) {
            let mut next_chunks = rank_chunks.clone();
            for (i, current_chunks) in rank_chunks.iter().enumerate() {
                let send_idx = (i + 1 + n - step) % n;
                let recv_rank = (i + 1) % n;
                let chunk_to_send = &current_chunks[send_idx];
                next_chunks[recv_rank][send_idx] = chunk_to_send.clone();
            }
            rank_chunks = next_chunks;
        }

        // Reconstruct each rank's fully reduced tensor by concatenating chunks along dimension 0
        let mut results = Vec::with_capacity(n);
        for chunks in rank_chunks {
            results.push(Tensor::cat(&chunks, 0)?);
        }

        Ok(results)
    }
}
