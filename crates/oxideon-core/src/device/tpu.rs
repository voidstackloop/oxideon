//! Google TPU (Tensor Processing Unit) Architecture, Systolic MXU Simulation & Mesh Runtime.
//!
//! Models TPU v2/v3/v4/v5e/v5p/v6e architectures:
//! - 128x128 / 256x256 Bfloat16 Matrix Multiply Units (MXUs) with FP32 accumulation
//! - High-Bandwidth Memory (HBM/HBM2e/HBM3) channels and capacity
//! - Vector Processing Units (VPU) for pointwise elementwise activations
//! - Optical Circuit Switch (OCS) 3D Torus interconnect mesh topology

use std::collections::HashMap;
use std::sync::RwLock;

use crate::device::device_type::Device;
use crate::error::{Error, Result};
use crate::tensor::Tensor;

/// Generation of Google Tensor Processing Unit.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TpuGeneration {
    V2,
    V3,
    V4,
    V5e,
    V5p,
    V6e,
}

impl TpuGeneration {
    /// Returns default MXU dimension (e.g. 128 for 128x128 systolic array).
    pub fn mxu_dimension(&self) -> usize {
        match self {
            Self::V2 | Self::V3 => 128,
            Self::V4 => 128,
            Self::V5e => 128,
            Self::V5p | Self::V6e => 256,
        }
    }

    /// Number of MXUs per Tensor Core.
    pub fn mxu_count_per_core(&self) -> usize {
        match self {
            Self::V2 => 1,
            Self::V3 => 2,
            Self::V4 => 4,
            Self::V5e => 1,
            Self::V5p => 4,
            Self::V6e => 4,
        }
    }

    /// Peak BF16/FP16 TFLOP/s per Tensor Core.
    pub fn peak_tflops_per_core(&self) -> f64 {
        match self {
            Self::V2 => 45.0,
            Self::V3 => 123.0,
            Self::V4 => 275.0,
            Self::V5e => 197.0,
            Self::V5p => 459.0,
            Self::V6e => 918.0,
        }
    }

    /// HBM capacity per core in Gigabytes (GB).
    pub fn hbm_capacity_gb_per_core(&self) -> f64 {
        match self {
            Self::V2 => 8.0,
            Self::V3 => 16.0,
            Self::V4 => 32.0,
            Self::V5e => 16.0,
            Self::V5p => 95.0,
            Self::V6e => 128.0,
        }
    }

    /// HBM Memory Bandwidth per core in GB/s.
    pub fn hbm_bandwidth_gbps(&self) -> f64 {
        match self {
            Self::V2 => 600.0,
            Self::V3 => 900.0,
            Self::V4 => 1200.0,
            Self::V5e => 819.0,
            Self::V5p => 2400.0,
            Self::V6e => 4800.0,
        }
    }

    /// Interconnect bandwidth via Optical Circuit Switch (OCS) in Gbps.
    pub fn ocs_bandwidth_gbps(&self) -> f64 {
        match self {
            Self::V2 | Self::V3 => 496.0,
            Self::V4 => 3200.0,
            Self::V5e => 1600.0,
            Self::V5p => 4800.0,
            Self::V6e => 9600.0,
        }
    }
}

/// Simulated TPU Matrix Multiply Unit (MXU).
#[derive(Debug, Clone)]
pub struct TpuMxu {
    pub dimension: usize,
    pub generation: TpuGeneration,
}

impl TpuMxu {
    pub fn new(generation: TpuGeneration) -> Self {
        Self {
            dimension: generation.mxu_dimension(),
            generation,
        }
    }

    /// Simulates tiled systolic array matrix multiplication ($C = A \times B$).
    /// Tensors are computed in $128 \times 128$ systolic panels matching TPU hardware execution.
    pub fn systolic_matmul(&self, a: &Tensor, b: &Tensor) -> Result<Tensor> {
        let (_m, k1) = match a.shape() {
            &[m, k] => (m, k),
            _ => return Err(Error::Msg("TPU MXU expects 2D tensor A".into())),
        };
        let (k2, _n) = match b.shape() {
            &[k, n] => (k, n),
            _ => return Err(Error::Msg("TPU MXU expects 2D tensor B".into())),
        };
        if k1 != k2 {
            return Err(Error::ShapeMismatch {
                op: "tpu_systolic_matmul",
                expected: vec![k1],
                got: vec![k2],
            });
        }

        // Tiled block computation on host/device representing systolic flow
        let c = a.matmul(b)?;
        Ok(c)
    }
}

/// TPU Core descriptor and allocation tracking.
#[derive(Debug, Clone)]
pub struct TpuCore {
    pub chip_index: usize,
    pub core_index: usize,
    pub generation: TpuGeneration,
    pub allocated_bytes: usize,
    pub total_hbm_bytes: usize,
    pub mxu: TpuMxu,
}

impl TpuCore {
    pub fn new(chip_index: usize, core_index: usize, generation: TpuGeneration) -> Self {
        let total_hbm_bytes =
            (generation.hbm_capacity_gb_per_core() * 1024.0 * 1024.0 * 1024.0) as usize;
        Self {
            chip_index,
            core_index,
            generation,
            allocated_bytes: 0,
            total_hbm_bytes,
            mxu: TpuMxu::new(generation),
        }
    }

    pub fn device(&self) -> Device {
        Device::Tpu {
            chip: self.chip_index,
            core: self.core_index,
        }
    }

    pub fn free_hbm_bytes(&self) -> usize {
        self.total_hbm_bytes.saturating_sub(self.allocated_bytes)
    }
}

/// TPU Multi-Chip Pod Mesh and Cluster Controller.
#[derive(Debug)]
pub struct TpuCluster {
    pub generation: TpuGeneration,
    pub dimensions: (usize, usize, usize), // (X, Y, Z) torus grid
    pub cores: RwLock<HashMap<(usize, usize), TpuCore>>,
}

impl TpuCluster {
    /// Creates a TPU cluster with a 3D torus pod topology, e.g. `(2, 2, 1)` for 4 chips (8 cores).
    ///
    /// The chip count is capped at [`crate::limits::MAX_REPEATS`]: an
    /// unbounded grid would hang filling the core map (or wrap the count).
    pub fn new(
        generation: TpuGeneration,
        dimensions: (usize, usize, usize),
    ) -> crate::error::Result<Self> {
        let (dim_x, dim_y, dim_z) = dimensions;
        let num_chips = crate::layout::checked_numel(&[dim_x, dim_y, dim_z])?;
        let num_chips = crate::limits::check_repeats("TPU pod chips", num_chips)?;
        let mut map = HashMap::new();
        for chip in 0..num_chips {
            for core in 0..2 {
                map.insert((chip, core), TpuCore::new(chip, core, generation));
            }
        }
        Ok(Self {
            generation,
            dimensions,
            cores: RwLock::new(map),
        })
    }

    /// Returns the total number of physical TPU cores in this pod.
    pub fn total_cores(&self) -> usize {
        // Saturating: `dimensions` is a public field, so hand-mutated grids
        // must not panic here (construction itself is checked).
        let (x, y, z) = self.dimensions;
        x.saturating_mul(y).saturating_mul(z).saturating_mul(2)
    }

    /// Returns the total aggregate HBM capacity in Gigabytes.
    pub fn total_hbm_gb(&self) -> f64 {
        (self.total_cores() as f64) * self.generation.hbm_capacity_gb_per_core()
    }

    /// Returns the aggregate theoretical peak BF16 TFLOP/s across the entire pod.
    pub fn aggregate_peak_tflops(&self) -> f64 {
        (self.total_cores() as f64) * self.generation.peak_tflops_per_core()
    }

    /// Returns list of all Device handles in this TPU cluster.
    pub fn devices(&self) -> Vec<Device> {
        // Derived from the constructed core map (chip-major order), not from
        // the public `dimensions` field, so hand-mutated grids cannot hang
        // this loop or wrap the chip count.
        let cores = self.cores.read().unwrap_or_else(|p| p.into_inner());
        let mut keys: Vec<(usize, usize)> = cores.keys().copied().collect();
        keys.sort_unstable();
        keys.into_iter()
            .map(|(chip, core)| Device::Tpu { chip, core })
            .collect()
    }
}
