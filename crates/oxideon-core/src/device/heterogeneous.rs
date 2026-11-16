//! Heterogeneous Multi-Device Execution, Mixing & Pipelining Engine.
//!
//! Provides first-class control over mixing hardware compute devices:
//! - Pipelining layers across CPU, GPU (CUDA/Metal/WGPU), and TPU
//! - Automatic inter-device tensor migration with differentiable backward propagation
//! - Sharding tensors across heterogeneous hardware clusters
//! - Automated hardware workload profiling and placement recommendation

use std::sync::Arc;

use crate::device::device_type::Device;
use crate::error::{Error, Result};
use crate::tensor::Tensor;

/// Stage in a heterogeneous execution pipeline assigned to a specific compute device.
pub type StageComputeFn = Arc<dyn Fn(&Tensor) -> Result<Tensor> + Send + Sync>;

pub struct PipelineStage {
    pub name: String,
    pub device: Device,
    pub compute_fn: StageComputeFn,
}

impl std::fmt::Debug for PipelineStage {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PipelineStage")
            .field("name", &self.name)
            .field("device", &self.device)
            .finish()
    }
}

/// Heterogeneous Pipelined Computation Model mixing CPU, GPU, and TPU.
#[derive(Default)]
pub struct HeterogeneousPipeline {
    stages: Vec<PipelineStage>,
}

impl std::fmt::Debug for HeterogeneousPipeline {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("HeterogeneousPipeline")
            .field("stages", &self.stages)
            .finish()
    }
}

impl HeterogeneousPipeline {
    /// Creates a new empty heterogeneous execution pipeline.
    pub fn new() -> Self {
        Self { stages: Vec::new() }
    }

    /// Adds a stage assigned to run on `device`.
    pub fn add_stage<F>(&mut self, name: impl Into<String>, device: Device, compute_fn: F)
    where
        F: Fn(&Tensor) -> Result<Tensor> + Send + Sync + 'static,
    {
        self.stages.push(PipelineStage {
            name: name.into(),
            device,
            compute_fn: Arc::new(compute_fn),
        });
    }

    /// Number of configured pipeline stages.
    pub fn len(&self) -> usize {
        self.stages.len()
    }

    pub fn is_empty(&self) -> bool {
        self.stages.is_empty()
    }

    /// Executes the heterogeneous pipeline sequentially across all assigned devices.
    ///
    /// For each stage:
    /// 1. Migrates input tensor to `stage.device` via differentiable `to_device()`
    /// 2. Executes stage computation on the hardware device
    /// 3. Passes output to subsequent stage
    ///
    /// Autograd gradients automatically stream backward across the entire heterogeneous device mesh.
    pub fn forward(&self, input: &Tensor) -> Result<Tensor> {
        if self.stages.is_empty() {
            return Ok(input.clone());
        }

        let mut current = input.clone();
        for stage in &self.stages {
            // Differentiable cross-device transfer
            if current.device() != stage.device {
                current = current.to_device(&stage.device)?;
            }
            // Execute on target device
            current = (stage.compute_fn)(&current)?;
        }
        Ok(current)
    }

    /// Returns list of devices involved in this pipeline.
    pub fn devices(&self) -> Vec<Device> {
        self.stages.iter().map(|s| s.device.clone()).collect()
    }
}

/// A tensor whose dimensions are sharded across heterogeneous compute devices.
#[derive(Debug, Clone)]
pub struct HeterogeneousShardedTensor {
    pub shards: Vec<(Device, Tensor)>,
    pub concat_axis: usize,
}

impl HeterogeneousShardedTensor {
    /// Creates a heterogeneous sharded tensor across given devices.
    pub fn from_shards(shards: Vec<(Device, Tensor)>, concat_axis: usize) -> Result<Self> {
        if shards.is_empty() {
            return Err(Error::Msg(
                "Cannot create HeterogeneousShardedTensor with 0 shards".into(),
            ));
        }
        Ok(Self {
            shards,
            concat_axis,
        })
    }

    /// Gathers all shards from CPU, GPU, and TPU back into a single tensor on `target_device`.
    pub fn gather_to(&self, target_device: &Device) -> Result<Tensor> {
        let mut gathered = Vec::new();
        for (_, shard) in &self.shards {
            let on_target = shard.to_device(target_device)?;
            gathered.push(on_target);
        }
        Tensor::cat(&gathered, self.concat_axis as isize)
    }

    /// Distributes a tensor on `source` into slices placed on the specified target devices.
    pub fn scatter_from(source: &Tensor, target_devices: &[Device], axis: usize) -> Result<Self> {
        let num_devices = target_devices.len();
        if num_devices == 0 {
            return Err(Error::Msg("target_devices cannot be empty".into()));
        }
        if axis >= source.ndim() {
            return Err(Error::InvalidAxis {
                op: "scatter_from",
                axis: axis as isize,
                ndim: source.ndim(),
            });
        }
        let dim_len = source.shape()[axis];
        let chunk_size = dim_len.div_ceil(num_devices);
        let chunks = source.split(chunk_size, axis as isize)?;
        let mut shards = Vec::new();
        for (i, chunk) in chunks.into_iter().enumerate() {
            let dev = &target_devices[i % num_devices];
            let placed = chunk.to_device(dev)?;
            shards.push((dev.clone(), placed));
        }
        Ok(Self {
            shards,
            concat_axis: axis,
        })
    }
}

/// Automated workload placement optimizer across CPU, GPU, and TPU.
#[derive(Debug, Clone, Default)]
pub struct AutoDeviceMixer;

impl AutoDeviceMixer {
    /// Analyzes a layer specification and recommends an optimal hardware placement:
    /// - Massive embeddings or sparse retrieval -> CPU RAM
    /// - Attention matrices and self-attention -> GPU (VRAM + Tensor Cores)
    /// - Dense GEMM / FFN projections -> TPU (Systolic Matrix Units)
    pub fn recommend_placement(
        layer_name: &str,
        param_count: usize,
        is_gemm_heavy: bool,
    ) -> Device {
        if layer_name.contains("embed") || param_count > 500_000_000 {
            // Giant memory footprint -> CPU Host RAM
            Device::Cpu
        } else if is_gemm_heavy || layer_name.contains("mlp") || layer_name.contains("ffn") {
            // Dense systolic multiplication -> TPU
            Device::Tpu { chip: 0, core: 0 }
        } else {
            // Attention, LayerNorm, Conv -> GPU
            #[cfg(feature = "wgpu")]
            {
                Device::wgpu().unwrap_or(Device::Cuda(0))
            }
            #[cfg(not(feature = "wgpu"))]
            {
                Device::Cuda(0)
            }
        }
    }
}
