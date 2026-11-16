//! Advanced Acceleration Device Choice, Topologies, and Introspection.
//!
//! Provides PyTorch-style device selection and introspection, extended to support complex
//! accelerator topologies (CUDA, Apple Silicon Metal/MPS, Vulkan, TPU v4/v5e mesh coordinates,
//! Groq/Cerebras/Ascend custom NPUs, and zero-memory Meta devices).

pub mod context;
pub mod device_type;
pub mod gpu;
pub mod hardware;
pub mod heterogeneous;
pub mod mesh;
pub mod tiered_memory;
pub mod topology;
pub mod tpu;

pub use context::{get_default_device, set_default_device, with_device};
pub use device_type::{Device, DeviceProperties, DeviceType, InterconnectType};
pub use gpu::{GpuBackendKind, GpuDescriptor, GpuRegistry};
pub use hardware::HardwareManager;
pub use heterogeneous::{
    AutoDeviceMixer, HeterogeneousPipeline, HeterogeneousShardedTensor, PipelineStage,
};
pub use mesh::DeviceMesh;
pub use tiered_memory::{
    AutoDevicePlacer, LayerPlacement, LayerPlacementSpec, MemoryPriority, TieredMemoryManager,
    TieredTensorEntry,
};
pub use topology::{CompositeDevice, InterconnectMatrix};
pub use tpu::{TpuCluster, TpuCore, TpuGeneration, TpuMxu};
