//! GPU (Graphics Processing Unit) Architecture, Stream Concurrency & Multi-GPU Topology.
//!
//! Provides unified GPU hardware control across:
//! - NVIDIA CUDA (Streaming Multiprocessors, Tensor Cores, NVLink)
//! - Apple Silicon Metal / MPS (Unified Memory architecture)
//! - Vulkan Compute (SPIR-V shader pipelines)
//! - Cross-platform WebGPU (WGSL compute kernels)
//!
//! Only a real wgpu adapter (`wgpu` feature) is backed by hardware. Otherwise the registry holds a fictional
//! descriptor used for bookkeeping, and "GPU" tensors live in host memory (see `Device::is_simulated`).

use std::collections::HashMap;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, RwLock};

use crate::device::device_type::Device;

/// GPU Architecture and Driver Backend Kind.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum GpuBackendKind {
    Cuda,
    Metal,
    Vulkan,
    Wgpu,
}

/// GPU Compute Hardware Descriptor.
#[derive(Debug, Clone)]
pub struct GpuDescriptor {
    pub index: usize,
    pub name: String,
    pub backend: GpuBackendKind,
    pub total_vram_bytes: usize,
    pub allocated_vram_bytes: Arc<AtomicUsize>,
    pub compute_units: usize,
    pub peak_fp32_tflops: f64,
    pub peak_fp16_tflops: f64,
    pub has_tensor_cores: bool,
    pub is_unified_memory: bool,
    /// True when no hardware backs this entry (placement label only). Capacity figures of every entry are placeholders:
    /// wgpu does not expose VRAM or compute-unit counts.
    pub is_simulated: bool,
}

impl GpuDescriptor {
    pub fn new(
        index: usize,
        name: impl Into<String>,
        backend: GpuBackendKind,
        vram_bytes: usize,
        compute_units: usize,
        tflops_fp32: f64,
        has_tensor_cores: bool,
    ) -> Self {
        let is_unified = backend == GpuBackendKind::Metal;
        Self {
            index,
            name: name.into(),
            backend,
            total_vram_bytes: vram_bytes,
            allocated_vram_bytes: Arc::new(AtomicUsize::new(0)),
            compute_units,
            peak_fp32_tflops: tflops_fp32,
            peak_fp16_tflops: tflops_fp32 * 2.0,
            has_tensor_cores,
            is_unified_memory: is_unified,
            is_simulated: false,
        }
    }

    /// Marks the descriptor as a placement label with no hardware behind it.
    pub fn simulated(mut self) -> Self {
        self.is_simulated = true;
        self
    }

    pub fn device(&self) -> Device {
        match self.backend {
            GpuBackendKind::Cuda => Device::Cuda(self.index),
            GpuBackendKind::Metal => Device::Metal(self.index),
            GpuBackendKind::Vulkan => Device::Vulkan(self.index),
            GpuBackendKind::Wgpu => {
                #[cfg(feature = "wgpu")]
                {
                    Device::wgpu().unwrap_or(Device::Cpu)
                }
                #[cfg(not(feature = "wgpu"))]
                {
                    // No wgpu runtime: memory lives on the host, so report Cpu
                    // rather than a CUDA device that does not exist.
                    Device::Cpu
                }
            }
        }
    }

    pub fn allocated_bytes(&self) -> usize {
        self.allocated_vram_bytes.load(Ordering::Relaxed)
    }

    pub fn free_bytes(&self) -> usize {
        self.total_vram_bytes.saturating_sub(self.allocated_bytes())
    }
}

/// Global GPU Hardware Registry.
#[derive(Debug, Default)]
pub struct GpuRegistry {
    gpus: RwLock<HashMap<usize, GpuDescriptor>>,
}

impl GpuRegistry {
    pub fn new() -> Self {
        let mut map = HashMap::new();

        // Register a real wgpu adapter if one exists.
        #[cfg(feature = "wgpu")]
        {
            if let Ok(Device::Wgpu(dev)) = Device::wgpu() {
                let name = if dev.is_software() {
                    format!("{} (software renderer)", dev.name())
                } else {
                    dev.name().to_string()
                };
                map.insert(
                    0,
                    GpuDescriptor::new(
                        0,
                        name,
                        GpuBackendKind::Wgpu,
                        16 * 1024 * 1024 * 1024, // 16GB
                        64,
                        32.0,
                        true,
                    ),
                );
            }
        }

        // No real adapter: register a fictional device (capacity is made up) so placement code has something to track.
        if map.is_empty() {
            map.insert(
                0,
                GpuDescriptor::new(
                    0,
                    "Simulated Accelerator 0 (host memory)",
                    GpuBackendKind::Cuda,
                    24 * 1024 * 1024 * 1024, // 24GB
                    128,
                    65.0,
                    true,
                )
                .simulated(),
            );
        }

        Self {
            gpus: RwLock::new(map),
        }
    }

    pub fn count(&self) -> usize {
        self.gpus
            .read()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .len()
    }

    pub fn get_gpu(&self, index: usize) -> Option<GpuDescriptor> {
        self.gpus
            .read()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .get(&index)
            .cloned()
    }

    /// Returns every registered GPU ordered by index (custom GPUs may use
    /// non-contiguous indices, so callers must not assume `0..count()`).
    pub fn all_gpus(&self) -> Vec<GpuDescriptor> {
        let guard = self
            .gpus
            .read()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let mut gpus: Vec<GpuDescriptor> = guard.values().cloned().collect();
        gpus.sort_by_key(|g| g.index);
        gpus
    }

    pub fn register_custom_gpu(&self, gpu: GpuDescriptor) {
        let idx = gpu.index;
        self.gpus
            .write()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .insert(idx, gpu);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn poisoned_lock_does_not_brick_registry() {
        let r = GpuRegistry::new();
        let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _g = r.gpus.write().unwrap();
            panic!("intentional poison");
        }));
        // pre-fix: every accessor below panicked on the poisoned lock
        let _ = r.count();
        let _ = r.get_gpu(0);
        let _ = r.all_gpus();
        r.register_custom_gpu(GpuDescriptor::new(
            7,
            "poison-test",
            GpuBackendKind::Wgpu,
            8,
            1,
            1.0,
            false,
        ));
        assert!(r.get_gpu(7).is_some());
    }
}
