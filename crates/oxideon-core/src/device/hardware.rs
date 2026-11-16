//! Unified Hardware Manager, Inventory & Heterogeneous Resource Control.
//!
//! Provides comprehensive introspection and runtime authority over compute hardware:
//! - Full hardware inventory across CPU, GPU (CUDA, Metal, Vulkan, WGPU), and TPU
//! - Dynamic default device assignment and scoping
//! - Real-time memory allocation, bandwidth, and latency monitoring
//! - Hardware diagnostics and micro-benchmarking

use std::sync::atomic::AtomicUsize;
use std::sync::{Arc, OnceLock, RwLock};

use crate::device::context::{get_default_device, set_default_device};
use crate::device::device_type::Device;
use crate::device::gpu::GpuRegistry;
use crate::device::tpu::{TpuCluster, TpuGeneration};
use crate::error::{Error, Result};
use crate::tensor::Tensor;

static GLOBAL_HARDWARE_MANAGER: OnceLock<HardwareManager> = OnceLock::new();

/// Global Hardware Resource Manager.
pub struct HardwareManager {
    pub cpu_cores: usize,
    pub gpu_registry: GpuRegistry,
    pub tpu_cluster: TpuCluster,
    allocated_bytes_by_device: RwLock<std::collections::HashMap<String, Arc<AtomicUsize>>>,
}

impl HardwareManager {
    /// Returns the global singleton HardwareManager.
    pub fn global() -> &'static HardwareManager {
        GLOBAL_HARDWARE_MANAGER.get_or_init(HardwareManager::new)
    }

    /// Initializes a new HardwareManager.
    pub fn new() -> Self {
        let cpu_cores = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(8);

        // Default 2x2x1 TPU v5e pod (4 chips, 8 cores). The literal grid
        // cannot fail validation (4 chips << MAX_REPEATS, no overflow).
        let tpu_cluster = TpuCluster::new(TpuGeneration::V5e, (2, 2, 1))
            .expect("default 2x2x1 TPU pod is within limits");
        let gpu_registry = GpuRegistry::new();

        Self {
            cpu_cores,
            gpu_registry,
            tpu_cluster,
            allocated_bytes_by_device: RwLock::new(std::collections::HashMap::new()),
        }
    }

    /// Returns list of all detected devices (CPU, GPUs, and TPUs).
    pub fn all_devices(&self) -> Vec<Device> {
        let mut list = vec![Device::Cpu, Device::Meta];
        // GPUs (enumerated by registration, not 0..count, so custom
        // non-contiguous GPU indices are not dropped).
        for gpu in self.gpu_registry.all_gpus() {
            list.push(gpu.device());
        }
        // TPUs
        list.extend(self.tpu_cluster.devices());
        list
    }

    /// Returns count of available compute devices by category.
    pub fn device_counts(&self) -> (usize, usize, usize) {
        // (CPU count, GPU count, TPU core count)
        (
            self.cpu_cores,
            self.gpu_registry.count(),
            self.tpu_cluster.total_cores(),
        )
    }

    /// Gets current default compute device.
    pub fn current_device(&self) -> Device {
        get_default_device()
    }

    /// Sets default compute device.
    pub fn set_current_device(&self, device: Device) {
        set_default_device(device);
    }

    /// Returns tracked allocated memory bytes for a device.
    pub fn device_allocated_bytes(&self, device: &Device) -> usize {
        let key = device.to_string();
        let guard = self
            .allocated_bytes_by_device
            .read()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        guard
            .get(&key)
            .map(|a| a.load(std::sync::atomic::Ordering::Relaxed))
            .unwrap_or(0)
    }

    /// Tracks an allocation on a device.
    pub fn track_allocation(&self, device: &Device, bytes: usize) {
        let key = device.to_string();
        let mut guard = self
            .allocated_bytes_by_device
            .write()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let counter = guard
            .entry(key)
            .or_insert_with(|| Arc::new(AtomicUsize::new(0)));
        counter.fetch_add(bytes, std::sync::atomic::Ordering::Relaxed);
    }

    /// Tracks a deallocation on a device.
    pub fn track_deallocation(&self, device: &Device, bytes: usize) {
        use std::sync::atomic::Ordering;
        let key = device.to_string();
        let guard = self
            .allocated_bytes_by_device
            .read()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        if let Some(counter) = guard.get(&key) {
            // Saturating subtract via CAS loop: the old load/compare/subtract
            // check-then-act raced between threads and could wrap the counter.
            let mut current = counter.load(Ordering::Relaxed);
            loop {
                let next = current.saturating_sub(bytes);
                match counter.compare_exchange_weak(
                    current,
                    next,
                    Ordering::Relaxed,
                    Ordering::Relaxed,
                ) {
                    Ok(_) => break,
                    Err(actual) => current = actual,
                }
            }
        }
    }

    /// Returns a human-readable summary of all detected hardware.
    pub fn summary(&self) -> String {
        let (cpu, gpu, tpu) = self.device_counts();
        let gpu_names: Vec<String> = self
            .gpu_registry
            .all_gpus()
            .into_iter()
            .map(|g| {
                if g.is_simulated {
                    format!("{} [simulated]", g.name)
                } else {
                    g.name
                }
            })
            .collect();
        // There is no TPU runtime: the pod only exists as a placement/modeling target, so say so.
        format!(
            "Hardware Inventory: {} CPU cores, {} GPU accelerator(s) [{}], {} simulated TPU core(s) [modeled HBM: {:.1} GB, modeled peak: {:.1} TFLOP/s]",
            cpu,
            gpu,
            gpu_names.join(", "),
            tpu,
            self.tpu_cluster.total_hbm_gb(),
            self.tpu_cluster.aggregate_peak_tflops()
        )
    }

    /// Runs a micro-benchmark on a specified device measuring GEMM performance in GFLOP/s.
    pub fn benchmark_gemm(&self, device: &Device, matrix_dim: usize) -> Result<f64> {
        if matrix_dim == 0 {
            return Err(Error::Msg("benchmark_gemm requires matrix_dim > 0".into()));
        }
        let a = Tensor::randn_on(&[matrix_dim, matrix_dim], device)?;
        let b = Tensor::randn_on(&[matrix_dim, matrix_dim], device)?;
        let start = std::time::Instant::now();
        let _c = a.matmul(&b)?;
        let elapsed_sec = start.elapsed().as_secs_f64();
        if elapsed_sec <= 0.0 || !elapsed_sec.is_finite() {
            return Err(Error::Msg(
                "benchmark_gemm: elapsed time too small to measure".into(),
            ));
        }
        let total_ops = 2.0 * (matrix_dim as f64).powi(3);
        let gflops = (total_ops / elapsed_sec) / 1.0e9;
        Ok(gflops)
    }
}

impl Default for HardwareManager {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn poisoned_lock_does_not_brick_allocation_tracking() {
        let m = HardwareManager::new();
        m.track_allocation(&Device::Cpu, 100);
        let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _g = m.allocated_bytes_by_device.write().unwrap();
            panic!("intentional poison");
        }));
        // pre-fix: every accessor below panicked on the poisoned lock
        assert_eq!(m.device_allocated_bytes(&Device::Cpu), 100);
        m.track_allocation(&Device::Cpu, 50);
        m.track_deallocation(&Device::Cpu, 20);
        assert_eq!(m.device_allocated_bytes(&Device::Cpu), 130);
    }
}
