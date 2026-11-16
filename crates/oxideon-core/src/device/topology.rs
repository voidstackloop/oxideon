//! Hardware Interconnect Topology, Bandwidth Modeling, and Composite Placements.
//!
//! Provides point-to-point interconnect modeling (NVLink4/5, PCIe Gen4/5, Apple Unified Memory,
//! TPU Optical Circuit Switch, InfiniBand NDR) and heterogeneous multi-device abstractions.

use crate::device::device_type::Device;

/// Point-to-point bandwidth and latency matrix across hardware compute devices.
#[derive(Debug, Clone, Copy, Default)]
pub struct InterconnectMatrix;

impl InterconnectMatrix {
    /// Estimates inter-device point-to-point bidirectional bandwidth in GB/s.
    pub fn bandwidth_gbps(a: &Device, b: &Device) -> f64 {
        if a == b {
            // Intra-device internal memory bandwidth
            return match a {
                Device::Cuda(_) => 3350.0,    // NVIDIA H100 SXM5 HBM3: 3.35 TB/s
                Device::Metal(_) => 800.0,    // Apple M4 Max Unified Memory: 800 GB/s
                Device::Tpu { .. } => 1600.0, // TPU v5e HBM2e: 1.6 TB/s
                Device::Vulkan(_) => 1000.0,  // High-end Discrete GPU VRAM
                Device::Npu { .. } => 1200.0, // Custom NPU SRAM / HBM
                _ => 204.8,                   // Host CPU DDR5 octa-channel: ~205 GB/s
            };
        }

        match (a, b) {
            // NVLink4 / NVLink5 high-speed GPU interconnect
            (Device::Cuda(_), Device::Cuda(_)) => 900.0,

            // Host CPU to Accelerator via PCIe Gen5 x16 (64 GB/s bi-directional)
            (Device::Cpu, Device::Cuda(_)) | (Device::Cuda(_), Device::Cpu) => 64.0,

            // Apple Silicon Unified Memory: CPU and GPU share the exact same physical memory bus
            (Device::Cpu, Device::Metal(_)) | (Device::Metal(_), Device::Cpu) => 800.0,

            // Host CPU to Vulkan / WGPU accelerator via PCIe Gen4 x16
            (Device::Cpu, Device::Vulkan(_)) | (Device::Vulkan(_), Device::Cpu) => 32.0,
            #[cfg(feature = "wgpu")]
            (Device::Cpu, Device::Wgpu(_)) | (Device::Wgpu(_), Device::Cpu) => 32.0,
            #[cfg(not(feature = "wgpu"))]
            (Device::Cpu, Device::Wgpu(_)) | (Device::Wgpu(_), Device::Cpu) => 32.0,

            // TPU Optical Circuit Switch (OCS) inter-chip interconnect
            (Device::Tpu { .. }, Device::Tpu { .. }) => 400.0,
            (Device::Cpu, Device::Tpu { .. }) | (Device::Tpu { .. }, Device::Cpu) => 64.0,

            // Custom NPU CXL / PCIe Gen5 fabric
            (Device::Npu { .. }, Device::Npu { .. }) => 512.0,
            (Device::Cpu, Device::Npu { .. }) | (Device::Npu { .. }, Device::Cpu) => 64.0,

            // Virtual Meta device (zero-cost logical transfers)
            (Device::Meta, _) | (_, Device::Meta) => f64::INFINITY,

            // Standard fallback inter-device interconnect
            _ => 32.0,
        }
    }

    /// Estimates one-way transfer latency between devices in microseconds ($\mu\text{s}$).
    pub fn latency_us(a: &Device, b: &Device) -> f64 {
        if a == b {
            return 0.05;
        }
        match (a, b) {
            (Device::Cuda(_), Device::Cuda(_)) => 1.2, // NVLink4 direct peer-to-peer
            (Device::Cpu, Device::Metal(_)) | (Device::Metal(_), Device::Cpu) => 0.1, // Zero-copy Unified Memory
            (Device::Tpu { .. }, Device::Tpu { .. }) => 1.5,                          // OCS fabric
            (Device::Meta, _) | (_, Device::Meta) => 0.0,
            _ => 3.5, // Standard PCIe DMA round-trip
        }
    }

    /// Estimates communication transfer time for a payload of `bytes` size in seconds.
    pub fn estimate_transfer_seconds(a: &Device, b: &Device, bytes: usize) -> f64 {
        if a == b || a.is_meta() || b.is_meta() {
            return 0.0;
        }
        let bw_bytes_per_sec = Self::bandwidth_gbps(a, b) * 1e9;
        let latency_sec = Self::latency_us(a, b) * 1e-6;
        latency_sec + (bytes as f64) / bw_bytes_per_sec
    }
}

/// Composite heterogeneous placement abstractions.
#[derive(Debug, Clone, PartialEq)]
pub enum CompositeDevice {
    /// Host CPU driving a cluster of heterogeneous accelerators
    Hierarchical {
        host: Device,
        accelerators: Vec<Device>,
    },
    /// Pipeline-parallel partition where each stage resides on an allocated accelerator
    PipelineStages(Vec<(usize, Device)>),
}
