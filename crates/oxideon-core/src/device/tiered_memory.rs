//! Tiered Memory Manager & Automatic Device Placement Engine.
//!
//! Provides dynamic memory pressure offloading between Host RAM (CPU) and Accelerator VRAM
//! using an LRU eviction policy with priority weighting, alongside an automated
//! layer-wise model partitioner modeled over [`InterconnectMatrix`].

use crate::device::device_type::Device;
use crate::device::topology::InterconnectMatrix;
use crate::error::{Error, Result};
use std::collections::HashMap;

/// Priority of a tensor in tiered memory. High priority tensors are preserved in fast VRAM.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum MemoryPriority {
    Low = 0,
    Normal = 1,
    High = 2,
    Pinned = 3, // Cannot be evicted
}

/// Metadata tracking a tensor placed in tiered memory.
#[derive(Debug, Clone)]
pub struct TieredTensorEntry {
    pub tensor_id: usize,
    pub size_bytes: usize,
    pub current_device: Device,
    pub preferred_device: Device,
    pub priority: MemoryPriority,
    pub last_access_tick: u64,
}

/// Dynamic Tiered Memory Manager handling VRAM pressure, eviction, and migration.
#[derive(Debug, Clone)]
pub struct TieredMemoryManager {
    device_limits: HashMap<Device, usize>,
    device_allocated: HashMap<Device, usize>,
    high_watermark_ratio: f64,
    entries: HashMap<usize, TieredTensorEntry>,
    current_tick: u64,
}

impl TieredMemoryManager {
    /// Creates a new tiered memory manager with given watermark ratio (e.g. 0.85 = 85%).
    pub fn new(high_watermark_ratio: f64) -> Self {
        Self {
            device_limits: HashMap::new(),
            device_allocated: HashMap::new(),
            high_watermark_ratio: high_watermark_ratio.clamp(0.1, 1.0),
            entries: HashMap::new(),
            current_tick: 0,
        }
    }

    /// Registers a device memory capacity limit in bytes.
    pub fn register_device_capacity(&mut self, device: Device, capacity_bytes: usize) {
        self.device_limits.insert(device.clone(), capacity_bytes);
        self.device_allocated.entry(device).or_insert(0);
    }

    /// Gets current memory usage for a device `(allocated_bytes, capacity_bytes)`.
    pub fn get_device_usage(&self, device: &Device) -> (usize, usize) {
        let alloc = self.device_allocated.get(device).copied().unwrap_or(0);
        let cap = self
            .device_limits
            .get(device)
            .copied()
            .unwrap_or(usize::MAX);
        (alloc, cap)
    }

    /// Checks if device has exceeded its high watermark threshold.
    pub fn is_under_pressure(&self, device: &Device) -> bool {
        if let Some(&cap) = self.device_limits.get(device) {
            let alloc = self.device_allocated.get(device).copied().unwrap_or(0);
            (alloc as f64) >= (cap as f64) * self.high_watermark_ratio
        } else {
            false
        }
    }

    /// Registers a tensor into tiered memory management.
    /// If target device is under pressure or doesn't fit, evicts lower-priority LRU tensors to CPU.
    pub fn register_tensor(
        &mut self,
        tensor_id: usize,
        size_bytes: usize,
        target_device: Device,
        priority: MemoryPriority,
    ) -> Result<Vec<(usize, Device, Device)>> {
        self.current_tick += 1;
        let mut migrations = Vec::new();

        // Re-registering an id first releases its old placement; otherwise the
        // device totals would double-count the tensor.
        if let Some(old) = self.entries.remove(&tensor_id) {
            if let Some(alloc) = self.device_allocated.get_mut(&old.current_device) {
                *alloc = alloc.saturating_sub(old.size_bytes);
            }
        }

        // Check if eviction is necessary on target_device
        if target_device != Device::Cpu {
            if let Some(&cap) = self.device_limits.get(&target_device) {
                let mut current_alloc = self
                    .device_allocated
                    .get(&target_device)
                    .copied()
                    .unwrap_or(0);
                let threshold = (cap as f64 * self.high_watermark_ratio) as usize;

                if current_alloc.checked_add(size_bytes).ok_or_else(|| {
                    crate::error::Error::Msg(
                        "tiered_memory: allocation accounting overflows".into(),
                    )
                })? > threshold
                {
                    // Evict eligible candidates sorted by (priority ASC, last_access_tick ASC)
                    let mut candidates: Vec<(usize, MemoryPriority, u64, usize)> = self
                        .entries
                        .iter()
                        .filter(|(_, e)| {
                            e.current_device == target_device
                                && e.priority != MemoryPriority::Pinned
                        })
                        .map(|(&id, e)| (id, e.priority, e.last_access_tick, e.size_bytes))
                        .collect();

                    candidates.sort_by_key(|&(_, p, tick, _)| (p, tick));

                    for (cand_id, _, _, cand_bytes) in candidates {
                        if current_alloc.checked_add(size_bytes).ok_or_else(|| {
                            crate::error::Error::Msg(
                                "tiered_memory: allocation accounting overflows".into(),
                            )
                        })? <= threshold
                        {
                            break;
                        }

                        // Evict cand to Device::Cpu
                        if let Some(entry) = self.entries.get_mut(&cand_id) {
                            entry.current_device = Device::Cpu;
                            current_alloc = current_alloc.saturating_sub(cand_bytes);
                            *self.device_allocated.get_mut(&target_device).unwrap() = current_alloc;
                            let cpu_alloc = self.device_allocated.entry(Device::Cpu).or_insert(0);
                            *cpu_alloc = cpu_alloc.checked_add(cand_bytes).ok_or_else(|| {
                                crate::error::Error::Msg(
                                    "tiered_memory: allocation accounting overflows".into(),
                                )
                            })?;
                            migrations.push((cand_id, target_device.clone(), Device::Cpu));
                        }
                    }
                }
            }
        }

        // Place tensor
        let total = self
            .device_allocated
            .entry(target_device.clone())
            .or_insert(0);
        *total = total.checked_add(size_bytes).ok_or_else(|| {
            crate::error::Error::Msg("tiered_memory: allocation accounting overflows".into())
        })?;
        self.entries.insert(
            tensor_id,
            TieredTensorEntry {
                tensor_id,
                size_bytes,
                current_device: target_device.clone(),
                preferred_device: target_device,
                priority,
                last_access_tick: self.current_tick,
            },
        );

        Ok(migrations)
    }

    /// Marks a tensor as accessed, returning `Ok(Some(device))` with the preferred
    /// device when the tensor currently lives elsewhere (the caller should
    /// migrate/reload it), or `Ok(None)` when it is already home.
    pub fn access_tensor(&mut self, tensor_id: usize) -> Result<Option<Device>> {
        self.current_tick += 1;
        let entry = self.entries.get_mut(&tensor_id).ok_or_else(|| {
            Error::Msg(format!(
                "Tensor {tensor_id} not found in TieredMemoryManager"
            ))
        })?;

        entry.last_access_tick = self.current_tick;
        if entry.current_device != entry.preferred_device {
            Ok(Some(entry.preferred_device.clone()))
        } else {
            Ok(None)
        }
    }

    /// Deregisters a tensor and frees its tracked memory.
    pub fn deregister_tensor(&mut self, tensor_id: usize) {
        if let Some(entry) = self.entries.remove(&tensor_id) {
            if let Some(alloc) = self.device_allocated.get_mut(&entry.current_device) {
                *alloc = alloc.saturating_sub(entry.size_bytes);
            }
        }
    }
}

/// Layer specification for automatic placement.
#[derive(Debug, Clone)]
pub struct LayerPlacementSpec {
    pub layer_name: String,
    pub parameter_bytes: usize,
    pub flops_estimate: usize,
}

/// Assigned placement result for a model layer.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LayerPlacement {
    pub layer_name: String,
    pub assigned_device: Device,
    pub stage_index: usize,
}

/// Automatic Model Partitioner and Layer Placement Engine.
pub struct AutoDevicePlacer;

impl AutoDevicePlacer {
    /// Automatically partitions sequential layers across `devices` respecting capacity
    /// and minimizing pipeline boundary communication.
    pub fn partition_layers(
        layers: &[LayerPlacementSpec],
        devices: &[Device],
        memory_budget_per_device: usize,
    ) -> Result<Vec<LayerPlacement>> {
        if devices.is_empty() {
            return Err(Error::Msg("No devices provided for placement".into()));
        }
        if layers.is_empty() {
            return Ok(Vec::new());
        }

        let mut placements = Vec::new();
        let mut current_dev_idx = 0;
        let mut current_dev_mem = 0usize;

        for layer in layers {
            // Check if adding this layer exceeds the current device's memory budget
            if current_dev_mem + layer.parameter_bytes > memory_budget_per_device
                && current_dev_idx + 1 < devices.len()
            {
                current_dev_idx += 1;
                current_dev_mem = 0;
            }

            let assigned = devices[current_dev_idx].clone();
            current_dev_mem += layer.parameter_bytes;

            placements.push(LayerPlacement {
                layer_name: layer.layer_name.clone(),
                assigned_device: assigned,
                stage_index: current_dev_idx,
            });
        }

        Ok(placements)
    }

    /// Computes total communication latency between pipeline stage boundaries.
    pub fn evaluate_pipeline_transfer_latency(
        placements: &[LayerPlacement],
        activation_size_bytes: usize,
    ) -> f64 {
        let mut total_latency = 0.0;
        for i in 0..placements.len().saturating_sub(1) {
            let dev_a = &placements[i].assigned_device;
            let dev_b = &placements[i + 1].assigned_device;
            if dev_a != dev_b {
                total_latency += InterconnectMatrix::estimate_transfer_seconds(
                    dev_a,
                    dev_b,
                    activation_size_bytes,
                );
            }
        }
        total_latency
    }
}
