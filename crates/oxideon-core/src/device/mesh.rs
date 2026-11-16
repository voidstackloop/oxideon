//! Multi-Dimensional Distributed Device Mesh.
//!
//! Provides coordinate mapping, topological subgroup slicing, and communication
//! collective layout across multidimensional cluster grids (e.g. `[DP=2, TP=4, PP=2]`).

use crate::device::device_type::{Device, DeviceType};
use crate::error::{Error, Result};

/// Multi-dimensional grid of acceleration devices for distributed computation.
#[derive(Debug, Clone, PartialEq)]
pub struct DeviceMesh {
    pub shape: Vec<usize>,
    pub dim_names: Vec<String>,
    pub devices: Vec<Device>,
}

impl DeviceMesh {
    /// Creates an explicit device mesh with custom shape, dimension names, and device list.
    pub fn new(shape: &[usize], dim_names: &[&str], devices: Vec<Device>) -> Result<Self> {
        if shape.is_empty() {
            return Err(Error::Msg("DeviceMesh shape cannot be empty".into()));
        }
        let total_nodes = crate::layout::checked_numel(shape)?;
        if devices.len() != total_nodes {
            return Err(Error::Msg(format!(
                "DeviceMesh shape {shape:?} expects {total_nodes} devices, got {}",
                devices.len()
            )));
        }
        if dim_names.len() != shape.len() {
            return Err(Error::Msg(format!(
                "DeviceMesh dim_names count {} does not match shape ndim {}",
                dim_names.len(),
                shape.len()
            )));
        }
        Ok(Self {
            shape: shape.to_vec(),
            dim_names: dim_names.iter().map(|&s| s.to_string()).collect(),
            devices,
        })
    }

    /// Constructs a homogeneous multi-dimensional mesh of a single device type.
    pub fn homogeneous(
        shape: &[usize],
        dim_names: &[&str],
        device_type: DeviceType,
    ) -> Result<Self> {
        let total = crate::layout::checked_numel(shape)?;
        let total = crate::limits::check_repeats("mesh nodes", total)?;
        let devices: Vec<Device> = (0..total)
            .map(|idx| match device_type {
                DeviceType::Cpu => Device::Cpu,
                DeviceType::Cuda => Device::Cuda(idx),
                DeviceType::Metal => Device::Metal(idx),
                DeviceType::Vulkan => Device::Vulkan(idx),
                DeviceType::Meta => Device::Meta,
                DeviceType::Tpu => Device::Tpu {
                    chip: idx / 2,
                    core: idx % 2,
                },
                DeviceType::Npu => Device::Npu {
                    name: "npu".into(),
                    index: idx,
                },
                DeviceType::Wgpu => {
                    #[cfg(feature = "wgpu")]
                    {
                        Device::wgpu().unwrap_or(Device::Cpu)
                    }
                    #[cfg(not(feature = "wgpu"))]
                    {
                        Device::Wgpu(idx)
                    }
                }
            })
            .collect();
        Self::new(shape, dim_names, devices)
    }

    /// Returns the number of dimensions in the mesh.
    pub fn ndim(&self) -> usize {
        self.shape.len()
    }

    /// Total count of devices in the mesh.
    pub fn device_count(&self) -> usize {
        self.devices.len()
    }

    /// Retrieves the device at the given multi-dimensional coordinate `[idx_0, idx_1, ...]`.
    pub fn get_device(&self, coords: &[usize]) -> Result<&Device> {
        if coords.len() != self.shape.len() {
            return Err(Error::Msg(format!(
                "DeviceMesh coords {coords:?} length does not match mesh ndim {}",
                self.shape.len()
            )));
        }
        let flat_idx = self.coords_to_flat(coords)?;
        Ok(&self.devices[flat_idx])
    }

    /// Converts multi-dimensional coordinates to flat linear index.
    pub fn coords_to_flat(&self, coords: &[usize]) -> Result<usize> {
        let mut flat = 0;
        let mut stride = 1;
        for i in (0..self.shape.len()).rev() {
            if coords[i] >= self.shape[i] {
                return Err(Error::Msg(format!(
                    "Coordinate {} out of bounds for dimension '{}' (size {})",
                    coords[i], self.dim_names[i], self.shape[i]
                )));
            }
            flat += coords[i] * stride;
            stride *= self.shape[i];
        }
        Ok(flat)
    }

    /// Converts flat linear index to multi-dimensional coordinate tuple.
    pub fn flat_to_coords(&self, mut flat: usize) -> Result<Vec<usize>> {
        if flat >= self.devices.len() {
            return Err(Error::Msg(format!(
                "Flat index {flat} out of bounds for mesh of size {}",
                self.devices.len()
            )));
        }
        let mut coords = vec![0; self.shape.len()];
        for i in (0..self.shape.len()).rev() {
            coords[i] = flat % self.shape[i];
            flat /= self.shape[i];
        }
        Ok(coords)
    }

    /// Slices sub-mesh along a specified named dimension, returning the sub-group of devices.
    /// E.g. slicing along `"dp"` with `index=0` returns all TP/PP devices for DP group 0.
    pub fn slice_submesh(&self, dim_name: &str, index: usize) -> Result<Vec<Device>> {
        let dim_idx = self
            .dim_names
            .iter()
            .position(|d| d == dim_name)
            .ok_or_else(|| {
                Error::Msg(format!(
                    "Dimension '{dim_name}' not found in DeviceMesh dimensions {:?}",
                    self.dim_names
                ))
            })?;

        if index >= self.shape[dim_idx] {
            return Err(Error::Msg(format!(
                "Slice index {index} out of bounds for dimension '{dim_name}' (size {})",
                self.shape[dim_idx]
            )));
        }

        let mut sub_devices = Vec::new();
        for (flat, dev) in self.devices.iter().enumerate() {
            let coords = self.flat_to_coords(flat)?;
            if coords[dim_idx] == index {
                sub_devices.push(dev.clone());
            }
        }

        Ok(sub_devices)
    }
}
