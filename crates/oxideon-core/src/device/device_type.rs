//! Device Types, Properties, and Hardware Capabilities.
//!
//! Provides PyTorch-style device selection and introspection, extended to support complex
//! accelerator topologies (CUDA, Apple Silicon Metal/MPS, Vulkan, TPU v4/v5e mesh coordinates,
//! Groq/Cerebras/Ascend custom NPUs, and zero-memory Meta devices).
//!
//! Only `Cpu` and (with the `wgpu` feature) a real wgpu adapter execute natively. `Cuda`, `Metal`, `Vulkan`,
//! `Tpu` and `Npu` are simulated placement labels: the data stays in host memory and the CPU kernels run.
//! Use [`Device::is_simulated`] to tell them apart.

use std::fmt;
use std::str::FromStr;

#[cfg(feature = "wgpu")]
use crate::backend::WgpuDevice;
use crate::error::{Error, Result};

/// Hardware accelerator or compute device category.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DeviceType {
    Cpu,
    Cuda,
    Metal,
    Vulkan,
    Wgpu,
    Tpu,
    Npu,
    Meta,
}

impl DeviceType {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Cpu => "cpu",
            Self::Cuda => "cuda",
            Self::Metal => "metal",
            Self::Vulkan => "vulkan",
            Self::Wgpu => "wgpu",
            Self::Tpu => "tpu",
            Self::Npu => "npu",
            Self::Meta => "meta",
        }
    }
}

impl fmt::Display for DeviceType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

/// Interconnect bus and network fabric type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum InterconnectType {
    HostBus,
    PCIeGen4,
    PCIeGen5,
    NVLink4,
    NVLink5,
    UnifiedMemory,
    CXL,
    TpuOpticalCircuitSwitch,
    InfiniBand,
}

impl fmt::Display for InterconnectType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::HostBus => write!(f, "HostBus"),
            Self::PCIeGen4 => write!(f, "PCIe Gen4"),
            Self::PCIeGen5 => write!(f, "PCIe Gen5"),
            Self::NVLink4 => write!(f, "NVLink4"),
            Self::NVLink5 => write!(f, "NVLink5"),
            Self::UnifiedMemory => write!(f, "Apple/Grace UnifiedMemory"),
            Self::CXL => write!(f, "CXL 2.0"),
            Self::TpuOpticalCircuitSwitch => write!(f, "TPU Optical Circuit Switch (OCS)"),
            Self::InfiniBand => write!(f, "InfiniBand NDR"),
        }
    }
}

/// Hardware properties and compute capabilities for an accelerator device.
#[derive(Debug, Clone, PartialEq)]
pub struct DeviceProperties {
    pub name: String,
    pub device_type: DeviceType,
    pub index: Option<usize>,
    pub total_memory_bytes: usize,
    pub is_unified_memory: bool,
    pub compute_capability: Option<(u32, u32)>,
    pub peak_fp32_tflops: f64,
    pub peak_fp16_tflops: f64,
    pub peak_bf16_tflops: f64,
    pub peak_fp8_tflops: f64,
    pub interconnect: InterconnectType,
}

/// Concrete compute device instance.
#[derive(Clone, Debug)]
pub enum Device {
    /// Host CPU (optionally pinned to core or NUMA domain)
    Cpu,
    /// NVIDIA GPU via CUDA driver API or PTX
    Cuda(usize),
    /// Apple Silicon GPU (M1/M2/M3/M4) via Metal Shading Language
    Metal(usize),
    /// Discrete GPU via Vulkan SPIR-V compute pipeline
    Vulkan(usize),
    /// Cross-platform WebGPU device
    #[cfg(feature = "wgpu")]
    Wgpu(WgpuDevice),
    #[cfg(not(feature = "wgpu"))]
    Wgpu(usize),
    /// Google TPU mesh coordinate: chip index and core index
    Tpu { chip: usize, core: usize },
    /// Custom Neural Processing Unit (Groq, Cerebras, Ascend, Tenstorrent)
    Npu { name: String, index: usize },
    /// Zero-memory virtual device for dry-run shapes and graph compilation
    Meta,
}

impl PartialEq for Device {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::Cpu, Self::Cpu) => true,
            (Self::Cuda(a), Self::Cuda(b)) => a == b,
            (Self::Metal(a), Self::Metal(b)) => a == b,
            (Self::Vulkan(a), Self::Vulkan(b)) => a == b,
            #[cfg(feature = "wgpu")]
            (Self::Wgpu(a), Self::Wgpu(b)) => a == b,
            #[cfg(not(feature = "wgpu"))]
            (Self::Wgpu(a), Self::Wgpu(b)) => a == b,
            (Self::Tpu { chip: c1, core: k1 }, Self::Tpu { chip: c2, core: k2 }) => {
                c1 == c2 && k1 == k2
            }
            (
                Self::Npu {
                    name: n1,
                    index: i1,
                },
                Self::Npu {
                    name: n2,
                    index: i2,
                },
            ) => n1 == n2 && i1 == i2,
            (Self::Meta, Self::Meta) => true,
            _ => false,
        }
    }
}

impl Eq for Device {}

impl std::hash::Hash for Device {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        std::mem::discriminant(self).hash(state);
        match self {
            Self::Cpu | Self::Meta => {}
            Self::Cuda(idx) | Self::Metal(idx) | Self::Vulkan(idx) => idx.hash(state),
            #[cfg(feature = "wgpu")]
            Self::Wgpu(d) => d.id().hash(state),
            #[cfg(not(feature = "wgpu"))]
            Self::Wgpu(idx) => idx.hash(state),
            Self::Tpu { chip, core } => {
                chip.hash(state);
                core.hash(state);
            }
            Self::Npu { name, index } => {
                name.hash(state);
                index.hash(state);
            }
        }
    }
}

impl Device {
    /// Constructs default CPU device.
    pub fn cpu() -> Self {
        Self::Cpu
    }

    /// Constructs a simulated CUDA device with ordinal `index` (default 0): host memory, CPU kernels.
    pub fn cuda(index: usize) -> Self {
        Self::Cuda(index)
    }

    /// Constructs a simulated Apple Metal device with ordinal `index` (default 0): host memory, CPU kernels.
    pub fn metal(index: usize) -> Self {
        Self::Metal(index)
    }

    /// Constructs a simulated Vulkan device with ordinal `index` (default 0): host memory, CPU kernels.
    pub fn vulkan(index: usize) -> Self {
        Self::Vulkan(index)
    }

    /// Constructs a GPU device: a real wgpu adapter when the `wgpu` feature finds one, otherwise a
    /// simulated CUDA device (host memory, CPU kernels).
    pub fn gpu(index: usize) -> Self {
        #[cfg(feature = "wgpu")]
        {
            if let Ok(w) = Self::wgpu() {
                return w;
            }
        }
        Self::Cuda(index)
    }

    /// Constructs a simulated Google TPU device with `chip` and `core` indices: host memory, CPU kernels.
    pub fn tpu(chip: usize, core: usize) -> Self {
        Self::Tpu { chip, core }
    }

    /// Constructs a simulated custom NPU accelerator: host memory, CPU kernels.
    pub fn npu(name: impl Into<String>, index: usize) -> Self {
        Self::Npu {
            name: name.into(),
            index,
        }
    }

    /// Constructs zero-memory virtual Meta device.
    pub fn meta() -> Self {
        Self::Meta
    }

    /// Constructs WebGPU device instance.
    #[cfg(feature = "wgpu")]
    pub fn wgpu() -> Result<Self> {
        Ok(Self::Wgpu(WgpuDevice::new()?))
    }

    #[cfg(not(feature = "wgpu"))]
    pub fn wgpu() -> Result<Self> {
        Ok(Self::Wgpu(0))
    }

    /// Returns general device category.
    pub fn device_type(&self) -> DeviceType {
        match self {
            Self::Cpu => DeviceType::Cpu,
            Self::Cuda(_) => DeviceType::Cuda,
            Self::Metal(_) => DeviceType::Metal,
            Self::Vulkan(_) => DeviceType::Vulkan,
            Self::Wgpu(_) => DeviceType::Wgpu,
            Self::Tpu { .. } => DeviceType::Tpu,
            Self::Npu { .. } => DeviceType::Npu,
            Self::Meta => DeviceType::Meta,
        }
    }

    /// Device ordinal index if applicable.
    pub fn index(&self) -> Option<usize> {
        match self {
            Self::Cpu | Self::Meta => None,
            Self::Cuda(idx) | Self::Metal(idx) | Self::Vulkan(idx) => Some(*idx),
            #[cfg(feature = "wgpu")]
            Self::Wgpu(_) => Some(0),
            #[cfg(not(feature = "wgpu"))]
            Self::Wgpu(idx) => Some(*idx),
            Self::Tpu { chip, .. } => Some(*chip),
            Self::Npu { index, .. } => Some(*index),
        }
    }

    pub fn is_cpu(&self) -> bool {
        matches!(self, Self::Cpu)
    }

    pub fn is_cuda(&self) -> bool {
        matches!(self, Self::Cuda(_))
    }

    pub fn is_metal(&self) -> bool {
        matches!(self, Self::Metal(_))
    }

    pub fn is_vulkan(&self) -> bool {
        matches!(self, Self::Vulkan(_))
    }

    pub fn is_wgpu(&self) -> bool {
        matches!(self, Self::Wgpu(_))
    }

    pub fn is_gpu(&self) -> bool {
        matches!(
            self,
            Self::Cuda(_) | Self::Metal(_) | Self::Vulkan(_) | Self::Wgpu(_)
        )
    }

    pub fn is_tpu(&self) -> bool {
        matches!(self, Self::Tpu { .. })
    }

    pub fn is_npu(&self) -> bool {
        matches!(self, Self::Npu { .. })
    }

    /// True when the device is only a label: tensors on it live in host memory and run the CPU kernels.
    /// `Cpu`, `Meta` and (with the `wgpu` feature) `Wgpu` are not simulated; every other device is.
    pub fn is_simulated(&self) -> bool {
        let native = matches!(self, Self::Cpu | Self::Meta)
            || (cfg!(feature = "wgpu") && matches!(self, Self::Wgpu(_)));
        !native
    }

    pub fn is_meta(&self) -> bool {
        matches!(self, Self::Meta)
    }

    /// Returns true if this device is an accelerator (non-CPU, non-Meta).
    pub fn is_accelerator(&self) -> bool {
        !self.is_cpu() && !self.is_meta()
    }

    /// Discovers all available compute devices on the current host system.
    pub fn available_devices() -> Vec<Self> {
        let mut devices = vec![Self::Cpu, Self::Meta];

        // Check WebGPU availability
        #[cfg(feature = "wgpu")]
        {
            if let Ok(w) = Self::wgpu() {
                devices.push(w);
            }
        }

        // Probe CUDA devices
        let cuda_count = Self::cuda_device_count();
        for i in 0..cuda_count {
            devices.push(Self::Cuda(i));
        }

        // Probe Apple Metal devices
        let metal_count = Self::metal_device_count();
        for i in 0..metal_count {
            devices.push(Self::Metal(i));
        }

        // Probe Vulkan devices
        let vulkan_count = Self::vulkan_device_count();
        for i in 0..vulkan_count {
            devices.push(Self::Vulkan(i));
        }

        devices
    }

    /// Checks if a specific device category is available on the current system.
    pub fn is_available(dev_type: DeviceType) -> bool {
        match dev_type {
            DeviceType::Cpu | DeviceType::Meta => true,
            DeviceType::Cuda => Self::is_cuda_available(),
            DeviceType::Metal => Self::is_metal_available(),
            DeviceType::Vulkan => Self::is_vulkan_available(),
            DeviceType::Wgpu => Self::is_wgpu_available(),
            DeviceType::Tpu => false,
            DeviceType::Npu => false,
        }
    }

    /// Returns device count for the given device category.
    pub fn device_count(dev_type: DeviceType) -> usize {
        match dev_type {
            DeviceType::Cpu | DeviceType::Meta => 1,
            DeviceType::Cuda => Self::cuda_device_count(),
            DeviceType::Metal => Self::metal_device_count(),
            DeviceType::Vulkan => Self::vulkan_device_count(),
            DeviceType::Wgpu => {
                if Self::is_wgpu_available() {
                    1
                } else {
                    0
                }
            }
            DeviceType::Tpu => 0,
            DeviceType::Npu => 0,
        }
    }

    pub fn is_cuda_available() -> bool {
        Self::cuda_device_count() > 0
    }

    pub fn cuda_device_count() -> usize {
        #[cfg(target_os = "linux")]
        {
            if std::path::Path::new("/proc/driver/nvidia/version").exists()
                || std::path::Path::new("/dev/nvidiactl").exists()
            {
                let mut count = 0;
                while std::path::Path::new(&format!("/dev/nvidia{count}")).exists() {
                    count += 1;
                }
                if count > 0 {
                    return count;
                }
            }
        }
        if let Ok(val) =
            std::env::var("OXIDEON_CUDA_DEVICES").or_else(|_| std::env::var("CUDA_VISIBLE_DEVICES"))
        {
            if val == "-1" || val.is_empty() {
                return 0;
            }
            return val.split(',').filter(|s| !s.trim().is_empty()).count();
        }
        0
    }

    pub fn is_metal_available() -> bool {
        Self::metal_device_count() > 0
    }

    pub fn metal_device_count() -> usize {
        #[cfg(target_os = "macos")]
        {
            1
        }
        #[cfg(not(target_os = "macos"))]
        {
            if let Ok(val) = std::env::var("OXIDEON_METAL_DEVICES") {
                val.parse::<usize>().unwrap_or(0)
            } else {
                0
            }
        }
    }

    pub fn is_vulkan_available() -> bool {
        Self::vulkan_device_count() > 0
    }

    pub fn vulkan_device_count() -> usize {
        #[cfg(target_os = "linux")]
        {
            if std::path::Path::new("/usr/share/vulkan/icd.d").exists()
                || std::path::Path::new("/etc/vulkan/icd.d").exists()
            {
                return 1;
            }
        }
        if let Ok(val) = std::env::var("OXIDEON_VULKAN_DEVICES") {
            val.parse::<usize>().unwrap_or(0)
        } else {
            0
        }
    }

    pub fn is_wgpu_available() -> bool {
        #[cfg(feature = "wgpu")]
        {
            WgpuDevice::new().is_ok()
        }
        #[cfg(not(feature = "wgpu"))]
        {
            false
        }
    }

    /// Parses PyTorch-style device strings, e.g. `"cuda:0"`, `"mps"`, `"cpu"`, `"tpu:1:0"`, `"meta"`.
    pub fn parse_spec(spec: &str) -> Result<Self> {
        let s = spec.trim().to_lowercase();
        if s.is_empty() || s == "cpu" || s == "cpu:0" {
            return Ok(Self::Cpu);
        }
        if s == "meta" {
            return Ok(Self::Meta);
        }
        if s == "wgpu" || s == "wgpu:0" {
            return Self::wgpu();
        }
        // Round-trips for the `Display` forms: `wgpu(name)` with the feature
        // (verified against the default adapter) and `wgpu:{idx}` without it.
        #[cfg(feature = "wgpu")]
        if let Some(inner) = s.strip_prefix("wgpu(").and_then(|r| r.strip_suffix(")")) {
            let dev = Self::wgpu()?;
            let actual = match &dev {
                Self::Wgpu(d) => d.name().to_lowercase(),
                _ => unreachable!("wgpu() always returns Wgpu"),
            };
            if actual == inner {
                return Ok(dev);
            }
            return Err(Error::Msg(format!(
                "No WebGPU adapter named '{inner}' (have '{actual}')"
            )));
        }
        #[cfg(not(feature = "wgpu"))]
        if let Some(rest) = s.strip_prefix("wgpu") {
            let idx = parse_optional_index(rest)?;
            return Ok(Self::Wgpu(idx));
        }

        // Generic GPU alias
        if let Some(rest) = s.strip_prefix("gpu") {
            let idx = parse_optional_index(rest)?;
            #[cfg(feature = "wgpu")]
            {
                if let Ok(dev) = Self::wgpu() {
                    return Ok(dev);
                }
            }
            return Ok(Self::Cuda(idx));
        }

        // Handle prefixes
        if let Some(rest) = s.strip_prefix("cuda") {
            let idx = parse_optional_index(rest)?;
            return Ok(Self::Cuda(idx));
        }
        if let Some(rest) = s.strip_prefix("metal").or_else(|| s.strip_prefix("mps")) {
            let idx = parse_optional_index(rest)?;
            return Ok(Self::Metal(idx));
        }
        if let Some(rest) = s.strip_prefix("vulkan") {
            let idx = parse_optional_index(rest)?;
            return Ok(Self::Vulkan(idx));
        }
        if let Some(rest) = s.strip_prefix("tpu") {
            // Formats: "tpu", "tpu:chip", "tpu:chip:core"
            let trimmed = rest.strip_prefix(':').unwrap_or(rest);
            if trimmed.is_empty() {
                return Ok(Self::Tpu { chip: 0, core: 0 });
            }
            let parts: Vec<&str> = trimmed.split(':').collect();
            if parts.len() > 2 {
                return Err(Error::Msg(format!(
                    "Invalid TPU spec '{spec}': expected 'tpu', 'tpu:chip' or 'tpu:chip:core'"
                )));
            }
            let chip = parts[0]
                .parse::<usize>()
                .map_err(|_| Error::Msg(format!("Invalid TPU chip index in '{spec}'")))?;
            let core = if parts.len() > 1 {
                parts[1]
                    .parse::<usize>()
                    .map_err(|_| Error::Msg(format!("Invalid TPU core index in '{spec}'")))?
            } else {
                0
            };
            return Ok(Self::Tpu { chip, core });
        }
        if let Some(rest) = s.strip_prefix("npu:") {
            // e.g. "npu:groq:0" or "npu:ascend"
            let parts: Vec<&str> = rest.split(':').collect();
            if parts[0].is_empty() {
                return Err(Error::Msg(format!(
                    "Invalid NPU spec '{spec}': empty accelerator name"
                )));
            }
            if parts.len() > 2 {
                return Err(Error::Msg(format!(
                    "Invalid NPU spec '{spec}': expected 'npu:name' or 'npu:name:index'"
                )));
            }
            let name = parts[0].to_string();
            let index = if parts.len() > 1 {
                parts[1]
                    .parse::<usize>()
                    .map_err(|_| Error::Msg(format!("Invalid NPU index in '{spec}'")))?
            } else {
                0
            };
            return Ok(Self::Npu { name, index });
        }

        Err(Error::Msg(format!(
            "Unrecognized device specifier '{spec}'. Valid examples: 'cpu', 'cuda:0', 'metal', 'mps:0', 'vulkan:0', 'tpu:0:1', 'meta', 'wgpu'."
        )))
    }

    /// Queries detailed device hardware properties and compute capability.
    pub fn properties(&self) -> DeviceProperties {
        match self {
            Self::Cpu => DeviceProperties {
                name: "Host System CPU (x86_64/AArch64)".into(),
                device_type: DeviceType::Cpu,
                index: None,
                total_memory_bytes: 64 * 1024 * 1024 * 1024, // 64 GB
                is_unified_memory: true,
                compute_capability: None,
                peak_fp32_tflops: 4.5,
                peak_fp16_tflops: 9.0,
                peak_bf16_tflops: 9.0,
                peak_fp8_tflops: 18.0,
                interconnect: InterconnectType::HostBus,
            },
            Self::Cuda(idx) => DeviceProperties {
                name: format!("NVIDIA Tensor Core Accelerator (CUDA:{idx})"),
                device_type: DeviceType::Cuda,
                index: Some(*idx),
                total_memory_bytes: 80 * 1024 * 1024 * 1024, // 80 GB HBM3
                is_unified_memory: false,
                compute_capability: Some((9, 0)), // Hopper Architecture
                peak_fp32_tflops: 67.0,
                peak_fp16_tflops: 989.0,
                peak_bf16_tflops: 989.0,
                peak_fp8_tflops: 1979.0,
                interconnect: InterconnectType::NVLink4,
            },
            Self::Metal(idx) => DeviceProperties {
                name: format!("Apple Silicon GPU Unified Memory (Metal:{idx})"),
                device_type: DeviceType::Metal,
                index: Some(*idx),
                total_memory_bytes: 128 * 1024 * 1024 * 1024, // 128 GB Unified
                is_unified_memory: true,
                compute_capability: None,
                peak_fp32_tflops: 38.0,
                peak_fp16_tflops: 76.0,
                peak_bf16_tflops: 76.0,
                peak_fp8_tflops: 152.0,
                interconnect: InterconnectType::UnifiedMemory,
            },
            Self::Vulkan(idx) => DeviceProperties {
                name: format!("Vulkan SPIR-V Compute Device ({idx})"),
                device_type: DeviceType::Vulkan,
                index: Some(*idx),
                total_memory_bytes: 24 * 1024 * 1024 * 1024,
                is_unified_memory: false,
                compute_capability: None,
                peak_fp32_tflops: 35.0,
                peak_fp16_tflops: 70.0,
                peak_bf16_tflops: 70.0,
                peak_fp8_tflops: 140.0,
                interconnect: InterconnectType::PCIeGen5,
            },
            #[cfg(feature = "wgpu")]
            Self::Wgpu(d) => DeviceProperties {
                name: format!("WebGPU Device ({})", d.name()),
                device_type: DeviceType::Wgpu,
                index: Some(0),
                total_memory_bytes: 16 * 1024 * 1024 * 1024,
                is_unified_memory: false,
                compute_capability: None,
                peak_fp32_tflops: 15.0,
                peak_fp16_tflops: 30.0,
                peak_bf16_tflops: 30.0,
                peak_fp8_tflops: 60.0,
                interconnect: InterconnectType::PCIeGen4,
            },
            #[cfg(not(feature = "wgpu"))]
            Self::Wgpu(idx) => DeviceProperties {
                name: format!("WebGPU Device ({idx})"),
                device_type: DeviceType::Wgpu,
                index: Some(*idx),
                total_memory_bytes: 16 * 1024 * 1024 * 1024,
                is_unified_memory: false,
                compute_capability: None,
                peak_fp32_tflops: 15.0,
                peak_fp16_tflops: 30.0,
                peak_bf16_tflops: 30.0,
                peak_fp8_tflops: 60.0,
                interconnect: InterconnectType::PCIeGen4,
            },
            Self::Tpu { chip, core } => DeviceProperties {
                name: format!("Google Cloud TPU v5e (Chip {chip}, Core {core})"),
                device_type: DeviceType::Tpu,
                index: Some(*chip),
                total_memory_bytes: 32 * 1024 * 1024 * 1024, // 32 GB HBM2e
                is_unified_memory: false,
                compute_capability: None,
                peak_fp32_tflops: 25.0,
                peak_fp16_tflops: 197.0,
                peak_bf16_tflops: 197.0,
                peak_fp8_tflops: 394.0,
                interconnect: InterconnectType::TpuOpticalCircuitSwitch,
            },
            Self::Npu { name, index } => DeviceProperties {
                name: format!("Neural Processing Unit ({name}:{index})"),
                device_type: DeviceType::Npu,
                index: Some(*index),
                total_memory_bytes: 64 * 1024 * 1024 * 1024,
                is_unified_memory: true,
                compute_capability: None,
                peak_fp32_tflops: 50.0,
                peak_fp16_tflops: 200.0,
                peak_bf16_tflops: 200.0,
                peak_fp8_tflops: 400.0,
                interconnect: InterconnectType::PCIeGen5,
            },
            Self::Meta => DeviceProperties {
                name: "Meta Virtual Dry-Run Device (Zero Memory Allocation)".into(),
                device_type: DeviceType::Meta,
                index: None,
                total_memory_bytes: 0,
                is_unified_memory: false,
                compute_capability: None,
                peak_fp32_tflops: f64::INFINITY,
                peak_fp16_tflops: f64::INFINITY,
                peak_bf16_tflops: f64::INFINITY,
                peak_fp8_tflops: f64::INFINITY,
                interconnect: InterconnectType::HostBus,
            },
        }
    }
}

fn parse_optional_index(rest: &str) -> Result<usize> {
    if rest.is_empty() {
        return Ok(0);
    }
    if let Some(num_str) = rest.strip_prefix(':') {
        num_str
            .parse::<usize>()
            .map_err(|_| Error::Msg(format!("Invalid integer device index '{num_str}'")))
    } else {
        Err(Error::Msg(format!(
            "Expected ':' before device index, found '{rest}'"
        )))
    }
}

impl FromStr for Device {
    type Err = Error;

    fn from_str(s: &str) -> Result<Self> {
        Self::parse_spec(s)
    }
}

impl TryFrom<&str> for Device {
    type Error = Error;

    fn try_from(value: &str) -> Result<Self> {
        Self::parse_spec(value)
    }
}

impl fmt::Display for Device {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Cpu => write!(f, "cpu"),
            Self::Cuda(idx) => write!(f, "cuda:{idx}"),
            Self::Metal(idx) => write!(f, "metal:{idx}"),
            Self::Vulkan(idx) => write!(f, "vulkan:{idx}"),
            #[cfg(feature = "wgpu")]
            Self::Wgpu(d) => write!(f, "wgpu({})", d.name()),
            #[cfg(not(feature = "wgpu"))]
            Self::Wgpu(idx) => write!(f, "wgpu:{idx}"),
            Self::Tpu { chip, core } => write!(f, "tpu:{chip}:{core}"),
            Self::Npu { name, index } => write!(f, "npu:{name}:{index}"),
            Self::Meta => write!(f, "meta"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Only Cpu and Meta (and a real wgpu adapter) are not simulated; every other device is a label over host memory.
    #[test]
    fn simulated_devices() {
        let labels = [
            Device::Cuda(0),
            Device::Metal(0),
            Device::Vulkan(0),
            Device::Tpu { chip: 0, core: 0 },
            Device::Npu {
                name: "x".into(),
                index: 0,
            },
        ];
        for d in labels {
            assert!(d.is_simulated(), "{d}");
        }
        assert!(!Device::Cpu.is_simulated() && !Device::Meta.is_simulated());
    }
}
