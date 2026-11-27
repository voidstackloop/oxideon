//! WGPU compute backend (enabled with `--features wgpu`).
//!
//! `f32` tensors stay on the device for element-wise unary / binary ops, strided gathers (transposed, broadcast and sliced
//! views), reductions and a register-blocked, shared-memory matrix multiplication. Other dtypes and the index ops are staged
//! through the host. Every kernel launch goes through `WgpuDevice::run`.

use std::sync::{Arc, OnceLock};

use wgpu::util::DeviceExt;

use crate::backend::{BinaryOp, CpuStorage, ReduceOp, UnaryOp};
use crate::dtype::DType;
use crate::error::{Error, Result};
use crate::layout::{Layout, broadcast_shapes};

/// Threads per workgroup of the 1-D kernels.
const WG: usize = 256;
/// Elements one reduction workgroup consumes per pass (each of its 256 threads loops over 32 of them).
const REDUCE_CHUNK: usize = WG * 32;
/// Output tile of the matmul kernel (`BM` rows x `BN` columns per workgroup; each thread computes 4 x 4 of them).
const TILE: usize = 64;
/// Output tile of the kernel for large products (each thread computes 8 x 8). It needs enough workgroups to keep the GPU busy,
/// so it is used only when the grid has at least `LARGE_MIN_GROUPS` of them.
const TILE_LARGE: usize = 128;
const LARGE_MIN_GROUPS: usize = 48;

#[derive(Clone, Debug)]
pub struct WgpuDevice {
    inner: Arc<WgpuContext>,
}

#[derive(Debug)]
struct WgpuContext {
    adapter_name: String,
    /// True for CPU-based renderers such as llvmpipe / SwiftShader: they run WGSL kernels, but slower than the CPU backend.
    is_software: bool,
    /// Largest buffer one kernel can bind. wgpu answers a bigger request with a validation error that it raises as a panic.
    max_bytes: u64,
    /// Workgroups one dispatch may use per dimension (65535 on most adapters).
    max_workgroups: u32,
    /// Device-to-host copies so far: the number tests use to prove a chain of ops stayed on the GPU.
    readbacks: std::sync::atomic::AtomicUsize,
    /// Which matmul kernel to use: `None` picks by size; `OXIDEON_GPU_MATMUL=small` or `large` forces one (for comparing them
    /// on a given adapter).
    matmul_kernel: Option<bool>,
    device: wgpu::Device,
    queue: wgpu::Queue,
    unary_pipeline: wgpu::ComputePipeline,
    binary_pipeline: wgpu::ComputePipeline,
    matmul_pipeline: wgpu::ComputePipeline,
    matmul_large_pipeline: wgpu::ComputePipeline,
    gather_pipeline: wgpu::ComputePipeline,
    index_select_pipeline: wgpu::ComputePipeline,
    where_pipeline: wgpu::ComputePipeline,
    reduce_pipeline: wgpu::ComputePipeline,
    custom_pipelines: std::sync::Mutex<std::collections::HashMap<u64, wgpu::ComputePipeline>>,
}

static GLOBAL_WGPU: OnceLock<std::result::Result<WgpuDevice, String>> = OnceLock::new();

impl PartialEq for WgpuDevice {
    fn eq(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.inner, &other.inner)
    }
}

fn make_pipeline(device: &wgpu::Device, label: &str, wgsl: &str) -> wgpu::ComputePipeline {
    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some(label),
        source: wgpu::ShaderSource::Wgsl(wgsl.into()),
    });
    device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some(label),
        layout: None,
        module: &module,
        entry_point: Some("main"),
        compilation_options: Default::default(),
        cache: None,
    })
}

impl WgpuDevice {
    pub fn new() -> Result<Self> {
        GLOBAL_WGPU
            .get_or_init(|| Self::init().map_err(|e| e.to_string()))
            .clone()
            .map_err(Error::Backend)
    }

    pub fn name(&self) -> &str {
        &self.inner.adapter_name
    }

    /// Whether the adapter is a software (CPU) renderer rather than a real GPU.
    pub fn is_software(&self) -> bool {
        self.inner.is_software
    }

    pub fn id(&self) -> usize {
        Arc::as_ptr(&self.inner) as usize
    }

    /// How many times a buffer has been copied back to the host since the device was created.
    pub fn readbacks(&self) -> usize {
        self.inner
            .readbacks
            .load(std::sync::atomic::Ordering::Relaxed)
    }

    /// Compiles (with caching) and dispatches an arbitrary WGSL compute shader on this GPU.
    pub fn dispatch_wgsl(
        &self,
        wgsl: &str,
        buffers: &[&wgpu::Buffer],
        uniform: Option<&[u8]>,
        groups: (u32, u32, u32),
    ) -> Result<()> {
        use std::hash::{Hash, Hasher};
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        wgsl.hash(&mut hasher);
        let key = hasher.finish();
        let pipeline = {
            let mut cache = self
                .inner
                .custom_pipelines
                .lock()
                .map_err(|_| Error::Backend("WGPU pipeline cache lock poisoned".into()))?;
            cache
                .entry(key)
                .or_insert_with(|| make_pipeline(&self.inner.device, "oxideon-custom-wgsl", wgsl))
                .clone()
        };
        if let Some(params) = uniform {
            self.run(&pipeline, buffers, params, groups);
        } else {
            let ctx = &self.inner;
            let layout = pipeline.get_bind_group_layout(0);
            let entries: Vec<wgpu::BindGroupEntry> = buffers
                .iter()
                .enumerate()
                .map(|(i, b)| wgpu::BindGroupEntry {
                    binding: i as u32,
                    resource: b.as_entire_binding(),
                })
                .collect();
            let bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: None,
                layout: &layout,
                entries: &entries,
            });
            let mut enc = ctx
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
            {
                let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: None,
                    timestamp_writes: None,
                });
                pass.set_pipeline(&pipeline);
                pass.set_bind_group(0, &bind_group, &[]);
                pass.dispatch_workgroups(groups.0, groups.1, groups.2);
            }
            ctx.queue.submit(Some(enc.finish()));
        }
        Ok(())
    }

    /// An error instead of a wgpu validation panic when `bytes` does not fit in one buffer of this adapter.
    pub(crate) fn check_bytes(&self, what: &str, bytes: u64) -> Result<()> {
        let max = self.inner.max_bytes;
        if bytes > max {
            return Err(Error::Backend(format!(
                "{what}: {bytes} bytes exceeds the {max} bytes one buffer can hold on {}",
                self.inner.adapter_name
            )));
        }
        Ok(())
    }

    /// Dispatch grid for `items` work items at `per_group` per workgroup, folded into two dimensions when one is not enough
    /// (the kernels recover the linear index from `num_workgroups`).
    pub(crate) fn grid_1d(&self, items: usize, per_group: usize) -> (u32, u32, u32) {
        let groups = items.div_ceil(per_group).max(1);
        let x = groups.min(self.inner.max_workgroups as usize);
        (x as u32, groups.div_ceil(x) as u32, 1)
    }

    /// Whether `items` work items fit in a (two-dimensional) dispatch; larger ones run on the CPU path.
    pub(crate) fn fits_grid(&self, items: usize, per_group: usize) -> bool {
        let max = self.inner.max_workgroups as usize;
        items.div_ceil(per_group).div_ceil(max) <= max
    }

    /// A zero-initialised device buffer for `elems` `f32` values.
    pub(crate) fn out_buffer(&self, elems: usize) -> wgpu::Buffer {
        self.inner.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("oxideon-gpu-buf"),
            size: (elems.max(1) * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        })
    }

    /// Runs `pipeline` with `buffers` bound to bindings `0..` and `params` as the uniform buffer at the next binding.
    fn run(
        &self,
        pipeline: &wgpu::ComputePipeline,
        buffers: &[&wgpu::Buffer],
        params: &[u8],
        groups: (u32, u32, u32),
    ) {
        let ctx = &self.inner;
        let param_buf = ctx
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: None,
                contents: params,
                usage: wgpu::BufferUsages::UNIFORM,
            });
        let layout = pipeline.get_bind_group_layout(0);
        let mut entries: Vec<wgpu::BindGroupEntry> = buffers
            .iter()
            .enumerate()
            .map(|(i, b)| wgpu::BindGroupEntry {
                binding: i as u32,
                resource: b.as_entire_binding(),
            })
            .collect();
        entries.push(wgpu::BindGroupEntry {
            binding: buffers.len() as u32,
            resource: param_buf.as_entire_binding(),
        });
        let bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &layout,
            entries: &entries,
        });
        let mut enc = ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        {
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(groups.0, groups.1, groups.2);
        }
        ctx.queue.submit(Some(enc.finish()));
    }

    fn init() -> Result<Self> {
        let options = wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: None,
            force_fallback_adapter: false,
        };
        // Native backends (Vulkan / Metal / DX12) first. The GL backend goes through EGL, which prints driver warnings
        // (`MESA: error: ZINK ...`, `libEGL warning ...`) on many Linux systems and mostly ends in a software renderer.
        let (_instance, adapter) = [wgpu::Backends::PRIMARY, wgpu::Backends::GL]
            .into_iter()
            .find_map(|backends| {
                let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
                    backends,
                    ..Default::default()
                });
                pollster::block_on(instance.request_adapter(&options))
                    .map(|adapter| (instance, adapter))
            })
            .ok_or_else(|| Error::Backend("failed to request wgpu adapter".into()))?;

        let info = adapter.get_info();
        let adapter_name = info.name;
        let is_software = info.device_type == wgpu::DeviceType::Cpu;
        // The defaults cap one buffer at 128-256 MiB, far below what the adapter can hold; ask for what it supports.
        let supported = adapter.limits();
        let (device, queue) = pollster::block_on(adapter.request_device(
            &wgpu::DeviceDescriptor {
                label: Some("oxideon-wgpu-device"),
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits {
                    max_buffer_size: supported.max_buffer_size,
                    max_storage_buffer_binding_size: supported.max_storage_buffer_binding_size,
                    ..wgpu::Limits::default()
                },
                memory_hints: wgpu::MemoryHints::Performance,
            },
            None,
        ))
        .map_err(|e| Error::Backend(format!("failed to create wgpu device: {e}")))?;
        let limits = device.limits();
        let max_bytes = limits
            .max_buffer_size
            .min(u64::from(limits.max_storage_buffer_binding_size))
            .min(u64::from(u32::MAX)); // the kernels index with u32
        let max_workgroups = limits.max_compute_workgroups_per_dimension;

        Ok(Self {
            inner: Arc::new(WgpuContext {
                adapter_name,
                is_software,
                max_bytes,
                max_workgroups,
                readbacks: std::sync::atomic::AtomicUsize::new(0),
                matmul_kernel: match std::env::var("OXIDEON_GPU_MATMUL").as_deref() {
                    Ok("small") => Some(false),
                    Ok("large") => Some(true),
                    _ => None,
                },
                unary_pipeline: make_pipeline(&device, "oxideon-unary", &with_pow(UNARY_BODY)),
                binary_pipeline: make_pipeline(&device, "oxideon-binary", &with_pow(BINARY_BODY)),
                matmul_pipeline: make_pipeline(&device, "oxideon-matmul", MATMUL_WGSL),
                matmul_large_pipeline: make_pipeline(
                    &device,
                    "oxideon-matmul-large",
                    MATMUL_LARGE_WGSL,
                ),
                gather_pipeline: make_pipeline(&device, "oxideon-gather", GATHER_WGSL),
                index_select_pipeline: make_pipeline(
                    &device,
                    "oxideon-index-select",
                    INDEX_SELECT_WGSL,
                ),
                where_pipeline: make_pipeline(&device, "oxideon-where", WHERE_WGSL),
                reduce_pipeline: make_pipeline(&device, "oxideon-reduce", REDUCE_WGSL),
                custom_pipelines: std::sync::Mutex::new(std::collections::HashMap::new()),
                device,
                queue,
            }),
        })
    }
}

#[derive(Clone, Debug)]
pub struct WgpuStorage {
    buffer: Arc<wgpu::Buffer>,
    len: usize,
    dtype: DType,
    device: WgpuDevice,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct UnaryParams {
    numel: u32,
    op_code: u32,
    p0: f32,
    p1: f32,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct BinaryParams {
    numel: u32,
    op_code: u32,
    _pad0: u32,
    _pad1: u32,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct MatmulParams {
    m: u32,
    k: u32,
    n: u32,
    batch: u32,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct GatherParams {
    numel: u32,
    ndim: u32,
    offset: i32,
    _pad: u32,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct IndexSelectParams {
    numel: u32,
    outer: u32,
    dim: u32,
    inner: u32,
    idx_len: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct ReduceParams {
    outer: u32,
    inner: u32,
    splits: u32,
    chunk: u32,
    op: u32,
    scale: f32,
    /// Always 0; the shader adds it to constants it must not fold (infinity, NaN).
    zero: u32,
    _pad: u32,
}

impl WgpuStorage {
    pub fn dtype(&self) -> DType {
        self.dtype
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    pub fn device(&self) -> &WgpuDevice {
        &self.device
    }

    pub fn from_f32_slice(data: &[f32], dev: &WgpuDevice) -> Self {
        let bytes = if data.is_empty() {
            bytemuck::cast_slice(&[0.0f32])
        } else {
            bytemuck::cast_slice(data)
        };
        let buffer = dev
            .inner
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("oxideon-gpu-buf"),
                contents: bytes,
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_SRC
                    | wgpu::BufferUsages::COPY_DST,
            });
        Self {
            buffer: Arc::new(buffer),
            len: data.len(),
            dtype: DType::F32,
            device: dev.clone(),
        }
    }

    pub fn from_cpu(cpu: &CpuStorage, dev: &WgpuDevice) -> Result<Self> {
        if let CpuStorage::C64(v) = cpu {
            dev.check_bytes("GPU tensor", (v.len() as u64).saturating_mul(8))?;
            let flat: &[f32] = bytemuck::cast_slice(v);
            let mut s = Self::from_f32_slice(flat, dev);
            s.len = v.len();
            s.dtype = DType::C64;
            return Ok(s);
        }
        dev.check_bytes("GPU tensor", (cpu.len() as u64).saturating_mul(4))?;
        let layout = Layout::contiguous(vec![cpu.len()]);
        let data = cpu.to_vec::<f32>(&layout);
        let mut s = Self::from_f32_slice(&data, dev);
        s.dtype = cpu.dtype();
        Ok(s)
    }

    /// Wraps a freshly computed `f32` result buffer.
    pub(crate) fn from_buffer(buffer: wgpu::Buffer, len: usize, dev: &WgpuDevice) -> Self {
        Self {
            buffer: Arc::new(buffer),
            len,
            dtype: DType::F32,
            device: dev.clone(),
        }
    }

    pub(crate) fn buffer(&self) -> &wgpu::Buffer {
        &self.buffer
    }

    pub fn to_f32_vec(&self) -> Result<Vec<f32>> {
        if self.len == 0 {
            return Ok(Vec::new());
        }
        self.device
            .inner
            .readbacks
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let byte_len = (self.len * 4) as u64;
        let dev = &self.device.inner.device;
        let queue = &self.device.inner.queue;
        let staging = dev.create_buffer(&wgpu::BufferDescriptor {
            label: Some("oxideon-staging"),
            size: byte_len,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut enc = dev.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        enc.copy_buffer_to_buffer(&self.buffer, 0, &staging, 0, byte_len);
        queue.submit(Some(enc.finish()));

        let slice = staging.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx.send(r);
        });
        let _ = dev.poll(wgpu::MaintainBase::Wait);
        rx.recv()
            .map_err(|e| Error::Backend(e.to_string()))?
            .map_err(|e| Error::Backend(e.to_string()))?;

        let mapped = slice.get_mapped_range();
        let out: Vec<f32> = bytemuck::cast_slice(&mapped).to_vec();
        drop(mapped);
        staging.unmap();
        Ok(out)
    }

    pub fn to_cpu(&self) -> Result<CpuStorage> {
        if self.dtype == DType::C64 {
            let as_f32 = Self {
                buffer: self.buffer.clone(),
                len: self.len * 2,
                dtype: DType::F32,
                device: self.device.clone(),
            };
            let flat = as_f32.to_f32_vec()?;
            let pairs = flat.chunks_exact(2).map(|c| [c[0], c[1]]).collect();
            return Ok(CpuStorage::C64(pairs));
        }
        let v = self.to_f32_vec()?;
        let f32_cpu = CpuStorage::F32(v);
        if self.dtype == DType::F32 {
            Ok(f32_cpu)
        } else {
            f32_cpu.to_dtype(&Layout::contiguous(vec![self.len]), self.dtype)
        }
    }

    /// Copies the elements a strided `layout` selects into a new contiguous buffer on the device, so transposed, broadcast,
    /// sliced and flipped views never leave the GPU. `None` when the layout does not fit the kernel's `i32` arithmetic.
    fn gather(&self, layout: &Layout) -> Result<Option<Self>> {
        let numel = layout.numel();
        if numel == 0 || numel > i32::MAX as usize {
            return Ok(None);
        }
        let Ok(offset) = i32::try_from(layout.offset()) else {
            return Ok(None);
        };
        let mut meta: Vec<i32> = Vec::with_capacity(2 * layout.ndim() + 1);
        for &d in layout.shape() {
            let Ok(d) = i32::try_from(d) else {
                return Ok(None);
            };
            meta.push(d);
        }
        for &s in layout.strides() {
            let Ok(s) = i32::try_from(s) else {
                return Ok(None);
            };
            meta.push(s);
        }
        meta.push(0); // a storage buffer may not be empty (scalars have no dimensions)
        self.device.check_bytes("GPU tensor", numel as u64 * 4)?;
        let ctx = &self.device.inner;
        let meta_buf = ctx
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("oxideon-gather-meta"),
                contents: bytemuck::cast_slice(&meta),
                usage: wgpu::BufferUsages::STORAGE,
            });
        let out = self.device.out_buffer(numel);
        let params = GatherParams {
            numel: numel as u32,
            ndim: layout.ndim() as u32,
            offset,
            _pad: 0,
        };
        let groups = self.device.grid_1d(numel, WG);
        self.device.run(
            &ctx.gather_pipeline,
            &[&self.buffer, &out, &meta_buf],
            bytemuck::bytes_of(&params),
            groups,
        );
        Ok(Some(Self::from_buffer(out, numel, &self.device)))
    }

    pub fn to_contiguous(&self, layout: &Layout) -> Result<Self> {
        if layout.is_contiguous() && layout.offset() == 0 && layout.numel() == self.len {
            return Ok(self.clone());
        }
        if self.dtype == DType::F32 && self.device.fits_grid(layout.numel(), WG) {
            if let Some(out) = self.gather(layout)? {
                return Ok(out);
            }
        }
        let cpu = self.to_cpu()?.to_contiguous(layout)?;
        Self::from_cpu(&cpu, &self.device)
    }

    pub fn to_dtype(&self, layout: &Layout, dst: DType) -> Result<Self> {
        let cpu = self.to_cpu()?.to_dtype(layout, dst)?;
        Self::from_cpu(&cpu, &self.device)
    }

    pub fn unary_op(&self, layout: &Layout, op: UnaryOp) -> Result<Self> {
        let numel = layout.numel();
        if self.dtype != DType::F32
            || numel == 0
            || !self.device.fits_grid(numel, WG)
        {
            let cpu = self.to_cpu()?.unary_op(layout, op)?;
            return Self::from_cpu(&cpu, &self.device);
        }
        let (op_code, p0, p1) = match op {
            UnaryOp::Neg => (0, 0.0, 0.0),
            UnaryOp::Abs => (1, 0.0, 0.0),
            UnaryOp::Exp => (2, 0.0, 0.0),
            UnaryOp::Ln => (3, 0.0, 0.0),
            UnaryOp::Sqrt => (4, 0.0, 0.0),
            UnaryOp::Sin => (5, 0.0, 0.0),
            UnaryOp::Cos => (6, 0.0, 0.0),
            UnaryOp::Tanh => (7, 0.0, 0.0),
            UnaryOp::Sigmoid => (8, 0.0, 0.0),
            UnaryOp::Relu => (9, 0.0, 0.0),
            UnaryOp::LeakyRelu(s) => (10, s as f32, 0.0),
            UnaryOp::Silu => (11, 0.0, 0.0),
            UnaryOp::Square => (12, 0.0, 0.0),
            UnaryOp::Recip => (13, 0.0, 0.0),
            UnaryOp::Powf(p) => (14, p as f32, 0.0),
            UnaryOp::AddScalar(s) => (15, s as f32, 0.0),
            UnaryOp::MulScalar(s) => (16, s as f32, 0.0),
            UnaryOp::Clamp(lo, hi) => (17, lo as f32, hi as f32),
            UnaryOp::Sign => (18, 0.0, 0.0),
            UnaryOp::Rsqrt => (19, 0.0, 0.0),
            UnaryOp::Gelu => (20, 0.0, 0.0),
            UnaryOp::Elu(a) => (21, a as f32, 0.0),
            UnaryOp::Softplus => (22, 0.0, 0.0),
            UnaryOp::Floor => (23, 0.0, 0.0),
            UnaryOp::Ceil => (24, 0.0, 0.0),
            UnaryOp::Round => (25, 0.0, 0.0),
            UnaryOp::Erf => (26, 0.0, 0.0),
            UnaryOp::LogicalNot => (27, 0.0, 0.0),
        };
        let src = self.to_contiguous(layout)?;
        self.device.check_bytes("GPU tensor", numel as u64 * 4)?;
        let out = self.device.out_buffer(numel);
        let params = UnaryParams {
            numel: numel as u32,
            op_code,
            p0,
            p1,
        };
        let groups = self.device.grid_1d(numel, WG);
        self.device.run(
            &self.device.inner.unary_pipeline,
            &[&src.buffer, &out],
            bytemuck::bytes_of(&params),
            groups,
        );
        let mut res = Self::from_buffer(out, numel, &self.device);
        if op == UnaryOp::LogicalNot {
            res.dtype = DType::Bool;
        }
        Ok(res)
    }

    /// Native GPU `index_select` for `F32` tensors along `ax` (zero CPU readbacks).
    pub fn index_select(&self, layout: &Layout, ax: usize, idx: &[i64]) -> Result<Option<Self>> {
        let dim = layout.shape()[ax];
        let outer: usize = layout.shape()[..ax].iter().product();
        let inner: usize = layout.shape()[ax + 1..].iter().product();
        let numel = outer * idx.len() * inner;
        if self.dtype != DType::F32
            || numel == 0
            || !self.device.fits_grid(numel, WG)
            || outer > u32::MAX as usize
            || dim > u32::MAX as usize
            || inner > u32::MAX as usize
            || idx.len() > u32::MAX as usize
        {
            return Ok(None);
        }
        let mut norm_idx = Vec::with_capacity(idx.len());
        for &r in idx {
            let i = CpuStorage::check_index("index_select", r, dim)?;
            norm_idx.push(i as u32);
        }
        let src = self.to_contiguous(layout)?;
        self.device.check_bytes("GPU tensor", numel as u64 * 4)?;
        let ctx = &self.device.inner;
        let idx_buf = ctx
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("oxideon-index-select-idx"),
                contents: bytemuck::cast_slice(&norm_idx),
                usage: wgpu::BufferUsages::STORAGE,
            });
        let out = self.device.out_buffer(numel);
        let params = IndexSelectParams {
            numel: numel as u32,
            outer: outer as u32,
            dim: dim as u32,
            inner: inner as u32,
            idx_len: idx.len() as u32,
            _pad0: 0,
            _pad1: 0,
            _pad2: 0,
        };
        let groups = self.device.grid_1d(numel, WG);
        self.device.run(
            &ctx.index_select_pipeline,
            &[&src.buffer, &idx_buf, &out],
            bytemuck::bytes_of(&params),
            groups,
        );
        Ok(Some(Self::from_buffer(out, numel, &self.device)))
    }

    pub fn binary_op(
        &self,
        lhs_l: &Layout,
        rhs: &Self,
        rhs_l: &Layout,
        op: BinaryOp,
    ) -> Result<Self> {
        let numel = lhs_l.numel();
        if self.dtype != DType::F32
            || rhs.dtype != DType::F32
            || numel == 0
            || !self.device.fits_grid(numel, WG)
        {
            let ca = self.to_cpu()?;
            let cb = rhs.to_cpu()?;
            let out = ca.binary_op(lhs_l, &cb, rhs_l, op)?;
            return Self::from_cpu(&out, &self.device);
        }
        let a_contig = self.to_contiguous(lhs_l)?;
        let b_contig = rhs.to_contiguous(rhs_l)?;
        let op_code = match op {
            BinaryOp::Add => 0,
            BinaryOp::Sub => 1,
            BinaryOp::Mul => 2,
            BinaryOp::Div => 3,
            BinaryOp::Maximum => 4,
            BinaryOp::Minimum => 5,
            BinaryOp::Pow => 6,
            BinaryOp::Fmod => 7,
            BinaryOp::Remainder => 8,
            BinaryOp::FloorDiv => 9,
        };
        self.device.check_bytes("GPU tensor", numel as u64 * 4)?;
        let out = self.device.out_buffer(numel);
        let params = BinaryParams {
            numel: numel as u32,
            op_code,
            _pad0: 0,
            _pad1: 0,
        };
        let groups = self.device.grid_1d(numel, WG);
        self.device.run(
            &self.device.inner.binary_pipeline,
            &[&a_contig.buffer, &b_contig.buffer, &out],
            bytemuck::bytes_of(&params),
            groups,
        );
        Ok(Self::from_buffer(out, numel, &self.device))
    }

    /// Elementwise comparison on the device (`f32` operands; the Bool result is stored as 0 / 1 like every non-`f32`
    /// device tensor). `None` when the operands are not `f32` or do not fit one dispatch: the caller uses the CPU path.
    pub fn cmp_op(&self, lhs_l: &Layout, rhs: &Self, rhs_l: &Layout, op: super::CmpOp) -> Result<Option<Self>> {
        let numel = lhs_l.numel();
        if self.dtype != DType::F32 || rhs.dtype != DType::F32 || numel == 0 || !self.device.fits_grid(numel, WG) {
            return Ok(None);
        }
        let op_code = match op {
            super::CmpOp::Eq => 10,
            super::CmpOp::Ne => 11,
            super::CmpOp::Lt => 12,
            super::CmpOp::Le => 13,
            super::CmpOp::Gt => 14,
            super::CmpOp::Ge => 15,
        };
        let a = self.to_contiguous(lhs_l)?;
        let b = rhs.to_contiguous(rhs_l)?;
        self.device.check_bytes("GPU tensor", numel as u64 * 4)?;
        let out = self.device.out_buffer(numel);
        let params = BinaryParams { numel: numel as u32, op_code, _pad0: 0, _pad1: 0 };
        self.device.run(&self.device.inner.binary_pipeline, &[&a.buffer, &b.buffer, &out], bytemuck::bytes_of(&params), self.device.grid_1d(numel, WG));
        let mut s = Self::from_buffer(out, numel, &self.device);
        s.dtype = DType::Bool;
        Ok(Some(s))
    }

    /// `cond ? t : f` on the device for a Bool condition and `f32` branches (all three already broadcast to one shape).
    /// `None` for other dtypes or sizes beyond one dispatch.
    pub fn where_cond(cond: &Self, cond_l: &Layout, t: &Self, t_l: &Layout, f: &Self, f_l: &Layout) -> Result<Option<Self>> {
        let numel = t_l.numel();
        if cond.dtype != DType::Bool
            || t.dtype != DType::F32
            || f.dtype != DType::F32
            || numel == 0
            || cond_l.numel() != numel
            || f_l.numel() != numel
            || !t.device.fits_grid(numel, WG)
        {
            return Ok(None);
        }
        // a Bool tensor's buffer already holds 0 / 1 as f32; gather it like any other layout
        let as_f32 = Self { buffer: cond.buffer.clone(), len: cond.len, dtype: DType::F32, device: cond.device.clone() };
        let c = as_f32.to_contiguous(cond_l)?;
        let a = t.to_contiguous(t_l)?;
        let b = f.to_contiguous(f_l)?;
        t.device.check_bytes("GPU tensor", numel as u64 * 4)?;
        let out = t.device.out_buffer(numel);
        let params = BinaryParams { numel: numel as u32, op_code: 0, _pad0: 0, _pad1: 0 };
        t.device.run(&t.device.inner.where_pipeline, &[&c.buffer, &a.buffer, &b.buffer, &out], bytemuck::bytes_of(&params), t.device.grid_1d(numel, WG));
        Ok(Some(Self::from_buffer(out, numel, &t.device)))
    }

    /// Softmax or log-softmax over the last axis of an `F32` tensor, one workgroup per row, without leaving the device
    /// (the composed max / exp / sum / div path needs a host readback per call). Rows whose maximum is not finite follow the
    /// IEEE result (NaN), as PyTorch. `None` for other dtypes and shapes the grid does not cover.
    pub fn softmax_last_axis(&self, layout: &Layout, log: bool) -> Result<Option<Self>> {
        let shape = layout.shape();
        let (Some(&n), true) = (shape.last(), self.dtype == DType::F32) else {
            return Ok(None);
        };
        let total = layout.numel();
        if n == 0 || total == 0 || !self.device.fits_grid(total / n, 1) {
            return Ok(None);
        }
        let rows = total / n;
        let src = self.to_contiguous(layout)?;
        self.device.check_bytes("GPU tensor", total as u64 * 4)?;
        let out = self.device.out_buffer(total);
        let params = [rows as u32, n as u32, log as u32, 0u32];
        let groups = self.device.grid_1d(rows, 1);
        self.device
            .dispatch_wgsl(SOFTMAX_WGSL, &[&src.buffer, &out], Some(bytemuck::bytes_of(&params)), groups)?;
        Ok(Some(Self {
            buffer: Arc::new(out),
            len: total,
            dtype: DType::F32,
            device: self.device.clone(),
        }))
    }

    pub fn reduce_op(&self, layout: &Layout, axes: &[usize], op: ReduceOp) -> Result<Self> {
        if self.dtype == DType::F32 {
            if let Some(out) = self.reduce_on_device(layout, axes, op)? {
                return Ok(out);
            }
        }
        let cpu = self.to_cpu()?.reduce_op(layout, axes, op)?;
        Self::from_cpu(&cpu, &self.device)
    }

    /// Sum / mean / max / min / prod over any set of axes without leaving the device: the layout is permuted so the reduced
    /// axes come last, gathered to a contiguous `[outer, inner]` buffer, and reduced row by row in as many passes as needed.
    /// `None` for arg-reductions and shapes the kernels do not cover.
    fn reduce_on_device(
        &self,
        layout: &Layout,
        axes: &[usize],
        op: ReduceOp,
    ) -> Result<Option<Self>> {
        let code = match op {
            ReduceOp::Sum | ReduceOp::Mean => 0u32,
            ReduceOp::Max => 1,
            ReduceOp::Min => 2,
            ReduceOp::Prod => 3,
            ReduceOp::ArgMax | ReduceOp::ArgMin => return Ok(None),
        };
        let shape = layout.shape();
        let mut is_red = vec![false; shape.len()];
        for &a in axes {
            if a >= shape.len() {
                return Ok(None);
            }
            is_red[a] = true;
        }
        let kept: Vec<usize> = (0..shape.len()).filter(|&d| !is_red[d]).collect();
        let reduced: Vec<usize> = (0..shape.len()).filter(|&d| is_red[d]).collect();
        let outer: usize = kept.iter().map(|&d| shape[d]).product();
        let inner: usize = reduced.iter().map(|&d| shape[d]).product();
        if outer == 0
            || inner == 0
            || !self
                .device
                .fits_grid(outer.saturating_mul(inner.div_ceil(REDUCE_CHUNK)), 1)
        {
            return Ok(None);
        }
        let perm: Vec<usize> = kept.iter().chain(&reduced).copied().collect();
        let src = self.to_contiguous(&layout.permute(&perm)?)?;

        let mut current = src.buffer.clone();
        let mut width = inner;
        loop {
            let splits = width.div_ceil(REDUCE_CHUNK);
            let last = splits == 1;
            let scale = if last && op == ReduceOp::Mean {
                1.0 / inner as f32
            } else {
                1.0
            };
            let total = outer * splits;
            self.device.check_bytes("GPU tensor", total as u64 * 4)?;
            let out = self.device.out_buffer(total);
            let params = ReduceParams {
                outer: outer as u32,
                inner: width as u32,
                splits: splits as u32,
                chunk: REDUCE_CHUNK as u32,
                op: code,
                scale,
                zero: 0,
                _pad: 0,
            };
            let groups = self.device.grid_1d(total, 1);
            self.device.run(
                &self.device.inner.reduce_pipeline,
                &[&current, &out],
                bytemuck::bytes_of(&params),
                groups,
            );
            current = Arc::new(out);
            width = splits;
            if last {
                break;
            }
        }
        Ok(Some(Self {
            buffer: current,
            len: outer,
            dtype: DType::F32,
            device: self.device.clone(),
        }))
    }

    pub fn matmul(&self, lhs_l: &Layout, rhs: &Self, rhs_l: &Layout) -> Result<(Self, Vec<usize>)> {
        if self.dtype != DType::F32 || rhs.dtype != DType::F32 {
            let ca = self.to_cpu()?;
            let cb = rhs.to_cpu()?;
            let (out, shape) = ca.matmul(lhs_l, &cb, rhs_l)?;
            return Ok((Self::from_cpu(&out, &self.device)?, shape));
        }
        let (ar, br) = (lhs_l.ndim(), rhs_l.ndim());
        if ar < 2 || br < 2 {
            return Err(Error::Msg("matmul requires rank >= 2".into()));
        }
        let (m, k) = (lhs_l.shape()[ar - 2], lhs_l.shape()[ar - 1]);
        let (k2, n) = (rhs_l.shape()[br - 2], rhs_l.shape()[br - 1]);
        if k != k2 {
            return Err(Error::ShapeMismatch {
                op: "matmul",
                expected: lhs_l.shape().to_vec(),
                got: rhs_l.shape().to_vec(),
            });
        }
        let batch_shape =
            broadcast_shapes("matmul", &lhs_l.shape()[..ar - 2], &rhs_l.shape()[..br - 2])?;
        let batch: usize = batch_shape.iter().product();
        let mut lhs_target = batch_shape.clone();
        lhs_target.extend([m, k]);
        let mut rhs_target = batch_shape.clone();
        rhs_target.extend([k, n]);

        let mut out_shape = batch_shape;
        out_shape.extend([m, n]);
        // A wrapped count would pass check_bytes below and let the kernel
        // write out of bounds; refuse overflow before touching the device.
        let out_numel = batch
            .checked_mul(m)
            .and_then(|v| v.checked_mul(n))
            .ok_or_else(|| {
                Error::Msg(format!(
                    "matmul: output shape {out_shape:?} element count overflows"
                ))
            })?;
        // Kernel params are u32: dims that do not fit would silently truncate.
        for (name, v) in [("m", m), ("k", k), ("n", n), ("batch", batch)] {
            if v > u32::MAX as usize {
                return Err(Error::Msg(format!(
                    "matmul: GPU kernel dim {name}={v} exceeds u32::MAX"
                )));
            }
        }
        if out_numel == 0 || k == 0 {
            let zeros = CpuStorage::F32(vec![0.0; out_numel]);
            return Ok((Self::from_cpu(&zeros, &self.device)?, out_shape));
        }

        // One workgroup computes a `tile` x `tile` block of one batch element; the grid is (n / tile, m / tile, batch).
        let large = self.device.inner.matmul_kernel.unwrap_or(
            k >= 64 && m.div_ceil(TILE_LARGE) * n.div_ceil(TILE_LARGE) * batch >= LARGE_MIN_GROUPS,
        );
        let tile = if large { TILE_LARGE } else { TILE };
        let wg = self.device.inner.max_workgroups as usize;
        if n.div_ceil(tile) > wg || m.div_ceil(tile) > wg || batch > wg {
            let (out, shape) = self.to_cpu()?.matmul(lhs_l, &rhs.to_cpu()?, rhs_l)?;
            return Ok((Self::from_cpu(&out, &self.device)?, shape));
        }
        self.device
            .check_bytes("GPU tensor", out_numel as u64 * 4)?;

        let a_contig = self.to_contiguous(&lhs_l.broadcast_to(&lhs_target)?)?;
        let b_contig = rhs.to_contiguous(&rhs_l.broadcast_to(&rhs_target)?)?;
        let out = self.device.out_buffer(out_numel);
        let params = MatmulParams {
            m: m as u32,
            k: k as u32,
            n: n as u32,
            batch: batch as u32,
        };
        let groups = (
            n.div_ceil(tile) as u32,
            m.div_ceil(tile) as u32,
            batch as u32,
        );
        let ctx = &self.device.inner;
        self.device.run(
            if large {
                &ctx.matmul_large_pipeline
            } else {
                &ctx.matmul_pipeline
            },
            &[&a_contig.buffer, &b_contig.buffer, &out],
            bytemuck::bytes_of(&params),
            groups,
        );
        Ok((Self::from_buffer(out, out_numel, &self.device), out_shape))
    }
}

/// Shared by the unary and binary kernels: `x ^ y` with the sign handling `f32::powf` has (WGSL leaves `pow` of a negative base
/// undefined). `nan_seed` is any runtime value; OR-ed into the NaN bit pattern so the compiler cannot fold it to a constant.
const POW_WGSL: &str = r#"
fn pow_f(x: f32, y: f32, nan_seed: u32) -> f32 {
    if (y == 0.0) { return 1.0; }
    if (x >= 0.0) { return pow(x, y); }
    if (y != floor(y)) { return bitcast<f32>(0x7fc00000u | nan_seed); }
    let r = pow(-x, y);
    let odd = abs(y) < 16777216.0 && (i32(abs(y)) & 1) == 1;
    return select(r, -r, odd);
}
"#;

/// `floor(a / b)` evaluated exactly (c10::div_floor_floating), shared by the binary kernels.
const FLOOR_DIV_WGSL: &str = r#"
fn floor_div_f(x: f32, y: f32) -> f32 {
    if (y == 0.0) { return x / y; }
    let m = x % y;
    var div = (x - m) / y;
    if (m != 0.0 && ((y < 0.0) != (m < 0.0))) { div = div - 1.0; }
    if (div != 0.0) {
        let fl = floor(div);
        return select(fl, fl + 1.0, div - fl > 0.5);
    }
    return select(0.0, -0.0, (x / y) < 0.0 || (x < 0.0) != (y < 0.0));
}
"#;

fn with_pow(body: &str) -> String {
    format!("{POW_WGSL}\n{FLOOR_DIV_WGSL}\n{body}")
}

const UNARY_BODY: &str = r#"
struct Params {
    numel: u32,
    op_code: u32,
    p0: f32,
    p1: f32,
}
@group(0) @binding(0) var<storage, read> src: array<f32>;
@group(0) @binding(1) var<storage, read_write> dst: array<f32>;
@group(0) @binding(2) var<uniform> params: Params;

// NaN test on the bit pattern: GPU compilers assume `x != x` is false and fold float comparisons.
fn isnan_bits(x: f32) -> bool {
    return (bitcast<u32>(x) & 0x7fffffffu) > 0x7f800000u;
}

// Abramowitz & Stegun 7.1.26, the same approximation the CPU backend uses.
fn erf_approx(x: f32) -> f32 {
    let s = select(1.0, -1.0, x < 0.0);
    let ax = abs(x);
    let t = 1.0 / (1.0 + 0.3275911 * ax);
    let y = 1.0 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * exp(-ax * ax);
    return s * y;
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
    let i = gid.y * (nwg.x * 256u) + gid.x;
    if (i >= params.numel) { return; }
    let x = src[i];
    var y: f32 = x;
    switch params.op_code {
        case 0u: { y = -x; }
        case 1u: { y = abs(x); }
        case 2u: { y = exp(x); }
        case 3u: { y = log(x); }
        case 4u: { y = sqrt(x); }
        case 5u: { y = sin(x); }
        case 6u: { y = cos(x); }
        case 7u: { y = tanh(x); }
        case 8u: { y = 1.0 / (1.0 + exp(-x)); }
        case 9u: { y = select(select(x, 0.0, x < 0.0), x, isnan_bits(x)); } // NaN passes through, like torch.relu
        case 10u: { y = select(params.p0 * x, x, x > 0.0); }
        case 11u: { y = x / (1.0 + exp(-x)); }
        case 12u: { y = x * x; }
        case 13u: { y = 1.0 / x; }
        case 14u: { y = pow_f(x, params.p0, params.numel); }
        case 15u: { y = x + params.p0; }
        case 16u: { y = x * params.p0; }
        case 17u: { y = clamp(x, params.p0, params.p1); }
        case 18u: { y = select(select(0.0, -1.0, x < 0.0), 1.0, x > 0.0); }
        case 19u: { y = 1.0 / sqrt(x); }
        case 20u: { y = 0.5 * x * (1.0 + erf_approx(x * 0.70710678118)); }
        case 21u: { y = select(params.p0 * (exp(x) - 1.0), x, x > 0.0); }
        case 22u: {
            let e = exp(x);
            y = select(select(log(1.0 + e), e * (1.0 - 0.5 * e), e < 1.0e-4), x, x > 20.0);
        }
        case 23u: { y = floor(x); }
        case 24u: { y = ceil(x); }
        // WGSL `round` rounds halves to even, which is what the CPU backend and PyTorch do
        case 25u: { y = round(x); }
        case 26u: { y = erf_approx(x); }
        case 27u: { y = select(0.0, 1.0, x == 0.0); }
        default: { y = x; }
    }
    dst[i] = y;
}
"#;

const BINARY_BODY: &str = r#"
struct Params {
    numel: u32,
    op_code: u32,
    pad0: u32,
    pad1: u32,
}
@group(0) @binding(0) var<storage, read> lhs: array<f32>;
@group(0) @binding(1) var<storage, read> rhs: array<f32>;
@group(0) @binding(2) var<storage, read_write> dst: array<f32>;
@group(0) @binding(3) var<uniform> params: Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
    let i = gid.y * (nwg.x * 256u) + gid.x;
    if (i >= params.numel) { return; }
    let a = lhs[i];
    let b = rhs[i];
    var y: f32 = a;
    switch params.op_code {
        case 0u: { y = a + b; }
        case 1u: { y = a - b; }
        case 2u: { y = a * b; }
        case 3u: { y = a / b; }
        case 4u: { y = max(a, b); if (a != a || b != b) { y = a + b; } }
        case 5u: { y = min(a, b); if (a != a || b != b) { y = a + b; } }
        case 6u: { y = pow_f(a, b, params.numel); }
        case 7u: { y = a % b; }
        case 8u: { let m = a % b; y = select(m, m + b, m != 0.0 && ((m < 0.0) != (b < 0.0))); }
        case 9u: { y = floor_div_f(a, b); }
        case 10u: { y = f32(a == b); }
        case 11u: { y = f32(!(a == b)); }
        case 12u: { y = f32(a < b); }
        case 13u: { y = f32(a <= b); }
        case 14u: { y = f32(a > b); }
        case 15u: { y = f32(a >= b); }
        default: { y = a; }
    }
    dst[i] = y;
}
"#;

/// `dst[i] = cond[i] != 0 ? on_true[i] : on_false[i]`.
const WHERE_WGSL: &str = r#"
struct Params {
    numel: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}
@group(0) @binding(0) var<storage, read> cond: array<f32>;
@group(0) @binding(1) var<storage, read> on_true: array<f32>;
@group(0) @binding(2) var<storage, read> on_false: array<f32>;
@group(0) @binding(3) var<storage, read_write> dst: array<f32>;
@group(0) @binding(4) var<uniform> params: Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
    let i = gid.y * (nwg.x * 256u) + gid.x;
    if (i >= params.numel) { return; }
    dst[i] = select(on_false[i], on_true[i], cond[i] != 0.0);
}
"#;

/// `dst[i] = src[offset + sum_d coord_d(i) * stride_d]` for the row-major coordinates of `i` in `shape`.
/// `layout_info` holds the `ndim` sizes followed by the `ndim` strides (strides may be zero or negative).
const GATHER_WGSL: &str = r#"
struct Params {
    numel: u32,
    ndim: u32,
    offset: i32,
    pad: u32,
}
@group(0) @binding(0) var<storage, read> src: array<f32>;
@group(0) @binding(1) var<storage, read_write> dst: array<f32>;
@group(0) @binding(2) var<storage, read> layout_info: array<i32>;
@group(0) @binding(3) var<uniform> params: Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
    let i = gid.y * (nwg.x * 256u) + gid.x;
    if (i >= params.numel) { return; }
    var rem = i;
    var off = params.offset;
    for (var d = params.ndim; d > 0u; d = d - 1u) {
        let dim = u32(layout_info[d - 1u]);
        let idx = rem % dim;
        rem = rem / dim;
        off = off + i32(idx) * layout_info[params.ndim + d - 1u];
    }
    dst[i] = src[u32(off)];
}
"#;

const INDEX_SELECT_WGSL: &str = r#"
struct Params {
    numel: u32,
    outer: u32,
    dim: u32,
    inner: u32,
    idx_len: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}
@group(0) @binding(0) var<storage, read> src: array<f32>;
@group(0) @binding(1) var<storage, read> idx: array<u32>;
@group(0) @binding(2) var<storage, read_write> dst: array<f32>;
@group(0) @binding(3) var<uniform> params: Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
    let i = gid.y * (nwg.x * 256u) + gid.x;
    if (i >= params.numel) { return; }
    let j = i % params.inner;
    let t = i / params.inner;
    let k = t % params.idx_len;
    let o = t / params.idx_len;
    let row = idx[k];
    dst[i] = src[(o * params.dim + row) * params.inner + j];
}
"#;

/// One pass of a row reduction: workgroup `w` reduces chunk `w % splits` of row `w / splits` of an `[outer, inner]` buffer
/// and writes one value to `dst[w]`. Each thread folds a strided share of the chunk, then a shared-memory tree combines them.
/// op: 0 sum, 1 max, 2 min, 3 prod; `scale` multiplies the result (the mean's 1/n on the last pass).
const SOFTMAX_WGSL: &str = r#"
struct P { rows: u32, n: u32, log: u32, pad: u32 };
@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read_write> y: array<f32>;
@group(0) @binding(2) var<uniform> p: P;
var<workgroup> sh: array<f32, 256>;

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(num_workgroups) nw: vec3<u32>,
        @builtin(local_invocation_index) li: u32) {
    let row = wg.y * nw.x + wg.x;
    let base = row * p.n;
    var m = -3.0e38;
    for (var i = li; i < p.n; i += 256u) {
        m = max(m, x[base + i]);
    }
    sh[li] = m;
    workgroupBarrier();
    for (var s = 128u; s > 0u; s = s >> 1u) {
        if (li < s) {
            sh[li] = max(sh[li], sh[li + s]);
        }
        workgroupBarrier();
    }
    let mx = sh[0];
    workgroupBarrier();
    var acc = 0.0;
    for (var i = li; i < p.n; i += 256u) {
        acc += exp(x[base + i] - mx);
    }
    sh[li] = acc;
    workgroupBarrier();
    for (var s = 128u; s > 0u; s = s >> 1u) {
        if (li < s) {
            sh[li] = sh[li] + sh[li + s];
        }
        workgroupBarrier();
    }
    let total = sh[0];
    let lse = log(total);
    for (var i = li; i < p.n; i += 256u) {
        let v = x[base + i] - mx;
        if (p.log == 1u) {
            y[base + i] = v - lse;
        } else {
            y[base + i] = exp(v) / total;
        }
    }
}
"#;

const REDUCE_WGSL: &str = r#"
struct Params {
    outer: u32,
    inner: u32,
    splits: u32,
    chunk: u32,
    op: u32,
    scale: f32,
    zero: u32,
    pad: u32,
}
@group(0) @binding(0) var<storage, read> src: array<f32>;
@group(0) @binding(1) var<storage, read_write> dst: array<f32>;
@group(0) @binding(2) var<uniform> params: Params;

var<workgroup> scratch: array<f32, 256>;

fn identity() -> f32 {
    switch params.op {
        case 1u: { return bitcast<f32>(0xff800000u + params.zero); }
        case 2u: { return bitcast<f32>(0x7f800000u + params.zero); }
        case 3u: { return 1.0; }
        default: { return 0.0; }
    }
}

// NaN test on the bit pattern: GPU compilers assume `x != x` is false and fold float comparisons.
fn isnan_bits(x: f32) -> bool {
    return (bitcast<u32>(x) & 0x7fffffffu) > 0x7f800000u;
}

fn combine(a: f32, b: f32) -> f32 {
    switch params.op {
        // a NaN operand makes the result NaN (torch.max / amax); plain max / min leave it unspecified
        case 1u: { return select(max(a, b), a + b, isnan_bits(a) || isnan_bits(b)); }
        case 2u: { return select(min(a, b), a + b, isnan_bits(a) || isnan_bits(b)); }
        case 3u: { return a * b; }
        default: { return a + b; }
    }
}

@compute @workgroup_size(256)
fn main(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(num_workgroups) nwg: vec3<u32>,
) {
    let w = wid.y * nwg.x + wid.x;
    if (w >= params.outer * params.splits) { return; } // the same for every thread of the workgroup
    let row = w / params.splits;
    let start = (w % params.splits) * params.chunk;
    let end = min(start + params.chunk, params.inner);
    let base = row * params.inner;
    var acc = identity();
    for (var e = start + lid.x; e < end; e = e + 256u) {
        acc = combine(acc, src[base + e]);
    }
    scratch[lid.x] = acc;
    workgroupBarrier();
    for (var s = 128u; s > 0u; s = s >> 1u) {
        if (lid.x < s) {
            scratch[lid.x] = combine(scratch[lid.x], scratch[lid.x + s]);
        }
        workgroupBarrier();
    }
    if (lid.x == 0u) {
        dst[w] = scratch[0] * params.scale;
    }
}
"#;

/// Batched matmul, `C[b] = A[b] @ B[b]` for row-major `A: [m, k]`, `B: [k, n]`.
/// A workgroup of 16 x 16 threads owns a 64 x 64 block of C; each thread keeps a 4 x 4 block in registers (columns spaced 16
/// apart so the shared-memory reads are conflict free). The k loop streams 16-wide slabs of A and B through workgroup memory.
/// Shared memory for A is padded to 65 columns so the transposing stores are conflict free too.
const MATMUL_WGSL: &str = r#"
struct Params {
    m: u32,
    k: u32,
    n: u32,
    batch: u32,
}
@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read> b: array<f32>;
@group(0) @binding(2) var<storage, read_write> c: array<f32>;
@group(0) @binding(3) var<uniform> params: Params;

const BM: u32 = 64u;
const BN: u32 = 64u;
const BK: u32 = 16u;
const APAD: u32 = 65u;

var<workgroup> tile_a: array<f32, 1040>; // [BK][APAD], k-major
var<workgroup> tile_b: array<f32, 1024>; // [BK][BN]

fn store4(base: u32, row: u32, col: u32, v: vec4<f32>, m: u32, n: u32) {
    if (row >= m) { return; }
    let o = base + row * n + col;
    if (col < n) { c[o] = v.x; }
    if (col + 16u < n) { c[o + 16u] = v.y; }
    if (col + 32u < n) { c[o + 32u] = v.z; }
    if (col + 48u < n) { c[o + 48u] = v.w; }
}

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let m = params.m;
    let k = params.k;
    let n = params.n;
    let batch = wid.z;
    let a_base = batch * m * k;
    let b_base = batch * k * n;
    let c_base = batch * m * n;
    let row0 = wid.y * BM;
    let col0 = wid.x * BN;
    let tid = lid.y * 16u + lid.x;

    var acc0 = vec4<f32>(0.0);
    var acc1 = vec4<f32>(0.0);
    var acc2 = vec4<f32>(0.0);
    var acc3 = vec4<f32>(0.0);

    let tiles = (k + BK - 1u) / BK;
    for (var t = 0u; t < tiles; t = t + 1u) {
        let k0 = t * BK;
        for (var i = 0u; i < 4u; i = i + 1u) {
            let e = tid + i * 256u;
            // A slab: 64 rows x 16 columns, read along the rows of A (16 consecutive floats per row).
            let ar = e / BK;
            let ac = e % BK;
            var va = 0.0;
            if (row0 + ar < m && k0 + ac < k) { va = a[a_base + (row0 + ar) * k + k0 + ac]; }
            tile_a[ac * APAD + ar] = va;
            // B slab: 16 rows x 64 columns, read along the rows of B (64 consecutive floats per row).
            let br = e / BN;
            let bc = e % BN;
            var vb = 0.0;
            if (k0 + br < k && col0 + bc < n) { vb = b[b_base + (k0 + br) * n + col0 + bc]; }
            tile_b[br * BN + bc] = vb;
        }
        workgroupBarrier();
        for (var p = 0u; p < BK; p = p + 1u) {
            let ai = p * APAD + lid.y * 4u;
            let a0 = tile_a[ai];
            let a1 = tile_a[ai + 1u];
            let a2 = tile_a[ai + 2u];
            let a3 = tile_a[ai + 3u];
            let bi = p * BN + lid.x;
            let bv = vec4<f32>(tile_b[bi], tile_b[bi + 16u], tile_b[bi + 32u], tile_b[bi + 48u]);
            acc0 = acc0 + a0 * bv;
            acc1 = acc1 + a1 * bv;
            acc2 = acc2 + a2 * bv;
            acc3 = acc3 + a3 * bv;
        }
        workgroupBarrier();
    }

    let r = row0 + lid.y * 4u;
    let col = col0 + lid.x;
    store4(c_base, r, col, acc0, m, n);
    store4(c_base, r + 1u, col, acc1, m, n);
    store4(c_base, r + 2u, col, acc2, m, n);
    store4(c_base, r + 3u, col, acc3, m, n);
}
"#;

/// The same product for large matrices: a 128 x 128 block per workgroup, 8 x 8 outputs per thread (sixteen `vec4` accumulators),
/// 8-wide k slabs. Each k step does 64 multiply-adds for 16 shared-memory reads, twice the ratio of the 64 x 64 kernel.
const MATMUL_LARGE_WGSL: &str = r#"
struct Params {
    m: u32,
    k: u32,
    n: u32,
    batch: u32,
}
@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read> b: array<f32>;
@group(0) @binding(2) var<storage, read_write> c: array<f32>;
@group(0) @binding(3) var<uniform> params: Params;

const BM: u32 = 128u;
const BN: u32 = 128u;
const BK: u32 = 8u;
const APAD: u32 = 129u;

var<workgroup> tile_a: array<f32, 1032>; // [BK][APAD], k-major
var<workgroup> tile_b: array<f32, 1024>; // [BK][BN]

fn store8(base: u32, row: u32, col: u32, lo: vec4<f32>, hi: vec4<f32>, m: u32, n: u32) {
    if (row >= m) { return; }
    let o = base + row * n + col;
    if (col < n) { c[o] = lo.x; }
    if (col + 16u < n) { c[o + 16u] = lo.y; }
    if (col + 32u < n) { c[o + 32u] = lo.z; }
    if (col + 48u < n) { c[o + 48u] = lo.w; }
    if (col + 64u < n) { c[o + 64u] = hi.x; }
    if (col + 80u < n) { c[o + 80u] = hi.y; }
    if (col + 96u < n) { c[o + 96u] = hi.z; }
    if (col + 112u < n) { c[o + 112u] = hi.w; }
}

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let m = params.m;
    let k = params.k;
    let n = params.n;
    let batch = wid.z;
    let a_base = batch * m * k;
    let b_base = batch * k * n;
    let c_base = batch * m * n;
    let row0 = wid.y * BM;
    let col0 = wid.x * BN;
    let tid = lid.y * 16u + lid.x;

    var c00 = vec4<f32>(0.0); var c01 = vec4<f32>(0.0);
    var c10 = vec4<f32>(0.0); var c11 = vec4<f32>(0.0);
    var c20 = vec4<f32>(0.0); var c21 = vec4<f32>(0.0);
    var c30 = vec4<f32>(0.0); var c31 = vec4<f32>(0.0);
    var c40 = vec4<f32>(0.0); var c41 = vec4<f32>(0.0);
    var c50 = vec4<f32>(0.0); var c51 = vec4<f32>(0.0);
    var c60 = vec4<f32>(0.0); var c61 = vec4<f32>(0.0);
    var c70 = vec4<f32>(0.0); var c71 = vec4<f32>(0.0);

    let tiles = (k + BK - 1u) / BK;
    for (var t = 0u; t < tiles; t = t + 1u) {
        let k0 = t * BK;
        for (var i = 0u; i < 4u; i = i + 1u) {
            let e = tid + i * 256u;
            // A slab: 128 rows x 8 columns. B slab: 8 rows x 128 columns.
            let ar = e / BK;
            let ac = e % BK;
            var va = 0.0;
            if (row0 + ar < m && k0 + ac < k) { va = a[a_base + (row0 + ar) * k + k0 + ac]; }
            tile_a[ac * APAD + ar] = va;
            let br = e / BN;
            let bc = e % BN;
            var vb = 0.0;
            if (k0 + br < k && col0 + bc < n) { vb = b[b_base + (k0 + br) * n + col0 + bc]; }
            tile_b[br * BN + bc] = vb;
        }
        workgroupBarrier();
        for (var p = 0u; p < BK; p = p + 1u) {
            let ai = p * APAD + lid.y * 8u;
            let a0 = tile_a[ai];
            let a1 = tile_a[ai + 1u];
            let a2 = tile_a[ai + 2u];
            let a3 = tile_a[ai + 3u];
            let a4 = tile_a[ai + 4u];
            let a5 = tile_a[ai + 5u];
            let a6 = tile_a[ai + 6u];
            let a7 = tile_a[ai + 7u];
            let bi = p * BN + lid.x;
            let b0 = vec4<f32>(tile_b[bi], tile_b[bi + 16u], tile_b[bi + 32u], tile_b[bi + 48u]);
            let b1 = vec4<f32>(tile_b[bi + 64u], tile_b[bi + 80u], tile_b[bi + 96u], tile_b[bi + 112u]);
            c00 = c00 + a0 * b0; c01 = c01 + a0 * b1;
            c10 = c10 + a1 * b0; c11 = c11 + a1 * b1;
            c20 = c20 + a2 * b0; c21 = c21 + a2 * b1;
            c30 = c30 + a3 * b0; c31 = c31 + a3 * b1;
            c40 = c40 + a4 * b0; c41 = c41 + a4 * b1;
            c50 = c50 + a5 * b0; c51 = c51 + a5 * b1;
            c60 = c60 + a6 * b0; c61 = c61 + a6 * b1;
            c70 = c70 + a7 * b0; c71 = c71 + a7 * b1;
        }
        workgroupBarrier();
    }

    let r = row0 + lid.y * 8u;
    let col = col0 + lid.x;
    store8(c_base, r, col, c00, c01, m, n);
    store8(c_base, r + 1u, col, c10, c11, m, n);
    store8(c_base, r + 2u, col, c20, c21, m, n);
    store8(c_base, r + 3u, col, c30, c31, m, n);
    store8(c_base, r + 4u, col, c40, c41, m, n);
    store8(c_base, r + 5u, col, c50, c51, m, n);
    store8(c_base, r + 6u, col, c60, c61, m, n);
    store8(c_base, r + 7u, col, c70, c71, m, n);
}
"#;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn wgpu_logical_not_index_select_and_c64() {
        let Ok(dev) = WgpuDevice::new() else {
            return;
        };
        let t = WgpuStorage::from_f32_slice(&[0.0, 1.0, -2.0, 0.0, 3.0, 4.0], &dev);
        let layout = Layout::contiguous(vec![3, 2]);
        let r0 = dev.readbacks();

        let not_t = t.unary_op(&layout, UnaryOp::LogicalNot).unwrap();
        assert_eq!(not_t.dtype(), DType::Bool);
        let sel = t
            .index_select(&layout, 0, &[2, 0])
            .unwrap()
            .expect("index_select should run on GPU");
        // Neither LogicalNot nor index_select should have triggered a host readback!
        assert_eq!(dev.readbacks(), r0);

        assert_eq!(not_t.to_f32_vec().unwrap(), vec![1.0, 0.0, 0.0, 1.0, 0.0, 0.0]);
        assert_eq!(sel.to_f32_vec().unwrap(), vec![3.0, 4.0, 0.0, 1.0]);

        let c_cpu = CpuStorage::C64(vec![[1.0, 2.0], [3.0, -4.0]]);
        let c_gpu = WgpuStorage::from_cpu(&c_cpu, &dev).unwrap();
        assert_eq!(c_gpu.dtype(), DType::C64);
        match c_gpu.to_cpu().unwrap() {
            CpuStorage::C64(v) => assert_eq!(v, vec![[1.0, 2.0], [3.0, -4.0]]),
            other => panic!("expected C64, got {:?}", other.dtype()),
        }
    }
}
