//! Python Bindings for Hardware Management, TPU, GPU & Heterogeneous Execution.

use crate::nn::Module;
#[cfg(feature = "python")]
use pyo3::exceptions::PyRuntimeError;
#[cfg(feature = "python")]
use pyo3::prelude::*;

#[cfg(feature = "python")]
use crate::device::device_type::Device;
#[cfg(feature = "python")]
use crate::device::hardware::HardwareManager;
#[cfg(feature = "python")]
use crate::device::heterogeneous::{
    AutoDeviceMixer, HeterogeneousPipeline, HeterogeneousShardedTensor,
};
#[cfg(feature = "python")]
use crate::device::tpu::{TpuCluster, TpuGeneration};
#[cfg(feature = "python")]
use crate::error::Error;
#[cfg(feature = "python")]
use crate::nn::Linear;
#[cfg(feature = "python")]
use crate::python::PyTensor;

#[cfg(feature = "python")]
fn to_py_err(err: Error) -> PyErr {
    PyRuntimeError::new_err(err.to_string())
}

/// Unified Hardware Resource Manager.
#[cfg(feature = "python")]
#[pyclass(name = "HardwareManager", module = "oxideon")]
pub struct PyHardwareManager;

#[cfg(feature = "python")]
#[pymethods]
impl PyHardwareManager {
    #[new]
    pub fn new() -> Self {
        Self
    }

    /// Returns a human-readable summary of all detected compute hardware.
    pub fn summary(&self) -> String {
        HardwareManager::global().summary()
    }

    /// Returns a list of all detected devices (e.g. ['cpu', 'cuda:0', 'tpu:0:0']).
    pub fn all_devices(&self) -> Vec<String> {
        HardwareManager::global()
            .all_devices()
            .into_iter()
            .map(|d| d.to_string())
            .collect()
    }

    /// Returns count of available hardware: `(cpu_cores, gpu_count, tpu_cores)`.
    pub fn device_counts(&self) -> (usize, usize, usize) {
        HardwareManager::global().device_counts()
    }

    /// Returns the currently active default compute device.
    pub fn current_device(&self) -> String {
        HardwareManager::global().current_device().to_string()
    }

    /// Sets the current active default compute device.
    pub fn set_current_device(&self, spec: &str) -> PyResult<()> {
        let dev = Device::parse_spec(spec).map_err(to_py_err)?;
        HardwareManager::global().set_current_device(dev);
        Ok(())
    }

    /// Runs a GEMM micro-benchmark on the specified device and returns throughput in GFLOP/s.
    #[pyo3(signature = (device="cpu", matrix_dim=128))]
    pub fn benchmark_gemm(&self, py: Python<'_>, device: &str, matrix_dim: usize) -> PyResult<f64> {
        let dev = Device::parse_spec(device).map_err(to_py_err)?;
        py.detach(|| {
            HardwareManager::global()
                .benchmark_gemm(&dev, matrix_dim)
                .map_err(to_py_err)
        })
    }

    /// Returns tracked allocated memory bytes for a device.
    pub fn device_allocated_bytes(&self, device: &str) -> PyResult<usize> {
        let dev = Device::parse_spec(device).map_err(to_py_err)?;
        Ok(HardwareManager::global().device_allocated_bytes(&dev))
    }
}

impl Default for PyHardwareManager {
    fn default() -> Self {
        Self::new()
    }
}

/// Heterogeneous Pipelined Computation Model mixing CPU, GPU, and TPU.
#[cfg(feature = "python")]
#[pyclass(name = "HeterogeneousPipeline", module = "oxideon")]
pub struct PyHeterogeneousPipeline {
    inner: HeterogeneousPipeline,
}

#[cfg(feature = "python")]
#[pymethods]
impl PyHeterogeneousPipeline {
    #[new]
    pub fn new() -> Self {
        Self {
            inner: HeterogeneousPipeline::new(),
        }
    }

    /// Adds a linear transformation stage on the specified device.
    pub fn add_linear(
        &mut self,
        name: &str,
        device_spec: &str,
        in_features: usize,
        out_features: usize,
    ) -> PyResult<()> {
        let dev = Device::parse_spec(device_spec).map_err(to_py_err)?;
        let mut linear = Linear::new(in_features, out_features).map_err(to_py_err)?;
        linear.weight = linear.weight.to_device(&dev).map_err(to_py_err)?;
        if let Some(b) = linear.bias {
            linear.bias = Some(b.to_device(&dev).map_err(to_py_err)?);
        }
        self.inner.add_stage(name, dev, move |t| linear.forward(t));
        Ok(())
    }

    /// Adds a ReLU activation stage on the specified device.
    pub fn add_relu(&mut self, name: &str, device_spec: &str) -> PyResult<()> {
        let dev = Device::parse_spec(device_spec).map_err(to_py_err)?;
        self.inner.add_stage(name, dev, |t| t.relu());
        Ok(())
    }

    /// Adds a GELU activation stage on the specified device.
    pub fn add_gelu(&mut self, name: &str, device_spec: &str) -> PyResult<()> {
        let dev = Device::parse_spec(device_spec).map_err(to_py_err)?;
        self.inner.add_stage(name, dev, |t| t.gelu());
        Ok(())
    }

    /// Executes the pipeline sequentially with automatic cross-device tensor migrations.
    pub fn forward(&self, py: Python<'_>, input: &PyTensor) -> PyResult<PyTensor> {
        let out = py
            .detach(|| self.inner.forward(&input.inner))
            .map_err(to_py_err)?;
        Ok(PyTensor { inner: out })
    }

    /// Number of pipeline stages.
    pub fn __len__(&self) -> usize {
        self.inner.len()
    }

    /// Returns list of device specifications for each stage.
    pub fn devices(&self) -> Vec<String> {
        self.inner
            .devices()
            .into_iter()
            .map(|d| d.to_string())
            .collect()
    }
}

impl Default for PyHeterogeneousPipeline {
    fn default() -> Self {
        Self::new()
    }
}

/// A tensor partitioned across heterogeneous hardware devices.
#[cfg(feature = "python")]
#[pyclass(name = "HeterogeneousShardedTensor", module = "oxideon")]
pub struct PyHeterogeneousShardedTensor {
    inner: HeterogeneousShardedTensor,
}

#[cfg(feature = "python")]
#[pymethods]
impl PyHeterogeneousShardedTensor {
    /// Shards a tensor across target devices along the specified axis.
    #[staticmethod]
    #[pyo3(signature = (source, target_devices, axis=0))]
    pub fn scatter_from(
        source: &PyTensor,
        target_devices: Vec<String>,
        axis: usize,
    ) -> PyResult<Self> {
        let mut devs = Vec::new();
        for s in target_devices {
            devs.push(Device::parse_spec(&s).map_err(to_py_err)?);
        }
        let inner = HeterogeneousShardedTensor::scatter_from(&source.inner, &devs, axis)
            .map_err(to_py_err)?;
        Ok(Self { inner })
    }

    /// Gathers all shards back to a single device.
    #[pyo3(signature = (target_device="cpu"))]
    pub fn gather_to(&self, py: Python<'_>, target_device: &str) -> PyResult<PyTensor> {
        let dev = Device::parse_spec(target_device).map_err(to_py_err)?;
        let gathered = py
            .detach(|| self.inner.gather_to(&dev))
            .map_err(to_py_err)?;
        Ok(PyTensor { inner: gathered })
    }

    /// Number of device shards.
    pub fn num_shards(&self) -> usize {
        self.inner.shards.len()
    }
}

/// Automated workload placement optimizer across CPU, GPU, and TPU.
#[cfg(feature = "python")]
#[pyclass(name = "AutoDeviceMixer", module = "oxideon")]
pub struct PyAutoDeviceMixer;

#[cfg(feature = "python")]
#[pymethods]
impl PyAutoDeviceMixer {
    #[staticmethod]
    pub fn recommend_placement(
        layer_name: &str,
        param_count: usize,
        is_gemm_heavy: bool,
    ) -> String {
        AutoDeviceMixer::recommend_placement(layer_name, param_count, is_gemm_heavy).to_string()
    }
}

/// TPU Multi-Chip Pod and Mesh Topology Controller.
#[cfg(feature = "python")]
#[pyclass(name = "TpuCluster", module = "oxideon")]
pub struct PyTpuCluster {
    inner: TpuCluster,
}

#[cfg(feature = "python")]
#[pymethods]
impl PyTpuCluster {
    #[new]
    #[pyo3(signature = (generation="v5e", dim_x=2, dim_y=2, dim_z=1))]
    pub fn new(generation: &str, dim_x: usize, dim_y: usize, dim_z: usize) -> PyResult<Self> {
        let tpu_gen = match generation.to_ascii_lowercase().as_str() {
            "v2" => TpuGeneration::V2,
            "v3" => TpuGeneration::V3,
            "v4" => TpuGeneration::V4,
            "v5e" => TpuGeneration::V5e,
            "v5p" => TpuGeneration::V5p,
            "v6e" => TpuGeneration::V6e,
            other => {
                return Err(PyRuntimeError::new_err(format!(
                    "Unknown TPU generation '{other}'. Expected v2, v3, v4, v5e, v5p, v6e"
                )));
            }
        };
        // The pod is a list of simulated chips; the largest real one has under 10,000.
        let chips = dim_x.checked_mul(dim_y).and_then(|p| p.checked_mul(dim_z));
        if chips.is_none_or(|n| n > crate::limits::MAX_REPEATS) {
            return Err(PyRuntimeError::new_err(format!(
                "TPU topology {dim_x} x {dim_y} x {dim_z} has more than {} chips",
                crate::limits::MAX_REPEATS
            )));
        }
        let inner = TpuCluster::new(tpu_gen, (dim_x, dim_y, dim_z))
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        Ok(Self { inner })
    }

    pub fn total_cores(&self) -> usize {
        self.inner.total_cores()
    }

    pub fn total_hbm_gb(&self) -> f64 {
        self.inner.total_hbm_gb()
    }

    pub fn aggregate_peak_tflops(&self) -> f64 {
        self.inner.aggregate_peak_tflops()
    }

    pub fn devices(&self) -> Vec<String> {
        self.inner
            .devices()
            .into_iter()
            .map(|d| d.to_string())
            .collect()
    }
}

#[cfg(feature = "python")]
#[pyfunction]
fn cuda_available() -> bool {
    crate::backend::cuda_driver::cuda_available()
}

#[cfg(feature = "python")]
#[pyfunction]
fn cuda_device_count() -> usize {
    crate::backend::cuda_driver::cuda_device_count()
}

#[cfg(feature = "python")]
#[pyfunction]
#[pyo3(signature = (index=0))]
fn cuda_device_info(index: usize) -> PyResult<(String, usize, i32, i32, i32)> {
    let info = crate::backend::cuda_driver::cuda_device_info(index).map_err(to_py_err)?;
    Ok((
        info.name,
        info.total_memory,
        info.major,
        info.minor,
        info.multi_processor_count,
    ))
}

#[cfg(feature = "python")]
#[pyfunction]
#[pyo3(signature = (index=0))]
fn cuda_mem_info(index: usize) -> PyResult<(usize, usize)> {
    crate::backend::cuda_driver::cuda_mem_info(index).map_err(to_py_err)
}

#[cfg(feature = "python")]
#[pyfunction]
#[pyo3(signature = (index=0))]
fn cuda_synchronize(index: usize) -> PyResult<()> {
    crate::backend::cuda_driver::cuda_synchronize(index).map_err(to_py_err)
}

#[cfg(feature = "python")]
fn parse_reduce_op(op: &str) -> PyResult<crate::comm::ReduceOp> {
    match op.to_ascii_lowercase().as_str() {
        "sum" => Ok(crate::comm::ReduceOp::Sum),
        "mean" | "avg" => Ok(crate::comm::ReduceOp::Mean),
        "min" => Ok(crate::comm::ReduceOp::Min),
        "max" => Ok(crate::comm::ReduceOp::Max),
        _ => Err(PyRuntimeError::new_err(format!("unknown reduce op: {op}"))),
    }
}

#[cfg(feature = "python")]
#[pyclass(name = "CommHandle", module = "oxideon")]
pub struct PyCommHandle {
    inner: Option<crate::comm::CommHandle>,
}

#[cfg(feature = "python")]
#[pymethods]
impl PyCommHandle {
    pub fn wait(&mut self, py: Python<'_>) -> PyResult<()> {
        let Some(handle) = self.inner.take() else {
            return Ok(());
        };
        py.detach(|| handle.wait()).map_err(to_py_err)
    }
}

#[cfg(feature = "python")]
#[pyclass(name = "ProcessGroup", module = "oxideon", from_py_object)]
#[derive(Clone)]
pub struct PyProcessGroup {
    inner: crate::comm::ProcessGroup,
}

#[cfg(feature = "python")]
#[pymethods]
impl PyProcessGroup {
    #[staticmethod]
    pub fn from_env() -> PyResult<Self> {
        let inner = crate::comm::ProcessGroup::from_env().map_err(to_py_err)?;
        Ok(Self { inner })
    }

    #[staticmethod]
    pub fn tcp(py: Python<'_>, rank: usize, world: usize, master_addr: &str) -> PyResult<Self> {
        let addr = master_addr.to_string();
        let inner = py
            .detach(|| crate::comm::ProcessGroup::tcp(rank, world, &addr))
            .map_err(to_py_err)?;
        Ok(Self { inner })
    }

    pub fn rank(&self) -> usize {
        self.inner.rank()
    }

    pub fn world_size(&self) -> usize {
        self.inner.world_size()
    }

    pub fn barrier(&self, py: Python<'_>) -> PyResult<()> {
        let pg = self.inner.clone();
        py.detach(|| pg.barrier()).map_err(to_py_err)
    }

    #[pyo3(signature = (tensor, op="sum"))]
    pub fn all_reduce_tensor(&self, py: Python<'_>, tensor: &PyTensor, op: &str) -> PyResult<PyTensor> {
        let rop = parse_reduce_op(op)?;
        let pg = self.inner.clone();
        let t = tensor.inner.clone();
        let out = py.detach(|| pg.all_reduce_tensor(&t, rop)).map_err(to_py_err)?;
        Ok(PyTensor { inner: out })
    }

    #[pyo3(signature = (tensor, root=0))]
    pub fn broadcast_tensor(&self, py: Python<'_>, tensor: &PyTensor, root: usize) -> PyResult<()> {
        let pg = self.inner.clone();
        let t = tensor.inner.clone();
        py.detach(|| pg.broadcast_tensor(&t, root)).map_err(to_py_err)
    }

    #[pyo3(signature = (tensor, dim=0))]
    pub fn all_gather_tensor(&self, py: Python<'_>, tensor: &PyTensor, dim: isize) -> PyResult<PyTensor> {
        let pg = self.inner.clone();
        let t = tensor.inner.clone();
        let out = py.detach(|| pg.all_gather_tensor(&t, dim)).map_err(to_py_err)?;
        Ok(PyTensor { inner: out })
    }

    #[pyo3(signature = (tensor, op="sum"))]
    pub fn reduce_scatter_tensor(&self, py: Python<'_>, tensor: &PyTensor, op: &str) -> PyResult<PyTensor> {
        let rop = parse_reduce_op(op)?;
        let pg = self.inner.clone();
        let t = tensor.inner.clone();
        let out = py.detach(|| pg.reduce_scatter_tensor(&t, rop)).map_err(to_py_err)?;
        Ok(PyTensor { inner: out })
    }
}

#[cfg(feature = "python")]
#[pyclass(name = "FullyShardedParam", module = "oxideon")]
pub struct PyFullyShardedParam {
    inner: crate::comm::FullyShardedParam,
}

#[cfg(feature = "python")]
#[pymethods]
impl PyFullyShardedParam {
    #[staticmethod]
    pub fn from_tensor(full: &PyTensor, pg: &PyProcessGroup) -> PyResult<Self> {
        let inner = crate::comm::FullyShardedParam::from_tensor(&full.inner, &pg.inner).map_err(to_py_err)?;
        Ok(Self { inner })
    }

    pub fn shard(&self) -> PyTensor {
        PyTensor { inner: self.inner.shard().clone() }
    }

    pub fn full_shape(&self) -> Vec<usize> {
        self.inner.full_shape().to_vec()
    }

    pub fn gather(&self, py: Python<'_>, pg: &PyProcessGroup) -> PyResult<PyTensor> {
        let pg_c = pg.inner.clone();
        let out = py.detach(|| self.inner.gather(&pg_c)).map_err(to_py_err)?;
        Ok(PyTensor { inner: out })
    }

    #[pyo3(signature = (pg, full_grad, op="mean"))]
    pub fn reduce_scatter_grad(&self, py: Python<'_>, pg: &PyProcessGroup, full_grad: &PyTensor, op: &str) -> PyResult<PyTensor> {
        let rop = parse_reduce_op(op)?;
        let pg_c = pg.inner.clone();
        let g = full_grad.inner.clone();
        let out = py
            .detach(|| self.inner.reduce_scatter_grad(&pg_c, &g, rop))
            .map_err(to_py_err)?;
        Ok(PyTensor { inner: out })
    }
}

#[cfg(feature = "python")]
pub fn register_hardware_classes(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // the tokenizer classes ride along with this registration so that the large `python.rs` stays untouched
    crate::python_tokenizer::register_tokenizer_classes(m)?;
    m.add_class::<PyHardwareManager>()?;
    m.add_class::<PyHeterogeneousPipeline>()?;
    m.add_class::<PyHeterogeneousShardedTensor>()?;
    m.add_class::<PyAutoDeviceMixer>()?;
    m.add_class::<PyTpuCluster>()?;
    m.add_class::<PyCommHandle>()?;
    m.add_class::<PyProcessGroup>()?;
    m.add_class::<PyFullyShardedParam>()?;
    m.add_function(wrap_pyfunction!(cuda_available, m)?)?;
    m.add_function(wrap_pyfunction!(cuda_device_count, m)?)?;
    m.add_function(wrap_pyfunction!(cuda_device_info, m)?)?;
    m.add_function(wrap_pyfunction!(cuda_mem_info, m)?)?;
    m.add_function(wrap_pyfunction!(cuda_synchronize, m)?)?;
    Ok(())
}

#[cfg(not(feature = "python"))]
pub fn register_hardware_classes() {}
