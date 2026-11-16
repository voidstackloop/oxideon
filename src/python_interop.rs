//! High-Performance Interoperability Bridge for Oxideon.
//!
//! Provides seamless bidirectional zero-copy and zero-overhead conversions between
//! Oxideon's native [`Tensor`] and:
//! - **NumPy** (`ndarray`, `np.asarray`, `np.array`, `__array__`, `__array_interface__`)
//! - **PyTorch** (`torch.Tensor`, `to_torch()`, `from_torch()`, `to("cuda")`)
//! - **DLPack** Standard (`__dlpack__`, `__dlpack_device__` for PyTorch, JAX, CuPy, Arrow, Polars)
//! - **PIL / Pillow** (`PIL.Image`, `to_pil()`, `from_pil()`, RGB, RGBA, Grayscale L)
//! - **Pandas** (`pd.DataFrame`, `to_pandas()`, `from_pandas()`)
//! - **Polars** (`pl.DataFrame`, `to_polars()`, `from_polars()`)

#[cfg(feature = "python")]
use pyo3::exceptions::{PyImportError, PyOverflowError, PyRuntimeError, PyTypeError};
#[cfg(feature = "python")]
use pyo3::prelude::*;
#[cfg(feature = "python")]
use pyo3::types::PyDict;

#[cfg(feature = "python")]
use crate::device::Device;
#[cfg(feature = "python")]
use crate::dtype::DType;
#[cfg(feature = "python")]
use crate::error::Error;
#[cfg(feature = "python")]
use crate::python::PyTensor;
#[cfg(feature = "python")]
use crate::tensor::Tensor;

#[cfg(feature = "python")]
fn to_py_err(e: Error) -> PyErr {
    PyRuntimeError::new_err(e.to_string())
}

#[cfg(feature = "python")]
fn parse_device(s: &str) -> PyResult<Device> {
    Device::parse_spec(s).map_err(to_py_err)
}

/// Reinterprets a byte buffer as native elements (alignment- and length-checked).
#[cfg(feature = "python")]
fn cast_slice<T>(bytes: &[u8]) -> Option<&[T]> {
    let size = std::mem::size_of::<T>();
    if size == 0
        || bytes.as_ptr() as usize % std::mem::align_of::<T>() != 0
        || bytes.len() % size != 0
    {
        return None;
    }
    // SAFETY: alignment and exact length checked above; read-only view, same lifetime.
    Some(unsafe { std::slice::from_raw_parts(bytes.as_ptr() as *const T, bytes.len() / size) })
}

/// Converts an arbitrary Python array or sequence into an Oxideon [`PyTensor`].
#[cfg(feature = "python")]
pub fn from_numpy_impl(
    py: Python<'_>,
    array: &Bound<'_, PyAny>,
    device: &str,
) -> PyResult<PyTensor> {
    let np = py
        .import("numpy")
        .map_err(|_| PyImportError::new_err("from_numpy requires NumPy: pip install numpy"))?;
    let orig_is_0d = match array.getattr("shape") {
        Ok(s) => match s.extract::<Vec<usize>>() {
            Ok(shp) => shp.is_empty(),
            Err(_) => false,
        },
        Err(_) => false,
    };
    let ndarray = np.call_method1("ascontiguousarray", (array,))?;
    let shape: Vec<usize> = if orig_is_0d {
        Vec::new()
    } else {
        ndarray.getattr("shape")?.extract()?
    };
    let dtype_name: String = ndarray.getattr("dtype")?.getattr("name")?.extract()?;
    let dev = parse_device(device)?;

    // `dtype.name` drops the byte order, so a big-endian array would silently decode to
    // garbage below; normalize multi-byte dtypes to native order first ("|" is bytes/bool).
    let ndarray = {
        let order: String = ndarray.getattr("dtype")?.getattr("byteorder")?.extract()?;
        let native = order == "="
            || order == "|"
            || (cfg!(target_endian = "little") && order == "<")
            || (cfg!(target_endian = "big") && order == ">");
        if native {
            ndarray
        } else {
            let dt = ndarray
                .getattr("dtype")?
                .call_method1("newbyteorder", ("=",))?;
            ndarray.call_method1("astype", (dt,))?
        }
    };

    let inner = match dtype_name.as_str() {
        "float32" => {
            let py_bytes = ndarray.call_method0("tobytes")?;
            let bytes: &[u8] = py_bytes.extract()?;
            let mut owned = Vec::new();
            let data: &[f32] = cast_slice(bytes).unwrap_or_else(|| {
                owned = bytes
                    .chunks_exact(4)
                    .map(|c| f32::from_ne_bytes(c.try_into().unwrap()))
                    .collect();
                &owned
            });
            py.detach(|| Tensor::from_slice_on(data, &shape, &dev))
                .map_err(to_py_err)?
        }
        "float64" => {
            let py_bytes = ndarray.call_method0("tobytes")?;
            let bytes: &[u8] = py_bytes.extract()?;
            let mut owned = Vec::new();
            let data: &[f64] = cast_slice(bytes).unwrap_or_else(|| {
                owned = bytes
                    .chunks_exact(8)
                    .map(|c| f64::from_ne_bytes(c.try_into().unwrap()))
                    .collect();
                &owned
            });
            py.detach(|| Tensor::from_slice_on(data, &shape, &dev))
                .map_err(to_py_err)?
        }
        "int32" => {
            let py_bytes = ndarray.call_method0("tobytes")?;
            let bytes: &[u8] = py_bytes.extract()?;
            let mut owned = Vec::new();
            let data: &[i32] = cast_slice(bytes).unwrap_or_else(|| {
                owned = bytes
                    .chunks_exact(4)
                    .map(|c| i32::from_ne_bytes(c.try_into().unwrap()))
                    .collect();
                &owned
            });
            py.detach(|| Tensor::from_slice_on(data, &shape, &dev))
                .map_err(to_py_err)?
        }
        "int64" => {
            let py_bytes = ndarray.call_method0("tobytes")?;
            let bytes: &[u8] = py_bytes.extract()?;
            let mut owned = Vec::new();
            let data: &[i64] = cast_slice(bytes).unwrap_or_else(|| {
                owned = bytes
                    .chunks_exact(8)
                    .map(|c| i64::from_ne_bytes(c.try_into().unwrap()))
                    .collect();
                &owned
            });
            py.detach(|| Tensor::from_slice_on(data, &shape, &dev))
                .map_err(to_py_err)?
        }
        "uint8" => {
            let bytes: Vec<u8> = ndarray.call_method0("tobytes")?.extract()?;
            py.detach(|| Tensor::from_slice_on(&bytes, &shape, &dev))
                .map_err(to_py_err)?
        }
        // Narrow integers widen exactly (every value is representable).
        "int8" => {
            let py_bytes = ndarray.call_method0("tobytes")?;
            let bytes: &[u8] = py_bytes.extract()?;
            let data: Vec<i32> = bytes.iter().map(|&b| (b as i8) as i32).collect();
            py.detach(|| Tensor::from_slice_on(&data, &shape, &dev))
                .map_err(to_py_err)?
        }
        "int16" => {
            let py_bytes = ndarray.call_method0("tobytes")?;
            let bytes: &[u8] = py_bytes.extract()?;
            let data: Vec<i32> = bytes
                .chunks_exact(2)
                .map(|c| i16::from_ne_bytes(c.try_into().unwrap()) as i32)
                .collect();
            py.detach(|| Tensor::from_slice_on(&data, &shape, &dev))
                .map_err(to_py_err)?
        }
        "uint16" => {
            let py_bytes = ndarray.call_method0("tobytes")?;
            let bytes: &[u8] = py_bytes.extract()?;
            let data: Vec<i32> = bytes
                .chunks_exact(2)
                .map(|c| u16::from_ne_bytes(c.try_into().unwrap()) as i32)
                .collect();
            py.detach(|| Tensor::from_slice_on(&data, &shape, &dev))
                .map_err(to_py_err)?
        }
        "uint32" => {
            let py_bytes = ndarray.call_method0("tobytes")?;
            let bytes: &[u8] = py_bytes.extract()?;
            let data: Vec<i64> = bytes
                .chunks_exact(4)
                .map(|c| u32::from_ne_bytes(c.try_into().unwrap()) as i64)
                .collect();
            py.detach(|| Tensor::from_slice_on(&data, &shape, &dev))
                .map_err(to_py_err)?
        }
        "uint64" => {
            let py_bytes = ndarray.call_method0("tobytes")?;
            let bytes: &[u8] = py_bytes.extract()?;
            let mut data = Vec::with_capacity(bytes.len() / 8);
            for c in bytes.chunks_exact(8) {
                let v = u64::from_ne_bytes(c.try_into().unwrap());
                if v > i64::MAX as u64 {
                    return Err(PyOverflowError::new_err(format!(
                        "from_numpy: uint64 value {v} exceeds int64 range"
                    )));
                }
                data.push(v as i64);
            }
            py.detach(|| Tensor::from_slice_on(&data, &shape, &dev))
                .map_err(to_py_err)?
        }
        "bool" => {
            let py_bytes = ndarray.call_method0("tobytes")?;
            let bytes: &[u8] = py_bytes.extract()?;
            let data: Vec<bool> = bytes.iter().map(|&b| b != 0).collect();
            py.detach(|| Tensor::from_slice_on(&data, &shape, &dev))
                .map_err(to_py_err)?
        }
        "float16" => {
            let f32_arr = ndarray.call_method1("astype", ("float32",))?;
            let py_bytes = f32_arr.call_method0("tobytes")?;
            let bytes: &[u8] = py_bytes.extract()?;
            let mut owned = Vec::new();
            let data: &[f32] = cast_slice(bytes).unwrap_or_else(|| {
                owned = bytes
                    .chunks_exact(4)
                    .map(|c| f32::from_ne_bytes(c.try_into().unwrap()))
                    .collect();
                &owned
            });
            py.detach(|| {
                let t = Tensor::from_slice_on(data, &shape, &dev)?;
                t.to_dtype(DType::F16)
            })
            .map_err(to_py_err)?
        }
        "bfloat16" => {
            // ml-dtypes bfloat16: exact round-trip through f32, like the f16 arm.
            let f32_arr = ndarray.call_method1("astype", ("float32",))?;
            let py_bytes = f32_arr.call_method0("tobytes")?;
            let bytes: &[u8] = py_bytes.extract()?;
            let mut owned = Vec::new();
            let data: &[f32] = cast_slice(bytes).unwrap_or_else(|| {
                owned = bytes
                    .chunks_exact(4)
                    .map(|c| f32::from_ne_bytes(c.try_into().unwrap()))
                    .collect();
                &owned
            });
            py.detach(|| {
                let t = Tensor::from_slice_on(data, &shape, &dev)?;
                t.to_dtype(DType::BF16)
            })
            .map_err(to_py_err)?
        }
        "complex64" => {
            let c_arr = ndarray.call_method1("astype", ("<c8",))?;
            let py_bytes = c_arr.call_method0("tobytes")?;
            let bytes: &[u8] = py_bytes.extract()?;
            let pairs: Vec<[f32; 2]> = bytes
                .chunks_exact(8)
                .map(|c| [
                    f32::from_le_bytes(c[0..4].try_into().unwrap()),
                    f32::from_le_bytes(c[4..8].try_into().unwrap()),
                ])
                .collect();
            py.detach(|| {
                let t = Tensor::from_c64_vec(pairs, &shape)?;
                if dev.is_cpu() { Ok(t) } else { t.to_device(&dev) }
            })
            .map_err(to_py_err)?
        }
        "complex128" => {
            let c_arr = ndarray.call_method1("astype", ("<c16",))?;
            let py_bytes = c_arr.call_method0("tobytes")?;
            let bytes: &[u8] = py_bytes.extract()?;
            let pairs: Vec<[f64; 2]> = bytes
                .chunks_exact(16)
                .map(|c| [
                    f64::from_le_bytes(c[0..8].try_into().unwrap()),
                    f64::from_le_bytes(c[8..16].try_into().unwrap()),
                ])
                .collect();
            py.detach(|| {
                let t = Tensor::from_c128_vec(pairs, &shape)?;
                if dev.is_cpu() { Ok(t) } else { t.to_device(&dev) }
            })
            .map_err(to_py_err)?
        }
        _ => {
            return Err(PyTypeError::new_err(format!(
                "from_numpy: unsupported NumPy dtype '{dtype_name}' (supported: bool, uint8, int8, int16, uint16, int32, uint32, int64, uint64, float16, bfloat16, float32, float64)"
            )));
        }
    };
    Ok(PyTensor { inner })
}

/// Converts a [`PyTensor`] to a `torch.Tensor`.
#[cfg(feature = "python")]
pub fn to_torch_impl<'py>(
    py: Python<'py>,
    tensor: &PyTensor,
    device: Option<&str>,
) -> PyResult<Bound<'py, PyAny>> {
    let torch = py
        .import("torch")
        .map_err(|_| PyImportError::new_err("to_torch() requires PyTorch: pip install torch"))?;
    let np_arr = tensor.numpy(py)?;
    let torch_t = torch.call_method1("from_numpy", (np_arr,))?;
    if let Some(dev) = device {
        torch_t.call_method1("to", (dev,))
    } else {
        Ok(torch_t)
    }
}

/// Converts a `torch.Tensor` into an Oxideon [`PyTensor`].
#[cfg(feature = "python")]
pub fn from_torch_impl(
    py: Python<'_>,
    tensor: &Bound<'_, PyAny>,
    device: &str,
) -> PyResult<PyTensor> {
    let np_arr = if tensor.hasattr("detach")? {
        tensor
            .call_method0("detach")?
            .call_method0("cpu")?
            .call_method0("numpy")?
    } else {
        tensor.clone()
    };
    from_numpy_impl(py, &np_arr, device)
}

/// Converts an image [`PyTensor`] into a `PIL.Image.Image`.
#[cfg(feature = "python")]
pub fn to_pil_impl<'py>(
    py: Python<'py>,
    tensor: &PyTensor,
    mode: Option<&str>,
) -> PyResult<Bound<'py, PyAny>> {
    let pil_image = py
        .import("PIL.Image")
        .map_err(|_| PyImportError::new_err("to_pil() requires Pillow: pip install pillow"))?;
    let np = py
        .import("numpy")
        .map_err(|_| PyImportError::new_err("to_pil() requires NumPy: pip install numpy"))?;

    let mut arr = tensor.numpy(py)?;
    let shape: Vec<usize> = arr.getattr("shape")?.extract()?;

    if shape.len() == 4 && shape[0] == 1 {
        arr = arr.call_method1("squeeze", (0,))?;
    }
    let ndim: usize = arr.getattr("ndim")?.extract()?;
    if ndim == 3 {
        let dim0: usize = arr.getattr("shape")?.get_item(0)?.extract()?;
        if dim0 == 1 {
            arr = arr.call_method1("squeeze", (0,))?;
        } else if dim0 == 3 || dim0 == 4 {
            arr = arr.call_method1("transpose", ((1, 2, 0),))?;
        }
    }

    let dtype_name: String = arr.getattr("dtype")?.getattr("name")?.extract()?;
    if dtype_name.starts_with("float") {
        let max_val: f64 = arr.call_method0("max")?.extract()?;
        let min_val: f64 = arr.call_method0("min")?.extract()?;
        let scaled = if max_val <= 1.05 && min_val >= -0.05 {
            arr.call_method1("__mul__", (255.0,))?
        } else {
            arr
        };
        let clamped = np.call_method1("clip", (scaled, 0.0, 255.0))?;
        arr = clamped.call_method1("astype", ("uint8",))?;
    } else if dtype_name != "uint8" {
        arr = arr.call_method1("astype", ("uint8",))?;
    }

    let kwargs = PyDict::new(py);
    if let Some(m) = mode {
        kwargs.set_item("mode", m)?;
    }
    pil_image.call_method("fromarray", (arr,), Some(&kwargs))
}

/// Converts a `PIL.Image.Image` into an Oxideon [`PyTensor`].
#[cfg(feature = "python")]
pub fn from_pil_impl(
    py: Python<'_>,
    image: &Bound<'_, PyAny>,
    normalize: bool,
    as_chw: bool,
    device: &str,
) -> PyResult<PyTensor> {
    let np = py
        .import("numpy")
        .map_err(|_| PyImportError::new_err("from_pil requires NumPy: pip install numpy"))?;
    let mut arr = np.call_method1("asarray", (image,))?;
    let ndim: usize = arr.getattr("ndim")?.extract()?;

    if as_chw {
        if ndim == 3 {
            arr = arr.call_method1("transpose", ((2, 0, 1),))?;
        } else if ndim == 2 {
            arr = np.call_method1("expand_dims", (arr, 0))?;
        }
    }

    if normalize {
        let f32_arr = arr.call_method1("astype", ("float32",))?;
        arr = f32_arr.call_method1("__truediv__", (255.0,))?;
    }

    from_numpy_impl(py, &arr, device)
}

/// Converts a [`PyTensor`] to a `pandas.DataFrame`.
#[cfg(feature = "python")]
pub fn to_pandas_impl<'py>(
    py: Python<'py>,
    tensor: &PyTensor,
    columns: Option<Vec<String>>,
) -> PyResult<Bound<'py, PyAny>> {
    let pd = py
        .import("pandas")
        .map_err(|_| PyImportError::new_err("to_pandas() requires pandas: pip install pandas"))?;
    let arr = tensor.numpy(py)?;
    let kwargs = PyDict::new(py);
    if let Some(cols) = columns {
        kwargs.set_item("columns", cols)?;
    }
    pd.call_method("DataFrame", (arr,), Some(&kwargs))
}

/// Converts a `pandas.DataFrame` or Series into an Oxideon [`PyTensor`].
#[cfg(feature = "python")]
pub fn from_pandas_impl(py: Python<'_>, df: &Bound<'_, PyAny>, device: &str) -> PyResult<PyTensor> {
    let arr = if df.hasattr("to_numpy")? {
        df.call_method0("to_numpy")?
    } else {
        let np = py.import("numpy")?;
        np.call_method1("asarray", (df,))?
    };
    from_numpy_impl(py, &arr, device)
}

/// Converts a [`PyTensor`] to a `polars.DataFrame`.
#[cfg(feature = "python")]
pub fn to_polars_impl<'py>(
    py: Python<'py>,
    tensor: &PyTensor,
    schema: Option<Vec<String>>,
) -> PyResult<Bound<'py, PyAny>> {
    let pl = py
        .import("polars")
        .map_err(|_| PyImportError::new_err("to_polars() requires polars: pip install polars"))?;
    let arr = tensor.numpy(py)?;
    let kwargs = PyDict::new(py);
    if let Some(sch) = schema {
        kwargs.set_item("schema", sch)?;
    }
    pl.call_method("DataFrame", (arr,), Some(&kwargs))
}

/// Converts a `polars.DataFrame` or Series into an Oxideon [`PyTensor`].
#[cfg(feature = "python")]
pub fn from_polars_impl(py: Python<'_>, df: &Bound<'_, PyAny>, device: &str) -> PyResult<PyTensor> {
    let arr = if df.hasattr("to_numpy")? {
        df.call_method0("to_numpy")?
    } else {
        let np = py.import("numpy")?;
        np.call_method1("asarray", (df,))?
    };
    from_numpy_impl(py, &arr, device)
}

// ---------------------------------------------------------------------------
// Standalone Top-Level Module Functions
// ---------------------------------------------------------------------------

#[cfg(feature = "python")]
#[pyfunction]
#[pyo3(name = "from_numpy", signature = (array, device="cpu"))]
pub fn py_from_numpy(py: Python<'_>, array: &Bound<'_, PyAny>, device: &str) -> PyResult<PyTensor> {
    from_numpy_impl(py, array, device)
}

#[cfg(feature = "python")]
#[pyfunction]
#[pyo3(name = "to_numpy")]
pub fn py_to_numpy<'py>(py: Python<'py>, tensor: &PyTensor) -> PyResult<Bound<'py, PyAny>> {
    tensor.numpy(py)
}

#[cfg(feature = "python")]
#[pyfunction]
#[pyo3(name = "from_torch", signature = (tensor, device="cpu"))]
pub fn py_from_torch(
    py: Python<'_>,
    tensor: &Bound<'_, PyAny>,
    device: &str,
) -> PyResult<PyTensor> {
    from_torch_impl(py, tensor, device)
}

#[cfg(feature = "python")]
#[pyfunction]
#[pyo3(name = "to_torch", signature = (tensor, device=None))]
pub fn py_to_torch<'py>(
    py: Python<'py>,
    tensor: &PyTensor,
    device: Option<&str>,
) -> PyResult<Bound<'py, PyAny>> {
    to_torch_impl(py, tensor, device)
}

#[cfg(feature = "python")]
#[pyfunction]
#[pyo3(name = "from_pil", signature = (image, normalize=false, as_chw=true, device="cpu"))]
pub fn py_from_pil(
    py: Python<'_>,
    image: &Bound<'_, PyAny>,
    normalize: bool,
    as_chw: bool,
    device: &str,
) -> PyResult<PyTensor> {
    from_pil_impl(py, image, normalize, as_chw, device)
}

#[cfg(feature = "python")]
#[pyfunction]
#[pyo3(name = "to_pil", signature = (tensor, mode=None))]
pub fn py_to_pil<'py>(
    py: Python<'py>,
    tensor: &PyTensor,
    mode: Option<&str>,
) -> PyResult<Bound<'py, PyAny>> {
    to_pil_impl(py, tensor, mode)
}

#[cfg(feature = "python")]
#[pyfunction]
#[pyo3(name = "from_pandas", signature = (df, device="cpu"))]
pub fn py_from_pandas(py: Python<'_>, df: &Bound<'_, PyAny>, device: &str) -> PyResult<PyTensor> {
    from_pandas_impl(py, df, device)
}

#[cfg(feature = "python")]
#[pyfunction]
#[pyo3(name = "to_pandas", signature = (tensor, columns=None))]
pub fn py_to_pandas<'py>(
    py: Python<'py>,
    tensor: &PyTensor,
    columns: Option<Vec<String>>,
) -> PyResult<Bound<'py, PyAny>> {
    to_pandas_impl(py, tensor, columns)
}

#[cfg(feature = "python")]
#[pyfunction]
#[pyo3(name = "from_polars", signature = (df, device="cpu"))]
pub fn py_from_polars(py: Python<'_>, df: &Bound<'_, PyAny>, device: &str) -> PyResult<PyTensor> {
    from_polars_impl(py, df, device)
}

#[cfg(feature = "python")]
#[pyfunction]
#[pyo3(name = "to_polars", signature = (tensor, schema=None))]
pub fn py_to_polars<'py>(
    py: Python<'py>,
    tensor: &PyTensor,
    schema: Option<Vec<String>>,
) -> PyResult<Bound<'py, PyAny>> {
    to_polars_impl(py, tensor, schema)
}

/// Registers the interoperability functions into the top-level `oxideon` Python module.
#[cfg(feature = "python")]
pub fn register_interop_functions(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(py_from_numpy, m)?)?;
    m.add_function(wrap_pyfunction!(py_to_numpy, m)?)?;
    m.add_function(wrap_pyfunction!(py_from_torch, m)?)?;
    m.add_function(wrap_pyfunction!(py_to_torch, m)?)?;
    m.add_function(wrap_pyfunction!(py_from_pil, m)?)?;
    m.add_function(wrap_pyfunction!(py_to_pil, m)?)?;
    m.add_function(wrap_pyfunction!(py_from_pandas, m)?)?;
    m.add_function(wrap_pyfunction!(py_to_pandas, m)?)?;
    m.add_function(wrap_pyfunction!(py_from_polars, m)?)?;
    m.add_function(wrap_pyfunction!(py_to_polars, m)?)?;
    Ok(())
}
