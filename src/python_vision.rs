//! Python Bindings for Vision Datasets and Image Models.

#[cfg(feature = "python")]
use pyo3::exceptions::PyRuntimeError;
#[cfg(feature = "python")]
use pyo3::prelude::*;
#[cfg(feature = "python")]
use pyo3::types::PyDict;

#[cfg(feature = "python")]
use crate::error::Error;
#[cfg(feature = "python")]
use crate::nn::Module;
#[cfg(feature = "python")]
use crate::python::PyTensor;
#[cfg(feature = "python")]
use crate::vision::{MiniAutoEncoder, MiniImageClassifier, MiniImageDataset, MiniResNet};

#[cfg(feature = "python")]
fn to_py_err(err: Error) -> PyErr {
    PyRuntimeError::new_err(err.to_string())
}

/// Procedural 1000 Mini Image Dataset Generator.
#[cfg(feature = "python")]
#[pyclass(name = "MiniImageDataset", module = "oxideon")]
pub struct PyMiniImageDataset {
    pub(crate) inner: MiniImageDataset,
}

#[cfg(feature = "python")]
#[pymethods]
impl PyMiniImageDataset {
    /// Generates a procedural dataset of mini images with distinct geometric waveforms.
    #[staticmethod]
    #[pyo3(signature = (num_images=1000, channels=1, height=16, width=16, seed=42))]
    pub fn generate(
        num_images: usize,
        channels: usize,
        height: usize,
        width: usize,
        seed: u64,
    ) -> PyResult<Self> {
        let inner = MiniImageDataset::generate(num_images, channels, height, width, seed)
            .map_err(to_py_err)?;
        Ok(Self { inner })
    }

    /// Convenience constructor generating standard 1,000 mini images ($1 \times 16 \times 16$).
    #[staticmethod]
    pub fn standard_1000() -> PyResult<Self> {
        let inner = MiniImageDataset::standard_1000().map_err(to_py_err)?;
        Ok(Self { inner })
    }

    /// 4D Tensor containing all images `[N, channels, height, width]`.
    #[getter]
    pub fn images(&self) -> PyTensor {
        PyTensor {
            inner: self.inner.images.clone(),
        }
    }

    /// Ground-truth integer class labels `[N]`.
    #[getter]
    pub fn labels(&self) -> Vec<usize> {
        self.inner.labels.clone()
    }

    /// Number of distinct pattern classes (4: horizontal, vertical, radial, diagonal).
    #[getter]
    pub fn num_classes(&self) -> usize {
        self.inner.num_classes
    }

    /// Image dimensions `(channels, height, width)`.
    #[getter]
    pub fn image_shape(&self) -> (usize, usize, usize) {
        (self.inner.channels, self.inner.height, self.inner.width)
    }

    /// Total number of images in this dataset.
    pub fn __len__(&self) -> usize {
        self.inner.len()
    }

    /// Splits the dataset into train and validation subsets according to `train_ratio`.
    #[pyo3(signature = (train_ratio=0.8))]
    pub fn split(&self, train_ratio: f64) -> PyResult<(Self, Self)> {
        let (train, val) = self.inner.split(train_ratio).map_err(to_py_err)?;
        Ok((Self { inner: train }, Self { inner: val }))
    }

    /// Retrieves a mini-batch of images and integer labels.
    #[pyo3(signature = (batch_idx, batch_size=32))]
    pub fn batch(&self, batch_idx: usize, batch_size: usize) -> PyResult<(PyTensor, Vec<usize>)> {
        let (imgs, lbls) = self.inner.batch(batch_idx, batch_size).map_err(to_py_err)?;
        Ok((PyTensor { inner: imgs }, lbls))
    }

    /// Number of mini-batches given a batch size.
    #[pyo3(signature = (batch_size=32))]
    pub fn num_batches(&self, batch_size: usize) -> usize {
        self.inner.num_batches(batch_size)
    }

    pub fn __repr__(&self) -> String {
        format!(
            "MiniImageDataset(num_images={}, channels={}, height={}, width={}, num_classes={})",
            self.inner.len(),
            self.inner.channels,
            self.inner.height,
            self.inner.width,
            self.inner.num_classes
        )
    }
}

/// 4-Stage Mini Image Convolutional Neural Network Classifier.
#[cfg(feature = "python")]
#[pyclass(name = "MiniImageClassifier", module = "oxideon")]
pub struct PyMiniImageClassifier {
    pub(crate) inner: MiniImageClassifier,
}

#[cfg(feature = "python")]
#[pymethods]
impl PyMiniImageClassifier {
    #[new]
    #[pyo3(signature = (in_channels=1, height=16, width=16, num_classes=4))]
    pub fn new(
        in_channels: usize,
        height: usize,
        width: usize,
        num_classes: usize,
    ) -> PyResult<Self> {
        let inner =
            MiniImageClassifier::new(in_channels, height, width, num_classes).map_err(to_py_err)?;
        Ok(Self { inner })
    }

    /// Convenience constructor for standard $1 \times 16 \times 16$ 4-class mini images.
    #[staticmethod]
    pub fn standard() -> PyResult<Self> {
        let inner = MiniImageClassifier::standard().map_err(to_py_err)?;
        Ok(Self { inner })
    }

    /// Computes model output logits for input images `[Batch, C, H, W]`.
    pub fn forward(&self, py: Python<'_>, images: &PyTensor) -> PyResult<PyTensor> {
        let out = py
            .detach(|| self.inner.forward(&images.inner))
            .map_err(to_py_err)?;
        Ok(PyTensor { inner: out })
    }

    /// Returns predicted class indices for a batch of images.
    pub fn predict(&self, py: Python<'_>, images: &PyTensor) -> PyResult<Vec<usize>> {
        py.detach(|| self.inner.predict(&images.inner))
            .map_err(to_py_err)
    }

    /// Evaluates model on `images` and ground-truth `targets`, returning `(loss, accuracy)`.
    pub fn evaluate(
        &self,
        py: Python<'_>,
        images: &PyTensor,
        targets: Vec<usize>,
    ) -> PyResult<(f64, f64)> {
        py.detach(|| self.inner.evaluate(&images.inner, &targets))
            .map_err(to_py_err)
    }

    /// Trains the model on `train_dataset` with mini-batch Adam optimization.
    #[pyo3(signature = (train_dataset, val_dataset=None, epochs=5, batch_size=32, lr=0.003))]
    pub fn fit<'py>(
        &mut self,
        py: Python<'py>,
        train_dataset: &PyMiniImageDataset,
        val_dataset: Option<&PyMiniImageDataset>,
        epochs: usize,
        batch_size: usize,
        lr: f64,
    ) -> PyResult<Bound<'py, PyDict>> {
        let history = py
            .detach(|| {
                let val_ref = val_dataset.map(|v| &v.inner);
                self.inner
                    .fit(&train_dataset.inner, val_ref, epochs, batch_size, lr)
            })
            .map_err(to_py_err)?;

        let dict = PyDict::new(py);
        dict.set_item("final_train_accuracy", history.final_train_accuracy)?;
        if let Some(va) = history.final_val_accuracy {
            dict.set_item("final_val_accuracy", va)?;
        }

        let epochs_list = pyo3::types::PyList::empty(py);
        for m in history.history {
            let ep_dict = PyDict::new(py);
            ep_dict.set_item("epoch", m.epoch)?;
            ep_dict.set_item("train_loss", m.train_loss)?;
            ep_dict.set_item("train_accuracy", m.train_accuracy)?;
            if let Some(vl) = m.val_loss {
                ep_dict.set_item("val_loss", vl)?;
            }
            if let Some(va) = m.val_accuracy {
                ep_dict.set_item("val_accuracy", va)?;
            }
            epochs_list.append(ep_dict)?;
        }
        dict.set_item("history", epochs_list)?;

        Ok(dict)
    }

    /// Returns the total count of learnable parameters.
    pub fn num_parameters(&self) -> usize {
        self.inner
            .parameters()
            .iter()
            .map(|p| p.shape().iter().product::<usize>())
            .sum()
    }

    pub fn __repr__(&self) -> String {
        format!(
            "MiniImageClassifier(in_channels={}, num_classes={}, parameters={})",
            self.inner.in_channels,
            self.inner.num_classes,
            self.num_parameters()
        )
    }
}

/// Compact Residual Convolutional Neural Network (MiniResNet).
#[cfg(feature = "python")]
#[pyclass(name = "MiniResNet", module = "oxideon")]
pub struct PyMiniResNet {
    pub(crate) inner: MiniResNet,
}

#[cfg(feature = "python")]
#[pymethods]
impl PyMiniResNet {
    #[new]
    #[pyo3(signature = (in_channels=1, num_classes=4))]
    pub fn new(in_channels: usize, num_classes: usize) -> PyResult<Self> {
        let inner = MiniResNet::new(in_channels, num_classes).map_err(to_py_err)?;
        Ok(Self { inner })
    }

    #[staticmethod]
    pub fn standard() -> PyResult<Self> {
        let inner = MiniResNet::standard().map_err(to_py_err)?;
        Ok(Self { inner })
    }

    pub fn forward(&self, py: Python<'_>, images: &PyTensor) -> PyResult<PyTensor> {
        let out = py
            .detach(|| self.inner.forward(&images.inner))
            .map_err(to_py_err)?;
        Ok(PyTensor { inner: out })
    }

    pub fn predict(&self, py: Python<'_>, images: &PyTensor) -> PyResult<Vec<usize>> {
        py.detach(|| self.inner.predict(&images.inner))
            .map_err(to_py_err)
    }

    pub fn evaluate(
        &self,
        py: Python<'_>,
        images: &PyTensor,
        targets: Vec<usize>,
    ) -> PyResult<(f64, f64)> {
        py.detach(|| self.inner.evaluate(&images.inner, &targets))
            .map_err(to_py_err)
    }

    #[pyo3(signature = (train_dataset, val_dataset=None, epochs=4, batch_size=32, lr=0.005))]
    pub fn fit<'py>(
        &mut self,
        py: Python<'py>,
        train_dataset: &PyMiniImageDataset,
        val_dataset: Option<&PyMiniImageDataset>,
        epochs: usize,
        batch_size: usize,
        lr: f64,
    ) -> PyResult<Bound<'py, pyo3::types::PyList>> {
        let history = py
            .detach(|| {
                let val_ref = val_dataset.map(|v| &v.inner);
                self.inner
                    .fit(&train_dataset.inner, val_ref, epochs, batch_size, lr)
            })
            .map_err(to_py_err)?;

        let list = pyo3::types::PyList::empty(py);
        for m in history {
            let ep_dict = PyDict::new(py);
            ep_dict.set_item("epoch", m.epoch)?;
            ep_dict.set_item("train_loss", m.train_loss)?;
            ep_dict.set_item("train_accuracy", m.train_accuracy)?;
            if let Some(v_acc) = m.val_accuracy {
                ep_dict.set_item("val_accuracy", v_acc)?;
            }
            list.append(ep_dict)?;
        }
        Ok(list)
    }

    pub fn num_parameters(&self) -> usize {
        self.inner
            .parameters()
            .iter()
            .map(|p| p.shape().iter().product::<usize>())
            .sum()
    }

    pub fn __repr__(&self) -> String {
        format!(
            "MiniResNet(in_channels={}, num_classes={}, parameters={})",
            self.inner.in_channels,
            self.inner.num_classes,
            self.num_parameters()
        )
    }
}

/// Compact Generative Convolutional Autoencoder (MiniAutoEncoder).
#[cfg(feature = "python")]
#[pyclass(name = "MiniAutoEncoder", module = "oxideon")]
pub struct PyMiniAutoEncoder {
    pub(crate) inner: MiniAutoEncoder,
}

#[cfg(feature = "python")]
#[pymethods]
impl PyMiniAutoEncoder {
    #[new]
    #[pyo3(signature = (in_channels=1, latent_dim=16))]
    pub fn new(in_channels: usize, latent_dim: usize) -> PyResult<Self> {
        let inner = MiniAutoEncoder::new(in_channels, latent_dim).map_err(to_py_err)?;
        Ok(Self { inner })
    }

    #[staticmethod]
    pub fn standard() -> PyResult<Self> {
        let inner = MiniAutoEncoder::standard().map_err(to_py_err)?;
        Ok(Self { inner })
    }

    pub fn encode(&self, py: Python<'_>, images: &PyTensor) -> PyResult<PyTensor> {
        let z = py
            .detach(|| self.inner.encode(&images.inner))
            .map_err(to_py_err)?;
        Ok(PyTensor { inner: z })
    }

    pub fn decode(&self, py: Python<'_>, latent: &PyTensor) -> PyResult<PyTensor> {
        let out = py
            .detach(|| self.inner.decode(&latent.inner))
            .map_err(to_py_err)?;
        Ok(PyTensor { inner: out })
    }

    pub fn forward(&self, py: Python<'_>, images: &PyTensor) -> PyResult<PyTensor> {
        let out = py
            .detach(|| self.inner.forward(&images.inner))
            .map_err(to_py_err)?;
        Ok(PyTensor { inner: out })
    }

    pub fn reconstruction_loss(&self, py: Python<'_>, images: &PyTensor) -> PyResult<f64> {
        let loss = py
            .detach(|| self.inner.reconstruction_loss(&images.inner))
            .map_err(to_py_err)?;
        let val = loss.item::<f32>().map_err(to_py_err)?;
        Ok(val as f64)
    }

    #[pyo3(signature = (x1, x2, steps=8))]
    pub fn interpolate(
        &self,
        py: Python<'_>,
        x1: &PyTensor,
        x2: &PyTensor,
        steps: usize,
    ) -> PyResult<PyTensor> {
        let interp = py
            .detach(|| self.inner.interpolate(&x1.inner, &x2.inner, steps))
            .map_err(to_py_err)?;
        Ok(PyTensor { inner: interp })
    }

    #[pyo3(signature = (num_samples=4))]
    pub fn sample(&self, py: Python<'_>, num_samples: usize) -> PyResult<PyTensor> {
        let samples = py
            .detach(|| self.inner.sample(num_samples))
            .map_err(to_py_err)?;
        Ok(PyTensor { inner: samples })
    }

    #[pyo3(signature = (train_dataset, val_dataset=None, epochs=4, batch_size=32, lr=0.005))]
    pub fn fit<'py>(
        &mut self,
        py: Python<'py>,
        train_dataset: &PyMiniImageDataset,
        val_dataset: Option<&PyMiniImageDataset>,
        epochs: usize,
        batch_size: usize,
        lr: f64,
    ) -> PyResult<Bound<'py, pyo3::types::PyList>> {
        let history = py
            .detach(|| {
                let val_ref = val_dataset.map(|v| &v.inner);
                self.inner
                    .fit(&train_dataset.inner, val_ref, epochs, batch_size, lr)
            })
            .map_err(to_py_err)?;

        let list = pyo3::types::PyList::empty(py);
        for m in history {
            let ep_dict = PyDict::new(py);
            ep_dict.set_item("epoch", m.epoch)?;
            ep_dict.set_item("train_mse_loss", m.train_loss)?;
            if let Some(v_loss) = m.val_loss {
                ep_dict.set_item("val_mse_loss", v_loss)?;
            }
            list.append(ep_dict)?;
        }
        Ok(list)
    }

    pub fn num_parameters(&self) -> usize {
        self.inner
            .parameters()
            .iter()
            .map(|p| p.shape().iter().product::<usize>())
            .sum()
    }

    pub fn __repr__(&self) -> String {
        format!(
            "MiniAutoEncoder(in_channels={}, latent_dim={}, parameters={})",
            self.inner.in_channels,
            self.inner.latent_dim,
            self.num_parameters()
        )
    }
}

/// Registers vision dataset and model classes into the PyO3 module.
#[cfg(feature = "python")]
pub fn register_vision_classes(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyMiniImageDataset>()?;
    m.add_class::<PyMiniImageClassifier>()?;
    m.add_class::<PyMiniResNet>()?;
    m.add_class::<PyMiniAutoEncoder>()?;
    Ok(())
}
