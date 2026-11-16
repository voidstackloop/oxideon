//! Programmable Photonic Mesh & Mach-Zehnder Interferometer (MZI) Optical Computing.
//!
//! Implements coherent nanophotonic circuits (Shen et al., Nature Photonics 2017)
//! for speed-of-light, zero-energy analog linear transformations:
//! - [`MachZehnderInterferometer`]: 2x2 optical phase interference unit with internal phase $\theta$ and external phase $\phi$.
//! - [`PhotonicMesh`]: Clements-style rectangular mesh of MZIs parameterizing unitary transformations $U \in U(N)$.
//! - [`PhotonicLinear`]: Optical linear layer implementing $W = U \Sigma V^\dagger$ via SVD decomposition.
//! - [`SaturableAbsorber`]: Optical saturable absorption nonlinearity for all-optical activation.

use crate::error::{Error, Result};
use crate::nn::Module;
use crate::tensor::Tensor;

/// A 2x2 Mach-Zehnder Interferometer (MZI) optical component.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MachZehnderInterferometer {
    pub theta: f64,
    pub phi: f64,
}

impl MachZehnderInterferometer {
    pub fn new(theta: f64, phi: f64) -> Self {
        Self { theta, phi }
    }

    /// Evaluates optical transfer on 2D complex electric field $(E_1, E_2)$:
    /// Returns real and imaginary components of the output fields.
    pub fn forward_field(
        &self,
        e1_real: f64,
        e1_imag: f64,
        e2_real: f64,
        e2_imag: f64,
    ) -> (f64, f64, f64, f64) {
        let cos_th = self.theta.cos();
        let sin_th = self.theta.sin();
        let cos_ph = self.phi.cos();
        let sin_ph = self.phi.sin();

        // T11 = e^{i phi} * cos(theta) = (cos_ph + i sin_ph) * cos_th
        let t11_r = cos_ph * cos_th;
        let t11_i = sin_ph * cos_th;

        // T12 = -e^{i phi} * sin(theta) = -(cos_ph + i sin_ph) * sin_th
        let t12_r = -cos_ph * sin_th;
        let t12_i = -sin_ph * sin_th;

        // T21 = sin(theta)
        let t21_r = sin_th;
        let t21_i = 0.0;

        // T22 = cos(theta)
        let t22_r = cos_th;
        let t22_i = 0.0;

        // Out1 = T11 * E1 + T12 * E2
        let out1_r = (t11_r * e1_real - t11_i * e1_imag) + (t12_r * e2_real - t12_i * e2_imag);
        let out1_i = (t11_r * e1_imag + t11_i * e1_real) + (t12_r * e2_imag + t12_i * e2_real);

        // Out2 = T21 * E1 + T22 * E2
        let out2_r = (t21_r * e1_real - t21_i * e1_imag) + (t22_r * e2_real - t22_i * e2_imag);
        let out2_i = (t21_r * e1_imag + t21_i * e1_real) + (t22_r * e2_imag + t22_i * e2_real);

        (out1_r, out1_i, out2_r, out2_i)
    }
}

/// Clements-style rectangular MZI mesh realizing unitary operations $U(N)$.
#[derive(Debug, Clone)]
pub struct PhotonicMesh {
    pub dim: usize,
    pub mzis: Vec<MachZehnderInterferometer>,
}

impl PhotonicMesh {
    pub fn new(dim: usize) -> Result<Self> {
        if dim == 0 {
            return Err(Error::Msg("PhotonicMesh dim must be >= 1".to_string()));
        }
        let num_mzis = dim
            .checked_mul(dim - 1)
            .map(|p| p / 2)
            .ok_or_else(|| Error::Msg(format!("PhotonicMesh dim {dim} is too large")))?;
        let mzis = crate::layout::try_filled(num_mzis, MachZehnderInterferometer::new(0.0, 0.0))?;
        Ok(Self { dim, mzis })
    }

    /// Propagates $N$-dimensional optical field through the MZI mesh.
    pub fn forward_optical(
        &self,
        input_real: &[f64],
        input_imag: &[f64],
    ) -> Result<(Vec<f64>, Vec<f64>)> {
        if input_real.len() != self.dim || input_imag.len() != self.dim {
            return Err(Error::Msg(format!(
                "Optical field dimension mismatch: expected {}",
                self.dim
            )));
        }

        let mut out_r = input_real.to_vec();
        let mut out_i = input_imag.to_vec();

        let mut mzi_idx = 0;
        for i in 0..self.dim {
            for j in 0..i {
                if mzi_idx < self.mzis.len() {
                    let mzi = self.mzis[mzi_idx];
                    let (o1_r, o1_i, o2_r, o2_i) =
                        mzi.forward_field(out_r[j], out_i[j], out_r[j + 1], out_i[j + 1]);
                    out_r[j] = o1_r;
                    out_i[j] = o1_i;
                    out_r[j + 1] = o2_r;
                    out_i[j + 1] = o2_i;
                    mzi_idx += 1;
                }
            }
        }

        Ok((out_r, out_i))
    }
}

/// Optical Saturable Absorber non-linear activation:
/// $$I_{\text{out}} = \frac{I_{\text{in}}}{1 + I_{\text{in}} / I_{\text{sat}}}$$
#[derive(Debug, Clone, Copy)]
pub struct SaturableAbsorber {
    pub i_sat: f64,
}

impl Default for SaturableAbsorber {
    fn default() -> Self {
        Self { i_sat: 1.0 }
    }
}

impl SaturableAbsorber {
    pub fn new(i_sat: f64) -> Self {
        Self {
            i_sat: i_sat.max(1e-5),
        }
    }

    pub fn forward(&self, intensity: &Tensor) -> Result<Tensor> {
        let denom = intensity.mul_scalar(1.0 / self.i_sat)?.add_scalar(1.0)?;
        intensity.div(&denom)
    }
}

/// SVD-based Programmable Photonic Linear Layer:
/// Implements $W = U \Sigma V^\dagger$ through unitary meshes and diagonal attenuation.
pub struct PhotonicLinear {
    pub in_features: usize,
    pub out_features: usize,
    pub u_mesh: PhotonicMesh,
    pub v_mesh: PhotonicMesh,
    pub singular_values: Tensor,
}

impl PhotonicLinear {
    pub fn new(in_features: usize, out_features: usize) -> Result<Self> {
        let u_mesh = PhotonicMesh::new(out_features)?;
        let v_mesh = PhotonicMesh::new(in_features)?;
        let min_dim = in_features.min(out_features);
        let singular_values = Tensor::ones(
            &[min_dim],
            crate::dtype::DType::F32,
            &crate::device::Device::Cpu,
        )?;

        Ok(Self {
            in_features,
            out_features,
            u_mesh,
            v_mesh,
            singular_values,
        })
    }

    /// Evaluates forward pass mapping optical input intensities `[batch, in_features]`.
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        if x.ndim() != 2 {
            return Err(Error::Msg(format!(
                "PhotonicLinear expects [batch, in_features], got {:?}",
                x.shape()
            )));
        }
        let (b, in_dim) = (x.shape()[0], x.shape()[1]);
        if in_dim != self.in_features {
            return Err(Error::Msg(format!(
                "Input feature mismatch: expected {}",
                self.in_features
            )));
        }

        let x_data = x.to_vec::<f32>()?;
        let min_dim = self.in_features.min(self.out_features);
        let s_vals = self.singular_values.to_vec::<f32>()?;
        if s_vals.len() < min_dim {
            return Err(Error::Msg(format!(
                "PhotonicLinear needs {min_dim} singular values, has {}",
                s_vals.len()
            )));
        }

        let mut out_data = vec![0.0f32; b * self.out_features];

        for batch_idx in 0..b {
            let row = &x_data[batch_idx * in_dim..(batch_idx + 1) * in_dim];
            let r_f64: Vec<f64> = row.iter().map(|&v| v as f64).collect();
            let i_f64 = vec![0.0f64; in_dim];

            // 1. Pass through V^\dagger mesh
            let (v_r, v_i) = self.v_mesh.forward_optical(&r_f64, &i_f64)?;

            // 2. Optical diagonal attenuation / amplification (singular values Sigma)
            let mut u_in_r = vec![0.0f64; self.out_features];
            let mut u_in_i = vec![0.0f64; self.out_features];
            for i in 0..min_dim {
                let sigma = s_vals[i] as f64;
                u_in_r[i] = v_r[i] * sigma;
                u_in_i[i] = v_i[i] * sigma;
            }

            // 3. Pass through U mesh
            let (u_r, u_i) = self.u_mesh.forward_optical(&u_in_r, &u_in_i)?;

            // 4. Optical detection (intensity / energy): I = |E|^2
            for i in 0..self.out_features {
                let intensity = (u_r[i] * u_r[i] + u_i[i] * u_i[i]) as f32;
                out_data[batch_idx * self.out_features + i] = intensity;
            }
        }

        Tensor::from_vec(out_data, &[b, self.out_features])
    }
}

impl Module for PhotonicLinear {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        self.forward(input)
    }

    fn parameters(&self) -> Vec<Tensor> {
        vec![self.singular_values.clone()]
    }
}
