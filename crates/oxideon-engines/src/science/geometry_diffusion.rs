//! Molecular 3D Geometry Diffusion Engine (AlphaFold3 / RFdiffusion standard).
//!
//! Generates valid 3D protein conformations and ligand point clouds by denoising
//! 3D atomic coordinates in $\mathbb{R}^{N \times 3}$ with center-of-mass invariance.

use crate::error::Result;
use crate::science::se3::SE3EquivariantLayer;
use crate::tensor::Tensor;

/// 3D Geometry Diffusion Processor.
#[derive(Debug, Clone, Default)]
pub struct MolecularGeometryDiffusion;

impl MolecularGeometryDiffusion {
    /// Removes the center-of-mass translation from 3D coordinates `[N, 3]`:
    /// $$\mathbf{x}_i \leftarrow \mathbf{x}_i - \frac{1}{N} \sum_{j=1}^N \mathbf{x}_j$$
    pub fn remove_center_of_mass(coords: &Tensor) -> Result<Tensor> {
        let mean = coords.mean_axis(0, true)?; // [1, 3]
        coords.sub(&mean)
    }

    /// Predicts coordinate drift vector $\mathbf{v} \in \mathbb{R}^{N \times 3}$ using
    /// an $SE(3)$-equivariant layer and preserves zero center-of-mass.
    pub fn predict_drift(
        layer: &SE3EquivariantLayer,
        coords: &Tensor,
        features: &Tensor,
    ) -> Result<Tensor> {
        let centered = Self::remove_center_of_mass(coords)?;
        let (updated_coords, _) = layer.forward(&centered, features)?;
        let drift = updated_coords.sub(&centered)?;
        Self::remove_center_of_mass(&drift)
    }

    /// Runs iterative reverse coordinate diffusion from standard Gaussian noise.
    pub fn sample_conformation(
        layer: &SE3EquivariantLayer,
        init_coords: &Tensor,
        features: &Tensor,
        num_steps: usize,
    ) -> Result<Tensor> {
        let mut x = Self::remove_center_of_mass(init_coords)?;
        let dt = 1.0 / (num_steps.max(1) as f64);

        for _step in 0..num_steps {
            let drift = Self::predict_drift(layer, &x, features)?;
            x = x.add(&drift.mul_scalar(dt)?)?;
            x = Self::remove_center_of_mass(&x)?;
        }

        Ok(x)
    }
}
