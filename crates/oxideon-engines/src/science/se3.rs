//! $SE(3)$-Equivariant Graph Neural Network Layer for 3D Molecular Geometries.
//!
//! Guarantees exact rotational ($SO(3)$) and translational ($T(3)$) equivariance for
//! 3D atomic coordinates: $f(R X + \mathbf{t}) = R f(X) + \mathbf{t}$.

use crate::error::{Error, Result};
use crate::nn::{Linear, Module};
use crate::tensor::Tensor;

/// An $SE(3)$-Equivariant layer updating both scalar features and 3D coordinates.
pub struct SE3EquivariantLayer {
    pub feature_dim: usize,
    pub phi_coord: Linear,
    pub phi_feat: Linear,
}

impl SE3EquivariantLayer {
    pub fn new(feature_dim: usize) -> Result<Self> {
        // Message function takes [d_ij^2, h_i, h_j] -> size: 1 + 2 * feature_dim
        let msg_in_dim = 1 + 2 * feature_dim;
        let phi_coord = Linear::new(msg_in_dim, 1)?;
        let phi_feat = Linear::new(msg_in_dim, feature_dim)?;

        Ok(Self {
            feature_dim,
            phi_coord,
            phi_feat,
        })
    }

    /// Evaluates $SE(3)$-equivariant forward step over:
    /// - `coords`: `[N, 3]` 3D positions of atoms
    /// - `features`: `[N, D]` atomic representation features
    ///
    /// Returns `(updated_coords, updated_features)`.
    pub fn forward(&self, coords: &Tensor, features: &Tensor) -> Result<(Tensor, Tensor)> {
        if coords.ndim() != 2 || coords.shape()[1] != 3 {
            return Err(Error::Msg(format!(
                "SE3EquivariantLayer coords must be [N, 3], got {:?}",
                coords.shape()
            )));
        }
        let n = coords.shape()[0];
        if features.ndim() != 2 || features.shape()[0] != n {
            return Err(Error::Msg(format!(
                "SE3EquivariantLayer features must be [{n}, D], got {:?}",
                features.shape()
            )));
        }
        let d = features.shape()[1];
        if d != self.feature_dim {
            return Err(Error::Msg(format!(
                "feature_dim mismatch: expected {}, got {d}",
                self.feature_dim
            )));
        }

        let coords_vec = coords.to_vec::<f32>()?;
        let feats_vec = features.to_vec::<f32>()?;

        let mut coord_delta = vec![0.0f32; n * 3];
        let mut feat_delta = vec![0.0f32; n * d];

        let msg_dim = 1 + 2 * d;
        let mut pair_msg = vec![0.0f32; msg_dim];

        for i in 0..n {
            let xi = [
                coords_vec[i * 3],
                coords_vec[i * 3 + 1],
                coords_vec[i * 3 + 2],
            ];
            let hi = &feats_vec[i * d..(i + 1) * d];

            for j in 0..n {
                if i == j {
                    continue;
                }
                let xj = [
                    coords_vec[j * 3],
                    coords_vec[j * 3 + 1],
                    coords_vec[j * 3 + 2],
                ];
                let hj = &feats_vec[j * d..(j + 1) * d];

                // Relative vector r_ij = x_i - x_j
                let rx = xi[0] - xj[0];
                let ry = xi[1] - xj[1];
                let rz = xi[2] - xj[2];
                let dist_sq = rx * rx + ry * ry + rz * rz;

                // Form invariant message vector [dist_sq, h_i, h_j]
                pair_msg[0] = dist_sq;
                pair_msg[1..1 + d].copy_from_slice(hi);
                pair_msg[1 + d..1 + 2 * d].copy_from_slice(hj);

                let msg_tensor = Tensor::from_vec(pair_msg.clone(), &[1, msg_dim])?
                    .to_device(&coords.device())?
                    .to_dtype(coords.dtype())?;

                // Coordinate scale weight w_coord
                let w_coord = self.phi_coord.forward(&msg_tensor)?.item::<f32>()?;

                // Feature message m_feat
                let m_feat = self.phi_feat.forward(&msg_tensor)?.to_vec::<f32>()?;

                let scale = 1.0 / (n.max(2) - 1) as f32;
                coord_delta[i * 3] += rx * w_coord * scale;
                coord_delta[i * 3 + 1] += ry * w_coord * scale;
                coord_delta[i * 3 + 2] += rz * w_coord * scale;

                for k in 0..d {
                    feat_delta[i * d + k] += m_feat[k] * scale;
                }
            }
        }

        let delta_c = Tensor::from_vec(coord_delta, &[n, 3])?
            .to_device(&coords.device())?
            .to_dtype(coords.dtype())?;
        let delta_f = Tensor::from_vec(feat_delta, &[n, d])?
            .to_device(&features.device())?
            .to_dtype(features.dtype())?;

        let new_coords = coords.add(&delta_c)?;
        let new_features = features.add(&delta_f)?;

        Ok((new_coords, new_features))
    }
}
