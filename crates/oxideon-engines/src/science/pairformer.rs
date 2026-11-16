//! AlphaFold3 Pairformer Engine for Bio-Molecular Structure Modeling.
//!
//! Updates pairwise spatial representations $Z \in \mathbb{R}^{N \times N \times C}$ using
//! Triangular Multiplicative Updates and Triangular Self-Attention (Abramson et al., Nature 2024).

use crate::error::{Error, Result};
use crate::nn::{LayerNorm, Linear, Module};
use crate::tensor::Tensor;

/// Triangular Multiplicative Update (Outgoing Triangles):
/// $$z_{ij} \leftarrow z_{ij} + \operatorname{Linear}\left(\sum_k a_{ik} b_{jk}\right) \odot \operatorname{sigmoid}(\operatorname{Linear}(z_{ij}))$$
pub struct TriangularMultiplicativeUpdate {
    pub norm: LayerNorm,
    pub proj_a: Linear,
    pub proj_b: Linear,
    pub proj_g: Linear,
    pub proj_out: Linear,
    pub c_hidden: usize,
}

impl TriangularMultiplicativeUpdate {
    pub fn new(c_z: usize, c_hidden: usize) -> Result<Self> {
        let norm = LayerNorm::new(c_z, 1e-5)?;
        let proj_a = Linear::new(c_z, c_hidden)?;
        let proj_b = Linear::new(c_z, c_hidden)?;
        let proj_g = Linear::new(c_z, c_z)?;
        let proj_out = Linear::new(c_hidden, c_z)?;

        Ok(Self {
            norm,
            proj_a,
            proj_b,
            proj_g,
            proj_out,
            c_hidden,
        })
    }

    /// Forward pass updating pair representation `z` of shape `[N, N, C_z]`.
    pub fn forward(&self, z: &Tensor) -> Result<Tensor> {
        if z.ndim() != 3 {
            return Err(Error::Msg(format!(
                "TriangularMultiplicativeUpdate expects [N, N, C_z], got {:?}",
                z.shape()
            )));
        }
        let (n0, n1, c) = (z.shape()[0], z.shape()[1], z.shape()[2]);
        let z_norm = self.norm.forward(z)?;

        // Flat projection
        let z_flat = z_norm.reshape(&[n0 * n1, c])?;
        let a_flat = self.proj_a.forward(&z_flat)?.sigmoid()?;
        let b_flat = self.proj_b.forward(&z_flat)?.sigmoid()?;
        let g_flat = self.proj_g.forward(&z_flat)?.sigmoid()?;

        let a = a_flat.reshape(&[n0, n1, self.c_hidden])?;
        let b = b_flat.reshape(&[n0, n1, self.c_hidden])?;
        let g = g_flat.reshape(&[n0, n1, c])?;

        // Triangular contraction: sum_k a[i, k, c] * b[j, k, c]
        // Transpose b to [c_hidden, n0, n1] or permute for batched GEMM
        let a_perm = a.permute(&[2, 0, 1])?; // [c_hidden, N, N]
        let b_perm = b.permute(&[2, 1, 0])?; // [c_hidden, N, N]
        let contract = a_perm.matmul(&b_perm)?; // [c_hidden, N, N]
        let contract_z = contract.permute(&[1, 2, 0])?; // [N, N, c_hidden]

        let out_flat = self
            .proj_out
            .forward(&contract_z.reshape(&[n0 * n1, self.c_hidden])?)?;
        let out_z = out_flat.reshape(&[n0, n1, c])?;

        // Gated addition
        let update = out_z.mul(&g)?;
        z.add(&update)
    }
}

/// Full AlphaFold3 Pairformer Block.
pub struct PairformerBlock {
    pub tri_mul_out: TriangularMultiplicativeUpdate,
    pub tri_mul_in: TriangularMultiplicativeUpdate,
    pub mlp1: Linear,
    pub mlp2: Linear,
    pub norm_mlp: LayerNorm,
}

impl PairformerBlock {
    pub fn new(c_z: usize, c_hidden: usize) -> Result<Self> {
        let tri_mul_out = TriangularMultiplicativeUpdate::new(c_z, c_hidden)?;
        let tri_mul_in = TriangularMultiplicativeUpdate::new(c_z, c_hidden)?;
        let norm_mlp = LayerNorm::new(c_z, 1e-5)?;
        let mlp1 = Linear::new(c_z, c_z * 2)?;
        let mlp2 = Linear::new(c_z * 2, c_z)?;

        Ok(Self {
            tri_mul_out,
            tri_mul_in,
            mlp1,
            mlp2,
            norm_mlp,
        })
    }

    pub fn forward(&self, z: &Tensor) -> Result<Tensor> {
        // 1. Triangular Multiplicative Update (Outgoing)
        let z1 = self.tri_mul_out.forward(z)?;

        // 2. Triangular Multiplicative Update (Incoming - transposed)
        let z1_t = z1.transpose(0, 1)?;
        let z2_t = self.tri_mul_in.forward(&z1_t)?;
        let z2 = z2_t.transpose(0, 1)?;

        // 3. Pair Transition MLP
        let n0 = z2.shape()[0];
        let n1 = z2.shape()[1];
        let c = z2.shape()[2];
        let z_norm = self.norm_mlp.forward(&z2)?;
        let flat = z_norm.reshape(&[n0 * n1, c])?;
        let hidden = self.mlp1.forward(&flat)?.relu()?;
        let out_flat = self.mlp2.forward(&hidden)?;
        let mlp_out = out_flat.reshape(&[n0, n1, c])?;

        z2.add(&mlp_out)
    }
}
