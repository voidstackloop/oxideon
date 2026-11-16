//! Residual Vector Quantizer (RVQ) for continuous audio tokenization (SoundStream / EnCodec / DAC).
//!
//! Quantizes continuous audio embeddings into multi-codebook discrete token streams
//! using straight-through hierarchical residual approximation:
//!
//! $$r_0 = x, \quad r_{k+1} = r_k - \mathcal{C}_k(r_k), \quad \hat{x} = \sum_{k=1}^K \mathcal{C}_k(r_k)$$

use crate::error::{Error, Result};
use crate::tensor::Tensor;

/// Configuration for Residual Vector Quantizer.
#[derive(Debug, Clone)]
pub struct RVQConfig {
    pub num_codebooks: usize,
    pub codebook_size: usize,
    pub dim: usize,
    pub commitment_cost: f32,
}

impl RVQConfig {
    pub fn new(num_codebooks: usize, codebook_size: usize, dim: usize) -> Self {
        Self {
            num_codebooks,
            codebook_size,
            dim,
            commitment_cost: 0.25,
        }
    }
}

/// Output of RVQ encoding step.
#[derive(Debug, Clone)]
pub struct RVQOutput {
    /// Reconstructed continuous latent tensor of shape `[B, T, D]`.
    pub quantized: Tensor,
    /// Discrete code indices for each stage: `[B, num_codebooks, T]`.
    pub codes: Vec<Vec<Vec<usize>>>,
    /// Commitment loss enforcing latents to track codebook centroids.
    pub commitment_loss: f32,
}

/// Single vector quantization codebook stage.
#[derive(Debug, Clone)]
pub struct VQCodebookStage {
    pub codebook_size: usize,
    pub dim: usize,
    /// Centroid vectors of size `codebook_size * dim`.
    pub embeddings: Vec<f32>,
}

impl VQCodebookStage {
    pub fn new(codebook_size: usize, dim: usize) -> Result<Self> {
        if codebook_size == 0 || dim == 0 {
            return Err(Error::Msg(format!(
                "VQCodebookStage needs positive codebook_size and dim, got {codebook_size}/{dim}"
            )));
        }
        let t = Tensor::randn(&[codebook_size, dim])?;
        let embeddings = t.to_vec::<f32>()?;
        Ok(Self {
            codebook_size,
            dim,
            embeddings,
        })
    }

    /// Finds nearest codebook centroid for input vector of size `dim`.
    pub fn quantize_vec(&self, x: &[f32]) -> Result<(usize, Vec<f32>)> {
        // All fields are pub: validate the consistency the indexing below
        // assumes (an empty codebook skipped the loop, then sliced OOB).
        if self.codebook_size == 0 || self.dim == 0 {
            return Err(Error::Msg(format!(
                "VQCodebookStage needs positive codebook_size and dim, got {}/{}",
                self.codebook_size, self.dim
            )));
        }
        let need = self.codebook_size.checked_mul(self.dim).ok_or_else(|| {
            Error::Msg(format!(
                "VQCodebookStage embedding size overflows ({} * {})",
                self.codebook_size, self.dim
            ))
        })?;
        if self.embeddings.len() != need {
            return Err(Error::Msg(format!(
                "VQCodebookStage has {} embeddings, need {}",
                self.embeddings.len(),
                need
            )));
        }
        let mut min_dist = f32::INFINITY;
        let mut best_idx = 0;

        for c in 0..self.codebook_size {
            let offset = c * self.dim;
            let mut dist = 0.0f32;
            for (i, &val) in x.iter().enumerate().take(self.dim) {
                let diff = val - self.embeddings[offset + i];
                dist += diff * diff;
            }
            if dist < min_dist {
                min_dist = dist;
                best_idx = c;
            }
        }

        let best_offset = best_idx * self.dim;
        let centroid = self.embeddings[best_offset..best_offset + self.dim].to_vec();
        Ok((best_idx, centroid))
    }
}

/// Residual Vector Quantizer (RVQ).
#[derive(Debug, Clone)]
pub struct ResidualVectorQuantizer {
    pub config: RVQConfig,
    pub stages: Vec<VQCodebookStage>,
}

impl ResidualVectorQuantizer {
    pub fn new(config: RVQConfig) -> Result<Self> {
        crate::limits::check_repeats("num_codebooks", config.num_codebooks)?;
        let mut stages = Vec::with_capacity(config.num_codebooks);
        for _ in 0..config.num_codebooks {
            stages.push(VQCodebookStage::new(config.codebook_size, config.dim)?);
        }
        Ok(Self { config, stages })
    }

    /// Self-consistency guard: `stages`/`config` are public and hand-tamperable,
    /// and both `quantize` and `decode_codes` index into them unchecked.
    fn check_stages(&self) -> Result<()> {
        if self.stages.len() != self.config.num_codebooks {
            return Err(Error::Msg(format!(
                "ResidualVectorQuantizer: {} stages disagree with num_codebooks {}",
                self.stages.len(),
                self.config.num_codebooks
            )));
        }
        for (i, st) in self.stages.iter().enumerate() {
            if st.dim != self.config.dim {
                return Err(Error::Msg(format!(
                    "ResidualVectorQuantizer: stage {i} dim {} disagrees with config dim {}",
                    st.dim, self.config.dim
                )));
            }
            if st.codebook_size == 0 {
                return Err(Error::Msg(format!(
                    "ResidualVectorQuantizer: stage {i} has an empty codebook"
                )));
            }
            let need = st.codebook_size.checked_mul(st.dim).ok_or_else(|| {
                Error::Msg(format!(
                    "ResidualVectorQuantizer: stage {i} embedding size overflows"
                ))
            })?;
            if st.embeddings.len() != need {
                return Err(Error::Msg(format!(
                    "ResidualVectorQuantizer: stage {i} has {} embeddings, need {}",
                    st.embeddings.len(),
                    need
                )));
            }
        }
        Ok(())
    }

    /// Quantizes input latents of shape `[batch_size, seq_len, dim]`.
    pub fn quantize(&self, latents: &Tensor) -> Result<RVQOutput> {
        self.check_stages()?;
        let shape = latents.shape();
        if shape.len() != 3 {
            return Err(Error::Msg(
                "RVQ expects 3D latents [batch_size, seq_len, dim]".into(),
            ));
        }

        let batch_size = shape[0];
        let seq_len = shape[1];
        let d = shape[2];

        if d != self.config.dim {
            return Err(Error::ShapeMismatch {
                op: "ResidualVectorQuantizer",
                expected: vec![batch_size, seq_len, self.config.dim],
                got: shape.to_vec(),
            });
        }

        let latents_vec = latents.to_vec::<f32>()?;
        let mut all_codes =
            vec![vec![vec![0usize; seq_len]; self.config.num_codebooks]; batch_size];
        let mut reconstructed_vec = vec![0.0f32; batch_size * seq_len * d];
        let mut total_commitment_loss = 0.0f32;

        for b in 0..batch_size {
            for t in 0..seq_len {
                let offset = (b * seq_len + t) * d;
                let orig_x = &latents_vec[offset..offset + d];
                let mut residual = orig_x.to_vec();
                let mut accum_q = vec![0.0f32; d];

                for (stage_idx, stage) in self.stages.iter().enumerate() {
                    let (idx, centroid) = stage.quantize_vec(&residual)?;
                    all_codes[b][stage_idx][t] = idx;

                    for i in 0..d {
                        accum_q[i] += centroid[i];
                        residual[i] -= centroid[i];
                    }
                }

                // Compute commitment loss: ||orig_x - accum_q||^2
                let mut token_loss = 0.0f32;
                for i in 0..d {
                    let diff = orig_x[i] - accum_q[i];
                    token_loss += diff * diff;
                    reconstructed_vec[offset + i] = accum_q[i];
                }
                total_commitment_loss += token_loss;
            }
        }

        let n_elements = (batch_size * seq_len * d) as f32;
        let commitment_loss =
            self.config.commitment_cost * (total_commitment_loss / n_elements.max(1.0));

        let quantized =
            Tensor::from_slice(&reconstructed_vec, shape)?.to_device(&latents.device())?;

        Ok(RVQOutput {
            quantized,
            codes: all_codes,
            commitment_loss,
        })
    }

    /// Reconstructs continuous latent embeddings from discrete code sequences of shape `[num_codebooks, seq_len]`.
    pub fn decode_codes(&self, codes: &[Vec<usize>]) -> Result<Tensor> {
        self.check_stages()?;
        if codes.len() != self.config.num_codebooks {
            return Err(Error::Msg(format!(
                "Expected {} codebooks, got {}",
                self.config.num_codebooks,
                codes.len()
            )));
        }
        let Some(first) = codes.first() else {
            return Err(Error::Msg(
                "decode_codes needs at least one codebook to infer sequence length".into(),
            ));
        };

        let seq_len = first.len();
        let d = self.config.dim;
        let mut out_data = vec![0.0f32; seq_len * d];

        for (stage_idx, stage) in self.stages.iter().enumerate() {
            let stage_codes = &codes[stage_idx];
            if stage_codes.len() != seq_len {
                return Err(Error::Msg(
                    "Inconsistent sequence lengths across codebooks".into(),
                ));
            }

            for (t, &c_idx) in stage_codes.iter().enumerate() {
                if c_idx >= stage.codebook_size {
                    return Err(Error::Msg(format!(
                        "Code index {c_idx} out of bounds for codebook size {}",
                        stage.codebook_size
                    )));
                }

                let c_offset = c_idx * d;
                let t_offset = t * d;
                for i in 0..d {
                    out_data[t_offset + i] += stage.embeddings[c_offset + i];
                }
            }
        }

        Tensor::from_slice(&out_data, &[1, seq_len, d])
    }
}
