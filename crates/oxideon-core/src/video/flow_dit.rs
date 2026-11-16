//! Rectified Flow Spatio-Temporal Diffusion Transformer (`RectifiedFlowVideoDiT`, Sora / SD3 standard).
//!
//! Employs continuous-time Flow Matching with linear probability paths:
//! $$x_t = (1 - t) x_0 + t x_1, \quad v_t = \frac{d x_t}{dt} = x_1 - x_0$$
//! The DiT model predicts $v_\theta(x_t, t, c)$ modulated by AdaLN-Zero and conditioned on text representations.

use crate::error::{Error, Result};
use crate::nn::{LayerNorm, Linear, Module, MultiHeadAttention};
use crate::tensor::Tensor;
use crate::video::rope3d::RotaryEmbedding3D;

/// Configuration for Rectified Flow Video Diffusion Transformer.
#[derive(Debug, Clone)]
pub struct VideoDiTConfig {
    pub in_channels: usize,
    pub hidden_dim: usize,
    pub num_heads: usize,
    pub patch_size: (usize, usize, usize), // (p_t, p_h, p_w)
    pub text_dim: usize,
}

impl Default for VideoDiTConfig {
    fn default() -> Self {
        Self {
            in_channels: 4,
            hidden_dim: 64,
            num_heads: 4,
            patch_size: (1, 2, 2),
            text_dim: 32,
        }
    }
}

/// Continuous sinusoidal timestep embedding for Flow Matching time $t \in [0, 1]$.
pub fn continuous_timestep_embedding(t: &Tensor, dim: usize) -> Result<Tensor> {
    let half = dim / 2;
    let max_period = 10000.0f32;
    let t_vec = t.to_vec::<f32>()?;
    let b = t_vec.len();
    let n = crate::limits::check_elements("continuous_timestep_embedding", &[b, dim])?;
    let mut out = vec![0.0f32; n];

    for (row, &val) in t_vec.iter().enumerate() {
        let tf = val * 1000.0;
        for i in 0..half {
            let freq = (-max_period.ln() * (i as f32) / (half.max(1) as f32)).exp();
            let angle = tf * freq;
            out[row * dim + i] = angle.cos();
            out[row * dim + half + i] = angle.sin();
        }
    }

    Tensor::from_vec(out, &[b, dim])?
        .to_device(&t.device())?
        .to_dtype(t.dtype())
}

/// A DiT transformer block with adaptive normalization (AdaLN), 3D RoPE self-attention,
/// and cross-attention conditioning.
pub struct VideoDiTBlock {
    pub norm1: LayerNorm,
    pub self_attn: MultiHeadAttention,
    pub norm_cross: LayerNorm,
    pub cross_attn: MultiHeadAttention,
    pub text_proj: Linear,
    pub norm2: LayerNorm,
    pub mlp1: Linear,
    pub mlp2: Linear,
    pub adaln_linear: Linear,
    pub rope3d: RotaryEmbedding3D,
}

impl VideoDiTBlock {
    pub fn new(hidden_dim: usize, num_heads: usize, text_dim: usize) -> Result<Self> {
        let norm1 = LayerNorm::new(hidden_dim, 1e-5)?;
        let self_attn = MultiHeadAttention::new(hidden_dim, num_heads)?;
        let norm_cross = LayerNorm::new(hidden_dim, 1e-5)?;
        let cross_attn = MultiHeadAttention::new(hidden_dim, num_heads)?;
        let text_proj = Linear::new(text_dim, hidden_dim)?;
        let norm2 = LayerNorm::new(hidden_dim, 1e-5)?;
        let mlp1 = Linear::new(hidden_dim, hidden_dim * 4)?;
        let mlp2 = Linear::new(hidden_dim * 4, hidden_dim)?;
        // Produces 6 modulation scales/shifts: (gamma1, beta1, alpha1, gamma2, beta2, alpha2)
        let adaln_linear = Linear::new(hidden_dim, hidden_dim * 6)?;
        let head_dim = hidden_dim / num_heads;
        let rope3d = RotaryEmbedding3D::new(head_dim, 10000.0)?;

        Ok(Self {
            norm1,
            self_attn,
            norm_cross,
            cross_attn,
            text_proj,
            norm2,
            mlp1,
            mlp2,
            adaln_linear,
            rope3d,
        })
    }

    pub fn forward(
        &self,
        x: &Tensor,
        t_emb: &Tensor,
        text_cond: &Tensor,
        grid_size: (usize, usize, usize),
    ) -> Result<Tensor> {
        if x.ndim() != 3 {
            return Err(Error::Msg(format!(
                "VideoDiTBlock expects [B, T, H] input, got {:?}",
                x.shape()
            )));
        }
        // AdaLN modulation parameters, reshaped [B, H] -> [B, 1, H] so they
        // broadcast over tokens (without this, batch>1 fails or misaligns).
        let mod_params = self.adaln_linear.forward(t_emb)?.silu()?;
        let hidden = x.shape()[2];
        if mod_params.ndim() != 2 || mod_params.shape()[1] != hidden * 6 {
            return Err(Error::Msg(format!(
                "VideoDiTBlock AdaLN params must be [B, {}], got {:?}",
                hidden * 6,
                mod_params.shape()
            )));
        }
        let chunks: Vec<Tensor> = (0..6)
            .map(|i| mod_params.narrow(1, i * hidden, hidden)?.unsqueeze(1))
            .collect::<Result<Vec<_>>>()?;

        let (gamma1, beta1, alpha1, gamma2, beta2, alpha2) = (
            &chunks[0], &chunks[1], &chunks[2], &chunks[3], &chunks[4], &chunks[5],
        );

        // 1. Modulated Self-Attention with 3D RoPE on q/k (the rope3d
        // tables were built but never applied, so tokens had no position).
        let h_norm = self.norm1.forward(x)?;
        let h_mod = h_norm.mul(&gamma1.add_scalar(1.0)?)?.add(beta1)?;
        let attn_out = {
            let b = h_mod.shape()[0];
            let l = h_mod.shape()[1];
            let h = self.self_attn.num_heads;
            let d = self.self_attn.head_dim;
            let (gt, gh, gw) = grid_size;
            let q = self
                .self_attn
                .q_proj
                .forward(&h_mod)?
                .reshape(&[b, l, h, d])?;
            let k = self
                .self_attn
                .k_proj
                .forward(&h_mod)?
                .reshape(&[b, l, h, d])?;
            let v = self
                .self_attn
                .v_proj
                .forward(&h_mod)?
                .reshape(&[b, l, h, d])?
                .transpose(1, 2)?;
            let q = self.rope3d.forward(&q, gt, gh, gw)?.transpose(1, 2)?;
            let k = self.rope3d.forward(&k, gt, gh, gw)?.transpose(1, 2)?;
            let scale = 1.0 / (d as f64).sqrt();
            let attn = q.matmul(&k.t()?)?.mul_scalar(scale)?.softmax(-1)?;
            let ctx =
                attn.matmul(&v)?
                    .transpose(1, 2)?
                    .reshape(&[b, l, self.self_attn.embed_dim])?;
            self.self_attn.out_proj.forward(&ctx)?
        };
        let x1 = x.add(&attn_out.mul(alpha1)?)?;

        // 2. Cross-Attention to text condition
        let h_cross = self.norm_cross.forward(&x1)?;
        let text_projected = self.text_proj.forward(text_cond)?;
        let cross_out =
            self.cross_attn
                .forward_qkv(&h_cross, &text_projected, &text_projected, false)?;
        let x2 = x1.add(&cross_out)?;

        // 3. Modulated MLP
        let m_norm = self.norm2.forward(&x2)?;
        let m_mod = m_norm.mul(&gamma2.add_scalar(1.0)?)?.add(beta2)?;
        let mlp_out = self.mlp2.forward(&self.mlp1.forward(&m_mod)?.gelu()?)?;
        x2.add(&mlp_out.mul(alpha2)?)
    }
}

/// Full Spatio-Temporal Rectified Flow Diffusion Transformer for Video.
pub struct RectifiedFlowVideoDiT {
    pub config: VideoDiTConfig,
    pub patch_proj: Linear,
    pub time_mlp1: Linear,
    pub time_mlp2: Linear,
    pub block: VideoDiTBlock,
    pub final_norm: LayerNorm,
    pub final_proj: Linear,
}

impl RectifiedFlowVideoDiT {
    pub fn new(config: VideoDiTConfig) -> Result<Self> {
        let (pt, ph, pw) = config.patch_size;
        let patch_dim = config
            .in_channels
            .checked_mul(pt)
            .and_then(|v| v.checked_mul(ph))
            .and_then(|v| v.checked_mul(pw))
            .ok_or_else(|| {
                Error::Msg(format!(
                    "RectifiedFlowVideoDiT: patch dim (in_channels {} * patch {pt}x{ph}x{pw}) overflows",
                    config.in_channels
                ))
            })?;
        let patch_proj = Linear::new(patch_dim, config.hidden_dim)?;

        let time_mlp1 = Linear::new(config.hidden_dim, config.hidden_dim)?;
        let time_mlp2 = Linear::new(config.hidden_dim, config.hidden_dim)?;

        let block = VideoDiTBlock::new(config.hidden_dim, config.num_heads, config.text_dim)?;
        let final_norm = LayerNorm::new(config.hidden_dim, 1e-5)?;
        let final_proj = Linear::new(config.hidden_dim, patch_dim)?;

        Ok(Self {
            config,
            patch_proj,
            time_mlp1,
            time_mlp2,
            block,
            final_norm,
            final_proj,
        })
    }

    /// Patchifies 5D video `[B, C, T, H, W]` into 3D sequence tokens `[B, num_patches, patch_dim]`.
    fn patchify(&self, x: &Tensor) -> Result<(Tensor, (usize, usize, usize))> {
        if x.ndim() != 5 {
            return Err(Error::Msg(format!(
                "patchify: expected 5D video [B, C, T, H, W], got {:?}",
                x.shape()
            )));
        }
        let shape = x.shape();
        let (b, c, t, h, w) = (shape[0], shape[1], shape[2], shape[3], shape[4]);
        let (pt, ph, pw) = self.config.patch_size;
        if pt == 0 || ph == 0 || pw == 0 {
            return Err(Error::Msg("patchify: patch dims must be positive".into()));
        }

        if t % pt != 0 || h % ph != 0 || w % pw != 0 {
            return Err(Error::Msg(format!(
                "video dimensions ({t}, {h}, {w}) not divisible by patch size ({pt}, {ph}, {pw})"
            )));
        }

        let (nt, nh, nw) = (t / pt, h / ph, w / pw);
        // `patch_size` is pub on the config: an all-empty video passes the
        // divisibility checks above (0 % p == 0) with absurd patches, so the
        // products below must be checked, not multiplied raw.
        let num_patches = nt
            .checked_mul(nh)
            .and_then(|v| v.checked_mul(nw))
            .ok_or_else(|| Error::Msg("patchify: patch count overflows".into()))?;
        let patch_dim = c
            .checked_mul(pt)
            .and_then(|v| v.checked_mul(ph))
            .and_then(|v| v.checked_mul(pw))
            .ok_or_else(|| Error::Msg("patchify: patch dim overflows".into()))?;

        let src = x.to_vec::<f32>()?;
        let flat = b
            .checked_mul(num_patches)
            .and_then(|v| v.checked_mul(patch_dim))
            .ok_or_else(|| Error::Msg("patchify: flat patch buffer overflows".into()))?;
        let mut patches = Vec::with_capacity(flat);

        for batch in 0..b {
            for it in 0..nt {
                for ih in 0..nh {
                    for iw in 0..nw {
                        for ch in 0..c {
                            for dt in 0..pt {
                                for dh in 0..ph {
                                    for dw in 0..pw {
                                        let frame = it * pt + dt;
                                        let row = ih * ph + dh;
                                        let col = iw * pw + dw;
                                        let idx =
                                            ((((batch * c + ch) * t + frame) * h + row) * w) + col;
                                        patches.push(src[idx]);
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        let out = Tensor::from_vec(patches, &[b, num_patches, patch_dim])?
            .to_device(&x.device())?
            .to_dtype(x.dtype())?;

        Ok((out, (nt, nh, nw)))
    }

    /// Unpatchifies 3D sequence tokens `[B, num_patches, patch_dim]` back into `[B, C, T, H, W]`.
    fn unpatchify(
        &self,
        tokens: &Tensor,
        grid_size: (usize, usize, usize),
        b: usize,
    ) -> Result<Tensor> {
        let (nt, nh, nw) = grid_size;
        let (pt, ph, pw) = self.config.patch_size;
        let c = self.config.in_channels;
        let patch_dim = c
            .checked_mul(pt)
            .and_then(|v| v.checked_mul(ph))
            .and_then(|v| v.checked_mul(pw))
            .ok_or_else(|| Error::Msg("unpatchify: patch dim overflows".into()))?;
        let num_patches = nt
            .checked_mul(nh)
            .and_then(|v| v.checked_mul(nw))
            .ok_or_else(|| Error::Msg("unpatchify: patch count overflows".into()))?;
        // `final_proj` is pub: a width-swapped projection reaches here with a
        // short token row, and the scatter below would index past the buffer.
        if tokens.shape() != [b, num_patches, patch_dim] {
            return Err(Error::ShapeMismatch {
                op: "unpatchify",
                expected: vec![b, num_patches, patch_dim],
                got: tokens.shape().to_vec(),
            });
        }
        let (t, h, w) = (nt * pt, nh * ph, nw * pw);
        let total = b * c * t * h * w;

        let src = tokens.to_vec::<f32>()?;
        let mut out = vec![0.0f32; total];
        let mut patch_idx = 0;

        for batch in 0..b {
            for it in 0..nt {
                for ih in 0..nh {
                    for iw in 0..nw {
                        let mut elem_offset = 0;
                        for ch in 0..c {
                            for dt in 0..pt {
                                for dh in 0..ph {
                                    for dw in 0..pw {
                                        let frame = it * pt + dt;
                                        let row = ih * ph + dh;
                                        let col = iw * pw + dw;
                                        let out_idx =
                                            ((((batch * c + ch) * t + frame) * h + row) * w) + col;
                                        let in_idx = patch_idx * patch_dim + elem_offset;
                                        out[out_idx] = src[in_idx];
                                        elem_offset += 1;
                                    }
                                }
                            }
                        }
                        patch_idx += 1;
                    }
                }
            }
        }

        Tensor::from_vec(out, &[b, c, t, h, w])?
            .to_device(&tokens.device())?
            .to_dtype(tokens.dtype())
    }

    /// Forward pass predicting vector velocity $v_\theta(x_t, t, c)$.
    pub fn forward(&self, x_t: &Tensor, t: &Tensor, text_cond: &Tensor) -> Result<Tensor> {
        // Read the batch AFTER patchify's rank check (a rank-0 input panicked `shape()[0]`).
        let (patches, grid) = self.patchify(x_t)?;
        let b = x_t.shape()[0];

        // Embed tokens
        let h_tokens = self.patch_proj.forward(&patches)?;

        // Embed timestep t in [0, 1]
        let t_sin = continuous_timestep_embedding(t, self.config.hidden_dim)?;
        let t_emb = self
            .time_mlp2
            .forward(&self.time_mlp1.forward(&t_sin)?.silu()?)?;

        // Apply Transformer Block
        let h_out = self.block.forward(&h_tokens, &t_emb, text_cond, grid)?;

        // Final Projection back to velocity field
        let h_norm = self.final_norm.forward(&h_out)?;
        let patch_velocities = self.final_proj.forward(&h_norm)?;

        self.unpatchify(&patch_velocities, grid, b)
    }

    /// Computes Rectified Flow loss $\| v_\theta(x_t, t, c) - (x_1 - x_0) \|^2$.
    pub fn loss(
        &self,
        x_0: &Tensor,
        x_1: &Tensor,
        t: &Tensor,
        text_cond: &Tensor,
    ) -> Result<Tensor> {
        if x_0.ndim() != 5 || x_1.ndim() != 5 {
            return Err(Error::Msg(format!(
                "RectifiedFlowVideoDiT::loss: x_0/x_1 must be 5D video, got {:?}/{:?}",
                x_0.shape(),
                x_1.shape()
            )));
        }
        let b = x_0.shape()[0];
        if t.ndim() != 2 || t.shape()[0] != b || t.shape()[1] != 1 {
            return Err(Error::Msg(format!(
                "RectifiedFlowVideoDiT::loss: t must be [B, 1] with B={b}, got {:?}",
                t.shape()
            )));
        }
        // x_t = (1 - t) * x_0 + t * x_1, with t reshaped to [B,1,1,1,1]
        // so it broadcasts against 5D video.
        let t_mix = t.reshape(&[b, 1, 1, 1, 1])?;
        let one_minus_t = t_mix.neg()?.add_scalar(1.0)?;
        let x_t = x_0.mul(&one_minus_t)?.add(&x_1.mul(&t_mix)?)?;

        // Target velocity v = x_1 - x_0
        let target_v = x_1.sub(x_0)?;

        let pred_v = self.forward(&x_t, t, text_cond)?;
        let diff = pred_v.sub(&target_v)?;
        diff.powf(2.0)?.mean()
    }

    /// Solves the Rectified Flow ODE using Euler steps from $t=0$ to $t=1$.
    pub fn sample_euler(
        &self,
        init_noise: &Tensor,
        text_cond: &Tensor,
        num_steps: usize,
    ) -> Result<Tensor> {
        if num_steps == 0 {
            return Err(Error::Msg(
                "RectifiedFlowVideoDiT::sample_euler requires num_steps > 0".into(),
            ));
        }
        let num_steps = crate::limits::check_iterations("sample_euler num_steps", num_steps)?;
        if init_noise.ndim() != 5 {
            return Err(Error::Msg(format!(
                "RectifiedFlowVideoDiT::sample_euler: init_noise must be 5D video, got {:?}",
                init_noise.shape()
            )));
        }
        let mut x = init_noise.clone();
        let dt = 1.0 / (num_steps as f64);
        let b = init_noise.shape()[0];

        for step in 0..num_steps {
            let t_val = (step as f64) * dt;
            let t_tensor = Tensor::from_vec(vec![t_val as f32; b], &[b, 1])?
                .to_device(&init_noise.device())?
                .to_dtype(init_noise.dtype())?;

            let v = self.forward(&x, &t_tensor, text_cond)?;
            x = x.add(&v.mul_scalar(dt)?)?;
        }

        Ok(x)
    }
}
