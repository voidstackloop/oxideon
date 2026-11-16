//! Low-Rank Adaptation (`LoRA`, Hu et al. 2021) for Parameter-Efficient Fine-Tuning (PEFT).
//!
//! [`LoRALinear`] freezes the pretrained base weight $W_0 \in \mathbb{R}^{d_{\text{out}} \times d_{\text{in}}}$
//! and learns a low-rank update $\Delta W = \frac{\alpha}{r} B A$ with
//! $A \in \mathbb{R}^{r \times d_{\text{in}}}$ and $B \in \mathbb{R}^{d_{\text{out}} \times r}$ ($r \ll \min(d_{\text{in}}, d_{\text{out}})$).
//! Because [`LoRALinear::named_parameters`] exposes **only** `lora_a` and `lora_b`, any
//! optimizer automatically trains just the low-rank adapter!

use std::sync::atomic::{AtomicBool, Ordering};

use super::{Dropout, Linear, Module, init};
use crate::error::Result;
use crate::tensor::{Tensor, no_grad};

pub struct LoRALinear {
    pub base_weight: Tensor,
    pub bias: Option<Tensor>,
    pub lora_a: Tensor,
    pub lora_b: Tensor,
    pub dropout: Dropout,
    pub rank: usize,
    pub alpha: f64,
    pub scaling: f64,
    merged: AtomicBool,
}

impl LoRALinear {
    /// Wraps an existing [`Linear`] layer with a trainable LoRA adapter of `rank` and `alpha`.
    pub fn from_linear(linear: &Linear, rank: usize, alpha: f64, dropout_p: f32) -> Result<Self> {
        if rank == 0 {
            return Err(crate::error::Error::Msg(
                "LoRA rank must be > 0".to_string(),
            ));
        }
        let in_f = linear.in_features();
        let out_f = linear.out_features();
        if in_f == 0 {
            return Err(crate::error::Error::Msg(
                "LoRA in_features must be > 0".to_string(),
            ));
        }
        // `Dropout::new` asserts on NaN; reject it here with an Err instead.
        if !dropout_p.is_finite() {
            return Err(crate::error::Error::Msg(
                "LoRA dropout_p must be finite".to_string(),
            ));
        }
        let bound = 1.0 / (in_f as f32).sqrt();

        // Freeze base weight; initialize A ~ U(-1/sqrt(in), 1/sqrt(in)) and B = 0 so Delta W = 0 at start.
        let base_weight = linear.weight.detach();
        let bias = linear.bias.as_ref().map(Tensor::detach);
        let lora_a = init::uniform(&[rank, in_f], -bound, bound)?
            .to_dtype(base_weight.dtype())?
            .to_device(&base_weight.device())?
            .requires_grad_(true);
        let lora_b = Tensor::zeros(&[out_f, rank], base_weight.dtype(), &base_weight.device())?
            .requires_grad_(true);

        Ok(Self {
            base_weight,
            bias,
            lora_a,
            lora_b,
            dropout: Dropout::new(dropout_p),
            rank,
            alpha,
            scaling: alpha / (rank as f64),
            merged: AtomicBool::new(false),
        })
    }

    /// Number of frozen base parameters.
    pub fn num_frozen_parameters(&self) -> usize {
        self.base_weight.numel() + self.bias.as_ref().map_or(0, Tensor::numel)
    }

    /// Computes $\Delta W = \frac{\alpha}{r} B A$ of shape `[out_features, in_features]`.
    pub fn delta_weight(&self) -> Result<Tensor> {
        self.lora_b.matmul(&self.lora_a)?.mul_scalar(self.scaling)
    }

    /// Merges the learned LoRA adapter into `base_weight` in-place for zero-overhead inference.
    pub fn merge(&self) -> Result<()> {
        if !self.merged.swap(true, Ordering::Relaxed) {
            no_grad(|| {
                let merged_w = self.base_weight.add(&self.delta_weight()?)?;
                self.base_weight.assign(&merged_w)
            })?;
        }
        Ok(())
    }

    /// Unmerges the LoRA adapter from `base_weight` in-place to resume adapter training.
    pub fn unmerge(&self) -> Result<()> {
        if self.merged.swap(false, Ordering::Relaxed) {
            no_grad(|| {
                let unmerged_w = self.base_weight.sub(&self.delta_weight()?)?;
                self.base_weight.assign(&unmerged_w)
            })?;
        }
        Ok(())
    }

    /// Exports a standard [`Linear`] layer with the LoRA weights folded into `weight`.
    pub fn to_merged_linear(&self) -> Result<Linear> {
        let w = no_grad(|| {
            if self.merged.load(Ordering::Relaxed) {
                Ok(self.base_weight.clone())
            } else {
                self.base_weight.add(&self.delta_weight()?)
            }
        })?;
        Linear::from_tensors(w, self.bias.clone())
    }
}

impl Module for LoRALinear {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        let base_out = project_2d_or_nd(input, &self.base_weight)?;
        let out = if self.merged.load(Ordering::Relaxed) {
            base_out
        } else {
            let x_drop = self.dropout.forward(input)?;
            let low_rank = project_2d_or_nd(&x_drop, &self.lora_a)?;
            let lora_out = project_2d_or_nd(&low_rank, &self.lora_b)?.mul_scalar(self.scaling)?;
            base_out.add(&lora_out)?
        };
        match &self.bias {
            Some(b) => out.add(b),
            None => Ok(out),
        }
    }

    /// Returns **only** the trainable LoRA adapter parameters (`lora_a` and `lora_b`).
    fn named_parameters(&self) -> Vec<(String, Tensor)> {
        vec![
            ("lora_a".to_string(), self.lora_a.clone()),
            ("lora_b".to_string(), self.lora_b.clone()),
        ]
    }

    fn train(&self, mode: bool) {
        self.dropout.train(mode);
    }
}

fn project_2d_or_nd(input: &Tensor, weight: &Tensor) -> Result<Tensor> {
    let wt = weight.t()?;
    let out_f = *weight
        .shape()
        .first()
        .ok_or_else(|| crate::error::Error::Msg("LoRA weight must be 2D".into()))?;
    if input.ndim() == 2 {
        input.matmul(&wt)
    } else {
        let orig = input.shape().to_vec();
        let k = *orig
            .last()
            .ok_or_else(|| crate::error::Error::Msg("LoRA input requires rank >= 1".into()))?;
        if k == 0 {
            return Err(crate::error::Error::Msg(
                "LoRA input last dim must be non-empty".into(),
            ));
        }
        let rows = input.numel() / k;
        let flat = input.reshape(&[rows, k])?.matmul(&wt)?;
        let mut out_shape = orig[..orig.len() - 1].to_vec();
        out_shape.push(out_f);
        flat.reshape(&out_shape)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lora_rejects_zero_width_base() -> Result<()> {
        let lin = Linear::with_bias(0, 4, false)?;
        let err = match LoRALinear::from_linear(&lin, 2, 1.0, 0.0) {
            Ok(_) => panic!("0-width base must be an error, not a NaN adapter"),
            Err(e) => e,
        };
        assert!(err.to_string().contains("in_features"), "{err}");
        Ok(())
    }
}
