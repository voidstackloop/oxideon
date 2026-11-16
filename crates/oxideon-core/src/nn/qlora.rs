//! Quantized Low-Rank Adaptation (QLoRA) linear module.
//!
//! Bridges 4-bit / 8-bit quantized frozen base weights with high-precision low-rank adapters:
//! $$y = W_{\text{quant}} x + \frac{\alpha}{r} (x A^T) B^T + b$$

use crate::dtype::DType;
use crate::error::Result;
use crate::nn::{Dropout, Linear, Module, init};
use crate::quant::{QuantScheme, QuantizedLinear, QuantizedTensor};
use crate::tensor::{Tensor, no_grad};

/// Quantized Low-Rank Adaptation Linear layer.
pub struct QLoRALinear {
    /// Frozen quantized base linear layer (e.g. Q4_0 or Q8_0).
    pub base: QuantizedLinear,
    /// Down-projection adapter matrix: shape `[rank, in_features]`.
    pub lora_a: Tensor,
    /// Up-projection adapter matrix: shape `[out_features, rank]`.
    pub lora_b: Tensor,
    /// Optional dropout on adapter inputs.
    pub dropout: Dropout,
    /// Adapter rank $r$.
    pub rank: usize,
    /// Adapter scaling parameter $\alpha$.
    pub alpha: f64,
    /// Scaling multiplier: $\alpha / r$.
    pub scaling: f64,
    /// Whether the adapter path is currently active.
    pub enabled: bool,
}

impl QLoRALinear {
    /// Constructs a `QLoRALinear` wrapping an existing `QuantizedLinear` base layer.
    pub fn new(base: QuantizedLinear, rank: usize, alpha: f64, dropout_p: f32) -> Result<Self> {
        if rank == 0 {
            return Err(crate::error::Error::Msg(
                "QLoRA rank must be > 0".to_string(),
            ));
        }
        let in_f = base.in_features;
        let out_f = base.out_features;
        if in_f == 0 {
            return Err(crate::error::Error::Msg(
                "QLoRA in_features must be > 0".to_string(),
            ));
        }
        // `Dropout::new` asserts on NaN; reject it here with an Err instead.
        if !dropout_p.is_finite() {
            return Err(crate::error::Error::Msg(
                "QLoRA dropout_p must be finite".to_string(),
            ));
        }
        let bound = 1.0 / (in_f as f32).sqrt();

        // lora_a ~ U(-1/sqrt(in), 1/sqrt(in)) and lora_b = 0
        let lora_a = init::uniform(&[rank, in_f], -bound, bound)?
            .to_dtype(DType::F32)?
            .requires_grad_(true);
        let lora_b =
            Tensor::zeros(&[out_f, rank], DType::F32, &lora_a.device())?.requires_grad_(true);
        let scaling = alpha / (rank as f64);

        Ok(Self {
            base,
            lora_a,
            lora_b,
            dropout: Dropout::new(dropout_p),
            rank,
            alpha,
            scaling,
            enabled: true,
        })
    }

    /// Constructs a `QLoRALinear` from an unquantized `Linear` layer by quantizing its weights.
    pub fn from_linear(
        linear: &Linear,
        scheme: QuantScheme,
        rank: usize,
        alpha: f64,
        dropout_p: f32,
    ) -> Result<Self> {
        let base = QuantizedLinear::from_linear(linear, scheme)?;
        Self::new(base, rank, alpha, dropout_p)
    }

    /// Input features dimension.
    pub fn in_features(&self) -> usize {
        self.base.in_features
    }

    /// Output features dimension.
    pub fn out_features(&self) -> usize {
        self.base.out_features
    }

    /// Enables the LoRA adapter.
    pub fn enable_adapter(&mut self) {
        self.enabled = true;
    }

    /// Disables the LoRA adapter (runs pure quantized base model).
    pub fn disable_adapter(&mut self) {
        self.enabled = false;
    }

    /// Computes adapter update matrix $\Delta W = \frac{\alpha}{r} B A$.
    pub fn delta_weight(&self) -> Result<Tensor> {
        self.lora_b.matmul(&self.lora_a)?.mul_scalar(self.scaling)
    }

    /// Merges the LoRA adapter weights directly into the base quantized representation.
    pub fn merge_adapter(&mut self) -> Result<()> {
        if !self.enabled {
            return Ok(());
        }

        let scheme = self.base.qweight.scheme;
        no_grad(|| {
            // Dequantize base weights: [out_f, in_f]
            let base_w = self.base.qweight.dequantize(DType::F32)?;
            let delta = self.delta_weight()?;
            let merged_w = base_w.add(&delta)?;
            self.base.qweight = QuantizedTensor::quantize(&merged_w, scheme)?;
            self.enabled = false;
            Ok::<(), crate::error::Error>(())
        })?;

        Ok(())
    }
}

impl Module for QLoRALinear {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        let base_out = self.base.forward(input)?;
        if !self.enabled || self.rank == 0 {
            return Ok(base_out);
        }

        let x_drop = self.dropout.forward(input)?;
        let at = self.lora_a.t()?;
        let x_a = if x_drop.ndim() == 2 {
            x_drop.matmul(&at)?
        } else {
            let orig = x_drop.shape().to_vec();
            let k = *orig
                .last()
                .ok_or_else(|| crate::error::Error::Msg("QLoRA input requires rank >= 1".into()))?;
            if k == 0 {
                return Err(crate::error::Error::Msg(
                    "QLoRA input last dim must be non-empty".into(),
                ));
            }
            let rows = x_drop.numel() / k;
            let flat = x_drop.reshape(&[rows, k])?.matmul(&at)?;
            let mut out_shape = orig[..orig.len() - 1].to_vec();
            out_shape.push(self.rank);
            flat.reshape(&out_shape)?
        };

        let bt = self.lora_b.t()?;
        let x_ab = if x_a.ndim() == 2 {
            x_a.matmul(&bt)?
        } else {
            let orig = x_a.shape().to_vec();
            let k = *orig.last().ok_or_else(|| {
                crate::error::Error::Msg("QLoRA adapter state requires rank >= 1".into())
            })?;
            if k == 0 {
                return Err(crate::error::Error::Msg(
                    "QLoRA adapter last dim must be non-empty".into(),
                ));
            }
            let rows = x_a.numel() / k;
            let flat = x_a.reshape(&[rows, k])?.matmul(&bt)?;
            let mut out_shape = orig[..orig.len() - 1].to_vec();
            out_shape.push(self.base.out_features);
            flat.reshape(&out_shape)?
        };

        let adapter_out = x_ab.mul_scalar(self.scaling)?;
        base_out.add(&adapter_out)
    }

    /// Exposes only the trainable LoRA adapter parameters.
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn qlora_rejects_zero_width_base() -> Result<()> {
        let lin = Linear::with_bias(4, 4, false)?;
        let mut base = QuantizedLinear::from_linear(&lin, QuantScheme::DEFAULT_Q8_0)?;
        base.in_features = 0; // pub field: 1/sqrt(0) bound would silently NaN-fill the adapter
        let err = match QLoRALinear::new(base, 2, 1.0, 0.0) {
            Ok(_) => panic!("0-width base must be an error, not a NaN adapter"),
            Err(e) => e,
        };
        assert!(err.to_string().contains("in_features"), "{err}");
        Ok(())
    }
}
