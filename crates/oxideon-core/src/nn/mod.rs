//! Neural network modules, layers, normalization, RNNs, Transformers, losses, and initializers.

pub mod activations;
pub mod activations_ext;
pub mod architectures;
pub mod attention_ext;
pub mod containers;
pub mod conv_ext;
pub mod conv3d;
pub mod deepseek_moe;
pub mod deq;
pub mod frontier;
pub mod functional_ext;
pub mod gguf_model;
pub mod gguf_bert;
pub mod gguf_rwkv;
pub mod gguf_ssm;
pub mod gnn;
pub mod gnn_ext;
pub mod module_tools;
pub mod implicit;
pub mod layers_misc;
pub mod hooks;
pub mod init;
pub mod init_ext;
pub mod kan;
pub mod lora;
pub mod loss;
pub mod layers_ext;
pub mod loss_ext;
pub mod loss_modules;
pub mod loss_vision;
pub mod mamba2;
pub mod moe;
pub mod neuromorphic;
pub mod qlora;
pub mod rnn;
pub mod spiking;
pub mod ssm;
pub mod stdp;
pub mod titans;
pub mod transformer;
pub mod utils;
pub mod transfusion;
pub mod vision;
pub mod vla;

mod layers;
mod sequential;
pub mod sequence_ext;

use std::collections::HashMap;

pub use activations::{
    AlphaDropout, Bilinear, CELU, CosineLinear, DropPath, Dropout2d, GLU, Hardshrink, Hardsigmoid,
    Hardswish, Hardtanh, LogSigmoid, Mish, PReLU, ReLU6, SELU, Softplus, Softshrink, Softsign,
    SpectralNormLinear, WeightNormLinear,
};
pub use architectures::{
    ActivationKind, BiLSTM, ConvNeXt, ConvNeXtBlock, MLP, MLPMixer, MLPMixerBlock,
    SqueezeExcitation2d, TCNBlock, TemporalConvNet, VariationalAutoencoder,
};
pub use containers::{
    Identity, LambdaModule, ModuleDict, ModuleList, ParallelAdd, ParallelCat, Residual,
};
pub use conv3d::{
    AvgPool3d, BatchNorm3d, Conv3d, ConvTranspose3d, MaxPool3d, PixelShuffle, PixelUnshuffle,
};
pub use deepseek_moe::{AuxFreeRouter, DeepSeekV3MoE};
pub use deq::{DeepEquilibriumModel, DeqConfig, DeqForwardOutput};
pub use frontier::{
    DeepSeekBlock, DeepSeekMoELM, DiT, DiTBlock, HybridChannelMixer, HybridJambaLM,
    HybridSequenceMixer, MultiTokenPredictionModule, MultimodalVLM, RewardModel, SharedRoutedMoE,
    bradley_terry_reward_loss,
};
pub use gguf_model::{GGUFCausalLM, GGUFDecoderBlock, GGUFModelConfig, GGUFTokenizer, QkNorm};
pub use gnn::{GATConv, GCNConv, normalize_adjacency};
pub use kan::{KAN, KANLinear};
pub use layers::{
    AdaptiveAvgPool1d, AdaptiveAvgPool2d, AdaptiveMaxPool1d, AdaptiveMaxPool2d, AvgPool1d,
    AvgPool2d, BatchNorm1d, BatchNorm2d, Conv1d, Conv2d, ConvTranspose1d, ConvTranspose2d, Dropout,
    ELU, Embedding, EmbeddingBag, EmbeddingBagMode, Flatten, GELU, GroupNorm, InstanceNorm1d,
    InstanceNorm2d, InstanceNorm3d, LPPool1d, LPPool2d, LayerNorm, LeakyReLU, Linear, MaxPool1d,
    MaxPool2d, MultiHeadAttention, RMSNorm, ReLU, ResidualBlock, SiLU, Sigmoid, Tanh, Upsample2d,
    UpsampleMode,
};
pub use layers_misc::ConvLstmCell;
pub use lora::LoRALinear;
pub use loss::{
    CTCLoss, HingeEmbeddingLoss, MarginRankingLoss, SoftMarginLoss, barlow_twins_loss,
    binary_cross_entropy_with_logits, cosine_embedding_loss, cosine_similarity, cross_entropy,
    cross_entropy_with_smoothing, ctc_loss, focal_loss, hinge_embedding_loss, huber_loss,
    info_nce_loss, kl_div_loss, l1_loss, margin_ranking_loss, mse_loss, multilabel_margin_loss,
    nll_loss, siglip_loss, simpo_loss, soft_margin_loss, triplet_margin_loss,
};
pub use mamba2::Mamba2Block;
pub use moe::SparseMoE;
pub use neuromorphic::{EventDrivenSpikeRouter, PolychronousNeuralLayer, SpikeEvent};
pub use qlora::QLoRALinear;
pub use rnn::{GRU, GRUCell, LSTM, LSTMCell, RNN, RNNCell};
pub use sequence_ext::{Maxout, Snake, Swish};
pub use sequential::Sequential;
pub use spiking::{LIFNode, PoissonEncoder, ResetMode, SurrogateKind};
pub use ssm::{HyenaOperator, MambaBlock, SelectiveSSM};
pub use stdp::{STDPConfig, STDPSynapse};
pub use titans::{TitansBlock, TitansMemoryConfig, TitansNeuralMemory};
pub use transformer::{
    CausalLM, CausalLMBlock, DiffAttention, FeedForward, GroupedQueryAttention, KVCache,
    LayerKVCache, MultiHeadLatentAttention, RotaryEmbedding, SinusoidalPositionalEncoding,
    SlidingWindowAttention, SwiGLU, TransformerDecoder, TransformerDecoderLayer,
    TransformerEncoder, TransformerEncoderLayer, alibi_bias, flash_attention,
};
pub use transfusion::{TransfusionConfig, TransfusionModel};
pub use vision::{
    DoubleConv2d, PatchEmbedding, UNet2d, VisionTransformer, vae_kl_loss, vae_reparameterize,
};
pub use vla::{GatedCrossAttention, VisionLanguageActionModel};

use crate::error::{Error, Result};
use crate::tensor::Tensor;

/// A differentiable neural network building block.
pub trait Module {
    fn forward(&self, input: &Tensor) -> Result<Tensor>;

    /// Returns `(name, tensor)` pairs for all trainable parameters.
    fn named_parameters(&self) -> Vec<(String, Tensor)> {
        Vec::new()
    }

    /// Returns `(name, tensor)` pairs for the non-trainable state a module keeps (for example the running statistics
    /// of batch normalization). Buffers are saved and loaded with the parameters but receive no gradients.
    fn named_buffers(&self) -> Vec<(String, Tensor)> {
        Vec::new()
    }

    /// Returns all buffers.
    fn buffers(&self) -> Vec<Tensor> {
        self.named_buffers().into_iter().map(|(_, t)| t).collect()
    }

    /// Returns all trainable parameters.
    fn parameters(&self) -> Vec<Tensor> {
        self.named_parameters()
            .into_iter()
            .map(|(_, t)| t)
            .collect()
    }

    /// Switches between training (`true`) and evaluation (`false`) mode.
    fn train(&self, _mode: bool) {}

    fn eval(&self) {
        self.train(false);
    }

    fn num_parameters(&self) -> usize {
        self.parameters().iter().map(Tensor::numel).sum()
    }

    /// Counts only parameters that currently have `requires_grad() == true`.
    fn num_trainable_parameters(&self) -> usize {
        self.parameters()
            .iter()
            .filter(|p| p.requires_grad())
            .map(Tensor::numel)
            .sum()
    }

    /// Freezes all parameters of this module (`set_requires_grad(false)`).
    fn freeze(&self) {
        self.parameters()
            .iter()
            .for_each(|p| p.set_requires_grad(false).expect("freeze never fails"));
    }

    /// Unfreezes all parameters of this module (`set_requires_grad(true)`).
    /// Non-float parameters (which cannot require grad) are left frozen.
    fn unfreeze(&self) {
        self.parameters()
            .iter()
            .for_each(|p| _ = p.set_requires_grad(true));
    }

    fn zero_grad(&self) {
        self.parameters().iter().for_each(Tensor::zero_grad);
    }

    /// Exports a map of parameter names to tensors (compatible with [`crate::serialize`]).
    fn state_dict(&self) -> HashMap<String, Tensor> {
        self.named_parameters()
            .into_iter()
            .chain(self.named_buffers())
            .collect()
    }

    /// Loads parameter tensors from `state` into this module in-place.
    fn load_state_dict(&self, state: &HashMap<String, Tensor>) -> Result<()> {
        for (name, param) in self.named_parameters() {
            let Some(src) = state.get(&name) else {
                return Err(Error::Serialization(format!(
                    "missing parameter '{name}' in state_dict"
                )));
            };
            param.assign(src)?;
        }
        // Buffers are optional so checkpoints written before they were exported still load.
        for (name, buf) in self.named_buffers() {
            if let Some(src) = state.get(&name) {
                buf.assign(src)?;
            }
        }
        Ok(())
    }

    /// Saves all parameters of this module directly to a `.safetensors` file at `path`.
    fn save(&self, path: &std::path::Path) -> Result<()> {
        crate::serialize::save_safetensors(&self.state_dict(), path)
    }

    /// Loads all parameters of this module directly from a `.safetensors` file at `path`.
    fn load(&self, path: &std::path::Path) -> Result<()> {
        let state = crate::serialize::load_safetensors(path)?;
        self.load_state_dict(&state)
    }

    /// Formats a human-readable parameter table for this module.
    fn summary(&self) -> String {
        let params = self.named_parameters();
        let mut out = String::new();
        out.push_str(&format!(
            "{:<36} | {:<18} | {:<8} | {:>10}\n",
            "Parameter", "Shape", "DType", "Elements"
        ));
        out.push_str(&"-".repeat(80));
        out.push('\n');
        let mut total = 0usize;
        for (name, t) in &params {
            let n = t.numel();
            total += n;
            out.push_str(&format!(
                "{:<36} | {:<18} | {:<8} | {:>10}\n",
                name,
                format!("{:?}", t.shape()),
                t.dtype().to_string(),
                n
            ));
        }
        out.push_str(&"-".repeat(80));
        out.push_str(&format!("\nTotal trainable parameters: {total}\n"));
        out
    }
}

impl<M: Module + ?Sized> Module for Box<M> {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        (**self).forward(input)
    }
    fn named_parameters(&self) -> Vec<(String, Tensor)> {
        (**self).named_parameters()
    }
    fn named_buffers(&self) -> Vec<(String, Tensor)> {
        (**self).named_buffers()
    }
    fn train(&self, mode: bool) {
        (**self).train(mode)
    }
}

pub mod rademacher_vapnik_pac_bayes;
pub use rademacher_vapnik_pac_bayes::PacBayesMcAllesterEngine;

pub mod jacot_neural_tangent_kernel;
pub use jacot_neural_tangent_kernel::InfiniteWidthNtkEngine;

pub mod yang_maximal_update_mup;
pub use yang_maximal_update_mup::MaximalUpdateParamEngine;

pub mod chinchilla_kaplan_scaling_laws;
pub use chinchilla_kaplan_scaling_laws::ComputeOptimalChinchillaEngine;

pub mod belkin_double_descent_marchenko;
pub use belkin_double_descent_marchenko::OverparameterizedDoubleDescent;

pub mod candes_tao_compressed_sensing_rip;
pub use candes_tao_compressed_sensing_rip::RestrictedIsometryLassoEngine;

pub mod talagrand_concentration_log_sobolev;
pub use talagrand_concentration_log_sobolev::LogSobolevConcentrationEngine;

pub mod vovk_conformal_prediction_martingale;
pub use vovk_conformal_prediction_martingale::SplitConformalPredictionEngine;

pub mod rahimi_recht_random_fourier_features;
pub use rahimi_recht_random_fourier_features::RandomFourierBochnerEngine;
