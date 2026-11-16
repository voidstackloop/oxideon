//! GGUF Causal Language Model runner for direct quantized inference.
//!
//! Loads transformer architecture configurations and quantized weights directly
//! from GGUF containers without external converters.

use crate::error::{Error, Result};
use crate::nn::gguf_ssm::{GdnParts, KdaParts, SsmBlock, SsmConfig, SsmKind};
use crate::nn::{Embedding, Linear, Module, RotaryEmbedding};
use crate::serialize::gguf::GGUFFile;
use crate::tensor::Tensor;

/// RMS norm or LayerNorm with optional weight and bias (llama.cpp's `LLM_NORM_RMS` / `LLM_NORM`).
pub struct Norm {
    /// Per-feature scale. For per-head Q/K norms this is `[heads, head_dim]` and broadcasts over tokens.
    pub weight: Tensor,
    pub bias: Option<Tensor>,
    pub eps: f64,
    /// LayerNorm (subtract the mean) instead of RMS norm.
    pub layer: bool,
    /// Multiply by `weight` (false for OLMo 1's non-parametric LayerNorm).
    pub scaled: bool,
    /// The features are split into this many equal groups that are normalized separately (K2 Horizon); 1 = one norm.
    pub groups: usize,
}

impl Norm {
    /// Splits the normalized width into `groups` independently normalized slices.
    pub fn with_groups(mut self, groups: usize) -> Self {
        self.groups = groups.max(1);
        self
    }

    pub fn rms(dim: usize, eps: f64) -> Result<Self> {
        Self::build(false, false, true, dim, eps)
    }

    pub fn build(layer: bool, with_bias: bool, scaled: bool, dim: usize, eps: f64) -> Result<Self> {
        Ok(Self {
            weight: Tensor::ones_f32(&[dim])?,
            bias: if with_bias {
                Some(Tensor::zeros_f32(&[dim])?)
            } else {
                None
            },
            eps,
            layer,
            scaled,
            groups: 1,
        })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        if self.groups > 1 {
            let shape = x.shape().to_vec();
            let d = *shape
                .last()
                .ok_or_else(|| Error::Msg("norm input needs rank >= 1".into()))?;
            if d % self.groups != 0 {
                return Err(Error::Msg(format!(
                    "{d} features do not split into {} norm groups",
                    self.groups
                )));
            }
            let mut grouped = shape.clone();
            grouped.pop();
            grouped.extend([self.groups, d / self.groups]);
            let plain = Norm {
                weight: Tensor::ones_f32(&[d / self.groups])?,
                bias: None,
                eps: self.eps,
                layer: self.layer,
                scaled: false,
                groups: 1,
            };
            let y = plain.forward(&x.reshape(&grouped)?)?.reshape(&shape)?;
            let y = if self.scaled { y.mul(&self.weight)? } else { y };
            return match &self.bias {
                Some(b) => y.add(b),
                None => Ok(y),
            };
        }
        let normed = if self.layer {
            let centered = x.sub(&x.mean_axis(-1, true)?)?;
            let var = centered.square()?.mean_axis(-1, true)?;
            centered.mul(&var.add_scalar(self.eps)?.rsqrt()?)?
        } else {
            x.mul(
                &x.square()?
                    .mean_axis(-1, true)?
                    .add_scalar(self.eps)?
                    .rsqrt()?,
            )?
        };
        let y = if self.scaled {
            normed.mul(&self.weight)?
        } else {
            normed
        };
        match &self.bias {
            Some(b) => y.add(b),
            None => Ok(y),
        }
    }
}

/// Loads a norm's optional `.weight` / `.bias` tensors (`base` is e.g. `blk.0.attn_norm`).
/// ggml's ALiBi slope for head `h` of `n_head` (`ggml_soft_max_ext` with `max_bias`).
fn alibi_slopes(n_head: usize, max_bias: f32) -> Vec<f32> {
    let pow2 = 1usize << (usize::BITS - 1 - n_head.max(1).leading_zeros());
    let m0 = 2f32.powf(-max_bias / pow2 as f32);
    let m1 = 2f32.powf(-(max_bias / 2.0) / pow2 as f32);
    (0..n_head)
        .map(|h| {
            if h < pow2 {
                m0.powi(h as i32 + 1)
            } else {
                m1.powi(2 * (h - pow2) as i32 + 1)
            }
        })
        .collect()
}

pub(crate) fn load_norm(file: &GGUFFile, base: &str, norm: &mut Norm) -> Result<()> {
    if norm.scaled {
        // Some converters write the bare tensor name (`ffn_norm`) without `.weight`.
        let name = if !file.tensors.contains_key(&format!("{base}.weight"))
            && file.tensors.contains_key(base)
        {
            base.to_string()
        } else {
            format!("{base}.weight")
        };
        norm.weight = load_named(file, &name, norm.weight.shape())?;
    }
    if norm.bias.is_some() {
        norm.bias = load_bias(file, &format!("{base}.bias"), norm.weight.shape()[0])?;
    }
    Ok(())
}

/// Feed-forward gate activation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Activation {
    Silu,
    /// ggml's `GELU` (the tanh approximation), used by Gemma.
    GeluTanh,
    /// `relu(x)^2` (Arcee, Nemotron, Jais 2).
    ReluSqr,
    /// Kimi-K3's `beta tanh(g / beta) sigmoid(g)` gate and `linear_beta tanh(u / linear_beta)` up branch
    /// (`f32::to_bits` of `situ_beta` and `situ_linear_beta`; a non-positive linear beta leaves `u` alone).
    Situ(u32, u32),
    /// OpenAI's clamped SwiGLU on a dense FFN (MiniMax-M3): see [`MoeAct::SwigluOai`].
    SwigluOai,
    /// Apertus' xIELU on the (ungated) up projection: the layer's `[alpha_n, alpha_p, beta, eps]` as `f32::to_bits`.
    Xielu([u32; 4]),
}

/// ggml's `xielu`: `alpha_p' x^2 + beta x` above zero, `alpha_n' (expm1(min(x, eps)) - x) + beta x` below, with
/// `alpha_p' = softplus(alpha_p)` and `alpha_n' = beta + softplus(alpha_n)`.
fn xielu(x: &Tensor, p: [u32; 4]) -> Result<Tensor> {
    let [an, ap, beta, eps] = p.map(f32::from_bits);
    let softplus = |v: f32| if v > 20.0 { v } else { v.exp().ln_1p() };
    let (an, ap) = (beta + softplus(an), softplus(ap));
    let v: Vec<f32> = x
        .to_vec::<f32>()?
        .into_iter()
        .map(|x| {
            if x > 0.0 {
                ap * x * x + beta * x
            } else {
                (x.min(eps).exp_m1() - x) * an + beta * x
            }
        })
        .collect();
    Tensor::from_slice_on(&v, x.shape(), &x.device())
}

static GGML_GELU_TABLE: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// Makes [`Activation::GeluTanh`] reproduce ggml's CPU kernel, which looks `gelu` up in a table indexed by the f16 bits
/// of the input and stores f16 results (about 5e-4 relative noise per activation). Off by default; parity tests turn it
/// on so the noise of the reference does not compound over many layers into the comparison.
pub fn emulate_ggml_gelu_table(on: bool) {
    GGML_GELU_TABLE.store(on, std::sync::atomic::Ordering::Relaxed);
}

/// ggml's table GELU: `0` below -10, the identity above 10, else `f16(gelu(f16(x)))`.
fn ggml_gelu_table(x: &Tensor) -> Result<Tensor> {
    use crate::dtype::f16;
    let v: Vec<f32> = x
        .to_vec::<f32>()?
        .into_iter()
        .map(|x| {
            if x <= -10.0 {
                0.0
            } else if x >= 10.0 {
                x
            } else {
                let h = f16::from_f32(x).to_f32();
                let g = 0.5 * h * (1.0 + (0.797_884_6 * h * (1.0 + 0.044_715 * h * h)).tanh());
                f16::from_f32(g).to_f32()
            }
        })
        .collect();
    Tensor::from_slice_on(&v, x.shape(), &x.device())
}

/// `beta * tanh(x / beta)`.
fn soft_cap(x: &Tensor, beta: f32) -> Result<Tensor> {
    x.mul_scalar(1.0 / f64::from(beta))?
        .tanh()?
        .mul_scalar(f64::from(beta))
}

impl Activation {
    /// The transform of the up branch (identity for everything but SITU).
    fn up(self, u: &Tensor) -> Result<Tensor> {
        match self {
            Activation::Situ(_, lin) if f32::from_bits(lin) > 0.0 => {
                soft_cap(u, f32::from_bits(lin))
            }
            Activation::SwigluOai => u.clamp(-7.0, 7.0)?.add_scalar(1.0),
            _ => Ok(u.clone()),
        }
    }

    pub(crate) fn apply(self, x: &Tensor) -> Result<Tensor> {
        match self {
            Activation::SwigluOai => {
                let g = x.clamp(-1e30, 7.0)?;
                g.mul(&g.mul_scalar(1.702)?.sigmoid()?)
            }
            Activation::Silu => x.silu(),
            Activation::Situ(beta, _) => soft_cap(x, f32::from_bits(beta))?.mul(&x.sigmoid()?),
            Activation::ReluSqr => x.relu()?.square(),
            Activation::Xielu(p) => xielu(x, p),
            Activation::GeluTanh if GGML_GELU_TABLE.load(std::sync::atomic::Ordering::Relaxed) => {
                ggml_gelu_table(x)
            }
            Activation::GeluTanh => {
                // 0.5 x (1 + tanh(sqrt(2/pi) (x + 0.044715 x^3)))
                let inner = x
                    .add(&x.square()?.mul(x)?.mul_scalar(0.044_715)?)?
                    .mul_scalar(0.797_884_560_802_865_4)?;
                x.mul_scalar(0.5)?.mul(&inner.tanh()?.add_scalar(1.0)?)
            }
        }
    }
}

/// Per-head-dimension RMS norm applied to Q and K before RoPE.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QkNorm {
    None,
    /// Weight of size `head_dim`, applied to every head (Qwen3, Gemma 3).
    PerHead,
    /// Weight over the whole projection (`heads * head_dim`), applied before splitting heads (OLMo 2).
    Full,
}

/// Which layers use the sliding window.
#[derive(Debug, Clone, PartialEq)]
pub enum SwaPattern {
    /// Every layer (Mistral).
    All,
    /// Layer `il` is windowed unless `il % n == n - 1` (Gemma 2: `n = 2`, Gemma 3: `n = 6`).
    Modulo(usize),
    Explicit(Vec<bool>),
}

impl SwaPattern {
    pub fn is_swa(&self, il: usize) -> bool {
        match self {
            SwaPattern::All => true,
            SwaPattern::Modulo(n) => *n == 0 || il % n < n - 1,
            SwaPattern::Explicit(v) => v.get(il).copied().unwrap_or(false),
        }
    }
}

/// What a decoder layer contains. Most models use `Full` (attention then FFN); hybrids such as Nemotron-H give
/// each layer exactly one mixer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Mixer {
    Full,
    AttnOnly,
    FfnOnly,
    /// Pure Mamba layer: no FFN.
    Ssm,
    /// Mamba mixer followed by the layer's FFN (Granite-hybrid).
    SsmFfn,
    /// Attention and Mamba-2 side by side on the same normed input, outputs summed, then the FFN (Falcon-H1).
    AttnSsm,
}

/// How an architecture differs from plain Llama (mirrors llama.cpp's per-architecture graph builders:
/// which optional tensors the graph reads, RoPE pairing, norms, activation).
#[derive(Debug, Clone, Copy)]
struct ArchSpec {
    /// RoPE rotates adjacent pairs (llama.cpp `NORM`); weights are re-ordered on load for our half-split RoPE.
    rope_interleaved: bool,
    o_bias: bool,
    ffn_bias: bool,
    ffn_scale: bool,
    out_bias: bool,
    expert_scales: bool,
    norm_topk: bool,
    activation: Activation,
    qk_norm: QkNorm,
    post_norms: bool,
    embed_scale: bool,
    attn_softcap: bool,
    final_softcap: bool,
    /// Default `(window, pattern)` when the file does not say (Gemma 2).
    swa_default: Option<(usize, usize)>,
    /// Gemma 3 style: local layers use `rope.freq_base_swa` without the global scaling.
    local_rope_unscaled: bool,
    /// Attention / FFN inputs are normalized first (false for OLMo 2, which only post-norms).
    pre_norm: bool,
    /// Every n-th layer (`(il + 1) % n == 0`) skips RoPE (SmolLM3).
    no_rope_step: Option<usize>,
    /// Granite-style scalar multipliers: `embedding_scale`, `residual_scale`, `logit_scale`.
    mup: bool,
    /// DeepSeek-style multi-head latent attention (low-rank Q / shared KV latent).
    mla: bool,
    /// MoE renormalization comes from `expert_weights_norm` instead of being fixed by the architecture.
    norm_topk_key: bool,
    /// LayerNorm (mean-subtracting, `layer_norm_epsilon`) instead of RMS norm.
    layer_norm: bool,
    /// The LayerNorms carry a bias tensor.
    norm_bias: bool,
    /// The norms have no learned scale at all (OLMo 1).
    norm_param: bool,
    /// Gated (SwiGLU-style) FFN; false = `down(act(up(x)))`.
    ffn_gated: bool,
    /// Attention and FFN both read the attention-normed input and add into the residual (Phi-2, Command-R).
    parallel: bool,
    /// ... but only when the layer has no `ffn_norm` (StableLM).
    parallel_without_ffn_norm: bool,
    /// Q/K carry optional per-head LayerNorm weights (StableLM, Command-R).
    qk_layernorm_optional: bool,
    /// The attention / FFN / output norms split their width into `attention.group_norm_groups` groups (K2 Horizon).
    norm_groups_key: bool,
    /// Q / K norm weights are one `[heads, head_dim]` set, not one `head_dim` vector shared by the heads (K2 Horizon).
    qk_head_weights: bool,
    /// Attention gate is `log2(1 + 2^x)` and its own `.scale` applies (K2 Horizon).
    attn_gate_log2: bool,
    /// Muse-Spark blocks: per-head weightless attention-output norm, weightless post norms, `alpha`/`beta` residual
    /// weights, an expert-space pre / post norm around the latent MoE, and `post_ffw_norm` applied as a scale before
    /// the final norm. `{arch}.post_norm_epsilon` gives the post-norm epsilon.
    spark_blocks: bool,
    /// The graph applies RoPE without frequency factors, so `rope_freqs.weight` is not used.
    ignore_rope_freqs: bool,
    /// Attention has no causal mask (diffusion language models).
    non_causal: bool,
    /// Layers with `attn_v_exps` route each token to value experts instead of one V projection (K2 Horizon).
    value_experts: bool,
    /// Logits are multiplied by the file's `logit_scale` (Command-R) rather than divided (Granite).
    logit_mul_key: bool,
    /// Tensor-name stem of the FFN's pre-norm (`ffn_norm`, but `post_attention_norm` for Seed-OSS, ...).
    ffn_norm_name: &'static str,
    /// The MoE gating function comes from `expert_gating_func` (DeepSeek-style models).
    gating_key: bool,
    /// Per-head Q/K norms run after RoPE instead of before (Maincoder).
    qk_norm_after_rope: bool,
    /// With a sliding-window pattern, only the windowed layers rotate Q/K (Exaone 4).
    rope_only_swa: bool,
    /// Parallel residual only when the file's `use_parallel_residual` is true (GPT-NeoX)...
    parallel_key: bool,
    /// ... and then the FFN has its own norm applied to the layer input rather than reusing the attention norm.
    parallel_own_ffn_norm: bool,
    /// A second attention-input norm (`attn_norm_2`) when the layer has one (Falcon's new decoder).
    attn_norm_2_optional: bool,
    /// Learned per-head attention-sink logit (`attn_sinks.weight`) joins every softmax (GPT-OSS).
    sinks: bool,
    /// Router and per-expert biases, and OpenAI's clamped SwiGLU in the experts (GPT-OSS).
    moe_biases: bool,
    moe_act: MoeAct,
    /// MoE gating fixed by the architecture rather than read from the file.
    fixed_gating: Option<Gating>,
    /// The graph multiplies Q/K/V with bare matmuls, so their biases and weight scales are ignored (OpenELM, MiMo2).
    ignore_qkv_scale: bool,
    /// The graph adds no Q/K/V biases (weight scales still apply), although the file carries them (Qwen4-Exp).
    ignore_qkv_bias: bool,
    /// llama.cpp switches from interleaved to M-RoPE (half-split for text) when `rope.dimension_sections` is set (GLM4).
    mrope_switch: bool,
    /// With no `expert_gating_func` key the router is sigmoid, not softmax (GLM4-MoE).
    gating_default_sigmoid: bool,
    /// Per-head Q/K norms exist only when the tensors do (GLM4-MoE: `use_qk_norm`).
    qk_norm_optional: bool,
    /// `expert_weights_norm` is read from the file but defaults to true (Mellum).
    topk_key_default: bool,
    /// RoPE always rotates the whole head, whatever `rope.dimension_count` says (PLaMo).
    rope_full_head: bool,
    /// `attn_gate.weight`: sigmoid(W x) multiplies the attention output before `wo` (AfMoE per element, Spark per head).
    attn_gate: bool,
    /// Attention sinks are used only when the layer has the tensor (MiMo2).
    sinks_optional: bool,
    /// `{arch}.attention.value_scale` multiplies the attention block's output (MiMo2).
    value_scale: bool,
    /// The attention output gate uses softplus instead of sigmoid (Laguna).
    attn_gate_softplus: bool,
    /// Experts are chosen on raw router logits + `exp_probs_b`, not on probabilities + bias (Kolibri-1).
    select_on_logits: bool,
    /// The shared expert's matmuls carry `ffn_*_shexp.scale` (Flint).
    shexp_scales: bool,
    /// `attention.rope_pattern` is a per-layer array saying which layers rotate (Flint).
    rope_pattern_array: bool,
    /// Layers without a router (the leading dense ones) always rotate, whatever `rope_only_swa` says (Cohere2-MoE).
    dense_layers_rope: bool,
    /// Multiplies the whole MoE output (routed + shared), e.g. Cohere2-MoE averages the two paths.
    moe_out_scale: Option<f32>,
    /// Grok: `embedding_scale` / `attention.output_scale` keys with the model's built-in defaults.
    grok_scales: bool,
    /// Dense FFN and MoE both run in every layer that has both and their outputs add (Grok, Arctic).
    dense_with_moe: bool,
    /// Arctic's MoE reads the block input through its own `ffn_norm_exps`.
    moe_norm_exps: bool,
    /// Names of the post-attention / post-FFN norm tensors.
    post_norm_names: (&'static str, &'static str),
    /// Text side of a multimodal model that rotates with `ggml_rope_multi`: `Some(false)` = M-RoPE,
    /// `Some(true)` = interleaved M-RoPE (Qwen3-VL). Needs `rope.dimension_sections`.
    mrope: Option<bool>,
    /// `rope.scaling.alpha` (NTK-aware): base *= alpha^(dim / (dim - 2)) (Hunyuan-VL).
    rope_alpha: bool,
    /// Learned absolute position embeddings (`position_embd.weight`) instead of RoPE (GPT-2, StarCoder).
    abs_pos: bool,
    /// The position table may be absent (MPT); RoPE is off either way.
    abs_pos_optional: bool,
    /// ALiBi with a built-in `max_bias` (BLOOM, Refact).
    alibi_fixed: Option<f32>,
    /// ALiBi when `attention.max_alibi_bias` is set (MPT, Jais).
    alibi_key: bool,
    /// LayerNorm over the token embeddings (`token_embd_norm`, BLOOM).
    embed_norm: bool,
    /// Attention logits are scaled by 1 / head_dim instead of 1 / sqrt(head_dim) (Jais).
    attn_scale_inv: bool,
    /// `ffn.act.scales`: the FFN activation is divided by these per-channel scales (MPT).
    ffn_act_scales: bool,
    /// The routing weight multiplies the expert's *input* (Llama 4), not its output.
    weight_before_ffn: bool,
    /// Layers marked by the sliding-window pattern use chunked attention: a token sees only its own
    /// `sliding_window`-sized chunk (Llama 4). A window of 0 turns the whole scheme off.
    chunked_attn: bool,
    /// NoPE layers scale queries by `ln(floor((pos + 1) / floor) + 1) * temperature_scale + 1` (Llama 4).
    nope_temp_floor: Option<usize>,
    /// Q/K get a weightless RMS norm after RoPE, on rotary layers only (Llama 4, except the 128-expert model).
    kq_norm_weightless: bool,
    /// The router reads the block's un-normalized input instead of the FFN-normed one (SmallThinker).
    moe_router_raw: bool,
    /// `no_rope_step` counts layers from 0: layers 0, n, 2n, ... are NoPE (SmallThinker); without a
    /// sliding-window key every layer rotates.
    no_rope_from_zero: bool,
    /// MLA with the plain `key_length` / `value_length` keys, a combined `attn_kv_b` tensor, NEOX rope and a
    /// plain 1/sqrt(head_dim) attention scale (MiniCPM3).
    mla_plain: bool,
    /// MLA layers rotate nothing (Kimi-Linear): `q_pe` / `k_pe` are used as projected.
    mla_no_rope: bool,
    /// MLA rotates adjacent pairs unless the file carries multi-axis `rope.dimension_sections` (BailingMoE3's
    /// text-only files use NORM rope, its VL files MROPE, which pairs like NEOX).
    mla_rope_by_sections: bool,
    /// The MLA output projection is a plain matmul in the graph, so `attn_output.scale` does not apply.
    mla_ignore_o_scale: bool,
    /// Kimi-K3's SITU activation (betas from `{arch}.activation.situ_beta` / `situ_linear_beta`) in every FFN,
    /// and its attention-residual stack (`{arch}.attn_res.block_size`).
    situ: bool,
    /// `{arch}.residual_scale` scales the normalised layer input that serves as each half-layer's residual (MiniMax-01).
    norm_residual: bool,
    /// Hyper-connections: `hyper_connection.count` residual streams mixed by learned per-layer gates (HY-V4).
    hc: bool,
    /// `swiglu_clamp_exp` clamps the gate before SiLU (`silu(min(g, L)) * clamp(u, -L, L)`) instead of after.
    swiglu_clamp_gate_first: bool,
    /// MLA heads with no rotary part at all: `kv_a` is just the latent and `q` is all nope (GLM-5-Next).
    mla_nope_only: bool,
    /// Qwen4-Exp's low-rank hyper-connections: they replace every norm, so the model has none of its own.
    hc_low_rank: bool,
    /// DeepSeek-V4's shared-KV attention with compressed blocks and a grouped output projection.
    dsv4: bool,
    /// Sparse-attention MLA (DeepSeek-V3.2 family): run the lightning indexer and attend only its top-k tokens.
    dsa_indexer: bool,
    /// Layers without indexer tensors reuse the previous indexer layer's selection (GLM-DSA, HY-V4).
    dsa_share: bool,
    /// Indexer heads rotate their last (not first) `n_rot` dimensions.
    dsa_rope_end: bool,
    /// The indexer rotates adjacent pairs (NORM) rather than half-split pairs (NEOX).
    dsa_norm_pairs: bool,
    /// The indexer's LayerNorm uses the RMS epsilon (otherwise 1e-6).
    dsa_ln_rms_eps: bool,
    /// The indexer pools `indexer.kpool` consecutive keys and selects whole pools (GLM-5-Next).
    dsa_kpool: bool,
    /// xIELU feed-forward activation with per-layer parameters from the `xielu.*` arrays (Apertus).
    xielu: bool,
    /// Extra RMS norms over the attention output and the FFN's inner activation (BitNet).
    sub_norms: bool,
    /// Gemma 3n: AltUp streams, LAuReL, per-layer inputs and shared KV (see [`G3nModel`]).
    g3n: bool,
    /// Step 3.5: full-attention layers rotate half of `rope.dimension_count`, sliding-window layers
    /// `rope.dimension_count_swa` dimensions.
    step35_rot: bool,
    /// Gemma 4: sliding-window layers have their own head width (`attention.key_length_swa`) and rotary width
    /// (`rope.dimension_count_swa`).
    swa_geometry: bool,
    /// Gemma 4: per-layer input embeddings gating every layer, a per-layer output scalar, and the last
    /// `attention.shared_kv_layers` layers reusing the K/V of earlier ones.
    g4: bool,
    /// The output gate (`attn_gate.weight`) exists only on some layers.
    attn_gate_optional: bool,
    /// RWKV-6: attention-free layers that carry their own norms and residuals (see [`super::gguf_rwkv`]).
    rwkv6: bool,
    /// RWKV-7: like RWKV-6 with the delta-rule recurrence and a value residual from layer 0.
    rwkv7: bool,
    /// QRWKV6: RWKV-6's time mix as gated linear attention with grouped heads, behind RMS norms and a SwiGLU FFN.
    rwkv6qwen2: bool,
    /// Attention scale fixed by the architecture (llama.cpp ignores `attention.scale` there).
    fixed_attn_scale: Option<f32>,
    /// The per-head Q/K LayerNorms (`qk_layernorm_optional`) use eps 1e-5 and an optional bias (Chameleon).
    qk_layer_norm: bool,
    /// `{arch}.swin_norm`: the block norms move after the attention / FFN outputs (Chameleon 34B).
    swin_norm_key: bool,
    /// Logits of token ids in `[start, end)` are forced to `-f32::MAX` (Chameleon hides its image tokens).
    suppress_logits: Option<(usize, usize)>,
    /// Block-sparse attention indexer on the ordinary attention layers (MiniMax-M3).
    block_indexer: bool,
    /// Per-layer n-gram hash embeddings (Qwen4-Exp).
    ple: bool,
    /// Epsilon of the delta-attention L2 norms when it is not the RMS epsilon.
    kda_l2_eps: Option<f32>,
    /// `{arch}.swiglu_clamp_exp` / `_shexp` limit the experts' and the shared expert's SwiGLU.
    swiglu_clamp: bool,
    /// State-space model (Mamba / Mamba-2): no attention and no FFN, just a selective-scan mixer per layer.
    ssm: Option<SsmKind>,
    /// Per-layer mixers decided by the head / FFN arrays (Nemotron-H): SSM when a layer has no KV heads and no FFN,
    /// attention-only when it has no FFN, FFN-only otherwise.
    hybrid_ssm: Option<SsmKind>,
    /// In a hybrid every layer keeps its FFN; layers without KV heads swap attention for the SSM (Granite-hybrid).
    hybrid_with_ffn: bool,
    /// Every layer runs attention and the SSM in parallel (Falcon-H1).
    hybrid_parallel: bool,
    /// Name of the final norm tensor (LFM2 calls it `token_embd_norm`).
    output_norm_name: &'static str,
    /// `attn_q` also carries a per-head output gate: rows are `[q | gate]` for every head (Qwen3.5).
    q_gate_packed: bool,
    /// Layers follow `attention.recurrent_layers` or `full_attention_interval`: recurrent layers are gated delta nets.
    gdn_layers: bool,
    /// The gated-delta-net projections carry per-tensor `*.scale` multipliers (Qwen3.5; Qwen3-Next ignores them).
    gdn_scales: bool,
    /// DeepSeek sparse attention: a lightning indexer keeps the best `indexer.top_k` tokens. Up to `top_k` positions
    /// that is dense MLA, which is all this runner implements, so longer contexts are refused.
    dsa: bool,
    /// Epsilon of the post-attention / post-FFN norms when it differs from the layer-norm epsilon (Muse-Glimmer: 1e-8).
    post_norm_eps: Option<f64>,
    /// A weightless RMS norm over the token embeddings before the first layer (Muse-Glimmer).
    embed_rms: bool,
    /// The attention output gate has one bias per head (`attn_gate.bias` is `[heads]`), shared across the head dim (Beam).
    attn_gate_head_bias: bool,
    /// NoPE query temperature `(1 + (pos + 1) / length) ^ scale` instead of the log-floor form; `length` is
    /// `attention.temperature_length` (Beam).
    nope_temp_pow: bool,
    /// Residual adds are weighted: `h / sqrt(i) + r * sqrt((i - 1) / i)` with `i = 2 * (layer + 1)` after attention and
    /// `i + 1` after the FFN (Beam).
    weighted_residual: bool,
    /// `attn_q_norm.weight` is a `[heads]` gain on a weightless norm; K has none (Talkie).
    q_head_gain: bool,
    /// Only `ffn_down.scale` is applied; the graph passes no gate / up scales (Talkie).
    ffn_down_scale_only: bool,
    /// Every layer adds `layer_output_scale * embedding` to its output (Talkie).
    embed_skip: bool,
}

const BASE_SPEC: ArchSpec = ArchSpec {
    rope_interleaved: false,
    o_bias: true,
    ffn_bias: false,
    ffn_scale: false,
    out_bias: false,
    expert_scales: false,
    norm_topk: true,
    activation: Activation::Silu,
    qk_norm: QkNorm::None,
    post_norms: false,
    embed_scale: false,
    attn_softcap: false,
    final_softcap: false,
    swa_default: None,
    local_rope_unscaled: false,
    pre_norm: true,
    no_rope_step: None,
    mup: false,
    mla: false,
    norm_topk_key: false,
    layer_norm: false,
    norm_bias: false,
    norm_param: true,
    ffn_gated: true,
    parallel: false,
    parallel_without_ffn_norm: false,
    qk_layernorm_optional: false,
    norm_groups_key: false,
    qk_head_weights: false,
    attn_gate_log2: false,
    value_experts: false,
    spark_blocks: false,
    ignore_rope_freqs: false,
    non_causal: false,
    logit_mul_key: false,
    ffn_norm_name: "ffn_norm",
    gating_key: false,
    qk_norm_after_rope: false,
    rope_only_swa: false,
    parallel_key: false,
    parallel_own_ffn_norm: false,
    attn_norm_2_optional: false,
    sinks: false,
    moe_biases: false,
    moe_act: MoeAct::Silu,
    fixed_gating: None,
    ignore_qkv_scale: false,
    ignore_qkv_bias: false,
    mrope_switch: false,
    gating_default_sigmoid: false,
    qk_norm_optional: false,
    topk_key_default: false,
    rope_full_head: false,
    attn_gate: false,
    sinks_optional: false,
    value_scale: false,
    attn_gate_softplus: false,
    select_on_logits: false,
    shexp_scales: false,
    rope_pattern_array: false,
    dense_layers_rope: false,
    moe_out_scale: None,
    grok_scales: false,
    dense_with_moe: false,
    moe_norm_exps: false,
    post_norm_names: ("post_attention", "post_ffw"),
    mrope: None,
    rope_alpha: false,
    abs_pos: false,
    abs_pos_optional: false,
    alibi_fixed: None,
    alibi_key: false,
    embed_norm: false,
    attn_scale_inv: false,
    ffn_act_scales: false,
    weight_before_ffn: false,
    chunked_attn: false,
    nope_temp_floor: None,
    kq_norm_weightless: false,
    moe_router_raw: false,
    no_rope_from_zero: false,
    mla_plain: false,
    mla_no_rope: false,
    mla_rope_by_sections: false,
    mla_ignore_o_scale: false,
    situ: false,
    norm_residual: false,
    hc: false,
    swiglu_clamp_gate_first: false,
    mla_nope_only: false,
    hc_low_rank: false,
    dsv4: false,
    dsa_indexer: false,
    dsa_share: false,
    dsa_rope_end: false,
    dsa_norm_pairs: false,
    dsa_ln_rms_eps: false,
    dsa_kpool: false,
    xielu: false,
    sub_norms: false,
    g3n: false,
    rwkv6: false,
    rwkv7: false,
    rwkv6qwen2: false,
    step35_rot: false,
    swa_geometry: false,
    g4: false,
    attn_gate_optional: false,
    fixed_attn_scale: None,
    qk_layer_norm: false,
    swin_norm_key: false,
    suppress_logits: None,
    block_indexer: false,
    ple: false,
    kda_l2_eps: None,
    swiglu_clamp: false,
    ssm: None,
    hybrid_ssm: None,
    hybrid_with_ffn: false,
    hybrid_parallel: false,
    output_norm_name: "output_norm",
    q_gate_packed: false,
    gdn_layers: false,
    gdn_scales: false,
    dsa: false,
    post_norm_eps: None,
    embed_rms: false,
    q_head_gain: false,
    ffn_down_scale_only: false,
    embed_skip: false,
    attn_gate_head_bias: false,
    nope_temp_pow: false,
    weighted_residual: false,
};

fn arch_spec(arch: &str) -> Option<ArchSpec> {
    let b = BASE_SPEC;
    Some(match arch {
        "llama" => ArchSpec {
            rope_interleaved: true,
            ffn_bias: true,
            ffn_scale: true,
            expert_scales: true,
            ..b
        },
        "mistral3" => ArchSpec {
            rope_interleaved: true,
            ffn_bias: true,
            ffn_scale: true,
            expert_scales: true,
            ..b
        },
        "internlm2" => ArchSpec {
            rope_interleaved: true,
            ..b
        },
        "exaone" => b,
        "smollm3" => ArchSpec {
            rope_interleaved: true,
            ffn_bias: true,
            no_rope_step: Some(4),
            ..b
        },
        "granite" | "granitemoe" => ArchSpec {
            rope_interleaved: true,
            ffn_bias: true,
            mup: true,
            ..b
        },
        "olmo2" => ArchSpec {
            o_bias: false,
            qk_norm: QkNorm::Full,
            post_norms: true,
            pre_norm: false,
            swa_default: Some((0, 4)),
            local_rope_unscaled: true,
            ..b
        },
        "xverse" | "baichuan" => ArchSpec {
            rope_interleaved: true,
            ..b
        },
        "plamo" => ArchSpec {
            o_bias: false,
            parallel_without_ffn_norm: true,
            rope_full_head: true,
            ..b
        },
        "mellum" => ArchSpec {
            qk_norm: QkNorm::PerHead,
            swa_default: Some((0, 4)),
            local_rope_unscaled: true,
            norm_topk_key: true,
            topk_key_default: true,
            gating_key: true,
            expert_scales: true,
            ..b
        },
        "maple" => ArchSpec {
            qk_norm: QkNorm::PerHead,
            swa_default: Some((0, 2)),
            rope_only_swa: true,
            ..b
        },
        "minimax-m2" => ArchSpec {
            o_bias: false,
            qk_norm: QkNorm::Full,
            gating_key: true,
            ..b
        },
        "exaone-moe" => ArchSpec {
            o_bias: false,
            qk_norm: QkNorm::PerHead,
            swa_default: Some((128, 4)),
            rope_only_swa: true,
            norm_topk_key: true,
            gating_key: true,
            ..b
        },
        "afmoe" => ArchSpec {
            o_bias: false,
            qk_norm: QkNorm::PerHead,
            post_norms: true,
            embed_scale: true,
            attn_gate: true,
            no_rope_step: Some(4),
            swa_default: Some((0, 4)),
            gating_key: true,
            gating_default_sigmoid: true,
            norm_topk_key: true,
            ..b
        },
        "spark2_5" => ArchSpec {
            o_bias: false,
            activation: Activation::GeluTanh,
            attn_gate: true,
            swa_default: Some((0, 4)),
            ..b
        },
        "mimo2" => ArchSpec {
            o_bias: false,
            ffn_bias: true,
            fixed_gating: Some(Gating::Sigmoid),
            sinks_optional: true,
            value_scale: true,
            ignore_qkv_scale: true,
            swa_default: Some((0, 2)),
            local_rope_unscaled: true,
            ..b
        },
        "laguna" => ArchSpec {
            o_bias: false,
            qk_norm: QkNorm::PerHead,
            attn_gate: true,
            attn_gate_softplus: true,
            gating_key: true,
            gating_default_sigmoid: true,
            norm_topk_key: true,
            swa_default: Some((0, 4)),
            local_rope_unscaled: true,
            ..b
        },
        "kolibri1" => ArchSpec {
            o_bias: false,
            qk_norm: QkNorm::PerHead,
            post_norms: true,
            rope_only_swa: true,
            gating_key: true,
            norm_topk_key: true,
            select_on_logits: true,
            swa_default: Some((0, 2)),
            ..b
        },
        "flint" => ArchSpec {
            qk_norm: QkNorm::PerHead,
            sinks_optional: true,
            rope_pattern_array: true,
            gating_key: true,
            norm_topk_key: true,
            ffn_scale: true,
            expert_scales: true,
            shexp_scales: true,
            swa_default: Some((0, 2)),
            local_rope_unscaled: true,
            ..b
        },
        "cohere2moe" => ArchSpec {
            rope_interleaved: true,
            parallel: true,
            logit_mul_key: true,
            ffn_scale: true,
            expert_scales: true,
            shexp_scales: true,
            rope_only_swa: true,
            dense_layers_rope: true,
            moe_out_scale: Some(0.5),
            gating_key: true,
            gating_default_sigmoid: true,
            norm_topk_key: true,
            swa_default: Some((0, 4)),
            ..b
        },
        "grok" => ArchSpec {
            rope_interleaved: false,
            attn_softcap: true,
            grok_scales: true,
            logit_mul_key: true,
            post_norms: true,
            post_norm_names: ("attn_output", "layer_output"),
            dense_with_moe: true,
            moe_out_scale: Some(std::f32::consts::FRAC_1_SQRT_2),
            moe_act: MoeAct::Gelu,
            activation: Activation::GeluTanh,
            ..b
        },
        "arctic" => ArchSpec {
            rope_interleaved: true,
            o_bias: false,
            dense_with_moe: true,
            moe_norm_exps: true,
            ..b
        },
        "paddleocr" => ArchSpec {
            mrope: Some(false),
            ..b
        },
        "hunyuan_vl" => ArchSpec {
            mrope_switch: true,
            qk_norm: QkNorm::PerHead,
            qk_norm_after_rope: true,
            rope_alpha: true,
            ..b
        },
        "qwen2vl" => ArchSpec {
            mrope: Some(false),
            ..b
        },
        "qwen3vl" => ArchSpec {
            mrope: Some(true),
            qk_norm: QkNorm::PerHead,
            ..b
        },
        "qwen3vlmoe" => ArchSpec {
            mrope: Some(true),
            qk_norm: QkNorm::PerHead,
            ..b
        },
        "llama4" => ArchSpec {
            rope_interleaved: true,
            ffn_bias: true,
            fixed_gating: Some(Gating::Sigmoid),
            norm_topk: false,
            weight_before_ffn: true,
            chunked_attn: true,
            nope_temp_floor: Some(8192),
            kq_norm_weightless: true,
            qk_norm: QkNorm::PerHead,
            qk_norm_after_rope: true,
            no_rope_step: Some(4),
            swa_default: Some((8192, 4)),
            ..b
        },
        "hy_v3" => ArchSpec {
            qk_norm: QkNorm::PerHead,
            ffn_bias: true,
            ffn_scale: true,
            expert_scales: true,
            shexp_scales: true,
            gating_key: true,
            gating_default_sigmoid: true,
            norm_topk_key: true,
            ..b
        },
        "plamo3" => ArchSpec {
            o_bias: false,
            qk_norm: QkNorm::PerHead,
            post_norms: true,
            ignore_qkv_scale: true,
            swa_default: Some((0, 8)),
            ..b
        },
        "smallthinker" => ArchSpec {
            moe_router_raw: true,
            moe_act: MoeAct::Relu,
            no_rope_from_zero: true,
            no_rope_step: Some(4),
            gating_key: true,
            swa_default: Some((4096, 2)),
            ..b
        },
        "grovemoe" => ArchSpec {
            qk_norm: QkNorm::PerHead,
            ..b
        },
        "minicpm3" => ArchSpec {
            o_bias: false,
            mla: true,
            mla_plain: true,
            mup: true,
            ..b
        },
        "maion-coder" => ArchSpec {
            o_bias: false,
            qk_norm: QkNorm::PerHead,
            post_norms: true,
            rope_only_swa: true,
            swa_default: Some((0, 4)),
            ..b
        },
        "qwen35moe" => ArchSpec {
            o_bias: false,
            qk_norm: QkNorm::PerHead,
            mrope: Some(true),
            q_gate_packed: true,
            gdn_layers: true,
            gdn_scales: true,
            hybrid_ssm: Some(SsmKind::GatedDelta),
            expert_scales: true,
            shexp_scales: true,
            ffn_norm_name: "post_attention_norm",
            ..b
        },
        "qwen3next" => ArchSpec {
            gdn_scales: true,
            o_bias: false,
            qk_norm: QkNorm::PerHead,
            q_gate_packed: true,
            gdn_layers: true,
            hybrid_ssm: Some(SsmKind::GatedDelta),
            expert_scales: true,
            shexp_scales: true,
            ffn_norm_name: "post_attention_norm",
            ..b
        },
        "qwen35" => ArchSpec {
            o_bias: false,
            qk_norm: QkNorm::PerHead,
            mrope: Some(true),
            q_gate_packed: true,
            gdn_layers: true,
            gdn_scales: true,
            hybrid_ssm: Some(SsmKind::GatedDelta),
            ffn_scale: true,
            ffn_norm_name: "post_attention_norm",
            ..b
        },
        "lfm2" | "lfm2moe" => ArchSpec {
            expert_scales: true,
            ffn_scale: true,
            o_bias: false,
            qk_norm: QkNorm::PerHead,
            hybrid_ssm: Some(SsmKind::ShortConv),
            hybrid_with_ffn: true,
            output_norm_name: "token_embd_norm",
            gating_key: true,
            swa_default: Some((0, 0)),
            ..b
        },
        "jamba" => ArchSpec {
            o_bias: false,
            norm_topk: false,
            no_rope_step: Some(1),
            hybrid_ssm: Some(SsmKind::Mamba),
            hybrid_with_ffn: true,
            ..b
        },
        "falcon-h1" => ArchSpec {
            o_bias: false,
            ffn_bias: true,
            hybrid_ssm: Some(SsmKind::Mamba2),
            hybrid_parallel: true,
            ..b
        },
        "granitehybrid" => ArchSpec {
            rope_interleaved: true,
            ffn_bias: true,
            mup: true,
            hybrid_ssm: Some(SsmKind::Mamba2),
            hybrid_with_ffn: true,
            ..b
        },
        "muse-glimmer" => ArchSpec {
            o_bias: false,
            qk_norm: QkNorm::PerHead,
            attn_gate: true,
            post_norms: true,
            post_norm_eps: Some(1e-8),
            embed_rms: true,
            rope_only_swa: true,
            logit_mul_key: true,
            final_softcap: true,
            swa_default: Some((0, 4)),
            ..b
        },
        // Sliding-window RoPE layers and NoPE full layers (power-law query temperature), per-head gate bias,
        // sandwich norms, group-routed sigmoid MoE after the dense lead, and a weighted residual instead of `x + f(x)`.
        "beam" => ArchSpec {
            o_bias: false,
            qk_norm: QkNorm::PerHead,
            attn_gate: true,
            attn_gate_head_bias: true,
            post_norms: true,
            rope_only_swa: true,
            nope_temp_floor: Some(2048),
            nope_temp_pow: true,
            weighted_residual: true,
            gating_key: true,
            gating_default_sigmoid: true,
            norm_topk_key: true,
            final_softcap: true,
            swa_default: Some((0, 4)),
            ..b
        },
        // Weightless norms everywhere, Q/K norm after RoPE (K ungained, Q with a per-head gain), a normalized
        // embedding that every layer adds back scaled by `layer_output_scale`, and a logit scale.
        // Low / high stacks of weightless-norm gated-attention layers re-run with shared weights (see `hrm`).
        "hrm_text" => ArchSpec {
            o_bias: false,
            norm_param: false,
            attn_gate: true,
            mup: true,
            ..b
        },
        // Gated attention (log2-softplus gate) with per-head Q/K norm weights, grouped main norms, a sigmoid-routed
        // MoE with a shared expert, and on MoE layers an optional routed-value attention (MoVA).
        "k2-horizon" => ArchSpec {
            o_bias: false,
            qk_norm: QkNorm::PerHead,
            qk_head_weights: true,
            attn_gate: true,
            attn_gate_log2: true,
            norm_groups_key: true,
            value_experts: true,
            gating_key: true,
            gating_default_sigmoid: true,
            norm_topk_key: true,
            ..b
        },
        // Sliding-window RoPE layers and NoPE global layers, gated attention with a per-head output norm,
        // alpha / beta residuals, and a latent sigmoid MoE on every layer (see `spark_blocks`).
        "muse-spark" => ArchSpec {
            o_bias: false,
            qk_norm: QkNorm::PerHead,
            attn_gate: true,
            spark_blocks: true,
            embed_rms: true,
            rope_only_swa: true,
            logit_mul_key: true,
            final_softcap: true,
            swa_default: Some((0, 4)),
            gating_key: true,
            gating_default_sigmoid: true,
            norm_topk_key: true,
            ..b
        },
        // CogVLM's text path: fused QKV, a plain Llama block; the vision-expert tensors are unused for token input.
        // Diffusion language models: a Qwen2 / Llama / Qwen3-MoE block stack with bidirectional attention.
        "dream" => ArchSpec {
            non_causal: true,
            ..b
        },
        "llada" => ArchSpec {
            non_causal: true,
            rope_interleaved: true,
            o_bias: false,
            ignore_rope_freqs: true,
            ..b
        },
        "llada-moe" => ArchSpec {
            non_causal: true,
            qk_norm: QkNorm::PerHead,
            norm_topk: false,
            ..b
        },
        "rnd1" => ArchSpec {
            non_causal: true,
            qk_norm: QkNorm::PerHead,
            ..b
        },
        // Mamba-2-style layers and gated-free attention layers, both followed by a fused SwiGLU FFN, each with
        // a post norm; attention uses fused QKV and per-head Q / K norm weights.
        "plamo2" => ArchSpec {
            hybrid_ssm: Some(SsmKind::Plamo2),
            hybrid_with_ffn: true,
            o_bias: false,
            qk_norm: QkNorm::PerHead,
            qk_head_weights: true,
            post_norms: true,
            ignore_qkv_scale: true,
            ..b
        },
        "cogvlm" => ArchSpec {
            o_bias: false,
            ignore_qkv_scale: true,
            ignore_rope_freqs: true,
            ..b
        },
        // Qwen3-TTS runs Qwen3-VL's graph.
        "qwen3tts" => ArchSpec {
            mrope: Some(true),
            qk_norm: QkNorm::PerHead,
            ..b
        },
        // Pocket-TTS backbone: LayerNorm with bias, interleaved RoPE, ungated GELU FFN without biases.
        "pockettts" => ArchSpec {
            o_bias: false,
            rope_interleaved: true,
            layer_norm: true,
            norm_bias: true,
            ffn_gated: false,
            activation: Activation::GeluTanh,
            ..b
        },
        "talkie" => ArchSpec {
            o_bias: false,
            norm_param: false,
            qk_norm: QkNorm::PerHead,
            qk_norm_after_rope: true,
            kq_norm_weightless: true,
            q_head_gain: true,
            ffn_down_scale_only: true,
            embed_skip: true,
            embed_rms: true,
            logit_mul_key: true,
            ..b
        },
        "nemotron_h" => ArchSpec {
            hybrid_ssm: Some(SsmKind::Mamba2),
            ffn_gated: false,
            activation: Activation::ReluSqr,
            ffn_bias: true,
            ffn_scale: true,
            no_rope_step: Some(1),
            ..b
        },
        // Same layers, but the FFN layers may be sparse: ungated ReLU² experts behind a sigmoid router with a
        // selection bias, an optional latent projection, and an ungated shared expert on the full-width input.
        "nemotron_h_moe" => ArchSpec {
            hybrid_ssm: Some(SsmKind::Mamba2),
            ffn_gated: false,
            activation: Activation::ReluSqr,
            ffn_bias: true,
            ffn_scale: true,
            no_rope_step: Some(1),
            moe_act: MoeAct::ReluSqr,
            fixed_gating: Some(Gating::Sigmoid),
            norm_topk_key: true,
            expert_scales: true,
            shexp_scales: true,
            ..b
        },
        "mamba" => ArchSpec {
            ssm: Some(SsmKind::Mamba),
            ..b
        },
        "mamba2" => ArchSpec {
            ssm: Some(SsmKind::Mamba2),
            ..b
        },
        "qwen" => ArchSpec { o_bias: false, ..b },
        "deepseek" => ArchSpec {
            rope_interleaved: true,
            norm_topk: false,
            ..b
        },
        "glm4moe" => ArchSpec {
            o_bias: false,
            qk_norm: QkNorm::PerHead,
            qk_norm_optional: true,
            norm_topk_key: true,
            gating_key: true,
            gating_default_sigmoid: true,
            ffn_norm_name: "post_attention_norm",
            ..b
        },
        "glm4" => ArchSpec {
            rope_interleaved: true,
            o_bias: false,
            post_norms: true,
            mrope_switch: true,
            ..b
        },
        // Sigmoid-routed MoE with a shared expert and a head-wise attention gate; head counts, rotary widths and the
        // sliding-window pattern differ per layer. The NextN draft layers are not run.
        "step35" => ArchSpec {
            o_bias: false,
            qk_norm: QkNorm::PerHead,
            qk_norm_optional: true,
            attn_gate: true,
            attn_gate_optional: true,
            norm_topk_key: true,
            gating_key: true,
            gating_default_sigmoid: true,
            swiglu_clamp: true,
            step35_rot: true,
            swa_default: Some((0, 4)),
            local_rope_unscaled: true,
            ..b
        },
        "chatglm" => ArchSpec {
            rope_interleaved: true,
            o_bias: false,
            ..b
        },
        "deci" => ArchSpec {
            rope_interleaved: true,
            ffn_bias: true,
            ..b
        },
        "openelm" => ArchSpec {
            qk_norm: QkNorm::PerHead,
            ignore_qkv_scale: true,
            ..b
        },
        "cohere2" => ArchSpec {
            rope_interleaved: true,
            layer_norm: true,
            parallel: true,
            logit_mul_key: true,
            ffn_scale: true,
            rope_only_swa: true,
            swa_default: Some((0, 4)),
            ..b
        },
        "granite_swa" => ArchSpec {
            rope_interleaved: true,
            ffn_bias: true,
            mup: true,
            sinks: true,
            swa_default: Some((0, 1)),
            ..b
        },
        "gpt-oss" => ArchSpec {
            sinks: true,
            moe_biases: true,
            moe_act: MoeAct::SwigluOai,
            fixed_gating: Some(Gating::SoftmaxWeight),
            norm_topk: false,
            ffn_norm_name: "post_attention_norm",
            swa_default: Some((0, 2)),
            ..b
        },
        "dbrx" => ArchSpec {
            layer_norm: true,
            o_bias: false,
            ffn_norm_name: "attn_output_norm",
            ..b
        },
        "hunyuan-moe" => ArchSpec {
            qk_norm: QkNorm::PerHead,
            qk_norm_after_rope: true,
            ..b
        },
        "dots1" => ArchSpec {
            qk_norm: QkNorm::PerHead,
            norm_topk_key: true,
            gating_key: true,
            ..b
        },
        "bailingmoe" => ArchSpec {
            rope_interleaved: true,
            norm_topk_key: true,
            ..b
        },
        "bailingmoe2" => ArchSpec {
            qk_norm: QkNorm::PerHead,
            norm_topk_key: true,
            gating_key: true,
            ..b
        },
        // Ernie's graph never reads the attention-output bias even when the file carries one.
        "ernie4_5" | "ernie4_5-moe" => ArchSpec {
            rope_interleaved: true,
            o_bias: false,
            ..b
        },
        "nanbeige" => ArchSpec {
            rope_interleaved: true,
            ffn_bias: true,
            ffn_scale: true,
            expert_scales: true,
            ..b
        },
        "arcee" => ArchSpec {
            rope_interleaved: true,
            ffn_gated: false,
            activation: Activation::ReluSqr,
            ..b
        },
        // MLA with a single Q projection (no low-rank Q), plain `1/sqrt(head_dim)` scaling and an ungated ReLU^2 FFN.
        "plm" => ArchSpec {
            o_bias: false,
            mla: true,
            mla_plain: true,
            rope_interleaved: true,
            ffn_gated: false,
            activation: Activation::ReluSqr,
            ..b
        },
        // Llama with per-head LayerNorm Q/K norms (torch eps), an optional post-norm layout (`swin_norm`) and the image
        // tokens' logits hidden. In the swin layout the FFN reads the residual sum, as in the reference implementation.
        "chameleon" => ArchSpec {
            rope_interleaved: true,
            qk_layernorm_optional: true,
            qk_layer_norm: true,
            swin_norm_key: true,
            post_norm_names: ("attn", "ffn"),
            suppress_logits: Some((4, 8196)),
            ..b
        },
        // RWKV-6: no attention and no FFN tensors; every layer is a time mix + channel mix on LayerNorms.
        "rwkv6" => ArchSpec {
            layer_norm: true,
            norm_bias: true,
            embed_norm: true,
            rwkv6: true,
            ..b
        },
        "rwkv7" => ArchSpec {
            layer_norm: true,
            norm_bias: true,
            embed_norm: true,
            rwkv7: true,
            ..b
        },
        // RWKV-7 time mix behind RMS norms with a SwiGLU FFN and no embedding norm (the distilled ARWKV models).
        "arwkv7" => ArchSpec {
            rwkv7: true,
            ..b
        },
        // RWKV-6 time mix as gated linear attention with grouped heads (the distilled QRWKV models).
        "rwkv6qwen2" => ArchSpec {
            rwkv6qwen2: true,
            ..b
        },
        "bitnet" => ArchSpec {
            sub_norms: true,
            ..b
        },
        "apertus" => ArchSpec {
            qk_norm: QkNorm::PerHead,
            ffn_gated: false,
            xielu: true,
            ..b
        },
        "maincoder" => ArchSpec {
            rope_interleaved: true,
            qk_norm: QkNorm::PerHead,
            qk_norm_after_rope: true,
            ..b
        },
        "seed_oss" => ArchSpec {
            ffn_norm_name: "post_attention_norm",
            ..b
        },
        "exaone4" => ArchSpec {
            swa_default: Some((4096, 4)),
            o_bias: false,
            qk_norm: QkNorm::PerHead,
            post_norms: true,
            pre_norm: false,
            rope_only_swa: true,
            ..b
        },
        "nemotron" | "jais2" => ArchSpec {
            layer_norm: true,
            norm_bias: true,
            ffn_gated: false,
            activation: Activation::ReluSqr,
            ffn_bias: true,
            ..b
        },
        "codeshell" => ArchSpec {
            layer_norm: true,
            norm_bias: true,
            ffn_gated: false,
            activation: Activation::GeluTanh,
            ffn_bias: true,
            ..b
        },
        "gptneox" => ArchSpec {
            layer_norm: true,
            norm_bias: true,
            ffn_gated: false,
            activation: Activation::GeluTanh,
            ffn_bias: true,
            parallel_key: true,
            parallel_own_ffn_norm: true,
            ..b
        },
        "falcon" => ArchSpec {
            layer_norm: true,
            norm_bias: true,
            o_bias: false,
            ffn_gated: false,
            activation: Activation::GeluTanh,
            parallel: true,
            attn_norm_2_optional: true,
            ..b
        },
        // MiniCPM runs the Granite graph (with built-in defaults for its scales when the file omits them).
        "minicpm" => ArchSpec {
            rope_interleaved: true,
            ffn_bias: true,
            mup: true,
            ..b
        },
        "phimoe" => ArchSpec {
            out_bias: true,
            norm_bias: true,
            ..b
        },
        "pangu-embedded" | "hunyuan-dense" => b,
        "orion" => ArchSpec {
            layer_norm: true,
            norm_bias: true,
            o_bias: false,
            ..b
        },
        "stablelm" => ArchSpec {
            layer_norm: true,
            norm_bias: true,
            o_bias: false,
            parallel_without_ffn_norm: true,
            qk_layernorm_optional: true,
            ..b
        },
        "phi2" => ArchSpec {
            layer_norm: true,
            norm_bias: true,
            ffn_gated: false,
            activation: Activation::GeluTanh,
            ffn_bias: true,
            out_bias: true,
            parallel: true,
            ..b
        },
        "bloom" => ArchSpec {
            layer_norm: true,
            norm_bias: true,
            ffn_gated: false,
            activation: Activation::GeluTanh,
            ffn_bias: true,
            alibi_fixed: Some(8.0),
            embed_norm: true,
            ..b
        },
        "mpt" => ArchSpec {
            layer_norm: true,
            norm_bias: true,
            ffn_gated: false,
            activation: Activation::GeluTanh,
            ffn_bias: true,
            alibi_key: true,
            abs_pos: true,
            abs_pos_optional: true,
            qk_norm: QkNorm::Full,
            qk_norm_optional: true,
            ffn_act_scales: true,
            ..b
        },
        "refact" => ArchSpec {
            o_bias: false,
            alibi_fixed: Some(8.0),
            ..b
        },
        "jais" => ArchSpec {
            layer_norm: true,
            norm_bias: true,
            ffn_bias: true,
            alibi_key: true,
            attn_scale_inv: true,
            ..b
        },
        "gpt2" | "starcoder" => ArchSpec {
            layer_norm: true,
            norm_bias: true,
            ffn_gated: false,
            activation: Activation::GeluTanh,
            ffn_bias: true,
            abs_pos: true,
            // GPT-2's graph adds `output.bias` when the file has one (StarCoder's does not).
            out_bias: arch == "gpt2",
            ..b
        },
        "starcoder2" => ArchSpec {
            layer_norm: true,
            norm_bias: true,
            ffn_gated: false,
            activation: Activation::GeluTanh,
            ffn_bias: true,
            ..b
        },
        "command-r" => ArchSpec {
            rope_interleaved: true,
            layer_norm: true,
            parallel: true,
            qk_layernorm_optional: true,
            logit_mul_key: true,
            ..b
        },
        "olmo" => ArchSpec {
            rope_interleaved: true,
            layer_norm: true,
            norm_param: false,
            o_bias: false,
            ..b
        },
        "deepseek32" => ArchSpec {
            dsa_indexer: true,
            rope_interleaved: true,
            o_bias: false,
            mla: true,
            norm_topk_key: true,
            dsa: true,
            ffn_scale: true,
            expert_scales: true,
            shexp_scales: true,
            ..b
        },
        // MLA with a lightning indexer (dense for contexts within `top_k`), a normed rotary key, a per-head output gate
        // and sliding-window layers; the window layers' MLA geometry must equal the full layers' (checked at load).
        "dots3note" => ArchSpec {
            dsa_indexer: true,
            rope_interleaved: true,
            o_bias: false,
            mla: true,
            norm_topk_key: true,
            dsa: true,
            attn_gate: true,
            swa_default: Some((0, 2)),
            ffn_scale: true,
            expert_scales: true,
            shexp_scales: true,
            ..b
        },
        "glm-dsa" => ArchSpec {
            dsa_indexer: true,
            dsa_share: true,
            dsa_norm_pairs: true,
            rope_interleaved: true,
            o_bias: false,
            mla: true,
            norm_topk_key: true,
            dsa: true,
            gating_default_sigmoid: true,
            ffn_scale: true,
            expert_scales: true,
            shexp_scales: true,
            ..b
        },
        "deepseek2" | "mistral4" => ArchSpec {
            rope_interleaved: true,
            o_bias: false,
            mla: true,
            norm_topk_key: true,
            ..b
        },
        // MiniMax-M2 attention with per-head QK norm, SwiGLU-OAI FFNs, shared + sigmoid-routed experts and block-sparse
        // attention that is exact (dense) while the context fits in `top_k` blocks; longer contexts are refused.
        // MLA with attention sinks and a head gate, lightning-indexer sparse attention (dense within `top_k`), the iHC
        // multi-stream residual and gate-first clamped experts.
        // Kimi delta-attention layers and rope-free nope-only MLA layers (dense within the k-pool indexer's `top_k`),
        // manifold-constrained hyper-connections averaged at the output, and gate-first clamped sigmoid MoE.
        "glm5-next" => ArchSpec {
            o_bias: false,
            mla: true,
            mla_nope_only: true,
            mla_ignore_o_scale: true,
            gdn_layers: true,
            hybrid_ssm: Some(SsmKind::Kda),
            kda_l2_eps: Some(1e-6),
            dsa: true,
            dsa_indexer: true,
            dsa_share: true,
            dsa_kpool: true,
            hc: true,
            norm_topk_key: true,
            gating_default_sigmoid: true,
            swiglu_clamp: true,
            swiglu_clamp_gate_first: true,
            ..b
        },
        // Shared-KV attention with compressed blocks and a grouped output projection, iHC with Sinkhorn mixing, and
        // sqrt-softplus routed experts with gate-first clamped SwiGLU. Hash-routed layers are refused.
        "deepseek4" => ArchSpec {
            o_bias: false,
            dsv4: true,
            norm_topk_key: true,
            gating_key: true,
            hc: true,
            swiglu_clamp: true,
            swiglu_clamp_gate_first: true,
            ..b
        },
        "hy_v4" => ArchSpec {
            dsa_indexer: true,
            dsa_share: true,
            dsa_norm_pairs: true,
            dsa_rope_end: true,
            dsa_ln_rms_eps: true,
            rope_interleaved: true,
            o_bias: false,
            mla: true,
            mla_ignore_o_scale: true,
            attn_gate: true,
            norm_topk_key: true,
            gating_key: true,
            dsa: true,
            hc: true,
            swiglu_clamp: true,
            swiglu_clamp_gate_first: true,
            ..b
        },
        "minimax-m3" => ArchSpec {
            ffn_scale: true,
            expert_scales: true,
            shexp_scales: true,
            o_bias: false,
            qk_norm: QkNorm::PerHead,
            gating_key: true,
            norm_topk_key: true,
            dsa: true,
            block_indexer: true,
            activation: Activation::SwigluOai,
            moe_act: MoeAct::SwigluOai,
            ..b
        },
        "minimax-01" => ArchSpec {
            o_bias: false,
            gdn_layers: true,
            hybrid_ssm: Some(SsmKind::Lightning),
            norm_residual: true,
            ..b
        },
        // Gated-DeltaNet / gated-attention hybrid with low-rank hyper-connections in place of every norm, a sigmoid
        // output gate on the delta-net, per-layer n-gram hash embeddings and QSA attention (dense within `top_k`).
        "qwen4exp" => ArchSpec {
            o_bias: false,
            qk_norm: QkNorm::PerHead,
            mrope: Some(true),
            q_gate_packed: true,
            gdn_layers: true,
            gdn_scales: true,
            hybrid_ssm: Some(SsmKind::GatedDelta),
            expert_scales: true,
            shexp_scales: true,
            pre_norm: false,
            dsa: true,
            block_indexer: true,
            ignore_qkv_bias: true,
            hc: true,
            hc_low_rank: true,
            ple: true,
            ..b
        },
        "kimi-k3" => ArchSpec {
            o_bias: false,
            mla: true,
            mla_no_rope: true,
            mla_ignore_o_scale: true,
            attn_gate: true,
            gdn_layers: true,
            hybrid_ssm: Some(SsmKind::Kda),
            situ: true,
            norm_topk_key: true,
            ..b
        },
        "bailingmoe3" => ArchSpec {
            o_bias: false,
            mla: true,
            mla_rope_by_sections: true,
            mla_ignore_o_scale: true,
            attn_gate: true,
            hybrid_ssm: Some(SsmKind::Kda),
            hybrid_with_ffn: true,
            norm_topk_key: true,
            swiglu_clamp: true,
            ..b
        },
        "kimi-linear" => ArchSpec {
            o_bias: false,
            mla: true,
            mla_no_rope: true,
            gdn_layers: true,
            hybrid_ssm: Some(SsmKind::Kda),
            ..b
        },
        "phi3" => ArchSpec {
            out_bias: true,
            ..b
        },
        "olmoe" => ArchSpec {
            o_bias: false,
            qk_norm: QkNorm::Full,
            norm_topk: false,
            ..b
        },
        "qwen2" => ArchSpec {
            ffn_scale: true,
            out_bias: true,
            ..b
        },
        "qwen2moe" => ArchSpec {
            out_bias: true,
            norm_topk: false,
            ..b
        },
        "qwen3" => ArchSpec {
            qk_norm: QkNorm::PerHead,
            ffn_scale: true,
            ..b
        },
        "qwen3moe" => ArchSpec {
            qk_norm: QkNorm::PerHead,
            expert_scales: true,
            ..b
        },
        "gemma" => ArchSpec {
            o_bias: false,
            activation: Activation::GeluTanh,
            embed_scale: true,
            ..b
        },
        "gemma2" => ArchSpec {
            o_bias: false,
            activation: Activation::GeluTanh,
            embed_scale: true,
            post_norms: true,
            attn_softcap: true,
            final_softcap: true,
            swa_default: Some((4096, 2)),
            ..b
        },
        "gemma3" => ArchSpec {
            o_bias: false,
            activation: Activation::GeluTanh,
            embed_scale: true,
            qk_norm: QkNorm::PerHead,
            post_norms: true,
            final_softcap: true,
            swa_default: Some((0, 6)),
            local_rope_unscaled: true,
            ..b
        },
        // Gemma 3 blocks with per-layer head widths and rotary widths for the sliding-window layers, values taken from
        // the keys on layers without a V projection, per-layer input embeddings, a per-layer output scalar and the
        // last `attention.shared_kv_layers` layers reusing earlier K/V. MoE layers add softmax-routed experts (router on
        // the normed residual) next to the dense MLP.
        "gemma4" => ArchSpec {
            o_bias: false,
            activation: Activation::GeluTanh,
            embed_scale: true,
            qk_norm: QkNorm::PerHead,
            qk_norm_optional: true,
            post_norms: true,
            final_softcap: true,
            swa_default: Some((0, 0)),
            local_rope_unscaled: true,
            swa_geometry: true,
            g4: true,
            fixed_attn_scale: Some(1.0),
            // MoE layers: softmax routing with renormalized weights, GELU experts with per-expert scales, and the dense
            // MLP kept next to them.
            dense_with_moe: true,
            fixed_gating: Some(Gating::Softmax),
            norm_topk: true,
            moe_act: MoeAct::Gelu,
            expert_scales: true,
            ..b
        },
        // Gemma 3 blocks inside four AltUp streams with LAuReL side paths, per-layer input embeddings, activation
        // sparsity on the first ten layers and K/V shared from layers 18/19 by every layer from the twentieth on.
        "gemma3n" => ArchSpec {
            o_bias: false,
            activation: Activation::GeluTanh,
            embed_scale: true,
            qk_norm: QkNorm::PerHead,
            post_norms: true,
            final_softcap: true,
            swa_default: Some((0, 5)),
            local_rope_unscaled: true,
            g3n: true,
            fixed_attn_scale: Some(1.0),
            ..b
        },
        _ => return None,
    })
}

/// RoPE position scaling stored in the GGUF metadata (`{arch}.rope.scaling.*`), with llama.cpp's semantics.
#[derive(Debug, Clone, PartialEq)]
pub enum RopeScaling {
    None,
    /// Positions divided by `factor`.
    Linear {
        factor: f64,
    },
    /// YaRN: per-dimension blend of interpolated and extrapolated angles plus a magnitude factor.
    Yarn {
        factor: f64,
        orig_ctx: usize,
        attn_factor: f64,
        beta_fast: f64,
        beta_slow: f64,
        log_mul: f64,
    },
}

impl RopeScaling {
    /// `(cos, sin)` for rotary pair `i` (of `dim / 2`) at position `pos`, following ggml's `rope_yarn`.
    /// `freq_factor` is the optional per-pair divisor from the `rope_freqs.weight` tensor
    /// (how Llama-3.1 style scaling is stored).
    fn angle(&self, dim: usize, base: f64, pos: usize, i: usize, freq_factor: f64) -> (f64, f64) {
        let extrap = pos as f64 * base.powf(-(2.0 * i as f64) / dim as f64) / freq_factor;
        match self {
            RopeScaling::None => (extrap.cos(), extrap.sin()),
            RopeScaling::Linear { factor } => {
                let theta = extrap / factor;
                (theta.cos(), theta.sin())
            }
            RopeScaling::Yarn {
                factor,
                orig_ctx,
                attn_factor,
                beta_fast,
                beta_slow,
                log_mul,
            } => {
                let corr = |rot: f64| {
                    dim as f64 * (*orig_ctx as f64 / (rot * 2.0 * std::f64::consts::PI)).ln()
                        / (2.0 * base.ln())
                };
                let low = corr(*beta_fast).floor().max(0.0);
                let high = corr(*beta_slow).ceil().min(dim as f64 - 1.0);
                let ramp = 1.0 - ((i as f64 - low) / (high - low).max(0.001)).clamp(0.0, 1.0);
                let interp = extrap / factor;
                let theta = interp * (1.0 - ramp) + extrap * ramp; // ext_factor = 1
                // ggml cancels its own 1 + 0.1 ln(factor) and llama.cpp re-applies get_mscale(factor, 1) /
                // get_mscale(factor, log_mul) (the latter is 1 unless the model sets a log multiplier: DeepSeek).
                let get_mscale = |m: f64| {
                    if *factor > 1.0 {
                        1.0 + 0.1 * m * factor.ln()
                    } else {
                        1.0
                    }
                };
                let mscale = attn_factor * get_mscale(1.0) / get_mscale(*log_mul);
                (theta.cos() * mscale, theta.sin() * mscale)
            }
        }
    }
}

/// Configuration extracted from a GGUF model container.
#[derive(Debug, Clone)]
pub struct GGUFModelConfig {
    pub architecture: String,
    pub vocab_size: usize,
    pub d_model: usize,
    pub num_heads: usize,
    pub num_layers: usize,
    pub max_seq_len: usize,
    pub rope_freq_base: f32,
    /// Key/value heads (grouped-query attention); equals `num_heads` for plain multi-head models.
    pub num_kv_heads: usize,
    /// FFN hidden width (`feed_forward_length`); real models are not `8/3 * d_model`.
    pub intermediate_size: usize,
    pub rms_eps: f64,
    pub rope_scaling: RopeScaling,
    /// `{arch}.rope.dimension_count` when the file sets it (partial rotary embeddings).
    pub rope_dim: Option<usize>,
    /// `{arch}.attention.sliding_window`: each token attends to at most this many latest tokens.
    pub sliding_window: Option<usize>,
    /// Mixture-of-experts: routed experts per layer (0 = dense FFN) and how many each token uses.
    pub expert_count: usize,
    pub expert_used: usize,
    /// Renormalize the selected experts' routing weights to sum to 1 (Mixtral: yes, Qwen2-MoE: no).
    pub moe_norm_topk: bool,
    pub moe_weights_scale: f32,
    /// `{arch}.attention.clamp_kqv` (> 0): clamp Q, K and V to `[-c, c]` after projection.
    pub clamp_kqv: Option<f32>,
    /// Per-head width (`attention.key_length`); may differ from `d_model / num_heads`.
    pub head_dim: usize,
    /// Rotary dimensions per head (`rope.dimension_count`); less than `head_dim` = partial rotary.
    pub rope_n_rot: usize,
    /// Rotary width of the sliding-window layers when it differs from `rope_n_rot` (Step 3.5).
    pub rope_n_rot_swa: Option<usize>,
    /// Head width of the sliding-window layers when it differs from `head_dim` (Gemma 4).
    pub head_dim_swa: Option<usize>,
    /// Number of trailing layers that reuse the K/V of earlier layers (Gemma 4's `attention.shared_kv_layers`).
    pub kv_shared_layers: usize,
    pub activation: Activation,
    /// Multiply the token embeddings by this (Gemma: `sqrt(d_model)`).
    pub embedding_scale: Option<f32>,
    pub attn_softcap: Option<f32>,
    pub final_softcap: Option<f32>,
    /// Scale of `q . k` (default `1 / sqrt(head_dim)`).
    pub attn_scale: Option<f32>,
    pub qk_norm: QkNorm,
    /// Norms after the attention / FFN outputs (before the residual add).
    pub post_norms: bool,
    pub swa: SwaPattern,
    /// RoPE base of the sliding-window layers when it differs (`rope.freq_base_swa`).
    pub rope_base_swa: Option<f32>,
    pub local_rope_unscaled: bool,
    pub pre_norm: bool,
    pub no_rope_step: Option<usize>,
    /// Granite: scale the attention / FFN outputs before the residual add, and divide the logits.
    pub residual_scale: Option<f32>,
    /// MiniMax-01: the residual of each half-layer is `norm_residual * (normalised layer input)` instead of the raw stream.
    pub norm_residual: Option<f32>,
    pub logit_scale: Option<f32>,
    /// `rope.scaling.attn_factor`: multiplies the rotary cos/sin (magnitude scaling for every scaling type).
    pub rope_attn_factor: f64,
    /// `rope.scaling.original_context_length` (or the training context): picks LongRoPE long vs short factors.
    pub rope_orig_ctx: usize,
    /// MLA (DeepSeek): rank of the Q / KV low-rank factors (`q_lora_rank` 0 = single Q projection) and the
    /// per-head key / value widths. `kv_lora_rank == 0` means ordinary attention.
    pub kv_lora_rank: usize,
    pub q_lora_rank: usize,
    pub value_length_mla: usize,
    pub moe_gating: Gating,
    /// `rope.scaling.yarn_log_multiplier / 0.1` (DeepSeek), applied in the attention scale and YaRN's mscale.
    pub yarn_log_mul: f64,
    pub norm_layer: bool,
    pub norm_bias: bool,
    pub norm_scaled: bool,
    pub ffn_gated: bool,
    pub parallel_residual: bool,
    /// Logits multiplier (Command-R's `logit_scale`).
    pub logit_mul: Option<f32>,
    pub qk_norm_after_rope: bool,
    pub rope_only_swa: bool,
    pub parallel_own_ffn_norm: bool,
    /// Group-limited routing (DeepSeek-V3 style): experts form `expert_groups` groups and only the best
    /// `expert_groups_used` groups compete for the top-k.
    pub expert_groups: usize,
    pub expert_groups_used: usize,
    /// Per-layer attention head counts (Deci, OpenELM). Empty = `num_heads` for every layer.
    pub layer_heads: Vec<usize>,
    pub layer_kv_heads: Vec<usize>,
    /// Which layers rotate q/k (empty = all, subject to `no_rope_step`).
    pub rope_layers: Vec<bool>,
    /// Rotary pairs that sit on M-RoPE's fourth position channel, which is 0 for text: they never rotate.
    pub rope_zero_pairs: Vec<bool>,
    /// Positions come from a learned table added to the token embedding; no rotary embedding.
    pub abs_pos: bool,
    /// ALiBi `max_bias` (attention scores get `-slope_h * distance`); no rotary embedding then.
    pub alibi_max_bias: Option<f32>,
    pub chunked_attn: bool,
    pub kq_norm_weightless: bool,
    /// `(temperature_scale, floor)` of the NoPE query scaling.
    pub nope_temp: Option<(f32, usize)>,
    pub nope_temp_pow: bool,
    pub weighted_residual: bool,
    /// HRM-Text `(layers_per_stack, h_cycles, l_cycles)`: the low and high stacks are re-run with shared weights.
    pub hrm: Option<(usize, usize, usize)>,
    /// `attention.group_norm_groups` for architectures that group their main norms (1 otherwise).
    pub norm_groups: usize,
    /// `{arch}.post_norm_epsilon` of an architecture whose post norms carry their own epsilon.
    pub post_norm_eps: Option<f64>,
    /// `(value_expert_count, value_expert_used_count)` of a model with routed value experts.
    pub value_experts: Option<(usize, usize)>,
    /// 1: layers n-1, 2n-1, ... skip RoPE; 0: layers 0, n, 2n, ...
    pub no_rope_offset: usize,
    pub mla_plain: bool,
    pub ssm: Option<SsmConfig>,
    /// Layers mix attention, FFN and SSM one at a time (see [`Mixer`]); `layer_ff` holds each layer's FFN width.
    pub hybrid: bool,
    pub hybrid_with_ffn: bool,
    pub hybrid_parallel: bool,
    pub layer_ff: Vec<usize>,
    /// Per layer: gated-delta-net (true) or full attention (false). Empty when the model has no such layers.
    pub recurrent_layers: Vec<bool>,
    /// `(streams, epsilon, magnitude)` of the hyper-connection residual (HY-V4).
    pub hc: Option<(usize, f32, f32)>,
    /// Longest context the dense implementation of a sparse-attention model is exact for.
    pub dsa_top_k: Option<usize>,
    /// Per-layer xIELU parameters (`f32::to_bits` of `[alpha_n, alpha_p, beta, eps]`), empty for other activations.
    pub xielu: Vec<[u32; 4]>,
    /// Token ids `[start, end)` whose logits are forced to `-f32::MAX` (Chameleon's image tokens).
    pub suppress_logits: Option<(usize, usize)>,
    /// RWKV-6 model: every layer is attention-free ([`Mixer::Ssm`]) and carries its own norms.
    pub rwkv: bool,
}

impl GGUFModelConfig {
    /// The mixer layer `il` holds.
    pub fn mixer(&self, il: usize) -> Mixer {
        if self.rwkv {
            return Mixer::Ssm;
        }
        if let Some(&recurrent) = self.recurrent_layers.get(il) {
            return if recurrent {
                Mixer::SsmFfn
            } else {
                Mixer::Full
            };
        }
        if !self.hybrid {
            return if self.ssm.is_some() {
                Mixer::Ssm
            } else {
                Mixer::Full
            };
        }
        let (_, kv) = self.layer_dims(il);
        if self.hybrid_parallel {
            return Mixer::AttnSsm;
        }
        if self.hybrid_with_ffn {
            return if kv == 0 { Mixer::SsmFfn } else { Mixer::Full };
        }
        let ff = self
            .layer_ff
            .get(il)
            .copied()
            .unwrap_or(self.intermediate_size);
        match (kv, ff) {
            (0, 0) => Mixer::Ssm,
            (_, 0) => Mixer::AttnOnly,
            _ => Mixer::FfnOnly,
        }
    }

    /// `(heads, kv_heads)` of layer `il`.
    pub fn layer_dims(&self, il: usize) -> (usize, usize) {
        (
            self.layer_heads.get(il).copied().unwrap_or(self.num_heads),
            self.layer_kv_heads
                .get(il)
                .copied()
                .unwrap_or(self.num_kv_heads),
        )
    }

    /// Config for a model described only by hyperparameters (MHA, `8/3 * d_model` FFN, eps 1e-5).
    pub fn basic(
        vocab_size: usize,
        d_model: usize,
        num_heads: usize,
        num_layers: usize,
        max_seq_len: usize,
        rope_freq_base: f32,
    ) -> Result<Self> {
        let intermediate_size = d_model.checked_mul(8).map(|v| v / 3).ok_or_else(|| {
            Error::Msg(format!(
                "GGUFModelConfig::basic: intermediate size 8 * d_model ({d_model}) / 3 overflows"
            ))
        })?;
        Ok(Self {
            architecture: "llama".to_string(),
            vocab_size,
            d_model,
            num_heads,
            num_layers,
            max_seq_len,
            rope_freq_base,
            num_kv_heads: num_heads,
            intermediate_size,
            rms_eps: 1e-5,
            rope_scaling: RopeScaling::None,
            rope_dim: None,
            sliding_window: None,
            expert_count: 0,
            expert_used: 0,
            moe_norm_topk: true,
            moe_weights_scale: 1.0,
            clamp_kqv: None,
            head_dim: d_model / num_heads.max(1),
            rope_n_rot: d_model / num_heads.max(1),
            rope_n_rot_swa: None,
            head_dim_swa: None,
            kv_shared_layers: 0,
            activation: Activation::Silu,
            embedding_scale: None,
            attn_softcap: None,
            final_softcap: None,
            attn_scale: None,
            qk_norm: QkNorm::None,
            post_norms: false,
            swa: SwaPattern::All,
            rope_base_swa: None,
            local_rope_unscaled: false,
            pre_norm: true,
            no_rope_step: None,
            residual_scale: None,
            norm_residual: None,
            logit_scale: None,
            rope_attn_factor: 1.0,
            rope_orig_ctx: max_seq_len,
            kv_lora_rank: 0,
            q_lora_rank: 0,
            value_length_mla: 0,
            moe_gating: Gating::Softmax,
            yarn_log_mul: 0.0,
            norm_layer: false,
            norm_bias: false,
            norm_scaled: true,
            ffn_gated: true,
            parallel_residual: false,
            logit_mul: None,
            qk_norm_after_rope: false,
            rope_only_swa: false,
            parallel_own_ffn_norm: false,
            expert_groups: 0,
            expert_groups_used: 0,
            layer_heads: Vec::new(),
            layer_kv_heads: Vec::new(),
            rope_layers: Vec::new(),
            rope_zero_pairs: Vec::new(),
            abs_pos: false,
            alibi_max_bias: None,
            chunked_attn: false,
            kq_norm_weightless: false,
            nope_temp: None,
            nope_temp_pow: false,
            weighted_residual: false,
            hrm: None,
            norm_groups: 1,
            post_norm_eps: None,
            value_experts: None,
            no_rope_offset: 1,
            mla_plain: false,
            ssm: None,
            hybrid: false,
            hybrid_with_ffn: false,
            hybrid_parallel: false,
            layer_ff: Vec::new(),
            recurrent_layers: Vec::new(),
            hc: None,
            dsa_top_k: None,
            xielu: Vec::new(),
            suppress_logits: None,
            rwkv: false,
        })
    }

    /// Extracts model hyperparameters from GGUF metadata keys.
    pub fn from_gguf(file: &GGUFFile) -> Result<Self> {
        let arch = file
            .get_metadata_str("general.architecture")
            .unwrap_or("llama")
            .to_string();
        let spec = arch_spec(&arch).unwrap_or(BASE_SPEC);

        let vocab_size = file
            .get_metadata_u32(&format!("{arch}.vocab_size"))
            .or_else(|| file.get_metadata_u32("general.vocab_size"))
            .unwrap_or(32000) as usize;

        let d_model = file
            .get_metadata_u32(&format!("{arch}.embedding_length"))
            .unwrap_or(4096) as usize;
        if d_model == 0 {
            return Err(Error::Msg(format!(
                "{arch}.embedding_length must be positive (corrupt GGUF metadata)"
            )));
        }

        let per_layer = |key: &str| -> Result<Vec<usize>> {
            match file.metadata.get(&format!("{arch}.attention.{key}")) {
                Some(crate::serialize::gguf::GGUFValue::Array(a)) => a
                    .iter()
                    .map(|v| match v {
                        crate::serialize::gguf::GGUFValue::Uint32(n) => Ok(*n as usize),
                        crate::serialize::gguf::GGUFValue::Int32(n) if *n >= 0 => Ok(*n as usize),
                        _ => Err(Error::Msg(format!(
                            "{arch}.attention.{key} must hold non-negative integers"
                        ))),
                    })
                    .collect(),
                _ => Ok(Vec::new()),
            }
        };
        let (layer_heads, layer_kv_heads) = (per_layer("head_count")?, per_layer("head_count_kv")?);
        let layer_ff: Vec<usize> = match file.metadata.get(&format!("{arch}.feed_forward_length")) {
            Some(crate::serialize::gguf::GGUFValue::Array(a)) if spec.hybrid_ssm.is_some() => a
                .iter()
                .map(|v| match v {
                    crate::serialize::gguf::GGUFValue::Uint32(n) => Ok(*n as usize),
                    crate::serialize::gguf::GGUFValue::Int32(n) if *n >= 0 => Ok(*n as usize),
                    _ => Err(Error::Msg(format!(
                        "{arch}.feed_forward_length must hold non-negative integers"
                    ))),
                })
                .collect::<Result<_>>()?,
            _ => Vec::new(),
        };
        let rope_layers: Vec<bool> = if spec.rope_pattern_array {
            match file.metadata.get(&format!("{arch}.attention.rope_pattern")) {
                Some(crate::serialize::gguf::GGUFValue::Array(a)) => a
                    .iter()
                    .map(|v| match v {
                        crate::serialize::gguf::GGUFValue::Uint32(n) => *n != 0,
                        crate::serialize::gguf::GGUFValue::Int32(n) => *n != 0,
                        crate::serialize::gguf::GGUFValue::Bool(b) => *b,
                        _ => true,
                    })
                    .collect(),
                _ => Vec::new(),
            }
        } else {
            Vec::new()
        };
        let num_heads = file
            .get_metadata_u32(&format!("{arch}.attention.head_count"))
            .map_or_else(
                || layer_heads.iter().copied().max().unwrap_or(32),
                |v| v as usize,
            );

        // HRM-Text: `block_count` is the expanded cache-slot count; only the low and high stack hold weights.
        let hrm = if arch == "hrm_text" {
            let key = |k: &str| {
                file.get_metadata_u32(&format!("{arch}.hrm.{k}"))
                    .map(|v| v as usize)
                    .filter(|&v| v > 0)
            };
            let (Some(lps), Some(hc), Some(lc)) =
                (key("layers_per_stack"), key("h_cycles"), key("l_cycles"))
            else {
                return Err(Error::Msg(
                    "hrm_text needs positive hrm.layers_per_stack, hrm.h_cycles and hrm.l_cycles"
                        .into(),
                ));
            };
            Some((lps, hc, lc))
        } else {
            None
        };
        if let Some((lps, hc, lc)) = hrm {
            // The slot table has lps * hc * (lc + 1) entries (cache, loader scales, forward indexing):
            // refuse overflow and absurd counts here rather than panicking or hanging downstream.
            let slots = lps
                .checked_mul(hc)
                .and_then(|n| lc.checked_add(1).and_then(|m| n.checked_mul(m)))
                .ok_or_else(|| Error::Msg(format!("hrm_text cache slots overflow: layers_per_stack={lps}, h_cycles={hc}, l_cycles={lc}")))?;
            crate::limits::check_repeats("hrm_text cache slots", slots)?;
        }
        let num_layers = match hrm {
            Some((lps, ..)) => 2 * lps,
            None => file
                .get_metadata_u32(&format!("{arch}.block_count"))
                .unwrap_or(32) as usize,
        };

        for (name, v) in [
            ("head_count", &layer_heads),
            ("head_count_kv", &layer_kv_heads),
        ] {
            if !v.is_empty() && v.len() != num_layers {
                return Err(Error::Msg(format!(
                    "{arch}.attention.{name} has {} entries for {num_layers} layers",
                    v.len()
                )));
            }
        }
        let max_seq_len = file
            .get_metadata_u32(&format!("{arch}.context_length"))
            .unwrap_or(4096) as usize;

        let rope_freq_base = file
            .get_metadata_f32(&format!("{arch}.rope.freq_base"))
            .unwrap_or(10000.0);

        let num_kv_heads = file
            .get_metadata_u32(&format!("{arch}.attention.head_count_kv"))
            .map(|v| v as usize)
            .or_else(|| layer_kv_heads.iter().copied().max())
            .unwrap_or(num_heads);

        let intermediate_size = file
            .get_metadata_u32(&format!("{arch}.feed_forward_length"))
            .map(|v| v as usize)
            .unwrap_or((d_model * 8) / 3);

        let rms_eps = file
            .get_metadata_f32(&format!(
                "{arch}.attention.{}",
                if spec.layer_norm {
                    "layer_norm_epsilon"
                } else {
                    "layer_norm_rms_epsilon"
                }
            ))
            .map(f64::from)
            .unwrap_or(1e-5);

        let factor = file
            .get_metadata_f32(&format!("{arch}.rope.scaling.factor"))
            .map(f64::from)
            .unwrap_or(0.0);
        let attn_factor = file
            .get_metadata_f32(&format!("{arch}.rope.scaling.attn_factor"))
            .map(f64::from)
            .unwrap_or(1.0);
        let orig_ctx = file
            .get_metadata_u32(&format!("{arch}.rope.scaling.original_context_length"))
            .map_or(max_seq_len, |v| v as usize);
        // llama.cpp: type defaults to "linear"; a missing / zero factor means "no scaling".
        let rope_scaling = match file
            .get_metadata_str(&format!("{arch}.rope.scaling.type"))
            .unwrap_or("linear")
        {
            // LongRoPE keeps plain angles; its per-dimension factors come from the rope_factors_* tensors.
            "none" | "longrope" => RopeScaling::None,
            "linear" if factor > 0.0 && factor != 1.0 => RopeScaling::Linear { factor },
            "linear" => RopeScaling::None,
            "yarn" => RopeScaling::Yarn {
                factor: if factor > 0.0 { factor } else { 1.0 },
                orig_ctx,
                attn_factor,
                beta_fast: 32.0,
                beta_slow: 1.0,
                log_mul: if spec.mla {
                    file.get_metadata_f32(&format!("{arch}.rope.scaling.yarn_log_multiplier"))
                        .map_or(0.0, |v| f64::from(v) / 0.1)
                } else {
                    0.0
                },
            },
            other => {
                return Err(Error::Msg(format!(
                    "unsupported rope.scaling.type '{other}'"
                )));
            }
        };
        let rope_dim = file
            .get_metadata_u32(&format!("{arch}.rope.dimension_count"))
            .map(|v| v as usize);

        let sliding_window = file
            .get_metadata_u32(&format!("{arch}.attention.sliding_window"))
            .map(|v| v as usize)
            .or(spec_swa_window(&arch))
            // llama.cpp reads the window only for architectures that implement it (not `llama`/Mistral, and
            // Phi-3's is disabled); Exaone 4 uses it on its 64-layer model only.
            .filter(|&w| {
                w > 0 && spec.swa_default.is_some() && !(arch == "exaone4" && num_layers != 64)
            });
        let swa = match file
            .metadata
            .get(&format!("{arch}.attention.sliding_window_pattern"))
        {
            Some(crate::serialize::gguf::GGUFValue::Array(a)) => SwaPattern::Explicit(
                a.iter()
                    .map(|v| match v {
                        crate::serialize::gguf::GGUFValue::Uint32(n) => *n != 0,
                        crate::serialize::gguf::GGUFValue::Int32(n) => *n != 0,
                        crate::serialize::gguf::GGUFValue::Bool(b) => *b,
                        _ => false,
                    })
                    .collect(),
            ),
            Some(crate::serialize::gguf::GGUFValue::Uint32(n)) => SwaPattern::Modulo(*n as usize),
            _ => spec
                .swa_default
                .map_or(SwaPattern::All, |(_, p)| SwaPattern::Modulo(p)),
        };
        let nonzero = |key: &str| {
            file.get_metadata_f32(&format!("{arch}.{key}"))
                .filter(|&v| v != 0.0)
        };
        let rope_base_swa = if spec.swa_default.is_some() {
            nonzero("rope.freq_base_swa")
        } else {
            None
        };
        let expert_count = file
            .get_metadata_u32(&format!("{arch}.expert_count"))
            .map_or(0, |v| v as usize);
        let expert_used = file
            .get_metadata_u32(&format!("{arch}.expert_used_count"))
            .map_or(0, |v| v as usize);
        if expert_count > 0 && (expert_used == 0 || expert_used > expert_count) {
            return Err(Error::Msg(format!(
                "expert_used_count {expert_used} must be in 1..={expert_count} for an MoE model"
            )));
        }
        // llama.cpp hard-codes this per architecture (the `expert_weights_norm` key is not consulted for
        // these): Mixtral / `llama` renormalizes the selected weights, Qwen2-MoE keeps the raw softmax.
        let moe_norm_topk = if spec.norm_topk_key {
            match file.metadata.get(&format!("{arch}.expert_weights_norm")) {
                Some(crate::serialize::gguf::GGUFValue::Bool(b)) => *b,
                _ => spec.topk_key_default,
            }
        } else {
            spec.norm_topk
        };
        let moe_gating = if let Some(g) = spec.fixed_gating {
            g
        } else {
            match (
                spec.mla || spec.gating_key,
                file.get_metadata_u32(&format!("{arch}.expert_gating_func")),
            ) {
                (true, Some(2)) => Gating::Sigmoid,
                (true, Some(3)) => Gating::SoftmaxWeight,
                (true, Some(4)) => Gating::SqrtSoftplus,
                (true, Some(0) | None) if spec.gating_default_sigmoid => Gating::Sigmoid,
                (true, Some(0 | 1) | None) | (false, _) => Gating::Softmax,
                (true, Some(other)) => {
                    return Err(Error::Msg(format!(
                        "unsupported expert_gating_func {other}"
                    )));
                }
            }
        };
        // llama.cpp's Beam loader never reads the group keys, so its router ignores them even when present.
        let group_key = |k: &str| {
            if arch == "beam" {
                0
            } else {
                file.get_metadata_u32(&format!("{arch}.{k}"))
                    .map_or(0, |g| g as usize)
            }
        };
        let expert_groups = group_key("expert_group_count");
        let expert_groups_used = group_key("expert_group_used_count");
        if expert_groups > 1
            && (expert_groups_used == 0
                || expert_groups_used > expert_groups
                || expert_count % expert_groups != 0)
        {
            return Err(Error::Msg(format!(
                "invalid group routing: {expert_groups} groups, {expert_groups_used} used, {expert_count} experts"
            )));
        }
        // `swiglu_clamp_*` only matters to architectures that read it (DeepSeek-V4, GLM-5, ...), none of which are
        // supported here; every supported architecture ignores the keys, exactly like llama.cpp.
        let clamp_kqv = file
            .get_metadata_f32(&format!("{arch}.attention.clamp_kqv"))
            .filter(|&c| c > 0.0);
        let moe_weights_scale = file
            .get_metadata_f32(&format!("{arch}.expert_weights_scale"))
            .filter(|&s| s != 0.0)
            .unwrap_or(1.0);

        if file
            .get_metadata_u32(&format!("{arch}.num_loops"))
            .is_some_and(|n| n > 1)
        {
            return Err(Error::Msg(
                "looped layer stacks (num_loops > 1) are not supported".into(),
            ));
        }
        if spec.nope_temp_floor.is_none() && nonzero("attention.temperature_scale").is_some() {
            return Err(Error::Msg(format!(
                "{arch}.attention.temperature_scale (attention temperature scaling) is not supported"
            )));
        }
        let mla_dim = |key: &str| {
            if spec.mla {
                file.get_metadata_u32(&format!("{arch}.attention.{key}"))
                    .map_or(0, |v| v as usize)
            } else {
                0
            }
        };
        let (kv_lora_rank, q_lora_rank, value_length_mla) = (
            mla_dim("kv_lora_rank"),
            mla_dim("q_lora_rank"),
            mla_dim(if spec.mla_plain {
                "value_length"
            } else {
                "value_length_mla"
            }),
        );
        let yarn_log_mul = if spec.mla {
            file.get_metadata_f32(&format!("{arch}.rope.scaling.yarn_log_multiplier"))
                .map_or(0.0, |v| f64::from(v) / 0.1)
        } else {
            0.0
        };
        let parallel_residual = spec.parallel
            || (spec.parallel_key
                && matches!(
                    file.metadata.get(&format!("{arch}.use_parallel_residual")),
                    Some(crate::serialize::gguf::GGUFValue::Bool(true))
                ));
        // Block-sparse attention (MiniMax-M3) selects `top_k` blocks of `block_size` tokens; token-level DSA has block size 1.
        let indexer_block = file
            .get_metadata_u32(&format!("{arch}.attention.indexer.block_size"))
            .map_or(1, |v| v.max(1) as usize);
        let dsa_top_k = if spec.dsa {
            file.get_metadata_u32(&format!("{arch}.attention.indexer.top_k"))
                .map(|v| (v as usize).saturating_mul(indexer_block))
        } else {
            None
        };
        let nope_temp = spec.nope_temp_floor.map(|f| {
            // The power-law form reads its length from the file; the log-floor form has it built in.
            let length = if spec.nope_temp_pow {
                file.get_metadata_u32(&format!("{arch}.attention.temperature_length"))
                    .map_or(f, |v| v as usize)
            } else {
                f
            };
            (
                file.get_metadata_f32(&format!("{arch}.attention.temperature_scale"))
                    .unwrap_or(0.1),
                length,
            )
        });
        // A window of 0 switches Llama 4's chunking off and makes every layer rotary.
        let window_key = file.get_metadata_u32(&format!("{arch}.attention.sliding_window"));
        let no_rope_step = if (spec.chunked_attn && window_key == Some(0))
            || (spec.no_rope_from_zero && window_key.is_none())
        {
            None
        } else {
            spec.no_rope_step
        };
        let activation = if spec.situ {
            let beta = |k: &str| {
                file.get_metadata_f32(&format!("{arch}.activation.{k}"))
                    .unwrap_or(0.0)
            };
            Activation::Situ(
                beta("situ_beta").to_bits(),
                beta("situ_linear_beta").to_bits(),
            )
        } else {
            spec.activation
        };
        let rope_n_rot_swa = if spec.step35_rot || spec.swa_geometry {
            file.get_metadata_u32(&format!("{arch}.rope.dimension_count_swa"))
                .map(|v| v as usize)
        } else {
            None
        };
        let head_dim_swa = if spec.swa_geometry {
            file.get_metadata_u32(&format!("{arch}.attention.key_length_swa"))
                .map(|v| v as usize)
        } else {
            None
        };
        let kv_shared_layers = if spec.g4 {
            file.get_metadata_u32(&format!("{arch}.attention.shared_kv_layers"))
                .map_or(0, |v| v as usize)
        } else {
            0
        };
        let swin_norm = spec.swin_norm_key
            && matches!(
                file.metadata.get(&format!("{arch}.swin_norm")),
                Some(crate::serialize::gguf::GGUFValue::Bool(true))
            );
        // Apertus: one `[alpha_n, alpha_p, beta, eps]` per layer from the global `xielu.*` arrays.
        let xielu: Vec<[u32; 4]> = if spec.xielu {
            let col = |k: &str| -> Result<Vec<f32>> {
                match file.metadata.get(&format!("xielu.{k}")) {
                    Some(crate::serialize::gguf::GGUFValue::Array(a)) => a
                        .iter()
                        .map(|v| match v {
                            crate::serialize::gguf::GGUFValue::Float32(x) => Ok(*x),
                            _ => Err(Error::Msg(format!("xielu.{k} must hold floats"))),
                        })
                        .collect(),
                    _ => Err(Error::Msg(format!("xielu.{k} is missing"))),
                }
            };
            let (an, ap, beta, eps) = (col("alpha_n")?, col("alpha_p")?, col("beta")?, col("eps")?);
            (0..num_layers)
                .map(|i| {
                    let at = |c: &[f32]| c.get(i).or(c.first()).copied().unwrap_or(0.0).to_bits();
                    [at(&an), at(&ap), at(&beta), at(&eps)]
                })
                .collect()
        } else {
            Vec::new()
        };
        let recurrent_layers: Vec<bool> = if spec.gdn_layers {
            match file
                .metadata
                .get(&format!("{arch}.attention.recurrent_layers"))
            {
                Some(crate::serialize::gguf::GGUFValue::Array(a)) => a
                    .iter()
                    .map(|v| match v {
                        crate::serialize::gguf::GGUFValue::Uint32(n) => *n != 0,
                        crate::serialize::gguf::GGUFValue::Int32(n) => *n != 0,
                        crate::serialize::gguf::GGUFValue::Bool(b) => *b,
                        _ => false,
                    })
                    .collect(),
                _ => {
                    let every = file
                        .get_metadata_u32(&format!("{arch}.full_attention_interval"))
                        .unwrap_or(4)
                        .max(1) as usize;
                    (0..num_layers).map(|i| (i + 1) % every != 0).collect()
                }
            }
        } else {
            Vec::new()
        };
        let ssm = match spec.ssm.or(spec.hybrid_ssm) {
            None => None,
            Some(SsmKind::ShortConv) => {
                let l = file
                    .get_metadata_u32(&format!("{arch}.shortconv.l_cache"))
                    .ok_or_else(|| Error::Msg(format!("{arch}.shortconv.l_cache is missing")))?
                    as usize;
                Some(SsmConfig {
                    kind: SsmKind::ShortConv,
                    d_conv: l,
                    d_inner: d_model,
                    d_state: 0,
                    dt_rank: 0,
                    n_group: 1,
                    dt_b_c_rms: false,
                })
            }
            Some(SsmKind::Lightning) => {
                let head_dim = file
                    .get_metadata_u32(&format!("{arch}.attention.key_length"))
                    .ok_or_else(|| Error::Msg(format!("{arch}.attention.key_length is missing")))?
                    as usize;
                Some(SsmConfig {
                    kind: SsmKind::Lightning,
                    d_conv: 1,
                    d_inner: num_heads * head_dim,
                    d_state: head_dim,
                    dt_rank: num_heads,
                    n_group: 1,
                    dt_b_c_rms: false,
                })
            }
            Some(SsmKind::Kda) => {
                let head_dim = file
                    .get_metadata_u32(&format!("{arch}.kda.head_dim"))
                    .ok_or_else(|| Error::Msg(format!("{arch}.kda.head_dim is missing")))?
                    as usize;
                let d_conv = file
                    .get_metadata_u32(&format!("{arch}.ssm.conv_kernel"))
                    .ok_or_else(|| Error::Msg(format!("{arch}.ssm.conv_kernel is missing")))?
                    as usize;
                Some(SsmConfig {
                    kind: SsmKind::Kda,
                    d_conv,
                    d_inner: num_heads * head_dim,
                    d_state: head_dim,
                    dt_rank: num_heads,
                    n_group: 1,
                    dt_b_c_rms: false,
                })
            }
            Some(kind) => {
                let key = |k: &str| {
                    file.get_metadata_u32(&format!("{arch}.ssm.{k}"))
                        .map(|v| v as usize)
                };
                let need = |k: &str| {
                    key(k).ok_or_else(|| Error::Msg(format!("{arch}.ssm.{k} is missing")))
                };
                Some(SsmConfig {
                    kind,
                    d_conv: need("conv_kernel")?,
                    d_inner: need("inner_size")?,
                    d_state: need("state_size")?,
                    dt_rank: need("time_step_rank")?,
                    n_group: if matches!(kind, SsmKind::Mamba2 | SsmKind::GatedDelta) {
                        key("group_count").unwrap_or(1).max(1)
                    } else {
                        1
                    },
                    dt_b_c_rms: matches!(
                        file.metadata.get(&format!("{arch}.ssm.dt_b_c_rms")),
                        Some(crate::serialize::gguf::GGUFValue::Bool(true))
                    ),
                })
            }
        };
        if let Some(c) = &ssm {
            // Every SSM scan divides by dt_rank / n_group or subtracts 1 from d_conv, and the
            // loader multiplies these into shapes: refuse degenerate geometry and absurd dims
            // here instead of panicking in the loader or mid-scan.
            if c.d_conv == 0 {
                return Err(Error::Msg(format!("{arch}.ssm.conv_kernel must be >= 1")));
            }
            if c.dt_rank == 0
                && matches!(
                    c.kind,
                    SsmKind::Mamba2 | SsmKind::GatedDelta | SsmKind::Plamo2
                )
            {
                return Err(Error::Msg(format!(
                    "{arch}.ssm.time_step_rank must be >= 1"
                )));
            }
            if c.kind == SsmKind::Mamba2 {
                if c.dt_rank % c.n_group != 0 {
                    return Err(Error::Msg(format!(
                        "mamba2 needs time_step_rank ({}) divisible by group_count ({})",
                        c.dt_rank, c.n_group
                    )));
                }
                if c.d_inner % c.dt_rank != 0 {
                    return Err(Error::Msg(format!(
                        "mamba2 needs inner_size ({}) divisible by time_step_rank ({})",
                        c.d_inner, c.dt_rank
                    )));
                }
            }
            if c.kind == SsmKind::Plamo2 && c.d_inner % c.dt_rank != 0 {
                return Err(Error::Msg(format!(
                    "plamo2 needs inner_size ({}) divisible by time_step_rank ({})",
                    c.d_inner, c.dt_rank
                )));
            }
            for (name, v) in [
                ("inner_size", c.d_inner),
                ("state_size", c.d_state),
                ("time_step_rank", c.dt_rank),
                ("conv_kernel", c.d_conv),
                ("group_count", c.n_group),
            ] {
                crate::limits::check_at_most(&format!("ssm.{name}"), v, 1 << 20)?;
            }
        }
        let alibi_max_bias = spec.alibi_fixed.or_else(|| {
            if spec.alibi_key {
                nonzero("attention.max_alibi_bias")
            } else {
                None
            }
        });
        let logit_mul = if spec.logit_mul_key {
            nonzero("logit_scale").or(spec.grok_scales.then_some(0.577_350_26))
        } else {
            None
        };
        let key_length = file
            .get_metadata_u32(&format!(
                "{arch}.attention.{}",
                if spec.mla && !spec.mla_plain {
                    "key_length_mla"
                } else {
                    "key_length"
                }
            ))
            .map_or(0, |v| v as usize);
        let attn_softcap = if spec.attn_softcap {
            nonzero("attn_logit_softcapping").or(spec.grok_scales.then_some(30.0))
        } else {
            None
        };
        let final_softcap = if spec.final_softcap {
            nonzero("final_logit_softcapping")
        } else {
            None
        };
        let attn_scale = if let Some(fixed) = spec.fixed_attn_scale {
            Some(fixed)
        } else if spec.grok_scales {
            nonzero("attention.output_scale").or(Some(0.088_388_35))
        } else {
            nonzero("attention.scale")
        };
        // Granite-style scales: a present key wins (0 = off); MiniCPM falls back to its built-in defaults.
        let mup_scale = |key: &str, default: Option<f32>| match file
            .get_metadata_f32(&format!("{arch}.{key}"))
        {
            Some(v) if v != 0.0 => Some(v),
            Some(_) => None,
            None => default,
        };
        let hc = if spec.hc {
            let n = file
                .get_metadata_u32(&format!("{arch}.hyper_connection.count"))
                .ok_or_else(|| Error::Msg(format!("{arch}.hyper_connection.count is missing")))?
                as usize;
            let get = |k: &str| {
                file.get_metadata_f32(&format!("{arch}.hyper_connection.{k}"))
                    .unwrap_or(0.0)
            };
            Some((n, get("epsilon"), get("magnitude")))
        } else {
            None
        };
        let norm_residual = if spec.norm_residual {
            file.get_metadata_f32(&format!("{arch}.residual_scale"))
        } else {
            None
        };
        let minicpm = arch == "minicpm" || arch == "minicpm3";
        // MiniCPM3's graph hard-codes its scalings and ignores the keys.
        let hard_coded = arch == "minicpm3";
        let residual_scale = if hard_coded {
            Some(1.4 / (num_layers as f32).sqrt())
        } else if spec.mup {
            mup_scale(
                "residual_scale",
                minicpm.then(|| 1.4 / (num_layers as f32).sqrt()),
            )
        } else {
            None
        };
        let logit_scale = if spec.mup {
            mup_scale(
                "logit_scale",
                (arch == "minicpm").then(|| 256.0 / d_model as f32),
            )
        } else {
            None
        };
        let embedding_scale = if spec.grok_scales {
            nonzero("embedding_scale").or(Some(78.383_67))
        } else if hard_coded {
            Some(12.0)
        } else if spec.mup {
            mup_scale("embedding_scale", minicpm.then_some(12.0))
        } else {
            None
        };
        let norm_groups = if spec.norm_groups_key {
            file.get_metadata_u32(&format!("{arch}.attention.group_norm_groups"))
                .map_or(1, |g| (g as usize).max(1))
        } else {
            1
        };
        let post_norm_eps = if spec.spark_blocks {
            file.get_metadata_f32(&format!("{arch}.post_norm_epsilon"))
                .map(f64::from)
        } else {
            None
        };
        let value_experts = if spec.value_experts {
            let key = |k: &str| {
                file.get_metadata_u32(&format!("{arch}.attention.{k}"))
                    .map_or(0, |v| v as usize)
            };
            Some((key("value_expert_count"), key("value_expert_used_count")))
                .filter(|&(n, u)| n > 0 && u > 0 && u <= n)
        } else {
            None
        };

        Ok(Self {
            architecture: arch,
            vocab_size,
            d_model,
            num_heads,
            num_layers,
            max_seq_len,
            rope_freq_base,
            num_kv_heads,
            intermediate_size,
            rms_eps,
            rope_scaling,
            rope_dim,
            sliding_window,
            expert_count,
            expert_used,
            moe_norm_topk,
            moe_weights_scale,
            clamp_kqv,
            head_dim: key_length,
            rope_n_rot: if spec.rope_full_head {
                0
            } else if spec.step35_rot {
                rope_dim.unwrap_or(0) / 2
            } else {
                rope_dim.unwrap_or(0)
            },
            rope_n_rot_swa,
            head_dim_swa,
            kv_shared_layers,
            activation,
            embedding_scale, // Gemma's sqrt(d_model) needs the final d_model; set by GGUFCausalLM::from_gguf
            attn_softcap,
            final_softcap,
            attn_scale,
            qk_norm: spec.qk_norm,
            post_norms: spec.post_norms || swin_norm,
            suppress_logits: spec.suppress_logits,
            rwkv: spec.rwkv6 || spec.rwkv7 || spec.rwkv6qwen2,
            swa,
            rope_base_swa,
            local_rope_unscaled: spec.local_rope_unscaled,
            pre_norm: spec.pre_norm && !swin_norm,
            no_rope_step,
            residual_scale,
            norm_residual,
            logit_scale,
            rope_attn_factor: attn_factor,
            rope_orig_ctx: orig_ctx,
            kv_lora_rank,
            q_lora_rank,
            value_length_mla,
            moe_gating,
            yarn_log_mul,
            norm_layer: spec.layer_norm,
            norm_bias: spec.norm_bias,
            norm_scaled: spec.norm_param,
            ffn_gated: spec.ffn_gated,
            parallel_residual,
            logit_mul,
            qk_norm_after_rope: spec.qk_norm_after_rope,
            rope_only_swa: spec.rope_only_swa,
            parallel_own_ffn_norm: spec.parallel_own_ffn_norm,
            expert_groups,
            expert_groups_used,
            layer_heads,
            layer_kv_heads,
            rope_layers,
            rope_zero_pairs: Vec::new(),
            abs_pos: spec.abs_pos,
            alibi_max_bias,
            chunked_attn: spec.chunked_attn,
            kq_norm_weightless: spec.kq_norm_weightless && expert_count != 128,
            nope_temp,
            nope_temp_pow: spec.nope_temp_pow,
            weighted_residual: spec.weighted_residual,
            hrm,
            norm_groups,
            post_norm_eps,
            value_experts,
            no_rope_offset: usize::from(!spec.no_rope_from_zero),
            mla_plain: spec.mla_plain,
            ssm,
            hybrid: spec.hybrid_ssm.is_some(),
            hybrid_with_ffn: spec.hybrid_with_ffn,
            hybrid_parallel: spec.hybrid_parallel,
            layer_ff,
            recurrent_layers,
            hc,
            dsa_top_k,
            xielu,
        })
    }
}

/// Default sliding-window size an architecture assumes when the file has no `attention.sliding_window`.
fn spec_swa_window(arch: &str) -> Option<usize> {
    arch_spec(arch).and_then(|s| s.swa_default).map(|(w, _)| w)
}

/// Activation of the routed experts' gated FFN.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MoeAct {
    Silu,
    /// OpenAI's clamped SwiGLU: `min(g, 7) * sigmoid(1.702 * min(g, 7)) * (clamp(u, -7, 7) + 1)`.
    SwigluOai,
    /// GELU (tanh form) gate, as in Grok.
    Gelu,
    /// ReLU gate (SmallThinker).
    Relu,
    /// Ungated `relu(up)^2` (Nemotron-H experts): there is no gate tensor, `gate` is only a placeholder.
    ReluSqr,
    /// Kimi-K3's SITU, see [`Activation::Situ`].
    Situ(u32, u32),
}

/// How router logits become expert probabilities (`expert_gating_func`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Gating {
    Softmax,
    Sigmoid,
    /// Softmax over only the selected experts' raw logits.
    SoftmaxWeight,
    /// `sqrt(softplus(logit))` (DeepSeek-V4).
    SqrtSoftplus,
}

/// Shared (always-on) expert of Qwen2-MoE style layers, optionally gated by a sigmoid of a learned dot product.
pub struct SharedExpert {
    pub gate: Tensor,
    pub up: Tensor,
    pub down: Tensor,
    pub gate_inp: Option<Tensor>,
    /// Output scales of the gate / up / down matmuls (`ffn_*_shexp.scale`).
    pub scales: Option<[f32; 3]>,
    /// Ungated `relu(up)^2` expert (Nemotron-H); `gate` is then unused.
    pub relu_sqr: bool,
    /// `swiglu_clamp_shexp` of this layer (0 = off), as [`MoeFfn::limit`].
    pub limit: f32,
}

/// GroveMoE's second stage: a few chunk experts, each shared by `group` routed experts, run on the first
/// stage's output and are added with `scale`.
pub struct ChunkExperts {
    pub gate: Tensor, // [n_chunk, ff, d]
    pub up: Tensor,
    pub down: Tensor, // [n_chunk, d, ff]
    pub group: usize,
    pub scale: f32,
}

/// Maion-Coder's latent MoE: tokens are projected down, routed and processed in the latent space, then
/// projected back up.
pub struct LatentProj {
    pub down: Tensor, // [latent, d]
    pub up: Tensor,   // [d, latent]
    /// Weighted RMS norm applied to the projected-down tokens before routing to experts (Muse-Spark).
    pub pre_norm: Option<Norm>,
    /// RMS norm on the experts' combined output before it is projected back up (Kimi-K3).
    pub mid_norm: Option<Norm>,
}

/// K2 Horizon's routed values: `V = sum_k w_k * silu(W_k x)` over the `used` best of the value experts.
pub struct ValueMoe {
    /// Router, `[n_value_experts, d]`.
    pub gate: Tensor,
    /// Added to the router probabilities for *choosing* experts only.
    pub bias: Option<Vec<f32>>,
    /// `[n_value_experts, kv_width, d]`.
    pub exps: Tensor,
    pub used: usize,
    pub gating: Gating,
    pub norm: bool,
    pub scale: f32,
}

impl ValueMoe {
    /// `x`: `[b, s, d]` -> `[b, s, kv_width]`. Every expert is evaluated and masked by its routing weight.
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let (b, s, d) = (x.shape()[0], x.shape()[1], x.shape()[2]);
        let (t, e, kvd) = (b * s, self.gate.shape()[0], self.exps.shape()[1]);
        let flat = x.reshape(&[t, d])?;
        let logits = flat.matmul(&self.gate.t()?)?.to_vec::<f32>()?;
        let mut w = vec![0f32; t * e];
        for tok in 0..t {
            let l = &logits[tok * e..(tok + 1) * e];
            let probs: Vec<f32> = match self.gating {
                Gating::Sigmoid => l.iter().map(|v| 1.0 / (1.0 + (-v).exp())).collect(),
                Gating::Softmax => {
                    let max = l.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                    let ex: Vec<f32> = l.iter().map(|v| (v - max).exp()).collect();
                    let total: f32 = ex.iter().sum();
                    ex.iter().map(|v| v / total).collect()
                }
                Gating::SoftmaxWeight | Gating::SqrtSoftplus => {
                    return Err(Error::Msg(
                        "value experts support softmax or sigmoid gating only".into(),
                    ));
                }
            };
            let pick: Vec<f32> = probs
                .iter()
                .enumerate()
                .map(|(i, p)| p + self.bias.as_ref().map_or(0.0, |b| b[i]))
                .collect();
            let mut order: Vec<usize> = (0..e).collect();
            order.sort_by(|&a, &b| pick[b].total_cmp(&pick[a]).then(a.cmp(&b)));
            let chosen = &order[..self.used];
            let total = chosen
                .iter()
                .map(|&i| probs[i])
                .sum::<f32>()
                .max(6.103_515_6e-5);
            for &i in chosen {
                let mut weight = probs[i];
                if self.norm {
                    weight /= total;
                }
                if self.scale != 0.0 && self.scale != 1.0 {
                    weight *= self.scale;
                }
                w[tok * e + i] = weight;
            }
        }
        let w = Tensor::from_slice_on(&w, &[t, e], &x.device())?;
        let mut out: Option<Tensor> = None;
        for ei in 0..e {
            let v = flat
                .matmul(&self.exps.narrow(0, ei, 1)?.reshape(&[kvd, d])?.t()?)?
                .silu()?;
            let v = v.mul(&w.narrow(1, ei, 1)?.expand(v.shape())?)?;
            out = Some(match out {
                None => v,
                Some(o) => o.add(&v)?,
            });
        }
        out.ok_or_else(|| Error::Msg("value experts need at least one expert".into()))?
            .reshape(&[b, s, kvd])
    }
}

/// Sparse mixture-of-experts FFN: a router picks `top_k` of `n_expert` SwiGLU experts per token.
/// Expert weights are stacked: `gate`/`up` are `[n_expert, ff, d]`, `down` is `[n_expert, d, ff]`.
pub struct MoeFfn {
    /// Each expert's output is scaled by this weight and RMS-normalized (with this epsilon) before the router weight
    /// applies (Muse-Spark).
    pub post_exp_norm: Option<(Tensor, f64)>,
    pub router: Tensor, // [n_expert, d]
    pub gate: Tensor,
    pub up: Tensor,
    pub down: Tensor,
    pub top_k: usize,
    pub norm_topk: bool,
    pub weights_scale: f32,
    pub shared: Option<SharedExpert>,
    /// Per-expert scalar output scales of the gate / up / down matmuls (`ffn_*_exps.scale`).
    pub gate_scale: Option<Vec<f32>>,
    pub up_scale: Option<Vec<f32>>,
    pub down_scale: Option<Vec<f32>>,
    pub gating: Gating,
    /// DeepSeek-V3 `exp_probs_b`: added to the probabilities for *choosing* experts only.
    pub selection_bias: Option<Vec<f32>>,
    /// Choose experts on `logits + selection_bias` instead of `probs + selection_bias`.
    pub select_on_logits: bool,
    /// Multiply the routing weight into the expert's input instead of its output (Llama 4).
    pub weight_before_ffn: bool,
    pub chunk: Option<ChunkExperts>,
    pub latent: Option<LatentProj>,
    /// Group-limited routing: `n_groups` groups of experts, only the best `n_group_used` may be chosen from.
    pub n_groups: usize,
    pub n_group_used: usize,
    /// Router bias `[n_expert]` and per-expert biases `[n_expert, ff]` / `[n_expert, d]` (GPT-OSS).
    pub router_bias: Option<Vec<f32>>,
    pub gate_bias: Option<Tensor>,
    pub up_bias: Option<Tensor>,
    pub down_bias: Option<Tensor>,
    pub act: MoeAct,
    /// `swiglu_clamp_exp` of this layer (0 = off): `min(silu(g), L) * clamp(u, -L, L)`.
    pub limit: f32,
}

/// One hyper-connection gate: `mixes = rms_norm(flatten(streams)) @ fn^T`, split into `pre` and `post` coefficients.
pub struct HcParams {
    /// `[2 * streams, streams * d]` (`[streams, streams * d]` for the output head).
    pub w: Tensor,
    /// `[2 * streams]` (`[streams]`).
    pub base: Tensor,
    /// `[pre, post]` (`[pre]`).
    pub scale: Vec<f32>,
    /// Qwen4-Exp's low-rank form, which replaces `w` / `base` / `scale`.
    pub low_rank: Option<LowRankHc>,
}

/// A low-rank hyper-connection mixer: a per-stream RMS norm, a bottleneck that gates every element of the normed
/// streams, the mean over the streams as the layer input, and (inside the layers) a `[streams]` injection weight.
pub struct LowRankHc {
    /// `[streams, d]`
    pub norm: Tensor,
    /// `[rank, streams * d]` and `[streams * d, rank]`
    pub down: Tensor,
    pub up: Tensor,
    /// `[streams, streams * d]`; absent on the output head.
    pub inject: Option<Tensor>,
}

impl LowRankHc {
    /// `x`: `[b, s, streams, d]`. Returns the mixed `[b, s, d]` input and the flattened normed streams.
    fn mix(&self, x: &Tensor, norm_eps: f64) -> Result<(Tensor, Tensor)> {
        let (b, s, n, d) = (x.shape()[0], x.shape()[1], x.shape()[2], x.shape()[3]);
        let normed = x.mul(
            &x.square()?
                .mean_axis(-1, true)?
                .add_scalar(norm_eps)?
                .rsqrt()?,
        )?;
        let xn = normed
            .mul(&self.norm.reshape(&[1, 1, n, d])?.expand(&[b, s, n, d])?)?
            .reshape(&[b, s, n * d])?;
        let lo = xn
            .matmul(&self.down.t()?)?
            .mul_scalar(1.0 / n as f64)?
            .silu()?;
        let gate = lo.matmul(&self.up.t()?)?.sigmoid()?;
        let mixed = xn.mul(&gate)?.reshape(&[b, s, n, d])?.mean_axis(2, false)?;
        Ok((mixed, xn))
    }
}

/// Both gates of a layer plus the shared hyper-connection settings.
pub struct HcLayer {
    pub attn: HcParams,
    pub ffn: HcParams,
    pub streams: usize,
    pub eps: f32,
    pub magnitude: f32,
    pub norm_eps: f64,
    /// Manifold-constrained form (GLM-5-Next, DeepSeek-V4): the gates also produce an `n x n` stream-mixing matrix,
    /// made doubly stochastic with this many Sinkhorn iterations (`None` for plain iHC).
    pub sinkhorn: Option<usize>,
}

/// The `post` gates `[b, s, n]` of a hyper-connection and, in the manifold-constrained form, its mixing matrix
/// `[b, s, dst, src]`.
pub struct HcMix {
    pub post: Tensor,
    pub comb: Option<Tensor>,
}

impl HcLayer {
    /// `x`: `[b, s, streams, d]`. Returns the reduced stream `[b, s, d]` and the gates for [`hc_post`].
    fn pre(&self, p: &HcParams, x: &Tensor) -> Result<(Tensor, HcMix)> {
        let (b, s, n, d) = (x.shape()[0], x.shape()[1], x.shape()[2], x.shape()[3]);
        if let Some(lr) = &p.low_rank {
            // Qwen4-Exp: the layer reads the gated mean of the streams and writes back with `2 sigmoid(inject / n)`.
            let (mixed, xn) = lr.mix(x, self.norm_eps)?;
            let inject = lr.inject.as_ref().ok_or_else(|| {
                Error::Msg("a layer's hyper-connection needs its injection".into())
            })?;
            let post = xn
                .matmul(&inject.t()?)?
                .mul_scalar(1.0 / n as f64)?
                .sigmoid()?
                .mul_scalar(2.0)?;
            return Ok((mixed, HcMix { post, comb: None }));
        }
        let flat = x.reshape(&[b, s, n * d])?;
        let normed = flat.mul(
            &flat
                .square()?
                .mean_axis(-1, true)?
                .add_scalar(self.norm_eps)?
                .rsqrt()?,
        )?;
        let mixes = normed.matmul(&p.w.t()?)?; // [b, s, 2n] or [b, s, (2 + n) n]
        if let Some(iters) = self.sinkhorn {
            let base = |from: usize, len: usize| -> Result<Tensor> {
                p.base
                    .narrow(0, from, len)?
                    .reshape(&[1, 1, len])?
                    .expand(&[b, s, len])?
                    .contiguous()
            };
            let pre = mixes
                .narrow(2, 0, n)?
                .contiguous()?
                .mul_scalar(f64::from(p.scale[0]))?
                .add(&base(0, n)?)?
                .sigmoid()?
                .add_scalar(f64::from(self.eps))?;
            let post = mixes
                .narrow(2, n, n)?
                .contiguous()?
                .mul_scalar(f64::from(p.scale[1]))?
                .add(&base(n, n)?)?
                .sigmoid()?
                .mul_scalar(2.0)?;
            let raw = mixes
                .narrow(2, 2 * n, n * n)?
                .contiguous()?
                .mul_scalar(f64::from(p.scale[2]))?
                .add(&base(2 * n, n * n)?)?
                .to_vec::<f32>()?;
            // raw[t][dst + n * src]; Sinkhorn: softmax over dst, +eps, then alternate source / destination sums.
            let mut comb = vec![0f32; b * s * n * n];
            for t in 0..b * s {
                let mut c = vec![0f32; n * n]; // c[dst * n + src]
                for src in 0..n {
                    let col: Vec<f32> = (0..n).map(|dst| raw[t * n * n + dst + n * src]).collect();
                    let max = col.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                    let ex: Vec<f32> = col.iter().map(|v| (v - max).exp()).collect();
                    let sum: f32 = ex.iter().sum();
                    for dst in 0..n {
                        c[dst * n + src] = ex[dst] / sum + self.eps;
                    }
                }
                let norm_cols = |c: &mut [f32]| {
                    for dst in 0..n {
                        let sum: f32 = (0..n).map(|src| c[dst * n + src]).sum::<f32>() + self.eps;
                        (0..n).for_each(|src| c[dst * n + src] /= sum);
                    }
                };
                let norm_rows = |c: &mut [f32]| {
                    for src in 0..n {
                        let sum: f32 = (0..n).map(|dst| c[dst * n + src]).sum::<f32>() + self.eps;
                        (0..n).for_each(|dst| c[dst * n + src] /= sum);
                    }
                };
                norm_cols(&mut c);
                for _ in 1..iters {
                    norm_rows(&mut c);
                    norm_cols(&mut c);
                }
                comb[t * n * n..(t + 1) * n * n].copy_from_slice(&c);
            }
            let comb = Tensor::from_slice_on(&comb, &[b, s, n, n], &x.device())?;
            return Ok((
                reduce_streams(x, &pre)?,
                HcMix {
                    post,
                    comb: Some(comb),
                },
            ));
        }
        let base = |from: usize| -> Result<Tensor> {
            p.base
                .narrow(0, from, n)?
                .reshape(&[1, 1, n])?
                .expand(&[b, s, n])?
                .contiguous()
        };
        let pre = mixes
            .narrow(2, 0, n)?
            .contiguous()?
            .mul_scalar(f64::from(p.scale[0]))?
            .add(&base(0)?)?
            .sigmoid()?
            .add_scalar(f64::from(self.eps))?;
        let post = mixes
            .narrow(2, n, n)?
            .contiguous()?
            .mul_scalar(f64::from(p.scale[1]))?
            .add(&base(n)?)?
            .sigmoid()?
            .mul_scalar(f64::from(self.magnitude))?
            .add_scalar(f64::from(self.eps))?;
        Ok((reduce_streams(x, &pre)?, HcMix { post, comb: None }))
    }
}

/// DeepSeek-V4 attention: one shared key/value head (`K = V`), a per-head RMS-normed query, RoPE on the last
/// `rope` dimensions only, per-head sinks, a sliding window over the raw keys, optionally followed by compressed
/// blocks (a softmax-pooled window of `ratio` tokens, with the previous window overlapping for ratio 4), and a
/// grouped low-rank output projection. The output's rotary part is rotated back by the query position first.
///
/// The compressed-block indexer is not run: every complete block is attended, which is exactly what its top-k
/// selects while the number of blocks stays within `indexer.top_k`.
pub struct Dsv4Attn {
    pub q_a: Tensor,
    pub q_a_norm: Norm,
    pub q_b: Tensor,
    pub kv: Tensor,
    pub kv_norm: Norm,
    pub sinks: Vec<f32>,
    /// `[groups, lora, heads_per_group * head_dim]` and `[d, groups * lora]`
    pub wo_a: Tensor,
    pub wo_b: Tensor,
    pub heads: usize,
    pub head_dim: usize,
    pub rope: usize,
    pub groups: usize,
    pub lora: usize,
    pub window: usize,
    pub eps: f64,
    /// RoPE schedule of this layer's queries / keys.
    pub sched: Dsv4Rope,
    pub comp: Option<Dsv4Comp>,
    /// The compressed-block indexer of a ratio-4 layer.
    pub lid: Option<Dsv4Lid>,
}

/// Picks which compressed blocks a query attends: scores them with its own pooled keys (`comp`, over `dim` channels).
pub struct Dsv4Lid {
    /// `[heads * dim, q_lora]` and `[heads, d]`
    pub q_b: Tensor,
    pub proj: Tensor,
    pub heads: usize,
    pub dim: usize,
    pub top_k: usize,
    pub comp: Dsv4Comp,
}

impl Dsv4Comp {
    /// Width of one half (values or scores) of a state row over `dim` channels.
    fn cw(&self, dim: usize) -> usize {
        if self.overlap { 2 * dim } else { dim }
    }

    /// Writes every token's value row and score row (plus the position-in-window bias) at `col0` of its state row.
    #[allow(clippy::too_many_arguments)]
    fn fill(
        &self,
        x2: &Tensor,
        seq_pos: usize,
        s: usize,
        dim: usize,
        state: &mut [f32],
        width: usize,
        col0: usize,
    ) -> Result<()> {
        let cw = self.cw(dim);
        let rows = x2.shape()[0];
        let kvs = x2.matmul(&self.wkv.t()?)?.to_vec::<f32>()?;
        let mut sc = x2.matmul(&self.wgate.t()?)?.to_vec::<f32>()?;
        let ape = self.ape.to_vec::<f32>()?;
        for r in 0..rows {
            let row = (seq_pos + r % s) % self.ratio;
            for k in 0..cw {
                sc[r * cw + k] += ape[row * cw + k];
            }
            state[r * width + col0..r * width + col0 + cw]
                .copy_from_slice(&kvs[r * cw..(r + 1) * cw]);
            state[r * width + col0 + cw..r * width + col0 + 2 * cw]
                .copy_from_slice(&sc[r * cw..(r + 1) * cw]);
        }
        Ok(())
    }

    /// The compressed block of every complete window of sequence `bi`: a softmax over the window rows per channel
    /// (the previous window's first half joins in for the overlapping form), RMS norm, then RoPE at the window start.
    #[allow(clippy::too_many_arguments)]
    fn pool(
        &self,
        state: &[f32],
        width: usize,
        col0: usize,
        bi: usize,
        total: usize,
        dim: usize,
        rope: usize,
        eps: f32,
    ) -> Result<Vec<Vec<f32>>> {
        let cw = self.cw(dim);
        let norm = self.norm.weight.to_vec::<f32>()?;
        let mut blocks = Vec::new();
        for j in 0..total / self.ratio {
            let start = (j * self.ratio) as isize;
            // (column offset, source position) of every pooled row
            let mut rows: Vec<(usize, isize)> = Vec::new();
            if self.overlap {
                rows.extend((0..self.ratio as isize).map(|r| (0, start - self.ratio as isize + r)));
                rows.extend((0..self.ratio as isize).map(|r| (dim, start + r)));
            } else {
                rows.extend((0..self.ratio as isize).map(|r| (0, start + r)));
            }
            let mut pooled = vec![0f32; dim];
            for (ch, slot) in pooled.iter_mut().enumerate() {
                let vals: Vec<(f32, f32)> = rows
                    .iter()
                    .map(|&(off, pos)| {
                        if pos < 0 {
                            (0.0, f32::NEG_INFINITY)
                        } else {
                            let base = (bi * total + pos as usize) * width + col0;
                            (state[base + off + ch], state[base + cw + off + ch])
                        }
                    })
                    .collect();
                let max = vals.iter().map(|v| v.1).fold(f32::NEG_INFINITY, f32::max);
                let ex: Vec<f32> = vals.iter().map(|v| (v.1 - max).exp()).collect();
                let sum: f32 = ex.iter().sum();
                *slot = vals.iter().zip(&ex).map(|(v, e)| v.0 * e / sum).sum();
            }
            rms_unit(&mut pooled, Some(&norm), eps);
            rope_pairs(&mut pooled[dim - rope..], start as f32, &self.rope, false);
            blocks.push(pooled);
        }
        Ok(blocks)
    }
}

pub struct Dsv4Comp {
    pub ratio: usize,
    pub overlap: bool,
    /// `[coff * head_dim, d]` each, `[ratio, coff * head_dim]`
    pub wkv: Tensor,
    pub wgate: Tensor,
    pub ape: Tensor,
    pub norm: Norm,
    /// Rotary schedule of the compressed blocks (the layer's own, since compressing layers rotate with it).
    pub rope: Dsv4Rope,
}

/// The rotary schedule of a DeepSeek-V4 layer: `base^(-2i / n)` frequencies with an optional linear position scale or
/// YaRN blend of interpolated and extrapolated angles (its magnitude factor cancels in this architecture).
#[derive(Debug, Clone, Copy)]
pub struct Dsv4Rope {
    pub base: f32,
    /// Linear position scale (`1 / factor`), 1 when unscaled.
    pub scale: f32,
    /// YaRN `(low, high, ext_factor)` correction range in pair indices.
    pub yarn: Option<(f32, f32, f32)>,
}

impl Dsv4Rope {
    fn theta(&self, pos: f32, i: usize, n: usize) -> f32 {
        let extrap = pos * self.base.powf(-(2.0 * i as f32) / n as f32);
        let interp = extrap * self.scale;
        match self.yarn {
            None => interp,
            Some((low, high, ext)) => {
                let ramp =
                    (1.0 - ((i as f32 - low) / (high - low).max(0.001)).clamp(0.0, 1.0)) * ext;
                interp * (1.0 - ramp) + extrap * ramp
            }
        }
    }
}

/// Rotates adjacent pairs of `v` by the schedule's angles for position `pos`; `inverse` undoes it.
fn rope_pairs(v: &mut [f32], pos: f32, rope: &Dsv4Rope, inverse: bool) {
    let n = v.len();
    for i in 0..n / 2 {
        let theta = rope.theta(pos, i, n);
        let (s, c) = theta.sin_cos();
        let s = if inverse { -s } else { s };
        let (x0, x1) = (v[2 * i], v[2 * i + 1]);
        v[2 * i] = x0 * c - x1 * s;
        v[2 * i + 1] = x0 * s + x1 * c;
    }
}

fn rms_unit(v: &mut [f32], w: Option<&[f32]>, eps: f32) {
    let ms = v.iter().map(|x| x * x).sum::<f32>() / v.len() as f32;
    let inv = 1.0 / (ms + eps).sqrt();
    for (i, x) in v.iter_mut().enumerate() {
        *x *= inv * w.map_or(1.0, |w| w[i]);
    }
}

impl Dsv4Attn {
    fn forward(
        &self,
        x: &Tensor,
        seq_pos: usize,
        cache: &mut Option<(Tensor, Tensor)>,
    ) -> Result<Tensor> {
        let (b, s, d) = (x.shape()[0], x.shape()[1], x.shape()[2]);
        let (h, hd, rope) = (self.heads, self.head_dim, self.rope);
        let eps = self.eps as f32;
        let x2 = x.reshape(&[b * s, d])?;
        let mm = |a: &Tensor, w: &Tensor| -> Result<Tensor> { a.matmul(&w.t()?) };

        let qr = self.q_a_norm.forward(&mm(&x2, &self.q_a)?)?;
        let mut q = mm(&qr, &self.q_b)?.to_vec::<f32>()?; // [b*s, h*hd]
        let mut kvn = self.kv_norm.forward(&mm(&x2, &self.kv)?)?.to_vec::<f32>()?; // [b*s, hd]
        for r in 0..b * s {
            let pos = (seq_pos + r % s) as f32;
            for head in 0..h {
                let qh = &mut q[(r * h + head) * hd..(r * h + head + 1) * hd];
                rms_unit(qh, None, eps);
                rope_pairs(&mut qh[hd - rope..], pos, &self.sched, false);
            }
            rope_pairs(
                &mut kvn[r * hd + hd - rope..(r + 1) * hd],
                pos,
                &self.sched,
                false,
            );
        }

        // Compressor inputs per token (value rows, then score rows with the position-in-window bias) for the attention
        // compressor and, after it, the indexer's compressor.
        let attn_w = self.comp.as_ref().map_or(0, |c| 2 * c.cw(hd));
        let lid_w = self.lid.as_ref().map_or(0, |l| 2 * l.comp.cw(l.dim));
        let width = (attn_w + lid_w).max(1);
        let mut state = vec![0f32; b * s * width];
        if let Some(c) = &self.comp {
            c.fill(&x2, seq_pos, s, hd, &mut state, width, 0)?;
        }
        if let Some(l) = &self.lid {
            l.comp
                .fill(&x2, seq_pos, s, l.dim, &mut state, width, attn_w)?;
        }

        // History: raw keys and compressor inputs for every position so far. The slot is
        // caller-visible, so validate rank and exact volume (a short/wrong-rank cache used
        // to panic in the shape read and slice copies below) and the seq_pos == past
        // invariant the key gathering assumes.
        let (past_kv, past_state, past) = match cache.take() {
            Some((k, st)) => {
                let n =
                    k.shape().get(2).copied().ok_or_else(|| {
                        Error::Msg("dsv4 cache: key cache needs rank >= 3".into())
                    })?;
                let kv = k.to_vec::<f32>()?;
                let sv = st.to_vec::<f32>()?;
                let (kv_need, st_need) = (
                    b.checked_mul(n).and_then(|v| v.checked_mul(hd)),
                    b.checked_mul(n).and_then(|v| v.checked_mul(width)),
                );
                if kv_need.is_none_or(|e| kv.len() != e) || st_need.is_none_or(|e| sv.len() != e) {
                    return Err(Error::Msg(format!(
                        "dsv4 cache mismatch: have {} + {} values, need [b={b}, past={n}] keys of width {hd} and states of width {width}",
                        kv.len(),
                        sv.len()
                    )));
                }
                (kv, sv, n)
            }
            None => (Vec::new(), Vec::new(), 0),
        };
        if seq_pos != past {
            return Err(Error::Msg(format!(
                "dsv4 cache: seq_pos {seq_pos} disagrees with the cache length {past}"
            )));
        }
        let total = past + s;
        let mut all_kv = vec![0f32; b * total * hd];
        let mut all_state = vec![0f32; b * total * width];
        for bi in 0..b {
            all_kv[bi * total * hd..bi * total * hd + past * hd]
                .copy_from_slice(&past_kv[bi * past * hd..(bi + 1) * past * hd]);
            all_kv[(bi * total + past) * hd..(bi + 1) * total * hd]
                .copy_from_slice(&kvn[bi * s * hd..(bi + 1) * s * hd]);
            all_state[bi * total * width..bi * total * width + past * width]
                .copy_from_slice(&past_state[bi * past * width..(bi + 1) * past * width]);
            all_state[(bi * total + past) * width..(bi + 1) * total * width]
                .copy_from_slice(&state[bi * s * width..(bi + 1) * s * width]);
        }

        // The indexer's queries (rotated like the attention's) and per-head weights, for the layers that have one.
        let lid_q = match &self.lid {
            Some(l) => {
                let mut lq = mm(&qr, &l.q_b)?.to_vec::<f32>()?; // [b*s, heads * dim]
                for r in 0..b * s {
                    let pos = (seq_pos + r % s) as f32;
                    for head in 0..l.heads {
                        let qh =
                            &mut lq[(r * l.heads + head) * l.dim..(r * l.heads + head + 1) * l.dim];
                        rope_pairs(&mut qh[l.dim - rope..], pos, &l.comp.rope, false);
                    }
                }
                let w = mm(&x2, &l.proj)?
                    .mul_scalar(1.0 / ((l.dim * l.heads) as f64).sqrt())?
                    .to_vec::<f32>()?;
                Some((lq, w))
            }
            None => None,
        };
        let mut out = vec![0f32; b * s * h * hd];
        for bi in 0..b {
            // Compressed blocks of this sequence (attention keys, and the indexer's keys when the layer has an indexer).
            let blocks = match &self.comp {
                Some(c) => c.pool(&all_state, width, 0, bi, total, hd, rope, eps)?,
                None => Vec::new(),
            };
            let lid_blocks = match &self.lid {
                Some(l) => l
                    .comp
                    .pool(&all_state, width, attn_w, bi, total, l.dim, rope, eps)?,
                None => Vec::new(),
            };
            for t in 0..s {
                let pos = seq_pos + t;
                let lo = (pos + 1).saturating_sub(self.window);
                let n_comp = self
                    .comp
                    .as_ref()
                    .map_or(0, |c| ((pos + 1) / c.ratio).min(blocks.len()));
                // Blocks the indexer lets through: the best `top_k` by sum_h w_h * relu(q_h . k_block).
                let mut picked: Vec<usize> = (0..n_comp).collect();
                if let (Some(l), Some((lq, lw))) = (&self.lid, &lid_q) {
                    if n_comp > l.top_k {
                        let r = bi * s + t;
                        let score: Vec<f32> = (0..n_comp)
                            .map(|j| {
                                (0..l.heads)
                                    .map(|hh| {
                                        let qh = &lq[(r * l.heads + hh) * l.dim
                                            ..(r * l.heads + hh + 1) * l.dim];
                                        let dot: f32 =
                                            qh.iter().zip(&lid_blocks[j]).map(|(a, b)| a * b).sum();
                                        lw[r * l.heads + hh] * dot.max(0.0)
                                    })
                                    .sum()
                            })
                            .collect();
                        picked.sort_by(|&a, &b| score[b].total_cmp(&score[a]).then(a.cmp(&b)));
                        picked.truncate(l.top_k);
                    }
                }
                for head in 0..h {
                    let qh = &q[((bi * s + t) * h + head) * hd..((bi * s + t) * h + head + 1) * hd];
                    let mut keys: Vec<&[f32]> = (lo..=pos)
                        .map(|p| &all_kv[(bi * total + p) * hd..(bi * total + p + 1) * hd])
                        .collect();
                    keys.extend(picked.iter().map(|&j| &blocks[j][..]));
                    let mut logits: Vec<f32> = keys
                        .iter()
                        .map(|k| {
                            qh.iter().zip(*k).map(|(a, b)| a * b).sum::<f32>() / (hd as f32).sqrt()
                        })
                        .collect();
                    logits.push(self.sinks[head]);
                    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                    let ex: Vec<f32> = logits.iter().map(|v| (v - max).exp()).collect();
                    let sum: f32 = ex.iter().sum();
                    let o = &mut out
                        [((bi * s + t) * h + head) * hd..((bi * s + t) * h + head + 1) * hd];
                    for (k, e) in keys.iter().zip(&ex) {
                        for (acc, v) in o.iter_mut().zip(*k) {
                            *acc += v * e / sum;
                        }
                    }
                    rope_pairs(&mut o[hd - rope..], pos as f32, &self.sched, true);
                }
            }
        }
        *cache = Some((
            Tensor::from_slice(&all_kv, &[b, 1, total, hd])?,
            Tensor::from_slice(&all_state, &[b, total, width])?,
        ));

        // Grouped output projection: each group of heads goes through its own low-rank matrix.
        let gd = h / self.groups * hd;
        let out = Tensor::from_slice_on(&out, &[b * s, self.groups, gd], &x.device())?;
        let mut parts = Vec::with_capacity(self.groups);
        for g in 0..self.groups {
            let wa = self.wo_a.narrow(0, g, 1)?.reshape(&[self.lora, gd])?;
            parts.push(
                out.narrow(1, g, 1)?
                    .reshape(&[b * s, gd])?
                    .matmul(&wa.t()?)?,
            );
        }
        Tensor::cat(&parts, 1)?
            .matmul(&self.wo_b.t()?)?
            .reshape(&[b, s, d])
    }
}

/// Qwen4-Exp's per-layer n-gram hash embeddings: every token gathers `heads` rows of one shared table, picked by
/// hashing it together with the tokens before it, and the layer folds them into the hyper-connection streams.
pub struct PleModel {
    /// `[rows, head_dim]`
    pub table: Tensor,
    pub head_dim: usize,
    pub ngram: usize,
    pub heads_per_ngram: usize,
    pub kernel: usize,
    pub eos: i64,
    pub multipliers: Vec<u64>,
    pub offsets: Vec<u64>,
    pub vocab_sizes: Vec<u64>,
    pub layers: Vec<PleLayer>,
}

pub struct PleLayer {
    pub layer: usize,
    /// `[streams * d, d_embed]` and `[d, d_embed]`
    pub key: Tensor,
    pub value: Tensor,
    /// `[streams, d]` each
    pub norm_key: Tensor,
    pub norm_query: Tensor,
    pub norm_conv: Tensor,
    /// Depthwise conv taps, `[streams * d][kernel]`.
    pub conv: Vec<f32>,
}

impl PleModel {
    pub fn n_heads(&self) -> usize {
        (self.ngram - 1) * self.heads_per_ngram
    }

    /// Tokens of context a position needs besides its own.
    pub fn n_prev(&self) -> usize {
        self.ngram - 1
    }

    /// Conv history rows kept between calls.
    pub fn hist(&self) -> usize {
        (self.kernel - 1) * self.ngram
    }

    /// `ids`: this call's tokens per sequence, `prev`: the tokens before them (oldest first, `-1` = none).
    /// Returns `[b, s, heads * head_dim]`.
    fn embed(&self, ids: &[Vec<i64>], prev: &[Vec<i64>]) -> Result<Tensor> {
        let (b, s, heads) = (ids.len(), ids[0].len(), self.n_heads());
        let n_prev = self.n_prev();
        let mut rows = Vec::with_capacity(b * s * heads);
        for (seq, before) in ids.iter().zip(prev) {
            let all: Vec<i64> = before.iter().chain(seq).copied().collect();
            for t in 0..s {
                let at = n_prev + t; // index of the token in `all`
                // An EOS (or a missing predecessor) cuts the window: everything older reads as EOS.
                let mut ctx = vec![all[at]];
                let mut cut = false;
                for back in 1..self.ngram {
                    let tok = if cut { -1 } else { all[at - back] };
                    cut = cut || tok < 0 || tok == self.eos;
                    ctx.push(if cut { self.eos } else { tok });
                }
                for n in 2..=self.ngram {
                    let mut mixed = (ctx[0] as u64).wrapping_mul(self.multipliers[0]);
                    for (j, &c) in ctx.iter().enumerate().take(n).skip(1) {
                        mixed ^= (c as u64).wrapping_mul(self.multipliers[j]);
                    }
                    for g in 0..self.heads_per_ngram {
                        let h = (n - 2) * self.heads_per_ngram + g;
                        rows.push((mixed % self.vocab_sizes[h] + self.offsets[h]) as usize);
                    }
                }
            }
        }
        let picked = rows
            .iter()
            .map(|&r| self.table.narrow(0, r, 1))
            .collect::<Result<Vec<_>>>()?;
        Tensor::cat(&picked, 0)?.reshape(&[b, s, heads * self.head_dim])
    }

    /// Folds the hash embedding into `hidden` (`[b, s, streams, d]`). `hist` is the conv history `[b, hist, streams * d]`
    /// from the previous call; returns the new hidden state and the new history.
    fn apply(
        &self,
        l: &PleLayer,
        hidden: &Tensor,
        emb: &Tensor,
        hist: Option<&Tensor>,
        eps: f64,
    ) -> Result<(Tensor, Tensor)> {
        let (b, s, n, d) = (
            hidden.shape()[0],
            hidden.shape()[1],
            hidden.shape()[2],
            hidden.shape()[3],
        );
        let width = n * d;
        let norm = |x: &Tensor, w: &Tensor| -> Result<Tensor> {
            x.mul(&x.square()?.mean_axis(-1, true)?.add_scalar(eps)?.rsqrt()?)?
                .mul(&w.reshape(&[1, 1, n, d])?.expand(&[b, s, n, d])?)
        };
        let key = norm(
            &emb.matmul(&l.key.t()?)?.reshape(&[b, s, n, d])?,
            &l.norm_key,
        )?;
        let value = emb.matmul(&l.value.t()?)?;
        let query = norm(hidden, &l.norm_query)?;

        // Per-stream dot product, then a signed square root before the sigmoid.
        let score = key
            .mul(&query)?
            .sum_axis(-1, false)?
            .mul_scalar(1.0 / (d as f64).sqrt())?
            .to_vec::<f32>()?;
        let gate: Vec<f32> = score
            .iter()
            .map(|&v| {
                let signed = if v > 0.0 {
                    1.0
                } else if v < 0.0 {
                    -1.0
                } else {
                    0.0
                };
                1.0 / (1.0 + (-(signed * v.abs().max(1e-6).sqrt())).exp())
            })
            .collect();
        let gate = Tensor::from_slice_on(&gate, &[b, s, n, 1], &hidden.device())?;
        let gated = value
            .reshape(&[b, s, 1, d])?
            .expand(&[b, s, n, d])?
            .mul(&gate.expand(&[b, s, n, d])?)?;
        let normalized = norm(&gated, &l.norm_conv)?.reshape(&[b, s, width])?;

        // Depthwise causal conv dilated by the n-gram size, over the history plus this call's rows.
        let (hist_len, kern, dil) = (self.hist(), self.kernel, self.ngram);
        let past = match hist {
            Some(h) => h.to_vec::<f32>()?,
            None => vec![0.0; b * hist_len * width],
        };
        let now = normalized.to_vec::<f32>()?;
        let mut padded = vec![0f32; b * (hist_len + s) * width];
        for bi in 0..b {
            let dst = &mut padded[bi * (hist_len + s) * width..(bi + 1) * (hist_len + s) * width];
            dst[..hist_len * width]
                .copy_from_slice(&past[bi * hist_len * width..(bi + 1) * hist_len * width]);
            dst[hist_len * width..].copy_from_slice(&now[bi * s * width..(bi + 1) * s * width]);
        }
        let mut conv = vec![0f32; b * s * width];
        for bi in 0..b {
            let base = bi * (hist_len + s) * width;
            for t in 0..s {
                for c in 0..width {
                    let mut acc = 0f32;
                    for k in 0..kern {
                        let row = hist_len + t - (kern - 1 - k) * dil;
                        acc += l.conv[c * kern + k] * padded[base + row * width + c];
                    }
                    conv[(bi * s + t) * width + c] = acc / (1.0 + (-acc).exp());
                }
            }
        }
        let conv = Tensor::from_slice_on(&conv, &[b, s, n, d], &hidden.device())?;
        let mut tail = vec![0f32; b * hist_len * width];
        for bi in 0..b {
            let base = bi * (hist_len + s) * width + s * width;
            tail[bi * hist_len * width..(bi + 1) * hist_len * width]
                .copy_from_slice(&padded[base..base + hist_len * width]);
        }
        let new_hist = Tensor::from_slice_on(&tail, &[b, hist_len, width], &hidden.device())?;
        Ok((hidden.add(&gated.add(&conv)?)?, new_hist))
    }
}

/// `sum_i w[..., i] * x[..., i, :]` for `x: [b, s, n, d]`, `w: [b, s, n]`.
fn reduce_streams(x: &Tensor, w: &Tensor) -> Result<Tensor> {
    let (b, s, n, d) = (x.shape()[0], x.shape()[1], x.shape()[2], x.shape()[3]);
    w.reshape(&[b, s, n, 1])?
        .expand(&[b, s, n, d])?
        .mul(x)?
        .sum_axis(2, false)
}

/// `streams_i + out * post_i` (plain iHC) or `sum_src comb[i][src] * streams_src + out * post_i`, `[b, s, n, d]`.
fn hc_post(out: &Tensor, streams: &Tensor, mix: &HcMix) -> Result<Tensor> {
    let (b, s, n, d) = (
        streams.shape()[0],
        streams.shape()[1],
        streams.shape()[2],
        streams.shape()[3],
    );
    let o = out.reshape(&[b, s, 1, d])?.expand(&[b, s, n, d])?;
    let written = o.mul(&mix.post.reshape(&[b, s, n, 1])?.expand(&[b, s, n, d])?)?;
    match &mix.comb {
        Some(comb) => comb.matmul(&streams.contiguous()?)?.add(&written),
        None => streams.add(&written),
    }
}

/// Kimi-K3's attention residual: a softmax-weighted sum of the banked checkpoints and the current stream `cur`
/// (`[b, s, d]`), scored per token by `sum(rms_norm(t) * w)`. With nothing banked the stream passes through.
fn res_mix(stack: &[Tensor], cur: &Tensor, w: &Tensor, eps: f64) -> Result<Tensor> {
    if stack.is_empty() {
        return Ok(cur.clone());
    }
    let score = |t: &Tensor| -> Result<Tensor> {
        let r = t.mul(&t.square()?.mean_axis(-1, true)?.add_scalar(eps)?.rsqrt()?)?;
        r.mul(w)?.sum_axis(-1, true)
    };
    let n = stack.len();
    let mut scores = stack.iter().map(score).collect::<Result<Vec<_>>>()?;
    scores.push(score(cur)?);
    let probs = Tensor::cat(&scores, 2)?.softmax(2)?;
    let mut out = cur.mul(&probs.narrow(2, n, 1)?.expand(cur.shape())?)?;
    for (i, t) in stack.iter().enumerate() {
        out = out.add(&t.mul(&probs.narrow(2, i, 1)?.expand(t.shape())?)?)?;
    }
    Ok(out)
}

/// llama.cpp's limited SwiGLU: `min(silu(g), limit) * clamp(u, -limit, limit)`.
/// A negative `limit` selects the gate-first form `silu(min(g, L)) * clamp(u, -L, L)` (HY-V4, DeepSeek-V4).
fn clamped_swiglu(g: &Tensor, u: &Tensor, limit: f32) -> Result<Tensor> {
    let l = f64::from(limit.abs());
    let u = u.clamp(-l, l)?;
    if limit < 0.0 {
        g.clamp(-1e30, l)?.silu()?.mul(&u)
    } else {
        g.silu()?.clamp(-1e30, l)?.mul(&u)
    }
}

fn swiglu(x: &Tensor, gate: &Tensor, up: &Tensor, down: &Tensor) -> Result<Tensor> {
    // x: [1, d]; gate/up: [ff, d]; down: [d, ff]
    let h = x.matmul(&gate.t()?)?.silu()?.mul(&x.matmul(&up.t()?)?)?;
    h.matmul(&down.t()?)
}

impl MoeFfn {
    fn expert(&self, x: &Tensor, e: usize) -> Result<Tensor> {
        let out = self.expert_raw(x, e)?;
        match &self.post_exp_norm {
            Some((w, eps)) => {
                let y = out.mul(w)?;
                y.mul(&y.square()?.mean_axis(-1, true)?.add_scalar(*eps)?.rsqrt()?)
            }
            None => Ok(out),
        }
    }

    fn expert_raw(&self, x: &Tensor, e: usize) -> Result<Tensor> {
        let (ff, d) = (self.up.shape()[1], self.up.shape()[2]);
        let up = self.up.narrow(0, e, 1)?.reshape(&[ff, d])?;
        let down = self.down.narrow(0, e, 1)?.reshape(&[d, ff])?;
        let pick = |s: &Option<Vec<f32>>| s.as_ref().map(|v| v[e]);
        let row = |b: &Option<Tensor>| -> Result<Option<Tensor>> {
            b.as_ref()
                .map(|b| b.narrow(0, e, 1)?.contiguous())
                .transpose()
        };
        if self.act == MoeAct::ReluSqr {
            let h = scaled(x.matmul(&up.t()?)?, pick(&self.up_scale))?
                .relu()?
                .square()?;
            return scaled(h.matmul(&down.t()?)?, pick(&self.down_scale));
        }
        let gate = self.gate.narrow(0, e, 1)?.reshape(&[ff, d])?;
        let mut g = scaled(x.matmul(&gate.t()?)?, pick(&self.gate_scale))?;
        if let Some(b) = row(&self.gate_bias)? {
            g = g.add(&b)?;
        }
        let mut u = scaled(x.matmul(&up.t()?)?, pick(&self.up_scale))?;
        if let Some(b) = row(&self.up_bias)? {
            u = u.add(&b)?;
        }
        let h = match self.act {
            MoeAct::Silu if self.limit.abs() > 1e-6 => clamped_swiglu(&g, &u, self.limit)?,
            MoeAct::Silu => g.silu()?.mul(&u)?,
            MoeAct::Gelu => Activation::GeluTanh.apply(&g)?.mul(&u)?,
            MoeAct::Relu => g.relu()?.mul(&u)?,
            MoeAct::ReluSqr => unreachable!("handled above"),
            MoeAct::Situ(b, l) => Activation::Situ(b, l)
                .apply(&g)?
                .mul(&Activation::Situ(b, l).up(&u)?)?,
            MoeAct::SwigluOai => {
                let (alpha, limit) = (1.702, 7.0);
                let g = g.clamp(-1e30, limit)?;
                let u = u.clamp(-limit, limit)?;
                g.mul(&g.mul_scalar(alpha)?.sigmoid()?)?
                    .mul(&u.add_scalar(1.0)?)?
            }
        };
        let mut out = scaled(h.matmul(&down.t()?)?, pick(&self.down_scale))?;
        if let Some(b) = row(&self.down_bias)? {
            out = out.add(&b)?;
        }
        Ok(out)
    }

    /// `x`: `[..., d]`; every token is routed independently.
    ///
    /// ponytail: one tiny matmul per (token, selected expert); group tokens by expert for prompt-heavy use.
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        self.forward_routed(x, x)
    }

    /// Like [`forward`](Self::forward) but the router reads `router_in` (same shape as `x`).
    pub fn forward_routed(&self, x: &Tensor, router_in: &Tensor) -> Result<Tensor> {
        self.forward_routed_forced(x, router_in, None)
    }

    /// [`forward_routed`](Self::forward_routed) with the experts of every token fixed by `forced` (`top_k` per token,
    /// hash-routed layers); the router still weights them.
    pub fn forward_routed_forced(
        &self,
        x: &Tensor,
        router_in: &Tensor,
        forced: Option<&[usize]>,
    ) -> Result<Tensor> {
        let Some(l) = &self.latent else {
            return self.forward_inner(x, router_in, forced);
        };
        let shape = x.shape().to_vec();
        let d = *shape
            .last()
            .ok_or_else(|| Error::Msg("MoE input needs rank >= 1".into()))?;
        if d == 0 {
            return Err(Error::Msg("MoE input has a zero-size last dim".into()));
        }
        let tokens = x.numel() / d;
        let latent = l.down.shape()[0];
        let mut lat = x.reshape(&[tokens, d])?.matmul(&l.down.t()?)?;
        if let Some(n) = &l.pre_norm {
            lat = n.forward(&lat)?;
        }
        let mut lat_shape = shape.clone();
        *lat_shape.last_mut().unwrap_or(&mut 0) = latent;
        let out = self.forward_inner(&lat.reshape(&lat_shape)?, router_in, forced)?;
        let out = out.reshape(&[tokens, latent])?;
        let out = match &l.mid_norm {
            Some(n) => n.forward(&out)?,
            None => out,
        };
        let mut out = out.matmul(&l.up.t()?)?;
        if self.shared.is_some() {
            // Nemotron-H: the shared expert reads the full-width input, not the latent one.
            let flat = x.reshape(&[tokens, d])?;
            let rows = (0..tokens)
                .map(|tok| {
                    self.shared_out(&flat.narrow(0, tok, 1)?, d)
                        .map(|s| s.expect("checked above"))
                })
                .collect::<Result<Vec<_>>>()?;
            out = out.add(&Tensor::cat(&rows, 0)?)?;
        }
        out.reshape(&shape)
    }

    /// The always-on shared expert's output for one token `xt` of width `d` (`None` when the layer has none).
    fn shared_out(&self, xt: &Tensor, d: usize) -> Result<Option<Tensor>> {
        let Some(sh) = &self.shared else {
            return Ok(None);
        };
        let [sg, su, sd] = sh.scales.unwrap_or([1.0; 3]);
        let mut s = if sh.relu_sqr {
            let h = xt
                .matmul(&sh.up.t()?)?
                .mul_scalar(f64::from(su))?
                .relu()?
                .square()?;
            h.matmul(&sh.down.t()?)?.mul_scalar(f64::from(sd))?
        } else if self.act == MoeAct::SwigluOai {
            let a = Activation::SwigluOai;
            let g = xt.matmul(&sh.gate.t()?)?.mul_scalar(f64::from(sg))?;
            let u = xt.matmul(&sh.up.t()?)?.mul_scalar(f64::from(su))?;
            a.apply(&g)?
                .mul(&a.up(&u)?)?
                .matmul(&sh.down.t()?)?
                .mul_scalar(f64::from(sd))?
        } else if let MoeAct::Situ(b, l) = self.act {
            let a = Activation::Situ(b, l);
            let g = xt.matmul(&sh.gate.t()?)?.mul_scalar(f64::from(sg))?;
            let u = xt.matmul(&sh.up.t()?)?.mul_scalar(f64::from(su))?;
            a.apply(&g)?
                .mul(&a.up(&u)?)?
                .matmul(&sh.down.t()?)?
                .mul_scalar(f64::from(sd))?
        } else if sh.limit.abs() > 1e-6 {
            let g = xt.matmul(&sh.gate.t()?)?.mul_scalar(f64::from(sg))?;
            let u = xt.matmul(&sh.up.t()?)?.mul_scalar(f64::from(su))?;
            clamped_swiglu(&g, &u, sh.limit)?
                .matmul(&sh.down.t()?)?
                .mul_scalar(f64::from(sd))?
        } else if sh.scales.is_none() {
            swiglu(xt, &sh.gate, &sh.up, &sh.down)?
        } else {
            let g = xt
                .matmul(&sh.gate.t()?)?
                .mul_scalar(f64::from(sg))?
                .silu()?;
            let u = xt.matmul(&sh.up.t()?)?.mul_scalar(f64::from(su))?;
            g.mul(&u)?
                .matmul(&sh.down.t()?)?
                .mul_scalar(f64::from(sd))?
        };
        if let Some(gi) = &sh.gate_inp {
            s = s.mul(&xt.matmul(&gi.reshape(&[d, 1])?)?.sigmoid()?)?;
        }
        Ok(Some(s))
    }

    fn forward_inner(
        &self,
        x: &Tensor,
        router_in: &Tensor,
        forced: Option<&[usize]>,
    ) -> Result<Tensor> {
        let shape = x.shape().to_vec();
        let d = *shape
            .last()
            .ok_or_else(|| Error::Msg("MoE input needs rank >= 1".into()))?;
        if d == 0 {
            return Err(Error::Msg("MoE input has a zero-size last dim".into()));
        }
        let tokens = x.numel() / d;
        let flat = x.reshape(&[tokens, d])?;
        let n_expert = self.router.shape()[0];
        let d_router = *router_in.shape().last().unwrap_or(&d);
        let logits = router_in
            .reshape(&[tokens, d_router])?
            .matmul(&self.router.t()?)?
            .to_vec::<f32>()?;

        let mut rows = Vec::with_capacity(tokens);
        for tok in 0..tokens {
            let mut l = logits[tok * n_expert..(tok + 1) * n_expert].to_vec();
            if let Some(b) = &self.router_bias {
                l.iter_mut().zip(b).for_each(|(x, b)| *x += b);
            }
            let l = &l[..];
            let softmax = |v: &[f32]| -> Vec<f32> {
                let max = v.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let exp: Vec<f32> = v.iter().map(|x| (x - max).exp()).collect();
                let total: f32 = exp.iter().sum();
                exp.iter().map(|x| x / total).collect()
            };
            let probs: Vec<f32> = match self.gating {
                Gating::Softmax => softmax(l),
                Gating::Sigmoid => l.iter().map(|x| 1.0 / (1.0 + (-x).exp())).collect(),
                Gating::SoftmaxWeight => l.to_vec(),
                Gating::SqrtSoftplus => l
                    .iter()
                    .map(|x| {
                        if *x > 20.0 {
                            x.sqrt()
                        } else {
                            x.exp().ln_1p().sqrt()
                        }
                    })
                    .collect(),
            };
            let mut select: Vec<f32> = match &self.selection_bias {
                Some(b) => {
                    let base = if self.select_on_logits { l } else { &probs[..] };
                    base.iter().zip(b).map(|(p, b)| p + b).collect()
                }
                None => probs.clone(),
            };
            if self.n_groups > 1 {
                // Score each group by the sum of its two best experts, keep the top groups, mask the rest.
                let per = n_expert / self.n_groups;
                let by_desc =
                    |a: &f32, b: &f32| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal);
                let scores: Vec<f32> = (0..self.n_groups)
                    .map(|g| {
                        let mut v = select[g * per..(g + 1) * per].to_vec();
                        v.sort_by(by_desc);
                        v.iter().take(2).sum()
                    })
                    .collect();
                let mut groups: Vec<usize> = (0..self.n_groups).collect();
                groups.sort_by(|&a, &b| {
                    scores[b]
                        .partial_cmp(&scores[a])
                        .unwrap_or(std::cmp::Ordering::Equal)
                        .then(a.cmp(&b))
                });
                groups.truncate(self.n_group_used);
                for (e, s) in select.iter_mut().enumerate() {
                    if !groups.contains(&(e / per)) {
                        *s = f32::NEG_INFINITY;
                    }
                }
            }

            let mut order: Vec<usize> = (0..n_expert).collect();
            order.sort_by(|&a, &b| {
                select[b]
                    .partial_cmp(&select[a])
                    .unwrap_or(std::cmp::Ordering::Equal)
                    .then(a.cmp(&b))
            });
            order.truncate(self.top_k);
            if let Some(f) = forced {
                order = f[tok * self.top_k..(tok + 1) * self.top_k].to_vec();
            }
            let mut w: Vec<f32> = order.iter().map(|&e| probs[e]).collect();
            if self.gating == Gating::SoftmaxWeight {
                w = softmax(&w);
            }
            if self.norm_topk {
                let s = w.iter().sum::<f32>().max(6.103_515_6e-5); // llama.cpp clamps at the f16 minimum
                w.iter_mut().for_each(|v| *v /= s);
            }

            let xt = flat.narrow(0, tok, 1)?;
            let mut acc: Option<Tensor> = None;
            for (&e, &wt) in order.iter().zip(&w) {
                let w = f64::from(wt * self.weights_scale);
                let out = if self.weight_before_ffn {
                    self.expert(&xt.mul_scalar(w)?, e)?
                } else {
                    self.expert(&xt, e)?.mul_scalar(w)?
                };
                acc = Some(match acc {
                    None => out,
                    Some(a) => a.add(&out)?,
                });
            }
            let mut acc = acc.ok_or_else(|| Error::Msg("MoE top_k must be >= 1".into()))?;
            if let Some(ch) = &self.chunk {
                // llama.cpp quirk, reproduced: the chunk experts read the routed output, are picked by
                // `expert / group`, and take their weights from the first router probabilities.
                let n_chunk = ch.gate.shape()[0];
                let ids: Vec<usize> = order
                    .iter()
                    .take(self.top_k.min(n_chunk))
                    .map(|&e| e / ch.group)
                    .collect();
                let mut w2: Vec<f32> = ids.iter().map(|&c| probs[c]).collect();
                if self.norm_topk {
                    let s = w2.iter().sum::<f32>().max(6.103_515_6e-5);
                    w2.iter_mut().for_each(|v| *v /= s);
                }
                let (cff, dd) = (ch.gate.shape()[1], ch.gate.shape()[2]);
                let mut second: Option<Tensor> = None;
                for (&c, &wt) in ids.iter().zip(&w2) {
                    let g = ch.gate.narrow(0, c, 1)?.reshape(&[cff, dd])?;
                    let u = ch.up.narrow(0, c, 1)?.reshape(&[cff, dd])?;
                    let dn = ch.down.narrow(0, c, 1)?.reshape(&[dd, cff])?;
                    let out = swiglu(&acc, &g, &u, &dn)?
                        .mul_scalar(f64::from(wt * self.weights_scale))?;
                    second = Some(match second {
                        None => out,
                        Some(a) => a.add(&out)?,
                    });
                }
                if let Some(second) = second {
                    acc = acc.add(&second.mul_scalar(f64::from(ch.scale))?)?;
                }
            }
            if self.latent.is_none() {
                if let Some(s) = self.shared_out(&xt, d)? {
                    acc = acc.add(&s)?;
                }
            }
            rows.push(acc);
        }
        Tensor::cat(&rows, 0)?.reshape(&shape)
    }
}

/// DeepSeek multi-head latent attention. Q comes from a low-rank factorization (or one projection when
/// `q_a` is `None`), all heads share one compressed KV latent, and attention runs in latent space with the
/// per-head key / value up-projections folded in (`k_b` into the query, `v_b` after the weighted sum).
/// The cache holds the shared `[latent | rotated k_pe]` keys and the latent values.
pub struct MlaAttention {
    /// `[q_lora, d]`; `None` for the "lite" single-projection variant.
    pub q_a: Option<Tensor>,
    pub q_a_norm: Option<Norm>,
    /// `[heads * (nope + rope), q_lora]` (or `[.., d]` when `q_a` is `None`).
    pub q_b: Tensor,
    /// `[kv_lora + rope, d]`.
    pub kv_a: Tensor,
    pub kv_a_norm: Norm,
    /// `[heads, kv_lora, nope]` and `[heads, v_dim, kv_lora]`.
    pub k_b: Tensor,
    pub v_b: Tensor,
    pub heads: usize,
    pub nope: usize,
    pub rope: usize,
    pub kv_lora: usize,
    pub v_dim: usize,
    /// Leave `q_pe` / `k_pe` unrotated (Kimi-Linear's MLA layers carry no positional encoding).
    pub no_rope: bool,
    /// RMS norm on the shared rotary key before RoPE (dots.llm3-note).
    pub k_norm: Option<Norm>,
    /// Per-head attention sink logits `[heads]` (HY-V4).
    pub sinks: Option<Tensor>,
    /// The layer's lightning indexer: only the `top_k` best-scoring earlier tokens are attended (DeepSeek-V3.2 style).
    pub indexer: Option<DsaIndexer>,
    /// The layer has no indexer of its own and reuses the previous indexer layer's token selection.
    pub share_dsa: bool,
}

/// Lightning indexer of a sparse-attention MLA layer: a small multi-head dot-product scorer over its own cached keys.
pub struct DsaIndexer {
    /// `[heads * dim, q_lora]`, `[dim, d]`, `[heads, d]`
    pub q_b: Tensor,
    pub k: Tensor,
    pub proj: Tensor,
    /// LayerNorm (weight and bias) on the keys.
    pub k_norm: Norm,
    pub heads: usize,
    pub dim: usize,
    pub top_k: usize,
    /// Rotary width, placed at the start (`false`) or the end of each indexer head, with adjacent-pair or half-split pairing.
    pub rot: usize,
    pub rope_at_end: bool,
    pub norm_pairs: bool,
    /// Scores use Hadamard-rotated queries / keys (HY-V4 skips the rotation).
    pub hadamard: bool,
    /// K-pool compression (GLM-5-Next): tokens are scored as pools of `kpool.size` consecutive keys.
    pub kpool: Option<KPool>,
}

/// MiniMax-M3's block-sparse attention (MSA): one indexer head per KV group scores every earlier position, a block's
/// score is the best of its positions, and each group attends only its `top_blocks` best blocks plus the `local` latest.
pub struct BlockIndexer {
    /// `[heads * dim, d]` and `[dim, d]`.
    pub q_proj: Tensor,
    pub k_proj: Tensor,
    pub q_norm: Norm,
    pub k_norm: Norm,
    pub heads: usize,
    pub dim: usize,
    pub block: usize,
    pub top_blocks: usize,
    pub local: usize,
    /// QSA (Qwen4-Exp): when above 0, blocks of `pool` tokens are scored by one pooled key (the mean of their raw keys,
    /// normed and rotated to the first member) summed over all indexer heads, `top_blocks` counts tokens, and the
    /// incomplete tail is always attended. The block size / forced local blocks of MSA do not apply.
    pub pool: usize,
}

impl BlockIndexer {
    /// KV groups the selection differs over: one per indexer head for MSA, a single shared one for QSA.
    fn groups(&self) -> usize {
        if self.pool > 0 {
            1
        } else {
            self.heads
        }
    }

    /// New indexer keys `[b, 1, s, dim]` (rotated, rounded like the f16 cache; QSA caches the raw key).
    fn keys(&self, x: &Tensor, rope: &RotaryEmbedding, seq_pos: usize, n_rot: usize) -> Result<Tensor> {
        let (b, s, d) = (x.shape()[0], x.shape()[1], x.shape()[2]);
        let k = x.reshape(&[b * s, d])?.matmul(&self.k_proj.t()?)?.reshape(&[b, s, 1, self.dim])?;
        let k = if self.pool > 0 {
            k.transpose(1, 2)?.contiguous()?
        } else {
            apply_rope(rope, &self.k_norm.forward(&k)?.transpose(1, 2)?, seq_pos, n_rot)?
        };
        let v: Vec<f32> = k
            .to_vec::<f32>()?
            .into_iter()
            .map(|x| crate::dtype::f16::from_f32(x).to_f32())
            .collect();
        Tensor::from_slice_on(&v, k.shape(), &k.device())
    }

    /// `[b, heads, s, total]` flags (`true` = excluded) for the queries of `x` against all `total` cached keys.
    fn select(
        &self,
        x: &Tensor,
        keys: &Tensor,
        rope: &RotaryEmbedding,
        seq_pos: usize,
        n_rot: usize,
    ) -> Result<Vec<bool>> {
        let (b, s, d) = (x.shape()[0], x.shape()[1], x.shape()[2]);
        let total = keys.shape()[2];
        let q = x
            .reshape(&[b * s, d])?
            .matmul(&self.q_proj.t()?)?
            .reshape(&[b, s, self.heads, self.dim])?;
        if self.pool > 0 {
            let q = apply_rope(rope, &self.q_norm.forward(&q)?.transpose(1, 2)?, seq_pos, n_rot)?
                .to_vec::<f32>()?; // [b, heads, s, dim]
            return self.select_pooled(&q, keys, rope, (b, s, total), n_rot);
        }
        // ggml's mul_mat against the f16 key cache converts the query to f16 as well.
        let q: Vec<f32> = apply_rope(rope, &self.q_norm.forward(&q)?.transpose(1, 2)?, seq_pos, n_rot)?
            .to_vec::<f32>()? // [b, heads, s, dim]
            .into_iter()
            .map(|x| crate::dtype::f16::from_f32(x).to_f32())
            .collect();
        let k = keys.to_vec::<f32>()?; // [b, 1, total, dim]
        let nblk = total.div_ceil(self.block);
        let n_top = self.top_blocks.min(nblk);
        let mut excluded = vec![true; b * self.heads * s * total];
        for bi in 0..b {
            for g in 0..self.heads {
                for t in 0..s {
                    let pos = total - s + t;
                    let qh = &q[((bi * self.heads + g) * s + t) * self.dim..][..self.dim];
                    let mut score = vec![f32::NEG_INFINITY; nblk];
                    for j in 0..=pos {
                        let kj = &k[(bi * total + j) * self.dim..][..self.dim];
                        let dot: f32 = qh.iter().zip(kj).map(|(a, b)| a * b).sum();
                        score[j / self.block] = score[j / self.block].max(dot);
                    }
                    let latest = pos / self.block;
                    for l in 0..self.local.min(latest + 1) {
                        score[latest - l] += 1e30;
                    }
                    let mut order: Vec<usize> = (0..nblk).collect();
                    order.sort_by(|&a, &c| score[c].total_cmp(&score[a]).then(a.cmp(&c)));
                    let row = ((bi * self.heads + g) * s + t) * total;
                    for &blk in &order[..n_top] {
                        for j in blk * self.block..((blk + 1) * self.block).min(total) {
                            excluded[row + j] = false;
                        }
                    }
                }
            }
        }
        Ok(excluded)
    }

    /// QSA: the best `top_blocks / pool` complete blocks plus the incomplete tail, `[b, 1, s, total]` flags.
    fn select_pooled(
        &self,
        q: &[f32],
        keys: &Tensor,
        rope: &RotaryEmbedding,
        (b, s, total): (usize, usize, usize),
        n_rot: usize,
    ) -> Result<Vec<bool>> {
        let (dim, size) = (self.dim, self.pool);
        let k = keys.to_vec::<f32>()?; // [b, 1, total, dim] raw keys
        let n_pool = total / size;
        let mut excluded = vec![true; b * s * total];
        for bi in 0..b {
            let mut pooled: Vec<Vec<f32>> = Vec::with_capacity(n_pool);
            for p in 0..n_pool {
                let mean: Vec<f32> = (0..dim)
                    .map(|c| {
                        (0..size)
                            .map(|m| k[(bi * total + p * size + m) * dim + c])
                            .sum::<f32>()
                            / size as f32
                    })
                    .collect();
                let t = Tensor::from_slice_on(&mean, &[1, 1, 1, dim], &keys.device())?;
                let t = apply_rope(rope, &self.k_norm.forward(&t)?, p * size, n_rot)?;
                pooled.push(
                    t.to_vec::<f32>()?
                        .into_iter()
                        .map(|x| crate::dtype::f16::from_f32(x).to_f32())
                        .collect(),
                );
            }
            for t in 0..s {
                let pos = total - s + t;
                let n_vis = (pos + 1) / size;
                let mut score: Vec<(f32, usize)> = (0..n_vis)
                    .map(|p| {
                        let sum: f32 = (0..self.heads)
                            .map(|hh| {
                                let qh = &q[((bi * self.heads + hh) * s + t) * dim..][..dim];
                                qh.iter().zip(&pooled[p]).map(|(a, b)| a * b).sum::<f32>().max(0.0)
                            })
                            .sum();
                        (sum / (dim as f32).sqrt(), p)
                    })
                    .collect();
                score.sort_by(|a, b| b.0.total_cmp(&a.0).then(a.1.cmp(&b.1)));
                score.truncate(self.top_blocks / size);
                let row = (bi * s + t) * total;
                for (_, p) in score {
                    for m in 0..size {
                        excluded[row + p * size + m] = false;
                    }
                }
                for j in n_vis * size..=pos {
                    excluded[row + j] = false;
                }
            }
        }
        Ok(excluded)
    }
}

/// The pooling gate of a k-pool indexer: each pool's key is the per-channel softmax-weighted sum of its members' keys.
pub struct KPool {
    /// `[dim, d]`: per-token gate scores, cached next to the key.
    pub gate: Tensor,
    /// `[size, dim]` positional bias added to the gate scores.
    pub ape: Tensor,
    pub size: usize,
    /// The incomplete tail pool's tokens are always attended.
    pub select_tail: bool,
}

impl DsaIndexer {
    /// Columns one token's indexer entry takes in the cache (the key, plus the pool gate scores for a k-pool indexer).
    fn width(&self) -> usize {
        if self.kpool.is_some() {
            2 * self.dim
        } else {
            self.dim
        }
    }

    /// Pool-level selection: the best `top_k / size` complete pools plus (optionally) the incomplete tail, as in
    /// llama.cpp's `build_kpool_select`. `k` holds `[key | gate]` per cached token.
    #[allow(clippy::too_many_arguments)]
    fn select_pooled(
        &self,
        kp: &KPool,
        q: &[f32],
        w: &[f32],
        k: &[f32],
        (b, s, total): (usize, usize, usize),
    ) -> Vec<bool> {
        let (dim, size, wd) = (self.dim, kp.size, 2 * self.dim);
        let ape = kp.ape.to_vec::<f32>().unwrap_or_default(); // [size, dim]
        let n_pool = total / size;
        let mut excluded = vec![true; b * s * total];
        for bi in 0..b {
            // Pooled keys, rounded like the f16 cache.
            let pooled: Vec<Vec<f32>> = (0..n_pool)
                .map(|p| {
                    (0..dim)
                        .map(|c| {
                            let logit = |m: usize| {
                                k[(bi * total + p * size + m) * wd + dim + c] + ape[m * dim + c]
                            };
                            let mx = (0..size).map(logit).fold(f32::NEG_INFINITY, f32::max);
                            let den: f32 = (0..size).map(|m| (logit(m) - mx).exp()).sum();
                            let v: f32 = (0..size)
                                .map(|m| {
                                    (logit(m) - mx).exp() / den
                                        * k[(bi * total + p * size + m) * wd + c]
                                })
                                .sum();
                            crate::dtype::f16::from_f32(v).to_f32()
                        })
                        .collect()
                })
                .collect();
            for t in 0..s {
                let pos = total - s + t;
                let n_vis = (pos + 1) / size;
                let mut score: Vec<(f32, usize)> = (0..n_vis)
                    .map(|p| {
                        let sum: f32 = (0..self.heads)
                            .map(|hh| {
                                let qh = &q[((bi * self.heads + hh) * s + t) * dim
                                    ..((bi * self.heads + hh) * s + t + 1) * dim];
                                w[(bi * s + t) * self.heads + hh]
                                    * qh.iter().zip(&pooled[p]).map(|(a, b)| a * b).sum::<f32>().max(0.0)
                            })
                            .sum();
                        (sum, p)
                    })
                    .collect();
                score.sort_by(|a, b| b.0.total_cmp(&a.0).then(a.1.cmp(&b.1)));
                score.truncate(self.top_k / size);
                let row = (bi * s + t) * total;
                for (_, p) in score {
                    for m in 0..size {
                        excluded[row + p * size + m] = false;
                    }
                }
                if kp.select_tail {
                    for j in n_vis * size..=pos {
                        excluded[row + j] = false;
                    }
                }
            }
        }
        excluded
    }

    /// Rotates the rotary slice of `t` (`[b, h, s, dim]`) for positions `seq_pos..`.
    fn rotate(&self, rope: &RotaryEmbedding, t: &Tensor, seq_pos: usize) -> Result<Tensor> {
        if self.rot == 0 {
            return Ok(t.clone());
        }
        let (b, h, s, dim) = (t.shape()[0], t.shape()[1], t.shape()[2], t.shape()[3]);
        let start = if self.rope_at_end { dim - self.rot } else { 0 };
        let mut part = t.narrow(3, start, self.rot)?.contiguous()?;
        if self.norm_pairs {
            // Adjacent pairs (x0, x1), (x2, x3), ... are the half-split pairs of the sequence (evens | odds).
            let pairs = part.reshape(&[b, h, s, self.rot / 2, 2])?;
            part = Tensor::cat(
                &[
                    pairs
                        .narrow(4, 0, 1)?
                        .contiguous()?
                        .reshape(&[b, h, s, self.rot / 2])?,
                    pairs
                        .narrow(4, 1, 1)?
                        .contiguous()?
                        .reshape(&[b, h, s, self.rot / 2])?,
                ],
                3,
            )?;
        }
        let mut part = rope.apply(&part, seq_pos)?;
        if self.norm_pairs {
            part = part
                .reshape(&[b, h, s, 2, self.rot / 2])?
                .transpose(3, 4)?
                .contiguous()?
                .reshape(&[b, h, s, self.rot])?;
        }
        let mut pieces = Vec::new();
        if start > 0 {
            pieces.push(t.narrow(3, 0, start)?.contiguous()?);
        }
        pieces.push(part);
        if start + self.rot < dim {
            pieces.push(
                t.narrow(3, start + self.rot, dim - start - self.rot)?
                    .contiguous()?,
            );
        }
        Tensor::cat(&pieces, 3)
    }

    /// llama.cpp scores with Hadamard-rotated queries and keys (most architectures) and keeps the keys in an f16 cache:
    /// apply the same orthonormal transform and rounding, so near-tied top-k picks come out identically.
    fn quantize(&self, t: &Tensor, round: bool) -> Result<Tensor> {
        if self.dim == 0 {
            return Err(Error::Msg("DsaIndexer has width 0".into()));
        }
        let hadamard = self.hadamard && self.dim.is_power_of_two();
        if !hadamard && !round {
            return Ok(t.clone());
        }
        let mut v = t.to_vec::<f32>()?;
        for chunk in v.chunks_exact_mut(self.dim) {
            if hadamard {
                let mut len = 1;
                while len < self.dim {
                    for start in (0..self.dim).step_by(2 * len) {
                        for i in start..start + len {
                            let (a, b) = (chunk[i], chunk[i + len]);
                            chunk[i] = a + b;
                            chunk[i + len] = a - b;
                        }
                    }
                    len *= 2;
                }
                let norm = 1.0 / (self.dim as f32).sqrt();
                chunk.iter_mut().for_each(|x| *x *= norm);
            }
            if round {
                chunk
                    .iter_mut()
                    .for_each(|x| *x = crate::dtype::f16::from_f32(*x).to_f32());
            }
        }
        Tensor::from_slice_on(&v, t.shape(), &t.device())
    }
    /// New indexer keys `[b, 1, s, dim]` for the tokens in `x`.
    fn keys(&self, x: &Tensor, rope: &RotaryEmbedding, seq_pos: usize) -> Result<Tensor> {
        let (b, s, d) = (x.shape()[0], x.shape()[1], x.shape()[2]);
        let k = self
            .k_norm
            .forward(&x.reshape(&[b * s, d])?.matmul(&self.k.t()?)?)?
            .reshape(&[b, 1, s, self.dim])?;
        let k = self.quantize(&self.rotate(rope, &k, seq_pos)?, true)?;
        match &self.kpool {
            Some(kp) => {
                let g = x
                    .reshape(&[b * s, d])?
                    .matmul(&kp.gate.t()?)?
                    .reshape(&[b, 1, s, self.dim])?;
                Tensor::cat(&[k, self.quantize(&g, true)?], 3)
            }
            None => Ok(k),
        }
    }

    /// The tokens each query may attend, as a `[b, s, total]` flag per key (`true` = excluded).
    #[allow(clippy::too_many_arguments)]
    fn select(
        &self,
        x: &Tensor,
        qr: &Tensor,
        keys: &Tensor,
        rope: &RotaryEmbedding,
        seq_pos: usize,
        total: usize,
    ) -> Result<Vec<bool>> {
        let (b, s, d) = (x.shape()[0], x.shape()[1], x.shape()[2]);
        let q = qr
            .reshape(&[b * s, qr.shape()[qr.shape().len() - 1]])?
            .matmul(&self.q_b.t()?)?
            .reshape(&[b, s, self.heads, self.dim])?
            .transpose(1, 2)?
            .contiguous()?;
        let q = self
            .quantize(&self.rotate(rope, &q, seq_pos)?, false)?
            .to_vec::<f32>()?; // [b, heads, s, dim]
        let w = x
            .reshape(&[b * s, d])?
            .matmul(&self.proj.t()?)?
            .mul_scalar(1.0 / ((self.dim * self.heads) as f64).sqrt())?
            .to_vec::<f32>()?;
        let k = keys.to_vec::<f32>()?; // [b, 1, total, width]
        if let Some(kp) = &self.kpool {
            return Ok(self.select_pooled(kp, &q, &w, &k, (b, s, total)));
        }
        let mut excluded = vec![true; b * s * total];
        for bi in 0..b {
            for t in 0..s {
                let pos = total - s + t;
                let mut score: Vec<(f32, usize)> = (0..=pos)
                    .map(|j| {
                        let kj = &k[(bi * total + j) * self.dim..(bi * total + j + 1) * self.dim];
                        let sum: f32 = (0..self.heads)
                            .map(|hh| {
                                let qh = &q[((bi * self.heads + hh) * s + t) * self.dim
                                    ..((bi * self.heads + hh) * s + t + 1) * self.dim];
                                w[(bi * s + t) * self.heads + hh]
                                    * qh.iter().zip(kj).map(|(a, b)| a * b).sum::<f32>().max(0.0)
                            })
                            .sum();
                        (sum, j)
                    })
                    .collect();
                if score.len() > self.top_k {
                    score.sort_by(|a, b| b.0.total_cmp(&a.0).then(a.1.cmp(&b.1)));
                    score.truncate(self.top_k);
                }
                for (_, j) in score {
                    excluded[(bi * s + t) * total + j] = false;
                }
            }
        }
        Ok(excluded)
    }
}

impl MlaAttention {
    /// `x`: normalized hidden states `[b, s, d]`. Returns the per-head values `[b, s, heads * v_dim]`
    /// (before the output projection).
    #[allow(clippy::too_many_arguments)]
    pub fn forward(
        &self,
        x: &Tensor,
        rope: &RotaryEmbedding,
        seq_pos: usize,
        cache: &mut Option<(Tensor, Tensor)>,
        scale: f32,
        window: Option<usize>,
        dsa: Option<&mut Option<Vec<bool>>>,
    ) -> Result<Tensor> {
        let (b, s) = (x.shape()[0], x.shape()[1]);
        let (h, nope, rope_d, kv, vd) =
            (self.heads, self.nope, self.rope, self.kv_lora, self.v_dim);
        let hdk = nope + rope_d;
        let mm = |a: &Tensor, w: &Tensor| -> Result<Tensor> {
            let (m, k) = (w.shape()[0], w.shape()[1]);
            a.reshape(&[b * s, k])?.matmul(&w.t()?)?.reshape(&[b, s, m])
        };

        let qr = match (&self.q_a, &self.q_a_norm) {
            (Some(qa), Some(norm)) => Some(norm.forward(&mm(x, qa)?)?),
            _ => None,
        };
        let q = match &qr {
            Some(qr) => mm(qr, &self.q_b)?,
            None => mm(x, &self.q_b)?,
        };
        let q = q.reshape(&[b, s, h, hdk])?.transpose(1, 2)?; // [b, h, s, hdk]
        let q_nope = q.narrow(3, 0, nope)?.contiguous()?;
        // `None` when the heads carry no rotary part at all (GLM-5-Next's nope-only MLA).
        let q_pe = if rope_d == 0 {
            None
        } else {
            let q_pe = q.narrow(3, nope, rope_d)?.contiguous()?;
            Some(if self.no_rope {
                q_pe
            } else {
                rope.apply(&q_pe, seq_pos)?
            })
        };

        let kvp = mm(x, &self.kv_a)?; // [b, s, kv + rope]
        let kv_c = self
            .kv_a_norm
            .forward(&kvp.narrow(2, 0, kv)?.contiguous()?)?;
        let k_pe = if rope_d == 0 {
            None
        } else {
            let k_pe = kvp
                .narrow(2, kv, rope_d)?
                .contiguous()?
                .reshape(&[b, 1, s, rope_d])?;
            let k_pe = match &self.k_norm {
                Some(n) => n.forward(&k_pe)?,
                None => k_pe,
            };
            Some(if self.no_rope {
                k_pe
            } else {
                rope.apply(&k_pe, seq_pos)?
            })
        };

        // Fold the key up-projection into the query: [b,h,s,nope] x [h,nope,kv] -> [b,h,s,kv].
        let q_absorbed = if nope == 0 {
            // MiniCPM3-style heads that are all rotary: nothing to absorb (an empty contraction is not safe to matmul).
            let zeros_n = crate::limits::check_elements("MLA absorbed-q scratch", &[b, h, s, kv])?;
            Tensor::from_slice_on(
                &crate::layout::try_filled(zeros_n, 0.0f32)?,
                &[b, h, s, kv],
                &x.device(),
            )?
        } else {
            let kb = self
                .k_b
                .transpose(1, 2)?
                .contiguous()?
                .reshape(&[1, h, nope, kv])?
                .expand(&[b, h, nope, kv])?
                .contiguous()?;
            q_nope.matmul(&kb)?
        };
        let q_full = match q_pe {
            Some(q_pe) => Tensor::cat(&[q_absorbed, q_pe], 3)?, // [b, h, s, kv + rope]
            None => q_absorbed,
        };

        let kv_c = kv_c.reshape(&[b, 1, s, kv])?;
        let k_new = match k_pe {
            Some(k_pe) => Tensor::cat(&[kv_c.clone(), k_pe], 3)?,
            None => kv_c.clone(),
        };
        // A sparse-attention layer keeps its indexer keys next to the latent values (extra columns of `v_all`).
        let v_new = match &self.indexer {
            Some(ix) => Tensor::cat(&[kv_c.clone(), ix.keys(x, rope, seq_pos)?], 3)?,
            None => kv_c,
        };
        let (mut k_all, mut v_all) = (k_new, v_new);
        if let Some((pk, pv)) = cache.take() {
            k_all = Tensor::cat(&[pk, k_all], 2)?;
            v_all = Tensor::cat(&[pv, v_all], 2)?;
        }
        *cache = Some((k_all.clone(), v_all.clone()));
        let total = k_all.shape()[2];

        // Which keys each query may see beyond causality: its indexer's top-k, or the previous indexer layer's choice.
        let mut dsa = dsa;
        let selection: Option<Vec<bool>> = match (&self.indexer, &qr) {
            (Some(ix), Some(qr)) => {
                let keys = v_all.narrow(3, kv, ix.width())?.contiguous()?;
                let sel = ix.select(x, qr, &keys, rope, seq_pos, total)?;
                if let Some(slot) = dsa.as_deref_mut() {
                    *slot = Some(sel.clone());
                }
                Some(sel)
            }
            _ if self.share_dsa => dsa.as_deref().and_then(|slot| slot.clone()),
            _ => None,
        };

        let k_e = k_all.expand(&[b, h, total, kv + rope_d])?.contiguous()?;
        let v_e = v_all
            .narrow(3, 0, kv)?
            .contiguous()?
            .expand(&[b, h, total, kv])?
            .contiguous()?;
        let mut scores = q_full
            .matmul(&k_e.transpose(2, 3)?)?
            .mul_scalar(f64::from(scale))?;
        if let Some(sel) = &selection {
            let mask = Tensor::from_slice_on(sel, &[b, 1, s, total], &scores.device())?
                .expand(&[b, h, s, total])?;
            scores = scores.masked_fill(&mask, -1e9)?;
        }
        if window.is_some_and(|w| total > w) {
            // Sliding-window layers: a token sees itself and the `w - 1` before it.
            let (w, past) = (window.unwrap_or(usize::MAX), total - s);
            let mask: Vec<bool> = (0..s)
                .flat_map(|qi| (0..total).map(move |ki| ki > past + qi || past + qi >= ki + w))
                .collect();
            let mask = Tensor::from_slice_on(&mask, &[s, total], &scores.device())?;
            scores = scores.masked_fill(&mask, -1e9)?;
        } else if s > 1 {
            let mask =
                super::transformer::causal_mask_with_offset(s, total, total - s, &scores.device())?;
            scores = scores.masked_fill(&mask, -1e9)?;
        }
        let weights = match &self.sinks {
            Some(sinks) => {
                // The sink only enlarges the softmax denominator.
                let col = sinks
                    .reshape(&[1, h, 1, 1])?
                    .expand(&[b, h, s, 1])?
                    .contiguous()?;
                Tensor::cat(&[scores, col], 3)?
                    .softmax(3)?
                    .narrow(3, 0, total)?
                    .contiguous()?
            }
            None => scores.softmax(3)?,
        };
        let ctx = weights.matmul(&v_e)?; // [b, h, s, kv]

        let vb = self
            .v_b
            .transpose(1, 2)?
            .contiguous()?
            .reshape(&[1, h, kv, vd])?
            .expand(&[b, h, kv, vd])?
            .contiguous()?;
        ctx.matmul(&vb)?
            .transpose(1, 2)?
            .contiguous()?
            .reshape(&[b, s, h * vd])
    }
}

/// Per-call extras a layer may need on top of its hidden state.
#[derive(Default)]
pub struct LayerCtx<'a> {
    /// This token's banked checkpoints (attention-residual models).
    pub res: Option<&'a mut Vec<Tensor>>,
    /// The call's token ids, flattened batch-major (hash-routed experts).
    pub ids: Option<&'a [i64]>,
    /// The token selection of the latest sparse-attention indexer layer, shared with the layers that have no indexer.
    pub dsa: Option<&'a mut Option<Vec<bool>>>,
    /// RWKV-7's value vectors of layer 0, shared with the later layers of the same call.
    pub v_first: Option<&'a mut Option<Tensor>>,
    /// K/V of an earlier layer that this layer attends to instead of its own (Gemma 4).
    pub shared_kv: Option<(Tensor, Tensor)>,
}

/// A `[rows, cols]` table kept in its on-disk (usually quantized) form: a lookup dequantizes only the requested rows.
/// Gemma 3n / 4's per-layer input embedding has `vocab x (layers * width)` entries, which as `f32` would take several
/// gigabytes for a real checkpoint.
pub struct LazyRows {
    ty: crate::serialize::gguf::GGMLType,
    raw: Vec<u8>,
    rows: usize,
    cols: usize,
    row_bytes: usize,
}

impl LazyRows {
    pub fn from_file(file: &GGUFFile, name: &str, rows: usize, cols: usize) -> Result<Self> {
        let info = file
            .tensors
            .get(name)
            .ok_or_else(|| Error::Msg(format!("missing {name}")))?;
        if info.shape != [rows, cols] {
            return Err(Error::Msg(format!(
                "GGUF tensor '{name}' has shape {:?}, expected {:?} from model config",
                info.shape,
                [rows, cols]
            )));
        }
        let row_bytes = info.ggml_type.tensor_bytes(cols)?;
        let raw = file.get_tensor_data(name)?.to_vec();
        if raw.len() < rows.saturating_mul(row_bytes) {
            return Err(Error::Msg(format!("{name} holds fewer than {rows} rows")));
        }
        Ok(Self {
            ty: info.ggml_type,
            raw,
            rows,
            cols,
            row_bytes,
        })
    }

    /// The rows named by `ids` (any shape): `[.., cols]`.
    pub fn forward(&self, ids: &Tensor) -> Result<Tensor> {
        let mut shape = ids.shape().to_vec();
        let flat = ids.to_vec::<i64>()?;
        let mut picked = Vec::with_capacity(flat.len() * self.row_bytes);
        for &id in &flat {
            let r = usize::try_from(id)
                .ok()
                .filter(|&r| r < self.rows)
                .ok_or_else(|| {
                    Error::Msg(format!("token id {id} is outside the {}-row table", self.rows))
                })?;
            picked.extend_from_slice(&self.raw[r * self.row_bytes..(r + 1) * self.row_bytes]);
        }
        let out = dequantize_rows(self.ty, &picked, flat.len(), self.cols)?;
        shape.push(self.cols);
        Tensor::from_slice(&out, &shape)
    }
}

/// Dequantizes `rows` whole rows of `cols` values stored as `ty` in `raw`. The GGUF reader only dequantizes whole
/// tensors of a parsed file, so the rows are wrapped in a one-tensor GGUF container first.
fn dequantize_rows(
    ty: crate::serialize::gguf::GGMLType,
    raw: &[u8],
    rows: usize,
    cols: usize,
) -> Result<Vec<f32>> {
    if rows == 0 {
        return Ok(Vec::new());
    }
    let mut buf = Vec::with_capacity(raw.len() + 128);
    buf.extend_from_slice(&0x4655_4747u32.to_le_bytes()); // "GGUF"
    buf.extend_from_slice(&3u32.to_le_bytes());
    buf.extend_from_slice(&1u64.to_le_bytes()); // tensors
    buf.extend_from_slice(&0u64.to_le_bytes()); // metadata entries
    buf.extend_from_slice(&1u64.to_le_bytes()); // name length
    buf.push(b't');
    buf.extend_from_slice(&2u32.to_le_bytes()); // dimensions, innermost first
    buf.extend_from_slice(&(cols as u64).to_le_bytes());
    buf.extend_from_slice(&(rows as u64).to_le_bytes());
    buf.extend_from_slice(&(ty as u32).to_le_bytes());
    buf.extend_from_slice(&0u64.to_le_bytes()); // data offset
    buf.resize(buf.len().next_multiple_of(32), 0); // the default alignment
    buf.extend_from_slice(raw);
    GGUFFile::from_bytes(&buf)?
        .dequantize_tensor("t")?
        .to_vec::<f32>()
}

/// Model-level tensors of Gemma 3n: the embedding is lifted into `n_altup` AltUp streams, every layer also receives a
/// per-layer input derived from the token and the embedding, and the streams are merged again before the output norm.
pub struct G3nModel {
    /// `[n_altup - 1, d, d]`: project the embedding into the extra streams / bring them back.
    pub altup_proj: Tensor,
    pub altup_unembd_proj: Tensor,
    /// Per-layer input embedding `[vocab, layers * width]` and its projection `[layers * width, d]`.
    pub per_layer_emb: LazyRows,
    pub per_layer_model_proj: Tensor,
    pub per_layer_proj_norm: Norm,
    /// Width of one layer's input.
    pub width: usize,
    pub n_altup: usize,
}

/// One Gemma 3n layer's AltUp / LAuReL / per-layer-input tensors.
pub struct G3nLayer {
    pub inp_gate: Tensor,
    pub proj: Tensor,
    pub post_norm: Norm,
    pub correct_coef: Tensor,
    pub correct_scale: Tensor,
    pub predict_coef: Tensor,
    pub router: Tensor,
    pub router_norm: Norm,
    pub laurel_l: Tensor,
    pub laurel_r: Tensor,
    pub laurel_post_norm: Norm,
    /// The first ten layers keep only gate activations above `mean + 1.645 std`.
    pub sparse: bool,
    /// Layer whose K/V this one reuses (every layer from the twentieth on).
    pub kv_source: Option<usize>,
}

/// `x @ w^T` over the last axis of any-rank `x`.
pub(crate) fn lin_last(x: &Tensor, w: &Tensor) -> Result<Tensor> {
    let shape = x.shape().to_vec();
    let k = shape[shape.len() - 1];
    let rows: usize = shape[..shape.len() - 1].iter().product();
    let mut out = shape;
    let last = out.len() - 1;
    out[last] = w.shape()[0];
    x.contiguous()?
        .reshape(&[rows, k])?
        .matmul(&w.t()?)?
        .reshape(&out)
}

/// Euclidean length over the last axis (kept as a size-1 axis).
fn g3n_magnitude(x: &Tensor) -> Result<Tensor> {
    x.square()?.sum_axis(-1, true)?.sqrt()
}

/// Rescales `x` to the length of `target` (no epsilon, as in llama.cpp).
fn g3n_match_magnitude(x: &Tensor, target: &Tensor) -> Result<Tensor> {
    x.mul(target)?.div(&g3n_magnitude(x)?)
}

/// Activation sparsity: `relu(x - (mean + 1.645 std))` over the last axis (unbiased std).
fn g3n_gaussian_topk(x: &Tensor) -> Result<Tensor> {
    let n = x.shape()[x.shape().len() - 1] as f64;
    let mean = x.mean_axis(-1, true)?;
    let std = x
        .sub(&mean)?
        .square()?
        .sum_axis(-1, true)?
        .mul_scalar(1.0 / (n - 1.0))?
        .sqrt()?;
    x.sub(&mean.add(&std.mul_scalar(1.644_853_353_500_366_2)?)?)?
        .relu()
}

/// Model-level tensors of Gemma 4: per-layer input embeddings (the same construction as Gemma 3n's, without AltUp),
/// absent from the larger models.
pub struct G4Model {
    pub per_layer: Option<G4PerLayer>,
}

pub struct G4PerLayer {
    pub emb: LazyRows,
    pub model_proj: Tensor,
    pub proj_norm: Norm,
    /// Width of one layer's input.
    pub width: usize,
}

/// One Gemma 4 layer's extras: the per-layer input gate with its projection and norm, the layer's output scalar and
/// the layer whose K/V this one reuses.
pub struct G4Layer {
    /// `(inp_gate [width, d], proj [d, width], post_norm)`.
    pub per_layer: Option<(Tensor, Tensor, Norm)>,
    pub out_scale: Option<f32>,
    pub kv_source: Option<usize>,
}

/// Gemma 4's MoE layer extras: `mlp = post_norm_1(dense(ffn_norm(h)))`, `moe = post_norm_2(experts(pre_norm_2(h)))`
/// with the router reading `rms_norm(h) / sqrt(d) * router_scale`; the sum then goes through the usual post-FFN norm.
pub struct G4Moe {
    pub pre_norm_2: Norm,
    pub post_norm_1: Norm,
    pub post_norm_2: Norm,
    pub router_scale: Tensor,
}

impl G3nLayer {
    /// AltUp router: `tanh(router(rms_norm(x) / d))`, `[.., n_altup]`.
    fn modalities(&self, x: &Tensor) -> Result<Tensor> {
        let d = x.shape()[x.shape().len() - 1] as f64;
        let r = self.router_norm.forward(x)?.mul_scalar(1.0 / d)?;
        lin_last(&r, &self.router)?.tanh()
    }

    fn laurel(&self, cur: &Tensor) -> Result<Tensor> {
        let low = lin_last(&lin_last(cur, &self.laurel_l)?, &self.laurel_r)?;
        self.laurel_post_norm.forward(&low)?.add(cur)
    }
}

/// A decoder block in a GGUF-loaded causal language model.
pub struct GGUFDecoderBlock {
    pub attn_norm: Norm,
    pub ffn_norm: Norm,
    pub q_proj: Linear,
    pub k_proj: Linear,
    pub v_proj: Linear,
    pub o_proj: Linear,
    pub gate_proj: Linear,
    pub up_proj: Linear,
    pub down_proj: Linear,
    pub num_heads: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    /// Attend only to the latest `window` tokens (itself included).
    pub sliding_window: Option<usize>,
    /// When set, replaces `gate_proj` / `up_proj` / `down_proj` (which are then 1x1 placeholders).
    pub moe: Option<MoeFfn>,
    /// Per-projection scalar weight scales (`blk.N.*.scale`): `y = (W x) * s`, then `+ bias` for the
    /// attention projections; for the dense FFN llama.cpp adds the bias first and scales after.
    pub q_scale: Option<f32>,
    pub k_scale: Option<f32>,
    pub v_scale: Option<f32>,
    pub o_scale: Option<f32>,
    /// Output gate over the attention result: width `heads * head_dim` (per element) or `heads` (per head).
    pub attn_gate: Option<Linear>,
    /// Attention-residual scores `[attn, ffn]` (Kimi-K3): this layer mixes the banked checkpoints into its inputs.
    pub attn_res: Option<[Tensor; 2]>,
    /// This layer banks its input as a checkpoint and restarts the residual stream from its attention output.
    pub res_bank: bool,
    /// Hyper-connection gates of this layer (HY-V4).
    pub hc: Option<HcLayer>,
    /// DeepSeek-V4 attention (replaces the standard / MLA attention of this layer).
    pub dsv4: Option<Dsv4Attn>,
    /// Hash-routed MoE layer: `[vocab][used]` expert ids by token id, and `used` (DeepSeek-V4's first layers).
    pub moe_hash: Option<(Vec<i32>, usize)>,
    pub attn_gate_softplus: bool,
    /// Gate is `log2(1 + 2^x)` rather than a sigmoid or natural softplus.
    pub attn_gate_log2: bool,
    pub value_moe: Option<ValueMoe>,
    /// Every token attends to every other one (diffusion LMs: Dream, LLaDA, RND1). Meant for a single full-sequence pass.
    pub bidirectional: bool,
    /// Weightless RMS norm over each head of the attention output, before the gate (Muse-Spark).
    pub attn_o_norm: Option<Norm>,
    /// Norm-preserving residuals `[attn_alpha, attn_beta, ffn_alpha, ffn_beta]`: `stream * alpha + branch * beta`.
    pub resid_ab: Option<[Tensor; 4]>,
    /// The FFN output is multiplied by this before `post_ffn_norm`, not after (Muse-Spark).
    pub ffn_post_scale: Option<Tensor>,
    pub attn_gate_scale: Option<f32>,
    pub moe_scale: Option<f32>,
    pub dense_with_moe: bool,
    pub moe_norm: Option<Norm>,
    /// ALiBi slope per head.
    pub alibi_slopes: Option<Vec<f32>>,
    /// Chunked attention: a token only sees keys of its own `chunk_size` chunk.
    pub chunk_size: Option<usize>,
    pub moe_router_raw: bool,
    /// Mamba / Mamba-2 mixer; when set the block has no attention and no FFN.
    pub ssm: Option<SsmBlock>,
    pub mixer: Mixer,
    pub nope_temp: Option<(f32, usize)>,
    pub nope_temp_pow: bool,
    /// Beam weighted residual: `Some(i)` mixes with `i = 2 * (layer + 1)` after attention and `i + 1` after the FFN.
    pub residual_mix: Option<usize>,
    /// Per-head gain on the normalized queries, `[1, heads, 1, 1]`.
    pub q_gain: Option<Tensor>,
    /// Weight of the embedding added to this layer's output.
    pub skip_scale: Option<f32>,
    /// Reciprocals of `ffn.act.scales`, `[ff]`.
    pub ffn_act_recip: Option<Tensor>,
    pub gate_scale: Option<f32>,
    pub up_scale: Option<f32>,
    pub down_scale: Option<f32>,
    pub clamp_kqv: Option<f32>,
    /// Q/K norms (`qk_norm_full`: over the whole projection before heads are split, else per head).
    pub q_norm: Option<Norm>,
    pub k_norm: Option<Norm>,
    pub qk_norm_full: bool,
    pub post_attn_norm: Option<Norm>,
    /// BitNet's extra RMS norms: over the attention output before `wo`, and over `act(gate) * up` before `down`.
    pub attn_sub_norm: Option<Norm>,
    pub ffn_sub_norm: Option<Norm>,
    /// Gemma 3n's AltUp / LAuReL / per-layer-input tensors; the block then runs [`GGUFDecoderBlock::forward_g3n`].
    pub g3n: Option<G3nLayer>,
    /// An RWKV-6 layer (time mix + channel mix with their own norms), see [`super::gguf_rwkv`].
    pub rwkv: Option<super::gguf_rwkv::Rwkv6Layer>,
    /// An RWKV-7 layer; the model loop threads layer 0's value vectors through [`LayerCtx::v_first`].
    pub rwkv7: Option<super::gguf_rwkv::Rwkv7Layer>,
    /// A QRWKV6 layer (`rwkv6qwen2`).
    pub qrwkv: Option<super::gguf_rwkv::Qrwkv6Layer>,
    /// Weightless RMS norm on the values, per head (Gemma 3n).
    pub v_norm: bool,
    /// The values are the (un-normed, un-rotated) key projection: the layer has no `attn_v` (Gemma 4).
    pub v_from_k: bool,
    /// Gemma 4's per-layer input gate / output scalar / shared-K/V source.
    pub g4: Option<G4Layer>,
    /// Gemma 4's MoE layers: the dense MLP and the routed experts read the post-attention state through their own norms.
    pub g4_moe: Option<G4Moe>,
    pub post_ffn_norm: Option<Norm>,
    pub activation: Activation,
    /// Scale of `q . k`.
    pub attn_scale: f32,
    pub attn_softcap: Option<f32>,
    /// Rotary dimensions per head (`< head_dim` = partial rotary).
    pub n_rot: usize,
    /// Use the model's local (sliding-window) rotary table instead of the global one.
    pub local_rope: bool,
    /// Apply `attn_norm` / `ffn_norm` before the attention / FFN (false for OLMo 2).
    pub pre_norm: bool,
    /// False on SmolLM3's periodic NoPE layers.
    pub use_rope: bool,
    /// Multi-head latent attention (replaces the Q/K/V projections and RoPE path).
    pub mla: Option<MlaAttention>,
    /// Gated FFN (`down(act(gate) * up)`); false = `down(act(up))`.
    pub gated: bool,
    /// FFN reads the attention-normed input and adds into the same residual.
    pub parallel: bool,
    /// Per-head attention-sink logits `[heads]`: an extra softmax term that absorbs probability mass.
    pub sinks: Option<Tensor>,
    /// Block-sparse attention indexer (MiniMax-M3).
    pub block_indexer: Option<BlockIndexer>,
    /// Per-head Q/K norms run after RoPE.
    pub qk_norm_after_rope: bool,
    /// With `parallel`: the FFN norms the layer input itself (`ffn_norm`) instead of reusing the attention norm.
    pub parallel_own_ffn_norm: bool,
    /// Optional second norm feeding the attention while the FFN keeps the first (Falcon).
    pub attn_norm_2: Option<Norm>,
    /// Multiplies the attention and FFN outputs before they are added to the residual (Granite).
    pub residual_scale: Option<f32>,
    /// MiniMax-01: the residual of each half-layer is `norm_residual * (normalised layer input)` instead of the raw stream.
    pub norm_residual: Option<f32>,
}

/// Rotates the first `n_rot` dimensions of each head of `x` (`[b, heads, s, hd]`), leaving the rest.
fn apply_rope(rope: &RotaryEmbedding, x: &Tensor, pos: usize, n_rot: usize) -> Result<Tensor> {
    let hd = x.shape()[3];
    if n_rot >= hd {
        return rope.apply(x, pos);
    }
    let rotated = rope.apply(&x.narrow(3, 0, n_rot)?, pos)?;
    Tensor::cat(&[rotated, x.narrow(3, n_rot, hd - n_rot)?], 3)
}

/// `lin(x)` with llama.cpp's operation order for projections that carry a weight scale:
/// `(W x) * scale + bias` (plain `Linear::forward` is `W x + bias`).
fn proj(lin: &Linear, x: &Tensor, scale: Option<f32>) -> Result<Tensor> {
    let Some(s) = scale else {
        return lin.forward(x);
    };
    let mut shape = x.shape().to_vec();
    let k = *shape
        .last()
        .ok_or_else(|| Error::Msg("projection input needs rank >= 1".into()))?;
    if k == 0 {
        return Err(Error::Msg(
            "projection input has a zero-size last dim".into(),
        ));
    }
    let mut y = x
        .reshape(&[x.numel() / k, k])?
        .matmul(&lin.weight.t()?)?
        .mul_scalar(f64::from(s))?;
    if let Some(b) = &lin.bias {
        y = y.add(b)?;
    }
    *shape.last_mut().unwrap() = lin.weight.shape()[0];
    y.reshape(&shape)
}

fn scaled(t: Tensor, scale: Option<f32>) -> Result<Tensor> {
    match scale {
        Some(s) => t.mul_scalar(f64::from(s)),
        None => Ok(t),
    }
}

/// Beam's weighted residual: `branch / sqrt(i) + residual * sqrt((i - 1) / i)`.
fn mix_residual(branch: &Tensor, residual: &Tensor, i: usize) -> Result<Tensor> {
    let i = i as f64;
    branch
        .mul_scalar(1.0 / i.sqrt())?
        .add(&residual.mul_scalar(((i - 1.0) / i).sqrt())?)
}

impl GGUFDecoderBlock {
    pub fn new(
        d_model: usize,
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        intermediate_size: usize,
        rms_eps: f64,
    ) -> Result<Self> {
        if num_kv_heads == 0 || num_heads % num_kv_heads != 0 {
            return Err(Error::Msg(format!(
                "num_heads ({num_heads}) must be a multiple of num_kv_heads ({num_kv_heads})"
            )));
        }
        if head_dim == 0 || head_dim % 2 != 0 {
            return Err(Error::Msg(format!(
                "head dim must be a positive even number, got {head_dim}"
            )));
        }
        let q_dim = num_heads.checked_mul(head_dim).ok_or_else(|| {
            Error::Msg(format!(
                "GGUFDecoderBlock: q_dim ({num_heads} * {head_dim}) overflows"
            ))
        })?;
        let kv_dim = num_kv_heads.checked_mul(head_dim).ok_or_else(|| {
            Error::Msg(format!(
                "GGUFDecoderBlock: kv_dim ({num_kv_heads} * {head_dim}) overflows"
            ))
        })?;
        Ok(Self {
            attn_norm: Norm::rms(d_model, rms_eps)?,
            ffn_norm: Norm::rms(d_model, rms_eps)?,
            q_proj: Linear::with_bias(d_model, q_dim, false)?,
            k_proj: Linear::with_bias(d_model, kv_dim, false)?,
            v_proj: Linear::with_bias(d_model, kv_dim, false)?,
            o_proj: Linear::with_bias(q_dim, d_model, false)?,
            gate_proj: Linear::with_bias(d_model, intermediate_size, false)?,
            up_proj: Linear::with_bias(d_model, intermediate_size, false)?,
            down_proj: Linear::with_bias(intermediate_size, d_model, false)?,
            num_heads,
            num_kv_heads,
            head_dim,
            sliding_window: None,
            moe: None,
            q_scale: None,
            k_scale: None,
            v_scale: None,
            o_scale: None,
            attn_gate: None,
            attn_res: None,
            res_bank: false,
            hc: None,
            dsv4: None,
            moe_hash: None,
            attn_gate_softplus: false,
            attn_gate_log2: false,
            value_moe: None,
            bidirectional: false,
            attn_o_norm: None,
            resid_ab: None,
            ffn_post_scale: None,
            attn_gate_scale: None,
            moe_scale: None,
            dense_with_moe: false,
            moe_norm: None,
            alibi_slopes: None,
            chunk_size: None,
            moe_router_raw: false,
            ssm: None,
            mixer: Mixer::Full,
            nope_temp: None,
            nope_temp_pow: false,
            residual_mix: None,
            q_gain: None,
            skip_scale: None,
            ffn_act_recip: None,
            gate_scale: None,
            up_scale: None,
            down_scale: None,
            clamp_kqv: None,
            q_norm: None,
            k_norm: None,
            qk_norm_full: false,
            post_attn_norm: None,
            attn_sub_norm: None,
            ffn_sub_norm: None,
            g3n: None,
            rwkv: None,
            rwkv7: None,
            qrwkv: None,
            v_norm: false,
            v_from_k: false,
            g4: None,
            g4_moe: None,
            post_ffn_norm: None,
            activation: Activation::Silu,
            attn_scale: 1.0 / (head_dim as f32).sqrt(),
            attn_softcap: None,
            n_rot: head_dim,
            local_rope: false,
            pre_norm: true,
            use_rope: true,
            mla: None,
            gated: true,
            parallel: false,
            sinks: None,
            block_indexer: None,
            qk_norm_after_rope: false,
            parallel_own_ffn_norm: false,
            attn_norm_2: None,
            residual_scale: None,
            norm_residual: None,
        })
    }

    /// Ordinary (GQA) self-attention over `norm_x`: returns the per-head context `[b, s, heads * head_dim]`.
    fn standard_attention(
        &self,
        norm_x: &Tensor,
        rope: &RotaryEmbedding,
        seq_pos: usize,
        cache: &mut Option<(Tensor, Tensor)>,
        sc: [Option<f32>; 4],
        shared: Option<&(Tensor, Tensor)>,
    ) -> Result<Tensor> {
        let (b, s) = (norm_x.shape()[0], norm_x.shape()[1]);
        let h = self.num_heads;
        let hd = self.head_dim;
        let kvh = self.num_kv_heads;
        let clamp = |t: Tensor| match self.clamp_kqv {
            Some(c) => t.clamp(-f64::from(c), f64::from(c)),
            None => Ok(t),
        };
        let mut q = clamp(proj(&self.q_proj, norm_x, sc[0])?)?;
        let mut k = clamp(proj(&self.k_proj, norm_x, sc[1])?)?;
        let v_full = match &self.value_moe {
            Some(vm) => vm.forward(norm_x)?,
            None if self.v_from_k => k.clone(),
            None => proj(&self.v_proj, norm_x, sc[2])?,
        };
        let mut v = clamp(v_full)?.reshape(&[b, s, kvh, hd])?.transpose(1, 2)?;
        if self.v_norm {
            // Gemma 3n: a weightless RMS norm per value head.
            v = v.mul(
                &v.square()?
                    .mean_axis(-1, true)?
                    .add_scalar(self.attn_norm.eps)?
                    .rsqrt()?,
            )?;
        }
        if self.qk_norm_full {
            // OLMo 2: one norm over the whole projection, before the heads are split.
            if let Some(n) = &self.q_norm {
                q = n.forward(&q)?;
            }
            if let Some(n) = &self.k_norm {
                k = n.forward(&k)?;
            }
        }
        let mut q = q.reshape(&[b, s, h, hd])?;
        let mut k = k.reshape(&[b, s, kvh, hd])?;
        if !self.qk_norm_full && !self.qk_norm_after_rope {
            // Qwen3 / Gemma 3: per-head norm over head_dim, before RoPE.
            if let Some(n) = &self.q_norm {
                q = n.forward(&q)?;
            }
            if let Some(n) = &self.k_norm {
                k = n.forward(&k)?;
            }
        }
        let mut q = q.transpose(1, 2)?;
        let k = k.transpose(1, 2)?;
        if !self.use_rope {
            if let Some((scale, floor)) = self.nope_temp {
                let pow = self.nope_temp_pow;
                let f: Vec<f32> = (0..s)
                    .map(|i| {
                        let p = (seq_pos + i + 1) as f32;
                        if pow {
                            (scale as f64 * (p as f64 / floor as f64).ln_1p()).exp() as f32
                        } else {
                            (((seq_pos + i + 1) / floor) as f32 + 1.0).ln() * scale + 1.0
                        }
                    })
                    .collect();
                let f = Tensor::from_slice_on(&f, &[1, 1, s, 1], &q.device())?;
                q = q.mul(&f.expand(q.shape())?)?;
            }
        }

        let (mut q_rot, mut k_rot) = if self.use_rope {
            (
                apply_rope(rope, &q, seq_pos, self.n_rot)?,
                apply_rope(rope, &k, seq_pos, self.n_rot)?,
            )
        } else {
            (q, k)
        };
        if self.qk_norm_after_rope {
            // Maincoder: the per-head norms see the rotated vectors.
            if let Some(n) = &self.q_norm {
                q_rot = n.forward(&q_rot)?;
            }
            if let Some(n) = &self.k_norm {
                k_rot = n.forward(&k_rot)?;
            }
            if let Some(g) = &self.q_gain {
                q_rot = q_rot.mul(&g.expand(q_rot.shape())?)?;
            }
        }
        let mut v = v;
        // A block-sparse layer keeps its indexer keys next to the values (extra columns, repeated over the KV heads).
        if let Some(bi) = &self.block_indexer {
            let ik = bi.keys(norm_x, rope, seq_pos, self.n_rot)?;
            v = Tensor::cat(&[v, ik.expand(&[b, kvh, s, bi.dim])?.contiguous()?], 3)?;
        }
        if let Some((sk, sv)) = shared {
            // A layer that reuses an earlier layer's K/V attends to that cache and adds nothing to its own.
            (k_rot, v) = (sk.clone(), sv.clone());
        } else {
            if let Some((past_k, past_v)) = cache.take() {
                k_rot = Tensor::cat(&[past_k, k_rot], 2)?;
                v = Tensor::cat(&[past_v, v], 2)?;
            }
            *cache = Some((k_rot.clone(), v.clone()));
        }
        let total_k = k_rot.shape()[2];
        let mut block_mask: Option<Vec<bool>> = None;
        if let Some(bi) = &self.block_indexer {
            let keys = v.narrow(3, hd, bi.dim)?.narrow(1, 0, 1)?.contiguous()?;
            let sel = bi.select(norm_x, &keys, rope, seq_pos, self.n_rot)?; // [b, kvh, s, total]
            let (ng, g) = (bi.groups(), h / bi.groups());
            block_mask = Some(
                (0..b * h * s * total_k)
                    .map(|i| {
                        let (row, col) = (i / total_k, i % total_k);
                        let (bi_, hh, t) = (row / (h * s), (row / s) % h, row % s);
                        sel[((bi_ * ng + hh / g) * s + t) * total_k + col]
                    })
                    .collect(),
            );
            v = v.narrow(3, 0, hd)?.contiguous()?;
        }
        if kvh != h {
            // Grouped-query attention: each KV head serves h / kvh consecutive query heads.
            k_rot = k_rot.repeat_interleave(h / kvh, 1)?;
            v = v.repeat_interleave(h / kvh, 1)?;
        }

        let kt = k_rot.transpose(2, 3)?;
        let mut scores = q_rot.matmul(&kt)?.mul_scalar(f64::from(self.attn_scale))?;
        if let Some(cap) = self.attn_softcap {
            scores = scores
                .mul_scalar(1.0 / f64::from(cap))?
                .tanh()?
                .mul_scalar(f64::from(cap))?;
        }
        if let Some(slopes) = &self.alibi_slopes {
            let first = total_k - s;
            let mut bias = Vec::with_capacity(h * s * total_k);
            for &m in slopes {
                for qi in 0..s {
                    let abs_q = (first + qi) as f32;
                    for ki in 0..total_k {
                        bias.push(-m * (abs_q - ki as f32).abs());
                    }
                }
            }
            let bias = Tensor::from_slice_on(&bias, &[1, h, s, total_k], &scores.device())?;
            scores = scores.add(&bias.expand(&[b, h, s, total_k])?)?;
        }
        // A causal LM: position i must not attend to j > i (this block previously attended to
        // the whole sequence, i.e. bidirectionally).
        let past = total_k - s;
        if s > 1
            || self.sliding_window.is_some_and(|w| total_k > w)
            || self.chunk_size.is_some_and(|c| total_k > c)
        {
            let mut mask = Vec::with_capacity(s * total_k);
            for qi in 0..s {
                let abs_q = past + qi;
                for ki in 0..total_k {
                    // future keys, and keys that fell out of the sliding window
                    mask.push(
                        (!self.bidirectional && ki > abs_q)
                            || self.sliding_window.is_some_and(|w| abs_q >= ki + w)
                            || self.chunk_size.is_some_and(|c| abs_q / c != ki / c),
                    );
                }
            }
            let mask = Tensor::from_slice_on(&mask, &[s, total_k], &scores.device())?;
            scores = scores.masked_fill(&mask, -1e9)?;
        }
        if let Some(sel) = &block_mask {
            let mask = Tensor::from_slice_on(sel, &[b, h, s, total_k], &scores.device())?;
            scores = scores.masked_fill(&mask, -1e9)?;
        }
        let attn_weights = match &self.sinks {
            Some(sinks) => {
                // softmax over [scores | sink], then drop the sink column: it only enlarges the denominator.
                let col = sinks
                    .reshape(&[1, h, 1, 1])?
                    .expand(&[b, h, s, 1])?
                    .contiguous()?;
                Tensor::cat(&[scores, col], 3)?
                    .softmax(3)?
                    .narrow(3, 0, total_k)?
                    .contiguous()?
            }
            None => scores.softmax(3)?,
        };
        let attn_out = attn_weights
            .matmul(&v)?
            .transpose(1, 2)?
            .reshape(&[b, s, h * hd])?;
        Ok(attn_out)
    }

    /// One Gemma 3n layer on the AltUp streams `x: [b, s, n_altup, d]` (stream 0 is the active one): predict all streams
    /// from the active one, run attention + LAuReL + the FFN on the predicted active stream, correct the streams with
    /// the result and add the layer's gated per-layer input (`pli: [b, s, width]`) to the non-active streams.
    pub fn forward_g3n(
        &self,
        x: &Tensor,
        rope: &RotaryEmbedding,
        seq_pos: usize,
        cache: &mut Option<(Tensor, Tensor)>,
        shared: Option<&(Tensor, Tensor)>,
        pli: &Tensor,
    ) -> Result<Tensor> {
        let l = self
            .g3n
            .as_ref()
            .ok_or_else(|| Error::Msg("forward_g3n needs a Gemma 3n layer".into()))?;
        let (post_attn, post_ffn) = match (&self.post_attn_norm, &self.post_ffn_norm) {
            (Some(a), Some(f)) => (a, f),
            _ => return Err(Error::Msg("a Gemma 3n layer needs its post norms".into())),
        };
        let (b, s, na, d) = (x.shape()[0], x.shape()[1], x.shape()[2], x.shape()[3]);
        let stream = |t: &Tensor, k: usize| -> Result<Tensor> {
            t.narrow(2, k, 1)?.contiguous()?.reshape(&[b, s, d])
        };
        let inv_sqrt2 = std::f64::consts::FRAC_1_SQRT_2;

        // AltUp prediction: stream j = x_j + sum_i coef[j, i] x_i, the coefficients coming from the active stream.
        let active = stream(x, 0)?;
        let coefs = lin_last(&l.modalities(&active)?, &l.predict_coef)?.reshape(&[b, s, na, na])?;
        let predictions = coefs.matmul(x)?.add(x)?;
        let active_prediction = stream(&predictions, 0)?;

        let cur = self.attn_norm.forward(&active_prediction)?;
        let laurel = l.laurel(&cur)?;
        let ctx = self.standard_attention(&cur, rope, seq_pos, cache, [None; 4], shared)?;
        let attn = lin_last(&ctx, &self.o_proj.weight)?;
        let cur = post_attn.forward(&attn)?.add(&active_prediction)?;
        let attn_laurel = cur.add(&laurel)?.mul_scalar(inv_sqrt2)?;

        let cur = self.ffn_norm.forward(&attn_laurel)?;
        let up = lin_last(&cur, &self.up_proj.weight)?;
        let mut gate = lin_last(&cur, &self.gate_proj.weight)?;
        if l.sparse {
            gate = g3n_gaussian_topk(&gate)?;
        }
        let ffn = lin_last(
            &Activation::GeluTanh.apply(&gate)?.mul(&up)?,
            &self.down_proj.weight,
        )?;
        let attn_ffw_laurel = post_ffn.forward(&ffn)?.add(&attn_laurel)?;

        // AltUp correction: every stream moves by the innovation of the active one.
        let coefs = lin_last(&l.modalities(&attn_ffw_laurel)?, &l.correct_coef)?
            .add_scalar(1.0)?
            .reshape(&[b, s, na, 1])?;
        let innovation = attn_ffw_laurel.sub(&active_prediction)?.reshape(&[b, s, 1, d])?;
        let corrected = predictions.add(&innovation.mul(&coefs)?)?;

        // The per-layer input gates the corrected active stream and is added to the other streams.
        let first = stream(&corrected, 0)?.mul(&l.correct_scale)?;
        let first = Activation::GeluTanh
            .apply(&lin_last(&first, &l.inp_gate)?)?
            .mul(pli)?;
        let first = l
            .post_norm
            .forward(&lin_last(&first, &l.proj)?)?
            .reshape(&[b, s, 1, d])?;
        let rest = corrected.narrow(2, 1, na - 1)?.contiguous()?.add(&first)?;
        Tensor::cat(&[corrected.narrow(2, 0, 1)?.contiguous()?, rest], 2)
    }

    pub fn forward(&self, x: &Tensor, rope: &RotaryEmbedding, seq_pos: usize) -> Result<Tensor> {
        self.forward_cached(x, rope, seq_pos, &mut None)
    }

    /// [`forward_with_ssm_cache`](Self::forward_with_ssm_cache) for blocks that never need the second cache.
    pub fn forward_cached(
        &self,
        x: &Tensor,
        rope: &RotaryEmbedding,
        seq_pos: usize,
        cache: &mut Option<(Tensor, Tensor)>,
    ) -> Result<Tensor> {
        self.forward_with_ssm_cache(x, rope, seq_pos, cache, &mut None)
    }

    /// Like [`forward`](Self::forward), but appends this call's rotated keys / values to `cache`
    /// (un-repeated KV heads, `[b, kv_heads, past + s, head_dim]`) and attends over everything cached,
    /// so incremental decoding only computes the new tokens. `seq_pos` must equal the cached length.
    /// `ssm_cache` is the extra state of a block that runs attention and an SSM side by side.
    pub fn forward_with_ssm_cache(
        &self,
        x: &Tensor,
        rope: &RotaryEmbedding,
        seq_pos: usize,
        cache: &mut Option<(Tensor, Tensor)>,
        ssm_cache: &mut Option<(Tensor, Tensor)>,
    ) -> Result<Tensor> {
        self.forward_scaled(
            x,
            rope,
            seq_pos,
            cache,
            ssm_cache,
            [self.q_scale, self.k_scale, self.v_scale, self.o_scale],
            LayerCtx::default(),
        )
    }

    /// A layer of a sparse-attention MLA model: `slot` carries the latest indexer layer's token selection forward.
    pub fn forward_with_dsa(
        &self,
        x: &Tensor,
        rope: &RotaryEmbedding,
        seq_pos: usize,
        cache: &mut Option<(Tensor, Tensor)>,
        slot: &mut Option<Vec<bool>>,
    ) -> Result<Tensor> {
        let sc = [self.q_scale, self.k_scale, self.v_scale, self.o_scale];
        self.forward_scaled(
            x,
            rope,
            seq_pos,
            cache,
            &mut None,
            sc,
            LayerCtx {
                dsa: Some(slot),
                ..Default::default()
            },
        )
    }

    /// A layer of a model with hash-routed experts: `ids` are the tokens of this call, flattened batch-major.
    pub fn forward_with_ids(
        &self,
        x: &Tensor,
        rope: &RotaryEmbedding,
        seq_pos: usize,
        cache: &mut Option<(Tensor, Tensor)>,
        ids: &[i64],
    ) -> Result<Tensor> {
        let sc = [self.q_scale, self.k_scale, self.v_scale, self.o_scale];
        self.forward_scaled(
            x,
            rope,
            seq_pos,
            cache,
            &mut None,
            sc,
            LayerCtx {
                ids: Some(ids),
                ..Default::default()
            },
        )
    }

    /// A layer of an attention-residual model: `stack` holds this token's banked checkpoints.
    pub fn forward_residual_stack(
        &self,
        x: &Tensor,
        rope: &RotaryEmbedding,
        seq_pos: usize,
        cache: &mut Option<(Tensor, Tensor)>,
        stack: &mut Vec<Tensor>,
    ) -> Result<Tensor> {
        let sc = [self.q_scale, self.k_scale, self.v_scale, self.o_scale];
        self.forward_scaled(
            x,
            rope,
            seq_pos,
            cache,
            &mut None,
            sc,
            LayerCtx {
                res: Some(stack),
                ..Default::default()
            },
        )
    }

    /// [`forward_with_ssm_cache`](Self::forward_with_ssm_cache) with the `[q, k, v, o]` projection scales given
    /// explicitly: HRM-Text re-runs one set of weights in several slots that each carry their own scales.
    /// `ctx` carries the per-call extras some architectures need (see [`LayerCtx`]).
    #[allow(clippy::too_many_arguments)]
    pub fn forward_scaled(
        &self,
        x: &Tensor,
        rope: &RotaryEmbedding,
        seq_pos: usize,
        cache: &mut Option<(Tensor, Tensor)>,
        ssm_cache: &mut Option<(Tensor, Tensor)>,
        sc: [Option<f32>; 4],
        ctx: LayerCtx<'_>,
    ) -> Result<Tensor> {
        let LayerCtx {
            res,
            ids,
            dsa,
            v_first,
            shared_kv,
        } = ctx;
        let mut res = res;
        // 1. Self Attention with Norm and RoPE
        // Attention residuals (Kimi-K3): the layer reads a softmax-weighted mix of the banked checkpoints and `x`.
        let eps = self.attn_norm.eps;
        let res_scores = self.attn_res.as_ref().filter(|_| res.is_some());
        let mixed = match (res_scores, res.as_deref()) {
            (Some([a, _]), Some(stack)) => Some(res_mix(stack, x, a, eps)?),
            _ => None,
        };
        if let (Some(_), Some(stack)) = (res_scores, res.as_deref_mut()) {
            if self.res_bank {
                stack.push(x.clone()); // the raw input, not the mix
            }
        }
        // Hyper-connections (HY-V4): `x` is `[b, s, streams, d]`; the layer sees a gated reduction of the streams.
        let hc_attn = match &self.hc {
            Some(h) => Some(h.pre(&h.attn, x)?),
            None => None,
        };
        if let Some(r) = &self.rwkv {
            return r.forward(x, cache); // the whole layer, norms and residuals included
        }
        if let Some(r) = &self.qrwkv {
            return r.forward(x, cache);
        }
        if let Some(r) = &self.rwkv7 {
            let v_first = v_first
                .ok_or_else(|| Error::Msg("an RWKV-7 layer needs the shared value state".into()))?;
            return r.forward(x, cache, v_first);
        }
        let x_in = hc_attn.as_ref().map_or(x, |(c, _)| c);
        let norm_x = if self.pre_norm {
            self.attn_norm.forward(mixed.as_ref().unwrap_or(x_in))?
        } else {
            x_in.clone()
        };
        if self.mixer == Mixer::FfnOnly {
            return x.add(&self.ffn_branch(&norm_x, x, ids)?);
        }
        let mut proj_out = if let (true, Some(ssm)) =
            (matches!(self.mixer, Mixer::Ssm | Mixer::SsmFfn), &self.ssm)
        {
            let y = ssm.forward(&norm_x, cache)?;
            if self.mixer == Mixer::Ssm {
                return x.add(&y);
            }
            y
        } else {
            let attn_in = match &self.attn_norm_2 {
                Some(n2) => n2.forward(x)?,
                None => norm_x.clone(),
            };
            let attn_out = match (&self.dsv4, &self.mla) {
                (Some(a), _) => a.forward(&attn_in, seq_pos, cache)?, // already through the output projection
                (None, Some(mla)) => mla.forward(
                    &attn_in,
                    rope,
                    seq_pos,
                    cache,
                    self.attn_scale,
                    self.sliding_window,
                    dsa,
                )?,
                (None, None) => {
                    self.standard_attention(&attn_in, rope, seq_pos, cache, sc, shared_kv.as_ref())?
                }
            };
            let attn_out = match &self.attn_o_norm {
                Some(n) => {
                    let shape = attn_out.shape().to_vec();
                    let (b, s) = (shape[0], shape[1]);
                    n.forward(&attn_out.reshape(&[b, s, self.num_heads, self.head_dim])?)?
                        .reshape(&shape)?
                }
                None => attn_out,
            };
            let attn_out = match &self.attn_gate {
                Some(g) => {
                    let gate = scaled(g.forward(&attn_in)?, self.attn_gate_scale)?;
                    let gate = if self.attn_gate_log2 {
                        gate.mul_scalar(std::f64::consts::LN_2)?
                            .softplus()?
                            .mul_scalar(std::f64::consts::LOG2_E)?
                    } else if self.attn_gate_softplus {
                        gate.softplus()?
                    } else {
                        gate.sigmoid()?
                    };
                    let (b, s, w) = (
                        attn_out.shape()[0],
                        attn_out.shape()[1],
                        attn_out.shape()[2],
                    );
                    if gate.shape()[2] == w {
                        attn_out.mul(&gate)?
                    } else {
                        let h = gate.shape()[2];
                        let gate = gate
                            .reshape(&[b, s, h, 1])?
                            .expand(&[b, s, h, w / h])?
                            .reshape(&[b, s, w])?;
                        attn_out.mul(&gate)?
                    }
                }
                None => attn_out,
            };
            let attn_out = match &self.attn_sub_norm {
                Some(n) => n.forward(&attn_out)?,
                None => attn_out,
            };
            if self.dsv4.is_some() {
                attn_out
            } else {
                proj(&self.o_proj, &attn_out, sc[3])?
            }
        };
        if let (Mixer::AttnSsm, Some(ssm)) = (self.mixer, &self.ssm) {
            proj_out = proj_out.add(&ssm.forward(&norm_x, ssm_cache)?)?;
        }
        if let Some(n) = &self.post_attn_norm {
            proj_out = n.forward(&proj_out)?;
        }
        proj_out = scaled(proj_out, self.residual_scale)?;
        let h_res = match (&self.resid_ab, self.residual_mix) {
            (Some([a, b, _, _]), _) => x.mul(a)?.add(&proj_out.mul(b)?)?,
            (None, Some(i)) => mix_residual(&proj_out, x, i)?,
            (None, None) if self.norm_residual.is_some() => {
                proj_out.add(&norm_x.mul_scalar(f64::from(self.norm_residual.unwrap_or(1.0)))?)?
            }
            // A banking layer restarts the stream from its attention output alone.
            (None, None) if res_scores.is_some() && self.res_bank => proj_out,
            (None, None) if hc_attn.is_some() => match &hc_attn {
                Some((_, post)) => hc_post(&proj_out, x, post)?,
                None => unreachable!("guarded"),
            },
            (None, None) => x.add(&proj_out)?,
        };
        if self.mixer == Mixer::AttnOnly {
            return Ok(h_res);
        }

        // 2. SwiGLU Feed-Forward Network
        let ffn_mixed = match (res_scores, res.as_deref()) {
            (Some([_, f]), Some(stack)) => Some(res_mix(stack, &h_res, f, eps)?),
            _ => None,
        };
        let hc_ffn = match &self.hc {
            Some(h) => Some(h.pre(&h.ffn, &h_res)?),
            None => None,
        };
        let ffn_src = hc_ffn.as_ref().map_or(&h_res, |(c, _)| c);
        let norm_ffn = if self.parallel && self.parallel_own_ffn_norm {
            self.ffn_norm.forward(x)?
        } else if self.parallel {
            norm_x.clone()
        } else if self.pre_norm {
            self.ffn_norm
                .forward(ffn_mixed.as_ref().unwrap_or(ffn_src))?
        } else {
            ffn_src.clone()
        };
        let down = match &self.g4_moe {
            Some(g) => self.g4_moe_branch(g, &h_res, &norm_ffn)?,
            None => self.ffn_branch(&norm_ffn, x_in, ids)?,
        };
        if let Some((_, post)) = &hc_ffn {
            return hc_post(&down, &h_res, post);
        }
        if let Some(a) = self.norm_residual {
            return down.add(&norm_ffn.mul_scalar(f64::from(a))?);
        }
        match (&self.resid_ab, self.residual_mix) {
            (Some([_, _, a, b]), _) => h_res.mul(a)?.add(&down.mul(b)?),
            (None, Some(i)) => mix_residual(&down, &h_res, i + 1),
            (None, None) => h_res.add(&down),
        }
    }

    /// Gemma 4's MoE layer: a dense gated-GELU MLP on the normed residual plus routed experts on the residual through a
    /// second norm, each followed by its own post norm, summed and passed through the post-FFN norm.
    fn g4_moe_branch(&self, g: &G4Moe, h: &Tensor, norm_ffn: &Tensor) -> Result<Tensor> {
        let moe = self
            .moe
            .as_ref()
            .ok_or_else(|| Error::Msg("a Gemma 4 MoE layer needs its experts".into()))?;
        let d = h.shape()[h.shape().len() - 1] as f64;
        let gate = self.activation.apply(&self.gate_proj.forward(norm_ffn)?)?;
        let mlp = self.down_proj.forward(&gate.mul(&self.up_proj.forward(norm_ffn)?)?)?;
        let mlp = g.post_norm_1.forward(&mlp)?;
        let router_in = h
            .mul(&h.square()?.mean_axis(-1, true)?.add_scalar(self.attn_norm.eps)?.rsqrt()?)?
            .mul_scalar(1.0 / d.sqrt())?
            .mul(&g.router_scale)?;
        let experts = moe.forward_routed_forced(&g.pre_norm_2.forward(h)?, &router_in, None)?;
        let sum = mlp.add(&g.post_norm_2.forward(&experts)?)?;
        match &self.post_ffn_norm {
            Some(n) => n.forward(&sum),
            None => Ok(sum),
        }
    }

    /// The FFN / MoE half of a layer: `norm_ffn` is its (already normalized) input, `x` the layer input.
    fn ffn_branch(&self, norm_ffn: &Tensor, x: &Tensor, ids: Option<&[i64]>) -> Result<Tensor> {
        let dense = || -> Result<Tensor> {
            if !self.gated {
                let up = scaled(self.up_proj.forward(norm_ffn)?, self.up_scale)?;
                let mut act = self.activation.apply(&up)?;
                if let Some(r) = &self.ffn_act_recip {
                    let shape = act.shape().to_vec();
                    act = act.mul(&r.reshape(&[1, 1, shape[2]])?.expand(&shape)?)?;
                }
                scaled(self.down_proj.forward(&act)?, self.down_scale)
            } else {
                let gate = self
                    .activation
                    .apply(&scaled(self.gate_proj.forward(norm_ffn)?, self.gate_scale)?)?;
                let up = self
                    .activation
                    .up(&scaled(self.up_proj.forward(norm_ffn)?, self.up_scale)?)?;
                let mut inner = gate.mul(&up)?;
                if let Some(n) = &self.ffn_sub_norm {
                    inner = n.forward(&inner)?;
                }
                scaled(self.down_proj.forward(&inner)?, self.down_scale)
            }
        };
        let down = match &self.moe {
            Some(moe) if self.dense_with_moe => {
                let moe_in = match &self.moe_norm {
                    Some(n) => n.forward(x)?,
                    None => norm_ffn.clone(),
                };
                scaled(dense()?.add(&moe.forward(&moe_in)?)?, self.moe_scale)?
            }
            Some(moe) => {
                let router_in = if self.moe_router_raw { x } else { norm_ffn };
                // Hash-routed layers (DeepSeek-V4) pick each token's experts from a table indexed by its id.
                let forced: Option<Vec<usize>> = match (&self.moe_hash, ids) {
                    (Some((table, used)), Some(ids)) => Some(
                        ids.iter()
                            .flat_map(|&t| {
                                table[t as usize * used..(t as usize + 1) * used]
                                    .iter()
                                    .map(|&e| e as usize)
                            })
                            .collect(),
                    ),
                    (Some(_), None) => {
                        return Err(Error::Msg("a hash-routed layer needs the token ids".into()));
                    }
                    _ => None,
                };
                scaled(
                    moe.forward_routed_forced(norm_ffn, router_in, forced.as_deref())?,
                    self.moe_scale,
                )?
            }
            None => dense()?,
        };

        let down = match &self.ffn_post_scale {
            Some(w) => down.mul(w)?,
            None => down,
        };
        let down = match &self.post_ffn_norm {
            Some(n) => n.forward(&down)?,
            None => down,
        };
        scaled(down, self.residual_scale)
    }
}

/// End-to-end Causal Language Model constructed and executed directly from GGUF models.
pub struct GGUFCausalLM {
    pub config: GGUFModelConfig,
    pub tok_emb: Embedding,
    pub rope: RotaryEmbedding,
    /// Rotary table of the sliding-window layers when their base differs (Gemma 2/3).
    pub rope_local: Option<RotaryEmbedding>,
    pub blocks: Vec<GGUFDecoderBlock>,
    pub norm: Norm,
    pub lm_head: Linear,
    /// `output.scale`: logits = `(W h) * scale` (+ `output.bias` for Qwen2).
    pub lm_head_scale: Option<f32>,
    /// Learned absolute position table `[positions, d_model]` (GPT-2, StarCoder).
    pub pos_emb: Option<Tensor>,
    /// LayerNorm applied to the token embeddings (BLOOM).
    pub emb_norm: Option<Norm>,
    /// HRM-Text's learned initial low-cycle state, `[d_model]`.
    pub hrm_z_l: Option<Tensor>,
    /// `output_res_score.weight`: the final attention-residual mix (Kimi-K3).
    pub output_res_score: Option<Tensor>,
    /// `output_hc_{fn,base,scale}`: collapses the hyper-connection streams before the final norm.
    pub hc_head: Option<HcParams>,
    /// Per-layer n-gram hash embeddings (Qwen4-Exp).
    pub ple: Option<PleModel>,
    /// Gemma 3n's stream / per-layer-input tensors.
    pub g3n: Option<G3nModel>,
    /// Gemma 4's per-layer-input tensors.
    pub g4: Option<G4Model>,
    /// HRM-Text's `[q, k, v, o]` projection scales per cache slot (the scales are indexed by slot, the weights are shared).
    pub hrm_slot_scales: Vec<[Option<f32>; 4]>,
}

impl GGUFCausalLM {
    /// Builds a runnable causal language model initialized to GGUF model dimensions.
    pub fn from_config(config: GGUFModelConfig) -> Result<Self> {
        let head_dim = if config.head_dim > 0 {
            config.head_dim
        } else {
            super::transformer::check_heads(config.d_model, config.num_heads)?;
            config.d_model / config.num_heads
        };
        let n_rot = if config.rope_n_rot > 0 {
            config.rope_n_rot
        } else {
            head_dim
        };
        let mla = config.kv_lora_rank > 0;
        if n_rot > head_dim || n_rot % 2 != 0 {
            return Err(Error::Msg(format!(
                "rotary dims {n_rot} must be even and within the head dim {head_dim}"
            )));
        }
        crate::limits::check_repeats("num_layers", config.num_layers)?;
        let mut blocks = Vec::with_capacity(config.num_layers);
        for il in 0..config.num_layers {
            // Sliding-window layers may have a narrower (or wider) head than the full-attention ones (Gemma 4).
            let head_dim = if config.sliding_window.is_some() && config.swa.is_swa(il) {
                config.head_dim_swa.unwrap_or(head_dim)
            } else {
                head_dim
            };
            let mixer = config.mixer(il);
            let (layer_h, layer_kv) = match mixer {
                Mixer::Full | Mixer::AttnOnly | Mixer::AttnSsm => config.layer_dims(il),
                Mixer::Ssm | Mixer::FfnOnly | Mixer::SsmFfn => (1, 1),
            };
            if layer_h == 0 || layer_kv == 0 {
                return Err(Error::Msg(format!(
                    "layer {il} has {layer_h} heads / {layer_kv} KV heads: attention-free or linear-attention layers are not supported"
                )));
            }
            let mut blk = GGUFDecoderBlock::new(
                config.d_model,
                layer_h,
                layer_kv,
                // MLA blocks never use the ordinary Q/K/V projections: keep their placeholders tiny.
                if mla || matches!(mixer, Mixer::Ssm | Mixer::FfnOnly | Mixer::SsmFfn) {
                    2
                } else {
                    head_dim
                },
                match mixer {
                    Mixer::Ssm | Mixer::AttnOnly => 1,
                    Mixer::FfnOnly => config
                        .layer_ff
                        .get(il)
                        .copied()
                        .unwrap_or(config.intermediate_size),
                    Mixer::Full | Mixer::SsmFfn | Mixer::AttnSsm if config.expert_count > 0 => 1,
                    Mixer::Full | Mixer::SsmFfn | Mixer::AttnSsm => config.intermediate_size,
                },
                config.rms_eps,
            )?;
            blk.mixer = mixer;
            let swa = config.sliding_window.is_some() && config.swa.is_swa(il);
            blk.sliding_window = if swa && !config.chunked_attn {
                config.sliding_window
            } else {
                None
            };
            blk.chunk_size = if swa && config.chunked_attn {
                config.sliding_window
            } else {
                None
            };
            blk.nope_temp = config.nope_temp;
            blk.nope_temp_pow = config.nope_temp_pow;
            blk.residual_mix = config.weighted_residual.then_some(2 * (il + 1));
            blk.local_rope = swa && config.rope_base_swa.is_some();
            blk.clamp_kqv = config.clamp_kqv;
            blk.activation = match config.xielu.get(il) {
                Some(&p) => Activation::Xielu(p),
                None => config.activation,
            };
            blk.attn_scale = if mla && !config.mla_plain {
                mla_kq_scale(&config, head_dim)
            } else {
                config.attn_scale.unwrap_or(1.0 / (head_dim as f32).sqrt())
            };
            blk.attn_softcap = config.attn_softcap;
            blk.alibi_slopes = config.alibi_max_bias.map(|mb| alibi_slopes(layer_h, mb));
            blk.n_rot = if swa {
                config.rope_n_rot_swa.unwrap_or(n_rot)
            } else {
                n_rot
            };
            blk.pre_norm = config.pre_norm;
            blk.gated = config.ffn_gated;
            blk.parallel = config.parallel_residual;
            blk.parallel_own_ffn_norm = config.parallel_own_ffn_norm;
            let mk = |dim: usize| {
                Norm::build(
                    config.norm_layer,
                    config.norm_bias,
                    config.norm_scaled,
                    dim,
                    config.rms_eps,
                )
            };
            blk.attn_norm = mk(config.d_model)?.with_groups(config.norm_groups);
            blk.ffn_norm = mk(config.d_model)?.with_groups(config.norm_groups);
            blk.use_rope = config
                .no_rope_step
                .is_none_or(|n| (il + config.no_rope_offset) % n != 0)
                && config.rope_layers.get(il).copied().unwrap_or(true)
                && !config.abs_pos
                && config.alibi_max_bias.is_none()
                && (!config.rope_only_swa || config.sliding_window.is_none() || swa);
            blk.qk_norm_after_rope = config.qk_norm_after_rope;
            blk.residual_scale = config.residual_scale;
            blk.norm_residual = config.norm_residual;
            match config.qk_norm {
                QkNorm::None => {}
                QkNorm::PerHead if config.kq_norm_weightless => {
                    if blk.use_rope {
                        blk.q_norm =
                            Some(Norm::build(false, false, false, head_dim, config.rms_eps)?);
                        blk.k_norm =
                            Some(Norm::build(false, false, false, head_dim, config.rms_eps)?);
                    }
                }
                QkNorm::PerHead => {
                    blk.q_norm = Some(Norm::rms(head_dim, config.rms_eps)?);
                    blk.k_norm = Some(Norm::rms(head_dim, config.rms_eps)?);
                }
                QkNorm::Full => {
                    let q_dim = layer_h.checked_mul(head_dim).ok_or_else(|| {
                        Error::Msg(format!(
                            "from_config: q_dim ({layer_h} * {head_dim}) overflows"
                        ))
                    })?;
                    let kv_dim = layer_kv.checked_mul(head_dim).ok_or_else(|| {
                        Error::Msg(format!(
                            "from_config: kv_dim ({layer_kv} * {head_dim}) overflows"
                        ))
                    })?;
                    let mk = |dim: usize| {
                        if config.norm_layer {
                            Norm::build(
                                true,
                                config.norm_bias,
                                config.norm_scaled,
                                dim,
                                config.rms_eps,
                            )
                        } else {
                            Norm::rms(dim, config.rms_eps)
                        }
                    };
                    blk.q_norm = Some(mk(q_dim)?);
                    blk.k_norm = Some(mk(kv_dim)?);
                    blk.qk_norm_full = true;
                }
            }
            blk.bidirectional = arch_spec(&config.architecture).is_some_and(|s| s.non_causal);
            if arch_spec(&config.architecture).is_some_and(|s| s.spark_blocks) {
                let post = config.post_norm_eps.unwrap_or(config.rms_eps);
                blk.post_attn_norm = Some(Norm::build(false, false, false, config.d_model, post)?);
                blk.post_ffn_norm = Some(Norm::build(false, false, false, config.d_model, post)?);
                blk.attn_o_norm = Some(Norm::build(false, false, false, head_dim, config.rms_eps)?);
            }
            if config.post_norms {
                let eps = arch_spec(&config.architecture)
                    .and_then(|s| s.post_norm_eps)
                    .unwrap_or(config.rms_eps);
                blk.post_attn_norm = Some(Norm::rms(config.d_model, eps)?);
                blk.post_ffn_norm = Some(Norm::rms(config.d_model, eps)?);
            }
            blocks.push(blk);
        }

        let base = f64::from(config.rope_freq_base);
        let rope = rope_table(
            n_rot,
            config.max_seq_len,
            base,
            &config.rope_scaling,
            None,
            config.rope_attn_factor,
        )?;
        let rope_local = match config.rope_base_swa {
            Some(b) => {
                let scaling = if config.local_rope_unscaled {
                    RopeScaling::None
                } else {
                    config.rope_scaling.clone()
                };
                Some(rope_table(
                    config.rope_n_rot_swa.unwrap_or(n_rot),
                    config.max_seq_len,
                    f64::from(b),
                    &scaling,
                    None,
                    config.rope_attn_factor,
                )?)
            }
            None => None,
        };
        Ok(Self {
            tok_emb: Embedding::new(config.vocab_size, config.d_model)?,
            rope,
            rope_local,
            blocks,
            norm: Norm::build(
                config.norm_layer,
                config.norm_bias,
                config.norm_scaled,
                config.d_model,
                config.rms_eps,
            )?
            .with_groups(config.norm_groups),
            lm_head: Linear::with_bias(config.d_model, config.vocab_size, false)?,
            lm_head_scale: None,
            pos_emb: None,
            emb_norm: None,
            hrm_z_l: None,
            output_res_score: None,
            hc_head: None,
            ple: None,
            g3n: None,
            g4: None,
            hrm_slot_scales: Vec::new(),
            config,
        })
    }

    /// Forward pass computing next-token logits from input token sequence `[batch, seq_len]`.
    pub fn forward(&self, input_ids: &Tensor) -> Result<Tensor> {
        self.forward_cached(input_ids, &mut self.new_cache())
    }

    /// Gemma 4's layer stack: ordinary Gemma-style blocks (pre/post norms, sliding windows, per-layer head widths), the
    /// last few attending to earlier layers' K/V, each followed by its gated per-layer input and its output scalar.
    fn g4_layers(
        &self,
        g: &G4Model,
        h: &Tensor,
        ids: &Tensor,
        past: usize,
        cache: &mut [Option<(Tensor, Tensor)>],
    ) -> Result<Tensor> {
        let (b, s, d) = (h.shape()[0], h.shape()[1], h.shape()[2]);
        let n_layers = self.blocks.len();
        let per_layer = match &g.per_layer {
            Some(p) => {
                let w = p.width;
                let own = p
                    .emb
                    .forward(ids)?
                    .mul_scalar((w as f64).sqrt())?
                    .reshape(&[b, s, n_layers, w])?;
                let projected = lin_last(h, &p.model_proj)?
                    .mul_scalar(1.0 / (d as f64).sqrt())?
                    .reshape(&[b, s, n_layers, w])?;
                Some((
                    p.proj_norm
                        .forward(&projected)?
                        .add(&own)?
                        .mul_scalar(std::f64::consts::FRAC_1_SQRT_2)?,
                    w,
                ))
            }
            None => None,
        };
        let mut x = h.clone();
        for (il, block) in self.blocks.iter().enumerate() {
            let layer = block
                .g4
                .as_ref()
                .ok_or_else(|| Error::Msg(format!("layer {il} has no Gemma 4 tensors")))?;
            let rope = if block.local_rope {
                self.rope_local.as_ref().unwrap_or(&self.rope)
            } else {
                &self.rope
            };
            let shared_kv = layer.kv_source.and_then(|src| cache[src].clone());
            let mut y = block.forward_scaled(
                &x,
                rope,
                past,
                &mut cache[il],
                &mut None,
                [block.q_scale, block.k_scale, block.v_scale, block.o_scale],
                LayerCtx {
                    shared_kv,
                    ..Default::default()
                },
            )?;
            if let (Some((gate, proj, post)), Some((pl, w))) = (&layer.per_layer, &per_layer) {
                let pli = pl.narrow(2, il, 1)?.contiguous()?.reshape(&[b, s, *w])?;
                let gated = Activation::GeluTanh
                    .apply(&lin_last(&y, gate)?)?
                    .mul(&pli)?;
                y = y.add(&post.forward(&lin_last(&gated, proj)?)?)?;
            }
            if let Some(scale) = layer.out_scale {
                y = y.mul_scalar(f64::from(scale))?;
            }
            x = y;
        }
        Ok(x)
    }

    /// Gemma 3n's layer stack: lifts the scaled embedding `h` into the AltUp streams, runs every layer with its
    /// per-layer input, and merges the streams back into one `[b, s, d]` state.
    fn g3n_layers(
        &self,
        g: &G3nModel,
        h: &Tensor,
        ids: &Tensor,
        past: usize,
        cache: &mut [Option<(Tensor, Tensor)>],
    ) -> Result<Tensor> {
        let (b, s, d) = (h.shape()[0], h.shape()[1], h.shape()[2]);
        let (n_layers, w, na) = (self.blocks.len(), g.width, g.n_altup);
        // Per-layer inputs `[b, s, layers, width]`: the token's own embedding plus a projection of the state.
        let own = g
            .per_layer_emb
            .forward(ids)?
            .mul_scalar((w as f64).sqrt())?
            .reshape(&[b, s, n_layers, w])?;
        let projected = lin_last(h, &g.per_layer_model_proj)?
            .mul_scalar(1.0 / (d as f64).sqrt())?
            .reshape(&[b, s, n_layers, w])?;
        let per_layer = g
            .per_layer_proj_norm
            .forward(&projected)?
            .add(&own)?
            .mul_scalar(std::f64::consts::FRAC_1_SQRT_2)?;
        // Extra streams: projections of the state, rescaled to its length.
        let target = g3n_magnitude(h)?;
        let mut streams = vec![h.clone()];
        for k in 0..na - 1 {
            let wk = g.altup_proj.narrow(0, k, 1)?.contiguous()?.reshape(&[d, d])?;
            streams.push(g3n_match_magnitude(&lin_last(h, &wk)?, &target)?);
        }
        let mut x = Tensor::stack(&streams, 2)?;
        for (il, block) in self.blocks.iter().enumerate() {
            let layer = block
                .g3n
                .as_ref()
                .ok_or_else(|| Error::Msg(format!("layer {il} has no Gemma 3n tensors")))?;
            let rope = if block.local_rope {
                self.rope_local.as_ref().unwrap_or(&self.rope)
            } else {
                &self.rope
            };
            let shared = layer.kv_source.and_then(|src| cache[src].clone());
            let pli = per_layer.narrow(2, il, 1)?.contiguous()?.reshape(&[b, s, w])?;
            x = block.forward_g3n(&x, rope, past, &mut cache[il], shared.as_ref(), &pli)?;
        }
        // Merge: the active stream plus the other streams brought back to its length, averaged.
        let active = x.narrow(2, 0, 1)?.contiguous()?.reshape(&[b, s, d])?;
        let target = g3n_magnitude(&active)?;
        let mut sum = active;
        for k in 0..na - 1 {
            let wk = g
                .altup_unembd_proj
                .narrow(0, k, 1)?
                .contiguous()?
                .reshape(&[d, d])?;
            let stream = x.narrow(2, k + 1, 1)?.contiguous()?.reshape(&[b, s, d])?;
            sum = sum.add(&g3n_match_magnitude(&lin_last(&stream, &wk)?, &target)?)?;
        }
        sum.mul_scalar(1.0 / na as f64)
    }

    /// Whether this is a sparse-attention MLA model whose lightning indexers run (so contexts beyond `top_k` are fine).
    fn has_dsa_indexer(&self) -> bool {
        self.blocks.iter().any(|b| {
            b.block_indexer.is_some()
                || b.mla
                    .as_ref()
                    .is_some_and(|m| m.indexer.is_some() || m.share_dsa)
        })
    }

    /// One empty per-layer KV cache for [`forward_cached`](Self::forward_cached).
    pub fn new_cache(&self) -> Vec<Option<(Tensor, Tensor)>> {
        if let Some((lps, hc, lc)) = self.config.hrm {
            return vec![None; lps * hc * (lc + 1)];
        }
        // One slot per layer, plus one more per parallel attention + SSM layer for its SSM state.
        // The last slot, when the model has hash embeddings, holds their conv history and token history.
        vec![
            None;
            self.blocks.len()
                + self
                    .blocks
                    .iter()
                    .filter(|b| b.mixer == Mixer::AttnSsm)
                    .count()
                + usize::from(self.ple.is_some())
        ]
    }

    /// Runs `input_ids` (`[batch, new_tokens]`) on top of whatever `cache` already holds; the cache
    /// grows by `new_tokens` positions. Logits are for the new tokens only.
    pub fn forward_cached(
        &self,
        input_ids: &Tensor,
        cache: &mut [Option<(Tensor, Tensor)>],
    ) -> Result<Tensor> {
        if input_ids.shape().len() != 2 {
            return Err(Error::Msg(format!(
                "input_ids must be [batch, new_tokens], got rank {}",
                input_ids.shape().len()
            )));
        }
        let cached = self
            .blocks
            .iter()
            .position(|b| matches!(b.mixer, Mixer::Full | Mixer::AttnOnly | Mixer::AttnSsm))
            .and_then(|i| cache.get(i))
            .and_then(|c| c.as_ref());
        let past = match cached {
            None => 0,
            Some((k, _)) => k.shape().get(2).copied().ok_or_else(|| {
                Error::Msg("KV cache slot is malformed: the key cache needs rank >= 3".into())
            })?,
        };
        let mut h = self.tok_emb.forward(input_ids)?;
        if let Some(s) = self.config.embedding_scale {
            h = h.mul_scalar(f64::from(s))?;
        }
        if let Some(n) = &self.emb_norm {
            h = n.forward(&h)?;
        }
        if let Some(pe) = &self.pos_emb {
            let (b, s, d) = (h.shape()[0], h.shape()[1], h.shape()[2]);
            if past + s > pe.shape()[0] {
                return Err(Error::Msg(format!(
                    "position {} is beyond the model's {} learned positions",
                    past + s,
                    pe.shape()[0]
                )));
            }
            h = h.add(
                &pe.narrow(0, past, s)?
                    .reshape(&[1, s, d])?
                    .expand(&[b, s, d])?,
            )?;
        }
        if let Some(top_k) = self.config.dsa_top_k.filter(|_| !self.has_dsa_indexer()) {
            if past + input_ids.shape()[1] > top_k {
                return Err(Error::Msg(format!(
                    "sparse attention (indexer top_k = {top_k}) is not implemented: contexts beyond {top_k} tokens would need the indexer"
                )));
            }
        }
        if cache.len() < self.new_cache().len() {
            return Err(Error::Msg(
                "cache was not created by this model's new_cache()".into(),
            ));
        }
        if let Some((lps, hc, lc)) = self.config.hrm {
            // HRM-Text: the low stack (blocks 0..lps) runs `lc` times per high cycle, then the high stack
            // (blocks lps..2*lps) once; every pass has its own KV slot and ends in a parameterless norm.
            let z_l = self
                .hrm_z_l
                .as_ref()
                .ok_or_else(|| Error::Msg("hrm_text model has no hrm.z_l_init".into()))?;
            let mut z_l = z_l.reshape(&[1, 1, z_l.numel()])?.expand(h.shape())?;
            let mut z_h = h;
            let mut run = |x: Tensor, slot_base: usize, blk_base: usize| -> Result<Tensor> {
                let mut x = x;
                for il in 0..lps {
                    let sc = self.hrm_slot_scales[slot_base + il];
                    x = self.blocks[blk_base + il].forward_scaled(
                        &x,
                        &self.rope,
                        past,
                        &mut cache[slot_base + il],
                        &mut None,
                        sc,
                        LayerCtx::default(),
                    )?;
                }
                self.norm.forward(&x)
            };
            for hh in 0..hc {
                for l in 0..lc {
                    z_l = run(z_h.add(&z_l)?, (hh * (lc + 1) + l) * lps, 0)?;
                }
                z_h = run(z_h.add(&z_l)?, (hh * (lc + 1) + lc) * lps, lps)?;
            }
            h = z_h;
        }
        if let Some((n, _, _)) = self.config.hc {
            // Hyper-connections start from `n` copies of the embedding.
            let (b, s, d) = (h.shape()[0], h.shape()[1], h.shape()[2]);
            h = h
                .reshape(&[b, s, 1, d])?
                .expand(&[b, s, n, d])?
                .contiguous()?;
        }
        if let Some(g) = &self.g3n {
            h = self.g3n_layers(g, &h, input_ids, past, cache)?;
        }
        if let Some(g) = &self.g4 {
            h = self.g4_layers(g, &h, input_ids, past, cache)?;
        }
        let no_blocks: &[GGUFDecoderBlock] = &[];
        let blocks = if self.config.hrm.is_some() || self.g3n.is_some() || self.g4.is_some() {
            no_blocks
        } else {
            &self.blocks
        };
        let embedding = h.clone();
        // Hash embeddings: the tokens before this call and the conv history live in the cache's last slot.
        let (mut ple_emb, mut ple_hist, mut ple_ids) = (None, None, Vec::new());
        if let Some(ple) = &self.ple {
            let (b, s) = (input_ids.shape()[0], input_ids.shape()[1]);
            let flat = input_ids.to_vec::<i64>()?;
            ple_ids = (0..b)
                .map(|bi| flat[bi * s..(bi + 1) * s].to_vec())
                .collect::<Vec<_>>();
            let n_prev = ple.n_prev();
            let (conv_hist, tok_hist) = match cache.last().and_then(|c| c.as_ref()) {
                Some((c, t)) => (
                    Some(c.clone()),
                    t.to_vec::<f32>()?
                        .iter()
                        .map(|&v| v as i64)
                        .collect::<Vec<_>>(),
                ),
                None => (None, vec![-1; b * n_prev]),
            };
            let prev: Vec<Vec<i64>> = (0..b)
                .map(|bi| tok_hist[bi * n_prev..(bi + 1) * n_prev].to_vec())
                .collect();
            ple_emb = Some(ple.embed(&ple_ids, &prev)?);
            ple_hist = conv_hist;
            ple_ids = ple_ids
                .iter()
                .zip(&prev)
                .map(|(ids, p)| p.iter().chain(ids).copied().collect())
                .collect();
        }
        let flat_ids: Vec<i64> = if self.blocks.iter().any(|b| b.moe_hash.is_some()) {
            input_ids.to_vec::<i64>()?
        } else {
            Vec::new()
        };
        let mut new_ple_hist = None;
        let mut dsa_slot: Option<Vec<bool>> = None;
        let (layer_caches, extra) = cache.split_at_mut(self.blocks.len());
        let mut extra = extra.iter_mut();
        let mut res_stack: Vec<Tensor> = Vec::new();
        let mut v_first: Option<Tensor> = None;
        for (il, (block, layer_cache)) in blocks.iter().zip(layer_caches.iter_mut()).enumerate() {
            if let (Some(ple), Some(emb)) = (&self.ple, &ple_emb) {
                for l in ple.layers.iter().filter(|l| l.layer == il) {
                    let (mixed, hist) = ple.apply(l, &h, emb, ple_hist.as_ref(), self.norm.eps)?;
                    h = mixed;
                    new_ple_hist = Some(hist);
                }
            }
            let rope = if block.local_rope {
                self.rope_local.as_ref().unwrap_or(&self.rope)
            } else {
                &self.rope
            };
            h = if block.rwkv7.is_some() {
                block.forward_scaled(
                    &h,
                    rope,
                    past,
                    layer_cache,
                    &mut None,
                    [None; 4],
                    LayerCtx {
                        v_first: Some(&mut v_first),
                        ..Default::default()
                    },
                )?
            } else if block.moe_hash.is_some() {
                block.forward_with_ids(&h, rope, past, layer_cache, &flat_ids)?
            } else if self.output_res_score.is_some() {
                block.forward_residual_stack(&h, rope, past, layer_cache, &mut res_stack)?
            } else if block.mixer == Mixer::AttnSsm {
                let slot = extra
                    .next()
                    .ok_or_else(|| Error::Msg("missing SSM cache slot".into()))?;
                block.forward_with_ssm_cache(&h, rope, past, layer_cache, slot)?
            } else if self.has_dsa_indexer() {
                block.forward_with_dsa(&h, rope, past, layer_cache, &mut dsa_slot)?
            } else {
                block.forward_cached(&h, rope, past, layer_cache)?
            };
            if let Some(s) = block.skip_scale {
                h = h.add(&embedding.mul_scalar(f64::from(s))?)?;
            }
        }
        if let (Some(ple), Some(hist)) = (&self.ple, new_ple_hist) {
            let n_prev = ple.n_prev();
            let tail: Vec<f32> = ple_ids
                .iter()
                .flat_map(|all| all[all.len() - n_prev..].iter().map(|&t| t as f32))
                .collect();
            let tok_hist = Tensor::from_slice(&tail, &[ple_ids.len(), n_prev])?;
            if let Some(slot) = cache.last_mut() {
                *slot = Some((hist, tok_hist));
            }
        }
        let low_rank_head = self.hc_head.as_ref().and_then(|p| p.low_rank.as_ref());
        if let Some(lr) = low_rank_head {
            // Qwen4-Exp: the final mixer is the output norm too.
            h = lr.mix(&h, self.norm.eps)?.0;
        } else if let (Some(head), Some((n, eps, _))) = (&self.hc_head, self.config.hc) {
            // Collapse the streams with the head's sigmoid gate.
            let (b, s, d) = (h.shape()[0], h.shape()[1], h.shape()[3]);
            let flat = h.reshape(&[b, s, n * d])?;
            let normed = flat.mul(
                &flat
                    .square()?
                    .mean_axis(-1, true)?
                    .add_scalar(self.norm.eps)?
                    .rsqrt()?,
            )?;
            let base = head.base.reshape(&[1, 1, n])?.expand(&[b, s, n])?;
            let pre = normed
                .matmul(&head.w.t()?)?
                .mul_scalar(f64::from(head.scale[0]))?
                .add(&base)?
                .sigmoid()?
                .add_scalar(f64::from(eps))?;
            h = reduce_streams(&h, &pre)?;
        } else if self.config.hc.is_some() {
            // No output gate (GLM-5-Next): the streams are averaged.
            h = h.mean_axis(2, false)?;
        }
        if let Some(score) = &self.output_res_score {
            h = res_mix(&res_stack, &h, score, self.norm.eps)?;
        }
        // HRM-Text's stacks already end in the norm.
        let normed = if self.config.hrm.is_some() || low_rank_head.is_some() {
            h
        } else {
            self.norm.forward(&h)?
        };
        let mut logits = proj(&self.lm_head, &normed, self.lm_head_scale)?;
        if let Some(s) = self.config.logit_scale {
            logits = logits.mul_scalar(1.0 / f64::from(s))?;
        }
        if let Some(s) = self.config.logit_mul {
            logits = logits.mul_scalar(f64::from(s))?;
        }
        let logits = match self.config.final_softcap {
            Some(cap) => logits
                .mul_scalar(1.0 / f64::from(cap))?
                .tanh()?
                .mul_scalar(f64::from(cap))?,
            None => logits,
        };
        match self.config.suppress_logits {
            Some((start, end)) if start < logits.shape()[logits.shape().len() - 1] => {
                // Chameleon: image-token ids can never win, so text-only decoding stays text.
                let vocab = logits.shape()[logits.shape().len() - 1];
                let end = end.min(vocab);
                let rows = logits.numel() / vocab;
                let mask: Vec<bool> = (0..rows * vocab)
                    .map(|i| (start..end).contains(&(i % vocab)))
                    .collect();
                let mask = Tensor::from_slice_on(&mask, logits.shape(), &logits.device())?;
                logits.masked_fill(&mask, f64::from(f32::MIN))
            }
            _ => Ok(logits),
        }
    }

    /// Greedy (argmax) decoding: appends up to `max_new_tokens` tokens to `prompt`, stopping early
    /// after `eos`. Returns prompt + generated ids.
    ///
    /// Uses a per-layer KV cache: the prompt is prefilled once, then each step feeds one token.
    pub fn generate_greedy(
        &self,
        prompt: &[i64],
        max_new_tokens: usize,
        eos: Option<i64>,
    ) -> Result<Vec<i64>> {
        if prompt.is_empty() {
            return Err(Error::Msg(
                "generate_greedy needs a non-empty prompt".into(),
            ));
        }
        let vocab = self.config.vocab_size;
        let mut ids = prompt.to_vec();
        let mut cache = self.new_cache();
        let mut fed = 0; // tokens already in the cache
        for _ in 0..max_new_tokens {
            if ids.len() >= self.config.max_seq_len {
                break;
            }
            let new = &ids[fed..];
            let input = Tensor::from_slice(new, &[1, new.len()])?;
            let logits = self.forward_cached(&input, &mut cache)?.to_vec::<f32>()?;
            fed = ids.len();
            let last = &logits[(new.len() - 1) * vocab..new.len() * vocab];
            let next = last
                .iter()
                .enumerate()
                .fold((0usize, f32::NEG_INFINITY), |best, (i, &v)| {
                    if v > best.1 { (i, v) } else { best }
                })
                .0 as i64;
            ids.push(next);
            if Some(next) == eos {
                break;
            }
        }
        Ok(ids)
    }

    /// Builds a model from a GGUF file's metadata *and* loads its actual tensor data as weights,
    /// using the standard llama.cpp tensor naming (`token_embd.weight`, `blk.N.attn_q.weight`, ...).
    ///
    /// `from_config` alone only shapes the model from the metadata and leaves every weight at its
    /// random initialization — it never reads the file's tensor data, despite this module's doc
    /// comment claiming GGUF "quantized weights" are loaded directly. This is the constructor that
    /// actually does that; every required tensor missing or shape-mismatched is a loud error rather
    /// than a model that silently runs on noise.
    pub fn from_gguf(file: &GGUFFile) -> Result<Self> {
        let mut config = GGUFModelConfig::from_gguf(file)?;
        // Architectures not in the table differ in ways this block does not model (Phi-3 fused QKV,
        // LayerNorm families, MLA, SSMs, ...); running them would silently produce wrong logits.
        let mut spec = arch_spec(&config.architecture).ok_or_else(|| {
            Error::Msg(format!(
                "unsupported GGUF architecture '{}' (supported: {})",
                config.architecture,
                SUPPORTED_ARCHITECTURES.join(", ")
            ))
        })?;
        // Real GGUF files usually carry no `vocab_size` key: the embedding table is the source of truth.
        if let Some(info) = file.tensors.get("token_embd.weight") {
            if info.shape.len() == 2 {
                config.vocab_size = info.shape[0];
                config.d_model = info.shape[1];
            }
        }
        if spec.ssm.is_some() || spec.rwkv6 || spec.rwkv7 || spec.rwkv6qwen2 {
            // No attention: keep the (unused) head geometry and rotary tables tiny.
            (
                config.num_heads,
                config.num_kv_heads,
                config.head_dim,
                config.rope_n_rot,
            ) = (1, 1, 2, 2);
        }
        if config.head_dim == 0 {
            super::transformer::check_heads(config.d_model, config.num_heads)?;
            config.head_dim = config.d_model / config.num_heads;
        }
        if config.rope_n_rot == 0 {
            config.rope_n_rot = config.head_dim;
        }
        if spec.attn_scale_inv {
            config.attn_scale = Some(1.0 / config.head_dim as f32);
        }
        if config.architecture == "minicpm3" {
            config.logit_mul = Some(256.0 / config.d_model as f32);
        }
        if spec.rope_alpha {
            if let Some(a) = file
                .get_metadata_f32(&format!("{}.rope.scaling.alpha", config.architecture))
                .filter(|&a| a > 0.0)
            {
                let dim = config.head_dim as f32;
                config.rope_freq_base *= a.powf(dim / (dim - 2.0));
            }
        }
        if spec.mrope_switch || spec.mrope.is_some() {
            let sections: Vec<i64> = match file
                .metadata
                .get(&format!("{}.rope.dimension_sections", config.architecture))
            {
                Some(crate::serialize::gguf::GGUFValue::Array(a)) => a
                    .iter()
                    .filter_map(|v| match v {
                        crate::serialize::gguf::GGUFValue::Int32(n) => Some(i64::from(*n)),
                        crate::serialize::gguf::GGUFValue::Uint32(n) => Some(i64::from(*n)),
                        _ => None,
                    })
                    .collect(),
                _ => Vec::new(),
            };
            let s = |k: usize| sections.get(k).copied().unwrap_or(0);
            // GLM4 switches to M-RoPE only when the first two sections are set (llama.cpp `use_mrope`).
            let kind = match spec.mrope {
                Some(imrope) if s(0) > 0 || s(1) > 0 || s(2) > 0 => Some(imrope),
                Some(_) => {
                    return Err(Error::Msg(
                        "this architecture needs rope.dimension_sections".into(),
                    ));
                }
                None => (s(0) > 0 && s(1) > 0).then_some(false),
            };
            if let Some(imrope) = kind {
                // Text tokens carry positions [p, p, p, 0]: pairs on the first three channels rotate like
                // plain half-split RoPE, pairs that land on the fourth channel stay put.
                let total = s(0) + s(1) + s(2) + s(3);
                spec.rope_interleaved = false;
                config.rope_zero_pairs = (0..config.rope_n_rot / 2)
                    .map(|k| {
                        let sector = k as i64 % total;
                        if imrope {
                            !((sector % 3 == 0 && sector < 3 * s(0))
                                || (sector % 3 == 1 && sector < 3 * s(1))
                                || (sector % 3 == 2 && sector < 3 * s(2)))
                        } else {
                            sector >= s(0) + s(1) + s(2)
                        }
                    })
                    .collect();
            }
        }
        if spec.embed_scale {
            config.embedding_scale = Some((config.d_model as f32).sqrt());
        }
        // llama.cpp: the 27B Gemma 2 / Gemma 3 scale attention by 1 / sqrt(d_model / n_head), not 1 / sqrt(head_dim).
        if config.attn_scale.is_none()
            && matches!(
                (config.architecture.as_str(), config.num_layers),
                ("gemma2", 46) | ("gemma3", 62)
            )
        {
            if config.num_heads == 0 {
                return Err(Error::Msg(format!(
                    "{}.attention.head_count must be positive (corrupt GGUF metadata)",
                    config.architecture
                )));
            }
            config.attn_scale = Some(1.0 / ((config.d_model / config.num_heads) as f32).sqrt());
        }
        // Cap the position tables: they cost max_seq_len * n_rot / 2 * 8 bytes per table.
        config.max_seq_len = config.max_seq_len.min(MAX_ROPE_POSITIONS);
        let (heads, n_rot, hd) = (config.num_heads, config.rope_n_rot, config.head_dim);
        let mut model = Self::from_config(config)?;
        model.rope = build_rope(
            file,
            &model.config,
            f64::from(model.config.rope_freq_base),
            false,
            model.config.rope_n_rot,
        )?;
        if let Some(b) = model.config.rope_base_swa {
            model.rope_local = Some(build_rope(
                file,
                &model.config,
                f64::from(b),
                model.config.local_rope_unscaled,
                model
                    .config
                    .rope_n_rot_swa
                    .unwrap_or(model.config.rope_n_rot),
            )?);
        }

        model.tok_emb.weight = load_named(file, "token_embd.weight", model.tok_emb.weight.shape())?;
        let d = model.config.d_model;
        let eps = model.config.rms_eps;
        if let Some((n, _, _)) = model
            .config
            .hc
            .filter(|_| file.tensors.contains_key("output_hc_fn.weight"))
        {
            model.hc_head = Some(HcParams {
                w: load_named(
                    file,
                    "output_hc_fn.weight",
                    &[
                        n,
                        n.checked_mul(d).ok_or_else(|| {
                            Error::Msg("output_hc_fn.weight shape overflows".into())
                        })?,
                    ],
                )?,
                base: load_named(file, "output_hc_base.weight", &[n])?,
                scale: file
                    .dequantize_tensor("output_hc_scale.weight")?
                    .to_vec::<f32>()?,
                low_rank: None,
            });
        }
        if let (Some((n, _, _)), true) = (model.config.hc, spec.hc_low_rank) {
            // The final mixer doubles as the output norm.
            model.hc_head = Some(load_low_rank_hc(file, "output_hc", n, d, false)?);
        }
        if spec.ple {
            model.ple = Some(load_ple(file, &model.config)?);
        }
        if spec.g3n {
            model.g3n = Some(load_g3n_model(file, &model.config)?);
        }
        if spec.g4 {
            model.g4 = Some(load_g4_model(file, &model.config)?);
        }
        if spec.situ && file.tensors.contains_key("output_res_score.weight") {
            model.output_res_score = Some(load_named(file, "output_res_score.weight", &[d])?);
        }

        for (i, block) in model.blocks.iter_mut().enumerate() {
            if spec.rwkv6 {
                block.rwkv = Some(load_rwkv6_layer(file, i, &model.config)?);
                continue;
            }
            if spec.rwkv7 {
                block.rwkv7 = Some(load_rwkv7_layer(file, i, &model.config)?);
                continue;
            }
            if spec.rwkv6qwen2 {
                block.qrwkv = Some(load_qrwkv6_layer(file, i, &model.config)?);
                continue;
            }
            if block.mixer == Mixer::Ssm {
                load_norm(file, &format!("blk.{i}.attn_norm"), &mut block.attn_norm)?;
                block.ssm = Some(load_ssm(file, i, &model.config)?);
                continue;
            }
            if block.mixer == Mixer::SsmFfn {
                if spec.pre_norm {
                    load_norm(file, &format!("blk.{i}.attn_norm"), &mut block.attn_norm)?;
                }
                block.ssm = Some(load_ssm(file, i, &model.config)?);
            }
            if block.mixer == Mixer::AttnSsm {
                block.ssm = Some(load_ssm(file, i, &model.config)?);
            }
            if spec.dense_layers_rope
                && !file
                    .tensors
                    .contains_key(&format!("blk.{i}.ffn_gate_inp.weight"))
            {
                block.use_rope = true;
            }
            if let Some((n, hc_eps, hc_mag)) = model.config.hc {
                let gate = |kind: &str, rows: usize| -> Result<HcParams> {
                    if spec.hc_low_rank {
                        return load_low_rank_hc(file, &format!("blk.{i}.hc_{kind}"), n, d, true);
                    }
                    let scale = file
                        .dequantize_tensor(&format!("blk.{i}.hc_{kind}_scale.weight"))?
                        .to_vec::<f32>()?;
                    Ok(HcParams {
                        w: load_named(
                            file,
                            &format!("blk.{i}.hc_{kind}_fn.weight"),
                            &[rows, n * d],
                        )?,
                        base: load_named(file, &format!("blk.{i}.hc_{kind}_base.weight"), &[rows])?,
                        scale,
                        low_rank: None,
                    })
                };
                let sinkhorn = file
                    .get_metadata_u32(&format!(
                        "{}.hyper_connection.sinkhorn_iterations",
                        model.config.architecture
                    ))
                    .map(|v| v as usize)
                    .filter(|&v| v > 0); // hy_v4 writes the key as 0: plain iHC
                let rows = if sinkhorn.is_some() {
                    (2 + n) * n
                } else {
                    2 * n
                };
                block.hc = Some(HcLayer {
                    attn: gate("attn", rows)?,
                    ffn: gate("ffn", rows)?,
                    streams: n,
                    eps: hc_eps,
                    magnitude: hc_mag,
                    norm_eps: eps,
                    sinkhorn,
                });
            }
            if spec.situ {
                let bs = file
                    .get_metadata_u32(&format!(
                        "{}.attn_res.block_size",
                        model.config.architecture
                    ))
                    .unwrap_or(0) as usize;
                if bs > 0 {
                    let score = |nm: &str| load_named(file, &format!("blk.{i}.{nm}.weight"), &[d]);
                    block.attn_res = Some([score("attn_res_score")?, score("ffn_res_score")?]);
                    block.res_bank = i % bs == 0;
                }
            }
            block.moe_scale = spec.moe_out_scale;
            block.moe_router_raw = spec.moe_router_raw;
            if spec.pre_norm {
                load_norm(file, &format!("blk.{i}.attn_norm"), &mut block.attn_norm)?;
            }
            if spec.attn_norm_2_optional
                && file
                    .tensors
                    .contains_key(&format!("blk.{i}.attn_norm_2.weight"))
            {
                let mut n2 = Norm::build(
                    model.config.norm_layer,
                    model.config.norm_bias,
                    model.config.norm_scaled,
                    d,
                    eps,
                )?;
                load_norm(file, &format!("blk.{i}.attn_norm_2"), &mut n2)?;
                block.attn_norm_2 = Some(n2);
            }
            let scalar = |name: String| load_scalar(file, &name);
            if spec.dsv4 {
                let hashed = file
                    .get_metadata_u32(&format!("{}.hash_layer_count", model.config.architecture))
                    .unwrap_or(0) as usize;
                if i < hashed {
                    // The first layers route by token id: a `[vocab, experts_used]` table of expert ids.
                    let name = format!("blk.{i}.ffn_gate_tid2eid.weight");
                    let table: Vec<i32> = file
                        .dequantize_tensor(&name)?
                        .to_vec::<f32>()?
                        .iter()
                        .map(|&v| v as i32)
                        .collect();
                    let used = model.config.expert_used;
                    if used == 0
                        || table.len() != model.config.vocab_size * used
                        || table
                            .iter()
                            .any(|&e| e < 0 || e as usize >= model.config.expert_count)
                    {
                        return Err(Error::Msg(format!(
                            "{name} does not match the vocabulary and expert counts"
                        )));
                    }
                    block.moe_hash = Some((table, used));
                }
                block.dsv4 = Some(load_dsv4(file, i, &model.config)?);
            } else if spec.mla && block.mixer != Mixer::SsmFfn {
                block.mla = Some(load_mla(file, i, &model.config)?);
                let (h, vd) = (model.config.num_heads, model.config.value_length_mla);
                let hvd = h.checked_mul(vd).ok_or_else(|| {
                    Error::Msg(format!(
                        "blk.{i}.attn_output.weight shape overflows: heads={h}, value_length={vd}"
                    ))
                })?;
                block.o_proj = Linear::with_bias(hvd, d, false)?;
                block.o_proj.weight =
                    load_named(file, &format!("blk.{i}.attn_output.weight"), &[d, hvd])?;
                block.o_scale = scalar(format!("blk.{i}.attn_output.scale"))?
                    .filter(|_| !spec.mla_ignore_o_scale);
                if spec.attn_gate {
                    // One sigmoid gate per head (BailingMoE3) or per value channel (Kimi-K3); GGUF drops a size-1 dimension.
                    let w = file.dequantize_tensor(&format!("blk.{i}.attn_gate.weight"))?;
                    let rows = w.numel() / d;
                    let mut g = Linear::with_bias(d, rows, false)?;
                    g.weight = w.reshape(&[rows, d])?;
                    block.attn_gate = Some(g);
                }
            } else if !matches!(block.mixer, Mixer::FfnOnly | Mixer::SsmFfn) {
                let fused_qkv = format!("blk.{i}.attn_qkv.weight");
                let (q_rows, kv_rows) = (
                    block.q_proj.weight.shape()[0],
                    block.k_proj.weight.shape()[0],
                );
                let (q, k, v_fused) = if file.tensors.contains_key(&fused_qkv) {
                    // One [q | k | v] matrix (Phi-3 style): split the rows.
                    let w = load_named(file, &fused_qkv, &[q_rows + 2 * kv_rows, d])?;
                    (
                        w.narrow(0, 0, q_rows)?.contiguous()?,
                        w.narrow(0, q_rows, kv_rows)?.contiguous()?,
                        Some(w.narrow(0, q_rows + kv_rows, kv_rows)?.contiguous()?),
                    )
                } else {
                    let q_shape = if spec.q_gate_packed {
                        vec![2 * q_rows, d]
                    } else {
                        block.q_proj.weight.shape().to_vec()
                    };
                    let k_name = format!("blk.{i}.attn_k.weight");
                    (
                        load_named(file, &format!("blk.{i}.attn_q.weight"), &q_shape)?,
                        // Gemma 4's layers that reuse an earlier layer's K/V have no K projection at all.
                        if spec.g4 && !file.tensors.contains_key(&k_name) {
                            block.k_proj.weight.clone()
                        } else {
                            load_named(file, &k_name, block.k_proj.weight.shape())?
                        },
                        None,
                    )
                };
                // Qwen3.5: every head's rows are `[q | gate]`; split them into the query and an output gate.
                let (q, packed_gate) = if spec.q_gate_packed {
                    let per = q_rows / block.num_heads;
                    let full = q.reshape(&[block.num_heads, 2, per, d])?;
                    (
                        full.narrow(1, 0, 1)?.contiguous()?.reshape(&[q_rows, d])?,
                        Some(full.narrow(1, 1, 1)?.contiguous()?.reshape(&[q_rows, d])?),
                    )
                } else {
                    (q, None)
                };
                let mut packed_gate_bias = None;
                // Optional q/k/v biases and scalar weight scales (every graph reads these); the output
                // projection's bias and the FFN's biases / scales only where the architecture's graph does.
                for (name, lin) in [
                    ("q", &mut block.q_proj),
                    ("k", &mut block.k_proj),
                    ("v", &mut block.v_proj),
                ] {
                    if spec.q_gate_packed && name == "q" {
                        if let Some(b) =
                            load_bias(file, &format!("blk.{i}.attn_q.bias"), 2 * q_rows)?
                        {
                            let full =
                                b.reshape(&[block.num_heads, 2, q_rows / block.num_heads])?;
                            lin.bias =
                                Some(full.narrow(1, 0, 1)?.contiguous()?.reshape(&[q_rows])?);
                            packed_gate_bias =
                                Some(full.narrow(1, 1, 1)?.contiguous()?.reshape(&[q_rows])?);
                        }
                        continue;
                    }
                    lin.bias = load_bias(
                        file,
                        &format!("blk.{i}.attn_{name}.bias"),
                        lin.weight.shape()[0],
                    )?;
                }
                if let Some(b) = load_bias(
                    file,
                    &format!("blk.{i}.attn_qkv.bias"),
                    q_rows + 2 * kv_rows,
                )? {
                    block.q_proj.bias = Some(b.narrow(0, 0, q_rows)?.contiguous()?);
                    block.k_proj.bias = Some(b.narrow(0, q_rows, kv_rows)?.contiguous()?);
                    block.v_proj.bias = Some(b.narrow(0, q_rows + kv_rows, kv_rows)?.contiguous()?);
                }
                if spec.o_bias {
                    block.o_proj.bias = load_bias(file, &format!("blk.{i}.attn_output.bias"), d)?;
                }
                if spec.sinks
                    || (spec.sinks_optional
                        && file
                            .tensors
                            .contains_key(&format!("blk.{i}.attn_sinks.weight")))
                {
                    block.sinks = Some(load_named(
                        file,
                        &format!("blk.{i}.attn_sinks.weight"),
                        &[heads],
                    )?);
                }
                if spec.block_indexer
                    && file
                        .tensors
                        .contains_key(&format!("blk.{i}.indexer.q_proj.weight"))
                {
                    let ikey = |k: &str| {
                        file.get_metadata_u32(&format!("{}.attention.indexer.{k}", model.config.architecture))
                            .map(|v| v as usize)
                            .ok_or_else(|| {
                                Error::Msg(format!(
                                    "{}.attention.indexer.{k} is missing",
                                    model.config.architecture
                                ))
                            })
                    };
                    let (ih, idim) = (ikey("head_count")?, ikey("key_length")?);
                    let (mut q_norm, mut k_norm) =
                        (Norm::rms(idim, model.config.rms_eps)?, Norm::rms(idim, model.config.rms_eps)?);
                    load_norm(file, &format!("blk.{i}.indexer.q_norm"), &mut q_norm)?;
                    load_norm(file, &format!("blk.{i}.indexer.k_norm"), &mut k_norm)?;
                    block.block_indexer = Some(BlockIndexer {
                        q_proj: load_named(
                            file,
                            &format!("blk.{i}.indexer.q_proj.weight"),
                            &[ih * idim, d],
                        )?,
                        k_proj: load_named(
                            file,
                            &format!("blk.{i}.indexer.k_proj.weight"),
                            &[idim, d],
                        )?,
                        q_norm,
                        k_norm,
                        heads: ih,
                        dim: idim,
                        block: ikey("block_size")?.max(1),
                        top_blocks: ikey("top_k")?,
                        local: ikey("local_blocks").unwrap_or(0),
                        pool: ikey("kpool").unwrap_or(0),
                    });
                }
                block.q_scale = scalar(format!("blk.{i}.attn_q.scale"))?;
                block.k_scale = scalar(format!("blk.{i}.attn_k.scale"))?;
                block.v_scale = scalar(format!("blk.{i}.attn_v.scale"))?;
                if let Some(s) =
                    scalar(format!("blk.{i}.attn_qkv.scale"))?.filter(|_| !spec.ignore_qkv_scale)
                {
                    (block.q_scale, block.k_scale, block.v_scale) = (Some(s), Some(s), Some(s));
                }
                if let Some(gw) = packed_gate {
                    let mut g = Linear::with_bias(d, q_rows, false)?;
                    g.weight = gw;
                    g.bias = packed_gate_bias;
                    block.attn_gate = Some(g);
                    block.attn_gate_scale = block.q_scale;
                }
                if spec.ignore_qkv_scale {
                    (block.q_scale, block.k_scale, block.v_scale) = (None, None, None);
                    (block.q_proj.bias, block.k_proj.bias, block.v_proj.bias) = (None, None, None);
                }
                if spec.ignore_qkv_bias {
                    (block.q_proj.bias, block.k_proj.bias, block.v_proj.bias) = (None, None, None);
                    if let Some(g) = block.attn_gate.as_mut() {
                        g.bias = None;
                    }
                }
                block.o_scale = scalar(format!("blk.{i}.attn_output.scale"))?;
                if spec.value_scale {
                    if let Some(vs) = file
                        .get_metadata_f32(&format!(
                            "{}.attention.value_scale",
                            model.config.architecture
                        ))
                        .filter(|&v| v != 0.0 && v != 1.0)
                    {
                        block.o_scale = Some(block.o_scale.unwrap_or(1.0) * vs);
                    }
                }
                if spec.attn_gate
                    && (!spec.attn_gate_optional
                        || file
                            .tensors
                            .contains_key(&format!("blk.{i}.attn_gate.weight")))
                {
                    let name = format!("blk.{i}.attn_gate.weight");
                    let w = file
                        .tensors
                        .get(&name)
                        .ok_or_else(|| Error::Msg(format!("missing {name}")))?
                        .shape[0];
                    let mut g = Linear::with_bias(d, w, false)?;
                    g.weight = load_named(file, &name, &[w, d])?;
                    if spec.attn_gate_head_bias {
                        let h = block.num_heads;
                        let b = load_named(file, &format!("blk.{i}.attn_gate.bias"), &[h])?;
                        g.bias = Some(b.reshape(&[h, 1])?.expand(&[h, w / h])?.reshape(&[w])?);
                    }
                    block.attn_gate = Some(g);
                    block.attn_gate_softplus = spec.attn_gate_softplus;
                    if spec.attn_gate_log2 {
                        block.attn_gate_log2 = true;
                        block.attn_gate_scale =
                            load_scalar(file, &format!("blk.{i}.attn_gate.scale"))?;
                    }
                }
                let (heads, kv_heads) = (block.num_heads, block.num_kv_heads);
                (block.q_proj.weight, block.k_proj.weight) = if spec.rope_interleaved {
                    // Rows were laid out for interleaved RoPE, so a bias on those rows moves with them.
                    for (lin, n_heads) in
                        [(&mut block.q_proj, heads), (&mut block.k_proj, kv_heads)]
                    {
                        if let Some(b) = lin.bias.take() {
                            let n = b.numel();
                            lin.bias = Some(
                                interleaved_to_half_split_partial(
                                    &b.reshape(&[n, 1])?,
                                    n_heads,
                                    n_rot,
                                )?
                                .reshape(&[n])?,
                            );
                        }
                    }
                    (
                        interleaved_to_half_split_partial(&q, heads, n_rot)?,
                        interleaved_to_half_split_partial(&k, kv_heads, n_rot)?,
                    )
                } else {
                    (q, k)
                };
                let value_exps = format!("blk.{i}.attn_v_exps.weight");
                if let (Some((n_values, used)), true) = (
                    model.config.value_experts,
                    file.tensors.contains_key(&value_exps),
                ) {
                    let cfg = &model.config;
                    block.value_moe = Some(ValueMoe {
                        gate: load_named(
                            file,
                            &format!("blk.{i}.attn_v_gate.weight"),
                            &[n_values, d],
                        )?,
                        bias: load_expert_scales(
                            file,
                            &format!("blk.{i}.attn_v_gate.bias"),
                            n_values,
                        )?,
                        exps: load_named(
                            file,
                            &value_exps,
                            &[n_values, block.num_kv_heads * hd, d],
                        )?,
                        used,
                        gating: cfg.moe_gating,
                        norm: cfg.moe_norm_topk,
                        scale: cfg.moe_weights_scale,
                    });
                } else {
                    let v_name = format!("blk.{i}.attn_v.weight");
                    if spec.g4 && v_fused.is_none() && !file.tensors.contains_key(&v_name) {
                        block.v_from_k = true; // no V projection: the values are the keys
                    }
                    block.v_proj.weight = match v_fused {
                        Some(v) => v,
                        None if block.v_from_k => block.v_proj.weight.clone(),
                        None => load_named(
                            file,
                            &format!("blk.{i}.attn_v.weight"),
                            block.v_proj.weight.shape(),
                        )?,
                    };
                }
                block.o_proj.weight = load_named(
                    file,
                    &format!("blk.{i}.attn_output.weight"),
                    block.o_proj.weight.shape(),
                )?;
                if spec.qk_norm != QkNorm::None
                    && !spec.qk_head_weights
                    && !(spec.kq_norm_weightless && model.config.kq_norm_weightless)
                {
                    for (nm, norm) in [("q", &mut block.q_norm), ("k", &mut block.k_norm)] {
                        let name = format!("blk.{i}.attn_{nm}_norm.weight");
                        if spec.qk_norm_optional && !file.tensors.contains_key(&name) {
                            *norm = None;
                            continue;
                        }
                        let n = norm.as_mut().expect("from_config creates the norms");
                        load_norm(file, &format!("blk.{i}.attn_{nm}_norm"), n)?;
                    }
                }
                if spec.qk_head_weights {
                    for (nm, norm, n_heads) in [
                        ("q", &mut block.q_norm, block.num_heads),
                        ("k", &mut block.k_norm, block.num_kv_heads),
                    ] {
                        let name = format!("blk.{i}.attn_{nm}_norm.weight");
                        if file.tensors.contains_key(&name) {
                            let w = file.dequantize_tensor(&name)?;
                            if w.numel() != n_heads * hd {
                                return Err(Error::Msg(format!(
                                    "{name} has {} values, expected {}",
                                    w.numel(),
                                    n_heads * hd
                                )));
                            }
                            *norm = Some(Norm {
                                weight: w.reshape(&[n_heads, hd])?,
                                bias: None,
                                eps,
                                layer: false,
                                scaled: true,
                                groups: 1,
                            });
                        } else {
                            *norm = None;
                        }
                    }
                }
                if spec.spark_blocks {
                    let vec = |nm: &str| load_named(file, &format!("blk.{i}.{nm}.weight"), &[d]);
                    block.resid_ab = Some([
                        vec("attn_resid_alpha")?,
                        vec("attn_resid_beta")?,
                        vec("ffn_resid_alpha")?,
                        vec("ffn_resid_beta")?,
                    ]);
                    block.ffn_post_scale = Some(vec("post_ffw_norm")?);
                }
                if spec.q_head_gain {
                    let h = block.num_heads;
                    let g = load_named(file, &format!("blk.{i}.attn_q_norm.weight"), &[h, 1])?;
                    block.q_gain = Some(g.reshape(&[1, h, 1, 1])?);
                }
                if spec.embed_skip {
                    block.skip_scale =
                        load_scalar(file, &format!("blk.{i}.layer_output_scale.weight"))?;
                }
            }
            if let Some(g) = model.g3n.as_ref() {
                block.g3n = Some(load_g3n_layer(
                    file,
                    i,
                    &model.config,
                    g,
                    block.sliding_window.is_some(),
                )?);
                block.v_norm = true;
            }
            if let Some(g) = model.g4.as_ref() {
                block.g4 = Some(load_g4_layer(
                    file,
                    i,
                    &model.config,
                    g,
                    block.sliding_window.is_some(),
                )?);
                block.v_norm = true;
            }
            if spec.sub_norms {
                let width = block.num_heads * block.head_dim;
                let mut an = Norm::rms(width, model.config.rms_eps)?;
                load_norm(file, &format!("blk.{i}.attn_sub_norm"), &mut an)?;
                block.attn_sub_norm = Some(an);
                let mut fnorm = Norm::rms(block.up_proj.weight.shape()[0], model.config.rms_eps)?;
                load_norm(file, &format!("blk.{i}.ffn_sub_norm"), &mut fnorm)?;
                block.ffn_sub_norm = Some(fnorm);
            }
            if model.config.post_norms {
                for (nm, norm) in [
                    (spec.post_norm_names.0, &mut block.post_attn_norm),
                    (spec.post_norm_names.1, &mut block.post_ffn_norm),
                ] {
                    let n = norm.as_mut().expect("from_config creates the norms");
                    // Some converters write the bare tensor name (`post_attention_norm`) without `.weight`.
                    let base = format!("blk.{i}.{nm}_norm");
                    let name = if file.tensors.contains_key(&format!("{base}.weight")) {
                        format!("{base}.weight")
                    } else {
                        base
                    };
                    n.weight = load_named(file, &name, n.weight.shape())?;
                }
            }
            if spec.pre_norm && matches!(block.mixer, Mixer::Full | Mixer::SsmFfn | Mixer::AttnSsm)
            {
                if spec.parallel_without_ffn_norm
                    && !file
                        .tensors
                        .contains_key(&format!("blk.{i}.ffn_norm.weight"))
                {
                    block.parallel = true; // StableLM's parallel-residual variant has no ffn_norm
                } else if !block.parallel || block.parallel_own_ffn_norm {
                    load_norm(
                        file,
                        &format!("blk.{i}.{}", spec.ffn_norm_name),
                        &mut block.ffn_norm,
                    )?;
                }
            }
            if spec.qk_layernorm_optional {
                // Per-head LayerNorm weights `[heads, head_dim]`; rows follow the rotary layout of the projection.
                for (nm, norm, n_heads) in [
                    ("q", &mut block.q_norm, block.num_heads),
                    ("k", &mut block.k_norm, block.num_kv_heads),
                ] {
                    let name = format!("blk.{i}.attn_{nm}_norm.weight");
                    if file.tensors.contains_key(&name) {
                        let mut w = file.dequantize_tensor(&name)?;
                        if w.numel() != n_heads * hd {
                            return Err(Error::Msg(format!(
                                "{name} has {} values, expected {}",
                                w.numel(),
                                n_heads * hd
                            )));
                        }
                        if spec.rope_interleaved {
                            w = interleaved_to_half_split_partial(
                                &w.reshape(&[n_heads * hd, 1])?,
                                n_heads,
                                n_rot,
                            )?;
                        }
                        // Chameleon: torch's LayerNorm epsilon and an optional bias laid out like the weight.
                        let bias_name = format!("blk.{i}.attn_{nm}_norm.bias");
                        let bias = if spec.qk_layer_norm && file.tensors.contains_key(&bias_name) {
                            let mut b = file.dequantize_tensor(&bias_name)?;
                            if spec.rope_interleaved {
                                b = interleaved_to_half_split_partial(
                                    &b.reshape(&[n_heads * hd, 1])?,
                                    n_heads,
                                    n_rot,
                                )?;
                            }
                            Some(b.reshape(&[n_heads, hd])?)
                        } else {
                            None
                        };
                        *norm = Some(Norm {
                            weight: w.reshape(&[n_heads, hd])?,
                            bias,
                            eps: if spec.qk_layer_norm { 1e-5 } else { eps },
                            layer: true,
                            scaled: true,
                            groups: 1,
                        });
                    }
                }
            }
            // A layer is sparse iff it has a router; MoE models may keep some leading layers dense.
            let sparse = model.config.expert_count > 0
                && file
                    .tensors
                    .contains_key(&format!("blk.{i}.ffn_gate_inp.weight"));
            if sparse && block.mixer != Mixer::AttnOnly {
                block.moe = Some(load_moe(file, i, &model.config)?);
                block.dense_with_moe = spec.dense_with_moe;
                if spec.g4 {
                    let norm = |nm: &str| -> Result<Norm> {
                        let mut n = Norm::rms(d, eps)?;
                        load_norm(file, &format!("blk.{i}.{nm}"), &mut n)?;
                        Ok(n)
                    };
                    block.g4_moe = Some(G4Moe {
                        pre_norm_2: norm("pre_ffw_norm_2")?,
                        post_norm_1: norm("post_ffw_norm_1")?,
                        post_norm_2: norm("post_ffw_norm_2")?,
                        router_scale: load_named(file, &format!("blk.{i}.ffn_gate_inp.scale"), &[d])?,
                    });
                }
                if spec.moe_norm_exps {
                    let mut n = Norm::build(
                        model.config.norm_layer,
                        model.config.norm_bias,
                        model.config.norm_scaled,
                        d,
                        eps,
                    )?;
                    load_norm(file, &format!("blk.{i}.ffn_norm_exps"), &mut n)?;
                    block.moe_norm = Some(n);
                }
            }
            if block.mixer != Mixer::AttnOnly
                && (!sparse
                    || (spec.dense_with_moe
                        && file
                            .tensors
                            .contains_key(&format!("blk.{i}.ffn_down.weight"))))
            {
                let gate_name = format!("blk.{i}.ffn_gate.weight");
                let ff = file
                    .tensors
                    .get(&format!("blk.{i}.ffn_down.weight"))
                    .and_then(|t| t.shape.get(1).copied())
                    .ok_or_else(|| Error::Msg(format!("missing blk.{i}.ffn_down.weight")))?;
                block.gate_proj = Linear::with_bias(d, ff, false)?;
                block.up_proj = Linear::with_bias(d, ff, false)?;
                block.down_proj = Linear::with_bias(ff, d, false)?;
                if file.tensors.contains_key(&gate_name) {
                    block.gate_proj.weight = load_named(file, &gate_name, &[ff, d])?;
                    block.up_proj.weight =
                        load_named(file, &format!("blk.{i}.ffn_up.weight"), &[ff, d])?;
                } else if !spec.ffn_gated {
                    block.up_proj.weight =
                        load_named(file, &format!("blk.{i}.ffn_up.weight"), &[ff, d])?;
                } else {
                    // Phi-3: one [gate | up] matrix, SwiGLU with the first half activated.
                    let w = load_named(file, &format!("blk.{i}.ffn_up.weight"), &[2 * ff, d])?;
                    block.gate_proj.weight = w.narrow(0, 0, ff)?.contiguous()?;
                    block.up_proj.weight = w.narrow(0, ff, ff)?.contiguous()?;
                }
                block.down_proj.weight =
                    load_named(file, &format!("blk.{i}.ffn_down.weight"), &[d, ff])?;
                if spec.ffn_bias {
                    for (nm, lin) in [
                        ("gate", &mut block.gate_proj),
                        ("up", &mut block.up_proj),
                        ("down", &mut block.down_proj),
                    ] {
                        lin.bias = load_bias(
                            file,
                            &format!("blk.{i}.ffn_{nm}.bias"),
                            lin.weight.shape()[0],
                        )?;
                    }
                }
                if spec.ffn_act_scales
                    && file
                        .tensors
                        .contains_key(&format!("blk.{i}.ffn.act.scales"))
                {
                    let s = file
                        .dequantize_tensor(&format!("blk.{i}.ffn.act.scales"))?
                        .to_vec::<f32>()?;
                    if s.len() != ff {
                        return Err(Error::Msg(format!(
                            "blk.{i}.ffn.act.scales has {} entries, expected {ff}",
                            s.len()
                        )));
                    }
                    let recip: Vec<f32> = s.iter().map(|v| 1.0 / v).collect();
                    block.ffn_act_recip = Some(Tensor::from_slice(&recip, &[ff])?);
                }
                if spec.ffn_scale {
                    block.gate_scale = scalar(format!("blk.{i}.ffn_gate.scale"))?;
                    block.up_scale = scalar(format!("blk.{i}.ffn_up.scale"))?;
                }
                if spec.ffn_scale || spec.ffn_down_scale_only {
                    block.down_scale = scalar(format!("blk.{i}.ffn_down.scale"))?;
                }
            }
        }

        if !spec.hc_low_rank {
            load_norm(file, spec.output_norm_name, &mut model.norm)?;
        }
        // Llama-family GGUF files commonly tie the output projection to the input embedding and
        // omit "output.weight" entirely; fall back to the (already-loaded) embedding in that case.
        model.lm_head.weight = if file.tensors.contains_key("output.weight") {
            load_named(file, "output.weight", model.lm_head.weight.shape())?
        } else {
            model.tok_emb.weight.clone()
        };
        model.lm_head_scale = load_scalar(file, "output.scale")?;
        if spec.abs_pos
            && (!spec.abs_pos_optional || file.tensors.contains_key("position_embd.weight"))
        {
            model.pos_emb = Some(file.dequantize_tensor("position_embd.weight")?);
        }
        if let Some((lps, hc, lc)) = model.config.hrm {
            model.hrm_z_l = Some(load_named(file, "hrm.z_l_init", &[model.config.d_model])?);
            for s in 0..lps * hc * (lc + 1) {
                let mut sc = [None; 4];
                for (slot, nm) in sc.iter_mut().zip(["q", "k", "v", "output"]) {
                    let nm = if nm == "output" {
                        "attn_output".to_string()
                    } else {
                        format!("attn_{nm}")
                    };
                    *slot = load_scalar(file, &format!("blk.{s}.{nm}.scale"))?;
                }
                model.hrm_slot_scales.push(sc);
            }
        }
        if spec.embed_rms {
            model.emb_norm = Some(Norm::build(
                false,
                false,
                false,
                model.config.d_model,
                model.config.rms_eps,
            )?);
        }
        if spec.embed_norm {
            let mut n = Norm::build(
                model.config.norm_layer,
                model.config.norm_bias,
                model.config.norm_scaled,
                model.config.d_model,
                model.config.rms_eps,
            )?;
            load_norm(file, "token_embd_norm", &mut n)?;
            model.emb_norm = Some(n);
        }
        if spec.out_bias {
            model.lm_head.bias = load_bias(file, "output.bias", model.config.vocab_size)?;
        }

        Ok(model)
    }
}

/// Reorders rows `[start, start + len)` of every head from interleaved (adjacent-pair) to half-split
/// RoPE layout; the other rows keep their place.
fn interleaved_rows(
    w: &Tensor,
    heads: usize,
    hd: usize,
    start: usize,
    len: usize,
) -> Result<Tensor> {
    let cols = w.shape()[1];
    let per = w.reshape(&[heads, hd, cols])?;
    let mut parts = Vec::new();
    if start > 0 {
        parts.push(per.narrow(1, 0, start)?.contiguous()?);
    }
    parts.push(
        per.narrow(1, start, len)?
            .contiguous()?
            .reshape(&[heads, len / 2, 2, cols])?
            .transpose(1, 2)?
            .contiguous()?
            .reshape(&[heads, len, cols])?,
    );
    if hd > start + len {
        parts.push(per.narrow(1, start + len, hd - start - len)?.contiguous()?);
    }
    Tensor::cat(&parts, 1)?.reshape(&[heads * hd, cols])
}

/// Loads layer `i`'s Mamba / Mamba-2 tensors (llama.cpp's `ssm_*` names; `ssm_a` / `ssm_d` have no suffix).
fn load_ssm(file: &GGUFFile, i: usize, cfg: &GGUFModelConfig) -> Result<SsmBlock> {
    let mut c = cfg
        .ssm
        .clone()
        .ok_or_else(|| Error::Msg("not a state-space model".into()))?;
    let d = cfg.d_model;
    if c.kind == SsmKind::GatedDelta {
        let key_dim = c
            .n_group
            .checked_mul(c.d_state)
            .ok_or_else(|| Error::Msg("gdn key_dim overflows".into()))?;
        let (hv, ch) = (
            c.dt_rank,
            c.d_inner
                .checked_add(
                    key_dim
                        .checked_mul(2)
                        .ok_or_else(|| Error::Msg("gdn channels overflow".into()))?,
                )
                .ok_or_else(|| Error::Msg("gdn channels overflow".into()))?,
        );
        let vec = |name: &str, len: usize| -> Result<Vec<f32>> {
            let v = file
                .dequantize_tensor(&format!("blk.{i}.{name}"))?
                .to_vec::<f32>()?;
            if v.len() != len {
                return Err(Error::Msg(format!(
                    "blk.{i}.{name} has {} values, expected {len}",
                    v.len()
                )));
            }
            Ok(v)
        };
        let mat = |name: &str, shape: &[usize]| {
            load_named(file, &format!("blk.{i}.{name}.weight"), shape)
        };
        let sc = |name: &str| load_scalar(file, &format!("blk.{i}.{name}.scale"));
        let has = |n: &str| file.tensors.contains_key(&format!("blk.{i}.{n}.weight"));
        let use_scales = arch_spec(&cfg.architecture).is_some_and(|s| s.gdn_scales);
        let sc = |name: &str| if use_scales { sc(name) } else { Ok(None) };
        let fused = has("ssm_in") && !has("attn_qkv");
        // Qwen3-Next stores one `[q | k | v | z]` block per key-head group (and `[beta | alpha]` likewise).
        let (sk, sv, hk) = (c.d_state, c.d_inner / hv, c.n_group);
        let r = hv / hk;
        let (in_proj, z_proj) = if fused {
            fn oflo() -> Error {
                Error::Msg("gdn fused block overflows".into())
            }
            let (rsv, sk2) = (
                r.checked_mul(sv).ok_or_else(oflo)?,
                sk.checked_mul(2).ok_or_else(oflo)?,
            );
            let block = sk2
                .checked_add(rsv.checked_mul(2).ok_or_else(oflo)?)
                .ok_or_else(oflo)?;
            let rows = hk.checked_mul(block).ok_or_else(oflo)?;
            let w = mat("ssm_in", &[rows, d])?.reshape(&[hk, block, d])?;
            let part = |start: usize, len: usize| -> Result<Tensor> {
                let n = hk.checked_mul(len).ok_or_else(oflo)?;
                w.narrow(1, start, len)?.contiguous()?.reshape(&[n, d])
            };
            (
                Tensor::cat(&[part(0, sk)?, part(sk, sk)?, part(sk2, rsv)?], 0)?,
                part(sk2.checked_add(rsv).ok_or_else(oflo)?, rsv)?,
            )
        } else {
            (
                mat("attn_qkv", &[ch, d])?,
                mat("attn_gate", &[c.d_inner, d])?,
            )
        };
        let (beta_proj, alpha_proj) = if has("ssm_ba") && !has("ssm_beta") {
            let ba = mat("ssm_ba", &[2 * hv, d])?.reshape(&[hk, 2 * r, d])?;
            let half = |start: usize| -> Result<Tensor> {
                ba.narrow(1, start, r)?.contiguous()?.reshape(&[hv, d])
            };
            (half(0)?, half(r)?)
        } else {
            (mat("ssm_beta", &[hv, d])?, mat("ssm_alpha", &[hv, d])?)
        };
        return Ok(SsmBlock {
            in_proj,
            conv_w: vec(
                "ssm_conv1d.weight",
                ch.checked_mul(c.d_conv)
                    .ok_or_else(|| Error::Msg("gdn conv width overflows".into()))?,
            )?,
            conv_b: vec![0.0; ch],
            x_proj: None,
            dt_proj: None,
            dt_bias: vec("ssm_dt.bias", hv)?,
            a: vec("ssm_a", hv)?,
            d: Vec::new(),
            norm_w: Some(vec("ssm_norm.weight", c.d_inner / hv)?),
            dt_b_c_norm: None,
            out_proj: mat("ssm_out", &[d, c.d_inner])?,
            eps: cfg.rms_eps as f32,
            in_scale: sc("attn_qkv")?,
            // qwen3next's graph runs `ssm_out` as a plain matmul even when the file carries `ssm_out.scale`.
            out_scale: if cfg.architecture == "qwen3next" { None } else { sc("ssm_out")? },
            gdn: Some(GdnParts {
                z_proj,
                beta_proj,
                alpha_proj,
                z_scale: sc("attn_gate")?,
                beta_scale: sc("ssm_beta")?,
                alpha_scale: sc("ssm_alpha")?,
                sigmoid_gate: arch_spec(&cfg.architecture).is_some_and(|s| s.hc_low_rank),
            }),
            kda: None,
            cfg: c,
        });
    }
    if c.kind == SsmKind::Lightning {
        let (nh, di) = (c.dt_rank, c.d_state * c.dt_rank);
        // ALiBi-style head slopes `start^(h + 1)`, scaled down towards the last layer (llama.cpp's `slope_scale`).
        let start = 2f32.powf(-(2f32.powf(-((nh as f32).log2() - 3.0))));
        let layer_scale = 1.0 - i as f32 / (cfg.num_layers.saturating_sub(1).max(1)) as f32 + 1e-5;
        let lambda: Vec<f32> = (0..nh)
            .map(|h| start.powi(h as i32 + 1) * layer_scale)
            .collect();
        let norm = file
            .dequantize_tensor(&format!("blk.{i}.attn_norm_2.weight"))?
            .to_vec::<f32>()?;
        if norm.len() != di {
            return Err(Error::Msg(format!(
                "blk.{i}.attn_norm_2.weight has {} values, expected {di}",
                norm.len()
            )));
        }
        // llama.cpp's lightning layers use plain matmuls: no per-tensor scales.
        return Ok(SsmBlock {
            in_proj: load_named(file, &format!("blk.{i}.attn_qkv.weight"), &[3 * di, d])?,
            conv_w: Vec::new(),
            conv_b: Vec::new(),
            x_proj: Some(load_named(
                file,
                &format!("blk.{i}.attn_gate.weight"),
                &[di, d],
            )?),
            dt_proj: None,
            dt_bias: Vec::new(),
            a: lambda,
            d: Vec::new(),
            norm_w: Some(norm),
            dt_b_c_norm: None,
            out_proj: load_named(file, &format!("blk.{i}.attn_output.weight"), &[d, di])?,
            eps: cfg.rms_eps as f32,
            in_scale: None,
            out_scale: None,
            gdn: None,
            kda: None,
            cfg: c,
        });
    }
    if c.kind == SsmKind::Kda {
        let (nh, hd, di) = (c.dt_rank, c.d_state, c.d_inner);
        let vec = |name: &str, len: usize| -> Result<Vec<f32>> {
            let v = file
                .dequantize_tensor(&format!("blk.{i}.{name}"))?
                .to_vec::<f32>()?;
            if v.len() != len {
                return Err(Error::Msg(format!(
                    "blk.{i}.{name} has {} values, expected {len}",
                    v.len()
                )));
            }
            Ok(v)
        };
        let mat = |name: &str, shape: &[usize]| {
            load_named(file, &format!("blk.{i}.{name}.weight"), shape)
        };
        let has = |n: &str| file.tensors.contains_key(&format!("blk.{i}.{n}.weight"));
        let (low_rank, g_low) = (has("ssm_f_b"), has("ssm_g_b"));
        // `kda.gate_lower_bound` is -inf (or absent) unless the architecture swaps the decay gate for a bounded sigmoid.
        let lower_bound = file
            .get_metadata_f32(&format!("{}.kda.gate_lower_bound", cfg.architecture))
            .filter(|v| v.is_finite());
        // llama.cpp passes no per-tensor scales to any of these projections, and ignores the q / k / v biases.
        let di3 = di
            .checked_mul(3)
            .ok_or_else(|| Error::Msg("kda qkv width overflows".into()))?;
        let in_proj = if has("attn_qkv") {
            mat("attn_qkv", &[di3, d])?
        } else {
            Tensor::cat(
                &[
                    mat("attn_q", &[di, d])?,
                    mat("attn_k", &[di, d])?,
                    mat("attn_v", &[di, d])?,
                ],
                0,
            )?
        };
        let conv_len = di3
            .checked_mul(c.d_conv)
            .ok_or_else(|| Error::Msg("kda conv width overflows".into()))?;
        let mut conv_w = Vec::with_capacity(conv_len);
        for nm in ["q", "k", "v"] {
            conv_w.extend(vec(
                &format!("ssm_conv1d_{nm}.weight"),
                di.checked_mul(c.d_conv)
                    .ok_or_else(|| Error::Msg("kda conv width overflows".into()))?,
            )?);
        }
        return Ok(SsmBlock {
            in_proj,
            conv_w,
            conv_b: vec![0.0; 3 * di],
            x_proj: None,
            dt_proj: None,
            dt_bias: vec("ssm_dt.bias", di)?,
            a: vec("ssm_a", nh)?,
            d: Vec::new(),
            norm_w: Some(vec("ssm_norm.weight", hd)?),
            dt_b_c_norm: None,
            out_proj: mat("attn_output", &[d, di])?,
            eps: cfg.rms_eps as f32,
            in_scale: None,
            out_scale: None,
            gdn: None,
            kda: Some(KdaParts {
                // Kimi-Linear: low-rank f / g projections. BailingMoE3: one full-width projection each and a bounded gate.
                f_a: mat("ssm_f_a", &[if low_rank { hd } else { di }, d])?,
                f_b: if low_rank {
                    Some(mat("ssm_f_b", &[di, hd])?)
                } else {
                    None
                },
                lower_bound,
                // Kimi-K3 bounds a low-rank gate with the opposite sign of BailingMoE3's full-width one.
                bound_negated: low_rank && lower_bound.is_some(),
                l2_eps: arch_spec(&cfg.architecture).and_then(|s| s.kda_l2_eps),
                // A single head is stored as a vector (GGUF drops the size-1 dimension).
                beta_proj: if nh == 1 {
                    file.dequantize_tensor(&format!("blk.{i}.ssm_beta.weight"))?
                        .reshape(&[1, d])?
                } else {
                    mat("ssm_beta", &[nh, d])?
                },
                g_a: if g_low {
                    mat("ssm_g_a", &[hd, d])?
                } else {
                    mat(if has("ssm_g") { "ssm_g" } else { "ssm_g_a" }, &[di, d])?
                },
                g_b: if g_low {
                    Some(mat("ssm_g_b", &[di, hd])?)
                } else {
                    None
                },
            }),
            cfg: c,
        });
    }
    if c.kind == SsmKind::ShortConv {
        c.d_inner = d;
        let vec = |name: &str, len: usize| -> Result<Vec<f32>> {
            let v = file
                .dequantize_tensor(&format!("blk.{i}.{name}"))?
                .to_vec::<f32>()?;
            if v.len() != len {
                return Err(Error::Msg(format!(
                    "blk.{i}.{name} has {} values, expected {len}",
                    v.len()
                )));
            }
            Ok(v)
        };
        return Ok(SsmBlock {
            in_proj: load_named(
                file,
                &format!("blk.{i}.shortconv.in_proj.weight"),
                &[3 * d, d],
            )?,
            conv_w: vec("shortconv.conv.weight", d * c.d_conv)?,
            conv_b: vec![0.0; d],
            x_proj: None,
            dt_proj: None,
            dt_bias: Vec::new(),
            a: Vec::new(),
            d: Vec::new(),
            norm_w: None,
            dt_b_c_norm: None,
            out_proj: load_named(file, &format!("blk.{i}.shortconv.out_proj.weight"), &[d, d])?,
            eps: cfg.rms_eps as f32,
            in_scale: None,
            out_scale: None,
            gdn: None,
            kda: None,
            cfg: c,
        });
    }
    if c.kind == SsmKind::Plamo2 {
        let (di, ds, nh) = (c.d_inner, c.d_state, c.dt_rank);
        let dt_dim = (d / 16).max(64);
        // PLaMo-2 writes some of these bare (`ssm_dt_norm`) and some with `.weight`.
        let vec = |name: &str, len: usize| -> Result<Vec<f32>> {
            let bare = format!("blk.{i}.{name}");
            let full = if file.tensors.contains_key(&bare) {
                bare
            } else {
                format!("blk.{i}.{name}.weight")
            };
            let v = file.dequantize_tensor(&full)?.to_vec::<f32>()?;
            if v.len() != len {
                return Err(Error::Msg(format!(
                    "{full} has {} values, expected {len}",
                    v.len()
                )));
            }
            Ok(v)
        };
        let mat = |name: &str, shape: &[usize]| {
            load_named(file, &format!("blk.{i}.{name}.weight"), shape)
        };
        // llama.cpp's PLaMo-2 graph passes no per-tensor scales to the in / out projections.
        return Ok(SsmBlock {
            in_proj: mat("ssm_in", &[2 * di, d])?,
            conv_w: vec("ssm_conv1d.weight", di * c.d_conv)?,
            conv_b: vec![0.0; di],
            x_proj: Some(mat("ssm_x", &[dt_dim + 2 * ds, di])?),
            dt_proj: Some(mat("ssm_dt", &[nh, dt_dim])?),
            dt_bias: vec("ssm_dt.bias", nh)?,
            a: vec("ssm_a", nh)?,
            d: vec("ssm_d", nh)?,
            norm_w: None,
            dt_b_c_norm: Some([
                vec("ssm_dt_norm", dt_dim)?,
                vec("ssm_b_norm", ds)?,
                vec("ssm_c_norm", ds)?,
            ]),
            out_proj: mat("ssm_out", &[d, di])?,
            eps: cfg.rms_eps as f32,
            in_scale: None,
            out_scale: None,
            gdn: None,
            kda: None,
            cfg: c,
        });
    }
    let (mamba, di, ds) = (c.kind == SsmKind::Mamba, c.d_inner, c.d_state);
    fn oflo() -> Error {
        Error::Msg("ssm loader widths overflow".into())
    }
    let ch = if mamba {
        di
    } else {
        di.checked_add(
            c.n_group
                .checked_mul(ds)
                .ok_or_else(oflo)?
                .checked_mul(2)
                .ok_or_else(oflo)?,
        )
        .ok_or_else(oflo)?
    };
    let in_dim = if mamba {
        di.checked_mul(2).ok_or_else(oflo)?
    } else {
        di.checked_add(ch)
            .and_then(|v| v.checked_add(c.dt_rank))
            .ok_or_else(oflo)?
    };
    let vec = |name: &str, len: usize| -> Result<Vec<f32>> {
        let v = file
            .dequantize_tensor(&format!("blk.{i}.{name}"))?
            .to_vec::<f32>()?;
        if v.len() != len {
            return Err(Error::Msg(format!(
                "blk.{i}.{name} has {} values, expected {len}",
                v.len()
            )));
        }
        Ok(v)
    };
    let mat =
        |name: &str, shape: &[usize]| load_named(file, &format!("blk.{i}.{name}.weight"), shape);
    Ok(SsmBlock {
        in_proj: mat("ssm_in", &[in_dim, d])?,
        conv_w: vec(
            "ssm_conv1d.weight",
            ch.checked_mul(c.d_conv).ok_or_else(oflo)?,
        )?,
        conv_b: if file
            .tensors
            .contains_key(&format!("blk.{i}.ssm_conv1d.bias"))
        {
            vec("ssm_conv1d.bias", ch)?
        } else {
            vec![0.0; ch]
        },
        x_proj: if mamba {
            Some(mat("ssm_x", &[c.dt_rank + 2 * ds, di])?)
        } else {
            None
        },
        dt_proj: if mamba {
            Some(mat("ssm_dt", &[di, c.dt_rank])?)
        } else {
            None
        },
        dt_bias: vec("ssm_dt.bias", if mamba { di } else { c.dt_rank })?,
        a: vec("ssm_a", if mamba { di * ds } else { c.dt_rank })?,
        d: vec("ssm_d", if mamba { di } else { c.dt_rank })?,
        norm_w: if mamba
            || !file
                .tensors
                .contains_key(&format!("blk.{i}.ssm_norm.weight"))
        {
            None
        } else {
            Some(vec("ssm_norm.weight", di)?)
        },
        dt_b_c_norm: if mamba
            && (c.dt_b_c_rms
                || file
                    .tensors
                    .contains_key(&format!("blk.{i}.ssm_dt_norm.weight")))
        {
            Some([
                vec("ssm_dt_norm.weight", c.dt_rank)?,
                vec("ssm_b_norm.weight", ds)?,
                vec("ssm_c_norm.weight", ds)?,
            ])
        } else {
            None
        },
        out_proj: mat("ssm_out", &[d, di])?,
        eps: cfg.rms_eps as f32,
        in_scale: load_scalar(file, &format!("blk.{i}.ssm_in.scale"))?,
        out_scale: load_scalar(file, &format!("blk.{i}.ssm_out.scale"))?,
        gdn: None,
        kda: None,
        cfg: c,
    })
}

/// Loads layer `i`'s DeepSeek MLA tensors. The rotary slices of `q_b` / `kv_a` are stored for interleaved
/// RoPE, so those rows are re-ordered to the half-split layout this crate rotates.
fn load_mla(file: &GGUFFile, i: usize, cfg: &GGUFModelConfig) -> Result<MlaAttention> {
    let (h, d) = (cfg.num_heads, cfg.d_model);
    let nope_only = arch_spec(&cfg.architecture).is_some_and(|s| s.mla_nope_only);
    let (hd_k, rope, kv, vd) = (
        cfg.head_dim,
        if nope_only { 0 } else { cfg.rope_n_rot },
        cfg.kv_lora_rank,
        cfg.value_length_mla,
    );
    if vd == 0 || rope > hd_k {
        return Err(Error::Msg(
            "MLA needs a value length and a rope dimension within the key length".into(),
        ));
    }
    let nope = hd_k - rope;
    // Sliding-window MLA layers may carry their own latent / head sizes (`*_swa` keys); only equal geometry is supported.
    for (swa_key, full) in [
        ("kv_lora_rank_swa", kv),
        ("key_length_mla_swa", hd_k),
        ("value_length_mla_swa", vd),
    ] {
        if let Some(v) = file.get_metadata_u32(&format!("{}.attention.{swa_key}", cfg.architecture))
        {
            if v as usize != full
                && arch_spec(&cfg.architecture).is_some_and(|s| s.swa_default.is_some() && s.mla)
            {
                return Err(Error::Msg(format!(
                    "{}: per-layer MLA geometry ({swa_key} = {v}, full layers use {full}) is not supported",
                    cfg.architecture
                )));
            }
        }
    }
    let by_sections = |s: &ArchSpec| {
        let sections = match file
            .metadata
            .get(&format!("{}.rope.dimension_sections", cfg.architecture))
        {
            Some(crate::serialize::gguf::GGUFValue::Array(a)) => a
                .iter()
                .map(|v| match v {
                    crate::serialize::gguf::GGUFValue::Uint32(n) => *n,
                    crate::serialize::gguf::GGUFValue::Int32(n) => u32::try_from(*n).unwrap_or(0),
                    _ => 0,
                })
                .collect::<Vec<_>>(),
            _ => Vec::new(),
        };
        s.mla_rope_by_sections
            && !(sections.first().is_some_and(|&v| v > 0)
                && sections.get(1).is_some_and(|&v| v > 0))
    };
    let interleaved =
        arch_spec(&cfg.architecture).is_some_and(|s| s.rope_interleaved || by_sections(&s));
    let get = |name: &str, shape: &[usize]| -> Result<Tensor> {
        let full = format!("blk.{i}.{name}.weight");
        let t = file.dequantize_tensor(&full)?;
        let expected = crate::layout::checked_numel(shape)
            .map_err(|_| Error::Msg(format!("{full}: expected shape {shape:?} overflows")))?;
        if t.numel() != expected {
            return Err(Error::Msg(format!(
                "{full} has {} values, expected {shape:?}",
                t.numel()
            )));
        }
        t.reshape(shape) // GGUF drops size-1 dims (one head): restore them
    };
    let q_rows = h.checked_mul(hd_k).ok_or_else(|| {
        Error::Msg(format!(
            "blk.{i}.attn_q.weight shape overflows: heads={h}, key_length={hd_k}"
        ))
    })?;
    let (q_a, q_a_norm, q_b_raw) = if cfg.q_lora_rank > 0 {
        let qr = cfg.q_lora_rank;
        let mut norm = Norm::rms(qr, cfg.rms_eps)?;
        norm.weight = get("attn_q_a_norm", &[qr])?;
        (
            Some(get("attn_q_a", &[qr, d])?),
            Some(norm),
            get("attn_q_b", &[q_rows, qr])?,
        )
    } else {
        (None, None, get("attn_q", &[q_rows, d])?)
    };
    let mut kv_a_norm = Norm::rms(kv, cfg.rms_eps)?;
    kv_a_norm.weight = get("attn_kv_a_norm", &[kv])?;
    let kv_a_raw = get("attn_kv_a_mqa", &[kv + rope, d])?;
    // Combined `attn_kv_b` ([h * (nope + v), kv], MiniCPM3) holds each head's K-nope rows then V rows.
    let (k_b, v_b) = if file
        .tensors
        .contains_key(&format!("blk.{i}.attn_kv_b.weight"))
    {
        let kvb = get("attn_kv_b", &[h, nope + vd, kv])?;
        (
            kvb.narrow(1, 0, nope)?.transpose(1, 2)?.contiguous()?,
            kvb.narrow(1, nope, vd)?.contiguous()?,
        )
    } else {
        (
            get("attn_k_b", &[h, kv, nope])?,
            get("attn_v_b", &[h, vd, kv])?,
        )
    };
    // Sparse-attention layers: the lightning indexer (a layer without its tensors reuses the previous layer's selection).
    let spec = arch_spec(&cfg.architecture);
    let sparse = spec.as_ref().filter(|s| s.dsa_indexer);
    let dsa_indexer = match sparse {
        Some(s)
            if file
                .tensors
                .contains_key(&format!("blk.{i}.indexer.attn_q_b.weight")) =>
        {
            let ikey = |k: &str| {
                file.get_metadata_u32(&format!("{}.attention.indexer.{k}", cfg.architecture))
                    .map(|v| v as usize)
            };
            let need = |k: &str| {
                ikey(k).ok_or_else(|| {
                    Error::Msg(format!(
                        "{}.attention.indexer.{k} is missing",
                        cfg.architecture
                    ))
                })
            };
            let (heads, dim, top_k) = (need("head_count")?, need("key_length")?, need("top_k")?);
            if heads == 0 || dim == 0 {
                return Err(Error::Msg(format!(
                    "{}.attention.indexer needs positive head_count and key_length, got {heads} heads and width {dim}",
                    cfg.architecture
                )));
            }
            let ln_eps = if s.dsa_kpool {
                // GLM-5-Next normalizes with the file's LayerNorm epsilon (0 when it is absent).
                f64::from(
                    file.get_metadata_f32(&format!(
                        "{}.attention.layer_norm_epsilon",
                        cfg.architecture
                    ))
                    .unwrap_or(0.0),
                )
            } else if s.dsa_ln_rms_eps {
                cfg.rms_eps
            } else {
                1e-6
            };
            let mut k_norm = Norm::build(true, true, true, dim, ln_eps)?;
            load_norm(file, &format!("blk.{i}.indexer.k_norm"), &mut k_norm)?;
            if rope > dim {
                return Err(Error::Msg(
                    "the indexer's heads are narrower than the rotary dimension".into(),
                ));
            }
            let q_rows = heads.checked_mul(dim).ok_or_else(|| {
                Error::Msg(format!("blk.{i}.indexer.attn_q_b.weight shape overflows: heads={heads}, key_length={dim}"))
            })?;
            Some(DsaIndexer {
                q_b: load_named(
                    file,
                    &format!("blk.{i}.indexer.attn_q_b.weight"),
                    &[
                        q_rows,
                        q_a.as_ref().map_or(cfg.d_model, |_| cfg.q_lora_rank),
                    ],
                )?,
                k: load_named(file, &format!("blk.{i}.indexer.attn_k.weight"), &[dim, d])?,
                proj: load_named(file, &format!("blk.{i}.indexer.proj.weight"), &[heads, d])?,
                k_norm,
                heads,
                dim,
                top_k,
                rot: rope,
                rope_at_end: s.dsa_rope_end,
                norm_pairs: s.dsa_norm_pairs,
                hadamard: !s.dsa_rope_end && !s.dsa_kpool,
                kpool: if s.dsa_kpool {
                    let size = need("kpool")?;
                    if size < 2 {
                        return Err(Error::Msg(format!(
                            "{}.attention.indexer.kpool must be at least 2, got {size}",
                            cfg.architecture
                        )));
                    }
                    Some(KPool {
                        gate: load_named(
                            file,
                            &format!("blk.{i}.indexer_compressor_gate.weight"),
                            &[dim, d],
                        )?,
                        ape: load_named(
                            file,
                            &format!("blk.{i}.indexer_compressor_ape.weight"),
                            &[size, dim],
                        )?,
                        size,
                        select_tail: file
                            .metadata
                            .get(&format!(
                                "{}.attention.indexer.kpool_select_tail",
                                cfg.architecture
                            ))
                            .is_some_and(|v| {
                                matches!(v, crate::serialize::gguf::GGUFValue::Bool(true))
                            }),
                    })
                } else {
                    None
                },
            })
        }
        _ => None,
    };
    let share_dsa = sparse.is_some_and(|s| s.dsa_share) && dsa_indexer.is_none();
    Ok(MlaAttention {
        q_a,
        q_a_norm,
        q_b: if interleaved {
            interleaved_rows(&q_b_raw, h, hd_k, nope, rope)?
        } else {
            q_b_raw
        },
        kv_a: if interleaved {
            interleaved_rows(&kv_a_raw, 1, kv + rope, kv, rope)?
        } else {
            kv_a_raw
        },
        kv_a_norm,
        k_b,
        v_b,
        heads: h,
        nope,
        rope,
        kv_lora: kv,
        v_dim: vd,
        no_rope: arch_spec(&cfg.architecture).is_some_and(|s| s.mla_no_rope),
        sinks: if file
            .tensors
            .contains_key(&format!("blk.{i}.attn_sinks.weight"))
        {
            Some(
                file.dequantize_tensor(&format!("blk.{i}.attn_sinks.weight"))?
                    .reshape(&[h])?,
            )
        } else {
            None
        },
        k_norm: if file
            .tensors
            .contains_key(&format!("blk.{i}.attn_k_norm.weight"))
        {
            let mut n = Norm::rms(rope, cfg.rms_eps)?;
            let w = get("attn_k_norm", &[rope])?;
            // The norm sees the key before the weights were re-ordered for half-split RoPE, so re-order its weight too.
            n.weight = if interleaved {
                interleaved_rows(&w.reshape(&[rope, 1])?, 1, rope, 0, rope)?.reshape(&[rope])?
            } else {
                w
            };
            Some(n)
        } else {
            None
        },
        indexer: dsa_indexer,
        share_dsa,
    })
}

/// Loads layer `i`'s DeepSeek-V4 attention (and its compressor when the layer has a compression ratio).
fn load_dsv4(file: &GGUFFile, i: usize, cfg: &GGUFModelConfig) -> Result<Dsv4Attn> {
    let arch = &cfg.architecture;
    let (d, h, hd, rope) = (cfg.d_model, cfg.num_heads, cfg.head_dim, cfg.rope_n_rot);
    let get = |k: &str| {
        file.get_metadata_u32(&format!("{arch}.attention.{k}"))
            .map(|v| v as usize)
    };
    let need =
        |k: &str| get(k).ok_or_else(|| Error::Msg(format!("{arch}.attention.{k} is missing")));
    let (groups, lora, window, qr) = (
        need("output_group_count")?,
        need("output_lora_rank")?,
        need("sliding_window")?,
        need("q_lora_rank")?,
    );
    let ratio = int_array(file, &format!("{arch}.attention.compress_ratios"))?
        .get(i)
        .copied()
        .unwrap_or(0) as usize;
    if qr == 0 || groups == 0 || h % groups != 0 || rope > hd {
        return Err(Error::Msg(format!(
            "{arch}: inconsistent attention geometry"
        )));
    }
    if ratio != 0 && ratio != 4 && ratio != 128 {
        return Err(Error::Msg(format!(
            "{arch}: compression ratio {ratio} is not supported (only 0, 4 and 128)"
        )));
    }
    let compress_base = file
        .get_metadata_f32(&format!("{arch}.attention.compress_rope_freq_base"))
        .ok_or_else(|| {
            Error::Msg(format!(
                "{arch}.attention.compress_rope_freq_base is missing"
            ))
        })?;
    // Compressing layers rotate with the compress base and the model's position scaling; the others with the plain base.
    let rope_sched = if ratio == 0 {
        Dsv4Rope {
            base: cfg.rope_freq_base,
            scale: 1.0,
            yarn: None,
        }
    } else {
        match &cfg.rope_scaling {
            RopeScaling::None => Dsv4Rope {
                base: compress_base,
                scale: 1.0,
                yarn: None,
            },
            RopeScaling::Linear { factor } => Dsv4Rope {
                base: compress_base,
                scale: if *factor > 0.0 {
                    (1.0 / *factor) as f32
                } else {
                    1.0
                },
                yarn: None,
            },
            RopeScaling::Yarn {
                factor,
                orig_ctx,
                beta_fast,
                beta_slow,
                ..
            } => {
                // ggml's YaRN correction range over the `rope` rotated dimensions, with the extrapolation mix at full strength.
                let base = f64::from(compress_base);
                let corr = |rot: f64| {
                    rope as f64 * (*orig_ctx as f64 / (rot * 2.0 * std::f64::consts::PI)).ln()
                        / (2.0 * base.ln())
                };
                let low = corr(*beta_fast).floor().max(0.0) as f32;
                let high = corr(*beta_slow).ceil().min(rope as f64 - 1.0) as f32;
                Dsv4Rope {
                    base: compress_base,
                    scale: (1.0 / *factor) as f32,
                    yarn: Some((low, high, 1.0)),
                }
            }
        }
    };
    let name = |n: &str| format!("blk.{i}.{n}.weight");
    let norm = |nm: &str, len: usize| -> Result<Norm> {
        let mut n = Norm::rms(len, cfg.rms_eps)?;
        n.weight = load_named(file, &name(nm), &[len])?;
        Ok(n)
    };
    let comp = if ratio == 0 {
        None
    } else {
        let overlap = ratio == 4;
        let cw = if overlap { 2 * hd } else { hd };
        Some(Dsv4Comp {
            ratio,
            overlap,
            wkv: load_named(file, &name("attn_compressor_kv"), &[cw, d])?,
            wgate: load_named(file, &name("attn_compressor_gate"), &[cw, d])?,
            ape: load_named(file, &name("attn_compressor_ape"), &[ratio, cw])?,
            norm: norm("attn_compressor_norm", hd)?,
            rope: rope_sched,
        })
    };
    // The indexer of a ratio-4 layer (a layer without its tensors attends every complete block).
    let lid = if ratio == 4 && file.tensors.contains_key(&name("indexer.attn_q_b")) {
        let (lh, ldim, top_k) = (
            need("indexer.head_count")?,
            need("indexer.key_length")?,
            need("indexer.top_k")?,
        );
        let lcw = 2 * ldim;
        Some(Dsv4Lid {
            q_b: load_named(file, &name("indexer.attn_q_b"), &[lh * ldim, qr])?,
            proj: load_named(file, &name("indexer.proj"), &[lh, d])?,
            heads: lh,
            dim: ldim,
            top_k,
            comp: Dsv4Comp {
                ratio,
                overlap: true,
                wkv: load_named(file, &name("indexer_compressor_kv"), &[lcw, d])?,
                wgate: load_named(file, &name("indexer_compressor_gate"), &[lcw, d])?,
                ape: load_named(file, &name("indexer_compressor_ape"), &[ratio, lcw])?,
                norm: norm("indexer_compressor_norm", ldim)?,
                rope: rope_sched,
            },
        })
    } else {
        None
    };
    Ok(Dsv4Attn {
        q_a: load_named(file, &name("attn_q_a"), &[qr, d])?,
        q_a_norm: norm("attn_q_a_norm", qr)?,
        q_b: load_named(file, &name("attn_q_b"), &[h * hd, qr])?,
        kv: load_named(file, &name("attn_kv"), &[hd, d])?,
        kv_norm: norm("attn_kv_a_norm", hd)?,
        sinks: file
            .dequantize_tensor(&name("attn_sinks"))?
            .to_vec::<f32>()?,
        wo_a: load_named(
            file,
            &name("attn_output_a"),
            &[groups, lora, h / groups * hd],
        )?,
        wo_b: load_named(file, &name("attn_output_b"), &[d, groups * lora])?,
        heads: h,
        head_dim: hd,
        rope,
        groups,
        lora,
        window,
        eps: cfg.rms_eps,
        sched: rope_sched,
        comp,
        lid,
    })
}

/// Loads a Qwen4-Exp low-rank hyper-connection mixer (`{prefix}_norm` / `_down` / `_up` and, in a layer, `_inject`).
fn load_low_rank_hc(
    file: &GGUFFile,
    prefix: &str,
    n: usize,
    d: usize,
    inject: bool,
) -> Result<HcParams> {
    let down_name = format!("{prefix}_down.weight");
    let rank = file
        .tensors
        .get(&down_name)
        .and_then(|t| t.shape.first().copied())
        .ok_or_else(|| Error::Msg(format!("missing {down_name}")))?;
    Ok(HcParams {
        w: Tensor::from_slice(&[0f32], &[1])?,
        base: Tensor::from_slice(&[0f32], &[1])?,
        scale: Vec::new(),
        low_rank: Some(LowRankHc {
            norm: load_named(file, &format!("{prefix}_norm.weight"), &[n, d])?,
            down: load_named(file, &down_name, &[rank, n * d])?,
            up: load_named(file, &format!("{prefix}_up.weight"), &[n * d, rank])?,
            inject: if inject {
                Some(load_named(
                    file,
                    &format!("{prefix}_inject.weight"),
                    &[n, n * d],
                )?)
            } else {
                None
            },
        }),
    })
}

/// Reads an integer metadata array of any width.
fn int_array(file: &GGUFFile, key: &str) -> Result<Vec<u64>> {
    use crate::serialize::gguf::GGUFValue as V;
    match file.metadata.get(key) {
        Some(V::Array(a)) => a
            .iter()
            .map(|v| match v {
                V::Uint8(x) => Ok(u64::from(*x)),
                V::Uint16(x) => Ok(u64::from(*x)),
                V::Uint32(x) => Ok(u64::from(*x)),
                V::Uint64(x) => Ok(*x),
                V::Int32(x) if *x >= 0 => Ok(*x as u64),
                V::Int64(x) if *x >= 0 => Ok(*x as u64),
                _ => Err(Error::Msg(format!("{key} must hold non-negative integers"))),
            })
            .collect(),
        _ => Err(Error::Msg(format!("{key} is missing"))),
    }
}

/// One RWKV-6 layer. The head size, the rescale period and the low-rank widths come from the metadata / tensor shapes.
fn load_rwkv6_layer(file: &GGUFFile, i: usize, cfg: &GGUFModelConfig) -> Result<super::gguf_rwkv::Rwkv6Layer> {
    let arch = &cfg.architecture;
    let d = cfg.d_model;
    let head_size = file
        .get_metadata_u32(&format!("{arch}.wkv.head_size"))
        .map(|v| v as usize)
        .ok_or_else(|| Error::Msg(format!("{arch}.wkv.head_size is missing")))?;
    if head_size == 0 || d % head_size != 0 {
        return Err(Error::Msg(format!(
            "{arch}.wkv.head_size {head_size} does not divide the width {d}"
        )));
    }
    let rescale_every = file
        .get_metadata_u32(&format!("{arch}.rescale_every_n_layers"))
        .unwrap_or(0) as usize;
    let rows = |name: &str| -> Result<usize> {
        file.tensors
            .get(name)
            .and_then(|t| t.shape.first().copied())
            .ok_or_else(|| Error::Msg(format!("missing {name}")))
    };
    let t = |nm: &str| format!("blk.{i}.{nm}");
    let mat = |nm: &str, shape: &[usize]| load_named(file, &format!("{}.weight", t(nm)), shape);
    let extra = rows(&format!("{}.weight", t("time_mix_w1")))? / 5;
    let extra_decay = rows(&format!("{}.weight", t("time_mix_decay_w1")))?;
    let ffn = rows(&format!("{}.weight", t("channel_mix_key")))?;
    let norm = |nm: &str| -> Result<Norm> {
        let mut n = Norm::build(true, true, true, d, cfg.rms_eps)?;
        load_norm(file, &t(nm), &mut n)?;
        Ok(n)
    };
    // Static interpolation offsets, fused (`[5, d]`) or one tensor each, in the order w k v r g.
    let lerp = if file.tensors.contains_key(&format!("{}.weight", t("time_mix_lerp_fused"))) {
        mat("time_mix_lerp_fused", &[5, 1, 1, d])?.reshape(&[5, d])?
    } else {
        let parts = ["w", "k", "v", "r", "g"]
            .iter()
            .map(|c| mat(&format!("time_mix_lerp_{c}"), &[1, 1, d])?.reshape(&[d]))
            .collect::<Result<Vec<_>>>()?;
        Tensor::stack(&parts, 0)?
    };
    let ln = mat("time_mix_ln", &[d])?;
    let ln_bias = load_named(file, &format!("{}.bias", t("time_mix_ln")), &[d])?;
    Ok(super::gguf_rwkv::Rwkv6Layer {
        attn_norm: norm("attn_norm")?,
        ffn_norm: norm("attn_norm_2")?,
        mix_w1: mat("time_mix_w1", &[5 * extra, d])?,
        mix_w2: mat("time_mix_w2", &[5, d, extra])?,
        lerp_x: mat("time_mix_lerp_x", &[1, 1, d])?.reshape(&[d])?,
        lerp,
        first: mat("time_mix_first", &[d / head_size, head_size])?,
        decay: mat("time_mix_decay", &[d])?,
        decay_w1: mat("time_mix_decay_w1", &[extra_decay, d])?,
        decay_w2: mat("time_mix_decay_w2", &[d, extra_decay])?,
        key: mat("time_mix_key", &[d, d])?,
        value: mat("time_mix_value", &[d, d])?,
        receptance: mat("time_mix_receptance", &[d, d])?,
        gate: mat("time_mix_gate", &[d, d])?,
        ln_weight: ln,
        ln_bias,
        output: mat("time_mix_output", &[d, d])?,
        cm_lerp_k: mat("channel_mix_lerp_k", &[1, 1, d])?.reshape(&[d])?,
        cm_lerp_r: mat("channel_mix_lerp_r", &[1, 1, d])?.reshape(&[d])?,
        cm_key: mat("channel_mix_key", &[ffn, d])?,
        cm_value: mat("channel_mix_value", &[d, ffn])?,
        cm_receptance: mat("channel_mix_receptance", &[d, d])?,
        head_size,
        rescale: rescale_every != 0 && (i + 1) % rescale_every == 0,
    })
}

/// One RWKV-7 layer; the low-rank widths come from the tensor shapes, the head size from the metadata.
fn load_rwkv7_layer(file: &GGUFFile, i: usize, cfg: &GGUFModelConfig) -> Result<super::gguf_rwkv::Rwkv7Layer> {
    let arch = &cfg.architecture;
    let d = cfg.d_model;
    let head_size = file
        .get_metadata_u32(&format!("{arch}.wkv.head_size"))
        .map(|v| v as usize)
        .ok_or_else(|| Error::Msg(format!("{arch}.wkv.head_size is missing")))?;
    if head_size == 0 || d % head_size != 0 {
        return Err(Error::Msg(format!(
            "{arch}.wkv.head_size {head_size} does not divide the width {d}"
        )));
    }
    let t = |nm: &str| format!("blk.{i}.{nm}");
    let has = |nm: &str| file.tensors.contains_key(&format!("{}.weight", t(nm)));
    let rows = |nm: &str| -> Result<usize> {
        file.tensors
            .get(&format!("{}.weight", t(nm)))
            .and_then(|x| x.shape.first().copied())
            .ok_or_else(|| Error::Msg(format!("missing {}.weight", t(nm))))
    };
    let mat = |nm: &str, shape: &[usize]| load_named(file, &format!("{}.weight", t(nm)), shape);
    // ARWKV (a SwiGLU FFN behind RMS norms) vs. RWKV-7 proper (LayerNorms with bias and a channel mix).
    let swiglu = has("ffn_gate");
    let norm = |nm: &str| -> Result<Norm> {
        let mut n = if swiglu {
            Norm::rms(d, cfg.rms_eps)?
        } else {
            Norm::build(true, true, true, d, cfg.rms_eps)?
        };
        load_norm(file, &t(nm), &mut n)?;
        Ok(n)
    };
    let ffn = if swiglu {
        let ff = rows("ffn_gate")?;
        super::gguf_rwkv::RwkvFfn::SwiGlu {
            norm: norm("ffn_norm")?,
            gate: mat("ffn_gate", &[ff, d])?,
            up: mat("ffn_up", &[ff, d])?,
            down: mat("ffn_down", &[d, ff])?,
        }
    } else {
        let ff = rows("channel_mix_key")?;
        super::gguf_rwkv::RwkvFfn::ChannelMix {
            norm: norm("attn_norm_2")?,
            lerp_k: mat("channel_mix_lerp_k", &[1, 1, d])?.reshape(&[d])?,
            key: mat("channel_mix_key", &[ff, d])?,
            value: mat("channel_mix_value", &[d, ff])?,
        }
    };
    let (rd, ri, rv) = (rows("time_mix_w1")?, rows("time_mix_a1")?, rows("time_mix_v1")?);
    let gate = if has("time_mix_g1") {
        let rg = rows("time_mix_g1")?;
        Some((mat("time_mix_g1", &[rg, d])?, mat("time_mix_g2", &[d, rg])?))
    } else {
        None
    };
    let ln = if has("time_mix_ln") {
        Some((
            mat("time_mix_ln", &[d])?,
            load_named(file, &format!("{}.bias", t("time_mix_ln")), &[d])?,
        ))
    } else {
        None
    };
    let nlerp = if gate.is_some() { 6 } else { 5 };
    let lerp = mat("time_mix_lerp_fused", &[nlerp, 1, 1, d])?.reshape(&[nlerp, d])?;
    Ok(super::gguf_rwkv::Rwkv7Layer {
        attn_norm: norm("attn_norm")?,
        ffn,
        lerp,
        w0: mat("time_mix_w0", &[d])?,
        w1: mat("time_mix_w1", &[rd, d])?,
        w2: mat("time_mix_w2", &[d, rd])?,
        a0: mat("time_mix_a0", &[d])?,
        a1: mat("time_mix_a1", &[ri, d])?,
        a2: mat("time_mix_a2", &[d, ri])?,
        v0: mat("time_mix_v0", &[d])?,
        v1: mat("time_mix_v1", &[rv, d])?,
        v2: mat("time_mix_v2", &[d, rv])?,
        gate,
        k_k: mat("time_mix_k_k", &[d])?,
        k_a: mat("time_mix_k_a", &[d])?,
        r_k: mat("time_mix_r_k", &[d])?,
        key: mat("time_mix_key", &[d, d])?,
        value: mat("time_mix_value", &[d, d])?,
        receptance: mat("time_mix_receptance", &[d, d])?,
        ln,
        output: mat("time_mix_output", &[d, d])?,
        head_size,
        first: i == 0,
    })
}

/// One QRWKV6 layer; the key / value width (and so the number of grouped heads) comes from the tensor shapes.
fn load_qrwkv6_layer(file: &GGUFFile, i: usize, cfg: &GGUFModelConfig) -> Result<super::gguf_rwkv::Qrwkv6Layer> {
    let arch = &cfg.architecture;
    let d = cfg.d_model;
    let head_size = file
        .get_metadata_u32(&format!("{arch}.wkv.head_size"))
        .map(|v| v as usize)
        .ok_or_else(|| Error::Msg(format!("{arch}.wkv.head_size is missing")))?;
    if head_size == 0 || d % head_size != 0 {
        return Err(Error::Msg(format!(
            "{arch}.wkv.head_size {head_size} does not divide the width {d}"
        )));
    }
    let t = |nm: &str| format!("blk.{i}.{nm}");
    let rows = |nm: &str| -> Result<usize> {
        file.tensors
            .get(&format!("{}.weight", t(nm)))
            .and_then(|x| x.shape.first().copied())
            .ok_or_else(|| Error::Msg(format!("missing {}.weight", t(nm))))
    };
    let mat = |nm: &str, shape: &[usize]| load_named(file, &format!("{}.weight", t(nm)), shape);
    let norm = |nm: &str| -> Result<Norm> {
        let mut n = Norm::rms(d, cfg.rms_eps)?;
        load_norm(file, &t(nm), &mut n)?;
        Ok(n)
    };
    let proj = |nm: &str, out: usize| -> Result<super::gguf_rwkv::BiasedProj> {
        Ok(super::gguf_rwkv::BiasedProj {
            weight: mat(nm, &[out, d])?,
            bias: load_bias(file, &format!("{}.bias", t(nm)), out)?,
        })
    };
    let kv = rows("time_mix_key")?;
    let extra = rows("time_mix_w1")? / 5;
    let extra_decay = rows("time_mix_decay_w1")?;
    let ff = rows("ffn_gate")?;
    Ok(super::gguf_rwkv::Qrwkv6Layer {
        attn_norm: norm("attn_norm")?,
        ffn_norm: norm("ffn_norm")?,
        mix_w1: mat("time_mix_w1", &[5 * extra, d])?,
        mix_w2: mat("time_mix_w2", &[5, d, extra])?,
        lerp_x: mat("time_mix_lerp_x", &[1, 1, d])?.reshape(&[d])?,
        lerp: mat("time_mix_lerp_fused", &[5, 1, 1, d])?.reshape(&[5, d])?,
        decay: mat("time_mix_decay", &[d])?,
        decay_w1: mat("time_mix_decay_w1", &[extra_decay, d])?,
        decay_w2: mat("time_mix_decay_w2", &[d, extra_decay])?,
        key: proj("time_mix_key", kv)?,
        value: proj("time_mix_value", kv)?,
        receptance: proj("time_mix_receptance", d)?,
        gate: mat("time_mix_gate", &[d, d])?,
        output: mat("time_mix_output", &[d, d])?,
        ffn_gate: mat("ffn_gate", &[ff, d])?,
        ffn_up: mat("ffn_up", &[ff, d])?,
        ffn_down: mat("ffn_down", &[d, ff])?,
        head_size,
    })
}

/// Gemma 3n's model-level tensors; the stream count and the per-layer input width come from the tensor shapes.
fn load_g3n_model(file: &GGUFFile, cfg: &GGUFModelConfig) -> Result<G3nModel> {
    let d = cfg.d_model;
    let shape = |name: &str| -> Result<Vec<usize>> {
        file.tensors
            .get(name)
            .map(|t| t.shape.clone())
            .ok_or_else(|| Error::Msg(format!("missing {name}")))
    };
    let width = *shape("per_layer_proj_norm.weight")?
        .first()
        .ok_or_else(|| Error::Msg("per_layer_proj_norm.weight is empty".into()))?;
    let n_altup = shape("altup_proj.weight")?
        .first()
        .copied()
        .ok_or_else(|| Error::Msg("altup_proj.weight is empty".into()))?
        + 1;
    let layers = cfg.num_layers;
    let rows = layers
        .checked_mul(width)
        .ok_or_else(|| Error::Msg("per-layer input width overflows".into()))?;
    let per_layer_emb = LazyRows::from_file(file, "per_layer_token_embd.weight", cfg.vocab_size, rows)?;
    let mut per_layer_proj_norm = Norm::rms(width, cfg.rms_eps)?;
    load_norm(file, "per_layer_proj_norm", &mut per_layer_proj_norm)?;
    Ok(G3nModel {
        altup_proj: load_named(file, "altup_proj.weight", &[n_altup - 1, d, d])?,
        altup_unembd_proj: load_named(file, "altup_unembd_proj.weight", &[n_altup - 1, d, d])?,
        per_layer_emb,
        per_layer_model_proj: load_named(file, "per_layer_model_proj.weight", &[rows, d])?,
        per_layer_proj_norm,
        width,
        n_altup,
    })
}

/// One Gemma 3n layer's tensors. Layers from the twentieth on reuse the K/V of layer 18 (sliding window) or 19.
fn load_g3n_layer(
    file: &GGUFFile,
    i: usize,
    cfg: &GGUFModelConfig,
    g: &G3nModel,
    sliding: bool,
) -> Result<G3nLayer> {
    const KV_LAYERS: usize = 20;
    const SPARSE_LAYERS: usize = 10;
    let (d, w, na) = (cfg.d_model, g.width, g.n_altup);
    let mat = |nm: &str, shape: &[usize]| load_named(file, &format!("blk.{i}.{nm}.weight"), shape);
    let rank = file
        .tensors
        .get(&format!("blk.{i}.laurel_l.weight"))
        .and_then(|t| t.shape.first().copied())
        .ok_or_else(|| Error::Msg(format!("missing blk.{i}.laurel_l.weight")))?;
    let norm = |nm: &str| -> Result<Norm> {
        let mut n = Norm::rms(d, cfg.rms_eps)?;
        load_norm(file, &format!("blk.{i}.{nm}"), &mut n)?;
        Ok(n)
    };
    Ok(G3nLayer {
        inp_gate: mat("inp_gate", &[w, d])?,
        proj: mat("proj", &[d, w])?,
        post_norm: norm("post_norm")?,
        correct_coef: mat("altup_correct_coef", &[na, na])?,
        correct_scale: mat("altup_correct_scale", &[d])?,
        predict_coef: mat("altup_predict_coef", &[na * na, na])?,
        router: mat("altup_router", &[na, d])?,
        router_norm: norm("altup_router_norm")?,
        laurel_l: mat("laurel_l", &[rank, d])?,
        laurel_r: mat("laurel_r", &[d, rank])?,
        laurel_post_norm: norm("laurel_post_norm")?,
        sparse: i < SPARSE_LAYERS,
        kv_source: (i >= KV_LAYERS).then(|| KV_LAYERS - if sliding { 2 } else { 1 }),
    })
}

/// Gemma 4's model-level tensors: the per-layer input embedding exists only in the smaller models.
fn load_g4_model(file: &GGUFFile, cfg: &GGUFModelConfig) -> Result<G4Model> {
    if !file.tensors.contains_key("per_layer_token_embd.weight") {
        return Ok(G4Model { per_layer: None });
    }
    let width = file
        .tensors
        .get("per_layer_proj_norm.weight")
        .and_then(|t| t.shape.first().copied())
        .ok_or_else(|| Error::Msg("missing per_layer_proj_norm.weight".into()))?;
    let rows = cfg
        .num_layers
        .checked_mul(width)
        .ok_or_else(|| Error::Msg("per-layer input width overflows".into()))?;
    let emb = LazyRows::from_file(file, "per_layer_token_embd.weight", cfg.vocab_size, rows)?;
    let mut proj_norm = Norm::rms(width, cfg.rms_eps)?;
    load_norm(file, "per_layer_proj_norm", &mut proj_norm)?;
    Ok(G4Model {
        per_layer: Some(G4PerLayer {
            emb,
            model_proj: load_named(file, "per_layer_model_proj.weight", &[rows, cfg.d_model])?,
            proj_norm,
            width,
        }),
    })
}

/// One Gemma 4 layer's extras. The last `kv_shared_layers` layers attend to the K/V of layer `n - shared - 2`
/// (sliding window) or `n - shared - 1` (full attention).
fn load_g4_layer(
    file: &GGUFFile,
    i: usize,
    cfg: &GGUFModelConfig,
    g: &G4Model,
    sliding: bool,
) -> Result<G4Layer> {
    let d = cfg.d_model;
    let per_layer = match &g.per_layer {
        Some(p) => {
            let mut post = Norm::rms(d, cfg.rms_eps)?;
            load_norm(file, &format!("blk.{i}.post_norm"), &mut post)?;
            Some((
                load_named(file, &format!("blk.{i}.inp_gate.weight"), &[p.width, d])?,
                load_named(file, &format!("blk.{i}.proj.weight"), &[d, p.width])?,
                post,
            ))
        }
        None => None,
    };
    let from_start = cfg.num_layers.saturating_sub(cfg.kv_shared_layers);
    let back = if sliding { 2 } else { 1 };
    Ok(G4Layer {
        per_layer,
        out_scale: load_scalar(file, &format!("blk.{i}.layer_output_scale.weight"))?,
        kv_source: (cfg.kv_shared_layers > 0 && i >= from_start && from_start >= back)
            .then(|| from_start - back),
    })
}

fn load_ple(file: &GGUFFile, cfg: &GGUFModelConfig) -> Result<PleModel> {
    let arch = &cfg.architecture;
    let get = |k: &str| {
        file.get_metadata_u32(&format!("{arch}.{k}"))
            .map(|v| v as usize)
    };
    let need = |k: &str| get(k).ok_or_else(|| Error::Msg(format!("{arch}.{k} is missing")));
    let (ngram, heads_per_ngram, kernel) = (
        need("ple.ngram_size")?,
        need("ple.heads_per_ngram")?,
        need("ple.conv_kernel")?,
    );
    let head_dim = need("embedding_length_per_layer_input")?;
    if ngram < 2 || kernel == 0 || heads_per_ngram == 0 {
        return Err(Error::Msg(format!("{arch}: bad PLE n-gram settings")));
    }
    let n_heads = (ngram - 1) * heads_per_ngram;
    let multipliers = int_array(file, &format!("{arch}.ple.layer_multipliers"))?;
    let offsets = int_array(file, &format!("{arch}.ple.head_offsets"))?;
    let vocab_sizes = int_array(file, &format!("{arch}.ple.head_vocab_sizes"))?;
    if multipliers.len() < ngram
        || offsets.len() < n_heads
        || vocab_sizes.len() < n_heads
        || vocab_sizes.contains(&0)
    {
        return Err(Error::Msg(format!("{arch}: PLE hash tables are too short")));
    }
    let rows = file
        .tensors
        .get("per_layer_token_embd.weight")
        .and_then(|t| t.shape.first().copied())
        .ok_or_else(|| Error::Msg("missing per_layer_token_embd.weight".into()))?;
    let (n, d) = (cfg.hc.map_or(1, |h| h.0), cfg.d_model);
    let mut layers = Vec::new();
    for layer in int_array(file, &format!("{arch}.ple.layers"))? {
        let i = layer as usize;
        let mat =
            |nm: &str, shape: &[usize]| load_named(file, &format!("blk.{i}.{nm}.weight"), shape);
        let conv_name = format!("blk.{i}.ple_conv1d.weight");
        let conv = file.dequantize_tensor(&conv_name)?.to_vec::<f32>()?;
        if conv.len() != n * d * kernel {
            return Err(Error::Msg(format!(
                "{conv_name} has {} values, expected {}",
                conv.len(),
                n * d * kernel
            )));
        }
        layers.push(PleLayer {
            layer: i,
            key: mat("ple_key", &[n * d, n_heads * head_dim])?,
            value: mat("ple_value", &[d, n_heads * head_dim])?,
            norm_key: mat("ple_norm_key", &[n, d])?,
            norm_query: mat("ple_norm_query", &[n, d])?,
            norm_conv: mat("ple_norm_conv", &[n, d])?,
            conv,
        });
    }
    Ok(PleModel {
        table: load_named(file, "per_layer_token_embd.weight", &[rows, head_dim])?,
        head_dim,
        ngram,
        heads_per_ngram,
        kernel,
        eos: get("ple.eos_token_id").unwrap_or(0) as i64,
        multipliers,
        offsets,
        vocab_sizes,
        layers,
    })
}

/// DeepSeek's attention scale exactly as llama.cpp derives it: `mscale^2 / sqrt(head_dim_k)` where the
/// magnitude factor folds in the YaRN / linear frequency scale and `yarn_log_multiplier`.
fn mla_kq_scale(cfg: &GGUFModelConfig, head_dim_k: usize) -> f32 {
    let get_mscale = |scale: f64, m: f64| {
        if scale <= 1.0 {
            1.0
        } else {
            0.1 * m * scale.ln() + 1.0
        }
    };
    let (factor, yarn) = match &cfg.rope_scaling {
        RopeScaling::None => (1.0, false),
        RopeScaling::Linear { factor } => (*factor, false),
        RopeScaling::Yarn { factor, .. } => (*factor, true),
    };
    let mut attn_factor = 1.0;
    if yarn {
        attn_factor = get_mscale(factor, 1.0) / get_mscale(factor, cfg.yarn_log_mul);
        attn_factor *= 1.0 / (1.0 + 0.1 * factor.ln());
    }
    attn_factor *= cfg.rope_attn_factor;
    let ln_inv_freq_scale = factor.ln(); // freq_scale = 1 / factor
    let attn_factor_org = attn_factor * (1.0 + 0.1 * ln_inv_freq_scale);
    let mscale = attn_factor_org * (1.0 + 0.1 * cfg.yarn_log_mul * ln_inv_freq_scale);
    (mscale * mscale / (head_dim_k as f64).sqrt()) as f32
}

/// An optional `[len]` bias tensor.
pub(crate) fn load_bias(file: &GGUFFile, name: &str, len: usize) -> Result<Option<Tensor>> {
    if file.tensors.contains_key(name) {
        Ok(Some(load_named(file, name, &[len])?))
    } else {
        Ok(None)
    }
}

/// Optional per-expert scalars (`ffn_*_exps.scale`, one value per expert).
fn load_expert_scales(file: &GGUFFile, name: &str, n_expert: usize) -> Result<Option<Vec<f32>>> {
    if !file.tensors.contains_key(name) {
        return Ok(None);
    }
    let v = file.dequantize_tensor(name)?.to_vec::<f32>()?;
    if v.len() != n_expert {
        return Err(Error::Msg(format!(
            "{name} has {} values, expected {n_expert}",
            v.len()
        )));
    }
    Ok(Some(v))
}

/// An optional one-element tensor (`*.scale`).
fn load_scalar(file: &GGUFFile, name: &str) -> Result<Option<f32>> {
    if !file.tensors.contains_key(name) {
        return Ok(None);
    }
    let v = file.dequantize_tensor(name)?.to_vec::<f32>()?;
    match v.as_slice() {
        [s] => Ok(Some(*s)),
        _ => Err(Error::Msg(format!(
            "{name} must hold exactly one value, found {}",
            v.len()
        ))),
    }
}

/// llama.cpp's `llama` architecture stores Q/K rows for *interleaved* RoPE (rotating adjacent
/// pairs `(2i, 2i+1)`), whereas `RotaryEmbedding::apply` rotates the two halves of each head
/// (`(i, i + hd/2)`). Reorder each head's rows `[hd/2, 2] -> [2, hd/2]` so both agree; without
/// this, real Llama weights get the wrong positional rotation.
fn interleaved_to_half_split(w: &Tensor, heads: usize) -> Result<Tensor> {
    let (rows, cols) = (w.shape()[0], w.shape()[1]);
    let hd = rows / heads;
    w.reshape(&[heads, hd / 2, 2, cols])?
        .transpose(1, 2)?
        .contiguous()?
        .reshape(&[rows, cols])
}

/// [`interleaved_to_half_split`] restricted to the first `n_rot` rows of every head (partial rotary:
/// the remaining rows are not rotated and keep their order).
fn interleaved_to_half_split_partial(w: &Tensor, heads: usize, n_rot: usize) -> Result<Tensor> {
    let (rows, cols) = (w.shape()[0], w.shape()[1]);
    let hd = rows / heads;
    if n_rot >= hd {
        return interleaved_to_half_split(w, heads);
    }
    let per_head = w.reshape(&[heads, hd, cols])?;
    let rot = per_head
        .narrow(1, 0, n_rot)?
        .contiguous()?
        .reshape(&[heads, n_rot / 2, 2, cols])?
        .transpose(1, 2)?
        .contiguous()?
        .reshape(&[heads, n_rot, cols])?;
    let rest = per_head.narrow(1, n_rot, hd - n_rot)?.contiguous()?;
    Tensor::cat(&[rot, rest], 1)?.reshape(&[rows, cols])
}

/// Architectures whose GGUF tensors map onto [`GGUFDecoderBlock`] (Norm + RoPE + GQA + SwiGLU).
/// Mistral and Mixtral (MoE) use `llama`; Qwen2 adds q/k/v biases and half-split RoPE; Qwen2-MoE adds
/// a shared expert.
pub const SUPPORTED_ARCHITECTURES: [&str; 139] = [
    "llama",
    "arwkv7",
    "rwkv6qwen2",
    "gemma4",
    "step35",
    "rwkv6",
    "rwkv7",
    "gemma3n",
    "apertus",
    "bitnet",
    "plm",
    "chameleon",
    "mistral3",
    "internlm2",
    "exaone",
    "exaone4",
    "smollm3",
    "granite",
    "granitemoe",
    "olmo",
    "olmo2",
    "olmoe",
    "phi2",
    "phi3",
    "stablelm",
    "starcoder2",
    "command-r",
    "deepseek2",
    "qwen2",
    "qwen2moe",
    "qwen3",
    "qwen3moe",
    "gemma",
    "gemma2",
    "gemma3",
    "xverse",
    "baichuan",
    "nanbeige",
    "arcee",
    "maincoder",
    "seed_oss",
    "nemotron",
    "jais2",
    "orion",
    "codeshell",
    "gptneox",
    "falcon",
    "minicpm",
    "phimoe",
    "pangu-embedded",
    "hunyuan-dense",
    "dbrx",
    "hunyuan-moe",
    "dots1",
    "bailingmoe",
    "bailingmoe2",
    "ernie4_5",
    "ernie4_5-moe",
    "gpt-oss",
    "cohere2",
    "granite_swa",
    "deci",
    "openelm",
    "glm4",
    "chatglm",
    "glm4moe",
    "qwen",
    "deepseek",
    "plamo",
    "mellum",
    "maple",
    "minimax-m2",
    "exaone-moe",
    "afmoe",
    "spark2_5",
    "mimo2",
    "laguna",
    "kolibri1",
    "flint",
    "cohere2moe",
    "grok",
    "arctic",
    "qwen2vl",
    "qwen3vl",
    "qwen3vlmoe",
    "paddleocr",
    "hunyuan_vl",
    "gpt2",
    "starcoder",
    "bloom",
    "mpt",
    "refact",
    "jais",
    "mistral4",
    "llama4",
    "hy_v3",
    "plamo3",
    "smallthinker",
    "grovemoe",
    "minicpm3",
    "maion-coder",
    "mamba",
    "mamba2",
    "nemotron_h",
    "nemotron_h_moe",
    "granitehybrid",
    "falcon-h1",
    "jamba",
    "lfm2",
    "lfm2moe",
    "qwen35",
    "qwen35moe",
    "qwen3next",
    "deepseek32",
    "glm-dsa",
    "muse-glimmer",
    "beam",
    "talkie",
    "hrm_text",
    "k2-horizon",
    "muse-spark",
    "cogvlm",
    "qwen3tts",
    "pockettts",
    "dream",
    "llada",
    "llada-moe",
    "rnd1",
    "plamo2",
    "kimi-linear",
    "bailingmoe3",
    "kimi-k3",
    "dots3note",
    "minimax-01",
    "minimax-m3",
    "hy_v4",
    "glm5-next",
    "qwen4exp",
    "deepseek4",
];

/// Which regex family splits text into words before byte-level BPE (`tokenizer.ggml.pre`).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum PreTok {
    /// Original GPT-2 pattern.
    Gpt2,
    /// Llama-3: case-insensitive contractions, digits in groups of up to 3.
    Llama3,
    /// Qwen2: like Llama-3 but single digits.
    Qwen2,
}

enum TokKind {
    /// SentencePiece (`tokenizer.ggml.model = "llama"`): score-ranked pair merges.
    Spm,
    /// Byte-level BPE (`"gpt2"`): rank-ordered merges over GPT-2's byte->unicode alphabet.
    Bpe {
        ranks: std::collections::HashMap<String, usize>,
        pre: PreTok,
        enc: Vec<char>,
        dec: std::collections::HashMap<char, u8>,
    },
}

/// GPT-2's reversible byte -> printable-unicode map (and its inverse).
fn byte_unicode_tables() -> (Vec<char>, std::collections::HashMap<char, u8>) {
    let mut enc = Vec::with_capacity(256);
    let mut extra = 0u32;
    for b in 0..=255u32 {
        let keep = (33..=126).contains(&b) || (161..=172).contains(&b) || (174..=255).contains(&b);
        let c = if keep {
            char::from_u32(b).unwrap()
        } else {
            extra += 1;
            char::from_u32(255 + extra).unwrap()
        };
        enc.push(c);
    }
    let dec = enc.iter().enumerate().map(|(b, &c)| (c, b as u8)).collect();
    (enc, dec)
}

fn contraction_len(c: &[char], i: usize, case_insensitive: bool) -> Option<usize> {
    if c[i] != '\'' {
        return None;
    }
    for suffix in ["s", "t", "re", "ve", "m", "ll", "d"] {
        let s: Vec<char> = suffix.chars().collect();
        if i + 1 + s.len() <= c.len()
            && s.iter().enumerate().all(|(k, &sc)| {
                let ch = c[i + 1 + k];
                (if case_insensitive {
                    ch.to_ascii_lowercase()
                } else {
                    ch
                }) == sc
            })
        {
            return Some(i + 1 + s.len());
        }
    }
    None
}

/// End (exclusive) of the pre-token starting at `i`, following the GPT-2 / Llama-3 / Qwen2 regexes
/// (`\p{L}` ~ `is_alphabetic`, `\p{N}` ~ `is_numeric`, `\s` ~ `is_whitespace`).
fn pretoken_end(c: &[char], i: usize, mode: PreTok) -> usize {
    let n = c.len();
    let is_l = |ch: char| ch.is_alphabetic();
    let is_n = |ch: char| ch.is_numeric();
    let is_nl = |ch: char| ch == '\r' || ch == '\n';
    let is_sym = |ch: char| !ch.is_whitespace() && !ch.is_alphabetic() && !ch.is_numeric();
    let run = |mut j: usize, f: &dyn Fn(char) -> bool| {
        while j < n && f(c[j]) {
            j += 1;
        }
        j
    };
    let gpt2 = mode == PreTok::Gpt2;

    if let Some(e) = contraction_len(c, i, !gpt2) {
        return e;
    }
    if gpt2 {
        let j = i + usize::from(c[i] == ' ');
        if j < n && is_l(c[j]) {
            return run(j, &is_l);
        }
        if j < n && is_n(c[j]) {
            return run(j, &is_n);
        }
        if j < n && is_sym(c[j]) {
            return run(j, &is_sym);
        }
    } else {
        // [^\r\n\p{L}\p{N}]?\p{L}+
        if is_l(c[i]) {
            return run(i, &is_l);
        }
        if !is_nl(c[i]) && !is_n(c[i]) && i + 1 < n && is_l(c[i + 1]) {
            return run(i + 1, &is_l);
        }
        // \p{N}{1,3} (Llama-3) or \p{N} (Qwen2)
        if is_n(c[i]) {
            let max = if mode == PreTok::Qwen2 { 1 } else { 3 };
            let mut j = i;
            while j < n && j - i < max && is_n(c[j]) {
                j += 1;
            }
            return j;
        }
        //  ?[^\s\p{L}\p{N}]+[\r\n]*
        let j = i + usize::from(c[i] == ' ');
        if j < n && is_sym(c[j]) {
            return run(run(j, &is_sym), &is_nl);
        }
        // \s*[\r\n]+ : through the last newline of the whitespace run
        if c[i].is_whitespace() {
            let wend = run(i, &|ch| ch.is_whitespace());
            if let Some(k) = (i..wend).rev().find(|&k| is_nl(c[k])) {
                return k + 1;
            }
        }
    }
    // \s+(?!\S) then \s+
    if c[i].is_whitespace() {
        let wend = run(i, &|ch| ch.is_whitespace());
        return if wend == n || wend - i == 1 {
            wend
        } else {
            wend - 1
        };
    }
    i + 1
}

fn pretokenize(text: &str, mode: PreTok) -> Vec<String> {
    let c: Vec<char> = text.chars().collect();
    let mut out = Vec::new();
    let mut i = 0;
    while i < c.len() {
        let e = pretoken_end(&c, i, mode);
        out.push(c[i..e].iter().collect());
        i = e;
    }
    out
}

/// Tokenizer read from a GGUF file's `tokenizer.ggml.*` metadata: SentencePiece (`llama`: score-ranked
/// merges, `\u{2581}` for spaces, `<0xNN>` byte fallback) or byte-level BPE (`gpt2`: Llama-3, Qwen2,
/// GPT-2 style rank-ordered merges and word-splitting regex).
pub struct GGUFTokenizer {
    tokens: Vec<String>,
    scores: Vec<f32>,
    index: std::collections::HashMap<String, u32>,
    kind: TokKind,
    pub bos_id: Option<u32>,
    pub eos_id: Option<u32>,
    unk_id: u32,
    /// Whether the model expects a BOS token in front of the text (`tokenizer.ggml.add_bos_token`).
    pub add_bos_default: bool,
}

impl GGUFTokenizer {
    pub fn from_gguf(file: &GGUFFile) -> Result<Self> {
        use crate::serialize::gguf::GGUFValue;
        let strings = |key: &str| -> Result<Option<Vec<String>>> {
            match file.metadata.get(key) {
                Some(GGUFValue::Array(a)) => a
                    .iter()
                    .map(|v| match v {
                        GGUFValue::String(s) => Ok(s.clone()),
                        _ => Err(Error::Msg(format!("{key} must be strings"))),
                    })
                    .collect::<Result<_>>()
                    .map(Some),
                _ => Ok(None),
            }
        };
        let tokens = strings("tokenizer.ggml.tokens")?
            .ok_or_else(|| Error::Msg("GGUF file has no tokenizer.ggml.tokens".into()))?;
        let scores: Vec<f32> = match file.metadata.get("tokenizer.ggml.scores") {
            Some(GGUFValue::Array(a)) => a
                .iter()
                .map(|v| {
                    if let GGUFValue::Float32(f) = v {
                        *f
                    } else {
                        0.0
                    }
                })
                .collect(),
            _ => vec![0.0; tokens.len()],
        };
        if scores.len() != tokens.len() {
            return Err(Error::Msg(
                "tokenizer.ggml.scores length differs from tokens".into(),
            ));
        }
        let index = tokens
            .iter()
            .enumerate()
            .map(|(i, s)| (s.clone(), i as u32))
            .collect();

        let model = file
            .get_metadata_str("tokenizer.ggml.model")
            .unwrap_or("llama");
        let kind = match model {
            "llama" => TokKind::Spm,
            "gpt2" => {
                let merges = strings("tokenizer.ggml.merges")?.ok_or_else(|| {
                    Error::Msg("gpt2 tokenizer needs tokenizer.ggml.merges".into())
                })?;
                let pre = match file
                    .get_metadata_str("tokenizer.ggml.pre")
                    .unwrap_or("default")
                {
                    "llama-bpe" | "llama3" | "dbrx" | "smaug-bpe" => PreTok::Llama3,
                    "qwen2" | "deepseek-r1-qwen" => PreTok::Qwen2,
                    "default" | "gpt-2" | "gpt2" => PreTok::Gpt2,
                    other => {
                        return Err(Error::Msg(format!("unsupported pre-tokenizer '{other}'")));
                    }
                };
                let (enc, dec) = byte_unicode_tables();
                TokKind::Bpe {
                    ranks: merges
                        .into_iter()
                        .enumerate()
                        .map(|(r, m)| (m, r))
                        .collect(),
                    pre,
                    enc,
                    dec,
                }
            }
            other => return Err(Error::Msg(format!("unsupported tokenizer model '{other}'"))),
        };
        let add_bos_default = match file.metadata.get("tokenizer.ggml.add_bos_token") {
            Some(GGUFValue::Bool(b)) => *b,
            _ => matches!(kind, TokKind::Spm),
        };
        Ok(Self {
            bos_id: file.get_metadata_u32("tokenizer.ggml.bos_token_id"),
            eos_id: file.get_metadata_u32("tokenizer.ggml.eos_token_id"),
            unk_id: file
                .get_metadata_u32("tokenizer.ggml.unknown_token_id")
                .unwrap_or(0),
            add_bos_default,
            tokens,
            scores,
            index,
            kind,
        })
    }

    pub fn vocab_size(&self) -> usize {
        self.tokens.len()
    }

    /// Encodes `text` (control tokens such as `<|eot_id|>` written in `text` are *not* parsed;
    /// add them by id). SentencePiece: prefix-space + `\u{2581}` escaping, then repeatedly merge the
    /// adjacent pair with the highest vocabulary score, falling back to `<0xNN>` byte tokens.
    /// Byte-level BPE: regex word split, then lowest-rank merges per word.
    pub fn encode(&self, text: &str, add_bos: bool) -> Vec<i64> {
        let mut ids = Vec::new();
        if add_bos {
            ids.extend(self.bos_id.map(i64::from));
        }
        if text.is_empty() {
            return ids;
        }
        match &self.kind {
            TokKind::Spm => self.encode_spm(text, &mut ids),
            TokKind::Bpe {
                ranks, pre, enc, ..
            } => {
                for word in pretokenize(text, *pre) {
                    let mut syms: Vec<String> =
                        word.bytes().map(|b| enc[b as usize].to_string()).collect();
                    // ponytail: O(n^2) per word; a heap over ranks if very long words matter.
                    loop {
                        let best = (0..syms.len().saturating_sub(1))
                            .filter_map(|i| {
                                ranks
                                    .get(&format!("{} {}", syms[i], syms[i + 1]))
                                    .map(|&r| (r, i))
                            })
                            .min();
                        let Some((_, i)) = best else { break };
                        let right = syms.remove(i + 1);
                        syms[i].push_str(&right);
                    }
                    for s in syms {
                        ids.push(
                            self.index
                                .get(&s)
                                .map_or(i64::from(self.unk_id), |&id| i64::from(id)),
                        );
                    }
                }
            }
        }
        ids
    }

    fn encode_spm(&self, text: &str, ids: &mut Vec<i64>) {
        let escaped = format!("\u{2581}{}", text.replace(' ', "\u{2581}"));
        let mut syms: Vec<String> = escaped.chars().map(String::from).collect();
        // ponytail: O(n^2) merge search; use a priority queue if long inputs matter.
        loop {
            let mut best: Option<(usize, f32)> = None;
            for i in 0..syms.len().saturating_sub(1) {
                let joined = format!("{}{}", syms[i], syms[i + 1]);
                if let Some(&id) = self.index.get(&joined) {
                    let s = self.scores[id as usize];
                    if best.is_none_or(|(_, bs)| s > bs) {
                        best = Some((i, s));
                    }
                }
            }
            let Some((i, _)) = best else { break };
            let right = syms.remove(i + 1);
            syms[i].push_str(&right);
        }
        for s in syms {
            match self.index.get(&s) {
                Some(&id) => ids.push(i64::from(id)),
                None => {
                    for b in s.bytes() {
                        let id = self
                            .index
                            .get(&format!("<0x{b:02X}>"))
                            .copied()
                            .unwrap_or(self.unk_id);
                        ids.push(i64::from(id));
                    }
                }
            }
        }
    }

    /// Inverse of [`encode`](Self::encode); ids outside the vocabulary and BOS/EOS are skipped.
    pub fn decode(&self, ids: &[i64]) -> String {
        let mut bytes = Vec::new();
        for &id in ids {
            let Some(tok) = usize::try_from(id).ok().and_then(|i| self.tokens.get(i)) else {
                continue;
            };
            if Some(id as u32) == self.bos_id || Some(id as u32) == self.eos_id {
                continue;
            }
            match &self.kind {
                TokKind::Bpe { dec, .. } => {
                    for ch in tok.chars() {
                        match dec.get(&ch) {
                            Some(&b) => bytes.push(b),
                            None => bytes.extend(ch.to_string().bytes()), // special token text
                        }
                    }
                }
                TokKind::Spm => {
                    let byte = tok
                        .strip_prefix("<0x")
                        .and_then(|r| r.strip_suffix('>'))
                        .filter(|h| h.len() == 2)
                        .and_then(|h| u8::from_str_radix(h, 16).ok());
                    match byte {
                        Some(b) => bytes.push(b),
                        None => bytes.extend(tok.replace('\u{2581}', " ").bytes()),
                    }
                }
            }
        }
        let s = String::from_utf8_lossy(&bytes).into_owned();
        if matches!(self.kind, TokKind::Spm) {
            s.strip_prefix(' ').map(str::to_owned).unwrap_or(s)
        } else {
            s
        }
    }
}

/// Loads layer `i`'s stacked expert tensors (`ffn_gate_inp`, `ffn_{gate,up,down}_exps`) and, for
/// Qwen2-MoE, the shared expert (`ffn_*_shexp`, `ffn_gate_inp_shexp`).
fn load_moe(file: &GGUFFile, i: usize, cfg: &GGUFModelConfig) -> Result<MoeFfn> {
    let spec = arch_spec(&cfg.architecture).unwrap_or(BASE_SPEC);
    let (e, d_model) = (cfg.expert_count, cfg.d_model);
    let (down_nm, up_nm) = if file
        .tensors
        .contains_key(&format!("blk.{i}.ffn_routed_down.weight"))
    {
        ("ffn_routed_down", "ffn_routed_up")
    } else {
        ("ffn_latent_down", "ffn_latent_up")
    };
    let latent_name = format!("blk.{i}.{down_nm}.weight");
    let latent = if file.tensors.contains_key(&latent_name) {
        let lat = file.tensors[&latent_name].shape[0];
        let mid_name = format!("blk.{i}.ffn_routed_norm.weight");
        Some(LatentProj {
            mid_norm: if file.tensors.contains_key(&mid_name) {
                let mut n = Norm::rms(lat, cfg.rms_eps)?;
                n.weight = load_named(file, &mid_name, &[lat])?;
                Some(n)
            } else {
                None
            },
            down: load_named(file, &latent_name, &[lat, d_model])?,
            up: load_named(file, &format!("blk.{i}.{up_nm}.weight"), &[d_model, lat])?,
            pre_norm: if spec.spark_blocks {
                let mut n = Norm::rms(lat, cfg.rms_eps)?;
                n.weight = load_named(file, &format!("blk.{i}.ffn_pre_exp_norm.weight"), &[lat])?;
                Some(n)
            } else {
                None
            },
        })
    } else {
        None
    };
    // Experts (and the shared expert's input) live in the latent space when there is one.
    let d = latent.as_ref().map_or(d_model, |l| l.down.shape()[0]);
    let ungated = spec.moe_act == MoeAct::ReluSqr;
    // `{arch}.swiglu_clamp_exp` / `_shexp`: one limit per layer (or a single value); 0 switches the clamp off.
    let raw_limit = |key: &str| -> f32 {
        if !spec.swiglu_clamp {
            return 0.0;
        }
        match file.metadata.get(&format!("{}.{key}", cfg.architecture)) {
            Some(crate::serialize::gguf::GGUFValue::Array(a)) => match a.get(i) {
                Some(crate::serialize::gguf::GGUFValue::Float32(v)) => *v,
                _ => 0.0,
            },
            Some(crate::serialize::gguf::GGUFValue::Float32(v)) => *v,
            _ => 0.0,
        }
    };
    // A negative limit marks the gate-first form (see `clamped_swiglu`).
    let clamp_limit = |key: &str| {
        raw_limit(key)
            * if spec.swiglu_clamp_gate_first {
                -1.0
            } else {
                1.0
            }
    };
    let moe_limit = clamp_limit("swiglu_clamp_exp");
    let gate_name = format!("blk.{i}.ffn_gate_exps.weight");
    let up_name = format!("blk.{i}.ffn_up_exps.weight");
    let fused_name = format!("blk.{i}.ffn_gate_up_exps.weight");
    let fused = !ungated
        && !file.tensors.contains_key(&gate_name)
        && file.tensors.contains_key(&fused_name);
    let width_of = if ungated { &up_name } else { &gate_name };
    let ff = if fused {
        file.tensors[&fused_name]
            .shape
            .get(1)
            .copied()
            .map(|n| n / 2)
    } else {
        file.tensors
            .get(width_of)
            .and_then(|t| t.shape.get(1).copied())
    }
    .ok_or_else(|| Error::Msg(format!("MoE model is missing {width_of}")))?;
    let shared_probe = if ungated {
        "ffn_up_shexp"
    } else {
        "ffn_gate_shexp"
    };
    let shared = if file
        .tensors
        .contains_key(&format!("blk.{i}.{shared_probe}.weight"))
    {
        let sff = file.tensors[&format!("blk.{i}.{shared_probe}.weight")].shape[0];
        let sh_up = load_named(
            file,
            &format!("blk.{i}.ffn_up_shexp.weight"),
            &[sff, d_model],
        )?;
        Some(SharedExpert {
            gate: if ungated {
                sh_up.clone()
            } else {
                load_named(
                    file,
                    &format!("blk.{i}.ffn_gate_shexp.weight"),
                    &[sff, d_model],
                )?
            },
            up: sh_up,
            relu_sqr: ungated,
            limit: clamp_limit("swiglu_clamp_shexp"),
            down: load_named(
                file,
                &format!("blk.{i}.ffn_down_shexp.weight"),
                &[d_model, sff],
            )?,
            scales: if arch_spec(&cfg.architecture).is_some_and(|s| s.shexp_scales) {
                let sc = |nm: &str| load_scalar(file, &format!("blk.{i}.ffn_{nm}_shexp.scale"));
                match (sc("gate")?, sc("up")?, sc("down")?) {
                    (None, None, None) => None,
                    (g, u, d) => Some([g.unwrap_or(1.0), u.unwrap_or(1.0), d.unwrap_or(1.0)]),
                }
            } else {
                None
            },
            gate_inp: if file
                .tensors
                .contains_key(&format!("blk.{i}.ffn_gate_inp_shexp.weight"))
            {
                let t = file.dequantize_tensor(&format!("blk.{i}.ffn_gate_inp_shexp.weight"))?;
                if t.numel() != d_model {
                    return Err(Error::Msg(format!(
                        "ffn_gate_inp_shexp has {} entries, expected {d_model}",
                        t.numel()
                    )));
                }
                Some(t)
            } else {
                None
            },
        })
    } else {
        None
    };
    let expert_scales = |name: &str| {
        if arch_spec(&cfg.architecture).is_some_and(|s| s.expert_scales) {
            load_expert_scales(file, name, e)
        } else {
            Ok(None)
        }
    };
    let exp_up = if fused {
        load_named(file, &fused_name, &[e, 2 * ff, d])?
            .narrow(1, ff, ff)?
            .contiguous()?
    } else {
        load_named(file, &up_name, &[e, ff, d])?
    };
    let post_exp_norm = match (spec.spark_blocks, &latent) {
        (true, Some(l)) => {
            let w = load_named(
                file,
                &format!("blk.{i}.ffn_post_exp_norm.weight"),
                &[l.down.shape()[0]],
            )?;
            Some((w, cfg.post_norm_eps.unwrap_or(cfg.rms_eps)))
        }
        _ => None,
    };
    Ok(MoeFfn {
        post_exp_norm,
        router: load_named(file, &format!("blk.{i}.ffn_gate_inp.weight"), &[e, d_model])?,
        gate: if ungated {
            exp_up.clone()
        } else if fused {
            load_named(file, &fused_name, &[e, 2 * ff, d])?
                .narrow(1, 0, ff)?
                .contiguous()?
        } else {
            load_named(file, &gate_name, &[e, ff, d])?
        },
        up: exp_up,
        down: load_named(file, &format!("blk.{i}.ffn_down_exps.weight"), &[e, d, ff])?,
        top_k: cfg.expert_used,
        norm_topk: cfg.moe_norm_topk,
        weights_scale: cfg.moe_weights_scale,
        shared,
        // Only llama.cpp's `llama` graph passes the per-expert scales on; Qwen2-MoE ignores them.
        gate_scale: expert_scales(&format!("blk.{i}.ffn_gate_exps.scale"))?,
        up_scale: expert_scales(&format!("blk.{i}.ffn_up_exps.scale"))?,
        down_scale: expert_scales(&format!("blk.{i}.ffn_down_exps.scale"))?,
        gating: cfg.moe_gating,
        n_groups: cfg.expert_groups,
        n_group_used: cfg.expert_groups_used,
        router_bias: if spec.moe_biases {
            load_expert_scales(file, &format!("blk.{i}.ffn_gate_inp.bias"), e)?
        } else {
            None
        },
        gate_bias: if spec.moe_biases {
            Some(load_named(
                file,
                &format!("blk.{i}.ffn_gate_exps.bias"),
                &[e, ff],
            )?)
        } else {
            None
        },
        up_bias: if spec.moe_biases {
            Some(load_named(
                file,
                &format!("blk.{i}.ffn_up_exps.bias"),
                &[e, ff],
            )?)
        } else {
            None
        },
        down_bias: if spec.moe_biases {
            Some(load_named(
                file,
                &format!("blk.{i}.ffn_down_exps.bias"),
                &[e, d],
            )?)
        } else {
            None
        },
        act: match cfg.activation {
            Activation::Situ(b, l) => MoeAct::Situ(b, l),
            _ => spec.moe_act,
        },
        limit: moe_limit,
        selection_bias: load_expert_scales(
            file,
            &if file.tensors.contains_key(&format!("blk.{i}.exp_probs_b")) {
                format!("blk.{i}.exp_probs_b")
            } else {
                format!("blk.{i}.exp_probs_b.bias")
            },
            e,
        )?,
        select_on_logits: arch_spec(&cfg.architecture).is_some_and(|s| s.select_on_logits),
        weight_before_ffn: arch_spec(&cfg.architecture).is_some_and(|s| s.weight_before_ffn),
        latent,
        chunk: if file
            .tensors
            .contains_key(&format!("blk.{i}.ffn_gate_chexps.weight"))
        {
            let group = file
                .get_metadata_u32(&format!("{}.experts_per_group", cfg.architecture))
                .ok_or_else(|| Error::Msg("chunk experts need experts_per_group".into()))?
                as usize;
            let n_chunk = file.tensors[&format!("blk.{i}.ffn_gate_chexps.weight")].shape[0];
            let cff = file.tensors[&format!("blk.{i}.ffn_gate_chexps.weight")].shape[1];
            Some(ChunkExperts {
                gate: load_named(
                    file,
                    &format!("blk.{i}.ffn_gate_chexps.weight"),
                    &[n_chunk, cff, d],
                )?,
                up: load_named(
                    file,
                    &format!("blk.{i}.ffn_up_chexps.weight"),
                    &[n_chunk, cff, d],
                )?,
                down: load_named(
                    file,
                    &format!("blk.{i}.ffn_down_chexps.weight"),
                    &[n_chunk, d, cff],
                )?,
                group: group.max(1),
                scale: file
                    .get_metadata_f32(&format!("{}.expert_group_scale", cfg.architecture))
                    .unwrap_or(1.0),
            })
        } else {
            None
        },
    })
}

/// Longest context the rotary tables are built for when loading a file (a 1M-token model would
/// otherwise need hundreds of MB of cos/sin tables); generation stops at `config.max_seq_len`.
pub const MAX_ROPE_POSITIONS: usize = 131_072;

/// Cos/sin tables for `n_rot` rotary dims under `scaling`, optionally with per-pair frequency divisors.
fn rope_table(
    n_rot: usize,
    max_seq_len: usize,
    base: f64,
    scaling: &RopeScaling,
    factors: Option<&[f64]>,
    attn_factor: f64,
) -> Result<RotaryEmbedding> {
    rope_table_masked(n_rot, max_seq_len, base, scaling, factors, attn_factor, &[])
}

/// [`rope_table`] where the pairs flagged in `zero_pairs` are evaluated at position 0 (M-RoPE's text-only fourth channel).
fn rope_table_masked(
    n_rot: usize,
    max_seq_len: usize,
    base: f64,
    scaling: &RopeScaling,
    factors: Option<&[f64]>,
    attn_factor: f64,
    zero_pairs: &[bool],
) -> Result<RotaryEmbedding> {
    // YaRN folds `attn_factor` into its own magnitude term; for every other scheme llama.cpp still
    // multiplies cos/sin by it.
    let mscale = if matches!(scaling, RopeScaling::Yarn { .. }) {
        1.0
    } else {
        attn_factor
    };
    RotaryEmbedding::from_fn(n_rot, max_seq_len, |pos, i| {
        let pos = if zero_pairs.get(i).copied().unwrap_or(false) {
            0
        } else {
            pos
        };
        let (c, s) = scaling.angle(n_rot, base, pos, i, factors.map_or(1.0, |f| f[i]));
        (c * mscale, s * mscale)
    })
}

/// Per-pair frequency divisors for the rotary tables: `rope_freqs.weight` (Llama-3.1 style) if present,
/// else LongRoPE's `rope_factors_long` when the context exceeds the original one, else `rope_factors_short`.
fn rope_factors(file: &GGUFFile, cfg: &GGUFModelConfig) -> Result<Option<Vec<f64>>> {
    // CogVLM's graph calls plain `ggml_rope`, which takes no frequency factors, so the stored tensor is dead weight.
    if arch_spec(&cfg.architecture).is_some_and(|s| s.ignore_rope_freqs) {
        return Ok(None);
    }
    let name = if file.tensors.contains_key("rope_freqs.weight") {
        "rope_freqs.weight"
    } else if cfg.max_seq_len > cfg.rope_orig_ctx
        && file.tensors.contains_key("rope_factors_long.weight")
    {
        "rope_factors_long.weight"
    } else if cfg.max_seq_len <= cfg.rope_orig_ctx
        && file.tensors.contains_key("rope_factors_short.weight")
    {
        "rope_factors_short.weight"
    } else {
        return Ok(None);
    };
    let f = file.dequantize_tensor(name)?.to_vec::<f32>()?;
    if f.len() != cfg.rope_n_rot / 2 {
        return Err(Error::Msg(format!(
            "{name} has {} entries, expected {}",
            f.len(),
            cfg.rope_n_rot / 2
        )));
    }
    Ok(Some(f.into_iter().map(f64::from).collect()))
}

/// Rotary tables honoring the file's `rope.scaling.*` keys and frequency factors. `unscaled` ignores
/// both (Gemma 3's local layers).
fn build_rope(
    file: &GGUFFile,
    cfg: &GGUFModelConfig,
    base: f64,
    unscaled: bool,
    n_rot: usize,
) -> Result<RotaryEmbedding> {
    let factors = if unscaled {
        None
    } else {
        rope_factors(file, cfg)?
    };
    let scaling = if unscaled {
        &RopeScaling::None
    } else {
        &cfg.rope_scaling
    };
    let attn_factor = if unscaled { 1.0 } else { cfg.rope_attn_factor };
    rope_table_masked(
        n_rot,
        cfg.max_seq_len,
        base,
        scaling,
        factors.as_deref(),
        attn_factor,
        &cfg.rope_zero_pairs,
    )
}

/// Dequantizes `name` from `file` and checks it has exactly `expected_shape`, so a config/metadata
/// mismatch or a wrong GGUF file fails loudly here instead of producing silently-wrong inference.
pub(crate) fn load_named(file: &GGUFFile, name: &str, expected_shape: &[usize]) -> Result<Tensor> {
    let t = file.dequantize_tensor(name)?;
    if t.shape() != expected_shape {
        return Err(Error::Msg(format!(
            "GGUF tensor '{name}' has shape {:?}, expected {:?} from model config",
            t.shape(),
            expected_shape
        )));
    }
    Ok(t)
}

#[cfg(test)]
mod from_gguf_tests {
    use super::*;
    use crate::serialize::gguf::{GGMLType, GGUFValue, serialize_gguf};
    use std::collections::HashMap;

    /// A quantized per-layer table gathered row by row equals the same rows of its full dequantization.
    #[test]
    fn lazy_rows_match_full_dequantization() -> Result<()> {
        let (rows, cols) = (10usize, 64usize);
        let data: Vec<f32> = (0..rows * cols)
            .map(|i| ((i * 37 % 101) as f32 - 50.0) / 17.0)
            .collect();
        let table = Tensor::from_slice(&data, &[rows, cols])?;
        let name = "per_layer_token_embd.weight";
        for ty in [GGMLType::Q8_0, GGMLType::Q4_0, GGMLType::F32] {
            let bytes = serialize_gguf(&HashMap::new(), &[(name, ty, &table)], 32)?;
            let file = GGUFFile::from_bytes(&bytes)?;
            let full = file.dequantize_tensor(name)?.to_vec::<f32>()?;
            let lazy = LazyRows::from_file(&file, name, rows, cols)?;
            let ids = Tensor::from_slice(&[3i64, 7, 0, 9], &[2, 2])?;
            let got = lazy.forward(&ids)?;
            assert_eq!(got.shape(), &[2, 2, cols]);
            let got = got.to_vec::<f32>()?;
            for (k, &id) in [3usize, 7, 0, 9].iter().enumerate() {
                assert_eq!(
                    &got[k * cols..(k + 1) * cols],
                    &full[id * cols..(id + 1) * cols],
                    "{ty:?} row {id}"
                );
            }
            assert!(lazy.forward(&Tensor::from_slice(&[10i64], &[1])?).is_err());
        }
        Ok(())
    }

    /// `from_gguf` must actually load tensor data, not just shape the model from metadata and
    /// leave it randomly initialized (what `from_config` alone does). Builds a tiny synthetic
    /// GGUF file with real llama.cpp tensor names via `serialize_gguf` (so this exercises the
    /// real GGUF parser, not a hand-rolled stub), then checks every loaded weight is bit-exact
    /// to what was written, not merely the right shape.
    #[test]
    fn from_gguf_loads_real_weights_not_random_init() -> Result<()> {
        let (vocab, d_model, heads, layers) = (8usize, 4usize, 2usize, 1usize);
        let intermediate = (d_model * 8) / 3;

        let mut metadata = HashMap::new();
        metadata.insert(
            "general.architecture".into(),
            GGUFValue::String("llama".into()),
        );
        metadata.insert("llama.vocab_size".into(), GGUFValue::Uint32(vocab as u32));
        metadata.insert(
            "llama.embedding_length".into(),
            GGUFValue::Uint32(d_model as u32),
        );
        metadata.insert(
            "llama.attention.head_count".into(),
            GGUFValue::Uint32(heads as u32),
        );
        metadata.insert("llama.block_count".into(), GGUFValue::Uint32(layers as u32));
        metadata.insert("llama.context_length".into(), GGUFValue::Uint32(64));

        // Deterministic, distinct values per tensor so a mix-up (wrong name/shape) is detectable.
        let mut next = 0.0f32;
        let mut make = |shape: &[usize]| -> Result<Tensor> {
            let n: usize = shape.iter().product();
            let vals: Vec<f32> = (0..n)
                .map(|_| {
                    next += 1.0;
                    next
                })
                .collect();
            Tensor::from_slice(&vals, shape)
        };

        let tok_emb = make(&[vocab, d_model])?;
        let attn_norm = make(&[d_model])?;
        let q = make(&[d_model, d_model])?;
        let k = make(&[d_model, d_model])?;
        let v = make(&[d_model, d_model])?;
        let o = make(&[d_model, d_model])?;
        let ffn_norm = make(&[d_model])?;
        let gate = make(&[intermediate, d_model])?;
        let up = make(&[intermediate, d_model])?;
        let down = make(&[d_model, intermediate])?;
        let out_norm = make(&[d_model])?;
        let output = make(&[vocab, d_model])?;

        let tensors = vec![
            ("token_embd.weight", GGMLType::F32, &tok_emb),
            ("blk.0.attn_norm.weight", GGMLType::F32, &attn_norm),
            ("blk.0.attn_q.weight", GGMLType::F32, &q),
            ("blk.0.attn_k.weight", GGMLType::F32, &k),
            ("blk.0.attn_v.weight", GGMLType::F32, &v),
            ("blk.0.attn_output.weight", GGMLType::F32, &o),
            ("blk.0.ffn_norm.weight", GGMLType::F32, &ffn_norm),
            ("blk.0.ffn_gate.weight", GGMLType::F32, &gate),
            ("blk.0.ffn_up.weight", GGMLType::F32, &up),
            ("blk.0.ffn_down.weight", GGMLType::F32, &down),
            ("output_norm.weight", GGMLType::F32, &out_norm),
            ("output.weight", GGMLType::F32, &output),
        ];

        let bytes = serialize_gguf(&metadata, &tensors, 32)?;
        let file = GGUFFile::from_bytes(&bytes)?;
        let model = GGUFCausalLM::from_gguf(&file)?;

        assert_eq!(
            model.tok_emb.weight.to_vec::<f32>()?,
            tok_emb.to_vec::<f32>()?
        );
        // `llama` Q/K rows are re-ordered from interleaved to half-split RoPE layout on load.
        assert_eq!(
            model.blocks[0].q_proj.weight.to_vec::<f32>()?,
            interleaved_to_half_split(&q, heads)?.to_vec::<f32>()?
        );
        assert_eq!(
            model.blocks[0].v_proj.weight.to_vec::<f32>()?,
            v.to_vec::<f32>()?
        );
        assert_eq!(
            model.blocks[0].down_proj.weight.to_vec::<f32>()?,
            down.to_vec::<f32>()?
        );
        assert_eq!(
            model.norm.weight.to_vec::<f32>()?,
            out_norm.to_vec::<f32>()?
        );
        assert_eq!(
            model.lm_head.weight.to_vec::<f32>()?,
            output.to_vec::<f32>()?
        );

        // And the loaded weights actually run: forward() shouldn't error on real data.
        let input_ids = Tensor::from_slice(&[1i64, 2, 3], &[1, 3])?;
        let logits = model.forward(&input_ids)?;
        assert_eq!(logits.shape(), &[1, 3, vocab]);

        Ok(())
    }

    #[test]
    fn interleaved_to_half_split_reorders_rows_per_head() -> Result<()> {
        // 1 head, head_dim 4: interleaved rows [r0 r1 r2 r3] pair as (r0,r1),(r2,r3);
        // half-split wants (r0,r2),(r1,r3) => order r0 r2 r1 r3.
        let w = Tensor::from_slice(&[0.0f32, 1.0, 2.0, 3.0], &[4, 1])?;
        assert_eq!(
            interleaved_to_half_split(&w, 1)?.to_vec::<f32>()?,
            vec![0.0, 2.0, 1.0, 3.0]
        );
        // 2 heads reorder independently.
        let w = Tensor::from_slice(&[0.0f32, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0], &[8, 1])?;
        assert_eq!(
            interleaved_to_half_split(&w, 2)?.to_vec::<f32>()?,
            vec![0.0, 2.0, 1.0, 3.0, 4.0, 6.0, 5.0, 7.0]
        );
        Ok(())
    }

    #[test]
    fn tokenizer_merges_by_score_falls_back_to_bytes_and_round_trips() -> Result<()> {
        let tokens = [
            "<unk>",
            "<s>",
            "</s>",
            "\u{2581}",
            "a",
            "b",
            "\u{2581}a",
            "ab",
            "\u{2581}ab",
            "<0x21>",
        ];
        let scores = [0.0f32, 0.0, 0.0, -9.0, -9.0, -9.0, -1.0, -2.0, -0.5, 0.0];
        let mut metadata = HashMap::new();
        metadata.insert(
            "tokenizer.ggml.tokens".to_string(),
            GGUFValue::Array(
                tokens
                    .iter()
                    .map(|s| GGUFValue::String(s.to_string()))
                    .collect(),
            ),
        );
        metadata.insert(
            "tokenizer.ggml.scores".to_string(),
            GGUFValue::Array(scores.iter().map(|&s| GGUFValue::Float32(s)).collect()),
        );
        metadata.insert(
            "tokenizer.ggml.bos_token_id".to_string(),
            GGUFValue::Uint32(1),
        );
        metadata.insert(
            "tokenizer.ggml.eos_token_id".to_string(),
            GGUFValue::Uint32(2),
        );
        let file = GGUFFile::from_bytes(&serialize_gguf(&metadata, &[], 32)?)?;
        let tk = GGUFTokenizer::from_gguf(&file)?;

        // "\u{2581}ab" is reachable by two merge orders; scores must pick the one ending in id 8.
        assert_eq!(tk.encode("ab", false), vec![8]);
        assert_eq!(tk.encode("ab", true), vec![1, 8]);
        // 'a b' -> \u{2581}a \u{2581} b ; '!' is unknown but has a byte token.
        assert_eq!(tk.encode("a!", false), vec![6, 9]);
        assert_eq!(tk.decode(&[1, 8, 2]), "ab");
        assert_eq!(tk.decode(&tk.encode("a!", false)), "a!");
        assert_eq!(tk.decode(&tk.encode("ab ab", false)), "ab ab");
        assert!(
            GGUFTokenizer::from_gguf(&GGUFFile::from_bytes(&serialize_gguf(
                &HashMap::new(),
                &[],
                32
            )?)?)
            .is_err()
        );
        Ok(())
    }

    /// The KV-cached path must reproduce the full-sequence forward exactly: prefill + one-token
    /// steps give the same logits as running everything at once (with GQA, so the cache stores
    /// un-repeated KV heads).
    fn bpe_file(pre: &str) -> Result<GGUFFile> {
        let (enc, _) = byte_unicode_tables();
        let mut tokens: Vec<String> = enc.iter().map(|c| c.to_string()).collect();
        let g = enc[b' ' as usize]; // 'G-dot' for space
        let merges = [
            "h e",
            "l l",
            "he ll",
            "hell o",
            &format!("{g} w"),
            "o r",
            &format!("{g}w or"),
            "l d",
        ];
        for m in merges {
            tokens.push(m.replace(' ', ""));
        }
        tokens.push("<|bos|>".into());
        let mut md = HashMap::new();
        md.insert(
            "tokenizer.ggml.model".to_string(),
            GGUFValue::String("gpt2".into()),
        );
        md.insert(
            "tokenizer.ggml.pre".to_string(),
            GGUFValue::String(pre.into()),
        );
        md.insert(
            "tokenizer.ggml.tokens".to_string(),
            GGUFValue::Array(
                tokens
                    .iter()
                    .map(|s| GGUFValue::String(s.clone()))
                    .collect(),
            ),
        );
        md.insert(
            "tokenizer.ggml.merges".to_string(),
            GGUFValue::Array(
                merges
                    .iter()
                    .map(|s| GGUFValue::String(s.to_string()))
                    .collect(),
            ),
        );
        md.insert(
            "tokenizer.ggml.bos_token_id".to_string(),
            GGUFValue::Uint32(264),
        );
        GGUFFile::from_bytes(&serialize_gguf(&md, &[], 32)?)
    }

    #[test]
    fn byte_level_bpe_merges_by_rank_and_round_trips() -> Result<()> {
        let tk = GGUFTokenizer::from_gguf(&bpe_file("default")?)?;
        assert!(!tk.add_bos_default);
        // "hello": he(0) ll(1) -> he ll(2) -> hell o(3); " world": G w(4), o r(5), Gw or(6), l d(7).
        assert_eq!(tk.encode("hello world", false), vec![259, 262, 263]);
        assert_eq!(tk.encode("hello", true), vec![264, 259]);
        for text in [
            "hello world",
            "h\u{e9}llo  \u{1F980}!\n",
            "",
            " ",
            "world's end",
        ] {
            assert_eq!(
                tk.decode(&tk.encode(text, false)),
                text,
                "round trip of {text:?}"
            );
        }
        Ok(())
    }

    #[test]
    fn pretokenizer_follows_gpt2_llama3_and_qwen2_regexes() {
        let split = |t: &str, m| pretokenize(t, m);
        assert_eq!(
            split("Hello, world! 123", PreTok::Gpt2),
            ["Hello", ",", " world", "!", " 123"]
        );
        assert_eq!(
            split("Hello, world! 123", PreTok::Llama3),
            ["Hello", ",", " world", "!", " ", "123"]
        );
        assert_eq!(split("12345", PreTok::Llama3), ["123", "45"]);
        assert_eq!(split("12345", PreTok::Qwen2), ["1", "2", "3", "4", "5"]);
        assert_eq!(split("don't", PreTok::Gpt2), ["don", "'t"]);
        assert_eq!(split("DON'T", PreTok::Llama3), ["DON", "'T"]); // case-insensitive contraction
        assert_eq!(split("DON'T", PreTok::Gpt2), ["DON", "'", "T"]); // case-sensitive
        assert_eq!(split("a  b", PreTok::Gpt2), ["a", " ", " b"]);
        assert_eq!(split("a\n\nb", PreTok::Llama3), ["a", "\n\n", "b"]);
        assert_eq!(split("x,y", PreTok::Llama3), ["x", ",y"]); // punctuation sticks to the next word
        assert_eq!(split("a  ", PreTok::Gpt2), ["a", "  "]);
        // Whatever the split, the pieces always concatenate back to the input.
        for s in [
            "",
            "  \t\n x\r\n\r\ny 9 8 7 \u{1F980}\u{e9}'s'S''",
            "\n\n\n",
        ] {
            for m in [PreTok::Gpt2, PreTok::Llama3, PreTok::Qwen2] {
                assert_eq!(split(s, m).concat(), s);
            }
        }
    }

    /// Tiny GGUF with llama.cpp tensor names for `arch` (optionally with Qwen2-style q/k/v biases).
    fn tiny_model_file(arch: &str, biases: bool) -> Result<GGUFFile> {
        let (vocab, d, inter) = (8usize, 4usize, 10usize);
        let mut md = HashMap::new();
        md.insert(
            "general.architecture".into(),
            GGUFValue::String(arch.into()),
        );
        for (k, v) in [
            ("embedding_length", d),
            ("attention.head_count", 2),
            ("block_count", 1),
            ("feed_forward_length", inter),
        ] {
            md.insert(format!("{arch}.{k}"), GGUFValue::Uint32(v as u32));
        }
        let mut next = 0.0f32;
        let mut names: Vec<(String, Vec<usize>)> = vec![
            ("token_embd.weight".into(), vec![vocab, d]),
            ("blk.0.attn_norm.weight".into(), vec![d]),
            ("blk.0.attn_q.weight".into(), vec![d, d]),
            ("blk.0.attn_k.weight".into(), vec![d, d]),
            ("blk.0.attn_v.weight".into(), vec![d, d]),
            ("blk.0.attn_output.weight".into(), vec![d, d]),
            ("blk.0.ffn_norm.weight".into(), vec![d]),
            ("blk.0.ffn_gate.weight".into(), vec![inter, d]),
            ("blk.0.ffn_up.weight".into(), vec![inter, d]),
            ("blk.0.ffn_down.weight".into(), vec![d, inter]),
            ("output_norm.weight".into(), vec![d]),
        ];
        if biases {
            for q in ["q", "k", "v"] {
                names.push((format!("blk.0.attn_{q}.bias"), vec![d]));
            }
        }
        let tensors: Vec<(String, Tensor)> = names
            .into_iter()
            .map(|(n, shape)| {
                let count: usize = shape.iter().product();
                let vals: Vec<f32> = (0..count)
                    .map(|_| {
                        next += 1.0;
                        next * 0.01
                    })
                    .collect();
                Ok((n, Tensor::from_slice(&vals, &shape)?))
            })
            .collect::<Result<_>>()?;
        let refs: Vec<(&str, GGMLType, &Tensor)> = tensors
            .iter()
            .map(|(n, t)| (n.as_str(), GGMLType::F32, t))
            .collect();
        GGUFFile::from_bytes(&serialize_gguf(&md, &refs, 32)?)
    }

    #[test]
    fn qwen2_loads_biases_without_rope_permutation_and_other_archs_are_refused() -> Result<()> {
        let file = tiny_model_file("qwen2", true)?;
        let model = GGUFCausalLM::from_gguf(&file)?;
        let b = |n: &str| file.dequantize_tensor(n).unwrap().to_vec::<f32>().unwrap();
        assert_eq!(
            model.blocks[0]
                .k_proj
                .bias
                .as_ref()
                .unwrap()
                .to_vec::<f32>()?,
            b("blk.0.attn_k.bias")
        );
        assert_eq!(
            model.blocks[0]
                .v_proj
                .bias
                .as_ref()
                .unwrap()
                .to_vec::<f32>()?,
            b("blk.0.attn_v.bias")
        );
        // Qwen2 uses half-split RoPE natively: Q is loaded as stored, not re-ordered.
        assert_eq!(
            model.blocks[0].q_proj.weight.to_vec::<f32>()?,
            b("blk.0.attn_q.weight")
        );
        assert_eq!(
            model
                .forward(&Tensor::from_slice(&[1i64, 2], &[1, 2])?)?
                .shape(),
            &[1, 2, 8]
        );

        // The same file as llama has no biases and gets the llama Q/K reordering.
        let llama = GGUFCausalLM::from_gguf(&tiny_model_file("llama", false)?)?;
        assert!(llama.blocks[0].q_proj.bias.is_none());

        let err = GGUFCausalLM::from_gguf(&tiny_model_file("not-a-real-architecture", false)?)
            .err()
            .unwrap();
        assert!(
            err.to_string().contains("unsupported GGUF architecture"),
            "{err}"
        );
        Ok(())
    }

    #[test]
    fn dsv4_rejects_malformed_cache_and_position() -> Result<()> {
        let attn = Dsv4Attn {
            q_a: Tensor::zeros_f32(&[2, 4])?,
            q_a_norm: Norm::rms(2, 1e-5)?,
            q_b: Tensor::zeros_f32(&[4, 2])?,
            kv: Tensor::zeros_f32(&[2, 4])?,
            kv_norm: Norm::rms(2, 1e-5)?,
            sinks: vec![0.0, 0.0],
            wo_a: Tensor::zeros_f32(&[2, 1, 2])?,
            wo_b: Tensor::zeros_f32(&[4, 2])?,
            heads: 2,
            head_dim: 2,
            rope: 0,
            groups: 2,
            lora: 1,
            window: 8,
            eps: 1e-5,
            sched: Dsv4Rope {
                base: 10000.0,
                scale: 1.0,
                yarn: None,
            },
            comp: None,
            lid: None,
        };
        let x = Tensor::zeros_f32(&[1, 1, 4])?;
        // Rank-2 cache slot: shape()[2] panicked pre-fix.
        let mut bad_rank = Some((Tensor::zeros_f32(&[1, 2])?, Tensor::zeros_f32(&[1, 2])?));
        let err = attn
            .forward(&x, 0, &mut bad_rank)
            .expect_err("rank-2 cache must be an error, not a panic");
        assert!(err.to_string().contains("cache"), "{err}");
        // Right rank, wrong volume: slice copies would panic pre-fix.
        let mut bad_len = Some((
            Tensor::zeros_f32(&[1, 1, 5, 3])?,
            Tensor::zeros_f32(&[1, 5, 1])?,
        ));
        let err = attn
            .forward(&x, 0, &mut bad_len)
            .expect_err("short cache must be an error, not a panic");
        assert!(err.to_string().contains("cache"), "{err}");
        // Position past the cache end: key gathering would panic pre-fix.
        let mut empty = None;
        let err = attn
            .forward(&x, 3, &mut empty)
            .expect_err("seq_pos past the cache must be an error, not a panic");
        assert!(err.to_string().contains("seq_pos"), "{err}");
        // The consistent call still works.
        let mut ok = None;
        assert_eq!(attn.forward(&x, 0, &mut ok)?.shape(), &[1, 1, 4]);
        Ok(())
    }

    #[test]
    fn ssm_config_rejects_degenerate_geometry() -> Result<()> {
        let file = |arch: &str, kv: &[(&str, u32)]| -> Result<GGUFFile> {
            let mut md = HashMap::new();
            md.insert(
                "general.architecture".into(),
                GGUFValue::String(arch.into()),
            );
            for (k, v) in kv {
                md.insert(format!("{arch}.ssm.{k}"), GGUFValue::Uint32(*v));
            }
            GGUFFile::from_bytes(&serialize_gguf(&md, &[], 32)?)
        };
        let base = [("conv_kernel", 4u32), ("inner_size", 8), ("state_size", 8)];
        // time_step_rank = 0: loader divides by it (GatedDelta) at load time.
        let mut kv = base.to_vec();
        kv.extend([("time_step_rank", 0), ("group_count", 1)]);
        let err = GGUFModelConfig::from_gguf(&file("mamba2", &kv)?)
            .expect_err("dt_rank 0 must be an error");
        assert!(err.to_string().contains("time_step_rank"), "{err}");
        // dt_rank % group_count != 0: group index walks out of bounds in the scan.
        let mut kv = base.to_vec();
        kv.extend([("time_step_rank", 2), ("group_count", 4)]);
        let err = GGUFModelConfig::from_gguf(&file("mamba2", &kv)?)
            .expect_err("misaligned groups must be an error");
        assert!(err.to_string().contains("group"), "{err}");
        // Absurd state size: loader shape products overflow.
        let kv = [
            ("conv_kernel", 4u32),
            ("inner_size", 8),
            ("state_size", u32::MAX),
            ("time_step_rank", 2),
        ];
        let err = GGUFModelConfig::from_gguf(&file("mamba2", &kv)?)
            .expect_err("absurd dims must be an error");
        assert!(err.to_string().contains("ssm"), "{err}");
        // conv_kernel = 0: `d_conv - 1` underflows in every scan.
        let kv = [
            ("conv_kernel", 0u32),
            ("inner_size", 8),
            ("state_size", 8),
            ("time_step_rank", 2),
        ];
        let err = GGUFModelConfig::from_gguf(&file("mamba", &kv)?)
            .expect_err("conv_kernel 0 must be an error");
        assert!(err.to_string().contains("conv_kernel"), "{err}");
        Ok(())
    }

    #[test]
    fn ssm_forward_handles_empty_and_rank2_input() -> Result<()> {
        let blk = SsmBlock {
            cfg: SsmConfig {
                kind: SsmKind::Mamba,
                d_conv: 4,
                d_inner: 4,
                d_state: 4,
                dt_rank: 2,
                n_group: 1,
                dt_b_c_rms: false,
            },
            in_proj: Tensor::zeros_f32(&[8, 4])?,
            conv_w: vec![0.0; 16],
            conv_b: vec![0.0; 4],
            x_proj: Some(Tensor::zeros_f32(&[10, 4])?),
            dt_proj: Some(Tensor::zeros_f32(&[4, 2])?),
            dt_bias: vec![0.0; 4],
            a: vec![0.0; 16],
            d: vec![0.0; 4],
            norm_w: None,
            dt_b_c_norm: None,
            out_proj: Tensor::zeros_f32(&[4, 4])?,
            eps: 1e-5,
            in_scale: None,
            out_scale: None,
            gdn: None,
            kda: None,
        };
        // Empty sequence: `proj.len() / (b * s)` divided by zero pre-fix.
        let x0 = Tensor::from_slice(&[] as &[f32], &[1, 0, 4])?;
        assert_eq!(blk.forward(&x0, &mut None)?.shape(), &[1, 0, 4]);
        // Rank-2 input: shape()[2] panicked pre-fix.
        let x2 = Tensor::zeros_f32(&[2, 4])?;
        assert!(blk.forward(&x2, &mut None).is_err());
        Ok(())
    }

    #[test]
    fn forward_rejects_inputs_that_are_not_batch_by_tokens() -> Result<()> {
        let model = GGUFCausalLM::from_gguf(&tiny_model_file("llama", false)?)?;
        let one_d = Tensor::from_slice(&[1i64, 2], &[2])?;
        let err = model
            .forward(&one_d)
            .expect_err("1-D input_ids must be rejected");
        assert!(err.to_string().contains("[batch, new_tokens]"), "{err}");
        let three_d = Tensor::from_slice(&[1i64, 2], &[1, 2, 1])?;
        let err = model
            .forward(&three_d)
            .expect_err("3-D input_ids must be rejected");
        assert!(err.to_string().contains("[batch, new_tokens]"), "{err}");
        Ok(())
    }

    #[test]
    fn forward_cached_rejects_a_malformed_kv_cache() -> Result<()> {
        let model = GGUFCausalLM::from_gguf(&tiny_model_file("llama", false)?)?;
        let ids = Tensor::from_slice(&[1i64, 2], &[1, 2])?;
        let mut cache = model.new_cache();
        cache[0] = Some((
            Tensor::from_slice(&[1.0f32, 2.0], &[1, 2])?,
            Tensor::from_slice(&[1.0f32, 2.0], &[1, 2])?,
        ));
        let err = model
            .forward_cached(&ids, &mut cache)
            .expect_err("a rank-2 cache slot must be an error, not a panic");
        assert!(err.to_string().contains("cache"), "{err}");
        Ok(())
    }

    #[test]
    fn hrm_text_rejects_cycle_counts_beyond_the_slot_table() -> Result<()> {
        let file = |lps: u32, hc: u32, lc: u32| -> Result<GGUFFile> {
            let mut md = HashMap::new();
            md.insert(
                "general.architecture".into(),
                GGUFValue::String("hrm_text".into()),
            );
            for (k, v) in [
                ("hrm.layers_per_stack", lps),
                ("hrm.h_cycles", hc),
                ("hrm.l_cycles", lc),
            ] {
                md.insert(format!("hrm_text.{k}"), GGUFValue::Uint32(v));
            }
            GGUFFile::from_bytes(&serialize_gguf(&md, &[], 32)?)
        };
        // Triple product overflows usize: new_cache and the loader loop would panic or hang.
        let err = GGUFModelConfig::from_gguf(&file(u32::MAX, 1 << 20, 1 << 20)?)
            .expect_err("overflowing HRM cycles must be an error");
        assert!(err.to_string().contains("hrm"), "{err}");
        // No overflow, but 27M cache slots is absurd: refused by the repeats cap.
        let err = GGUFModelConfig::from_gguf(&file(300, 300, 300)?)
            .expect_err("absurd HRM cycles must be an error");
        assert!(err.to_string().contains("hrm"), "{err}");
        // A realistic triple still loads.
        assert!(GGUFModelConfig::from_gguf(&file(4, 2, 2)?).is_ok());
        Ok(())
    }

    #[test]
    fn mla_loader_rejects_head_dims_that_overflow() -> Result<()> {
        let mut md = HashMap::new();
        md.insert(
            "general.architecture".into(),
            GGUFValue::String("deepseek2".into()),
        );
        for (k, v) in [
            ("embedding_length", 4u32),
            ("attention.head_count", u32::MAX),
            ("block_count", 1),
            ("feed_forward_length", 10),
            ("attention.key_length_mla", u32::MAX),
            ("attention.kv_lora_rank", 4),
            ("attention.value_length_mla", 2),
            ("rope.dimension_count", 4),
        ] {
            md.insert(format!("deepseek2.{k}"), GGUFValue::Uint32(v));
        }
        // A present (tiny) attn_q so the loader reaches the expected-size product instead of
        // failing on the missing tensor first; the product itself is what overflows.
        let q = Tensor::from_slice(&[1.0f32], &[1, 1])?;
        let file = GGUFFile::from_bytes(&serialize_gguf(
            &md,
            &[("blk.0.attn_q.weight", GGMLType::F32, &q)],
            32,
        )?)?;
        let cfg = GGUFModelConfig::from_gguf(&file)?;
        let err = match load_mla(&file, 0, &cfg) {
            Ok(_) => panic!("overflowing head dims must be an error, not a panic"),
            Err(e) => e,
        };
        assert!(err.to_string().contains("overflow"), "{err}");
        Ok(())
    }

    #[test]
    fn mla_loader_rejects_a_zero_width_indexer() -> Result<()> {
        // deepseek32 with q_lora_rank omitted (= 0): the single-projection MLA path plus indexer tensors.
        // rope.dimension_count is omitted too (rope_n_rot = 0), so the rotary check passes and the
        // zero key_length is what must be refused.
        let mut md = HashMap::new();
        md.insert(
            "general.architecture".into(),
            GGUFValue::String("deepseek32".into()),
        );
        for (k, v) in [
            ("embedding_length", 4u32),
            ("attention.head_count", 2),
            ("block_count", 1),
            ("feed_forward_length", 10),
            ("attention.key_length_mla", 4),
            ("attention.kv_lora_rank", 4),
            ("attention.value_length_mla", 2),
            ("attention.indexer.head_count", 2),
            ("attention.indexer.key_length", 0),
            ("attention.indexer.top_k", 2),
        ] {
            md.insert(format!("deepseek32.{k}"), GGUFValue::Uint32(v));
        }
        let t = |n: usize, shape: &[usize]| Tensor::from_slice(&vec![0.0f32; n], shape);
        let tensors = [
            ("blk.0.attn_q.weight", t(8 * 4, &[8, 4])?),
            ("blk.0.attn_kv_a_mqa.weight", t(4 * 4, &[4, 4])?),
            ("blk.0.attn_kv_a_norm.weight", t(4, &[4])?),
            ("blk.0.attn_k_b.weight", t(2 * 4 * 4, &[2, 4, 4])?),
            ("blk.0.attn_v_b.weight", t(2 * 2 * 4, &[2, 2, 4])?),
            ("blk.0.indexer.attn_q_b.weight", t(0, &[0, 4])?),
            ("blk.0.indexer.attn_k.weight", t(0, &[0, 4])?),
            ("blk.0.indexer.proj.weight", t(2 * 4, &[2, 4])?),
            ("blk.0.indexer.k_norm.weight", t(0, &[0])?),
            ("blk.0.indexer.k_norm.bias", t(0, &[0])?),
        ];
        let refs: Vec<(&str, GGMLType, &Tensor)> = tensors
            .iter()
            .map(|(n, ten)| (*n, GGMLType::F32, ten))
            .collect();
        let file = GGUFFile::from_bytes(&serialize_gguf(&md, &refs, 32)?)?;
        let cfg = GGUFModelConfig::from_gguf(&file)?;
        let err = match load_mla(&file, 0, &cfg) {
            Ok(_) => panic!("a zero-width indexer must be an error"),
            Err(e) => e,
        };
        assert!(err.to_string().contains("indexer"), "{err}");
        Ok(())
    }

    #[test]
    fn indexer_with_zero_width_is_an_error_not_a_panic() -> Result<()> {
        // Direct construction (all fields are pub): the loader can no longer build this, but
        // inference must still refuse it instead of panicking in the quantizer.
        let ix = DsaIndexer {
            q_b: Tensor::zeros_f32(&[0, 4])?,
            k: Tensor::zeros_f32(&[0, 4])?,
            proj: Tensor::zeros_f32(&[1, 4])?,
            k_norm: Norm::build(true, true, true, 0, 1e-6)?,
            heads: 1,
            dim: 0,
            top_k: 1,
            rot: 0,
            rope_at_end: false,
            norm_pairs: false,
            hadamard: false,
            kpool: None,
        };
        let mla = MlaAttention {
            q_a: None,
            q_a_norm: None,
            q_b: Tensor::zeros_f32(&[2, 4])?,
            kv_a: Tensor::zeros_f32(&[2, 4])?,
            kv_a_norm: Norm::rms(2, 1e-5)?,
            k_b: Tensor::zeros_f32(&[1, 2, 2])?,
            v_b: Tensor::zeros_f32(&[1, 2, 2])?,
            heads: 1,
            nope: 2,
            rope: 0,
            kv_lora: 2,
            v_dim: 2,
            no_rope: true,
            k_norm: None,
            sinks: None,
            indexer: Some(ix),
            share_dsa: false,
        };
        let rope = RotaryEmbedding::from_fn(2, 64, |_, _| (1.0, 0.0))?;
        let x = Tensor::zeros_f32(&[1, 2, 4])?;
        let err = mla
            .forward(&x, &rope, 0, &mut None, 1.0, None, None)
            .expect_err("a zero-width indexer must be an error, not a panic");
        assert!(err.to_string().contains('0'), "{err}");
        Ok(())
    }

    #[test]
    fn rope_scaling_matches_llama_cpp_semantics() -> Result<()> {
        let (dim, base) = (8usize, 10000.0f64);
        let rope = |s: &RopeScaling, ff: f64| {
            RotaryEmbedding::from_fn(dim, 64, |p, i| s.angle(dim, base, p, i, ff)).unwrap()
        };
        // cos / sin of pair 0 at `pos`: rotate x = [1, 0, ...] and read the two halves back.
        let pair0 = |r: &RotaryEmbedding, pos: usize| -> (f32, f32) {
            let mut x = vec![0.0f32; dim];
            x[0] = 1.0;
            let y = r
                .apply(&Tensor::from_slice(&x, &[1, 1, 1, dim]).unwrap(), pos)
                .unwrap()
                .to_vec::<f32>()
                .unwrap();
            (y[0], y[dim / 2])
        };
        let plain = rope(&RopeScaling::None, 1.0);
        let (c, s) = pair0(&plain, 3);
        assert!((c - 3f32.cos()).abs() < 1e-5 && (s - 3f32.sin()).abs() < 1e-5);

        // Linear factor 2: position 6 behaves like position 3 of the unscaled table.
        let lin = rope(&RopeScaling::Linear { factor: 2.0 }, 1.0);
        let (lc, ls) = pair0(&lin, 6);
        assert!((lc - c).abs() < 1e-5 && (ls - s).abs() < 1e-5);
        // ... and a uniform rope_freqs divisor of 2 is the same thing.
        let (fc, fs) = pair0(&rope(&RopeScaling::None, 2.0), 6);
        assert!((fc - c).abs() < 1e-5 && (fs - s).abs() < 1e-5);

        // YaRN keeps the highest-frequency pair (i = 0) un-interpolated and scales magnitude by
        // 1 + 0.1 ln(factor) (times attn_factor).
        let yarn = rope(
            &RopeScaling::Yarn {
                factor: 4.0,
                orig_ctx: 128,
                attn_factor: 1.0,
                beta_fast: 32.0,
                beta_slow: 1.0,
                log_mul: 0.0,
            },
            1.0,
        );
        let m = 1.0 + 0.1 * 4f64.ln();
        let (yc, ys) = pair0(&yarn, 3);
        assert!(
            (yc as f64 - 3f64.cos() * m).abs() < 1e-5 && (ys as f64 - 3f64.sin() * m).abs() < 1e-5
        );
        // The lowest-frequency pair is fully interpolated: angle / factor.
        let low = |r: &RotaryEmbedding, pos| {
            let mut x = vec![0.0f32; dim];
            x[dim / 2 - 1] = 1.0;
            let y = r
                .apply(&Tensor::from_slice(&x, &[1, 1, 1, dim]).unwrap(), pos)
                .unwrap()
                .to_vec::<f32>()
                .unwrap();
            (y[dim / 2 - 1], y[dim - 1])
        };
        let theta = 40.0 * base.powf(-(2.0 * 3.0) / dim as f64) / 4.0;
        let (lc, ls) = low(&yarn, 40);
        assert!(
            (lc as f64 - theta.cos() * m).abs() < 1e-4
                && (ls as f64 - theta.sin() * m).abs() < 1e-4
        );
        Ok(())
    }

    #[test]
    fn from_gguf_reads_rope_scaling_and_rejects_partial_rotary() -> Result<()> {
        let with = |extra: &[(&str, GGUFValue)]| -> Result<GGUFFile> {
            let mut md = HashMap::new();
            md.insert(
                "general.architecture".into(),
                GGUFValue::String("llama".into()),
            );
            for (k, v) in extra {
                md.insert(k.to_string(), v.clone());
            }
            GGUFFile::from_bytes(&serialize_gguf(&md, &[], 32)?)
        };
        let cfg = |f: &GGUFFile| GGUFModelConfig::from_gguf(f);
        assert_eq!(cfg(&with(&[])?)?.rope_scaling, RopeScaling::None);
        let lin = with(&[
            (
                "llama.rope.scaling.type",
                GGUFValue::String("linear".into()),
            ),
            ("llama.rope.scaling.factor", GGUFValue::Float32(4.0)),
        ])?;
        assert_eq!(cfg(&lin)?.rope_scaling, RopeScaling::Linear { factor: 4.0 });
        let yarn = with(&[
            ("llama.rope.scaling.type", GGUFValue::String("yarn".into())),
            ("llama.rope.scaling.factor", GGUFValue::Float32(8.0)),
            (
                "llama.rope.scaling.original_context_length",
                GGUFValue::Uint32(4096),
            ),
        ])?;
        assert!(matches!(
            cfg(&yarn)?.rope_scaling,
            RopeScaling::Yarn { orig_ctx: 4096, .. }
        ));
        // LongRoPE is accepted (its factors come from tensors); an unknown type is still an error.
        let longrope = with(&[(
            "llama.rope.scaling.type",
            GGUFValue::String("longrope".into()),
        )])?;
        assert_eq!(cfg(&longrope)?.rope_scaling, RopeScaling::None);
        let bad = with(&[("llama.rope.scaling.type", GGUFValue::String("bogus".into()))])?;
        assert!(cfg(&bad).is_err());
        Ok(())
    }

    #[test]
    fn sliding_window_limits_attention_in_full_and_cached_decoding() -> Result<()> {
        let mut cfg = GGUFModelConfig::basic(16, 8, 2, 1, 32, 10000.0)?;
        cfg.sliding_window = Some(2);
        let model = GGUFCausalLM::from_config(cfg)?;
        let row = |ids: &[i64]| -> Result<Vec<f32>> {
            let l = model
                .forward(&Tensor::from_slice(ids, &[1, ids.len()])?)?
                .to_vec::<f32>()?;
            Ok(l[(ids.len() - 1) * 16..].to_vec())
        };
        // One layer, window 2: the last position sees tokens 1 and 2 only, never token 0.
        assert_eq!(row(&[1, 5, 7])?, row(&[9, 5, 7])?);
        assert_ne!(row(&[1, 5, 7])?, row(&[1, 6, 7])?);

        // Cached single-token decoding applies the same window.
        let ids = [3i64, 1, 4, 1, 5, 9];
        let full = model
            .forward(&Tensor::from_slice(&ids, &[1, 6])?)?
            .to_vec::<f32>()?;
        let mut cache = model.new_cache();
        let mut got = model
            .forward_cached(&Tensor::from_slice(&ids[..2], &[1, 2])?, &mut cache)?
            .to_vec::<f32>()?;
        for &id in &ids[2..] {
            got.extend(
                model
                    .forward_cached(&Tensor::from_slice(&[id], &[1, 1])?, &mut cache)?
                    .to_vec::<f32>()?,
            );
        }
        for (a, b) in full.iter().zip(&got) {
            assert!((a - b).abs() < 1e-4);
        }
        Ok(())
    }

    #[test]
    fn moe_routes_top_k_and_matches_dense_when_experts_are_identical() -> Result<()> {
        let (d, ff) = (4usize, 6usize);
        let mat = |rows: usize, cols: usize, seed: f32| {
            let v: Vec<f32> = (0..rows * cols)
                .map(|i| ((i as f32 + seed) * 0.37).sin())
                .collect();
            Tensor::from_slice(&v, &[rows, cols]).unwrap()
        };
        let (g, u, dn) = (mat(ff, d, 1.0), mat(ff, d, 2.0), mat(d, ff, 3.0));
        let stack = |t: &Tensor, n: usize| {
            Tensor::cat(
                &vec![t.reshape(&[1, t.shape()[0], t.shape()[1]]).unwrap(); n],
                0,
            )
            .unwrap()
        };
        let x = Tensor::from_slice(&[0.5f32, -1.0, 2.0, 0.25, 1.5, 0.5, -0.5, 1.0], &[1, 2, d])?;
        let dense = swiglu(&x.narrow(1, 0, 1)?.reshape(&[1, d])?, &g, &u, &dn)?.to_vec::<f32>()?;

        // Four identical experts + any router: renormalized top-2 weights sum to 1 => the dense FFN.
        let moe = MoeFfn {
            router: mat(4, d, 5.0),
            gate: stack(&g, 4),
            up: stack(&u, 4),
            down: stack(&dn, 4),
            top_k: 2,
            norm_topk: true,
            weights_scale: 1.0,
            shared: None,
            gate_scale: None,
            up_scale: None,
            down_scale: None,
            gating: Gating::Softmax,
            selection_bias: None,
            select_on_logits: false,
            weight_before_ffn: false,
            chunk: None,
            post_exp_norm: None,
            latent: None,
            n_groups: 0,
            n_group_used: 0,
            router_bias: None,
            gate_bias: None,
            up_bias: None,
            down_bias: None,
            act: MoeAct::Silu,
            limit: 0.0,
        };
        let out = moe.forward(&x)?;
        assert_eq!(out.shape(), &[1, 2, d]);
        for (a, b) in out.to_vec::<f32>()?[..d].iter().zip(&dense) {
            assert!((a - b).abs() < 1e-5, "{a} vs {b}");
        }

        // Distinct experts, top-1: a router that saturates on one expert selects exactly that expert.
        let (g2, u2, dn2) = (mat(ff, d, 7.0), mat(ff, d, 8.0), mat(d, ff, 9.0));
        let router = Tensor::from_slice(&[0.0f32, 0.0, 0.0, 0.0, 50.0, 0.0, 0.0, 0.0], &[2, d])?; // expert 1 wins iff x[0] > 0
        let moe = MoeFfn {
            router,
            gate: Tensor::cat(&[g.reshape(&[1, ff, d])?, g2.reshape(&[1, ff, d])?], 0)?,
            up: Tensor::cat(&[u.reshape(&[1, ff, d])?, u2.reshape(&[1, ff, d])?], 0)?,
            down: Tensor::cat(&[dn.reshape(&[1, d, ff])?, dn2.reshape(&[1, d, ff])?], 0)?,
            top_k: 1,
            norm_topk: true,
            weights_scale: 1.0,
            shared: None,
            gate_scale: None,
            up_scale: None,
            down_scale: None,
            gating: Gating::Softmax,
            selection_bias: None,
            select_on_logits: false,
            weight_before_ffn: false,
            chunk: None,
            post_exp_norm: None,
            latent: None,
            n_groups: 0,
            n_group_used: 0,
            router_bias: None,
            gate_bias: None,
            up_bias: None,
            down_bias: None,
            act: MoeAct::Silu,
            limit: 0.0,
        };
        let x = Tensor::from_slice(&[1.0f32, 0.5, -0.5, 2.0], &[1, d])?; // x[0] > 0 -> expert 1
        let want = swiglu(&x, &g2, &u2, &dn2)?.to_vec::<f32>()?;
        for (a, b) in moe.forward(&x)?.to_vec::<f32>()?.iter().zip(&want) {
            assert!((a - b).abs() < 1e-5);
        }
        Ok(())
    }

    #[test]
    fn moe_shared_expert_is_added_with_a_sigmoid_gate_and_no_renorm() -> Result<()> {
        let (d, ff) = (4usize, 6usize);
        let mat = |rows: usize, cols: usize, seed: f32| {
            let v: Vec<f32> = (0..rows * cols)
                .map(|i| ((i as f32 + seed) * 0.41).cos())
                .collect();
            Tensor::from_slice(&v, &[rows, cols]).unwrap()
        };
        let (g, u, dn) = (mat(ff, d, 1.0), mat(ff, d, 2.0), mat(d, ff, 3.0));
        let (sg, su, sd) = (mat(3, d, 4.0), mat(3, d, 5.0), mat(d, 3, 6.0));
        let gate_inp = Tensor::from_slice(&[0.3f32, -0.2, 0.5, 0.1], &[d])?;
        let one = |t: &Tensor, a: usize, b: usize| t.reshape(&[1, a, b]).unwrap();
        let moe = MoeFfn {
            router: Tensor::from_slice(&[0.0f32; 4], &[1, d])?, // a single expert: probability 1
            gate: one(&g, ff, d),
            up: one(&u, ff, d),
            down: one(&dn, d, ff),
            top_k: 1,
            norm_topk: false,
            weights_scale: 1.0,
            shared: Some(SharedExpert {
                gate: sg.clone(),
                up: su.clone(),
                down: sd.clone(),
                gate_inp: Some(gate_inp.clone()),
                scales: None,
                relu_sqr: false,
                limit: 0.0,
            }),
            gate_scale: None,
            up_scale: None,
            down_scale: None,
            gating: Gating::Softmax,
            selection_bias: None,
            select_on_logits: false,
            weight_before_ffn: false,
            chunk: None,
            post_exp_norm: None,
            latent: None,
            n_groups: 0,
            n_group_used: 0,
            router_bias: None,
            gate_bias: None,
            up_bias: None,
            down_bias: None,
            act: MoeAct::Silu,
            limit: 0.0,
        };
        let x = Tensor::from_slice(&[0.7f32, -0.4, 1.1, 0.2], &[1, d])?;
        let routed = swiglu(&x, &g, &u, &dn)?;
        let shared = swiglu(&x, &sg, &su, &sd)?;
        let gate = x.matmul(&gate_inp.reshape(&[d, 1])?)?.sigmoid()?;
        let want = routed.add(&shared.mul(&gate)?)?.to_vec::<f32>()?;
        for (a, b) in moe.forward(&x)?.to_vec::<f32>()?.iter().zip(&want) {
            assert!((a - b).abs() < 1e-5, "{a} vs {b}");
        }
        Ok(())
    }

    /// Group-limited routing: the best single expert (0) sits in a weak group; the strongest *group* (2, 3) must win.
    #[test]
    fn group_limited_routing_picks_the_best_group_first() -> Result<()> {
        let (d, ff) = (4usize, 6usize);
        let mat = |rows: usize, cols: usize, seed: f32| {
            let v: Vec<f32> = (0..rows * cols)
                .map(|i| ((i as f32 + seed) * 0.37).sin())
                .collect();
            Tensor::from_slice(&v, &[rows, cols]).unwrap()
        };
        let (g, u, dn) = (mat(ff, d, 1.0), mat(ff, d, 2.0), mat(d, ff, 3.0));
        let stack = |t: &Tensor, scales: &[f64]| {
            let parts: Vec<Tensor> = scales
                .iter()
                .map(|s| {
                    t.mul_scalar(*s)
                        .unwrap()
                        .reshape(&[1, t.shape()[0], t.shape()[1]])
                        .unwrap()
                })
                .collect();
            Tensor::cat(&parts, 0).unwrap()
        };
        // Expert e returns (e + 1) x the dense FFN output, so the output reveals which expert ran.
        let build = |groups: usize, used: usize| -> Result<MoeFfn> {
            Ok(MoeFfn {
                // logits for x = [1,0,0,0]: [3, 0, 2.9, 2.9] -> expert 0 alone is best, group {2,3} is strongest
                router: Tensor::from_slice(
                    &[
                        3.0f32, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.9, 0.0, 0.0, 0.0, 2.9, 0.0,
                        0.0, 0.0,
                    ],
                    &[4, d],
                )?,
                gate: stack(&g, &[1.0; 4]),
                up: stack(&u, &[1.0; 4]),
                down: stack(&dn, &[1.0, 2.0, 3.0, 4.0]),
                top_k: 1,
                norm_topk: true,
                weights_scale: 1.0,
                shared: None,
                gate_scale: None,
                up_scale: None,
                down_scale: None,
                gating: Gating::Softmax,
                selection_bias: None,
                select_on_logits: false,
                weight_before_ffn: false,
                chunk: None,
                post_exp_norm: None,
                latent: None,
                n_groups: groups,
                n_group_used: used,
                router_bias: None,
                gate_bias: None,
                up_bias: None,
                down_bias: None,
                act: MoeAct::Silu,
                limit: 0.0,
            })
        };
        let x = Tensor::from_slice(&[1.0f32, 0.0, 0.0, 0.0], &[1, d])?;
        let dense = swiglu(&x, &g, &u, &dn)?.to_vec::<f32>()?;
        let ratio = |m: &MoeFfn| -> Result<f32> {
            let out = m.forward(&x)?.to_vec::<f32>()?;
            let i = dense
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.abs().partial_cmp(&b.1.abs()).unwrap())
                .unwrap()
                .0;
            Ok(out[i] / dense[i])
        };
        assert!((ratio(&build(0, 0)?)? - 1.0).abs() < 1e-4); // ungrouped: expert 0
        assert!((ratio(&build(2, 1)?)? - 3.0).abs() < 1e-4); // grouped: group {2,3} wins, tie -> expert 2
        assert!((ratio(&build(2, 2)?)? - 1.0).abs() < 1e-4); // both groups allowed: back to expert 0
        Ok(())
    }

    #[test]
    fn mixtral_style_file_loads_experts_and_runs() -> Result<()> {
        let (vocab, d, ff, e) = (8usize, 4usize, 6usize, 3usize);
        let mut md = HashMap::new();
        md.insert(
            "general.architecture".into(),
            GGUFValue::String("llama".into()),
        );
        for (k, v) in [
            ("embedding_length", d),
            ("attention.head_count", 2),
            ("block_count", 1),
            ("expert_count", e),
            ("expert_used_count", 2),
        ] {
            md.insert(format!("llama.{k}"), GGUFValue::Uint32(v as u32));
        }
        let mut n = 0.0f32;
        let mut mk = |shape: &[usize]| {
            let c: usize = shape.iter().product();
            let v: Vec<f32> = (0..c)
                .map(|_| {
                    n += 1.0;
                    (n * 0.173).sin()
                })
                .collect();
            Tensor::from_slice(&v, shape).unwrap()
        };
        let spec: Vec<(&str, Vec<usize>)> = vec![
            ("token_embd.weight", vec![vocab, d]),
            ("blk.0.attn_norm.weight", vec![d]),
            ("blk.0.attn_q.weight", vec![d, d]),
            ("blk.0.attn_k.weight", vec![d, d]),
            ("blk.0.attn_v.weight", vec![d, d]),
            ("blk.0.attn_output.weight", vec![d, d]),
            ("blk.0.ffn_norm.weight", vec![d]),
            ("blk.0.ffn_gate_inp.weight", vec![e, d]),
            ("blk.0.ffn_gate_exps.weight", vec![e, ff, d]),
            ("blk.0.ffn_up_exps.weight", vec![e, ff, d]),
            ("blk.0.ffn_down_exps.weight", vec![e, d, ff]),
            ("output_norm.weight", vec![d]),
        ];
        let tensors: Vec<(&str, Tensor)> = spec.iter().map(|(name, s)| (*name, mk(s))).collect();
        let refs: Vec<(&str, GGMLType, &Tensor)> = tensors
            .iter()
            .map(|(n, t)| (*n, GGMLType::F32, t))
            .collect();
        let file = GGUFFile::from_bytes(&serialize_gguf(&md, &refs, 32)?)?;
        let model = GGUFCausalLM::from_gguf(&file)?;
        let moe = model.blocks[0].moe.as_ref().expect("experts loaded");
        assert_eq!((moe.top_k, moe.gate.shape().to_vec()), (2, vec![e, ff, d]));
        assert!(moe.norm_topk); // Mixtral renormalizes
        let logits = model.forward(&Tensor::from_slice(&[1i64, 2, 3], &[1, 3])?)?;
        assert_eq!(logits.shape(), &[1, 3, vocab]);
        assert!(logits.to_vec::<f32>()?.iter().all(|v| v.is_finite()));
        assert_eq!(model.generate_greedy(&[1, 2], 3, None)?.len(), 5);
        Ok(())
    }

    #[test]
    fn longrope_uses_long_factors_only_beyond_the_original_context() -> Result<()> {
        let half = 4usize;
        let long: Vec<f32> = vec![4.0; half];
        let short: Vec<f32> = vec![1.0; half];
        let (tl, ts) = (
            Tensor::from_slice(&long, &[half])?,
            Tensor::from_slice(&short, &[half])?,
        );
        let tensors = vec![
            ("rope_factors_long.weight", GGMLType::F32, &tl),
            ("rope_factors_short.weight", GGMLType::F32, &ts),
        ];
        let file = GGUFFile::from_bytes(&serialize_gguf(&HashMap::new(), &tensors, 32)?)?;
        let mut cfg = GGUFModelConfig::basic(8, 8, 1, 1, 64, 10000.0)?; // head_dim 8 -> 4 pairs
        cfg.rope_orig_ctx = 64;
        assert_eq!(rope_factors(&file, &cfg)?, Some(vec![1.0; half])); // 64 <= 64: short
        cfg.max_seq_len = 128;
        assert_eq!(rope_factors(&file, &cfg)?, Some(vec![4.0; half])); // beyond: long
        // attn_factor scales cos/sin of plain (non-YaRN) tables.
        let plain = rope_table(8, 4, 10000.0, &RopeScaling::None, None, 1.0)?;
        let scaled = rope_table(8, 4, 10000.0, &RopeScaling::None, None, 2.0)?;
        let probe = |r: &RotaryEmbedding| {
            let mut x = vec![0.0f32; 8];
            x[0] = 1.0;
            r.apply(&Tensor::from_slice(&x, &[1, 1, 1, 8]).unwrap(), 3)
                .unwrap()
                .to_vec::<f32>()
                .unwrap()[0]
        };
        assert!((probe(&scaled) - 2.0 * probe(&plain)).abs() < 1e-5);
        Ok(())
    }

    /// Partial rotary: rotating only the first `n_rot` dims of a head, with the interleaved-pair rows of a
    /// llama-style projection re-ordered by `interleaved_rows`, must equal the textbook adjacent-pair rotation.
    #[test]
    fn partial_rotary_matches_adjacent_pair_rotation() -> Result<()> {
        let (hd, n_rot, pos, base) = (8usize, 4usize, 5usize, 10000.0f64);
        let y: Vec<f32> = (0..hd).map(|i| (i as f32 + 1.0) * 0.7 - 2.0).collect();

        // Reference: adjacent pairs (2i, 2i+1) of the first n_rot dims, theta_i = pos * base^(-2i/n_rot).
        let mut want = y.clone();
        for i in 0..n_rot / 2 {
            let th = pos as f64 * base.powf(-(2.0 * i as f64) / n_rot as f64);
            let (c, s) = (th.cos() as f32, th.sin() as f32);
            want[2 * i] = y[2 * i] * c - y[2 * i + 1] * s;
            want[2 * i + 1] = y[2 * i] * s + y[2 * i + 1] * c;
        }

        // Ours: permute the rotary rows once (as the loader does for weights), then half-split RoPE.
        let rows = Tensor::from_slice(&y, &[hd, 1])?;
        let permuted = interleaved_rows(&rows, 1, hd, 0, n_rot)?.reshape(&[1, 1, 1, hd])?;
        let rope = rope_table(n_rot, 16, base, &RopeScaling::None, None, 1.0)?;
        let got = apply_rope(&rope, &permuted, pos, n_rot)?.to_vec::<f32>()?;
        // Undo the permutation to compare in the original order.
        let mut back = got.clone();
        for i in 0..n_rot / 2 {
            back[2 * i] = got[i];
            back[2 * i + 1] = got[n_rot / 2 + i];
        }
        for (a, b) in back.iter().zip(&want) {
            assert!((a - b).abs() < 1e-5, "{back:?} vs {want:?}");
        }
        // The non-rotary tail is untouched.
        assert_eq!(&got[n_rot..], &y[n_rot..]);
        // And the full-head case agrees with the older whole-head helper.
        let w = Tensor::from_slice(&(0..16).map(|v| v as f32).collect::<Vec<_>>(), &[8, 2])?;
        assert_eq!(
            interleaved_rows(&w, 2, 4, 0, 4)?.to_vec::<f32>()?,
            interleaved_to_half_split(&w, 2)?.to_vec::<f32>()?
        );
        Ok(())
    }

    /// Deci / OpenELM: head counts differ per layer (array-valued metadata).
    #[test]
    fn per_layer_head_counts_build_and_run() -> Result<()> {
        let mut cfg = GGUFModelConfig::basic(16, 8, 4, 3, 32, 10000.0)?;
        cfg.head_dim = 2;
        cfg.rope_n_rot = 2;
        cfg.layer_heads = vec![4, 2, 1];
        cfg.layer_kv_heads = vec![2, 1, 1];
        let model = GGUFCausalLM::from_config(cfg.clone())?;
        let dims: Vec<(usize, usize)> = model
            .blocks
            .iter()
            .map(|b| (b.num_heads, b.num_kv_heads))
            .collect();
        assert_eq!(dims, vec![(4, 2), (2, 1), (1, 1)]);
        assert_eq!(model.blocks[1].q_proj.weight.shape(), &[4, 8]); // 2 heads x head_dim 2
        let logits = model.forward(&Tensor::from_slice(&[1i64, 2, 3, 4], &[1, 4])?)?;
        assert_eq!(logits.shape(), &[1, 4, 16]);
        assert!(logits.to_vec::<f32>()?.iter().all(|v| v.is_finite()));
        // Attention-free layers are refused, not silently mis-run.
        cfg.layer_heads[1] = 0;
        assert!(GGUFCausalLM::from_config(cfg).is_err());
        Ok(())
    }

    /// Nemotron-H MoE layer: sigmoid router on the full-width input, ungated ReLU² experts in a latent space,
    /// and an ungated ReLU² shared expert on the full-width input - checked against plain f32 arithmetic.
    #[test]
    fn ungated_latent_moe_matches_scalar_reference() -> Result<()> {
        let (d, lat, ff, sff, e) = (4usize, 2usize, 3usize, 2usize, 2usize);
        let mut seed = 0.37f32;
        let mut next = move || {
            seed = (seed * 9301.0 + 49297.0) % 233_280.0;
            seed / 233_280.0 - 0.5
        };
        let mut mat = |n: usize| -> Vec<f32> { (0..n).map(|_| next()).collect() };
        let (router, down_w, up_w) = (mat(e * d), mat(lat * d), mat(d * lat));
        let (eup, edown) = (mat(e * ff * lat), mat(e * lat * ff));
        let (sup, sdown) = (mat(sff * d), mat(d * sff));
        let x = mat(d);

        let t = |v: &[f32], s: &[usize]| Tensor::from_slice(v, s);
        let moe = MoeFfn {
            router: t(&router, &[e, d])?,
            gate: t(&eup, &[e, ff, lat])?, // placeholder for the ungated experts
            up: t(&eup, &[e, ff, lat])?,
            down: t(&edown, &[e, lat, ff])?,
            top_k: e,
            norm_topk: false,
            weights_scale: 1.0,
            shared: Some(SharedExpert {
                gate: t(&sup, &[sff, d])?,
                up: t(&sup, &[sff, d])?,
                down: t(&sdown, &[d, sff])?,
                gate_inp: None,
                scales: None,
                relu_sqr: true,
                limit: 0.0,
            }),
            gate_scale: None,
            up_scale: None,
            down_scale: None,
            gating: Gating::Sigmoid,
            selection_bias: None,
            select_on_logits: false,
            weight_before_ffn: false,
            chunk: None,
            post_exp_norm: None,
            latent: Some(LatentProj {
                down: t(&down_w, &[lat, d])?,
                up: t(&up_w, &[d, lat])?,
                pre_norm: None,
                mid_norm: None,
            }),
            n_groups: 0,
            n_group_used: 0,
            router_bias: None,
            gate_bias: None,
            up_bias: None,
            down_bias: None,
            act: MoeAct::ReluSqr,
            limit: 0.0,
        };
        let got = moe.forward(&t(&x, &[1, d])?)?.to_vec::<f32>()?;

        let dot = |a: &[f32], b: &[f32]| a.iter().zip(b).map(|(p, q)| p * q).sum::<f32>();
        let sigmoid = |v: f32| 1.0 / (1.0 + (-v).exp());
        let x_lat: Vec<f32> = (0..lat)
            .map(|j| dot(&down_w[j * d..(j + 1) * d], &x))
            .collect();
        let mut routed = vec![0f32; lat];
        for ex in 0..e {
            let w = sigmoid(dot(&router[ex * d..(ex + 1) * d], &x));
            let h: Vec<f32> = (0..ff)
                .map(|j| {
                    dot(&eup[(ex * ff + j) * lat..(ex * ff + j + 1) * lat], &x_lat)
                        .max(0.0)
                        .powi(2)
                })
                .collect();
            for (o, r) in routed.iter_mut().enumerate() {
                *r += w * dot(&edown[(ex * lat + o) * ff..(ex * lat + o + 1) * ff], &h);
            }
        }
        let sh: Vec<f32> = (0..sff)
            .map(|j| dot(&sup[j * d..(j + 1) * d], &x).max(0.0).powi(2))
            .collect();
        for o in 0..d {
            let want = dot(&up_w[o * lat..(o + 1) * lat], &routed)
                + dot(&sdown[o * sff..(o + 1) * sff], &sh);
            assert!(
                (got[o] - want).abs() < 1e-5,
                "output {o}: {} vs {want}",
                got[o]
            );
        }
        Ok(())
    }

    /// Llama 4 chunked attention: a token only sees its own chunk, with and without the KV cache.
    #[test]
    fn chunked_attention_isolates_chunks() -> Result<()> {
        let mut cfg = GGUFModelConfig::basic(16, 8, 2, 2, 16, 10000.0)?;
        cfg.head_dim = 4;
        cfg.rope_n_rot = 4;
        cfg.sliding_window = Some(4);
        cfg.chunked_attn = true;
        cfg.swa = SwaPattern::All;
        let model = GGUFCausalLM::from_config(cfg)?;
        let run = |first: i64| -> Result<Vec<f32>> {
            let ids = [first, 2, 3, 4, 5, 6, 7, 8];
            model
                .forward(&Tensor::from_slice(&ids, &[1, 8])?)?
                .to_vec::<f32>()
        };
        let (a, b) = (run(1)?, run(9)?);
        let row = |v: &[f32], pos: usize| v[pos * 16..(pos + 1) * 16].to_vec();
        // Token 0 lives in chunk 0: it changes positions 0..4 but must not reach chunk 1 (positions 4..8).
        assert_ne!(row(&a, 3), row(&b, 3));
        for pos in 4..8 {
            assert_eq!(
                row(&a, pos),
                row(&b, pos),
                "position {pos} leaked across the chunk boundary"
            );
        }
        // The cached path (one token at a time) agrees with the full forward.
        let mut cache = model.new_cache();
        let mut got = Vec::new();
        for id in [1i64, 2, 3, 4, 5, 6, 7, 8] {
            got.extend(
                model
                    .forward_cached(&Tensor::from_slice(&[id], &[1, 1])?, &mut cache)?
                    .to_vec::<f32>()?,
            );
        }
        for (x, y) in got.iter().zip(&a) {
            assert!((x - y).abs() < 1e-4, "cached {x} vs full {y}");
        }
        Ok(())
    }

    #[test]
    fn kv_cache_matches_full_forward() -> Result<()> {
        let mut cfg = GGUFModelConfig::basic(16, 8, 4, 2, 32, 10000.0)?;
        cfg.num_kv_heads = 2;
        let model = GGUFCausalLM::from_config(cfg)?;
        let ids = [3i64, 1, 4, 1, 5, 9];
        let full = model
            .forward(&Tensor::from_slice(&ids, &[1, 6])?)?
            .to_vec::<f32>()?;

        let mut cache = model.new_cache();
        let mut got = model
            .forward_cached(&Tensor::from_slice(&ids[..3], &[1, 3])?, &mut cache)?
            .to_vec::<f32>()?;
        for &id in &ids[3..] {
            got.extend(
                model
                    .forward_cached(&Tensor::from_slice(&[id], &[1, 1])?, &mut cache)?
                    .to_vec::<f32>()?,
            );
        }
        assert_eq!(full.len(), got.len());
        for (i, (a, b)) in full.iter().zip(&got).enumerate() {
            assert!((a - b).abs() < 1e-4, "logit {i}: full {a} vs cached {b}");
        }
        // And a multi-token chunk on top of a non-empty cache (offset causal mask).
        let mut cache = model.new_cache();
        model.forward_cached(&Tensor::from_slice(&ids[..2], &[1, 2])?, &mut cache)?;
        let tail = model
            .forward_cached(&Tensor::from_slice(&ids[2..], &[1, 4])?, &mut cache)?
            .to_vec::<f32>()?;
        for (a, b) in full[2 * 16..].iter().zip(&tail) {
            assert!((a - b).abs() < 1e-4);
        }
        Ok(())
    }

    #[test]
    fn generate_greedy_extends_prompt_and_stops_at_eos() -> Result<()> {
        let model = GGUFCausalLM::from_config(GGUFModelConfig::basic(16, 8, 2, 1, 32, 10000.0)?)?;
        let out = model.generate_greedy(&[1, 2, 3], 5, None)?;
        assert_eq!(out.len(), 8);
        assert_eq!(&out[..3], &[1, 2, 3]);
        assert!(out.iter().all(|&t| (0..16).contains(&t)));
        // Deterministic, and the first generated token is the argmax of the last-position logits.
        assert_eq!(out, model.generate_greedy(&[1, 2, 3], 5, None)?);
        // EOS = the first generated token stops after one step.
        assert_eq!(model.generate_greedy(&[1, 2, 3], 5, Some(out[3]))?.len(), 4);
        // The context window bounds generation; an empty prompt is an error.
        assert_eq!(model.generate_greedy(&[1; 30], 10, None)?.len(), 32);
        assert!(model.generate_greedy(&[], 1, None).is_err());
        Ok(())
    }

    /// Causal LM: logits at position i must not depend on tokens after i; and GQA (fewer KV
    /// heads than query heads) with a non-8/3 FFN width must run.
    #[test]
    fn forward_is_causal_and_supports_gqa() -> Result<()> {
        let mut cfg = GGUFModelConfig::basic(16, 8, 4, 2, 32, 10000.0)?;
        cfg.num_kv_heads = 2;
        cfg.intermediate_size = 12;
        let model = GGUFCausalLM::from_config(cfg)?;
        assert_eq!(model.blocks[0].k_proj.weight.shape(), &[4, 8]); // kv_dim = 2 heads * head_dim 2

        let a = Tensor::from_slice(&[1i64, 2, 3, 4], &[1, 4])?;
        let b = Tensor::from_slice(&[1i64, 2, 3, 9], &[1, 4])?; // differs only in the last token
        let (la, lb) = (
            model.forward(&a)?.to_vec::<f32>()?,
            model.forward(&b)?.to_vec::<f32>()?,
        );
        let first3 = 3 * 16;
        for i in 0..first3 {
            assert!(
                (la[i] - lb[i]).abs() < 1e-5,
                "position {} leaked future token",
                i / 16
            );
        }
        assert!(
            la[first3..]
                .iter()
                .zip(&lb[first3..])
                .any(|(x, y)| (x - y).abs() > 1e-6)
        );
        Ok(())
    }

    /// A GGUF file missing a required tensor (e.g. a non-llama architecture, or a truncated
    /// export) must fail loudly, not silently fall back to the random init `from_config` used.
    #[test]
    fn from_gguf_errors_on_missing_tensor() {
        let mut metadata = HashMap::new();
        metadata.insert(
            "general.architecture".into(),
            GGUFValue::String("llama".into()),
        );
        metadata.insert("llama.vocab_size".into(), GGUFValue::Uint32(4));
        metadata.insert("llama.embedding_length".into(), GGUFValue::Uint32(4));
        metadata.insert("llama.attention.head_count".into(), GGUFValue::Uint32(2));
        metadata.insert("llama.block_count".into(), GGUFValue::Uint32(1));
        let bytes = serialize_gguf(&metadata, &[], 32).unwrap();
        let file = GGUFFile::from_bytes(&bytes).unwrap();
        assert!(GGUFCausalLM::from_gguf(&file).is_err());
    }
}
