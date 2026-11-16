//! BERT-family encoders (`bert`, `nomic-bert`, `jina-bert-v2` GGUF architectures), computed the way llama.cpp's `build_bert` does.
//!
//! Tokens are embedded as `tok + type[0] (+ learned position)`, normalized, and run through post-LayerNorm blocks:
//! `x = LN(x + Wo(attn(x)))`, `x = LN(x + ffn(x))` with bidirectional attention. `bert` adds learned positions and
//! has a GELU FFN with biases; `nomic-bert` rotates Q / K (RoPE) instead and has a SwiGLU FFN. Q / K / V may be one
//! fused `attn_qkv` tensor. There is no language-model head: the result is one vector per token, or one pooled vector
//! per sequence.

use super::gguf_model::{Activation, Norm, lin_last, load_bias, load_named, load_norm};
use super::transformer::RotaryEmbedding;
use crate::error::{Error, Result};
use crate::serialize::gguf::GGUFFile;
use crate::tensor::Tensor;

/// `{arch}.pooling_type` (llama.cpp's `llama_pooling_type`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Pooling {
    None,
    Mean,
    Cls,
    Last,
    /// Reranker / classifier: pool (CLS or mean), then the `cls` and `cls.output` projections.
    Rank,
}

struct Proj {
    w: Tensor,
    b: Option<Tensor>,
}

impl Proj {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let y = lin_last(x, &self.w)?;
        match &self.b {
            Some(b) => y.add(b),
            None => Ok(y),
        }
    }
}

/// Separate Q / K / V projections, or one fused `[3d, d]` projection.
enum Qkv {
    Split(Box<[Proj; 3]>),
    Fused(Proj),
}

struct BertLayer {
    qkv: Qkv,
    o: Proj,
    attn_norm: Norm,
    up: Proj,
    /// Separate gate (`nomic-bert` SwiGLU, `jina-bert-v2` GEGLU): `down(act(gate x) * up x)`. Without one, an `up` of
    /// twice the FFN width is a fused GEGLU (`gelu(first half) * second half`) and a plain one is `down(gelu(up x))`.
    gate: Option<Proj>,
    /// LayerNorms over the whole Q / K width (`jina-bert-v2`).
    q_norm: Option<Norm>,
    k_norm: Option<Norm>,
    /// A second post-attention norm that re-adds the layer input first (`jina-bert-v2`).
    norm_2: Option<Norm>,
    ff: usize,
    down: Proj,
    out_norm: Norm,
}

pub struct BertEncoder {
    tok: Tensor,
    types: Option<Tensor>,
    pos: Option<Tensor>,
    rope: Option<RotaryEmbedding>,
    /// ALiBi with llama.cpp's fixed maximum bias of 8 (`jina-bert-v2`).
    alibi: bool,
    gate_act: Activation,
    emb_norm: Norm,
    layers: Vec<BertLayer>,
    heads: usize,
    pub pooling: Pooling,
    /// Rank head: `(cls, activation, norm, cls_out)`; `classifier_mean` selects mean pooling before the head.
    head: Option<RankHead>,
    classifier_mean: bool,
}

/// Classification head of a reranker.
struct RankHead {
    cls: Option<Proj>,
    act: HeadAct,
    norm: Option<Norm>,
    out: Option<Proj>,
}

#[derive(Clone, Copy)]
enum HeadAct {
    Tanh,
    Gelu,
    Silu,
}

impl BertEncoder {
    pub fn from_gguf(file: &GGUFFile) -> Result<Self> {
        let arch = file.get_metadata_str("general.architecture").unwrap_or("");
        let (nomic, jina) = match arch {
            "bert" => (false, false),
            "nomic-bert" => (true, false),
            "jina-bert-v2" => (false, true),
            _ => return Err(Error::Msg(format!("'{arch}' is not a supported BERT encoder"))),
        };
        let u = |k: &str| -> Result<usize> {
            file.get_metadata_u32(&format!("{arch}.{k}"))
                .map(|v| v as usize)
                .ok_or_else(|| Error::Msg(format!("{arch}.{k} is missing")))
        };
        let (d, n_layer, heads) = (
            u("embedding_length")?,
            u("block_count")?,
            u("attention.head_count")?,
        );
        let ff = u("feed_forward_length")?;
        if heads == 0 || d % heads != 0 {
            return Err(Error::Msg(format!("{heads} heads do not divide width {d}")));
        }
        let eps = f64::from(
            file.get_metadata_f32(&format!("{arch}.attention.layer_norm_epsilon"))
                .unwrap_or(1e-12),
        );
        let pooling = match file.get_metadata_u32(&format!("{arch}.pooling_type")) {
            Some(1) => Pooling::Mean,
            Some(2) => Pooling::Cls,
            Some(3) => Pooling::Last,
            Some(4) => Pooling::Rank,
            _ => Pooling::None,
        };
        let rows = |name: &str| -> Result<usize> {
            file.tensors
                .get(name)
                .and_then(|t| t.shape.first().copied())
                .ok_or_else(|| Error::Msg(format!("missing {name}")))
        };
        let vocab = rows("token_embd.weight")?;
        let types = if file.tensors.contains_key("token_types.weight") {
            Some(load_named(
                file,
                "token_types.weight",
                &[rows("token_types.weight")?, d],
            )?)
        } else {
            None
        };
        let (pos, rope) = if nomic {
            let base = file
                .get_metadata_f32(&format!("{arch}.rope.freq_base"))
                .unwrap_or(10_000.0);
            let max = u("context_length")?;
            (
                None,
                Some(RotaryEmbedding::new(d / heads, max, f64::from(base))?),
            )
        } else if jina {
            (None, None) // positions enter through ALiBi
        } else {
            let n = rows("position_embd.weight")?;
            (Some(load_named(file, "position_embd.weight", &[n, d])?), None)
        };
        let norm = |base: &str| -> Result<Norm> {
            let mut n = Norm::build(true, true, true, d, eps)?;
            load_norm(file, base, &mut n)?;
            Ok(n)
        };
        let proj = |base: &str, out: usize, inp: usize| -> Result<Proj> {
            Ok(Proj {
                w: load_named(file, &format!("{base}.weight"), &[out, inp])?,
                b: load_bias(file, &format!("{base}.bias"), out)?,
            })
        };
        let mut layers = Vec::with_capacity(n_layer);
        for i in 0..n_layer {
            let t = |nm: &str| format!("blk.{i}.{nm}");
            let qkv = if file.tensors.contains_key(&t("attn_qkv.weight")) {
                Qkv::Fused(proj(&t("attn_qkv"), 3 * d, d)?)
            } else {
                Qkv::Split(Box::new([
                    proj(&t("attn_q"), d, d)?,
                    proj(&t("attn_k"), d, d)?,
                    proj(&t("attn_v"), d, d)?,
                ]))
            };
            let gate = if file.tensors.contains_key(&t("ffn_gate.weight")) {
                Some(proj(&t("ffn_gate"), ff, d)?)
            } else {
                None
            };
            let opt_norm = |nm: &str| -> Result<Option<Norm>> {
                if file.tensors.contains_key(&t(&format!("{nm}.weight"))) {
                    Ok(Some(norm(&t(nm))?))
                } else {
                    Ok(None)
                }
            };
            let up_rows = rows(&t("ffn_up.weight"))?;
            if up_rows != ff && !(jina && up_rows == 2 * ff) {
                return Err(Error::Msg(format!("ffn_up has {up_rows} rows, expected {ff}")));
            }
            layers.push(BertLayer {
                qkv,
                o: proj(&t("attn_output"), d, d)?,
                attn_norm: norm(&t("attn_output_norm"))?,
                up: proj(&t("ffn_up"), up_rows, d)?,
                gate,
                q_norm: opt_norm("attn_q_norm")?,
                k_norm: opt_norm("attn_k_norm")?,
                norm_2: opt_norm("attn_norm_2")?,
                ff,
                down: proj(&t("ffn_down"), d, ff)?,
                out_norm: norm(&t("layer_output_norm"))?,
            });
        }
        let classifier_mean = file.get_metadata_u32(&format!("{arch}.classifier.pooling_type")) == Some(1);
        let act = match file.get_metadata_str(&format!("{arch}.classifier.activation")) {
            Some("gelu") => HeadAct::Gelu,
            Some("silu") => HeadAct::Silu,
            _ => HeadAct::Tanh,
        };
        let has = |n: &str| file.tensors.contains_key(n);
        let head = if pooling == Pooling::Rank {
            Some(RankHead {
                cls: if has("cls.weight") { Some(proj("cls", d, d)?) } else { None },
                act,
                norm: if has("cls.norm.weight") { Some(norm("cls.norm")?) } else { None },
                out: if has("cls.output.weight") { Some(proj("cls.output", rows("cls.output.weight")?, d)?) } else { None },
            })
        } else {
            None
        };
        Ok(Self {
            tok: load_named(file, "token_embd.weight", &[vocab, d])?,
            types,
            pos,
            rope,
            alibi: jina,
            gate_act: if jina { Activation::GeluTanh } else { Activation::Silu },
            emb_norm: norm("token_embd_norm")?,
            layers,
            heads,
            pooling,
            head,
            classifier_mean,
        })
    }

    /// One hidden vector per token: `[ids.len(), d]`.
    pub fn encode(&self, ids: &[i64]) -> Result<Tensor> {
        let s = ids.len();
        if s == 0 {
            return Err(Error::Msg("no tokens to encode".into()));
        }
        let d = self.tok.shape()[1];
        let idx = Tensor::from_vec(ids.to_vec(), &[s])?;
        let mut x = self.tok.index_select(0, &idx)?;
        if let Some(pos) = &self.pos {
            if s > pos.shape()[0] {
                return Err(Error::Msg(format!(
                    "{s} tokens do not fit the {} learned positions",
                    pos.shape()[0]
                )));
            }
            x = x.add(&pos.narrow(0, 0, s)?)?;
        }
        if let Some(t) = &self.types {
            x = x.add(&t.narrow(0, 0, 1)?)?; // every token is segment 0
        }
        x = self.emb_norm.forward(&x)?;
        let (h, hd) = (self.heads, d / self.heads);
        let scale = 1.0 / (hd as f64).sqrt();
        // ALiBi: head `h` adds `-slope_h * |i - j|` to the scores (ggml's `soft_max_ext` with max_bias = 8).
        let bias = if self.alibi {
            let log2 = 1usize << (usize::BITS - 1 - h.leading_zeros()); // largest power of two <= h
            let (m0, m1) = (
                2f32.powf(-8.0 / log2 as f32),
                2f32.powf(-4.0 / log2 as f32),
            );
            let mut v = Vec::with_capacity(h * s * s);
            for head in 0..h {
                let slope = if head < log2 {
                    m0.powi(head as i32 + 1)
                } else {
                    m1.powi(2 * (head - log2) as i32 + 1)
                };
                for i in 0..s {
                    for j in 0..s {
                        v.push(-slope * i.abs_diff(j) as f32);
                    }
                }
            }
            Some(Tensor::from_vec(v, &[h, s, s])?)
        } else {
            None
        };
        for l in &self.layers {
            let x_in = x.clone();
            // [heads, s, hd]
            let heads_of = |t: &Tensor| -> Result<Tensor> {
                t.reshape(&[s, h, hd])?.transpose(0, 1)?.contiguous()
            };
            let (q, k, v) = match &l.qkv {
                Qkv::Split(p) => (p[0].forward(&x)?, p[1].forward(&x)?, p[2].forward(&x)?),
                Qkv::Fused(p) => {
                    let all = p.forward(&x)?;
                    (
                        all.narrow(1, 0, d)?.contiguous()?,
                        all.narrow(1, d, d)?.contiguous()?,
                        all.narrow(1, 2 * d, d)?.contiguous()?,
                    )
                }
            };
            let q = match &l.q_norm {
                Some(n) => n.forward(&q)?,
                None => q,
            };
            let k = match &l.k_norm {
                Some(n) => n.forward(&k)?,
                None => k,
            };
            let (mut q, mut k, v) = (heads_of(&q)?, heads_of(&k)?, heads_of(&v)?);
            if let Some(rope) = &self.rope {
                let turn = |t: &Tensor| -> Result<Tensor> {
                    rope.apply(&t.reshape(&[1, h, s, hd])?, 0)?
                        .reshape(&[h, s, hd])
                };
                q = turn(&q)?;
                k = turn(&k)?;
            }
            let mut scores = q
                .matmul(&k.transpose(1, 2)?.contiguous()?)?
                .mul_scalar(scale)?;
            if let Some(b) = &bias {
                scores = scores.add(b)?;
            }
            let p = scores.softmax(-1)?;
            let a = p.matmul(&v)?.transpose(0, 1)?.contiguous()?.reshape(&[s, d])?;
            x = l.attn_norm.forward(&l.o.forward(&a)?.add(&x)?)?;
            if let Some(n2) = &l.norm_2 {
                x = n2.forward(&x.add(&x_in)?)?;
            }
            let inner = match &l.gate {
                Some(g) => self
                    .gate_act
                    .apply(&g.forward(&x)?)?
                    .mul(&l.up.forward(&x)?)?,
                None => {
                    let up = l.up.forward(&x)?;
                    if up.shape()[1] == 2 * l.ff {
                        Activation::GeluTanh
                            .apply(&up.narrow(1, 0, l.ff)?.contiguous()?)?
                            .mul(&up.narrow(1, l.ff, l.ff)?.contiguous()?)?
                    } else {
                        Activation::GeluTanh.apply(&up)?
                    }
                }
            };
            x = l.out_norm.forward(&l.down.forward(&inner)?.add(&x)?)?;
        }
        Ok(x)
    }

    /// The sequence embedding under the model's pooling (`[d]`), or every token's vector (`[n, d]`) without pooling.
    pub fn embed(&self, ids: &[i64]) -> Result<Tensor> {
        let x = self.encode(ids)?;
        let s = ids.len();
        match self.pooling {
            Pooling::None => Ok(x),
            Pooling::Mean => x.mean_axis(0, false),
            Pooling::Cls => x.narrow(0, 0, 1)?.reshape(&[x.shape()[1]]),
            Pooling::Last => x.narrow(0, s - 1, 1)?.reshape(&[x.shape()[1]]),
            Pooling::Rank => {
                let d = x.shape()[1];
                let mut cur = if self.classifier_mean { x.mean_axis(0, false)? } else { x.narrow(0, 0, 1)?.reshape(&[d])? };
                if let Some(h) = &self.head {
                    if let Some(c) = &h.cls {
                        cur = c.forward(&cur)?;
                        cur = match h.act {
                            HeadAct::Tanh => cur.tanh()?,
                            HeadAct::Gelu => cur.gelu()?,
                            HeadAct::Silu => cur.silu()?,
                        };
                        if let Some(n) = &h.norm {
                            cur = n.forward(&cur)?;
                        }
                    }
                    if let Some(o) = &h.out {
                        cur = o.forward(&cur)?;
                    }
                }
                Ok(cur)
            }
        }
    }
}
