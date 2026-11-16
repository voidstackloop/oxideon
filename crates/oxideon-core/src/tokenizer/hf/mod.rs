//! A complete, pure-Rust reimplementation of the Hugging Face `tokenizers` pipeline: normalizer -> pre-tokenizer ->
//! model (BPE / WordPiece / Unigram / WordLevel) -> post-processor, plus decoders, added tokens, truncation, padding,
//! offsets and batching. [`Tokenizer::from_file`] reads any `tokenizer.json` (GPT-2, Llama, BERT, RoBERTa, T5, Qwen,
//! Mistral, ...); the same pipeline backs the `vocab.json` + `merges.txt` and `vocab.txt` loaders.

pub mod charsmap;
pub mod decoder;
pub mod model;
pub mod normalizer;
pub mod pretokenizer;
pub mod trainer;

use crate::error::{Error, Result};
use crate::tokenizer::json::Json;
use decoder::Decoder;
use model::Model;
use normalizer::{NormalizedString, Normalizer};
use pretokenizer::PreTokenizer;
use std::collections::HashMap;

pub(crate) fn base64_decode(s: &str) -> Result<Vec<u8>> {
    let mut out = Vec::with_capacity(s.len() * 3 / 4);
    let (mut acc, mut bits) = (0u32, 0);
    for c in s.bytes() {
        let v = match c {
            b'A'..=b'Z' => c - b'A',
            b'a'..=b'z' => c - b'a' + 26,
            b'0'..=b'9' => c - b'0' + 52,
            b'+' | b'-' => 62,
            b'/' | b'_' => 63,
            b'=' | b'\n' | b'\r' | b' ' => continue,
            _ => return Err(Error::Msg("base64: invalid character".into())),
        };
        acc = (acc << 6) | v as u32;
        bits += 6;
        if bits >= 8 {
            bits -= 8;
            out.push((acc >> bits) as u8);
            acc &= (1 << bits) - 1;
        }
    }
    Ok(out)
}

/// Result of tokenizing one sequence or a pair. Offsets are byte ranges into the original text.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Encoding {
    pub ids: Vec<u32>,
    pub type_ids: Vec<u32>,
    pub tokens: Vec<String>,
    pub word_ids: Vec<Option<u32>>,
    pub sequence_ids: Vec<Option<usize>>,
    pub offsets: Vec<(usize, usize)>,
    pub special_tokens_mask: Vec<u32>,
    pub attention_mask: Vec<u32>,
    pub overflowing: Vec<Encoding>,
}

impl Encoding {
    pub fn len(&self) -> usize {
        self.ids.len()
    }
    pub fn is_empty(&self) -> bool {
        self.ids.is_empty()
    }

    #[allow(clippy::too_many_arguments)]
    fn push(&mut self, id: u32, token: String, offsets: (usize, usize), word: Option<u32>, type_id: u32, special: bool, seq: Option<usize>) {
        self.ids.push(id);
        self.tokens.push(token);
        self.offsets.push(offsets);
        self.word_ids.push(word);
        self.type_ids.push(type_id);
        self.special_tokens_mask.push(special as u32);
        self.attention_mask.push(1);
        self.sequence_ids.push(seq);
    }

    fn append(&mut self, mut o: Encoding) {
        self.ids.append(&mut o.ids);
        self.type_ids.append(&mut o.type_ids);
        self.tokens.append(&mut o.tokens);
        self.word_ids.append(&mut o.word_ids);
        self.sequence_ids.append(&mut o.sequence_ids);
        self.offsets.append(&mut o.offsets);
        self.special_tokens_mask.append(&mut o.special_tokens_mask);
        self.attention_mask.append(&mut o.attention_mask);
        self.overflowing.append(&mut o.overflowing);
    }

    fn slice(&self, a: usize, b: usize) -> Encoding {
        Encoding {
            ids: self.ids[a..b].to_vec(),
            type_ids: self.type_ids[a..b].to_vec(),
            tokens: self.tokens[a..b].to_vec(),
            word_ids: self.word_ids[a..b].to_vec(),
            sequence_ids: self.sequence_ids[a..b].to_vec(),
            offsets: self.offsets[a..b].to_vec(),
            special_tokens_mask: self.special_tokens_mask[a..b].to_vec(),
            attention_mask: self.attention_mask[a..b].to_vec(),
            overflowing: vec![],
        }
    }

    /// Keeps `max_len` tokens; the rest becomes overflowing windows overlapping by `stride`.
    fn truncate(&mut self, max_len: usize, stride: usize, dir: Direction) {
        let n = self.len();
        if max_len >= n || max_len == 0 && n == 0 {
            return;
        }
        let step = max_len.saturating_sub(stride).max(1);
        let mut ranges = Vec::new();
        match dir {
            Direction::Right => {
                let mut s = 0;
                loop {
                    let e = (s + max_len).min(n);
                    ranges.push((s, e));
                    if e == n {
                        break;
                    }
                    s += step;
                }
            }
            Direction::Left => {
                let mut e = n;
                loop {
                    let s = e.saturating_sub(max_len);
                    ranges.push((s, e));
                    if s == 0 {
                        break;
                    }
                    e = e.saturating_sub(step).max(max_len.min(e));
                    if e <= s {
                        break;
                    }
                }
            }
        }
        let mut main = self.slice(ranges[0].0, ranges[0].1);
        main.overflowing = ranges[1..].iter().map(|&(a, b)| self.slice(a, b)).collect();
        *self = main;
    }

    fn pad(&mut self, target: usize, pad_id: u32, pad_type_id: u32, pad_token: &str, dir: Direction) {
        if self.len() >= target {
            return;
        }
        let k = target - self.len();
        let pad = Encoding {
            ids: vec![pad_id; k],
            type_ids: vec![pad_type_id; k],
            tokens: vec![pad_token.to_string(); k],
            word_ids: vec![None; k],
            sequence_ids: vec![None; k],
            offsets: vec![(0, 0); k],
            special_tokens_mask: vec![1; k],
            attention_mask: vec![0; k],
            overflowing: vec![],
        };
        match dir {
            Direction::Right => self.append(pad),
            Direction::Left => {
                let mut p = pad;
                let old = std::mem::take(self);
                let overflow = old.overflowing.clone();
                let mut old = old;
                old.overflowing.clear();
                p.append(old);
                p.overflowing = overflow;
                *self = p;
            }
        }
        for o in &mut self.overflowing {
            o.pad(target, pad_id, pad_type_id, pad_token, dir);
        }
    }

    /// Converts the byte offsets into `char` offsets of `text`.
    pub fn char_offsets(&self, text: &str) -> Vec<(usize, usize)> {
        let mut map = vec![0usize; text.len() + 1];
        let mut ci = 0;
        for (bi, c) in text.char_indices() {
            for k in 0..c.len_utf8() {
                map[bi + k] = ci;
            }
            ci += 1;
        }
        map[text.len()] = ci;
        self.offsets.iter().map(|&(a, b)| (map[a.min(text.len())], map[b.min(text.len())])).collect()
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Direction {
    Left,
    Right,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum TruncationStrategy {
    LongestFirst,
    OnlyFirst,
    OnlySecond,
}

#[derive(Debug, Clone)]
pub struct Truncation {
    pub max_length: usize,
    pub strategy: TruncationStrategy,
    pub stride: usize,
    pub direction: Direction,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum PaddingStrategy {
    BatchLongest,
    Fixed(usize),
}

#[derive(Debug, Clone)]
pub struct Padding {
    pub strategy: PaddingStrategy,
    pub direction: Direction,
    pub pad_to_multiple_of: Option<usize>,
    pub pad_id: u32,
    pub pad_type_id: u32,
    pub pad_token: String,
}

fn dir_from(j: Option<&Json>) -> Direction {
    if j.and_then(Json::as_str) == Some("Left") { Direction::Left } else { Direction::Right }
}

#[derive(Debug, Clone)]
pub enum Piece {
    Seq(char, u32),
    Special(String, u32),
}

#[derive(Debug, Clone)]
pub enum PostProcessor {
    Bert { sep: (String, u32), cls: (String, u32) },
    Roberta { sep: (String, u32), cls: (String, u32), trim_offsets: bool, add_prefix_space: bool },
    ByteLevel { trim_offsets: bool, add_prefix_space: bool },
    Template { single: Vec<Piece>, pair: Vec<Piece>, special: HashMap<String, (Vec<u32>, Vec<String>)> },
    Sequence(Vec<PostProcessor>),
}

fn tok_pair(j: Option<&Json>) -> Result<(String, u32)> {
    match j.and_then(Json::as_array) {
        Some([t, i]) => Ok((t.as_str().unwrap_or("").to_string(), i.as_u64().unwrap_or(0) as u32)),
        _ => Err(Error::Msg("tokenizer: expected [token, id]".into())),
    }
}

fn pieces(j: Option<&Json>) -> Result<Vec<Piece>> {
    let mut out = Vec::new();
    for p in j.and_then(Json::as_array).unwrap_or(&[]) {
        if let Some(s) = p.get("Sequence") {
            let c = if s.get("id").and_then(Json::as_str) == Some("B") { 'B' } else { 'A' };
            out.push(Piece::Seq(c, s.get("type_id").and_then(Json::as_u64).unwrap_or(0) as u32));
        } else if let Some(s) = p.get("SpecialToken") {
            out.push(Piece::Special(s.get("id").and_then(Json::as_str).unwrap_or("").to_string(), s.get("type_id").and_then(Json::as_u64).unwrap_or(0) as u32));
        } else {
            return Err(Error::Msg("tokenizer: bad template piece".into()));
        }
    }
    Ok(out)
}

fn trim_offsets_bytelevel(e: &mut Encoding, add_prefix_space: bool) {
    for i in 0..e.len() {
        let tok = &e.tokens[i];
        let is_sp = |c: char| c == '\u{120}' || c.is_whitespace();
        let mut lead = tok.chars().take_while(|&c| is_sp(c)).count();
        let trail = tok.chars().rev().take_while(|&c| is_sp(c)).count();
        let (mut s, mut en) = e.offsets[i];
        if lead > 0 || trail > 0 {
            if lead > 0 {
                let is_first = i == 0 || s == 0;
                if is_first && add_prefix_space && lead == 1 {
                    lead = 0;
                }
                s = en.min(s + lead);
            }
            if trail > 0 && en >= trail {
                en = s.max(en - trail);
            }
        }
        e.offsets[i] = (s, en);
    }
}

impl PostProcessor {
    pub fn from_json(j: &Json) -> Result<PostProcessor> {
        let ty = j.get("type").and_then(Json::as_str).ok_or_else(|| Error::Msg("tokenizer: post_processor without type".into()))?;
        let flag = |k: &str, d: bool| j.get(k).and_then(Json::as_bool).unwrap_or(d);
        Ok(match ty {
            "BertProcessing" => PostProcessor::Bert { sep: tok_pair(j.get("sep"))?, cls: tok_pair(j.get("cls"))? },
            "RobertaProcessing" => PostProcessor::Roberta { sep: tok_pair(j.get("sep"))?, cls: tok_pair(j.get("cls"))?, trim_offsets: flag("trim_offsets", true), add_prefix_space: flag("add_prefix_space", true) },
            "ByteLevel" => PostProcessor::ByteLevel { trim_offsets: flag("trim_offsets", true), add_prefix_space: flag("add_prefix_space", true) },
            "TemplateProcessing" => {
                let mut special = HashMap::new();
                if let Some(o) = j.get("special_tokens").and_then(Json::as_object) {
                    for (k, v) in o {
                        let ids = v.get("ids").and_then(Json::as_array).unwrap_or(&[]).iter().filter_map(|x| x.as_u64().map(|n| n as u32)).collect();
                        let toks = v.get("tokens").and_then(Json::as_array).unwrap_or(&[]).iter().filter_map(|x| x.as_str().map(str::to_string)).collect();
                        special.insert(k.clone(), (ids, toks));
                    }
                }
                PostProcessor::Template { single: pieces(j.get("single"))?, pair: pieces(j.get("pair"))?, special }
            }
            "Sequence" => PostProcessor::Sequence(
                j.get("processors").and_then(Json::as_array).ok_or_else(|| Error::Msg("tokenizer: Sequence needs processors".into()))?.iter().map(PostProcessor::from_json).collect::<Result<_>>()?,
            ),
            other => return Err(Error::Msg(format!("tokenizer: unsupported post-processor {other}"))),
        })
    }

    /// Number of special tokens added around a sequence (or pair).
    pub fn added_tokens(&self, pair: bool) -> usize {
        match self {
            PostProcessor::Bert { .. } => if pair { 3 } else { 2 },
            PostProcessor::Roberta { .. } => if pair { 4 } else { 2 },
            PostProcessor::ByteLevel { .. } => 0,
            PostProcessor::Template { single, pair: p, special } => {
                let t = if pair { p } else { single };
                t.iter().map(|x| if let Piece::Special(n, _) = x { special.get(n).map_or(0, |s| s.0.len()) } else { 0 }).sum()
            }
            PostProcessor::Sequence(v) => v.iter().map(|p| p.added_tokens(pair)).sum(),
        }
    }

    /// Runs the processor over one or two encodings; single processors merge them, a `ByteLevel` one keeps them apart.
    pub fn process_list(&self, encs: Vec<Encoding>, add_special: bool) -> Result<Vec<Encoding>> {
        match self {
            PostProcessor::Sequence(v) => v.iter().try_fold(encs, |e, p| p.process_list(e, add_special)),
            PostProcessor::ByteLevel { trim_offsets, add_prefix_space } => Ok(encs
                .into_iter()
                .enumerate()
                .map(|(k, mut e)| {
                    if *trim_offsets {
                        trim_offsets_bytelevel(&mut e, *add_prefix_space);
                    }
                    e.sequence_ids = vec![Some(k); e.len()];
                    e
                })
                .collect()),
            _ => {
                let mut it = encs.into_iter();
                let a = it.next().unwrap_or_default();
                Ok(vec![self.process_pair(a, it.next(), add_special)?])
            }
        }
    }

    pub fn process(&self, a: Encoding, b: Option<Encoding>, add_special: bool) -> Result<Encoding> {
        let mut list = vec![a];
        list.extend(b);
        let mut out = Encoding::default();
        for e in self.process_list(list, add_special)? {
            out.append(e);
        }
        Ok(out)
    }

    fn process_pair(&self, mut a: Encoding, b: Option<Encoding>, add_special: bool) -> Result<Encoding> {
        let default = |mut a: Encoding, b: Option<Encoding>| -> Encoding {
            a.sequence_ids = vec![Some(0); a.len()];
            if let Some(mut b) = b {
                b.sequence_ids = vec![Some(1); b.len()];
                a.append(b);
            }
            a
        };
        match self {
            PostProcessor::ByteLevel { trim_offsets, add_prefix_space } => {
                let mut b = b;
                if *trim_offsets {
                    trim_offsets_bytelevel(&mut a, *add_prefix_space);
                    if let Some(x) = b.as_mut() {
                        trim_offsets_bytelevel(x, *add_prefix_space);
                    }
                }
                Ok(default(a, b))
            }
            PostProcessor::Roberta { sep, cls, trim_offsets, add_prefix_space } => {
                let mut b = b;
                if *trim_offsets {
                    trim_offsets_bytelevel(&mut a, *add_prefix_space);
                    if let Some(x) = b.as_mut() {
                        trim_offsets_bytelevel(x, *add_prefix_space);
                    }
                }
                if !add_special {
                    return Ok(default(a, b));
                }
                let mut out = Encoding::default();
                out.push(cls.1, cls.0.clone(), (0, 0), None, 0, true, None);
                let (la, mut a) = (a.len(), a);
                a.type_ids = vec![0; la];
                a.sequence_ids = vec![Some(0); la];
                out.append(a);
                out.push(sep.1, sep.0.clone(), (0, 0), None, 0, true, None);
                if let Some(mut b) = b {
                    out.push(sep.1, sep.0.clone(), (0, 0), None, 0, true, None);
                    let lb = b.len();
                    b.type_ids = vec![0; lb];
                    b.sequence_ids = vec![Some(1); lb];
                    out.append(b);
                    out.push(sep.1, sep.0.clone(), (0, 0), None, 0, true, None);
                }
                Ok(out)
            }
            PostProcessor::Bert { sep, cls } => {
                if !add_special {
                    return Ok(default(a, b));
                }
                let mut out = Encoding::default();
                out.push(cls.1, cls.0.clone(), (0, 0), None, 0, true, None);
                let la = a.len();
                a.type_ids = vec![0; la];
                a.sequence_ids = vec![Some(0); la];
                out.append(a);
                out.push(sep.1, sep.0.clone(), (0, 0), None, 0, true, None);
                if let Some(mut b) = b {
                    let lb = b.len();
                    b.type_ids = vec![1; lb];
                    b.sequence_ids = vec![Some(1); lb];
                    out.append(b);
                    out.push(sep.1, sep.0.clone(), (0, 0), None, 1, true, None);
                }
                Ok(out)
            }
            PostProcessor::Template { single, pair, special } => {
                if !add_special {
                    return Ok(default(a, b));
                }
                let tpl = if b.is_some() { pair } else { single };
                let mut out = Encoding::default();
                let (mut a, mut b) = (Some(a), b);
                for p in tpl {
                    match p {
                        Piece::Seq(which, type_id) => {
                            let src = if *which == 'A' { a.take() } else { b.take() };
                            let mut src = src.ok_or_else(|| Error::Msg("tokenizer: template uses a sequence that is not available".into()))?;
                            let n = src.len();
                            src.type_ids = vec![*type_id; n];
                            src.sequence_ids = vec![Some(if *which == 'A' { 0 } else { 1 }); n];
                            out.append(src);
                        }
                        Piece::Special(name, type_id) => {
                            let (ids, toks) = special.get(name).ok_or_else(|| Error::Msg(format!("tokenizer: template special token {name} is not defined")))?;
                            for (id, t) in ids.iter().zip(toks) {
                                out.push(*id, t.clone(), (0, 0), None, *type_id, true, None);
                            }
                        }
                    }
                }
                Ok(out)
            }
            PostProcessor::Sequence(_) => unreachable!("sequences are handled by process_list"),
        }
    }
}

#[derive(Debug, Clone)]
pub struct AddedToken {
    pub content: String,
    pub id: u32,
    pub single_word: bool,
    pub lstrip: bool,
    pub rstrip: bool,
    pub normalized: bool,
    pub special: bool,
}

impl AddedToken {
    pub fn new(content: &str, special: bool) -> Self {
        Self { content: content.to_string(), id: 0, single_word: false, lstrip: false, rstrip: false, normalized: !special, special }
    }
}

pub struct Tokenizer {
    pub normalizer: Option<Normalizer>,
    pub pre_tokenizer: Option<PreTokenizer>,
    pub model: Model,
    pub post_processor: Option<PostProcessor>,
    pub decoder: Option<Decoder>,
    pub added: Vec<AddedToken>,
    pub truncation: Option<Truncation>,
    pub padding: Option<Padding>,
    source: Json,
}

enum Seg {
    Text(NormalizedString),
    Added(usize, (usize, usize)),
}

fn is_word_char(c: char) -> bool {
    c.is_alphanumeric() || c == '_'
}

impl Tokenizer {
    pub fn from_file(path: impl AsRef<std::path::Path>) -> Result<Self> {
        let text = std::fs::read_to_string(path.as_ref()).map_err(|e| Error::Msg(format!("tokenizer: cannot read {}: {e}", path.as_ref().display())))?;
        Self::from_str(&text)
    }

    #[allow(clippy::should_implement_trait)]
    pub fn from_str(text: &str) -> Result<Self> {
        Self::from_json(&Json::parse(text)?)
    }

    pub fn from_json(j: &Json) -> Result<Self> {
        let opt = |k: &str| j.get(k).filter(|v| !v.is_null());
        let model = Model::from_json(opt("model").ok_or_else(|| Error::Msg("tokenizer: missing model".into()))?)?;
        let mut added = Vec::new();
        for t in opt("added_tokens").and_then(Json::as_array).unwrap_or(&[]) {
            let flag = |k: &str, d: bool| t.get(k).and_then(Json::as_bool).unwrap_or(d);
            let special = flag("special", false);
            added.push(AddedToken {
                content: t.get("content").and_then(Json::as_str).unwrap_or("").to_string(),
                id: t.get("id").and_then(Json::as_u64).unwrap_or(0) as u32,
                single_word: flag("single_word", false),
                lstrip: flag("lstrip", false),
                rstrip: flag("rstrip", false),
                normalized: flag("normalized", !special),
                special,
            });
        }
        let truncation = opt("truncation").map(|t| Truncation {
            max_length: t.get("max_length").and_then(Json::as_u64).unwrap_or(512) as usize,
            strategy: match t.get("strategy").and_then(Json::as_str) {
                Some("OnlyFirst") => TruncationStrategy::OnlyFirst,
                Some("OnlySecond") => TruncationStrategy::OnlySecond,
                _ => TruncationStrategy::LongestFirst,
            },
            stride: t.get("stride").and_then(Json::as_u64).unwrap_or(0) as usize,
            direction: dir_from(t.get("direction")),
        });
        let padding = opt("padding").map(|p| Padding {
            strategy: match p.get("strategy") {
                Some(Json::Obj(o)) if !o.is_empty() => PaddingStrategy::Fixed(o[0].1.as_u64().unwrap_or(0) as usize),
                _ => PaddingStrategy::BatchLongest,
            },
            direction: dir_from(p.get("direction")),
            pad_to_multiple_of: p.get("pad_to_multiple_of").and_then(Json::as_u64).map(|n| n as usize),
            pad_id: p.get("pad_id").and_then(Json::as_u64).unwrap_or(0) as u32,
            pad_type_id: p.get("pad_type_id").and_then(Json::as_u64).unwrap_or(0) as u32,
            pad_token: p.get("pad_token").and_then(Json::as_str).unwrap_or("[PAD]").to_string(),
        });
        Ok(Self {
            normalizer: opt("normalizer").map(Normalizer::from_json).transpose()?,
            pre_tokenizer: opt("pre_tokenizer").map(PreTokenizer::from_json).transpose()?,
            model,
            post_processor: opt("post_processor").map(PostProcessor::from_json).transpose()?,
            decoder: opt("decoder").map(Decoder::from_json).transpose()?,
            added,
            truncation,
            padding,
            source: j.clone(),
        })
    }

    /// The `tokenizer.json` document: the loaded components with the current added tokens, truncation and padding.
    pub fn to_json(&self) -> Json {
        let mut j = self.source.clone();
        let b = Json::Bool;
        let n = |x: usize| Json::Num(x as f64);
        let added = self
            .added
            .iter()
            .map(|a| {
                Json::Obj(vec![
                    ("id".into(), Json::Num(a.id as f64)),
                    ("content".into(), Json::Str(a.content.clone())),
                    ("single_word".into(), b(a.single_word)),
                    ("lstrip".into(), b(a.lstrip)),
                    ("rstrip".into(), b(a.rstrip)),
                    ("normalized".into(), b(a.normalized)),
                    ("special".into(), b(a.special)),
                ])
            })
            .collect();
        j.set("added_tokens", Json::Arr(added));
        let dir = |d: Direction| Json::Str(if d == Direction::Left { "Left" } else { "Right" }.into());
        let trunc = match &self.truncation {
            None => Json::Null,
            Some(t) => {
                let strategy = match t.strategy {
                    TruncationStrategy::LongestFirst => "LongestFirst",
                    TruncationStrategy::OnlyFirst => "OnlyFirst",
                    TruncationStrategy::OnlySecond => "OnlySecond",
                };
                Json::Obj(vec![("direction".into(), dir(t.direction)), ("max_length".into(), n(t.max_length)), ("strategy".into(), Json::Str(strategy.into())), ("stride".into(), n(t.stride))])
            }
        };
        j.set("truncation", trunc);
        let pad = match &self.padding {
            None => Json::Null,
            Some(p) => Json::Obj(vec![
                ("strategy".into(), match p.strategy {
                    PaddingStrategy::BatchLongest => Json::Str("BatchLongest".into()),
                    PaddingStrategy::Fixed(k) => Json::Obj(vec![("Fixed".into(), n(k))]),
                }),
                ("direction".into(), dir(p.direction)),
                ("pad_to_multiple_of".into(), p.pad_to_multiple_of.map_or(Json::Null, n)),
                ("pad_id".into(), Json::Num(p.pad_id as f64)),
                ("pad_type_id".into(), Json::Num(p.pad_type_id as f64)),
                ("pad_token".into(), Json::Str(p.pad_token.clone())),
            ]),
        };
        j.set("padding", pad);
        j
    }

    pub(crate) fn set_source_model(&mut self, model: Json) {
        self.source.set("model", model);
    }

    /// Sets the normalizer from its `tokenizer.json` description (`Json::Null` removes it).
    pub fn set_normalizer_json(&mut self, j: &Json) -> Result<()> {
        self.normalizer = if j.is_null() { None } else { Some(Normalizer::from_json(j)?) };
        self.source.set("normalizer", j.clone());
        Ok(())
    }
    /// Sets the pre-tokenizer from its `tokenizer.json` description.
    pub fn set_pre_tokenizer_json(&mut self, j: &Json) -> Result<()> {
        self.pre_tokenizer = if j.is_null() { None } else { Some(PreTokenizer::from_json(j)?) };
        self.source.set("pre_tokenizer", j.clone());
        Ok(())
    }
    /// Sets the post-processor from its `tokenizer.json` description.
    pub fn set_post_processor_json(&mut self, j: &Json) -> Result<()> {
        self.post_processor = if j.is_null() { None } else { Some(PostProcessor::from_json(j)?) };
        self.source.set("post_processor", j.clone());
        Ok(())
    }
    /// Sets the decoder from its `tokenizer.json` description.
    pub fn set_decoder_json(&mut self, j: &Json) -> Result<()> {
        self.decoder = if j.is_null() { None } else { Some(Decoder::from_json(j)?) };
        self.source.set("decoder", j.clone());
        Ok(())
    }

    /// Writes `tokenizer.json`.
    pub fn save(&self, path: impl AsRef<std::path::Path>) -> Result<()> {
        std::fs::write(path.as_ref(), self.to_json().to_text()).map_err(|e| Error::Msg(format!("tokenizer: cannot write {}: {e}", path.as_ref().display())))
    }

    /// GPT-2 style `vocab.json` + `merges.txt` (GPT-2, OPT, GPT-Neo, CodeGen, ...; byte-level BPE). With
    /// `roberta_specials` (`<s>` / `</s>` looked up in the vocabulary) the RoBERTa post-processing is added.
    pub fn from_vocab_and_merges(vocab_json: &str, merges_txt: &str, add_prefix_space: bool, special_tokens: &[&str], roberta_specials: bool) -> Result<Self> {
        let vocab = Json::parse(vocab_json)?;
        let merges: Vec<Json> = merges_txt.lines().filter(|l| !l.is_empty() && !l.starts_with("#version")).map(|l| Json::Str(l.trim_end_matches('\r').to_string())).collect();
        let id_of = |t: &str| vocab.get(t).and_then(Json::as_u64);
        let s = |x: &str| Json::Str(x.to_string());
        let special_entry = |t: &str, id: u64| {
            Json::Obj(vec![
                ("id".into(), Json::Num(id as f64)),
                ("content".into(), s(t)),
                ("single_word".into(), Json::Bool(false)),
                // RoBERTa's mask token swallows the space before it
                ("lstrip".into(), Json::Bool(roberta_specials && t == "<mask>")),
                ("rstrip".into(), Json::Bool(false)),
                ("normalized".into(), Json::Bool(false)),
                ("special".into(), Json::Bool(true)),
            ])
        };
        let byte_level = |prefix: bool, trim: bool| Json::Obj(vec![("add_prefix_space".into(), Json::Bool(prefix)), ("trim_offsets".into(), Json::Bool(trim)), ("use_regex".into(), Json::Bool(true))]);
        let typed = |ty: &str, mut j: Json| {
            if let Json::Obj(o) = &mut j {
                o.insert(0, ("type".into(), s(ty)));
            }
            j
        };
        let post = if roberta_specials {
            let sep = id_of("</s>").ok_or_else(|| Error::Msg("tokenizer: the vocabulary has no </s>".into()))?;
            let cls = id_of("<s>").ok_or_else(|| Error::Msg("tokenizer: the vocabulary has no <s>".into()))?;
            Json::Obj(vec![
                ("type".into(), s("RobertaProcessing")),
                ("sep".into(), Json::Arr(vec![s("</s>"), Json::Num(sep as f64)])),
                ("cls".into(), Json::Arr(vec![s("<s>"), Json::Num(cls as f64)])),
                ("trim_offsets".into(), Json::Bool(true)),
                ("add_prefix_space".into(), Json::Bool(add_prefix_space)),
            ])
        } else {
            typed("ByteLevel", byte_level(true, false))
        };
        let doc = Json::Obj(vec![
            ("version".into(), s("1.0")),
            ("added_tokens".into(), Json::Arr(special_tokens.iter().filter_map(|t| id_of(t).map(|id| special_entry(t, id))).collect())),
            ("normalizer".into(), Json::Null),
            ("pre_tokenizer".into(), typed("ByteLevel", byte_level(add_prefix_space, true))),
            ("post_processor".into(), post),
            ("decoder".into(), typed("ByteLevel", byte_level(true, true))),
            ("model".into(), Json::Obj(vec![("type".into(), s("BPE")), ("unk_token".into(), Json::Null), ("fuse_unk".into(), Json::Bool(false)), ("vocab".into(), vocab.clone()), ("merges".into(), Json::Arr(merges))])),
        ]);
        Self::from_json(&doc)
    }

    /// BERT style `vocab.txt` (one token per line, the id is the line number).
    pub fn from_vocab_txt(vocab_txt: &str, lowercase: bool, strip_accents: Option<bool>) -> Result<Self> {
        let s = |x: &str| Json::Str(x.to_string());
        let vocab: Vec<(String, Json)> = vocab_txt.lines().enumerate().map(|(i, l)| (l.trim_end_matches('\r').to_string(), Json::Num(i as f64))).filter(|(l, _)| !l.is_empty()).collect();
        let id_of = |t: &str| vocab.iter().find(|(k, _)| k == t).and_then(|(_, v)| v.as_u64());
        let added: Vec<Json> = ["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"]
            .iter()
            .filter_map(|t| {
                id_of(t).map(|id| {
                    Json::Obj(vec![
                        ("id".into(), Json::Num(id as f64)),
                        ("content".into(), s(t)),
                        ("single_word".into(), Json::Bool(false)),
                        ("lstrip".into(), Json::Bool(false)),
                        ("rstrip".into(), Json::Bool(false)),
                        ("normalized".into(), Json::Bool(false)),
                        ("special".into(), Json::Bool(true)),
                    ])
                })
            })
            .collect();
        let post = match (id_of("[CLS]"), id_of("[SEP]")) {
            (Some(c), Some(sp)) => Json::Obj(vec![
                ("type".into(), s("BertProcessing")),
                ("sep".into(), Json::Arr(vec![s("[SEP]"), Json::Num(sp as f64)])),
                ("cls".into(), Json::Arr(vec![s("[CLS]"), Json::Num(c as f64)])),
            ]),
            _ => Json::Null,
        };
        let doc = Json::Obj(vec![
            ("version".into(), s("1.0")),
            ("added_tokens".into(), Json::Arr(added)),
            ("normalizer".into(), Json::Obj(vec![
                ("type".into(), s("BertNormalizer")),
                ("clean_text".into(), Json::Bool(true)),
                ("handle_chinese_chars".into(), Json::Bool(true)),
                ("strip_accents".into(), strip_accents.map_or(Json::Null, Json::Bool)),
                ("lowercase".into(), Json::Bool(lowercase)),
            ])),
            ("pre_tokenizer".into(), Json::Obj(vec![("type".into(), s("BertPreTokenizer"))])),
            ("post_processor".into(), post),
            ("decoder".into(), Json::Obj(vec![("type".into(), s("WordPiece")), ("prefix".into(), s("##")), ("cleanup".into(), Json::Bool(true))])),
            ("model".into(), Json::Obj(vec![
                ("type".into(), s("WordPiece")),
                ("unk_token".into(), s("[UNK]")),
                ("continuing_subword_prefix".into(), s("##")),
                ("max_input_chars_per_word".into(), Json::Num(100.0)),
                ("vocab".into(), Json::Obj(vocab)),
            ])),
        ]);
        Self::from_json(&doc)
    }

    pub fn token_to_id(&self, t: &str) -> Option<u32> {
        self.added.iter().find(|a| a.content == t).map(|a| a.id).or_else(|| self.model.token_to_id(t))
    }

    pub fn id_to_token(&self, id: u32) -> Option<String> {
        // like the reference, a token added with `normalized: true` is reported in its normalized form
        // (Llama-2's `</s>` is "\u{2581}</s>"), which is what the decoder then sees
        if let Some(a) = self.added.iter().find(|a| a.id == id) {
            if a.normalized {
                if let Some(nz) = &self.normalizer {
                    let mut n = NormalizedString::new(&a.content);
                    nz.normalize(&mut n);
                    return Some(n.get());
                }
            }
            return Some(a.content.clone());
        }
        self.model.id_to_token(id)
    }

    pub fn get_vocab_size(&self, with_added_tokens: bool) -> usize {
        let base = self.model.vocab_size();
        if !with_added_tokens {
            return base;
        }
        base + self.added.iter().filter(|a| self.model.token_to_id(&a.content).is_none()).count()
    }

    /// Adds tokens (ids continue after the current vocabulary); returns how many were new.
    pub fn add_tokens(&mut self, tokens: &[AddedToken]) -> usize {
        let mut n = 0;
        for t in tokens {
            if self.added.iter().any(|a| a.content == t.content) {
                continue;
            }
            let mut t = t.clone();
            t.id = match self.model.token_to_id(&t.content) {
                Some(id) => id,
                None => {
                    n += 1;
                    let max_added = self.added.iter().map(|a| a.id + 1).max().unwrap_or(0);
                    (self.model.vocab_size() as u32).max(max_added)
                }
            };
            self.added.push(t);
        }
        n
    }

    pub fn add_special_tokens(&mut self, tokens: &[&str]) -> usize {
        let v: Vec<AddedToken> = tokens.iter().map(|t| AddedToken::new(t, true)).collect();
        self.add_tokens(&v)
    }

    pub fn with_truncation(&mut self, t: Option<Truncation>) -> &mut Self {
        self.truncation = t;
        self
    }
    pub fn with_padding(&mut self, p: Option<Padding>) -> &mut Self {
        self.padding = p;
        self
    }

    /// Finds added tokens in `text` (chars with spans). Returns segments; text between tokens is `Seg::Text`.
    fn split_added(&self, ns: &NormalizedString, normalized_pass: bool) -> Vec<Seg> {
        let chars = ns.plain();
        let cands: Vec<usize> = (0..self.added.len()).filter(|&k| self.added[k].normalized == normalized_pass && !self.added[k].content.is_empty()).collect();
        if cands.is_empty() || chars.is_empty() {
            return vec![Seg::Text(ns.clone())];
        }
        let mut by_first: HashMap<char, Vec<(usize, Vec<char>)>> = HashMap::new();
        for &k in &cands {
            // normalized tokens are matched in their normalized form (Llama-2: "<s>" becomes "\u{2581}<s>")
            let cs: Vec<char> = if normalized_pass {
                let mut n = NormalizedString::new(&self.added[k].content);
                if let Some(nz) = &self.normalizer {
                    nz.normalize(&mut n);
                }
                n.plain()
            } else {
                self.added[k].content.chars().collect()
            };
            if !cs.is_empty() {
                by_first.entry(cs[0]).or_default().push((k, cs));
            }
        }
        for v in by_first.values_mut() {
            v.sort_by_key(|(_, c)| std::cmp::Reverse(c.len()));
        }
        let mut segs: Vec<Seg> = Vec::new();
        let (mut i, mut text_start) = (0usize, 0usize);
        let mut last_end = 0usize;
        while i < chars.len() {
            let hit = by_first.get(&chars[i]).and_then(|v| {
                v.iter().find(|(k, c)| {
                    chars[i..].starts_with(c) && {
                        let t = &self.added[*k];
                        !t.single_word || ((i == 0 || !is_word_char(chars[i - 1])) && (i + c.len() >= chars.len() || !is_word_char(chars[i + c.len()])))
                    }
                })
            });
            match hit {
                Some((k, c)) => {
                    let t = &self.added[*k];
                    let (mut s, mut e) = (i, i + c.len());
                    if t.lstrip {
                        while s > last_end.max(text_start) && chars[s - 1].is_whitespace() {
                            s -= 1;
                        }
                    }
                    if t.rstrip {
                        while e < chars.len() && chars[e].is_whitespace() {
                            e += 1;
                        }
                    }
                    if s > text_start {
                        segs.push(Seg::Text(NormalizedString { chars: ns.chars[text_start..s].to_vec() }));
                    }
                    let core = &ns.chars[i..i + c.len()];
                    let span = (core.iter().map(|x| x.1.0).min().unwrap_or(0), core.iter().map(|x| x.1.1).max().unwrap_or(0));
                    segs.push(Seg::Added(*k, span));
                    i = e;
                    text_start = e;
                    last_end = e;
                }
                None => i += 1,
            }
        }
        if text_start < chars.len() {
            segs.push(Seg::Text(NormalizedString { chars: ns.chars[text_start..].to_vec() }));
        }
        segs
    }

    fn encode_single(&self, text: &str, type_id: u32, seq: usize) -> Result<Encoding> {
        let mut enc = Encoding::default();
        let mut word = 0u32;
        let root = NormalizedString::new(text);
        // pass 1: added tokens that are matched on the raw text
        let mut stage: Vec<Seg> = self.split_added(&root, false);
        // normalize the remaining text, then pass 2: added tokens matched on the normalized text
        let mut next: Vec<Seg> = Vec::new();
        for s in stage.drain(..) {
            match s {
                Seg::Added(..) => next.push(s),
                Seg::Text(mut ns) => {
                    if let Some(n) = &self.normalizer {
                        n.normalize(&mut ns);
                    }
                    next.extend(self.split_added(&ns, true));
                }
            }
        }
        for s in next {
            match s {
                Seg::Added(k, span) => {
                    let t = &self.added[k];
                    enc.push(t.id, t.content.clone(), span, Some(word), type_id, false, Some(seq));
                    word += 1;
                }
                Seg::Text(ns) => {
                    let pieces = match &self.pre_tokenizer {
                        Some(p) => p.pre_tokenize(vec![ns]),
                        None => if ns.is_empty() { vec![] } else { vec![ns] },
                    };
                    for w in pieces {
                        let chars = w.plain();
                        for t in self.model.tokenize(&chars)? {
                            let spans = &w.chars[t.start..t.end];
                            let off = (spans.iter().map(|x| x.1.0).min().unwrap_or(0), spans.iter().map(|x| x.1.1).max().unwrap_or(0));
                            enc.push(t.id, t.value, off, Some(word), type_id, false, Some(seq));
                        }
                        word += 1;
                    }
                }
            }
        }
        Ok(enc)
    }

    fn post_process(&self, mut a: Encoding, mut b: Option<Encoding>, add_special: bool) -> Result<Encoding> {
        let added = if add_special { self.post_processor.as_ref().map_or(0, |p| p.added_tokens(b.is_some())) } else { 0 };
        if let Some(t) = &self.truncation {
            let max = t.max_length.saturating_sub(added);
            let total = a.len() + b.as_ref().map_or(0, Encoding::len);
            if total > max {
                let remove = total - max;
                let too_short = || Error::Msg("tokenizer: the sequence to truncate is too short to respect max_length".into());
                let strategy = if b.is_none() && t.strategy == TruncationStrategy::LongestFirst { TruncationStrategy::OnlyFirst } else { t.strategy };
                match strategy {
                    TruncationStrategy::LongestFirst => {
                        let bb = b.as_mut().unwrap();
                        let (mut na, mut nb) = (a.len(), bb.len());
                        for _ in 0..remove {
                            if na > nb { na -= 1 } else { nb -= 1 }
                        }
                        a.truncate(na, t.stride, t.direction);
                        bb.truncate(nb, t.stride, t.direction);
                    }
                    TruncationStrategy::OnlyFirst => {
                        if a.len() < remove {
                            return Err(too_short());
                        }
                        let keep = a.len() - remove;
                        a.truncate(keep, t.stride, t.direction);
                    }
                    TruncationStrategy::OnlySecond => {
                        let Some(bb) = b.as_mut() else { return Err(Error::Msg("tokenizer: OnlySecond truncation needs a pair".into())) };
                        if bb.len() < remove {
                            return Err(too_short());
                        }
                        let keep = bb.len() - remove;
                        bb.truncate(keep, t.stride, t.direction);
                    }
                }
            }
        }
        let windows = if b.is_none() { std::mem::take(&mut a.overflowing) } else { Vec::new() };
        let mut out = self.run_processor(a, b, add_special)?;
        out.overflowing = windows.into_iter().map(|w| self.run_processor(w, None, add_special)).collect::<Result<_>>()?;
        Ok(out)
    }

    fn run_processor(&self, mut a: Encoding, mut b: Option<Encoding>, add_special: bool) -> Result<Encoding> {
        match &self.post_processor {
            Some(p) => p.process(a, b, add_special),
            None => {
                a.sequence_ids = vec![Some(0); a.len()];
                if let Some(mut bb) = b.take() {
                    bb.sequence_ids = vec![Some(1); bb.len()];
                    a.append(bb);
                }
                Ok(a)
            }
        }
    }

    fn pad_all(&self, encs: &mut [Encoding]) {
        let Some(p) = &self.padding else { return };
        let mut target = match p.strategy {
            PaddingStrategy::Fixed(n) => n,
            PaddingStrategy::BatchLongest => encs.iter().flat_map(|e| std::iter::once(e.len()).chain(e.overflowing.iter().map(Encoding::len))).max().unwrap_or(0),
        };
        if let Some(m) = p.pad_to_multiple_of {
            if m > 0 && target % m != 0 {
                target = (target / m + 1) * m;
            }
        }
        for e in encs {
            e.pad(target, p.pad_id, p.pad_type_id, &p.pad_token, p.direction);
        }
    }

    pub fn encode(&self, text: &str, add_special_tokens: bool) -> Result<Encoding> {
        let a = self.encode_single(text, 0, 0)?;
        let mut e = [self.post_process(a, None, add_special_tokens)?];
        self.pad_all(&mut e);
        let [e] = e;
        Ok(e)
    }

    pub fn encode_pair(&self, a: &str, b: &str, add_special_tokens: bool) -> Result<Encoding> {
        let ea = self.encode_single(a, 0, 0)?;
        let eb = self.encode_single(b, 1, 1)?;
        let mut e = [self.post_process(ea, Some(eb), add_special_tokens)?];
        self.pad_all(&mut e);
        let [e] = e;
        Ok(e)
    }

    pub fn encode_batch(&self, texts: &[&str], add_special_tokens: bool) -> Result<Vec<Encoding>> {
        let mut out = texts
            .iter()
            .map(|t| {
                let a = self.encode_single(t, 0, 0)?;
                self.post_process(a, None, add_special_tokens)
            })
            .collect::<Result<Vec<_>>>()?;
        self.pad_all(&mut out);
        Ok(out)
    }

    pub fn encode_pair_batch(&self, pairs: &[(&str, &str)], add_special_tokens: bool) -> Result<Vec<Encoding>> {
        let mut out = pairs
            .iter()
            .map(|(a, b)| {
                let ea = self.encode_single(a, 0, 0)?;
                let eb = self.encode_single(b, 1, 1)?;
                self.post_process(ea, Some(eb), add_special_tokens)
            })
            .collect::<Result<Vec<_>>>()?;
        self.pad_all(&mut out);
        Ok(out)
    }

    pub fn decode(&self, ids: &[u32], skip_special_tokens: bool) -> Result<String> {
        let mut tokens = Vec::with_capacity(ids.len());
        for &id in ids {
            if skip_special_tokens && self.added.iter().any(|a| a.id == id && a.special) {
                continue;
            }
            if let Some(t) = self.id_to_token(id) {
                tokens.push(t);
            }
        }
        Ok(match &self.decoder {
            Some(d) => d.decode(tokens),
            None => tokens.join(" "),
        })
    }

    pub fn decode_batch(&self, batch: &[Vec<u32>], skip_special_tokens: bool) -> Result<Vec<String>> {
        batch.iter().map(|ids| self.decode(ids, skip_special_tokens)).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const BERT: &str = r###"{"version":"1.0","truncation":null,"padding":null,"added_tokens":[{"id":0,"content":"[UNK]","single_word":false,"lstrip":false,"rstrip":false,"normalized":false,"special":true},{"id":1,"content":"[CLS]","single_word":false,"lstrip":false,"rstrip":false,"normalized":false,"special":true},{"id":2,"content":"[SEP]","single_word":false,"lstrip":false,"rstrip":false,"normalized":false,"special":true}],
    "normalizer":{"type":"BertNormalizer","clean_text":true,"handle_chinese_chars":true,"strip_accents":null,"lowercase":true},
    "pre_tokenizer":{"type":"BertPreTokenizer"},
    "post_processor":{"type":"TemplateProcessing","single":[{"SpecialToken":{"id":"[CLS]","type_id":0}},{"Sequence":{"id":"A","type_id":0}},{"SpecialToken":{"id":"[SEP]","type_id":0}}],"pair":[{"SpecialToken":{"id":"[CLS]","type_id":0}},{"Sequence":{"id":"A","type_id":0}},{"SpecialToken":{"id":"[SEP]","type_id":0}},{"Sequence":{"id":"B","type_id":1}},{"SpecialToken":{"id":"[SEP]","type_id":1}}],"special_tokens":{"[CLS]":{"id":"[CLS]","ids":[1],"tokens":["[CLS]"]},"[SEP]":{"id":"[SEP]","ids":[2],"tokens":["[SEP]"]}}},
    "decoder":{"type":"WordPiece","prefix":"##","cleanup":true},
    "model":{"type":"WordPiece","unk_token":"[UNK]","continuing_subword_prefix":"##","max_input_chars_per_word":100,"vocab":{"[UNK]":0,"[CLS]":1,"[SEP]":2,"hello":3,",":4,"world":5,"un":6,"##believ":7,"##able":8,"cafe":9}}}"###;

    #[test]
    fn bert_pipeline_end_to_end() {
        let t = Tokenizer::from_str(BERT).unwrap();
        let e = t.encode("Hello, wörld! Unbelievable café", true).unwrap();
        assert_eq!(e.tokens, ["[CLS]", "hello", ",", "world", "[UNK]", "un", "##believ", "##able", "cafe", "[SEP]"]);
        assert_eq!(e.special_tokens_mask, [1, 0, 0, 0, 0, 0, 0, 0, 0, 1]);
        assert_eq!(e.offsets[1], (0, 5));
        assert_eq!(t.decode(&e.ids, true).unwrap(), "hello, world unbelievable cafe");
        let p = t.encode_pair("hello", "world", true).unwrap();
        assert_eq!(p.type_ids, [0, 0, 0, 1, 1]);
        assert_eq!(t.token_to_id("world"), Some(5));
        // special tokens inside the text are recognised
        assert_eq!(t.encode("hello [SEP] world", false).unwrap().ids, [3, 2, 5]);
    }

    #[test]
    fn truncation_and_padding() {
        let mut t = Tokenizer::from_str(BERT).unwrap();
        t.with_truncation(Some(Truncation { max_length: 5, strategy: TruncationStrategy::LongestFirst, stride: 0, direction: Direction::Right }));
        t.with_padding(Some(Padding { strategy: PaddingStrategy::Fixed(7), direction: Direction::Right, pad_to_multiple_of: None, pad_id: 0, pad_type_id: 0, pad_token: "[PAD]".into() }));
        let e = t.encode("hello world hello world hello", true).unwrap();
        assert_eq!(e.ids, [1, 3, 5, 3, 2, 0, 0]);
        assert_eq!(e.attention_mask, [1, 1, 1, 1, 1, 0, 0]);
        assert_eq!(e.overflowing.len(), 1);
    }
}
