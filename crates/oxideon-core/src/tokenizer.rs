//! Byte-level Byte-Pair Encoding (`BpeTokenizer`) for training and running LLMs on raw UTF-8 text.
//!
//! Uses a byte-level base vocabulary (`0..256`) plus 4 reserved special tokens
//! (`<pad>=256`, `<bos>=257`, `<eos>=258`, `<unk>=259`), followed by learned merge rules
//! (`260..vocab_size`), guaranteeing lossless encoding and decoding of any UTF-8 string.

pub mod omni_tokenizer;
pub mod auto;
pub mod chat_template;
pub mod hf;
pub mod json;
pub mod regex;
pub mod sentencepiece;
pub mod tiktoken;
pub mod unicode;
#[allow(clippy::unreadable_literal)]
pub mod unicode_data;

pub use omni_tokenizer::{ModalityKind, OmniSequence, OmniTokenSegment, OmniTokenizer};

use std::collections::HashMap;

use crate::error::Result;
use crate::tensor::Tensor;

pub const PAD_TOKEN_ID: usize = 256;
pub const BOS_TOKEN_ID: usize = 257;
pub const EOS_TOKEN_ID: usize = 258;
pub const UNK_TOKEN_ID: usize = 259;
pub const BASE_VOCAB_SIZE: usize = 260;

#[derive(Debug, Clone)]
pub struct BpeTokenizer {
    /// Ordered list of learned merge rules `((left_id, right_id), merged_id)`.
    pub merges: Vec<((usize, usize), usize)>,
    merge_map: HashMap<(usize, usize), usize>,
    /// Byte sequences corresponding to each token ID (`0..vocab_size`).
    vocab_bytes: Vec<Vec<u8>>,
}

impl Default for BpeTokenizer {
    fn default() -> Self {
        Self::new_byte_level()
    }
}

impl BpeTokenizer {
    /// Creates a byte-level tokenizer with the 256 base byte tokens and 4 special tokens.
    pub fn new_byte_level() -> Self {
        let mut vocab_bytes = Vec::with_capacity(BASE_VOCAB_SIZE);
        for b in 0..=255u8 {
            vocab_bytes.push(vec![b]);
        }
        // Special tokens: 256=<pad>, 257=<bos>, 258=<eos>, 259=<unk>
        vocab_bytes.push(Vec::new());
        vocab_bytes.push(Vec::new());
        vocab_bytes.push(Vec::new());
        vocab_bytes.push(Vec::new());

        Self {
            merges: Vec::new(),
            merge_map: HashMap::new(),
            vocab_bytes,
        }
    }

    /// Trains a byte-level BPE tokenizer on `corpus` up to `target_vocab_size`.
    pub fn train(corpus: &str, target_vocab_size: usize) -> Self {
        let mut tok = Self::new_byte_level();
        if target_vocab_size <= BASE_VOCAB_SIZE || corpus.is_empty() {
            return tok;
        }
        let num_merges = target_vocab_size - BASE_VOCAB_SIZE;
        let mut ids: Vec<usize> = corpus.as_bytes().iter().map(|&b| b as usize).collect();

        for i in 0..num_merges {
            if ids.len() < 2 {
                break;
            }
            let mut counts: HashMap<(usize, usize), usize> = HashMap::new();
            for pair in ids.windows(2) {
                *counts.entry((pair[0], pair[1])).or_insert(0) += 1;
            }
            // Pick highest-frequency pair (tie-breaking deterministically by pair tuple)
            let best_pair = counts
                .into_iter()
                .max_by(|(p1, c1), (p2, c2)| c1.cmp(c2).then_with(|| p2.cmp(p1)));

            let Some((pair, count)) = best_pair else {
                break;
            };
            if count < 2 {
                break;
            }

            let new_id = BASE_VOCAB_SIZE + i;
            let mut merged_bytes = tok.vocab_bytes[pair.0].clone();
            merged_bytes.extend_from_slice(&tok.vocab_bytes[pair.1]);
            tok.vocab_bytes.push(merged_bytes);
            tok.merges.push((pair, new_id));
            tok.merge_map.insert(pair, new_id);

            ids = apply_single_merge(&ids, pair, new_id);
        }
        tok
    }

    pub fn vocab_size(&self) -> usize {
        self.vocab_bytes.len()
    }

    /// Encodes a UTF-8 string into token IDs.
    pub fn encode(&self, text: &str) -> Vec<usize> {
        let mut ids: Vec<usize> = text.as_bytes().iter().map(|&b| b as usize).collect();
        for &(pair, new_id) in &self.merges {
            if ids.len() < 2 {
                break;
            }
            ids = apply_single_merge(&ids, pair, new_id);
        }
        ids
    }

    /// Encodes `text` with optional `<bos>` and `<eos>` tokens.
    pub fn encode_with_special(&self, text: &str, add_bos: bool, add_eos: bool) -> Vec<usize> {
        let mut out = Vec::new();
        if add_bos {
            out.push(BOS_TOKEN_ID);
        }
        out.extend(self.encode(text));
        if add_eos {
            out.push(EOS_TOKEN_ID);
        }
        out
    }

    /// Decodes a slice of token IDs back into a UTF-8 `String` (ignoring special tokens).
    pub fn decode(&self, ids: &[usize]) -> String {
        let mut bytes = Vec::new();
        for &id in ids {
            if id < 256 || (id >= BASE_VOCAB_SIZE && id < self.vocab_bytes.len()) {
                bytes.extend_from_slice(&self.vocab_bytes[id]);
            }
        }
        String::from_utf8_lossy(&bytes).into_owned()
    }

    /// Encodes `text` directly into a `[1, seq_len]` `I64` [`Tensor`].
    pub fn encode_tensor(&self, text: &str) -> Result<Tensor> {
        let ids: Vec<i64> = self.encode(text).into_iter().map(|x| x as i64).collect();
        let len = ids.len().max(1);
        let ids = if ids.is_empty() {
            vec![PAD_TOKEN_ID as i64]
        } else {
            ids
        };
        Tensor::from_vec(ids, &[1, len])
    }

    /// Encodes a batch of strings with right-padding (`PAD_TOKEN_ID`), returning
    /// `(input_ids: [batch, max_len] (I64), attention_mask: [batch, max_len] (F32))`.
    pub fn encode_batch_padded(
        &self,
        texts: &[&str],
        max_len: Option<usize>,
    ) -> Result<(Tensor, Tensor)> {
        let encoded: Vec<Vec<usize>> = texts.iter().map(|t| self.encode(t)).collect();
        let target_len = max_len
            .unwrap_or_else(|| encoded.iter().map(Vec::len).max().unwrap_or(1))
            .max(1);
        let b = texts.len();
        // Refuse overflowing/giant `batch * max_len` before the two allocs below.
        let n = crate::limits::check_elements("encode_batch_padded", &[b, target_len])?;

        let mut batch_ids = vec![PAD_TOKEN_ID as i64; n];
        let mut mask = vec![0.0f32; n];

        for (row, seq) in encoded.iter().enumerate() {
            let take = seq.len().min(target_len);
            for col in 0..take {
                batch_ids[row * target_len + col] = seq[col] as i64;
                mask[row * target_len + col] = 1.0;
            }
        }

        let ids_t = Tensor::from_vec(batch_ids, &[b, target_len])?;
        let mask_t = Tensor::from_vec(mask, &[b, target_len])?;
        Ok((ids_t, mask_t))
    }
}

fn apply_single_merge(ids: &[usize], pair: (usize, usize), new_id: usize) -> Vec<usize> {
    let mut out = Vec::with_capacity(ids.len());
    let mut i = 0;
    while i < ids.len() {
        if i + 1 < ids.len() && ids[i] == pair.0 && ids[i + 1] == pair.1 {
            out.push(new_id);
            i += 2;
        } else {
            out.push(ids[i]);
            i += 1;
        }
    }
    out
}
