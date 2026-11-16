//! High-performance text vectorization, n-gram extraction, hashing, and inverted indexing.
//!
//! Provides native Rust implementations of:
//! - Canonical MurmurHash3 32-bit streaming hashing with signed feature projection.
//! - Fast tokenization and n-gram extraction (word-level and subword character-level).
//! - [`FastCountVectorizer`] producing sparse [`SparseCsr`] or dense [`Tensor`].
//! - [`FastTfidfVectorizer`] with smooth IDF and L1/L2 normalization.
//! - [`FastHashingVectorizer`] stateless hashing trick with sign alternation.
//! - [`FastBM25Index`] inverted index for BM25Okapi and BM25+ ranked retrieval.
//! - [`FastTextCharNGramVectorizer`] subword character n-gram morphology vectorizer.
//! - [`KernelSpectralVectorizer`] Random Fourier Features mapping to Euclidean RKHS.

use std::collections::{HashMap, HashSet};

use crate::backend::Device;
use crate::dtype::DType;
use crate::error::{Error, Result};
use crate::sparse::SparseCsr;
use crate::tensor::Tensor;

/// Canonical 32-bit MurmurHash3 implementation matching standard C and Python implementations.
#[inline]
pub fn murmurhash3_32(key: &[u8], seed: u32) -> u32 {
    let len = key.len();
    let nblocks = len / 4;
    let mut h1 = seed;
    const C1: u32 = 0xcc9e2d51;
    const C2: u32 = 0x1b873593;

    for i in 0..nblocks {
        let idx = i * 4;
        let mut k1 = u32::from_le_bytes([key[idx], key[idx + 1], key[idx + 2], key[idx + 3]]);
        k1 = k1.wrapping_mul(C1);
        k1 = k1.rotate_left(15);
        k1 = k1.wrapping_mul(C2);

        h1 ^= k1;
        h1 = h1.rotate_left(13);
        h1 = h1.wrapping_mul(5).wrapping_add(0xe6546b64);
    }

    let tail = &key[nblocks * 4..];
    let mut k1 = 0u32;
    match tail.len() {
        3 => {
            k1 ^= (tail[2] as u32) << 16;
            k1 ^= (tail[1] as u32) << 8;
            k1 ^= tail[0] as u32;
            k1 = k1.wrapping_mul(C1);
            k1 = k1.rotate_left(15);
            k1 = k1.wrapping_mul(C2);
            h1 ^= k1;
        }
        2 => {
            k1 ^= (tail[1] as u32) << 8;
            k1 ^= tail[0] as u32;
            k1 = k1.wrapping_mul(C1);
            k1 = k1.rotate_left(15);
            k1 = k1.wrapping_mul(C2);
            h1 ^= k1;
        }
        1 => {
            k1 ^= tail[0] as u32;
            k1 = k1.wrapping_mul(C1);
            k1 = k1.rotate_left(15);
            k1 = k1.wrapping_mul(C2);
            h1 ^= k1;
        }
        _ => {}
    }

    h1 ^= len as u32;
    h1 ^= h1 >> 16;
    h1 = h1.wrapping_mul(0x85ebca6b);
    h1 ^= h1 >> 13;
    h1 = h1.wrapping_mul(0xc2b2ae35);
    h1 ^= h1 >> 16;
    h1
}

/// Tokenizes text into lowercase alphanumeric words.
pub fn tokenize_words(text: &str) -> Vec<String> {
    let mut tokens = Vec::new();
    let mut current = String::new();
    for ch in text.chars() {
        if ch.is_alphanumeric() {
            for lower in ch.to_lowercase() {
                current.push(lower);
            }
        } else if !current.is_empty() {
            tokens.push(std::mem::take(&mut current));
        }
    }
    if !current.is_empty() {
        tokens.push(current);
    }
    tokens
}

/// Extracts word n-grams from a token sequence with $n \in [\text{min\_n}, \text{max\_n}]$.
pub fn extract_ngrams(tokens: &[String], min_n: usize, max_n: usize) -> Vec<String> {
    let mut ngrams = Vec::new();
    let n_tokens = tokens.len();
    for n in min_n..=max_n {
        if n == 0 || n > n_tokens {
            continue;
        }
        for i in 0..=(n_tokens - n) {
            let slice = &tokens[i..i + n];
            ngrams.push(slice.join(" "));
        }
    }
    ngrams
}

/// Extracts character n-grams from a word with optional `<` and `>` word boundary tags.
pub fn extract_char_ngrams(
    word: &str,
    min_n: usize,
    max_n: usize,
    add_boundaries: bool,
) -> Vec<String> {
    let bounded = if add_boundaries {
        format!("<{word}>")
    } else {
        word.to_string()
    };
    let chars: Vec<char> = bounded.chars().collect();
    let n_chars = chars.len();
    let mut ngrams = Vec::new();
    for n in min_n..=max_n {
        if n == 0 || n > n_chars {
            continue;
        }
        for i in 0..=(n_chars - n) {
            let slice: String = chars[i..i + n].iter().collect();
            ngrams.push(slice);
        }
    }
    ngrams
}

/// Native Bag-of-Words and N-Gram Count Vectorizer.
#[derive(Debug, Clone)]
pub struct FastCountVectorizer {
    pub min_df: usize,
    pub max_df: Option<usize>,
    pub max_features: Option<usize>,
    pub ngram_range: (usize, usize),
    pub binary: bool,
    pub vocab: HashMap<String, usize>,
    pub inverse_vocab: Vec<String>,
    pub stop_words: Option<HashSet<String>>,
}

impl FastCountVectorizer {
    pub fn new(
        ngram_range: (usize, usize),
        binary: bool,
        max_features: Option<usize>,
    ) -> Result<Self> {
        if ngram_range.0 == 0 || ngram_range.0 > ngram_range.1 {
            return Err(Error::Msg(format!(
                "Invalid ngram_range: ({}, {})",
                ngram_range.0, ngram_range.1
            )));
        }
        Ok(Self {
            min_df: 1,
            max_df: None,
            max_features,
            ngram_range,
            binary,
            vocab: HashMap::new(),
            inverse_vocab: Vec::new(),
            stop_words: None,
        })
    }

    pub fn with_stop_words(mut self, stop_words: HashSet<String>) -> Self {
        self.stop_words = Some(stop_words);
        self
    }

    pub fn with_min_df(mut self, min_df: usize) -> Self {
        self.min_df = min_df.max(1);
        self
    }

    pub fn with_max_df(mut self, max_df: Option<usize>) -> Self {
        self.max_df = max_df;
        self
    }

    pub fn fit(&mut self, documents: &[&str]) {
        let mut doc_freq: HashMap<String, usize> = HashMap::new();
        let mut term_freq: HashMap<String, usize> = HashMap::new();

        for &doc in documents {
            let tokens = tokenize_words(doc);
            let ngrams = extract_ngrams(&tokens, self.ngram_range.0, self.ngram_range.1);
            let mut seen_in_doc: HashSet<String> = HashSet::new();

            for ng in ngrams {
                if let Some(ref sw) = self.stop_words {
                    if sw.contains(&ng) {
                        continue;
                    }
                }
                *term_freq.entry(ng.clone()).or_insert(0) += 1;
                seen_in_doc.insert(ng);
            }

            for term in seen_in_doc {
                *doc_freq.entry(term).or_insert(0) += 1;
            }
        }

        let mut candidates: Vec<(String, usize, usize)> = doc_freq
            .into_iter()
            .filter(|&(_, df)| {
                df >= self.min_df && self.max_df.is_none_or(|max_val| df <= max_val)
            })
            .map(|(term, df)| {
                let tf = term_freq.get(&term).copied().unwrap_or(0);
                (term, df, tf)
            })
            .collect();

        // Sort by total term frequency descending, then lexicographically
        candidates.sort_by(|a, b| b.2.cmp(&a.2).then_with(|| a.0.cmp(&b.0)));

        if let Some(max_f) = self.max_features {
            if candidates.len() > max_f {
                candidates.truncate(max_f);
            }
        }

        // Final vocabulary is sorted alphabetically to maintain canonical feature indexing
        candidates.sort_by(|a, b| a.0.cmp(&b.0));

        self.vocab.clear();
        self.inverse_vocab.clear();
        for (idx, (term, _, _)) in candidates.into_iter().enumerate() {
            self.vocab.insert(term.clone(), idx);
            self.inverse_vocab.push(term);
        }
    }

    pub fn transform(&self, documents: &[&str]) -> Result<SparseCsr> {
        let rows = documents.len();
        let cols = self.vocab.len();

        let mut row_ptr = Vec::with_capacity(rows + 1);
        row_ptr.push(0);
        let mut col_idx = Vec::new();
        let mut values = Vec::new();

        for &doc in documents {
            let tokens = tokenize_words(doc);
            let ngrams = extract_ngrams(&tokens, self.ngram_range.0, self.ngram_range.1);

            let mut counts: HashMap<usize, f64> = HashMap::new();
            for ng in ngrams {
                if let Some(ref sw) = self.stop_words {
                    if sw.contains(&ng) {
                        continue;
                    }
                }
                if let Some(&col) = self.vocab.get(&ng) {
                    let entry = counts.entry(col).or_insert(0.0);
                    if self.binary {
                        *entry = 1.0;
                    } else {
                        *entry += 1.0;
                    }
                }
            }

            let mut sorted_entries: Vec<(usize, f64)> = counts.into_iter().collect();
            sorted_entries.sort_by_key(|&(c, _)| c);

            for (c, v) in sorted_entries {
                col_idx.push(c);
                values.push(v);
            }
            row_ptr.push(col_idx.len());
        }

        SparseCsr::new(row_ptr, col_idx, values, [rows, cols])
    }

    pub fn transform_dense(&self, documents: &[&str]) -> Result<Tensor> {
        let csr = self.transform(documents)?;
        csr.to_dense(DType::F32, &Device::Cpu)
    }

    pub fn vocab_size(&self) -> usize {
        self.vocab.len()
    }
}

/// Native TF-IDF Vectorizer with smooth IDF calculation and L1/L2 sample normalization.
#[derive(Debug, Clone)]
pub struct FastTfidfVectorizer {
    pub count_vec: FastCountVectorizer,
    pub idf: Vec<f64>,
    pub sublinear_tf: bool,
    pub norm: Option<String>,
}

impl FastTfidfVectorizer {
    pub fn new(
        ngram_range: (usize, usize),
        sublinear_tf: bool,
        norm: Option<String>,
    ) -> Result<Self> {
        let count_vec = FastCountVectorizer::new(ngram_range, false, None)?;
        Ok(Self {
            count_vec,
            idf: Vec::new(),
            sublinear_tf,
            norm,
        })
    }

    pub fn fit(&mut self, documents: &[&str]) {
        self.count_vec.fit(documents);
        let n_docs = documents.len() as f64;
        let vocab_size = self.count_vec.vocab_size();

        // Calculate document frequency for each vocabulary feature
        let mut doc_freq = vec![0.0f64; vocab_size];
        for &doc in documents {
            let tokens = tokenize_words(doc);
            let ngrams = extract_ngrams(
                &tokens,
                self.count_vec.ngram_range.0,
                self.count_vec.ngram_range.1,
            );
            let mut seen: HashSet<usize> = HashSet::new();
            for ng in ngrams {
                if let Some(&idx) = self.count_vec.vocab.get(&ng) {
                    seen.insert(idx);
                }
            }
            for idx in seen {
                doc_freq[idx] += 1.0;
            }
        }

        // Smooth IDF: ln((1 + N) / (1 + df)) + 1.0
        self.idf = doc_freq
            .into_iter()
            .map(|df| ((1.0 + n_docs) / (1.0 + df)).ln() + 1.0)
            .collect();
    }

    pub fn transform(&self, documents: &[&str]) -> Result<SparseCsr> {
        let raw_csr = self.count_vec.transform(documents)?;
        let rows = raw_csr.shape[0];
        let cols = raw_csr.shape[1];

        let mut row_ptr = Vec::with_capacity(rows + 1);
        row_ptr.push(0);
        let mut col_idx = Vec::with_capacity(raw_csr.col_idx.len());
        let mut values = Vec::with_capacity(raw_csr.values.len());

        for r in 0..rows {
            let start = raw_csr.row_ptr[r];
            let end = raw_csr.row_ptr[r + 1];

            let mut row_vals: Vec<(usize, f64)> = Vec::with_capacity(end - start);
            for i in start..end {
                let c = raw_csr.col_idx[i];
                let raw_val = raw_csr.values[i];
                let tf = if self.sublinear_tf {
                    1.0 + raw_val.ln()
                } else {
                    raw_val
                };
                let idf_val = self.idf.get(c).copied().unwrap_or(1.0);
                row_vals.push((c, tf * idf_val));
            }

            // Normalization
            if let Some(ref norm_type) = self.norm {
                match norm_type.to_lowercase().as_str() {
                    "l2" => {
                        let sum_sq: f64 = row_vals.iter().map(|&(_, v)| v * v).sum();
                        let l2_norm = (sum_sq + 1e-12).sqrt();
                        for item in &mut row_vals {
                            item.1 /= l2_norm;
                        }
                    }
                    "l1" => {
                        let sum_abs: f64 = row_vals.iter().map(|&(_, v)| v.abs()).sum();
                        let l1_norm = (sum_abs + 1e-12).max(1e-12);
                        for item in &mut row_vals {
                            item.1 /= l1_norm;
                        }
                    }
                    _ => {}
                }
            }

            for (c, v) in row_vals {
                col_idx.push(c);
                values.push(v);
            }
            row_ptr.push(col_idx.len());
        }

        SparseCsr::new(row_ptr, col_idx, values, [rows, cols])
    }

    pub fn transform_dense(&self, documents: &[&str]) -> Result<Tensor> {
        let csr = self.transform(documents)?;
        csr.to_dense(DType::F32, &Device::Cpu)
    }

    pub fn idf_weights(&self) -> &[f64] {
        &self.idf
    }
}

/// Stateless Hashing Vectorizer using MurmurHash3 with optional sign alternation.
#[derive(Debug, Clone)]
pub struct FastHashingVectorizer {
    pub n_features: usize,
    pub ngram_range: (usize, usize),
    pub alternate_sign: bool,
    pub norm: Option<String>,
}

impl FastHashingVectorizer {
    pub fn new(
        n_features: usize,
        ngram_range: (usize, usize),
        alternate_sign: bool,
        norm: Option<String>,
    ) -> Result<Self> {
        if n_features == 0 {
            return Err(Error::Msg(
                "n_features must be strictly positive for HashingVectorizer".into(),
            ));
        }
        if ngram_range.0 == 0 || ngram_range.0 > ngram_range.1 {
            return Err(Error::Msg(format!(
                "Invalid ngram_range: ({}, {})",
                ngram_range.0, ngram_range.1
            )));
        }
        Ok(Self {
            n_features,
            ngram_range,
            alternate_sign,
            norm,
        })
    }

    pub fn transform(&self, documents: &[&str]) -> Result<SparseCsr> {
        let rows = documents.len();
        let cols = self.n_features;

        let mut row_ptr = Vec::with_capacity(rows + 1);
        row_ptr.push(0);
        let mut col_idx = Vec::new();
        let mut values = Vec::new();

        for &doc in documents {
            let tokens = tokenize_words(doc);
            let ngrams = extract_ngrams(&tokens, self.ngram_range.0, self.ngram_range.1);

            let mut counts: HashMap<usize, f64> = HashMap::new();
            for ng in ngrams {
                let bytes = ng.as_bytes();
                let h1 = murmurhash3_32(bytes, 0);
                let col = (h1 as usize) % self.n_features;

                let sign = if self.alternate_sign {
                    let h2 = murmurhash3_32(bytes, 1);
                    if (h2 & 1) == 0 { 1.0 } else { -1.0 }
                } else {
                    1.0
                };

                *counts.entry(col).or_insert(0.0) += sign;
            }

            let mut sorted_entries: Vec<(usize, f64)> =
                counts.into_iter().filter(|&(_, v)| v != 0.0).collect();
            sorted_entries.sort_by_key(|&(c, _)| c);

            // Normalization
            if let Some(ref norm_type) = self.norm {
                match norm_type.to_lowercase().as_str() {
                    "l2" => {
                        let sum_sq: f64 = sorted_entries.iter().map(|&(_, v)| v * v).sum();
                        let l2_norm = (sum_sq + 1e-12).sqrt();
                        for item in &mut sorted_entries {
                            item.1 /= l2_norm;
                        }
                    }
                    "l1" => {
                        let sum_abs: f64 = sorted_entries.iter().map(|&(_, v)| v.abs()).sum();
                        let l1_norm = (sum_abs + 1e-12).max(1e-12);
                        for item in &mut sorted_entries {
                            item.1 /= l1_norm;
                        }
                    }
                    _ => {}
                }
            }

            for (c, v) in sorted_entries {
                col_idx.push(c);
                values.push(v);
            }
            row_ptr.push(col_idx.len());
        }

        SparseCsr::new(row_ptr, col_idx, values, [rows, cols])
    }

    pub fn transform_dense(&self, documents: &[&str]) -> Result<Tensor> {
        let csr = self.transform(documents)?;
        csr.to_dense(DType::F32, &Device::Cpu)
    }
}

/// High-speed Inverted Index with Robertson-Spärck Jones BM25Okapi and BM25+ scoring.
#[derive(Debug, Clone)]
pub struct FastBM25Index {
    pub k1: f64,
    pub b: f64,
    pub delta: f64,
    pub doc_lens: Vec<usize>,
    pub avgdl: f64,
    pub num_docs: usize,
    pub inverted_index: HashMap<String, Vec<(usize, u32)>>,
    pub idf: HashMap<String, f64>,
}

impl FastBM25Index {
    pub fn new(k1: f64, b: f64, delta: f64) -> Self {
        Self {
            k1,
            b,
            delta,
            doc_lens: Vec::new(),
            avgdl: 0.0,
            num_docs: 0,
            inverted_index: HashMap::new(),
            idf: HashMap::new(),
        }
    }

    /// Creates a standard BM25Okapi ranker ($k_1=1.5, b=0.75, \delta=0.0$).
    pub fn new_okapi() -> Self {
        Self::new(1.5, 0.75, 0.0)
    }

    /// Creates a BM25+ ranker ($k_1=1.5, b=0.75, \delta=1.0$) preventing over-penalization of long documents.
    pub fn new_plus(delta: f64) -> Self {
        Self::new(1.5, 0.75, delta)
    }

    /// Fits the inverted index on a full corpus of documents.
    pub fn fit(&mut self, corpus: &[&str]) {
        self.doc_lens.clear();
        self.inverted_index.clear();
        self.idf.clear();
        self.num_docs = corpus.len();

        let mut total_len = 0usize;
        for (doc_id, &doc) in corpus.iter().enumerate() {
            let tokens = tokenize_words(doc);
            let len = tokens.len();
            self.doc_lens.push(len);
            total_len += len;

            let mut counts: HashMap<String, u32> = HashMap::new();
            for t in tokens {
                *counts.entry(t).or_insert(0) += 1;
            }

            for (term, tf) in counts {
                self.inverted_index
                    .entry(term)
                    .or_default()
                    .push((doc_id, tf));
            }
        }

        self.avgdl = if self.num_docs > 0 {
            (total_len as f64) / (self.num_docs as f64)
        } else {
            1.0
        };

        // Compute Robertson-Spärck Jones IDF: ln(1 + (N - n(q) + 0.5) / (n(q) + 0.5))
        let n = self.num_docs as f64;
        for (term, postings) in &self.inverted_index {
            let n_q = postings.len() as f64;
            let val = ((n - n_q + 0.5) / (n_q + 0.5) + 1.0).ln().max(0.0);
            self.idf.insert(term.clone(), val);
        }
    }

    /// Computes BM25 relevance scores for all documents given a query string.
    pub fn score_query(&self, query: &str) -> Vec<f64> {
        let mut scores = vec![0.0f64; self.num_docs];
        if self.num_docs == 0 {
            return scores;
        }

        let query_tokens = tokenize_words(query);
        let avgdl = if self.avgdl > 0.0 { self.avgdl } else { 1.0 };

        for q in query_tokens {
            if let Some(postings) = self.inverted_index.get(&q) {
                let idf_val = self.idf.get(&q).copied().unwrap_or(0.0);
                for &(doc_id, tf) in postings {
                    let dl = self.doc_lens[doc_id] as f64;
                    let tf_f = tf as f64;
                    let denom = tf_f + self.k1 * (1.0 - self.b + self.b * (dl / avgdl));
                    let term_score = idf_val * ((tf_f * (self.k1 + 1.0)) / (denom + 1e-12) + self.delta);
                    scores[doc_id] += term_score;
                }
            }
        }
        scores
    }

    /// Returns top $k$ scored documents `(doc_id, score)` sorted descending.
    pub fn top_k(&self, query: &str, k: usize) -> Vec<(usize, f64)> {
        let scores = self.score_query(query);
        let mut indexed: Vec<(usize, f64)> = scores.into_iter().enumerate().collect();
        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        if indexed.len() > k {
            indexed.truncate(k);
        }
        indexed
    }
}

/// Subword character n-gram morphology vectorizer (FastText style).
#[derive(Debug, Clone)]
pub struct FastTextCharNGramVectorizer {
    pub dim: usize,
    pub min_n: usize,
    pub max_n: usize,
    pub seed: u32,
    pub norm: bool,
}

impl FastTextCharNGramVectorizer {
    pub fn new(dim: usize, min_n: usize, max_n: usize) -> Result<Self> {
        if dim == 0 {
            return Err(Error::Msg("dim must be strictly positive".into()));
        }
        if min_n == 0 || min_n > max_n {
            return Err(Error::Msg(format!(
                "Invalid char n-gram range: ({min_n}, {max_n})"
            )));
        }
        Ok(Self {
            dim,
            min_n,
            max_n,
            seed: 42,
            norm: true,
        })
    }

    pub fn transform_word(&self, word: &str) -> Vec<f64> {
        let char_ngrams = extract_char_ngrams(word, self.min_n, self.max_n, true);
        let mut vec = vec![0.0f64; self.dim];
        for ng in &char_ngrams {
            let bytes = ng.as_bytes();
            let h1 = murmurhash3_32(bytes, self.seed);
            let h2 = murmurhash3_32(bytes, self.seed.wrapping_add(1));
            let idx = (h1 as usize) % self.dim;
            let sign = if (h2 & 1) == 0 { 1.0 } else { -1.0 };
            vec[idx] += sign;
        }
        if self.norm {
            let sum_sq: f64 = vec.iter().map(|&v| v * v).sum();
            let l2 = (sum_sq + 1e-12).sqrt();
            for v in &mut vec {
                *v /= l2;
            }
        }
        vec
    }

    pub fn transform_document(&self, doc: &str) -> Vec<f64> {
        let words = tokenize_words(doc);
        let mut doc_vec = vec![0.0f64; self.dim];
        if words.is_empty() {
            return doc_vec;
        }

        for w in &words {
            let w_vec = self.transform_word(w);
            for (d, &val) in doc_vec.iter_mut().zip(&w_vec) {
                *d += val;
            }
        }

        if self.norm {
            let sum_sq: f64 = doc_vec.iter().map(|&v| v * v).sum();
            let l2 = (sum_sq + 1e-12).sqrt();
            for v in &mut doc_vec {
                *v /= l2;
            }
        }
        doc_vec
    }

    pub fn transform_dense(&self, docs: &[&str]) -> Result<Tensor> {
        let rows = docs.len();
        let cols = self.dim;
        let mut flat = Vec::with_capacity(rows * cols);
        for &doc in docs {
            let v = self.transform_document(doc);
            flat.extend_from_slice(&v);
        }
        Tensor::from_slice_on(&flat, &[rows, cols], &Device::Cpu)
    }
}

/// Random Fourier Features (RFF) lifting text/bag-of-words vectors into continuous reproducing kernel Hilbert spaces (RKHS).
#[derive(Debug, Clone)]
pub struct KernelSpectralVectorizer {
    pub in_dim: usize,
    pub out_dim: usize,
    pub gamma: f64,
    pub weights: Vec<f64>,
    pub bias: Vec<f64>,
}

impl KernelSpectralVectorizer {
    pub fn new(in_dim: usize, out_dim: usize, gamma: f64, seed: u64) -> Result<Self> {
        if in_dim == 0 || out_dim == 0 {
            return Err(Error::Msg("Dimensions must be strictly positive".into()));
        }
        let std = (2.0 * gamma).sqrt();
        let mut weights = Vec::with_capacity(in_dim * out_dim);
        let mut bias = Vec::with_capacity(out_dim);

        // Deterministic pseudo-random generation using 64-bit LCG
        let mut state = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        for _ in 0..(in_dim * out_dim) {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            let u1 = ((state >> 33) as f64 + 1.0) / ((1u64 << 31) as f64 + 2.0);
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            let u2 = ((state >> 33) as f64 + 1.0) / ((1u64 << 31) as f64 + 2.0);
            // Box-Muller transform for standard Gaussian
            let z0 = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
            weights.push(z0 * std);
        }

        for _ in 0..out_dim {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            let u = (state >> 33) as f64 / ((1u64 << 31) as f64);
            bias.push(u * 2.0 * std::f64::consts::PI);
        }

        Ok(Self {
            in_dim,
            out_dim,
            gamma,
            weights,
            bias,
        })
    }

    /// Transforms an $[N, D_{\text{in}}]$ dense tensor into $[N, D_{\text{out}}]$ via $Z(x) = \sqrt{\frac{2}{D}} \cos(x W^\top + b)$.
    pub fn transform(&self, x: &Tensor) -> Result<Tensor> {
        let shape = x.shape();
        if shape.len() != 2 || shape[1] != self.in_dim {
            return Err(Error::Msg(format!(
                "KernelSpectralVectorizer expects shape [N, {}], got {:?}",
                self.in_dim, shape
            )));
        }
        let n = shape[0];
        let x_vec = x.to_vec::<f64>()?;
        let scale = (2.0 / (self.out_dim as f64)).sqrt();
        let mut out = vec![0.0f64; n * self.out_dim];

        for i in 0..n {
            let row_x = &x_vec[i * self.in_dim..(i + 1) * self.in_dim];
            for j in 0..self.out_dim {
                let mut dot = 0.0f64;
                for k in 0..self.in_dim {
                    dot += row_x[k] * self.weights[k * self.out_dim + j];
                }
                let angle = dot + self.bias[j];
                out[i * self.out_dim + j] = scale * angle.cos();
            }
        }

        Tensor::from_slice_on(&out, &[n, self.out_dim], &Device::Cpu)
    }
}

/// Native MinHash Vectorizer for estimating Jaccard similarities and near-duplicate detection.
#[derive(Debug, Clone)]
pub struct FastMinHashVectorizer {
    pub num_hashes: usize,
    pub ngram_range: (usize, usize),
    pub seed: u32,
    pub a_params: Vec<u32>,
    pub b_params: Vec<u32>,
}

impl FastMinHashVectorizer {
    pub fn new(num_hashes: usize, ngram_range: (usize, usize), seed: u32) -> Result<Self> {
        if num_hashes == 0 {
            return Err(Error::Msg("num_hashes must be positive".into()));
        }
        if ngram_range.0 == 0 || ngram_range.0 > ngram_range.1 {
            return Err(Error::Msg(format!(
                "Invalid ngram_range: ({}, {})",
                ngram_range.0, ngram_range.1
            )));
        }
        let mut a_params = Vec::with_capacity(num_hashes);
        let mut b_params = Vec::with_capacity(num_hashes);
        let mut state = (seed as u64)
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        for _ in 0..num_hashes {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let a = ((state >> 32) as u32) | 1; // ensure odd for full cycle in 2^32
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let b = (state >> 32) as u32;
            a_params.push(a);
            b_params.push(b);
        }
        Ok(Self {
            num_hashes,
            ngram_range,
            seed,
            a_params,
            b_params,
        })
    }

    /// Computes MinHash signatures for a document: `Vec<u32>` of length `num_hashes`.
    pub fn transform_document(&self, doc: &str) -> Vec<u32> {
        let tokens = tokenize_words(doc);
        let ngrams = extract_ngrams(&tokens, self.ngram_range.0, self.ngram_range.1);
        let mut sig = vec![u32::MAX; self.num_hashes];
        if ngrams.is_empty() {
            return sig;
        }

        for ng in ngrams {
            let h0 = murmurhash3_32(ng.as_bytes(), self.seed);
            for k in 0..self.num_hashes {
                let h_k = self.a_params[k].wrapping_mul(h0).wrapping_add(self.b_params[k]);
                if h_k < sig[k] {
                    sig[k] = h_k;
                }
            }
        }
        sig
    }

    /// Computes MinHash signatures for multiple documents returning an $[N, K]$ Tensor of f64.
    pub fn transform_dense(&self, docs: &[&str]) -> Result<Tensor> {
        let rows = docs.len();
        let cols = self.num_hashes;
        let mut flat = Vec::with_capacity(rows * cols);
        for &doc in docs {
            let sig = self.transform_document(doc);
            for s in sig {
                flat.push(s as f64);
            }
        }
        Tensor::from_slice_on(&flat, &[rows, cols], &Device::Cpu)
    }

    /// Estimates the Jaccard similarity between two MinHash signatures: fraction of matching hashes.
    pub fn jaccard_similarity(&self, sig_a: &[u32], sig_b: &[u32]) -> f64 {
        if sig_a.is_empty() || sig_a.len() != sig_b.len() {
            return 0.0;
        }
        let matches = sig_a.iter().zip(sig_b).filter(|&(a, b)| a == b).count();
        (matches as f64) / (sig_a.len() as f64)
    }
}

/// Native SimHash Vectorizer for cosine similarity approximation and bitwise Hamming indexing (Charikar 2002).
#[derive(Debug, Clone)]
pub struct FastSimHashVectorizer {
    pub dim: usize,
    pub ngram_range: (usize, usize),
    pub seed: u32,
}

impl FastSimHashVectorizer {
    pub fn new(dim: usize, ngram_range: (usize, usize), seed: u32) -> Result<Self> {
        if dim == 0 || dim > 1024 {
            return Err(Error::Msg("dim must be between 1 and 1024".into()));
        }
        if ngram_range.0 == 0 || ngram_range.0 > ngram_range.1 {
            return Err(Error::Msg(format!(
                "Invalid ngram_range: ({}, {})",
                ngram_range.0, ngram_range.1
            )));
        }
        Ok(Self {
            dim,
            ngram_range,
            seed,
        })
    }

    /// Computes real-valued accumulator vector $[D]$ and binary bit-signature for a document.
    pub fn transform_document(&self, doc: &str) -> (Vec<f64>, Vec<u8>) {
        let tokens = tokenize_words(doc);
        let ngrams = extract_ngrams(&tokens, self.ngram_range.0, self.ngram_range.1);
        let mut acc = vec![0.0f64; self.dim];
        if ngrams.is_empty() {
            let bits = vec![0u8; self.dim.div_ceil(8)];
            return (acc, bits);
        }

        // Count term frequency
        let mut counts: HashMap<String, f64> = HashMap::new();
        for ng in ngrams {
            *counts.entry(ng).or_insert(0.0) += 1.0;
        }

        for (term, weight) in counts {
            let bytes = term.as_bytes();
            for j in 0..self.dim {
                let h = murmurhash3_32(bytes, self.seed.wrapping_add(j as u32));
                let sign = if (h & 1) == 0 { 1.0 } else { -1.0 };
                acc[j] += sign * weight;
            }
        }

        // Generate bit-pack
        let mut bits = vec![0u8; self.dim.div_ceil(8)];
        for j in 0..self.dim {
            if acc[j] >= 0.0 {
                bits[j / 8] |= 1 << (j % 8);
            }
        }
        (acc, bits)
    }

    pub fn transform_dense(&self, docs: &[&str]) -> Result<Tensor> {
        let rows = docs.len();
        let cols = self.dim;
        let mut flat = Vec::with_capacity(rows * cols);
        for &doc in docs {
            let (acc, _) = self.transform_document(doc);
            let sum_sq: f64 = acc.iter().map(|&v| v * v).sum();
            let norm = (sum_sq + 1e-12).sqrt();
            for v in acc {
                flat.push(v / norm);
            }
        }
        Tensor::from_slice_on(&flat, &[rows, cols], &Device::Cpu)
    }

    /// Computes Hamming distance between two packed bit-signatures.
    pub fn hamming_distance(&self, bits_a: &[u8], bits_b: &[u8]) -> usize {
        bits_a.iter().zip(bits_b).map(|(&a, &b)| (a ^ b).count_ones() as usize).sum()
    }
}

/// Native 256-bin UTF-8 Byte Histogram Vectorizer.
#[derive(Debug, Clone)]
pub struct FastByteHistogramVectorizer {
    pub log_scale: bool,
    pub norm: Option<String>,
}

impl FastByteHistogramVectorizer {
    pub fn new(log_scale: bool, norm: Option<String>) -> Self {
        Self { log_scale, norm }
    }

    /// Computes 256-dimensional byte count vector for a text or byte slice.
    pub fn transform_bytes(&self, data: &[u8]) -> Vec<f64> {
        let mut hist = vec![0.0f64; 256];
        for &b in data {
            hist[b as usize] += 1.0;
        }

        if self.log_scale {
            for v in &mut hist {
                if *v > 0.0 {
                    *v = 1.0 + (*v).ln();
                }
            }
        }

        if let Some(ref norm_type) = self.norm {
            match norm_type.to_lowercase().as_str() {
                "l2" => {
                    let sum_sq: f64 = hist.iter().map(|&v| v * v).sum();
                    let l2 = (sum_sq + 1e-12).sqrt();
                    for v in &mut hist {
                        *v /= l2;
                    }
                }
                "l1" => {
                    let sum_abs: f64 = hist.iter().map(|&v| v.abs()).sum();
                    let l1 = (sum_abs + 1e-12).max(1e-12);
                    for v in &mut hist {
                        *v /= l1;
                    }
                }
                _ => {}
            }
        }

        hist
    }

    pub fn transform_dense(&self, docs: &[&str]) -> Result<Tensor> {
        let rows = docs.len();
        let cols = 256;
        let mut flat = Vec::with_capacity(rows * cols);
        for &doc in docs {
            let h = self.transform_bytes(doc.as_bytes());
            flat.extend_from_slice(&h);
        }
        Tensor::from_slice_on(&flat, &[rows, cols], &Device::Cpu)
    }
}

/// High-speed late-interaction MaxSim scorer (ColBERT, Khattab & Zaharia 2020).
pub struct FastMaxSimScorer;

impl FastMaxSimScorer {
    /// Computes late-interaction MaxSim score between query tokens $[S_q, D]$ and doc tokens $[S_d, D]$.
    pub fn score_maxsim(query: &Tensor, doc: &Tensor) -> Result<f64> {
        let q_shape = query.shape();
        let d_shape = doc.shape();
        if q_shape.len() != 2 || d_shape.len() != 2 || q_shape[1] != d_shape[1] {
            return Err(Error::Msg(format!(
                "MaxSim expects 2D tensors with matching dimension D, got {:?} and {:?}",
                q_shape, d_shape
            )));
        }
        let sq = q_shape[0];
        let sd = d_shape[0];
        let dim = q_shape[1];

        let q_vec = query.to_vec::<f64>()?;
        let d_vec = doc.to_vec::<f64>()?;

        let mut total_score = 0.0f64;
        for i in 0..sq {
            let q_row = &q_vec[i * dim..(i + 1) * dim];
            let mut max_sim = f64::NEG_INFINITY;
            for j in 0..sd {
                let d_row = &d_vec[j * dim..(j + 1) * dim];
                let mut dot = 0.0f64;
                for k in 0..dim {
                    dot += q_row[k] * d_row[k];
                }
                if dot > max_sim {
                    max_sim = dot;
                }
            }
            if max_sim != f64::NEG_INFINITY {
                total_score += max_sim;
            }
        }

        Ok(total_score)
    }

    /// Computes batched pairwise MaxSim scores between $B_q$ queries and $B_d$ documents.
    pub fn batch_maxsim(queries: &[Tensor], docs: &[Tensor]) -> Result<Tensor> {
        let bq = queries.len();
        let bd = docs.len();
        let mut scores = Vec::with_capacity(bq * bd);
        for q in queries {
            for d in docs {
                scores.push(Self::score_maxsim(q, d)?);
            }
        }
        Tensor::from_slice_on(&scores, &[bq, bd], &Device::Cpu)
    }
}

/// Native Latent Semantic Analysis (LSA) Vectorizer via Truncated SVD over TF-IDF.
#[derive(Debug, Clone)]
pub struct FastLsaVectorizer {
    pub tfidf: FastTfidfVectorizer,
    pub n_components: usize,
    pub components: Vec<f64>,
}

impl FastLsaVectorizer {
    pub fn new(n_components: usize, ngram_range: (usize, usize)) -> Result<Self> {
        if n_components == 0 {
            return Err(Error::Msg("n_components must be positive".into()));
        }
        let tfidf = FastTfidfVectorizer::new(ngram_range, true, Some("l2".into()))?;
        Ok(Self {
            tfidf,
            n_components,
            components: Vec::new(),
        })
    }

    pub fn fit(&mut self, corpus: &[&str]) -> Result<()> {
        self.tfidf.fit(corpus);
        let tfidf_dense = self.tfidf.transform_dense(corpus)?;
        let (_u, _s, vt) = tfidf_dense.svd()?;
        let (v_rows, v_cols) = (vt.shape()[0], vt.shape()[1]);
        let k = self.n_components.min(v_rows);

        let vt_vec = vt.to_vec::<f64>()?;
        let mut comps = Vec::with_capacity(k * v_cols);
        for i in 0..k {
            comps.extend_from_slice(&vt_vec[i * v_cols..(i + 1) * v_cols]);
        }
        self.components = comps;
        self.n_components = k;
        Ok(())
    }

    pub fn transform(&self, docs: &[&str]) -> Result<Tensor> {
        if self.components.is_empty() {
            return Err(Error::Msg(
                "FastLsaVectorizer must be fitted before transforming".into(),
            ));
        }
        let tfidf_dense = self.tfidf.transform_dense(docs)?;
        let n_docs = docs.len();
        let vocab_size = self.tfidf.count_vec.vocab_size();
        let x_vec = tfidf_dense.to_vec::<f64>()?;

        let k = self.n_components;
        let mut out = vec![0.0f64; n_docs * k];
        for i in 0..n_docs {
            let row_x = &x_vec[i * vocab_size..(i + 1) * vocab_size];
            for c in 0..k {
                let comp_row = &self.components[c * vocab_size..(c + 1) * vocab_size];
                let mut dot = 0.0f64;
                for j in 0..vocab_size {
                    dot += row_x[j] * comp_row[j];
                }
                out[i * k + c] = dot;
            }
        }

        Tensor::from_slice_on(&out, &[n_docs, k], &Device::Cpu)
    }
}

/// Real Spherical Harmonics Vectorizer for 3D coordinates and directions (degrees 0 to 3).
///
/// Maps $[N, 3]$ Cartesian coordinates $(x, y, z)$ into $[N, (L+1)^2]$ spherical harmonics representations.
#[derive(Debug, Clone)]
pub struct FastSphericalHarmonics {
    pub degree: usize,
}

impl FastSphericalHarmonics {
    pub fn new(degree: usize) -> Result<Self> {
        if degree > 3 {
            return Err(Error::Msg("degree must be between 0 and 3".into()));
        }
        Ok(Self { degree })
    }

    pub fn output_dim(&self) -> usize {
        (self.degree + 1) * (self.degree + 1)
    }

    /// Computes spherical harmonics basis vector for a single 3D vector $(x, y, z)$.
    pub fn transform_point(&self, p: [f64; 3]) -> Vec<f64> {
        let (x, y, z) = (p[0], p[1], p[2]);
        let norm = (x * x + y * y + z * z + 1e-12).sqrt();
        let (xn, yn, zn) = (x / norm, y / norm, z / norm);

        let mut out = Vec::with_capacity(self.output_dim());

        // Degree 0
        const C0: f64 = 0.28209479177387814;
        out.push(C0);

        if self.degree >= 1 {
            const C1: f64 = 0.4886025119029199;
            out.push(C1 * yn);
            out.push(C1 * zn);
            out.push(C1 * xn);
        }

        if self.degree >= 2 {
            const C2A: f64 = 1.0925484305920792;
            const C2B: f64 = 0.31539156525252005;
            const C2C: f64 = 0.5462742152960396;
            out.push(C2A * xn * yn);
            out.push(C2A * yn * zn);
            out.push(C2B * (3.0 * zn * zn - 1.0));
            out.push(C2A * xn * zn);
            out.push(C2C * (xn * xn - yn * yn));
        }

        if self.degree >= 3 {
            const C3A: f64 = 0.5900435899266435;
            const C3B: f64 = 1.445305721320277;
            const C3C: f64 = 0.4570457994644658;
            const C3D: f64 = 0.3731763325901154;
            out.push(C3A * yn * (3.0 * xn * xn - yn * yn));
            out.push(C3B * xn * yn * zn);
            out.push(C3C * yn * (5.0 * zn * zn - 1.0));
            out.push(C3D * zn * (5.0 * zn * zn - 3.0));
            out.push(C3C * xn * (5.0 * zn * zn - 1.0));
            out.push(0.5 * C3B * zn * (xn * xn - yn * yn));
            out.push(C3A * xn * (xn * xn - 3.0 * yn * yn));
        }

        out
    }

    /// Transforms an $[N, 3]$ Tensor of coordinates into an $[N, (L+1)^2]$ Tensor.
    pub fn transform(&self, points: &Tensor) -> Result<Tensor> {
        let shape = points.shape();
        if shape.len() != 2 || shape[1] != 3 {
            return Err(Error::Msg(format!(
                "FastSphericalHarmonics expects shape [N, 3], got {:?}",
                shape
            )));
        }
        let n = shape[0];
        let p_vec = points.to_vec::<f64>()?;
        let out_d = self.output_dim();
        let mut flat = Vec::with_capacity(n * out_d);

        for i in 0..n {
            let p = [p_vec[i * 3], p_vec[i * 3 + 1], p_vec[i * 3 + 2]];
            let basis = self.transform_point(p);
            flat.extend_from_slice(&basis);
        }

        Tensor::from_slice_on(&flat, &[n, out_d], &Device::Cpu)
    }
}

/// Continuous-to-discrete numerical binning vectorizer (KBinsDiscretizer).
#[derive(Debug, Clone)]
pub struct FastDiscretizer {
    pub n_bins: usize,
    pub bin_edges: Vec<f64>,
    pub one_hot: bool,
}

impl FastDiscretizer {
    pub fn new_uniform(n_bins: usize, min_val: f64, max_val: f64, one_hot: bool) -> Result<Self> {
        if n_bins < 2 {
            return Err(Error::Msg("n_bins must be >= 2".into()));
        }
        if min_val >= max_val {
            return Err(Error::Msg("min_val must be strictly less than max_val".into()));
        }
        let step = (max_val - min_val) / (n_bins as f64);
        let mut bin_edges = Vec::with_capacity(n_bins - 1);
        for i in 1..n_bins {
            bin_edges.push(min_val + (i as f64) * step);
        }
        Ok(Self {
            n_bins,
            bin_edges,
            one_hot,
        })
    }

    pub fn fit_quantile(&mut self, values: &[f64]) -> Result<()> {
        if values.is_empty() {
            return Err(Error::Msg("Cannot fit quantile on empty slice".into()));
        }
        let mut sorted = values.to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let n = sorted.len();

        self.bin_edges.clear();
        for i in 1..self.n_bins {
            let q = (i as f64) / (self.n_bins as f64);
            let idx = ((q * (n as f64)).round() as usize).min(n - 1);
            self.bin_edges.push(sorted[idx]);
        }
        Ok(())
    }

    /// Assigns a continuous value $x$ to bin $b \in [0, \text{n\_bins}-1]$.
    pub fn assign_bin(&self, x: f64) -> usize {
        match self.bin_edges.binary_search_by(|e| e.partial_cmp(&x).unwrap_or(std::cmp::Ordering::Equal)) {
            Ok(idx) => (idx + 1).min(self.n_bins - 1),
            Err(idx) => idx.min(self.n_bins - 1),
        }
    }

    pub fn transform(&self, values: &[f64]) -> Result<Tensor> {
        let n = values.len();
        if self.one_hot {
            let mut flat = vec![0.0f64; n * self.n_bins];
            for (i, &v) in values.iter().enumerate() {
                let bin = self.assign_bin(v);
                flat[i * self.n_bins + bin] = 1.0;
            }
            Tensor::from_slice_on(&flat, &[n, self.n_bins], &Device::Cpu)
        } else {
            let mut flat = Vec::with_capacity(n);
            for &v in values {
                flat.push(self.assign_bin(v) as f64);
            }
            Tensor::from_slice_on(&flat, &[n, 1], &Device::Cpu)
        }
    }
}

/// Native Dictionary / Tabular Feature Vectorizer.
#[derive(Debug, Clone, PartialEq)]
pub enum DictVal {
    Str(String),
    Num(f64),
    Bool(bool),
}

#[derive(Debug, Clone)]
pub struct FastDictVectorizer {
    pub feature_names: Vec<String>,
    pub vocab: HashMap<String, usize>,
    pub sort: bool,
}

impl FastDictVectorizer {
    pub fn new(sort: bool) -> Self {
        Self {
            feature_names: Vec::new(),
            vocab: HashMap::new(),
            sort,
        }
    }

    pub fn fit(&mut self, samples: &[HashMap<String, DictVal>]) {
        let mut names = HashSet::new();
        for sample in samples {
            for (k, v) in sample {
                match v {
                    DictVal::Str(s) => {
                        names.insert(format!("{k}={s}"));
                    }
                    DictVal::Bool(b) => {
                        names.insert(format!("{k}={b}"));
                    }
                    DictVal::Num(_) => {
                        names.insert(k.clone());
                    }
                }
            }
        }
        let mut list: Vec<String> = names.into_iter().collect();
        if self.sort {
            list.sort();
        }
        self.vocab.clear();
        for (i, name) in list.iter().enumerate() {
            self.vocab.insert(name.clone(), i);
        }
        self.feature_names = list;
    }

    pub fn transform_sparse(&self, samples: &[HashMap<String, DictVal>]) -> Result<SparseCsr> {
        let n_samples = samples.len();
        let n_features = self.feature_names.len();
        let mut row_ptr = Vec::with_capacity(n_samples + 1);
        row_ptr.push(0);
        let mut col_idx = Vec::new();
        let mut values = Vec::new();

        for sample in samples {
            let mut row_entries: HashMap<usize, f64> = HashMap::new();
            for (k, v) in sample {
                let (fname, val) = match v {
                    DictVal::Str(s) => (format!("{k}={s}"), 1.0),
                    DictVal::Bool(b) => (format!("{k}={b}"), if *b { 1.0 } else { 0.0 }),
                    DictVal::Num(num) => (k.clone(), *num),
                };
                if let Some(&col) = self.vocab.get(&fname) {
                    row_entries.insert(col, val);
                }
            }
            let mut sorted: Vec<(usize, f64)> = row_entries.into_iter().collect();
            sorted.sort_by_key(|&(c, _)| c);
            for (c, v) in sorted {
                col_idx.push(c);
                values.push(v);
            }
            row_ptr.push(col_idx.len());
        }

        SparseCsr::new(row_ptr, col_idx, values, [n_samples, n_features])
    }

    pub fn transform_dense(&self, samples: &[HashMap<String, DictVal>]) -> Result<Tensor> {
        self.transform_sparse(samples)?.to_dense(DType::F32, &Device::Cpu)
    }

    pub fn num_features(&self) -> usize {
        self.feature_names.len()
    }
}

/// Native One-Hot Vectorizer for categorical sequences and tabular columns.
#[derive(Debug, Clone)]
pub struct FastOneHotVectorizer {
    pub categories: Vec<Vec<String>>,
    pub cat_to_idx: Vec<HashMap<String, usize>>,
    pub drop_first: bool,
    pub ignore_unknown: bool,
}

impl FastOneHotVectorizer {
    pub fn new(drop_first: bool, ignore_unknown: bool) -> Self {
        Self {
            categories: Vec::new(),
            cat_to_idx: Vec::new(),
            drop_first,
            ignore_unknown,
        }
    }

    pub fn fit(&mut self, table: &[Vec<String>]) -> Result<()> {
        if table.is_empty() {
            return Ok(());
        }
        let n_cols = table[0].len();
        self.categories = Vec::with_capacity(n_cols);
        self.cat_to_idx = Vec::with_capacity(n_cols);

        for c in 0..n_cols {
            let mut unique: HashSet<String> = HashSet::new();
            for row in table {
                if c < row.len() {
                    unique.insert(row[c].clone());
                }
            }
            let mut sorted: Vec<String> = unique.into_iter().collect();
            sorted.sort();
            if self.drop_first && sorted.len() > 1 {
                sorted.remove(0);
            }
            let mut map = HashMap::with_capacity(sorted.len());
            for (i, cat) in sorted.iter().enumerate() {
                map.insert(cat.clone(), i);
            }
            self.categories.push(sorted);
            self.cat_to_idx.push(map);
        }
        Ok(())
    }

    pub fn total_features(&self) -> usize {
        self.categories.iter().map(|c| c.len()).sum()
    }

    pub fn transform(&self, table: &[Vec<String>]) -> Result<Tensor> {
        let n_samples = table.len();
        let n_cols = self.categories.len();
        let total_feats = self.total_features();
        let mut flat = vec![0.0f64; n_samples * total_feats];

        for (i, row) in table.iter().enumerate() {
            let mut offset = 0;
            for c in 0..n_cols {
                if c < row.len() {
                    let cat = &row[c];
                    if let Some(&idx) = self.cat_to_idx[c].get(cat) {
                        flat[i * total_feats + offset + idx] = 1.0;
                    } else if !self.ignore_unknown {
                        return Err(Error::Msg(format!("Unknown category '{cat}' in column {c}")));
                    }
                }
                offset += self.categories[c].len();
            }
        }

        Tensor::from_slice_on(&flat, &[n_samples, total_feats], &Device::Cpu)
    }
}

/// Native Hierarchical JSON and Tree Structure Vectorizer.
#[derive(Debug, Clone)]
pub struct FastJsonTreeVectorizer {
    pub n_features: usize,
    pub decay_rate: f64,
    pub seed: u32,
}

impl FastJsonTreeVectorizer {
    pub fn new(n_features: usize, decay_rate: f64, seed: u32) -> Result<Self> {
        if n_features == 0 {
            return Err(Error::Msg("n_features must be positive".into()));
        }
        Ok(Self {
            n_features,
            decay_rate,
            seed,
        })
    }

    pub fn transform_entries(&self, entries: &[(&str, f64)]) -> Vec<f64> {
        let mut vec = vec![0.0f64; self.n_features];
        for &(path, val) in entries {
            let h1 = murmurhash3_32(path.as_bytes(), self.seed);
            let h2 = murmurhash3_32(path.as_bytes(), self.seed.wrapping_add(1));
            let idx = (h1 as usize) % self.n_features;
            let sign = if (h2 & 1) == 0 { 1.0 } else { -1.0 };
            vec[idx] += sign * val;
        }
        let sum_sq: f64 = vec.iter().map(|&v| v * v).sum();
        let norm = (sum_sq + 1e-12).sqrt();
        for v in &mut vec {
            *v /= norm;
        }
        vec
    }

    pub fn transform_dense(&self, batch: &[Vec<(&str, f64)>]) -> Result<Tensor> {
        let n_samples = batch.len();
        let mut flat = Vec::with_capacity(n_samples * self.n_features);
        for item in batch {
            let v = self.transform_entries(item);
            flat.extend_from_slice(&v);
        }
        Tensor::from_slice_on(&flat, &[n_samples, self.n_features], &Device::Cpu)
    }
}

/// Native Time-Series and Sensor Stream Spectral Vectorizer.
#[derive(Debug, Clone)]
pub struct FastTimeSeriesSpectralVectorizer {
    pub n_fft_bands: usize,
    pub normalize: bool,
}

impl FastTimeSeriesSpectralVectorizer {
    pub fn new(n_fft_bands: usize, normalize: bool) -> Result<Self> {
        if n_fft_bands == 0 {
            return Err(Error::Msg("n_fft_bands must be positive".into()));
        }
        Ok(Self {
            n_fft_bands,
            normalize,
        })
    }

    pub fn output_dim(&self) -> usize {
        5 + self.n_fft_bands + 2 + 4 + 1
    }

    pub fn transform_series(&self, x: &[f64]) -> Vec<f64> {
        let n = x.len();
        if n == 0 {
            return vec![0.0f64; self.output_dim()];
        }

        let mean = x.iter().sum::<f64>() / (n as f64);
        let var = x.iter().map(|&v| (v - mean) * (v - mean)).sum::<f64>() / (n as f64);
        let std = (var + 1e-12).sqrt();
        let skew = x.iter().map(|&v| ((v - mean) / std).powi(3)).sum::<f64>() / (n as f64);
        let kurt = x.iter().map(|&v| ((v - mean) / std).powi(4)).sum::<f64>() / (n as f64) - 3.0;

        let mut min_val = x[0];
        let mut max_val = x[0];
        for &v in x {
            if v < min_val {
                min_val = v;
            }
            if v > max_val {
                max_val = v;
            }
        }
        let ptp = max_val - min_val;

        let n_freqs = n / 2 + 1;
        let mut power = Vec::with_capacity(n_freqs);
        let pi2 = 2.0 * std::f64::consts::PI;
        for k in 0..n_freqs {
            let mut re = 0.0f64;
            let mut im = 0.0f64;
            for (t, &val) in x.iter().enumerate() {
                let angle = pi2 * (k as f64) * (t as f64) / (n as f64);
                re += val * angle.cos();
                im -= val * angle.sin();
            }
            power.push(re * re + im * im);
        }

        let total_power = power.iter().sum::<f64>().max(1e-12);

        let band_size = n_freqs.div_ceil(self.n_fft_bands);
        let mut band_powers = Vec::with_capacity(self.n_fft_bands);
        for b in 0..self.n_fft_bands {
            let start = (b * band_size).min(n_freqs);
            let end = ((b + 1) * band_size).min(n_freqs);
            let p_sum: f64 = power[start..end].iter().sum();
            band_powers.push(p_sum / total_power);
        }

        let mut num_centroid = 0.0f64;
        for (k, &p) in power.iter().enumerate() {
            num_centroid += (k as f64) * p;
        }
        let centroid = num_centroid / (total_power * (n_freqs.max(2) - 1) as f64);

        let log_sum: f64 = power.iter().map(|&p| (p.max(1e-12)).ln()).sum();
        let geo_mean = (log_sum / (n_freqs as f64)).exp();
        let arith_mean = total_power / (n_freqs as f64);
        let flatness = geo_mean / (arith_mean + 1e-12);

        let mut autocorr = Vec::with_capacity(4);
        for &lag in &[1usize, 2, 5, 10] {
            if lag < n {
                let mut sum = 0.0f64;
                for t in 0..(n - lag) {
                    sum += (x[t] - mean) * (x[t + lag] - mean);
                }
                autocorr.push(sum / ((n as f64) * var + 1e-12));
            } else {
                autocorr.push(0.0);
            }
        }

        let mut zcr_count = 0usize;
        for t in 1..n {
            if (x[t] - mean).signum() != (x[t - 1] - mean).signum() {
                zcr_count += 1;
            }
        }
        let zcr = (zcr_count as f64) / ((n - 1).max(1) as f64);

        let mut out = Vec::with_capacity(self.output_dim());
        out.push(mean);
        out.push(std);
        out.push(skew);
        out.push(kurt);
        out.push(ptp);
        out.extend(band_powers);
        out.push(centroid);
        out.push(flatness);
        out.extend(autocorr);
        out.push(zcr);

        if self.normalize {
            let sum_sq: f64 = out.iter().map(|&v| v * v).sum();
            let l2 = (sum_sq + 1e-12).sqrt();
            for v in &mut out {
                *v /= l2;
            }
        }

        out
    }

    pub fn transform(&self, series_list: &[Vec<f64>]) -> Result<Tensor> {
        let n = series_list.len();
        let dim = self.output_dim();
        let mut flat = Vec::with_capacity(n * dim);
        for s in series_list {
            let feats = self.transform_series(s);
            flat.extend_from_slice(&feats);
        }
        Tensor::from_slice_on(&flat, &[n, dim], &Device::Cpu)
    }
}

/// Native Truncated Rough Path Signature Vectorizer (Chen's iterated integrals).
#[derive(Debug, Clone)]
pub struct FastPathSignatureVectorizer {
    pub depth: usize,
}

impl FastPathSignatureVectorizer {
    pub fn new(depth: usize) -> Result<Self> {
        if depth != 1 && depth != 2 {
            return Err(Error::Msg("depth must be 1 or 2".into()));
        }
        Ok(Self { depth })
    }

    pub fn output_dim(&self, d: usize) -> usize {
        if self.depth == 1 {
            d
        } else {
            d + d * d
        }
    }

    pub fn signature_single(&self, path: &[Vec<f64>], d: usize) -> Vec<f64> {
        let t_steps = path.len();
        let out_dim = self.output_dim(d);
        if t_steps < 2 {
            return vec![0.0f64; out_dim];
        }

        let mut level1 = vec![0.0f64; d];
        for col in 0..d {
            level1[col] = path[t_steps - 1][col] - path[0][col];
        }

        if self.depth == 1 {
            return level1;
        }

        let mut level2 = vec![0.0f64; d * d];
        let mut running_x = vec![0.0f64; d];

        for t in 0..(t_steps - 1) {
            let mut dx = vec![0.0f64; d];
            for c in 0..d {
                dx[c] = path[t + 1][c] - path[t][c];
            }
            for i in 0..d {
                let x_mid = running_x[i] + 0.5 * dx[i];
                for j in 0..d {
                    level2[i * d + j] += x_mid * dx[j];
                }
            }
            for c in 0..d {
                running_x[c] += dx[c];
            }
        }

        let mut out = Vec::with_capacity(out_dim);
        out.extend(level1);
        out.extend(level2);
        out
    }

    pub fn transform(&self, paths: &[Vec<Vec<f64>>], d: usize) -> Result<Tensor> {
        let n = paths.len();
        let dim = self.output_dim(d);
        let mut flat = Vec::with_capacity(n * dim);
        for p in paths {
            let sig = self.signature_single(p, d);
            flat.extend_from_slice(&sig);
        }
        Tensor::from_slice_on(&flat, &[n, dim], &Device::Cpu)
    }
}

/// Native Deep Sets Permutation-Invariant Vectorizer (Zaheer et al. 2017).
#[derive(Debug, Clone)]
pub struct FastDeepSetsVectorizer {
    pub in_features: usize,
    pub out_features: usize,
    pub pooling: String,
    pub weights: Vec<f64>,
    pub bias: Vec<f64>,
}

impl FastDeepSetsVectorizer {
    pub fn new(in_features: usize, out_features: usize, pooling: &str, seed: u64) -> Result<Self> {
        if in_features == 0 || out_features == 0 {
            return Err(Error::Msg("features must be positive".into()));
        }
        let std = 1.0 / (in_features as f64).sqrt();
        let mut weights = Vec::with_capacity(in_features * out_features);
        let bias = vec![0.0f64; out_features];

        let mut state = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        for _ in 0..(in_features * out_features) {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let u1 = ((state >> 33) as f64 + 1.0) / ((1u64 << 31) as f64 + 2.0);
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let u2 = ((state >> 33) as f64 + 1.0) / ((1u64 << 31) as f64 + 2.0);
            let z = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
            weights.push(z * std);
        }

        Ok(Self {
            in_features,
            out_features,
            pooling: pooling.to_lowercase(),
            weights,
            bias,
        })
    }

    pub fn transform_set(&self, set: &[Vec<f64>]) -> Vec<f64> {
        let n_items = set.len();
        if n_items == 0 {
            return vec![0.0f64; self.out_features];
        }

        let mut projected = vec![0.0f64; n_items * self.out_features];
        for (i, item) in set.iter().enumerate() {
            for j in 0..self.out_features {
                let mut sum = self.bias[j];
                for k in 0..self.in_features.min(item.len()) {
                    sum += item[k] * self.weights[k * self.out_features + j];
                }
                projected[i * self.out_features + j] = sum.max(0.0);
            }
        }

        let mut pooled = vec![0.0f64; self.out_features];
        match self.pooling.as_str() {
            "sum" => {
                for i in 0..n_items {
                    for j in 0..self.out_features {
                        pooled[j] += projected[i * self.out_features + j];
                    }
                }
            }
            "max" => {
                pooled.fill(f64::NEG_INFINITY);
                for i in 0..n_items {
                    for j in 0..self.out_features {
                        let v = projected[i * self.out_features + j];
                        if v > pooled[j] {
                            pooled[j] = v;
                        }
                    }
                }
            }
            "lse" => {
                let mut max_val = vec![f64::NEG_INFINITY; self.out_features];
                for i in 0..n_items {
                    for j in 0..self.out_features {
                        let v = projected[i * self.out_features + j];
                        if v > max_val[j] {
                            max_val[j] = v;
                        }
                    }
                }
                for i in 0..n_items {
                    for j in 0..self.out_features {
                        let v = projected[i * self.out_features + j];
                        pooled[j] += (v - max_val[j]).exp();
                    }
                }
                for j in 0..self.out_features {
                    pooled[j] = max_val[j] + (pooled[j] + 1e-12).ln();
                }
            }
            _ => {
                for i in 0..n_items {
                    for j in 0..self.out_features {
                        pooled[j] += projected[i * self.out_features + j];
                    }
                }
                for j in 0..self.out_features {
                    pooled[j] /= n_items as f64;
                }
            }
        }

        let sum_sq: f64 = pooled.iter().map(|&v| v * v).sum();
        let l2 = (sum_sq + 1e-12).sqrt();
        for v in &mut pooled {
            *v /= l2;
        }
        pooled
    }

    pub fn transform(&self, sets: &[Vec<Vec<f64>>]) -> Result<Tensor> {
        let n = sets.len();
        let mut flat = Vec::with_capacity(n * self.out_features);
        for s in sets {
            let p = self.transform_set(s);
            flat.extend_from_slice(&p);
        }
        Tensor::from_slice_on(&flat, &[n, self.out_features], &Device::Cpu)
    }
}

/// Native 1-Weisfeiler-Lehman (1-WL) Graph Subtree Pattern Vectorizer.
#[derive(Debug, Clone)]
pub struct FastWeisfeilerLehmanVectorizer {
    pub iterations: usize,
    pub n_features: usize,
    pub seed: u32,
}

impl FastWeisfeilerLehmanVectorizer {
    pub fn new(iterations: usize, n_features: usize, seed: u32) -> Result<Self> {
        if n_features == 0 {
            return Err(Error::Msg("n_features must be positive".into()));
        }
        Ok(Self {
            iterations,
            n_features,
            seed,
        })
    }

    pub fn transform_graph(&self, adj: &[Vec<i32>], node_labels: &[String]) -> Vec<f64> {
        let n_nodes = adj.len();
        if n_nodes == 0 {
            return vec![0.0f64; self.n_features];
        }

        let mut feature_counts: HashMap<String, f64> = HashMap::new();
        for l in node_labels {
            *feature_counts.entry(format!("0:{l}")).or_insert(0.0) += 1.0;
        }

        let mut curr_labels = node_labels.to_vec();
        for h in 1..=self.iterations {
            let mut next_labels = Vec::with_capacity(n_nodes);
            for i in 0..n_nodes {
                let mut neighbors = Vec::new();
                for j in 0..n_nodes {
                    if adj[i][j] > 0 {
                        neighbors.push(curr_labels[j].clone());
                    }
                }
                neighbors.sort();
                let pattern = format!("{}_{}", curr_labels[i], neighbors.join("_"));
                *feature_counts.entry(format!("{h}:{pattern}")).or_insert(0.0) += 1.0;
                next_labels.push(pattern);
            }
            curr_labels = next_labels;
        }

        let mut vec = vec![0.0f64; self.n_features];
        for (pattern, count) in feature_counts {
            let h = murmurhash3_32(pattern.as_bytes(), self.seed);
            let idx = (h as usize) % self.n_features;
            vec[idx] += count;
        }

        let sum_sq: f64 = vec.iter().map(|&v| v * v).sum();
        let l2 = (sum_sq + 1e-12).sqrt();
        for v in &mut vec {
            *v /= l2;
        }
        vec
    }

    pub fn transform_dense(&self, graphs: &[(&[Vec<i32>], &[String])]) -> Result<Tensor> {
        let n = graphs.len();
        let mut flat = Vec::with_capacity(n * self.n_features);
        for &(adj, labels) in graphs {
            let v = self.transform_graph(adj, labels);
            flat.extend_from_slice(&v);
        }
        Tensor::from_slice_on(&flat, &[n, self.n_features], &Device::Cpu)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_murmurhash3_32_canonical() {
        assert_eq!(murmurhash3_32(b"", 0), 0);
        assert_eq!(murmurhash3_32(b"hello", 0), 0x248bfa47);
        assert_eq!(murmurhash3_32(b"oxideon", 42), 0xd7e5b211);
    }

    #[test]
    fn test_tokenize_and_ngrams() {
        let text = "Deep Learning with Oxideon!";
        let tokens = tokenize_words(text);
        assert_eq!(tokens, vec!["deep", "learning", "with", "oxideon"]);

        let ngrams = extract_ngrams(&tokens, 1, 2);
        assert_eq!(
            ngrams,
            vec![
                "deep",
                "learning",
                "with",
                "oxideon",
                "deep learning",
                "learning with",
                "with oxideon"
            ]
        );
    }

    #[test]
    fn test_char_ngrams() {
        let char_ng = extract_char_ngrams("deep", 3, 3, true);
        assert_eq!(char_ng, vec!["<de", "dee", "eep", "ep>"]);
    }

    #[test]
    fn test_count_vectorizer_basic() {
        let docs = ["hello world", "world of deep learning", "deep hello"];
        let mut cv = FastCountVectorizer::new((1, 1), false, None).unwrap();
        cv.fit(&docs);
        assert_eq!(cv.vocab_size(), 5);

        let csr = cv.transform(&docs).unwrap();
        assert_eq!(csr.shape, [3, 5]);
        assert_eq!(csr.row_ptr, vec![0, 2, 6, 8]);
    }

    #[test]
    fn test_tfidf_vectorizer_normalization() {
        let docs = ["artificial intelligence", "intelligence and deep learning"];
        let mut tfidf = FastTfidfVectorizer::new((1, 1), false, Some("l2".into())).unwrap();
        tfidf.fit(&docs);
        let csr = tfidf.transform(&docs).unwrap();
        assert_eq!(csr.shape[0], 2);

        // Check L2 norm of each row is 1.0
        let norm0: f64 = csr.values[csr.row_ptr[0]..csr.row_ptr[1]]
            .iter()
            .map(|v| v * v)
            .sum::<f64>()
            .sqrt();
        assert!((norm0 - 1.0).abs() < 1e-4);
    }

    #[test]
    fn test_hashing_vectorizer() {
        let docs = ["quick brown fox", "jumped over the lazy dog"];
        let hv = FastHashingVectorizer::new(1024, (1, 1), true, Some("l2".into())).unwrap();
        let csr = hv.transform(&docs).unwrap();
        assert_eq!(csr.shape, [2, 1024]);
        assert!(csr.nnz() > 0);
    }

    #[test]
    fn test_bm25_index_ranking() {
        let corpus = [
            "the quick brown fox jumps over the lazy dog",
            "deep learning neural network tensor gradient autograd",
            "quick brown fox runs in the forest",
        ];
        let mut bm25 = FastBM25Index::new_okapi();
        bm25.fit(&corpus);

        let scores = bm25.score_query("quick brown fox");
        assert_eq!(scores.len(), 3);
        assert!(scores[0] > 0.0);
        assert!(scores[2] > 0.0);
        assert_eq!(scores[1], 0.0); // Doc 1 has none of the query terms

        let top = bm25.top_k("neural network", 1);
        assert_eq!(top.len(), 1);
        assert_eq!(top[0].0, 1); // Doc 1 is top ranked
    }

    #[test]
    fn test_fasttext_char_ngrams() {
        let ft = FastTextCharNGramVectorizer::new(64, 3, 5).unwrap();
        let doc1 = ft.transform_document("computation");
        let doc2 = ft.transform_document("computations");
        let doc3 = ft.transform_document("butterfly");

        // Cosine similarity between computation and computations should be high
        let dot12: f64 = doc1.iter().zip(&doc2).map(|(a, b)| a * b).sum();
        let dot13: f64 = doc1.iter().zip(&doc3).map(|(a, b)| a * b).sum();
        assert!(
            dot12 > dot13,
            "computation should be closer to computations than butterfly"
        );
    }

    #[test]
    fn test_kernel_spectral_vectorizer() {
        let rff = KernelSpectralVectorizer::new(4, 16, 0.5, 42).unwrap();
        let x = Tensor::from_slice_on(
            &[1.0f64, 2.0, 3.0, 4.0, 1.1, 2.1, 2.9, 4.0],
            &[2, 4],
            &Device::Cpu,
        )
        .unwrap();
        let z = rff.transform(&x).unwrap();
        assert_eq!(z.shape(), &[2, 16]);
    }

    #[test]
    fn test_minhash_vectorizer() {
        let mh = FastMinHashVectorizer::new(64, (1, 1), 42).unwrap();
        let doc1 = "the quick brown fox jumps over the lazy dog";
        let doc2 = "the fast brown fox jumps over the lazy dog";
        let doc3 = "quantum gravity and string theory";

        let sig1 = mh.transform_document(doc1);
        let sig2 = mh.transform_document(doc2);
        let sig3 = mh.transform_document(doc3);

        let sim12 = mh.jaccard_similarity(&sig1, &sig2);
        let sim13 = mh.jaccard_similarity(&sig1, &sig3);
        assert!(sim12 > sim13);
        assert!(sim12 > 0.5);

        let dense = mh.transform_dense(&[doc1, doc2]).unwrap();
        assert_eq!(dense.shape(), &[2, 64]);
    }

    #[test]
    fn test_simhash_vectorizer() {
        let sh = FastSimHashVectorizer::new(64, (1, 1), 42).unwrap();
        let doc1 = "neural sequence transduction models";
        let doc2 = "neural sequence transduction models with attention";
        let doc3 = "cooking recipes with potatoes";

        let (_, bits1) = sh.transform_document(doc1);
        let (_, bits2) = sh.transform_document(doc2);
        let (_, bits3) = sh.transform_document(doc3);

        let dist12 = sh.hamming_distance(&bits1, &bits2);
        let dist13 = sh.hamming_distance(&bits1, &bits3);
        assert!(dist12 < dist13);

        let dense = sh.transform_dense(&[doc1, doc2]).unwrap();
        assert_eq!(dense.shape(), &[2, 64]);
    }

    #[test]
    fn test_byte_histogram_vectorizer() {
        let bh = FastByteHistogramVectorizer::new(true, Some("l2".into()));
        let h = bh.transform_bytes(b"hello world");
        assert_eq!(h.len(), 256);
        let norm: f64 = h.iter().map(|v| v * v).sum::<f64>().sqrt();
        assert!((norm - 1.0).abs() < 1e-4);
    }

    #[test]
    fn test_maxsim_scorer() {
        // Query: 2 tokens, dim 4
        let q = Tensor::from_slice_on(
            &[1.0f64, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
            &[2, 4],
            &Device::Cpu,
        )
        .unwrap();
        // Doc: 3 tokens, dim 4
        let d = Tensor::from_slice_on(
            &[
                0.9f64, 0.0, 0.0, 0.0, // matches token 0 closely
                0.0, 0.8, 0.0, 0.0, // matches token 1 closely
                0.0, 0.0, 1.0, 0.0,
            ],
            &[3, 4],
            &Device::Cpu,
        )
        .unwrap();

        let score = FastMaxSimScorer::score_maxsim(&q, &d).unwrap();
        assert!((score - 1.7).abs() < 1e-4);
    }

    #[test]
    fn test_spherical_harmonics() {
        let sh = FastSphericalHarmonics::new(2).unwrap();
        assert_eq!(sh.output_dim(), 9);

        let p = Tensor::from_slice_on(&[0.0f64, 0.0, 1.0, 1.0, 0.0, 0.0], &[2, 3], &Device::Cpu)
            .unwrap();
        let basis = sh.transform(&p).unwrap();
        assert_eq!(basis.shape(), &[2, 9]);
    }

    #[test]
    fn test_discretizer() {
        let disc = FastDiscretizer::new_uniform(4, 0.0, 100.0, true).unwrap();
        let vals = [10.0f64, 30.0, 60.0, 90.0];
        let t = disc.transform(&vals).unwrap();
        assert_eq!(t.shape(), &[4, 4]);

        // Each row should be one-hot
        let t_vec = t.to_vec::<f64>().unwrap();
        assert_eq!(t_vec[0..4], [1.0, 0.0, 0.0, 0.0]); // bin 0
        assert_eq!(t_vec[4..8], [0.0, 1.0, 0.0, 0.0]); // bin 1
        assert_eq!(t_vec[8..12], [0.0, 0.0, 1.0, 0.0]); // bin 2
        assert_eq!(t_vec[12..16], [0.0, 0.0, 0.0, 1.0]); // bin 3
    }

    #[test]
    fn test_dict_vectorizer() {
        let mut dv = FastDictVectorizer::new(true);
        let mut sample1 = HashMap::new();
        sample1.insert("city".to_string(), DictVal::Str("NY".to_string()));
        sample1.insert("temp".to_string(), DictVal::Num(72.0));
        sample1.insert("rain".to_string(), DictVal::Bool(false));

        let mut sample2 = HashMap::new();
        sample2.insert("city".to_string(), DictVal::Str("SF".to_string()));
        sample2.insert("temp".to_string(), DictVal::Num(65.0));
        sample2.insert("rain".to_string(), DictVal::Bool(true));

        let samples = vec![sample1, sample2];
        dv.fit(&samples);
        assert_eq!(dv.num_features(), 5); // city=NY, city=SF, rain=false, rain=true, temp

        let dense = dv.transform_dense(&samples).unwrap();
        assert_eq!(dense.shape(), &[2, 5]);

        let sparse = dv.transform_sparse(&samples).unwrap();
        assert_eq!(sparse.shape, [2, 5]);
        assert_eq!(sparse.nnz(), 6); // 3 non-zeros per sample
    }

    #[test]
    fn test_one_hot_vectorizer() {
        let mut ohv = FastOneHotVectorizer::new(false, true);
        let table = vec![
            vec!["cat".to_string(), "red".to_string()],
            vec!["dog".to_string(), "blue".to_string()],
            vec!["cat".to_string(), "blue".to_string()],
        ];
        ohv.fit(&table).unwrap();
        assert_eq!(ohv.total_features(), 4); // 2 animals + 2 colors

        let t = ohv.transform(&table).unwrap();
        assert_eq!(t.shape(), &[3, 4]);

        // Test unknown category handling
        let unknown_row = vec![vec!["bird".to_string(), "green".to_string()]];
        let t_unk = ohv.transform(&unknown_row).unwrap();
        assert_eq!(t_unk.shape(), &[1, 4]);
        let unk_slice = t_unk.to_vec::<f64>().unwrap();
        assert_eq!(unk_slice, vec![0.0, 0.0, 0.0, 0.0]);
    }

    #[test]
    fn test_json_tree_vectorizer() {
        let jtv = FastJsonTreeVectorizer::new(32, 0.5, 42).unwrap();
        let entries1 = vec![("user.profile.age", 30.0), ("user.active", 1.0)];
        let v1 = jtv.transform_entries(&entries1);
        assert_eq!(v1.len(), 32);

        let sum_sq: f64 = v1.iter().map(|&x| x * x).sum();
        assert!((sum_sq - 1.0).abs() < 1e-4);

        let batch = vec![entries1];
        let t = jtv.transform_dense(&batch).unwrap();
        assert_eq!(t.shape(), &[1, 32]);
    }

    #[test]
    fn test_time_series_spectral_vectorizer() {
        let tsv = FastTimeSeriesSpectralVectorizer::new(4, true).unwrap();
        assert_eq!(tsv.output_dim(), 16); // 5 moments + 4 bands + 2 spectral + 4 autocorr + 1 zcr = 16

        let series = (0..64)
            .map(|i| (2.0 * std::f64::consts::PI * (i as f64) / 16.0).sin())
            .collect::<Vec<f64>>();
        let feats = tsv.transform_series(&series);
        assert_eq!(feats.len(), 16);

        let t = tsv.transform(&[series]).unwrap();
        assert_eq!(t.shape(), &[1, 16]);
    }

    #[test]
    fn test_path_signature_vectorizer() {
        let psv = FastPathSignatureVectorizer::new(2).unwrap();
        assert_eq!(psv.output_dim(2), 6); // 2 (level 1) + 4 (level 2)

        // Straight line path from (0,0) to (1,2)
        let path = vec![vec![0.0, 0.0], vec![0.5, 1.0], vec![1.0, 2.0]];
        let sig = psv.signature_single(&path, 2);
        assert_eq!(sig.len(), 6);
        assert!((sig[0] - 1.0).abs() < 1e-4); // delta x
        assert!((sig[1] - 2.0).abs() < 1e-4); // delta y

        let t = psv.transform(&[path], 2).unwrap();
        assert_eq!(t.shape(), &[1, 6]);
    }

    #[test]
    fn test_deep_sets_vectorizer_permutation_invariance() {
        let dsv = FastDeepSetsVectorizer::new(3, 8, "sum", 42).unwrap();

        let item_a = vec![1.0, 2.0, 3.0];
        let item_b = vec![4.0, 5.0, 6.0];
        let item_c = vec![-1.0, 0.0, 1.0];

        let set1 = vec![item_a.clone(), item_b.clone(), item_c.clone()];
        let set2_permuted = vec![item_c, item_a, item_b]; // permuted order

        let v1 = dsv.transform_set(&set1);
        let v2 = dsv.transform_set(&set2_permuted);

        for (a, b) in v1.iter().zip(&v2) {
            assert!(
                (a - b).abs() < 1e-6,
                "Deep sets representation must be strictly permutation-invariant"
            );
        }

        let t = dsv.transform(&[set1, set2_permuted]).unwrap();
        assert_eq!(t.shape(), &[2, 8]);
    }

    #[test]
    fn test_weisfeiler_lehman_vectorizer() {
        let wlv = FastWeisfeilerLehmanVectorizer::new(2, 64, 42).unwrap();

        // 3-node path: 0 - 1 - 2
        let adj = vec![vec![0, 1, 0], vec![1, 0, 1], vec![0, 1, 0]];
        let labels = vec!["C".to_string(), "C".to_string(), "O".to_string()];

        let v = wlv.transform_graph(&adj, &labels);
        assert_eq!(v.len(), 64);

        let sum_sq: f64 = v.iter().map(|&x| x * x).sum();
        assert!((sum_sq - 1.0).abs() < 1e-4);

        let t = wlv.transform_dense(&[(&adj, &labels)]).unwrap();
        assert_eq!(t.shape(), &[1, 64]);
    }
}
