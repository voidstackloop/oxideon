//! Python bindings for the pure-Rust tokenizer stack (`oxideon_core::tokenizer`): Hugging Face `tokenizer.json`
//! pipelines, SentencePiece models, tiktoken encodings and an auto-detecting loader.

use crate::tokenizer::auto::AutoTokenizer;
use crate::tokenizer::hf::trainer::{BpeTrainer, WordLevelTrainer};
use crate::tokenizer::hf::{AddedToken, Direction, Encoding, Padding, PaddingStrategy, Tokenizer, Truncation, TruncationStrategy};
use crate::tokenizer::sentencepiece::SentencePiece;
use crate::tokenizer::tiktoken::{Special, Tiktoken};
use pyo3::exceptions::{PyKeyError, PyValueError};
use pyo3::prelude::*;
use std::sync::Arc;

fn err<E: std::fmt::Debug>(e: E) -> PyErr {
    PyValueError::new_err(format!("{e:?}"))
}

fn parse_direction(s: &str) -> PyResult<Direction> {
    match s {
        "left" | "Left" => Ok(Direction::Left),
        "right" | "Right" => Ok(Direction::Right),
        other => Err(PyValueError::new_err(format!("direction must be 'left' or 'right', not {other:?}"))),
    }
}

/// The result of encoding one sequence or a pair (like `tokenizers.Encoding`; offsets are character offsets).
#[pyclass(name = "Encoding", module = "oxideon", skip_from_py_object)]
#[derive(Clone)]
pub struct PyEncoding {
    ids: Vec<u32>,
    tokens: Vec<String>,
    offsets: Vec<(usize, usize)>,
    type_ids: Vec<u32>,
    attention_mask: Vec<u32>,
    special_tokens_mask: Vec<u32>,
    word_ids: Vec<Option<u32>>,
    sequence_ids: Vec<Option<usize>>,
    overflowing: Vec<PyEncoding>,
}

impl PyEncoding {
    fn from_core(e: &Encoding, offsets: Vec<(usize, usize)>) -> Self {
        Self {
            ids: e.ids.clone(),
            tokens: e.tokens.clone(),
            offsets,
            type_ids: e.type_ids.clone(),
            attention_mask: e.attention_mask.clone(),
            special_tokens_mask: e.special_tokens_mask.clone(),
            word_ids: e.word_ids.clone(),
            sequence_ids: e.sequence_ids.clone(),
            overflowing: e.overflowing.iter().map(|o| Self::from_core(o, o.offsets.clone())).collect(),
        }
    }
}

#[pymethods]
impl PyEncoding {
    #[getter]
    fn ids(&self) -> Vec<u32> {
        self.ids.clone()
    }
    #[getter]
    fn tokens(&self) -> Vec<String> {
        self.tokens.clone()
    }
    #[getter]
    fn offsets(&self) -> Vec<(usize, usize)> {
        self.offsets.clone()
    }
    #[getter]
    fn type_ids(&self) -> Vec<u32> {
        self.type_ids.clone()
    }
    #[getter]
    fn attention_mask(&self) -> Vec<u32> {
        self.attention_mask.clone()
    }
    #[getter]
    fn special_tokens_mask(&self) -> Vec<u32> {
        self.special_tokens_mask.clone()
    }
    #[getter]
    fn word_ids(&self) -> Vec<Option<u32>> {
        self.word_ids.clone()
    }
    #[getter]
    fn sequence_ids(&self) -> Vec<Option<usize>> {
        self.sequence_ids.clone()
    }
    #[getter]
    fn overflowing(&self) -> Vec<PyEncoding> {
        self.overflowing.clone()
    }
    fn __len__(&self) -> usize {
        self.ids.len()
    }
    fn __repr__(&self) -> String {
        format!("Encoding(num_tokens={}, ids={:?})", self.ids.len(), self.ids)
    }
}

/// A Hugging Face style tokenizer (normalizer, pre-tokenizer, model, post-processor, decoder), loadable from `tokenizer.json`.
#[pyclass(name = "Tokenizer", module = "oxideon", skip_from_py_object)]
pub struct PyTokenizer {
    inner: Tokenizer,
}

fn wrap(t: Tokenizer) -> PyTokenizer {
    PyTokenizer { inner: t }
}

#[pymethods]
impl PyTokenizer {
    #[staticmethod]
    fn from_file(path: &str) -> PyResult<Self> {
        Tokenizer::from_file(path).map(wrap).map_err(err)
    }
    #[staticmethod]
    fn from_str(json: &str) -> PyResult<Self> {
        Tokenizer::from_str(json).map(wrap).map_err(err)
    }
    /// GPT-2 style `vocab.json` + `merges.txt` contents (byte-level BPE).
    #[staticmethod]
    #[pyo3(signature = (vocab_json, merges_txt, add_prefix_space=false, special_tokens=vec![], roberta=false))]
    fn from_vocab_and_merges(vocab_json: &str, merges_txt: &str, add_prefix_space: bool, special_tokens: Vec<String>, roberta: bool) -> PyResult<Self> {
        let sp: Vec<&str> = special_tokens.iter().map(String::as_str).collect();
        Tokenizer::from_vocab_and_merges(vocab_json, merges_txt, add_prefix_space, &sp, roberta).map(wrap).map_err(err)
    }
    /// BERT style `vocab.txt` contents.
    #[staticmethod]
    #[pyo3(signature = (vocab_txt, lowercase=true, strip_accents=None))]
    fn from_vocab_txt(vocab_txt: &str, lowercase: bool, strip_accents: Option<bool>) -> PyResult<Self> {
        Tokenizer::from_vocab_txt(vocab_txt, lowercase, strip_accents).map(wrap).map_err(err)
    }
    fn to_str(&self) -> String {
        self.inner.to_json().to_text()
    }
    fn save(&self, path: &str) -> PyResult<()> {
        self.inner.save(path).map_err(err)
    }

    #[pyo3(signature = (text, pair=None, add_special_tokens=true))]
    fn encode(&self, text: &str, pair: Option<&str>, add_special_tokens: bool) -> PyResult<PyEncoding> {
        match pair {
            None => {
                let e = self.inner.encode(text, add_special_tokens).map_err(err)?;
                let off = e.char_offsets(text);
                Ok(PyEncoding::from_core(&e, off))
            }
            Some(p) => {
                let e = self.inner.encode_pair(text, p, add_special_tokens).map_err(err)?;
                // offsets of a pair index the text of the sequence each token came from
                let (oa, ob) = (Encoding { offsets: e.offsets.clone(), ..Encoding::default() }.char_offsets(text), Encoding { offsets: e.offsets.clone(), ..Encoding::default() }.char_offsets(p));
                let off = e.offsets.iter().enumerate().map(|(i, _)| if e.sequence_ids[i] == Some(1) { ob[i] } else { oa[i] }).collect();
                Ok(PyEncoding::from_core(&e, off))
            }
        }
    }

    #[pyo3(signature = (texts, add_special_tokens=true))]
    fn encode_batch(&self, py: Python<'_>, texts: Vec<String>, add_special_tokens: bool) -> PyResult<Vec<PyEncoding>> {
        let encs = py.detach(|| {
            let refs: Vec<&str> = texts.iter().map(String::as_str).collect();
            self.inner.encode_batch(&refs, add_special_tokens)
        });
        let encs = encs.map_err(err)?;
        Ok(encs.iter().zip(&texts).map(|(e, t)| PyEncoding::from_core(e, e.char_offsets(t))).collect())
    }

    #[pyo3(signature = (ids, skip_special_tokens=true))]
    fn decode(&self, ids: Vec<u32>, skip_special_tokens: bool) -> PyResult<String> {
        self.inner.decode(&ids, skip_special_tokens).map_err(err)
    }
    #[pyo3(signature = (batch, skip_special_tokens=true))]
    fn decode_batch(&self, batch: Vec<Vec<u32>>, skip_special_tokens: bool) -> PyResult<Vec<String>> {
        self.inner.decode_batch(&batch, skip_special_tokens).map_err(err)
    }

    fn token_to_id(&self, token: &str) -> Option<u32> {
        self.inner.token_to_id(token)
    }
    fn id_to_token(&self, id: u32) -> Option<String> {
        self.inner.id_to_token(id)
    }
    #[pyo3(signature = (with_added_tokens=true))]
    fn get_vocab_size(&self, with_added_tokens: bool) -> usize {
        self.inner.get_vocab_size(with_added_tokens)
    }
    fn add_tokens(&mut self, tokens: Vec<String>) -> usize {
        let v: Vec<AddedToken> = tokens.iter().map(|t| AddedToken::new(t, false)).collect();
        self.inner.add_tokens(&v)
    }
    fn add_special_tokens(&mut self, tokens: Vec<String>) -> usize {
        let refs: Vec<&str> = tokens.iter().map(String::as_str).collect();
        self.inner.add_special_tokens(&refs)
    }

    #[pyo3(signature = (max_length, stride=0, strategy="longest_first", direction="right"))]
    fn enable_truncation(&mut self, max_length: usize, stride: usize, strategy: &str, direction: &str) -> PyResult<()> {
        let strategy = match strategy {
            "longest_first" => TruncationStrategy::LongestFirst,
            "only_first" => TruncationStrategy::OnlyFirst,
            "only_second" => TruncationStrategy::OnlySecond,
            other => return Err(PyValueError::new_err(format!("unknown truncation strategy {other:?}"))),
        };
        self.inner.with_truncation(Some(Truncation { max_length, strategy, stride, direction: parse_direction(direction)? }));
        Ok(())
    }
    fn no_truncation(&mut self) {
        self.inner.with_truncation(None);
    }
    #[pyo3(signature = (length=None, direction="right", pad_to_multiple_of=None, pad_id=0, pad_type_id=0, pad_token="[PAD]"))]
    fn enable_padding(&mut self, length: Option<usize>, direction: &str, pad_to_multiple_of: Option<usize>, pad_id: u32, pad_type_id: u32, pad_token: &str) -> PyResult<()> {
        self.inner.with_padding(Some(Padding {
            strategy: length.map_or(PaddingStrategy::BatchLongest, PaddingStrategy::Fixed),
            direction: parse_direction(direction)?,
            pad_to_multiple_of,
            pad_id,
            pad_type_id,
            pad_token: pad_token.to_string(),
        }));
        Ok(())
    }
    fn no_padding(&mut self) {
        self.inner.with_padding(None);
    }

    /// Normalized and pre-tokenized words of `text` (what the trainers count).
    fn words(&self, text: &str) -> Vec<String> {
        self.inner.words(text)
    }

    /// Replaces the model by a BPE trained on `texts` (the normalizer / pre-tokenizer of this tokenizer define the words).
    #[pyo3(signature = (texts, vocab_size=30000, min_frequency=0, special_tokens=vec![], limit_alphabet=None, initial_alphabet=vec![], continuing_subword_prefix=None, end_of_word_suffix=None))]
    #[allow(clippy::too_many_arguments)]
    fn train_bpe(&mut self, py: Python<'_>, texts: Vec<String>, vocab_size: usize, min_frequency: u64, special_tokens: Vec<String>, limit_alphabet: Option<usize>, initial_alphabet: Vec<String>, continuing_subword_prefix: Option<String>, end_of_word_suffix: Option<String>) -> PyResult<()> {
        let trainer = BpeTrainer {
            vocab_size,
            min_frequency,
            special_tokens,
            limit_alphabet,
            initial_alphabet: initial_alphabet.iter().filter_map(|s| s.chars().next()).collect(),
            continuing_subword_prefix,
            end_of_word_suffix,
        };
        py.detach(|| self.inner.train_bpe(&texts, &trainer)).map_err(err)
    }
    /// Replaces the model by a WordLevel vocabulary of the most frequent words of `texts`.
    #[pyo3(signature = (texts, vocab_size=30000, min_frequency=0, special_tokens=vec![]))]
    fn train_word_level(&mut self, py: Python<'_>, texts: Vec<String>, vocab_size: usize, min_frequency: u64, special_tokens: Vec<String>) -> PyResult<()> {
        let trainer = WordLevelTrainer { vocab_size, min_frequency, special_tokens };
        py.detach(|| self.inner.train_word_level(&texts, &trainer)).map_err(err)
    }
    /// An untrained tokenizer ("BPE", "WordPiece" or "WordLevel"); set a pre-tokenizer JSON with `set_pre_tokenizer`.
    #[staticmethod]
    #[pyo3(signature = (kind, unk_token=None))]
    fn untrained(kind: &str, unk_token: Option<&str>) -> PyResult<Self> {
        Tokenizer::untrained(kind, unk_token).map(wrap).map_err(err)
    }
    fn set_normalizer(&mut self, json: &str) -> PyResult<()> {
        self.inner.set_normalizer_json(&crate::tokenizer::json::Json::parse(json).map_err(err)?).map_err(err)
    }
    fn set_pre_tokenizer(&mut self, json: &str) -> PyResult<()> {
        self.inner.set_pre_tokenizer_json(&crate::tokenizer::json::Json::parse(json).map_err(err)?).map_err(err)
    }
    fn set_post_processor(&mut self, json: &str) -> PyResult<()> {
        self.inner.set_post_processor_json(&crate::tokenizer::json::Json::parse(json).map_err(err)?).map_err(err)
    }
    fn set_decoder(&mut self, json: &str) -> PyResult<()> {
        self.inner.set_decoder_json(&crate::tokenizer::json::Json::parse(json).map_err(err)?).map_err(err)
    }
}

/// A SentencePiece model (`.model` file): Unigram or BPE.
#[pyclass(name = "SentencePieceProcessor", module = "oxideon", skip_from_py_object)]
pub struct PySentencePiece {
    inner: SentencePiece,
}

#[pymethods]
impl PySentencePiece {
    #[new]
    fn new(model_file: &str) -> PyResult<Self> {
        SentencePiece::from_file(model_file).map(|inner| Self { inner }).map_err(err)
    }
    #[staticmethod]
    fn from_bytes(data: &[u8]) -> PyResult<Self> {
        SentencePiece::from_bytes(data).map(|inner| Self { inner }).map_err(err)
    }
    #[pyo3(signature = (text, add_bos=false, add_eos=false))]
    fn encode(&self, text: &str, add_bos: bool, add_eos: bool) -> PyResult<Vec<u32>> {
        self.inner.encode_with_special(text, add_bos, add_eos).map_err(err)
    }
    fn encode_as_pieces(&self, text: &str) -> PyResult<Vec<String>> {
        Ok(self.inner.encode_pieces_ids(text).map_err(err)?.into_iter().map(|(p, _)| p).collect())
    }
    fn decode(&self, ids: Vec<u32>) -> PyResult<String> {
        self.inner.decode(&ids).map_err(err)
    }
    fn normalize(&self, text: &str) -> String {
        self.inner.normalize(text)
    }
    fn get_piece_size(&self) -> usize {
        self.inner.vocab_size()
    }
    fn id_to_piece(&self, id: u32) -> PyResult<String> {
        self.inner.id_to_piece(id).map(str::to_string).ok_or_else(|| PyKeyError::new_err(id))
    }
    fn piece_to_id(&self, piece: &str) -> Option<u32> {
        self.inner.piece_to_id(piece)
    }
    #[getter]
    fn bos_id(&self) -> i32 {
        self.inner.bos_id
    }
    #[getter]
    fn eos_id(&self) -> i32 {
        self.inner.eos_id
    }
    #[getter]
    fn unk_id(&self) -> i32 {
        self.inner.unk_id
    }
    #[getter]
    fn pad_id(&self) -> i32 {
        self.inner.pad_id
    }
}

/// A tiktoken-style byte-level BPE encoding.
#[pyclass(name = "TiktokenEncoding", module = "oxideon", skip_from_py_object)]
pub struct PyTiktoken {
    inner: Arc<Tiktoken>,
}

#[pymethods]
impl PyTiktoken {
    /// `path` is a `.tiktoken` rank file; `pattern` the splitting regex (see `oxideon.tiktoken_pattern`).
    #[new]
    #[pyo3(signature = (path, pattern, special_tokens=vec![]))]
    fn new(path: &str, pattern: &str, special_tokens: Vec<(String, u32)>) -> PyResult<Self> {
        let sp: Vec<(&str, u32)> = special_tokens.iter().map(|(s, i)| (s.as_str(), *i)).collect();
        Tiktoken::from_tiktoken_file(path, pattern, &sp).map(|t| Self { inner: Arc::new(t) }).map_err(err)
    }
    /// One of the OpenAI encodings ("r50k_base", "p50k_base", "cl100k_base", "o200k_base") from its rank file.
    #[staticmethod]
    fn preset(name: &str, path: &str) -> PyResult<Self> {
        let text = std::fs::read_to_string(path).map_err(err)?;
        Tiktoken::preset(name, &text).map(|t| Self { inner: Arc::new(t) }).map_err(err)
    }
    fn encode_ordinary(&self, text: &str) -> PyResult<Vec<u32>> {
        self.inner.encode_ordinary(text).map_err(err)
    }
    /// `allowed_special`: "all", or a list of special-token strings; with `disallowed_special="all"` any other special-token
    /// string in the text is an error, like the reference.
    #[pyo3(signature = (text, allowed_special=None, disallowed_special=true))]
    fn encode(&self, text: &str, allowed_special: Option<Bound<'_, PyAny>>, disallowed_special: bool) -> PyResult<Vec<u32>> {
        let mode = match &allowed_special {
            Some(a) if a.extract::<String>().is_ok_and(|s| s == "all") => Special::All,
            Some(a) => {
                let owned: Vec<String> = a.extract()?;
                return self.encode_only(text, &owned);
            }
            None if disallowed_special => Special::Disallow,
            None => Special::Ignore,
        };
        self.inner.encode(text, mode).map_err(err)
    }
    fn decode(&self, ids: Vec<u32>) -> PyResult<String> {
        self.inner.decode(&ids).map_err(err)
    }
    fn decode_bytes(&self, ids: Vec<u32>) -> PyResult<Vec<u8>> {
        self.inner.decode_bytes(&ids).map_err(err)
    }
    #[getter]
    fn n_vocab(&self) -> usize {
        self.inner.n_vocab()
    }
}

impl PyTiktoken {
    fn encode_only(&self, text: &str, allowed: &[String]) -> PyResult<Vec<u32>> {
        let refs: Vec<&str> = allowed.iter().map(String::as_str).collect();
        self.inner.encode(text, Special::Only(&refs)).map_err(err)
    }
}

/// Loads whatever tokenizer files a model directory holds (`tokenizer.json`, `tokenizer.model`, `vocab.json` +
/// `merges.txt`, `vocab.txt`) and reads `tokenizer_config.json`.
#[pyclass(name = "AutoTokenizer", module = "oxideon", skip_from_py_object)]
pub struct PyAutoTokenizer {
    inner: AutoTokenizer,
}

#[pymethods]
impl PyAutoTokenizer {
    #[staticmethod]
    fn from_pretrained(path: &str) -> PyResult<Self> {
        let p = std::path::Path::new(path);
        let inner = if p.is_dir() { AutoTokenizer::from_dir(p) } else { AutoTokenizer::from_file(p) };
        inner.map(|inner| Self { inner }).map_err(err)
    }
    #[pyo3(signature = (text, add_special_tokens=true))]
    fn encode(&self, text: &str, add_special_tokens: bool) -> PyResult<Vec<u32>> {
        self.inner.encode(text, add_special_tokens).map_err(err)
    }
    #[pyo3(signature = (texts, add_special_tokens=true))]
    fn encode_batch(&self, texts: Vec<String>, add_special_tokens: bool) -> PyResult<Vec<Vec<u32>>> {
        let refs: Vec<&str> = texts.iter().map(String::as_str).collect();
        self.inner.encode_batch(&refs, add_special_tokens).map_err(err)
    }
    #[pyo3(signature = (ids, skip_special_tokens=true))]
    fn decode(&self, ids: Vec<u32>, skip_special_tokens: bool) -> PyResult<String> {
        self.inner.decode(&ids, skip_special_tokens).map_err(err)
    }
    fn token_to_id(&self, token: &str) -> Option<u32> {
        self.inner.token_to_id(token)
    }
    fn id_to_token(&self, id: u32) -> Option<String> {
        self.inner.id_to_token(id)
    }
    #[getter]
    fn vocab_size(&self) -> usize {
        self.inner.vocab_size()
    }
    /// Renders the chat template over `messages` (a list of dicts). With `tokenize=True` the rendered text is encoded
    /// (without adding further special tokens) and the ids are returned instead of the string.
    #[pyo3(signature = (messages, tools=None, add_generation_prompt=false, tokenize=false, **kwargs))]
    fn apply_chat_template(&self, py: Python<'_>, messages: Bound<'_, PyAny>, tools: Option<Bound<'_, PyAny>>, add_generation_prompt: bool, tokenize: bool, kwargs: Option<Bound<'_, pyo3::types::PyDict>>) -> PyResult<Py<PyAny>> {
        let json = py.import("json")?;
        let to_json = |o: &Bound<'_, PyAny>| -> PyResult<crate::tokenizer::json::Json> {
            let text: String = json.call_method1("dumps", (o,))?.extract()?;
            crate::tokenizer::json::Json::parse(&text).map_err(err)
        };
        let msgs = to_json(&messages)?;
        let tools_j = tools.as_ref().map(|t| to_json(t)).transpose()?;
        let mut extra: Vec<(String, crate::tokenizer::json::Json)> = Vec::new();
        if let Some(k) = kwargs {
            for (key, val) in k.iter() {
                extra.push((key.extract::<String>()?, to_json(&val)?));
            }
        }
        let extra_refs: Vec<(&str, crate::tokenizer::json::Json)> = extra.iter().map(|(k, v)| (k.as_str(), v.clone())).collect();
        let text = self.inner.apply_chat_template(&msgs, tools_j.as_ref(), add_generation_prompt, &extra_refs).map_err(err)?;
        if tokenize {
            let ids = self.inner.encode(&text, false).map_err(err)?;
            Ok(ids.into_pyobject(py)?.into_any().unbind())
        } else {
            Ok(text.into_pyobject(py)?.into_any().unbind())
        }
    }
    #[getter]
    fn bos_token(&self) -> Option<String> {
        self.inner.config.bos_token.clone()
    }
    #[getter]
    fn eos_token(&self) -> Option<String> {
        self.inner.config.eos_token.clone()
    }
    #[getter]
    fn unk_token(&self) -> Option<String> {
        self.inner.config.unk_token.clone()
    }
    #[getter]
    fn pad_token(&self) -> Option<String> {
        self.inner.config.pad_token.clone()
    }
    #[getter]
    fn bos_token_id(&self) -> Option<u32> {
        self.inner.bos_token_id()
    }
    #[getter]
    fn eos_token_id(&self) -> Option<u32> {
        self.inner.eos_token_id()
    }
    #[getter]
    fn pad_token_id(&self) -> Option<u32> {
        self.inner.pad_token_id()
    }
    #[getter]
    fn unk_token_id(&self) -> Option<u32> {
        self.inner.unk_token_id()
    }
    #[getter]
    fn model_max_length(&self) -> Option<usize> {
        self.inner.config.model_max_length
    }
    #[getter]
    fn chat_template(&self) -> Option<String> {
        self.inner.config.chat_template.clone()
    }
}

/// Registers the tokenizer classes on the Python module.
pub fn register_tokenizer_classes(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyEncoding>()?;
    m.add_class::<PyTokenizer>()?;
    m.add_class::<PySentencePiece>()?;
    m.add_class::<PyTiktoken>()?;
    m.add_class::<PyAutoTokenizer>()?;
    m.add("TIKTOKEN_R50K_PATTERN", crate::tokenizer::tiktoken::R50K_PATTERN)?;
    m.add("TIKTOKEN_CL100K_PATTERN", crate::tokenizer::tiktoken::CL100K_PATTERN)?;
    m.add("TIKTOKEN_O200K_PATTERN", crate::tokenizer::tiktoken::O200K_PATTERN)?;
    Ok(())
}
