//! Tokenizer training: BPE (also the WordPiece vocabulary builder, with the `##` prefix) and WordLevel, following the
//! reference trainers (alphabet first, merges by highest pair count, ties to the smaller pair of ids). Words are visited
//! in sorted order, so results are reproducible run to run.

use super::model::{Bpe, Model, Vocab, WordLevel, WordPiece};
use super::{AddedToken, Tokenizer};
use crate::error::{Error, Result};
use crate::tokenizer::json::Json;
use std::cmp::Ordering;
use std::collections::{BTreeMap, BinaryHeap, HashMap, HashSet};

#[derive(Debug, Clone)]
pub struct BpeTrainer {
    pub vocab_size: usize,
    pub min_frequency: u64,
    pub special_tokens: Vec<String>,
    pub limit_alphabet: Option<usize>,
    pub initial_alphabet: Vec<char>,
    pub continuing_subword_prefix: Option<String>,
    pub end_of_word_suffix: Option<String>,
}

impl Default for BpeTrainer {
    fn default() -> Self {
        Self { vocab_size: 30000, min_frequency: 0, special_tokens: vec![], limit_alphabet: None, initial_alphabet: vec![], continuing_subword_prefix: None, end_of_word_suffix: None }
    }
}

impl BpeTrainer {
    /// The WordPiece flavour: `##` continuing prefix.
    pub fn word_piece(vocab_size: usize) -> Self {
        Self { vocab_size, continuing_subword_prefix: Some("##".into()), ..Self::default() }
    }
}

#[derive(Debug, Clone, Default)]
pub struct WordLevelTrainer {
    pub vocab_size: usize,
    pub min_frequency: u64,
    pub special_tokens: Vec<String>,
}

type Pair = (u32, u32);

#[derive(Eq, PartialEq)]
struct Merge {
    pair: Pair,
    count: u64,
    pos: HashSet<usize>,
}

impl Ord for Merge {
    fn cmp(&self, other: &Self) -> Ordering {
        // higher count first; on equal counts the smaller pair first
        self.count.cmp(&other.count).then_with(|| other.pair.cmp(&self.pair))
    }
}
impl PartialOrd for Merge {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

/// `(vocabulary in id order, merges as token strings)`.
pub struct TrainedBpe {
    pub vocab: Vec<String>,
    pub merges: Vec<(String, String)>,
}

impl BpeTrainer {
    pub fn train(&self, word_counts: &BTreeMap<String, u64>) -> TrainedBpe {
        let mut word_to_id: HashMap<String, u32> = HashMap::new();
        let mut id_to_word: Vec<String> = Vec::new();
        let add = |w2i: &mut HashMap<String, u32>, i2w: &mut Vec<String>, s: String| -> u32 {
            if let Some(&i) = w2i.get(&s) {
                return i;
            }
            i2w.push(s.clone());
            w2i.insert(s, (i2w.len() - 1) as u32);
            (i2w.len() - 1) as u32
        };
        for t in &self.special_tokens {
            add(&mut word_to_id, &mut id_to_word, t.clone());
        }
        // alphabet
        let mut alphabet: HashMap<char, usize> = HashMap::new();
        for (w, &c) in word_counts {
            for ch in w.chars() {
                *alphabet.entry(ch).or_insert(0) += c as usize;
            }
        }
        for &c in &self.initial_alphabet {
            alphabet.insert(c, usize::MAX);
        }
        let mut kept: Vec<(char, usize)> = alphabet.iter().map(|(&c, &n)| (c, n)).collect();
        let to_remove = self.limit_alphabet.map_or(0, |l| alphabet.len().saturating_sub(l));
        if to_remove > 0 {
            kept.sort_by_key(|&(c, n)| (n, c));
            kept.drain(..to_remove);
        }
        kept.sort_by_key(|&(c, _)| c as u32);
        for (c, _) in kept {
            add(&mut word_to_id, &mut id_to_word, c.to_string());
        }
        // words as sequences of symbol ids
        let mut words: Vec<Vec<u32>> = Vec::new();
        let mut counts: Vec<u64> = Vec::new();
        for (w, &c) in word_counts {
            let chars: Vec<char> = w.chars().collect();
            let mut cur = Vec::with_capacity(chars.len());
            for (i, &ch) in chars.iter().enumerate() {
                let mut s = ch.to_string();
                if !word_to_id.contains_key(&s) {
                    continue; // not in the (limited) alphabet
                }
                if i > 0 {
                    if let Some(p) = &self.continuing_subword_prefix {
                        s = format!("{p}{s}");
                    }
                }
                if i == chars.len() - 1 {
                    if let Some(suf) = &self.end_of_word_suffix {
                        s = format!("{s}{suf}");
                    }
                }
                cur.push(add(&mut word_to_id, &mut id_to_word, s));
            }
            words.push(cur);
            counts.push(c);
        }
        // initial pair counts
        let mut pair_counts: HashMap<Pair, i64> = HashMap::new();
        let mut where_to_update: HashMap<Pair, HashSet<usize>> = HashMap::new();
        for (i, w) in words.iter().enumerate() {
            for win in w.windows(2) {
                let p = (win[0], win[1]);
                *pair_counts.entry(p).or_insert(0) += counts[i] as i64;
                where_to_update.entry(p).or_default().insert(i);
            }
        }
        let mut queue: BinaryHeap<Merge> = BinaryHeap::with_capacity(pair_counts.len());
        for (pair, pos) in where_to_update.drain() {
            let count = pair_counts[&pair];
            if count > 0 {
                queue.push(Merge { pair, count: count as u64, pos });
            }
        }
        let prefix_len = self.continuing_subword_prefix.as_ref().map_or(0, |p| p.len());
        let mut merges: Vec<(Pair, u32)> = Vec::new();
        while word_to_id.len() < self.vocab_size {
            let Some(mut top) = queue.pop() else { break };
            let cur = pair_counts.get(&top.pair).copied().unwrap_or(0) as u64;
            if top.count != cur {
                top.count = cur;
                queue.push(top);
                continue;
            }
            if top.count < 1 || top.count < self.min_frequency {
                break;
            }
            let part_a = id_to_word[top.pair.0 as usize].clone();
            let mut part_b = id_to_word[top.pair.1 as usize].clone();
            if self.continuing_subword_prefix.is_some() && prefix_len > 0 && part_b.starts_with(self.continuing_subword_prefix.as_deref().unwrap()) {
                part_b = part_b[prefix_len..].to_string();
            }
            let new_token = format!("{part_a}{part_b}");
            let new_id = add(&mut word_to_id, &mut id_to_word, new_token);
            merges.push((top.pair, new_id));
            // apply the merge in every word that has the pair
            let mut positions: Vec<usize> = top.pos.iter().copied().collect();
            positions.sort_unstable();
            let mut changes: Vec<(Pair, i64, usize)> = Vec::new();
            for &iw in &positions {
                let w = &mut words[iw];
                let mut i = 0;
                while i < w.len() {
                    if w[i] == top.pair.0 && i + 1 < w.len() && w[i + 1] == top.pair.1 {
                        let (first, second) = (w[i], w[i + 1]);
                        if i > 0 {
                            changes.push(((w[i - 1], first), -1, iw));
                            changes.push(((w[i - 1], new_id), 1, iw));
                        }
                        w[i] = new_id;
                        w.remove(i + 1);
                        if i + 1 < w.len() {
                            changes.push(((second, w[i + 1]), -1, iw));
                            changes.push(((new_id, w[i + 1]), 1, iw));
                        }
                    }
                    i += 1;
                }
            }
            for (pair, change, iw) in changes {
                let c = change * counts[iw] as i64;
                *pair_counts.entry(pair).or_insert(0) += c;
                if change > 0 {
                    where_to_update.entry(pair).or_default().insert(iw);
                }
            }
            for (pair, pos) in where_to_update.drain() {
                let count = pair_counts[&pair];
                if count > 0 {
                    queue.push(Merge { pair, count: count as u64, pos });
                }
            }
        }
        TrainedBpe { vocab: id_to_word.clone(), merges: merges.into_iter().map(|((a, b), _)| (id_to_word[a as usize].clone(), id_to_word[b as usize].clone())).collect() }
    }
}

impl WordLevelTrainer {
    pub fn train(&self, word_counts: &BTreeMap<String, u64>) -> Vec<String> {
        let mut ordered: Vec<(&String, &u64)> = word_counts.iter().collect();
        ordered.sort_by(|a, b| b.1.cmp(a.1).then_with(|| a.0.cmp(b.0)));
        let mut out: Vec<String> = self.special_tokens.clone();
        out.extend(ordered.into_iter().filter(|(_, n)| **n >= self.min_frequency).map(|(w, _)| w.clone()));
        let mut seen = HashSet::new();
        out.retain(|w| seen.insert(w.clone()));
        if self.vocab_size > 0 {
            out.truncate(self.vocab_size);
        }
        out
    }
}


impl Model {
    /// The `model` section of a `tokenizer.json`.
    pub fn to_json(&self) -> Json {
        let s = |x: &str| Json::Str(x.to_string());
        let opt = |x: &Option<String>| x.as_ref().map_or(Json::Null, |v| s(v));
        let ordered = |v: &Vocab| {
            let mut pairs: Vec<(&String, &u32)> = v.map.iter().collect();
            pairs.sort_by_key(|&(_, &id)| id);
            Json::Obj(pairs.into_iter().map(|(t, &id)| (t.clone(), Json::Num(id as f64))).collect())
        };
        match self {
            Model::Bpe(m) => Json::Obj(vec![
                ("type".into(), s("BPE")),
                ("dropout".into(), Json::Null),
                ("unk_token".into(), opt(&m.unk_token)),
                ("continuing_subword_prefix".into(), opt(&m.continuing_subword_prefix)),
                ("end_of_word_suffix".into(), opt(&m.end_of_word_suffix)),
                ("fuse_unk".into(), Json::Bool(m.fuse_unk)),
                ("byte_fallback".into(), Json::Bool(m.byte_fallback)),
                ("ignore_merges".into(), Json::Bool(m.ignore_merges)),
                ("vocab".into(), ordered(&m.vocab)),
                ("merges".into(), Json::Arr(m.merge_list.iter().map(|(a, b)| Json::Arr(vec![s(a), s(b)])).collect())),
            ]),
            Model::WordPiece(m) => Json::Obj(vec![
                ("type".into(), s("WordPiece")),
                ("unk_token".into(), s(&m.unk_token)),
                ("continuing_subword_prefix".into(), s(&m.continuing_subword_prefix)),
                ("max_input_chars_per_word".into(), Json::Num(m.max_input_chars_per_word as f64)),
                ("vocab".into(), ordered(&m.vocab)),
            ]),
            Model::WordLevel(m) => Json::Obj(vec![("type".into(), s("WordLevel")), ("vocab".into(), ordered(&m.vocab)), ("unk_token".into(), s(&m.unk_token))]),
            Model::Unigram(m) => Json::Obj(vec![
                ("type".into(), s("Unigram")),
                ("unk_id".into(), m.unk_id.map_or(Json::Null, |u| Json::Num(u as f64))),
                ("vocab".into(), Json::Arr(m.pieces.iter().map(|(p, sc)| Json::Arr(vec![s(p), Json::Num(*sc)])).collect())),
                ("byte_fallback".into(), Json::Bool(m.byte_fallback)),
            ]),
        }
    }
}

impl Tokenizer {
    /// Normalized, pre-tokenized words of `text` (what the trainers count).
    pub fn words(&self, text: &str) -> Vec<String> {
        let mut ns = super::normalizer::NormalizedString::new(text);
        if let Some(n) = &self.normalizer {
            n.normalize(&mut ns);
        }
        let pieces = match &self.pre_tokenizer {
            Some(p) => p.pre_tokenize(vec![ns]),
            None => if ns.is_empty() { vec![] } else { vec![ns] },
        };
        pieces.into_iter().map(|p| p.get()).collect()
    }

    fn count_words<S: AsRef<str>>(&self, texts: impl IntoIterator<Item = S>) -> BTreeMap<String, u64> {
        let mut counts: BTreeMap<String, u64> = BTreeMap::new();
        for t in texts {
            for w in self.words(t.as_ref()) {
                *counts.entry(w).or_insert(0) += 1;
            }
        }
        counts
    }

    fn install_specials(&mut self, specials: &[String]) {
        let toks: Vec<AddedToken> = specials.iter().map(|t| AddedToken::new(t, true)).collect();
        self.add_tokens(&toks);
    }

    fn set_model(&mut self, model: Model) {
        let j = model.to_json();
        self.model = model;
        self.set_source_model(j);
    }

    /// Trains a BPE model (or, with a `##` prefix, a WordPiece vocabulary) from `texts`, replacing the model.
    /// The normalizer and pre-tokenizer already set on the tokenizer define the words; `unk_token` is kept from
    /// the current BPE model when there is one.
    pub fn train_bpe<S: AsRef<str>>(&mut self, texts: impl IntoIterator<Item = S>, trainer: &BpeTrainer) -> Result<()> {
        let counts = self.count_words(texts);
        let trained = trainer.train(&counts);
        let vocab = Vocab::from_pairs(trained.vocab.iter().enumerate().map(|(i, t)| (t.clone(), i as u32)));
        let (unk, fuse, byte_fb) = match &self.model {
            Model::Bpe(m) => (m.unk_token.clone(), m.fuse_unk, m.byte_fallback),
            _ => (None, false, false),
        };
        let model = if trainer.continuing_subword_prefix.as_deref() == Some("##") && trainer.end_of_word_suffix.is_none() && matches!(self.model, Model::WordPiece(_)) {
            let unk = match &self.model {
                Model::WordPiece(m) => m.unk_token.clone(),
                _ => "[UNK]".into(),
            };
            Model::WordPiece(WordPiece { vocab, unk_token: unk, continuing_subword_prefix: "##".into(), max_input_chars_per_word: 100 })
        } else {
            Model::Bpe(Bpe::new(vocab, trained.merges, unk, trainer.continuing_subword_prefix.clone(), trainer.end_of_word_suffix.clone(), fuse, byte_fb, false)?)
        };
        self.set_model(model);
        self.install_specials(&trainer.special_tokens);
        Ok(())
    }

    /// Trains a WordLevel vocabulary (the most frequent words) from `texts`.
    pub fn train_word_level<S: AsRef<str>>(&mut self, texts: impl IntoIterator<Item = S>, trainer: &WordLevelTrainer) -> Result<()> {
        let counts = self.count_words(texts);
        let vocab = trainer.train(&counts);
        let unk = match &self.model {
            Model::WordLevel(m) => m.unk_token.clone(),
            _ => "[UNK]".into(),
        };
        if vocab.is_empty() {
            return Err(Error::Msg("tokenizer: training produced an empty vocabulary".into()));
        }
        self.set_model(Model::WordLevel(WordLevel { vocab: Vocab::from_pairs(vocab.iter().enumerate().map(|(i, t)| (t.clone(), i as u32))), unk_token: unk }));
        self.install_specials(&trainer.special_tokens);
        Ok(())
    }
}

impl Tokenizer {
    /// An untrained tokenizer around an empty model of the given kind (`"BPE"`, `"WordPiece"`, `"WordLevel"`) with no
    /// normalizer or pre-tokenizer; set those fields, then train.
    pub fn untrained(kind: &str, unk_token: Option<&str>) -> Result<Self> {
        let s = |x: &str| Json::Str(x.to_string());
        let model = match kind {
            "BPE" => Json::Obj(vec![("type".into(), s("BPE")), ("unk_token".into(), unk_token.map_or(Json::Null, s)), ("vocab".into(), Json::Obj(vec![])), ("merges".into(), Json::Arr(vec![]))]),
            "WordPiece" => Json::Obj(vec![("type".into(), s("WordPiece")), ("unk_token".into(), s(unk_token.unwrap_or("[UNK]"))), ("vocab".into(), Json::Obj(vec![]))]),
            "WordLevel" => Json::Obj(vec![("type".into(), s("WordLevel")), ("unk_token".into(), s(unk_token.unwrap_or("[UNK]"))), ("vocab".into(), Json::Obj(vec![]))]),
            "Unigram" => Json::Obj(vec![("type".into(), s("Unigram")), ("unk_id".into(), Json::Null), ("vocab".into(), Json::Arr(vec![]))]),
            other => return Err(Error::Msg(format!("tokenizer: cannot train a {other} model"))),
        };
        Self::from_json(&Json::Obj(vec![("version".into(), s("1.0")), ("added_tokens".into(), Json::Arr(vec![])), ("model".into(), model)]))
    }
}

/// Unigram (SentencePiece style) trainer: frequent-substring seed vocabulary, EM re-estimation of piece probabilities and
/// loss-based pruning down to `vocab_size`. Deterministic; not bit-identical to the reference (whose own runs differ).
#[derive(Debug, Clone)]
pub struct UnigramTrainer {
    pub vocab_size: usize,
    pub special_tokens: Vec<String>,
    pub unk_token: Option<String>,
    pub shrinking_factor: f64,
    pub n_sub_iterations: usize,
    pub max_piece_length: usize,
    pub seed_size: usize,
}

impl Default for UnigramTrainer {
    fn default() -> Self {
        Self { vocab_size: 8000, special_tokens: vec![], unk_token: None, shrinking_factor: 0.75, n_sub_iterations: 2, max_piece_length: 16, seed_size: 1_000_000 }
    }
}

fn digamma(mut x: f64) -> f64 {
    let mut r = 0.0;
    while x < 6.0 {
        r -= 1.0 / x;
        x += 1.0;
    }
    let f = 1.0 / (x * x);
    r + x.ln() - 0.5 / x - f * (1.0 / 12.0 - f * (1.0 / 120.0 - f * (1.0 / 252.0 - f * (1.0 / 240.0 - f / 132.0))))
}

fn logsumexp(a: f64, b: f64) -> f64 {
    if a == f64::NEG_INFINITY {
        return b;
    }
    if b == f64::NEG_INFINITY {
        return a;
    }
    let m = a.max(b);
    m + ((a - m).exp() + (b - m).exp()).ln()
}

impl UnigramTrainer {
    /// Returns `(piece, score)` for the specials (score 0) followed by the learned pieces, best first.
    pub fn train(&self, word_counts: &BTreeMap<String, u64>) -> Vec<(String, f64)> {
        let words: Vec<(Vec<char>, f64)> = word_counts.iter().map(|(w, &c)| (w.chars().collect(), c as f64)).collect();
        // seed: every character plus substrings seen at least twice, ranked by count * length
        let mut sub: HashMap<String, f64> = HashMap::new();
        for (w, c) in &words {
            for b in 0..w.len() {
                for e in b + 1..=w.len().min(b + self.max_piece_length) {
                    *sub.entry(w[b..e].iter().collect()).or_insert(0.0) += c;
                }
            }
        }
        let mut chars: Vec<(String, f64)> = sub.iter().filter(|(k, _)| k.chars().count() == 1).map(|(k, &v)| (k.clone(), v)).collect();
        chars.sort_by(|a, b| a.0.cmp(&b.0));
        let mut multi: Vec<(String, f64)> = sub.into_iter().filter(|(k, v)| k.chars().count() > 1 && *v >= 2.0).collect();
        multi.sort_by(|a, b| (b.1 * b.0.chars().count() as f64).partial_cmp(&(a.1 * a.0.chars().count() as f64)).unwrap_or(Ordering::Equal).then_with(|| a.0.cmp(&b.0)));
        multi.truncate(self.seed_size.saturating_sub(chars.len()));
        let mut pieces: Vec<(String, f64)> = chars.iter().chain(multi.iter()).cloned().collect();
        let total: f64 = pieces.iter().map(|p| p.1).sum();
        for p in &mut pieces {
            p.1 = (p.1 / total).ln();
        }
        let desired = self.vocab_size.saturating_sub(self.special_tokens.len()).max(chars.len());
        loop {
            for _ in 0..self.n_sub_iterations {
                pieces = Self::em_step(&pieces, &words, self.max_piece_length);
            }
            if pieces.len() <= desired {
                break;
            }
            let target = ((pieces.len() as f64 * self.shrinking_factor) as usize).max(desired);
            pieces = Self::prune(&pieces, &words, target, self.max_piece_length);
            if pieces.len() <= desired {
                pieces = Self::em_step(&pieces, &words, self.max_piece_length);
                break;
            }
        }
        pieces.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal).then_with(|| a.0.cmp(&b.0)));
        let mut out: Vec<(String, f64)> = self.special_tokens.iter().map(|s| (s.clone(), 0.0)).collect();
        out.extend(pieces.into_iter().filter(|p| !self.special_tokens.contains(&p.0)));
        out.truncate(self.vocab_size.max(self.special_tokens.len()));
        out
    }

    fn em_step(pieces: &[(String, f64)], words: &[(Vec<char>, f64)], maxlen: usize) -> Vec<(String, f64)> {
        let idx: HashMap<&str, usize> = pieces.iter().enumerate().map(|(i, p)| (p.0.as_str(), i)).collect();
        let mut expected = vec![0.0f64; pieces.len()];
        for (w, c) in words {
            let n = w.len();
            let mut edges: Vec<(usize, usize, usize)> = Vec::new();
            for b in 0..n {
                let mut s = String::new();
                for e in b + 1..=n.min(b + maxlen) {
                    s.push(w[e - 1]);
                    if let Some(&i) = idx.get(s.as_str()) {
                        edges.push((b, e, i));
                    }
                }
            }
            let mut alpha = vec![f64::NEG_INFINITY; n + 1];
            alpha[0] = 0.0;
            for &(b, e, i) in &edges {
                if alpha[b] > f64::NEG_INFINITY {
                    alpha[e] = logsumexp(alpha[e], alpha[b] + pieces[i].1);
                }
            }
            if alpha[n] == f64::NEG_INFINITY {
                continue;
            }
            let mut beta = vec![f64::NEG_INFINITY; n + 1];
            beta[n] = 0.0;
            for &(b, e, i) in edges.iter().rev() {
                if beta[e] > f64::NEG_INFINITY {
                    beta[b] = logsumexp(beta[b], beta[e] + pieces[i].1);
                }
            }
            for &(b, e, i) in &edges {
                if alpha[b] > f64::NEG_INFINITY && beta[e] > f64::NEG_INFINITY {
                    expected[i] += c * (alpha[b] + pieces[i].1 + beta[e] - alpha[n]).exp();
                }
            }
        }
        let kept: Vec<(usize, f64)> = expected.iter().enumerate().filter(|&(i, &f)| f > 0.5 || pieces[i].0.chars().count() == 1).map(|(i, &f)| (i, f.max(0.5))).collect();
        let logsum = digamma(kept.iter().map(|k| k.1).sum());
        kept.into_iter().map(|(i, f)| (pieces[i].0.clone(), digamma(f) - logsum)).collect()
    }

    /// Keeps all single characters and the pieces whose removal would cost the most likelihood.
    fn prune(pieces: &[(String, f64)], words: &[(Vec<char>, f64)], target: usize, maxlen: usize) -> Vec<(String, f64)> {
        let idx: HashMap<&str, usize> = pieces.iter().enumerate().map(|(i, p)| (p.0.as_str(), i)).collect();
        let mut freq = vec![0.0f64; pieces.len()];
        let viterbi = |w: &[char], skip: Option<usize>| -> Option<(f64, Vec<usize>)> {
            let n = w.len();
            let mut best: Vec<Option<(f64, usize, usize)>> = vec![None; n + 1];
            best[0] = Some((0.0, 0, 0));
            for b in 0..n {
                let Some((base, _, _)) = best[b] else { continue };
                let mut s = String::new();
                for e in b + 1..=n.min(b + maxlen) {
                    s.push(w[e - 1]);
                    if let Some(&i) = idx.get(s.as_str()) {
                        if Some(i) == skip {
                            continue;
                        }
                        let cand = base + pieces[i].1;
                        if best[e].is_none_or(|(c, _, _)| cand > c) {
                            best[e] = Some((cand, b, i));
                        }
                    }
                }
            }
            let (score, _, _) = best[n]?;
            let (mut e, mut path) = (n, Vec::new());
            while e > 0 {
                let (_, b, i) = best[e]?;
                path.push(i);
                e = b;
            }
            Some((score, path))
        };
        for (w, c) in words {
            if let Some((_, path)) = viterbi(w, None) {
                for i in path {
                    freq[i] += c;
                }
            }
        }
        let mut loss: Vec<(usize, f64)> = Vec::new();
        for (i, p) in pieces.iter().enumerate() {
            if p.0.chars().count() == 1 {
                continue;
            }
            let chars: Vec<char> = p.0.chars().collect();
            let alt = viterbi(&chars, Some(i)).map_or(f64::NEG_INFINITY, |a| a.0);
            loss.push((i, if alt == f64::NEG_INFINITY { f64::INFINITY } else { freq[i] * (p.1 - alt) }));
        }
        loss.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal).then(a.0.cmp(&b.0)));
        let n_chars = pieces.iter().filter(|p| p.0.chars().count() == 1).count();
        let keep: HashSet<usize> = loss.iter().take(target.saturating_sub(n_chars)).map(|l| l.0).collect();
        pieces.iter().enumerate().filter(|(i, p)| p.0.chars().count() == 1 || keep.contains(i)).map(|(_, p)| p.clone()).collect()
    }
}

impl Tokenizer {
    /// Trains a Unigram model from `texts`, replacing the model (`unk_token`, when in the vocabulary, is the unknown piece).
    pub fn train_unigram<S: AsRef<str>>(&mut self, texts: impl IntoIterator<Item = S>, trainer: &UnigramTrainer) -> Result<()> {
        let counts = self.count_words(texts);
        let pieces = trainer.train(&counts);
        if pieces.is_empty() {
            return Err(Error::Msg("tokenizer: training produced an empty vocabulary".into()));
        }
        let unk = trainer.unk_token.as_ref().and_then(|u| pieces.iter().position(|p| &p.0 == u)).map(|i| i as u32);
        self.set_model(Model::Unigram(super::model::Unigram::new(pieces, unk, false)));
        self.install_specials(&trainer.special_tokens);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unigram_training_yields_a_working_tokenizer() {
        let corpus: Vec<String> = ["the cat sat on the mat", "the dog sat on the log", "a cat and a dog"].iter().cycle().take(30).map(|s| s.to_string()).collect();
        let mut t = Tokenizer::untrained("Unigram", None).unwrap();
        t.set_pre_tokenizer_json(&Json::parse(r#"{"type":"Whitespace"}"#).unwrap()).unwrap();
        t.train_unigram(&corpus, &UnigramTrainer { vocab_size: 40, special_tokens: vec!["<unk>".into()], unk_token: Some("<unk>".into()), ..Default::default() }).unwrap();
        assert!(t.get_vocab_size(false) <= 40);
        let e = t.encode("the cat sat", false).unwrap();
        assert_eq!(e.tokens.concat(), "thecatsat");
        assert!(e.tokens.len() < 9, "frequent words should be merged into pieces: {:?}", e.tokens);
        assert_eq!(t.encode("zzz", false).unwrap().ids, [t.token_to_id("<unk>").unwrap()]);
        // saved and reloaded it behaves the same
        let back = Tokenizer::from_json(&t.to_json()).unwrap();
        assert_eq!(back.encode("the dog", false).unwrap().ids, t.encode("the dog", false).unwrap().ids);
    }
}
