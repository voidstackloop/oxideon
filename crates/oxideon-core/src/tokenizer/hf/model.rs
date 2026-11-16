//! Tokenization models: BPE, WordPiece, Unigram and WordLevel (Hugging Face `tokenizers` semantics).

use crate::error::{Error, Result};
use crate::tokenizer::json::Json;
use std::collections::HashMap;
use std::sync::Mutex;

/// One token of a word; `start..end` index the characters of that word.
#[derive(Debug, Clone, PartialEq)]
pub struct Token {
    pub id: u32,
    pub value: String,
    pub start: usize,
    pub end: usize,
}

#[derive(Debug, Clone, Default)]
pub struct Vocab {
    pub map: HashMap<String, u32>,
    pub rev: Vec<Option<String>>,
}

impl Vocab {
    pub fn from_pairs(pairs: impl IntoIterator<Item = (String, u32)>) -> Self {
        let mut v = Vocab::default();
        for (s, id) in pairs {
            v.insert(s, id);
        }
        v
    }
    pub fn insert(&mut self, s: String, id: u32) {
        if self.rev.len() <= id as usize {
            self.rev.resize(id as usize + 1, None);
        }
        self.rev[id as usize] = Some(s.clone());
        self.map.insert(s, id);
    }
    pub fn id(&self, s: &str) -> Option<u32> {
        self.map.get(s).copied()
    }
    pub fn token(&self, id: u32) -> Option<&str> {
        self.rev.get(id as usize).and_then(|o| o.as_deref())
    }
    pub fn len(&self) -> usize {
        self.map.len()
    }
    pub fn is_empty(&self) -> bool {
        self.map.is_empty()
    }
}

fn vocab_from_json(j: &Json) -> Result<Vocab> {
    let o = j.as_object().ok_or_else(|| Error::Msg("tokenizer: model vocab must be an object".into()))?;
    o.iter().map(|(k, v)| v.as_u64().map(|id| (k.clone(), id as u32)).ok_or_else(|| Error::Msg("tokenizer: vocab ids must be integers".into()))).collect::<Result<Vec<_>>>().map(Vocab::from_pairs)
}

pub struct Bpe {
    pub vocab: Vocab,
    /// `(left id, right id) -> (rank, merged id)`
    merges: HashMap<(u32, u32), (u32, u32)>,
    pub merge_list: Vec<(String, String)>,
    pub unk_token: Option<String>,
    pub continuing_subword_prefix: Option<String>,
    pub end_of_word_suffix: Option<String>,
    pub fuse_unk: bool,
    pub byte_fallback: bool,
    pub ignore_merges: bool,
    cache: Mutex<HashMap<String, Vec<Token>>>,
}

impl std::fmt::Debug for Bpe {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Bpe(vocab={}, merges={})", self.vocab.len(), self.merge_list.len())
    }
}

impl Bpe {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        vocab: Vocab,
        merge_list: Vec<(String, String)>,
        unk_token: Option<String>,
        continuing_subword_prefix: Option<String>,
        end_of_word_suffix: Option<String>,
        fuse_unk: bool,
        byte_fallback: bool,
        ignore_merges: bool,
    ) -> Result<Self> {
        let prefix_len = continuing_subword_prefix.as_ref().map_or(0, |p| p.len());
        let mut merges = HashMap::new();
        for (rank, (a, b)) in merge_list.iter().enumerate() {
            let ai = vocab.id(a).ok_or_else(|| Error::Msg(format!("tokenizer: merge token {a:?} not in vocab")))?;
            let bi = vocab.id(b).ok_or_else(|| Error::Msg(format!("tokenizer: merge token {b:?} not in vocab")))?;
            let merged = format!("{a}{}", b.get(prefix_len..).unwrap_or(b));
            let ni = vocab.id(&merged).ok_or_else(|| Error::Msg(format!("tokenizer: merged token {merged:?} not in vocab")))?;
            merges.entry((ai, bi)).or_insert((rank as u32, ni));
        }
        Ok(Self { vocab, merges, merge_list, unk_token, continuing_subword_prefix, end_of_word_suffix, fuse_unk, byte_fallback, ignore_merges, cache: Mutex::new(HashMap::new()) })
    }

    pub fn from_json(m: &Json) -> Result<Self> {
        let vocab = vocab_from_json(m.get("vocab").ok_or_else(|| Error::Msg("tokenizer: BPE needs vocab".into()))?)?;
        let mut merge_list = Vec::new();
        for e in m.get("merges").and_then(Json::as_array).unwrap_or(&[]) {
            match e {
                Json::Str(s) => {
                    let (a, b) = s.split_once(' ').ok_or_else(|| Error::Msg("tokenizer: bad merge entry".into()))?;
                    merge_list.push((a.to_string(), b.to_string()));
                }
                Json::Arr(p) if p.len() == 2 => merge_list.push((p[0].as_str().unwrap_or("").to_string(), p[1].as_str().unwrap_or("").to_string())),
                _ => return Err(Error::Msg("tokenizer: bad merge entry".into())),
            }
        }
        let s = |k: &str| m.get(k).and_then(Json::as_str).map(str::to_string);
        let b = |k: &str, d: bool| m.get(k).and_then(Json::as_bool).unwrap_or(d);
        Self::new(vocab, merge_list, s("unk_token"), s("continuing_subword_prefix").filter(|p| !p.is_empty()), s("end_of_word_suffix").filter(|p| !p.is_empty()), b("fuse_unk", false), b("byte_fallback", false), b("ignore_merges", false))
    }

    pub fn tokenize(&self, word: &[char]) -> Result<Vec<Token>> {
        if word.is_empty() {
            return Ok(vec![]);
        }
        let key: String = word.iter().collect();
        if word.len() < 256 {
            if let Some(hit) = self.cache.lock().unwrap().get(&key) {
                return Ok(hit.clone());
            }
        }
        let out = self.tokenize_uncached(word, &key)?;
        if word.len() < 256 {
            let mut c = self.cache.lock().unwrap();
            if c.len() < 100_000 {
                c.insert(key, out.clone());
            }
        }
        Ok(out)
    }

    fn tokenize_uncached(&self, word: &[char], key: &str) -> Result<Vec<Token>> {
        if self.ignore_merges {
            if let Some(id) = self.vocab.id(key) {
                return Ok(vec![Token { id, value: key.to_string(), start: 0, end: word.len() }]);
            }
        }
        // initial symbols: (id, start, end, is_unk)
        struct Sym {
            id: u32,
            start: usize,
            end: usize,
        }
        let mut syms: Vec<Sym> = Vec::with_capacity(word.len());
        let mut unk_run: Option<usize> = None; // index of the last unk symbol, for fuse_unk
        for (i, &c) in word.iter().enumerate() {
            let mut piece = String::new();
            if i > 0 {
                if let Some(p) = &self.continuing_subword_prefix {
                    piece.push_str(p);
                }
            }
            piece.push(c);
            if i == word.len() - 1 {
                if let Some(s) = &self.end_of_word_suffix {
                    piece.push_str(s);
                }
            }
            if let Some(id) = self.vocab.id(&piece) {
                syms.push(Sym { id, start: i, end: i + 1 });
                unk_run = None;
                continue;
            }
            if self.byte_fallback {
                let mut buf = [0u8; 4];
                let ids: Option<Vec<u32>> = c.encode_utf8(&mut buf).as_bytes().iter().map(|b| self.vocab.id(&format!("<0x{b:02X}>"))).collect();
                if let Some(ids) = ids {
                    for id in ids {
                        syms.push(Sym { id, start: i, end: i + 1 });
                    }
                    unk_run = None;
                    continue;
                }
            }
            if let Some(unk) = &self.unk_token {
                let uid = self.vocab.id(unk).ok_or_else(|| Error::Msg(format!("tokenizer: unk token {unk:?} not in vocab")))?;
                match unk_run {
                    Some(k) if self.fuse_unk => syms[k].end = i + 1,
                    _ => {
                        syms.push(Sym { id: uid, start: i, end: i + 1 });
                        unk_run = Some(syms.len() - 1);
                    }
                }
            }
        }
        // repeatedly merge the lowest-rank adjacent pair (leftmost on ties)
        loop {
            let mut best: Option<(u32, usize, u32)> = None;
            for k in 0..syms.len().saturating_sub(1) {
                if let Some(&(rank, new_id)) = self.merges.get(&(syms[k].id, syms[k + 1].id)) {
                    if best.is_none_or(|(r, _, _)| rank < r) {
                        best = Some((rank, k, new_id));
                    }
                }
            }
            let Some((_, k, new_id)) = best else { break };
            syms[k].id = new_id;
            syms[k].end = syms[k + 1].end;
            syms.remove(k + 1);
        }
        Ok(syms.into_iter().map(|s| Token { id: s.id, value: self.vocab.token(s.id).unwrap_or("").to_string(), start: s.start, end: s.end }).collect())
    }
}

#[derive(Debug)]
pub struct WordPiece {
    pub vocab: Vocab,
    pub unk_token: String,
    pub continuing_subword_prefix: String,
    pub max_input_chars_per_word: usize,
}

impl WordPiece {
    pub fn from_json(m: &Json) -> Result<Self> {
        Ok(Self {
            vocab: vocab_from_json(m.get("vocab").ok_or_else(|| Error::Msg("tokenizer: WordPiece needs vocab".into()))?)?,
            unk_token: m.get("unk_token").and_then(Json::as_str).unwrap_or("[UNK]").to_string(),
            continuing_subword_prefix: m.get("continuing_subword_prefix").and_then(Json::as_str).unwrap_or("##").to_string(),
            max_input_chars_per_word: m.get("max_input_chars_per_word").and_then(Json::as_u64).unwrap_or(100) as usize,
        })
    }

    pub fn tokenize(&self, word: &[char]) -> Result<Vec<Token>> {
        let unk = || -> Result<Vec<Token>> {
            let id = self.vocab.id(&self.unk_token).ok_or_else(|| Error::Msg(format!("tokenizer: unk token {:?} not in vocab", self.unk_token)))?;
            Ok(vec![Token { id, value: self.unk_token.clone(), start: 0, end: word.len() }])
        };
        if word.len() > self.max_input_chars_per_word {
            return unk();
        }
        let mut out = Vec::new();
        let mut start = 0;
        while start < word.len() {
            let mut end = word.len();
            let mut found = None;
            while start < end {
                let mut sub: String = word[start..end].iter().collect();
                if start > 0 {
                    sub = format!("{}{sub}", self.continuing_subword_prefix);
                }
                if let Some(id) = self.vocab.id(&sub) {
                    found = Some(Token { id, value: sub, start, end });
                    break;
                }
                end -= 1;
            }
            match found {
                Some(t) => {
                    start = t.end;
                    out.push(t);
                }
                None => return unk(),
            }
        }
        Ok(out)
    }
}

#[derive(Debug)]
pub struct WordLevel {
    pub vocab: Vocab,
    pub unk_token: String,
}

impl WordLevel {
    pub fn from_json(m: &Json) -> Result<Self> {
        Ok(Self {
            vocab: vocab_from_json(m.get("vocab").ok_or_else(|| Error::Msg("tokenizer: WordLevel needs vocab".into()))?)?,
            unk_token: m.get("unk_token").and_then(Json::as_str).unwrap_or("<unk>").to_string(),
        })
    }
    pub fn tokenize(&self, word: &[char]) -> Result<Vec<Token>> {
        let w: String = word.iter().collect();
        let (id, value) = match self.vocab.id(&w) {
            Some(id) => (id, w),
            None => (self.vocab.id(&self.unk_token).ok_or_else(|| Error::Msg(format!("tokenizer: unk token {:?} not in vocab", self.unk_token)))?, self.unk_token.clone()),
        };
        Ok(vec![Token { id, value, start: 0, end: word.len() }])
    }
}

#[derive(Debug)]
pub struct Unigram {
    pub pieces: Vec<(String, f64)>,
    index: HashMap<String, u32>,
    pub unk_id: Option<u32>,
    pub byte_fallback: bool,
    min_score: f64,
    max_piece_chars: usize,
}

impl Unigram {
    pub fn new(pieces: Vec<(String, f64)>, unk_id: Option<u32>, byte_fallback: bool) -> Self {
        let index = pieces.iter().enumerate().map(|(i, (p, _))| (p.clone(), i as u32)).collect();
        let min_score = pieces.iter().map(|p| p.1).fold(f64::INFINITY, f64::min);
        let max_piece_chars = pieces.iter().map(|(p, _)| p.chars().count()).max().unwrap_or(1);
        Self { pieces, index, unk_id, byte_fallback, min_score, max_piece_chars }
    }

    pub fn from_json(m: &Json) -> Result<Self> {
        let vocab = m.get("vocab").and_then(Json::as_array).ok_or_else(|| Error::Msg("tokenizer: Unigram needs vocab".into()))?;
        let pieces = vocab
            .iter()
            .map(|e| match e.as_array() {
                Some([p, s]) => Ok((p.as_str().unwrap_or("").to_string(), s.as_f64().unwrap_or(0.0))),
                _ => Err(Error::Msg("tokenizer: Unigram vocab entries must be [piece, score]".into())),
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(Self::new(pieces, m.get("unk_id").and_then(Json::as_u64).map(|u| u as u32), m.get("byte_fallback").and_then(Json::as_bool).unwrap_or(false)))
    }

    pub fn id(&self, s: &str) -> Option<u32> {
        self.index.get(s).copied()
    }

    /// Viterbi segmentation of one word into pieces (with the unknown-character penalty of SentencePiece).
    pub fn tokenize(&self, word: &[char]) -> Result<Vec<Token>> {
        let n = word.len();
        if n == 0 {
            return Ok(vec![]);
        }
        let unk_score = self.min_score - 10.0;
        // best[e] = (score, start, id or None for unknown)
        let mut best: Vec<Option<(f64, usize, Option<u32>)>> = vec![None; n + 1];
        best[0] = Some((0.0, 0, None));
        // candidate nodes ending at e, in order of ascending start, evaluated with strict improvement (as HF does)
        let mut ends: Vec<Vec<(usize, f64, Option<u32>)>> = vec![vec![]; n + 1];
        for b in 0..n {
            let mut has_single = false;
            let mut s = String::new();
            for e in b + 1..=n.min(b + self.max_piece_chars) {
                s.push(word[e - 1]);
                if let Some(&id) = self.index.get(&s) {
                    ends[e].push((b, self.pieces[id as usize].1, Some(id)));
                    if e - b == 1 {
                        has_single = true;
                    }
                }
            }
            if !has_single {
                ends[b + 1].push((b, unk_score, None));
            }
        }
        for e in 1..=n {
            let mut cur: Option<(f64, usize, Option<u32>)> = None;
            for &(b, sc, id) in &ends[e] {
                if let Some((prev, _, _)) = best[b] {
                    let total = prev + sc;
                    if cur.is_none_or(|(c, _, _)| total > c) {
                        cur = Some((total, b, id));
                    }
                }
            }
            best[e] = cur;
        }
        // backtrace, fusing consecutive unknown characters
        let mut spans: Vec<(usize, usize, Option<u32>)> = Vec::new();
        let mut e = n;
        while e > 0 {
            let (_, b, id) = best[e].ok_or_else(|| Error::Msg("tokenizer: Unigram lattice has no path".into()))?;
            if id.is_none() {
                if let Some(last) = spans.last_mut() {
                    if last.2.is_none() && last.0 == e {
                        last.0 = b;
                        e = b;
                        continue;
                    }
                }
            }
            spans.push((b, e, id));
            e = b;
        }
        spans.reverse();
        let mut out = Vec::new();
        for (b, e, id) in spans {
            let text: String = word[b..e].iter().collect();
            match id {
                Some(id) => out.push(Token { id, value: text, start: b, end: e }),
                None => {
                    if self.byte_fallback {
                        let ids: Option<Vec<(u32, String)>> = text.bytes().map(|x| { let s = format!("<0x{x:02X}>"); self.id(&s).map(|i| (i, s)) }).collect();
                        if let Some(ids) = ids {
                            out.extend(ids.into_iter().map(|(id, value)| Token { id, value, start: b, end: e }));
                            continue;
                        }
                    }
                    let uid = self.unk_id.ok_or_else(|| Error::Msg("tokenizer: Unigram has no unk_id for an unknown character".into()))?;
                    out.push(Token { id: uid, value: text, start: b, end: e });
                }
            }
        }
        Ok(out)
    }
}

#[derive(Debug)]
pub enum Model {
    Bpe(Bpe),
    WordPiece(WordPiece),
    WordLevel(WordLevel),
    Unigram(Unigram),
}

impl Model {
    pub fn from_json(m: &Json) -> Result<Model> {
        // older files carry no "type": the shape decides (as the reference's untagged deserialization does)
        let inferred = if m.get("merges").is_some() {
            Some("BPE")
        } else if m.get("vocab").is_some_and(|v| v.as_array().is_some()) {
            Some("Unigram")
        } else if m.get("continuing_subword_prefix").is_some() || m.get("max_input_chars_per_word").is_some() {
            Some("WordPiece")
        } else if m.get("vocab").is_some() {
            Some("WordLevel")
        } else {
            None
        };
        Ok(match m.get("type").and_then(Json::as_str).or(inferred) {
            Some("BPE") | None => Model::Bpe(Bpe::from_json(m)?),
            Some("WordPiece") => Model::WordPiece(WordPiece::from_json(m)?),
            Some("WordLevel") => Model::WordLevel(WordLevel::from_json(m)?),
            Some("Unigram") => Model::Unigram(Unigram::from_json(m)?),
            Some(o) => return Err(Error::Msg(format!("tokenizer: unsupported model {o}"))),
        })
    }

    pub fn tokenize(&self, word: &[char]) -> Result<Vec<Token>> {
        match self {
            Model::Bpe(m) => m.tokenize(word),
            Model::WordPiece(m) => m.tokenize(word),
            Model::WordLevel(m) => m.tokenize(word),
            Model::Unigram(m) => m.tokenize(word),
        }
    }

    pub fn token_to_id(&self, t: &str) -> Option<u32> {
        match self {
            Model::Bpe(m) => m.vocab.id(t),
            Model::WordPiece(m) => m.vocab.id(t),
            Model::WordLevel(m) => m.vocab.id(t),
            Model::Unigram(m) => m.id(t),
        }
    }

    pub fn id_to_token(&self, id: u32) -> Option<String> {
        match self {
            Model::Bpe(m) => m.vocab.token(id).map(str::to_string),
            Model::WordPiece(m) => m.vocab.token(id).map(str::to_string),
            Model::WordLevel(m) => m.vocab.token(id).map(str::to_string),
            Model::Unigram(m) => m.pieces.get(id as usize).map(|p| p.0.clone()),
        }
    }

    pub fn vocab_size(&self) -> usize {
        match self {
            Model::Bpe(m) => m.vocab.len(),
            Model::WordPiece(m) => m.vocab.len(),
            Model::WordLevel(m) => m.vocab.len(),
            Model::Unigram(m) => m.pieces.len(),
        }
    }
}
