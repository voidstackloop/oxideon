//! SentencePiece `.model` files (protobuf): Unigram (Viterbi over a piece lattice, `f32` scores as in the reference) and BPE
//! (score-ordered pair merging), with the precompiled normalization map, dummy prefix, whitespace handling, user-defined
//! symbols, byte fallback and unknown-run merging of the original implementation.

use super::hf::charsmap::CharsMap;
use super::hf::normalizer::NormalizedString;
use crate::error::{Error, Result};
use std::collections::{BinaryHeap, HashMap};

const SPACE: char = '\u{2581}';
const UNK_PENALTY: f32 = 10.0;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PieceKind {
    Normal,
    Unknown,
    Control,
    UserDefined,
    Unused,
    Byte,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModelType {
    Unigram,
    Bpe,
    Word,
    Char,
}

#[derive(Debug, Clone)]
pub struct Piece {
    pub text: String,
    pub score: f32,
    pub kind: PieceKind,
}

struct Reader<'a> {
    b: &'a [u8],
    i: usize,
}

impl<'a> Reader<'a> {
    fn varint(&mut self) -> Result<u64> {
        let (mut v, mut shift) = (0u64, 0);
        loop {
            let c = *self.b.get(self.i).ok_or_else(|| Error::Msg("sentencepiece: truncated protobuf".into()))?;
            self.i += 1;
            v |= ((c & 0x7f) as u64) << shift;
            if c & 0x80 == 0 {
                return Ok(v);
            }
            shift += 7;
            if shift > 63 {
                return Err(Error::Msg("sentencepiece: bad varint".into()));
            }
        }
    }
    /// Next `(field, wire type, payload)`; payload is the varint value, the fixed bytes or the length-delimited bytes.
    fn field(&mut self) -> Result<Option<(u32, Value<'a>)>> {
        if self.i >= self.b.len() {
            return Ok(None);
        }
        let tag = self.varint()?;
        let (num, wt) = ((tag >> 3) as u32, tag & 7);
        let v = match wt {
            0 => Value::Varint(self.varint()?),
            1 => {
                let s = self.take(8)?;
                Value::Bytes(s)
            }
            2 => {
                let n = self.varint()? as usize;
                Value::Bytes(self.take(n)?)
            }
            5 => Value::Bytes(self.take(4)?),
            _ => return Err(Error::Msg("sentencepiece: unsupported protobuf wire type".into())),
        };
        Ok(Some((num, v)))
    }
    fn take(&mut self, n: usize) -> Result<&'a [u8]> {
        let s = self.b.get(self.i..self.i + n).ok_or_else(|| Error::Msg("sentencepiece: truncated protobuf".into()))?;
        self.i += n;
        Ok(s)
    }
}

enum Value<'a> {
    Varint(u64),
    Bytes(&'a [u8]),
}

pub struct SentencePiece {
    pub pieces: Vec<Piece>,
    pub model_type: ModelType,
    pub byte_fallback: bool,
    pub unk_id: i32,
    pub bos_id: i32,
    pub eos_id: i32,
    pub pad_id: i32,
    add_dummy_prefix: bool,
    remove_extra_whitespaces: bool,
    escape_whitespaces: bool,
    charsmap: Option<CharsMap>,
    index: HashMap<String, u32>,
    byte_ids: HashMap<u8, u32>,
    user_defined: Vec<(Vec<char>, u32)>,
    min_score: f32,
    max_score: f32,
    max_piece_chars: usize,
}

impl SentencePiece {
    pub fn from_file(path: impl AsRef<std::path::Path>) -> Result<Self> {
        let b = std::fs::read(path.as_ref()).map_err(|e| Error::Msg(format!("sentencepiece: cannot read {}: {e}", path.as_ref().display())))?;
        Self::from_bytes(&b)
    }

    pub fn from_bytes(bytes: &[u8]) -> Result<Self> {
        let mut pieces = Vec::new();
        let (mut model_type, mut byte_fallback) = (ModelType::Unigram, false);
        let (mut unk_id, mut bos_id, mut eos_id, mut pad_id) = (0i32, 1i32, 2i32, -1i32);
        let (mut dummy, mut rm_ws, mut esc_ws) = (true, true, true);
        let mut charsmap_bytes: Vec<u8> = Vec::new();
        let mut r = Reader { b: bytes, i: 0 };
        while let Some((f, v)) = r.field()? {
            match (f, v) {
                (1, Value::Bytes(b)) => {
                    let mut pr = Reader { b, i: 0 };
                    let (mut text, mut score, mut kind) = (String::new(), 0f32, 1u64);
                    while let Some((pf, pv)) = pr.field()? {
                        match (pf, pv) {
                            (1, Value::Bytes(s)) => text = String::from_utf8_lossy(s).into_owned(),
                            (2, Value::Bytes(s)) if s.len() == 4 => score = f32::from_le_bytes([s[0], s[1], s[2], s[3]]),
                            (3, Value::Varint(k)) => kind = k,
                            _ => {}
                        }
                    }
                    let kind = match kind {
                        1 => PieceKind::Normal,
                        2 => PieceKind::Unknown,
                        3 => PieceKind::Control,
                        4 => PieceKind::UserDefined,
                        5 => PieceKind::Unused,
                        6 => PieceKind::Byte,
                        k => return Err(Error::Msg(format!("sentencepiece: unknown piece type {k}"))),
                    };
                    pieces.push(Piece { text, score, kind });
                }
                (2, Value::Bytes(b)) => {
                    let mut tr = Reader { b, i: 0 };
                    while let Some((tf, tv)) = tr.field()? {
                        match (tf, tv) {
                            (3, Value::Varint(m)) => {
                                model_type = match m {
                                    1 => ModelType::Unigram,
                                    2 => ModelType::Bpe,
                                    3 => ModelType::Word,
                                    4 => ModelType::Char,
                                    _ => return Err(Error::Msg("sentencepiece: unknown model type".into())),
                                }
                            }
                            (35, Value::Varint(x)) => byte_fallback = x != 0,
                            (40, Value::Varint(x)) => unk_id = x as i64 as i32,
                            (41, Value::Varint(x)) => bos_id = x as i64 as i32,
                            (42, Value::Varint(x)) => eos_id = x as i64 as i32,
                            (43, Value::Varint(x)) => pad_id = x as i64 as i32,
                            _ => {}
                        }
                    }
                }
                (3, Value::Bytes(b)) => {
                    let mut nr = Reader { b, i: 0 };
                    while let Some((nf, nv)) = nr.field()? {
                        match (nf, nv) {
                            (2, Value::Bytes(c)) => charsmap_bytes = c.to_vec(),
                            (3, Value::Varint(x)) => dummy = x != 0,
                            (4, Value::Varint(x)) => rm_ws = x != 0,
                            (5, Value::Varint(x)) => esc_ws = x != 0,
                            _ => {}
                        }
                    }
                }
                _ => {}
            }
        }
        if pieces.is_empty() {
            return Err(Error::Msg("sentencepiece: the model has no pieces".into()));
        }
        let charsmap = if charsmap_bytes.is_empty() { None } else { Some(CharsMap::from_bytes(&charsmap_bytes)?) };
        let mut index = HashMap::new();
        let mut byte_ids = HashMap::new();
        let mut user_defined = Vec::new();
        let (mut min_score, mut max_score) = (f32::MAX, f32::MIN_POSITIVE);
        let mut max_piece_chars = 1;
        for (i, p) in pieces.iter().enumerate() {
            match p.kind {
                PieceKind::Normal | PieceKind::UserDefined => {
                    index.insert(p.text.clone(), i as u32);
                    max_piece_chars = max_piece_chars.max(p.text.chars().count());
                    if p.kind == PieceKind::UserDefined {
                        user_defined.push((p.text.chars().collect::<Vec<_>>(), i as u32));
                    } else {
                        min_score = min_score.min(p.score);
                        max_score = max_score.max(p.score);
                    }
                }
                PieceKind::Byte => {
                    if let Some(h) = p.text.strip_prefix("<0x").and_then(|t| t.strip_suffix('>')).and_then(|h| u8::from_str_radix(h, 16).ok()) {
                        byte_ids.insert(h, i as u32);
                    }
                    index.entry(p.text.clone()).or_insert(i as u32);
                }
                _ => {
                    index.entry(p.text.clone()).or_insert(i as u32);
                }
            }
        }
        if min_score == f32::MAX {
            min_score = 0.0;
        }
        user_defined.sort_by_key(|(c, _)| std::cmp::Reverse(c.len()));
        Ok(Self { pieces, model_type, byte_fallback, unk_id, bos_id, eos_id, pad_id, add_dummy_prefix: dummy, remove_extra_whitespaces: rm_ws, escape_whitespaces: esc_ws, charsmap, index, byte_ids, user_defined, min_score, max_score, max_piece_chars })
    }

    pub fn vocab_size(&self) -> usize {
        self.pieces.len()
    }
    pub fn piece_to_id(&self, p: &str) -> Option<u32> {
        self.index.get(p).copied()
    }
    pub fn id_to_piece(&self, id: u32) -> Option<&str> {
        self.pieces.get(id as usize).map(|p| p.text.as_str())
    }

    /// The normalized text the model sees (precompiled map, whitespace rules, dummy prefix, `\u{2581}` escaping).
    pub fn normalize(&self, text: &str) -> String {
        let mut ns = NormalizedString::new(text);
        if let Some(m) = &self.charsmap {
            m.normalize(&mut ns);
        }
        let mut chars: Vec<char> = ns.plain();
        if self.remove_extra_whitespaces {
            let mut out: Vec<char> = Vec::with_capacity(chars.len());
            for c in chars {
                if c == ' ' && out.last().is_none_or(|&p| p == ' ') {
                    continue;
                }
                out.push(c);
            }
            while out.last() == Some(&' ') {
                out.pop();
            }
            chars = out;
        }
        if chars.is_empty() {
            return String::new();
        }
        if self.add_dummy_prefix {
            chars.insert(0, ' ');
        }
        chars.into_iter().map(|c| if self.escape_whitespaces && c == ' ' { SPACE } else { c }).collect()
    }

    /// Id of a normal or user-defined piece, else the unknown id.
    fn lookup(&self, s: &str) -> i32 {
        match self.index.get(s) {
            Some(&id) if matches!(self.pieces[id as usize].kind, PieceKind::Normal | PieceKind::UserDefined) => id as i32,
            _ => self.unk_id,
        }
    }

    fn unigram(&self, text: &[char]) -> Vec<(usize, usize, i32)> {
        let n = text.len();
        let unk_score = self.min_score - UNK_PENALTY;
        // best[e] = (score, start, id)
        let mut best: Vec<Option<(f32, usize, i32)>> = vec![None; n + 1];
        best[0] = Some((0.0, 0, -1));
        // trie replaced by a bounded scan over candidate lengths
        let mut s = String::new();
        for b in 0..n {
            let Some((base, _, _)) = best[b] else { continue };
            let mut has_single = false;
            s.clear();
            for e in b + 1..=n.min(b + self.max_piece_chars) {
                s.push(text[e - 1]);
                if let Some(&id) = self.index.get(&s) {
                    let p = &self.pieces[id as usize];
                    if !matches!(p.kind, PieceKind::Normal | PieceKind::UserDefined) {
                        continue;
                    }
                    let len = (e - b) as f32;
                    let score = if p.kind == PieceKind::UserDefined { len * self.max_score - 0.1 } else { p.score };
                    if e - b == 1 {
                        has_single = true;
                    }
                    let cand = score + base;
                    if best[e].is_none_or(|(c, _, _)| cand > c) {
                        best[e] = Some((cand, b, id as i32));
                    }
                }
            }
            if !has_single {
                let cand = unk_score + base;
                if best[b + 1].is_none_or(|(c, _, _)| cand > c) {
                    best[b + 1] = Some((cand, b, self.unk_id));
                }
            }
        }
        let mut spans = Vec::new();
        let mut e = n;
        while e > 0 {
            let (_, b, id) = best[e].unwrap_or((0.0, e - 1, self.unk_id));
            spans.push((b, e, id));
            e = b;
        }
        spans.reverse();
        spans
    }

    fn bpe(&self, text: &[char]) -> Vec<(usize, usize, i32)> {
        // symbols start as characters; user-defined pieces are frozen single symbols
        struct Sym {
            start: usize,
            end: usize,
            prev: i64,
            next: i64,
            frozen: bool,
            alive: bool,
        }
        let mut syms: Vec<Sym> = Vec::new();
        let mut i = 0;
        while i < text.len() {
            let ud = self.user_defined.iter().find(|(c, _)| text[i..].starts_with(c));
            let len = ud.map_or(1, |(c, _)| c.len());
            syms.push(Sym { start: i, end: i + len, prev: syms.len() as i64 - 1, next: -1, frozen: ud.is_some(), alive: true });
            i += len;
        }
        let m = syms.len();
        for k in 0..m {
            syms[k].next = if k + 1 < m { k as i64 + 1 } else { -1 };
        }
        #[derive(PartialEq)]
        struct Cand {
            score: f32,
            left: usize,
            right: usize,
            size: usize,
        }
        impl Eq for Cand {}
        impl PartialOrd for Cand {
            fn partial_cmp(&self, o: &Self) -> Option<std::cmp::Ordering> {
                Some(self.cmp(o))
            }
        }
        impl Ord for Cand {
            fn cmp(&self, o: &Self) -> std::cmp::Ordering {
                // higher score first; on equal scores the smaller left position first
                self.score.partial_cmp(&o.score).unwrap_or(std::cmp::Ordering::Equal).then(o.left.cmp(&self.left))
            }
        }
        let mut heap: BinaryHeap<Cand> = BinaryHeap::new();
        let try_pair = |syms: &Vec<Sym>, l: i64, r: i64, heap: &mut BinaryHeap<Cand>| {
            if l < 0 || r < 0 {
                return;
            }
            let (l, r) = (l as usize, r as usize);
            if syms[l].frozen || syms[r].frozen {
                return;
            }
            let s: String = text[syms[l].start..syms[r].end].iter().collect();
            if let Some(&id) = self.index.get(&s) {
                let p = &self.pieces[id as usize];
                if matches!(p.kind, PieceKind::Normal | PieceKind::UserDefined) {
                    heap.push(Cand { score: p.score, left: l, right: r, size: syms[r].end - syms[l].start });
                }
            }
        };
        for k in 1..m {
            try_pair(&syms, k as i64 - 1, k as i64, &mut heap);
        }
        while let Some(c) = heap.pop() {
            if !syms[c.left].alive || !syms[c.right].alive || syms[c.left].end != syms[c.right].start || syms[c.right].end - syms[c.left].start != c.size || syms[c.left].next != c.right as i64 {
                continue;
            }
            syms[c.left].end = syms[c.right].end;
            syms[c.right].alive = false;
            let nx = syms[c.right].next;
            syms[c.left].next = nx;
            if nx >= 0 {
                syms[nx as usize].prev = c.left as i64;
            }
            let pv = syms[c.left].prev;
            try_pair(&syms, pv, c.left as i64, &mut heap);
            try_pair(&syms, c.left as i64, nx, &mut heap);
        }
        syms.iter()
            .filter(|s| s.alive)
            .map(|s| {
                let t: String = text[s.start..s.end].iter().collect();
                (s.start, s.end, self.lookup(&t))
            })
            .collect()
    }

    /// Encodes to ids. Unknown runs are merged into one unknown piece, or spelled as byte pieces with byte fallback.
    pub fn encode(&self, text: &str) -> Result<Vec<u32>> {
        Ok(self.encode_pieces_ids(text)?.into_iter().map(|(_, id)| id).collect())
    }

    /// Encodes to `(piece text, id)` pairs.
    pub fn encode_pieces_ids(&self, text: &str) -> Result<Vec<(String, u32)>> {
        let norm: Vec<char> = self.normalize(text).chars().collect();
        if norm.is_empty() {
            return Ok(vec![]);
        }
        let spans = match self.model_type {
            ModelType::Unigram => self.unigram(&norm),
            ModelType::Bpe => self.bpe(&norm),
            ModelType::Char => (0..norm.len()).map(|i| (i, i + 1, self.lookup(&norm[i].to_string()))).collect(),
            ModelType::Word => return Err(Error::Msg("sentencepiece: WORD models are not supported".into())),
        };
        let mut out: Vec<(String, u32)> = Vec::new();
        let mut prev_unk = false;
        for (b, e, id) in spans {
            let surface: String = norm[b..e].iter().collect();
            let is_unk = id < 0 || id == self.unk_id;
            if is_unk {
                if self.byte_fallback && surface.bytes().all(|x| self.byte_ids.contains_key(&x)) {
                    for x in surface.bytes() {
                        let bid = self.byte_ids[&x];
                        out.push((self.pieces[bid as usize].text.clone(), bid));
                    }
                    prev_unk = false;
                } else if prev_unk {
                    out.last_mut().unwrap().0.push_str(&surface);
                } else {
                    out.push((surface, self.unk_id.max(0) as u32));
                    prev_unk = true;
                }
            } else {
                out.push((surface, id as u32));
                prev_unk = false;
            }
        }
        Ok(out)
    }

    /// Ids with optional `<s>` / `</s>` around them.
    pub fn encode_with_special(&self, text: &str, add_bos: bool, add_eos: bool) -> Result<Vec<u32>> {
        let mut ids = Vec::new();
        if add_bos && self.bos_id >= 0 {
            ids.push(self.bos_id as u32);
        }
        ids.extend(self.encode(text)?);
        if add_eos && self.eos_id >= 0 {
            ids.push(self.eos_id as u32);
        }
        Ok(ids)
    }

    pub fn decode(&self, ids: &[u32]) -> Result<String> {
        let mut bytes: Vec<u8> = Vec::new();
        let mut first = true;
        for &id in ids {
            let p = self.pieces.get(id as usize).ok_or_else(|| Error::Msg(format!("sentencepiece: invalid id {id}")))?;
            match p.kind {
                PieceKind::Control => {}
                PieceKind::Unknown => bytes.extend_from_slice(" \u{2047} ".as_bytes()),
                PieceKind::Byte => match p.text.strip_prefix("<0x").and_then(|t| t.strip_suffix('>')).and_then(|h| u8::from_str_radix(h, 16).ok()) {
                    Some(b) => bytes.push(b),
                    None => bytes.extend_from_slice(p.text.as_bytes()),
                },
                _ => {
                    let mut t: String = p.text.chars().map(|c| if c == SPACE { ' ' } else { c }).collect();
                    if first && self.add_dummy_prefix && t.starts_with(' ') {
                        t.remove(0);
                    }
                    bytes.extend_from_slice(t.as_bytes());
                    first = false;
                    continue;
                }
            }
            if p.kind != PieceKind::Control {
                first = false;
            }
        }
        Ok(String::from_utf8_lossy(&bytes).into_owned())
    }
}
