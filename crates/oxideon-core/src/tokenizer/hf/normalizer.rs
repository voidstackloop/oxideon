//! Normalizers (Hugging Face `tokenizers` semantics) over a [`NormalizedString`] that remembers, for every
//! normalized character, the byte range of the original text it came from (for offsets).

use crate::error::{Error, Result};
use crate::tokenizer::json::Json;
use crate::tokenizer::regex::Regex;
use crate::tokenizer::unicode;

/// Byte range in the original text.
pub type Span = (usize, usize);

#[derive(Debug, Clone)]
pub struct NormalizedString {
    pub chars: Vec<(char, Span)>,
}

/// A string or regular-expression pattern (`{"String": ".."}` / `{"Regex": ".."}` in tokenizer.json).
#[derive(Debug, Clone)]
pub enum Pattern {
    Str(String),
    Regex(Regex),
}

impl Pattern {
    pub fn from_json(j: &Json) -> Result<Pattern> {
        if let Some(s) = j.get("String").and_then(Json::as_str) {
            return Ok(Pattern::Str(s.to_string()));
        }
        if let Some(r) = j.get("Regex").and_then(Json::as_str) {
            return Ok(Pattern::Regex(Regex::new(r)?));
        }
        Err(Error::Msg("tokenizer: pattern must be {\"String\": ..} or {\"Regex\": ..}".into()))
    }

    /// Non-overlapping matches as char ranges.
    pub fn find_all(&self, text: &[char]) -> Vec<(usize, usize)> {
        match self {
            Pattern::Regex(r) => r.find_all(text).into_iter().filter(|(a, b)| a != b).collect(),
            Pattern::Str(s) => {
                let pat: Vec<char> = s.chars().collect();
                let mut out = Vec::new();
                if pat.is_empty() {
                    return out;
                }
                let mut i = 0;
                while i + pat.len() <= text.len() {
                    if text[i..i + pat.len()] == pat[..] {
                        out.push((i, i + pat.len()));
                        i += pat.len();
                    } else {
                        i += 1;
                    }
                }
                out
            }
        }
    }
}

impl NormalizedString {
    pub fn new(text: &str) -> Self {
        Self { chars: text.char_indices().map(|(i, c)| (c, (i, i + c.len_utf8()))).collect() }
    }

    pub fn get(&self) -> String {
        self.chars.iter().map(|&(c, _)| c).collect()
    }
    pub fn plain(&self) -> Vec<char> {
        self.chars.iter().map(|&(c, _)| c).collect()
    }
    pub fn is_empty(&self) -> bool {
        self.chars.is_empty()
    }

    /// Replaces every character by zero or more characters; the results keep the source character's span.
    pub fn flat_map(&mut self, mut f: impl FnMut(char) -> Vec<char>) {
        let old = std::mem::take(&mut self.chars);
        for (c, sp) in old {
            for n in f(c) {
                self.chars.push((n, sp));
            }
        }
    }
    pub fn retain(&mut self, mut f: impl FnMut(char) -> bool) {
        self.chars.retain(|&(c, _)| f(c));
    }

    /// Inserts text at the front; the new characters take the span of the first character.
    pub fn prepend(&mut self, s: &str) {
        let sp = self.chars.first().map(|&(_, sp)| sp).unwrap_or((0, 0));
        let mut v: Vec<(char, Span)> = s.chars().map(|c| (c, sp)).collect();
        v.append(&mut self.chars);
        self.chars = v;
    }
    /// Appends text; the new characters take the span of the last character.
    pub fn append(&mut self, s: &str) {
        let sp = self.chars.last().map(|&(_, sp)| sp).unwrap_or((0, 0));
        self.chars.extend(s.chars().map(|c| (c, sp)));
    }

    pub fn lowercase(&mut self) {
        self.flat_map(|c| c.to_lowercase().collect());
    }
    pub fn uppercase(&mut self) {
        self.flat_map(|c| c.to_uppercase().collect());
    }

    pub fn nfd(&mut self) {
        self.chars = unicode::decompose_with(&self.chars, false);
    }
    pub fn nfkd(&mut self) {
        self.chars = unicode::decompose_with(&self.chars, true);
    }
    pub fn nfc(&mut self) {
        let d = unicode::decompose_with(&self.chars, false);
        self.chars = unicode::compose_with(d, |a, _| a);
    }
    pub fn nfkc(&mut self) {
        let d = unicode::decompose_with(&self.chars, true);
        self.chars = unicode::compose_with(d, |a, _| a);
    }

    pub fn strip(&mut self, left: bool, right: bool) {
        if right {
            while self.chars.last().is_some_and(|&(c, _)| c.is_whitespace()) {
                self.chars.pop();
            }
        }
        if left {
            let n = self.chars.iter().take_while(|&&(c, _)| c.is_whitespace()).count();
            self.chars.drain(..n);
        }
    }

    /// Replaces every match; replacement characters take the span of the last replaced character (as the reference does).
    pub fn replace(&mut self, pat: &Pattern, content: &str) {
        let plain = self.plain();
        let matches = pat.find_all(&plain);
        if matches.is_empty() {
            return;
        }
        let mut out = Vec::with_capacity(self.chars.len());
        let mut pos = 0;
        for (a, b) in matches {
            out.extend_from_slice(&self.chars[pos..a]);
            let span = self.chars[b - 1].1;
            out.extend(content.chars().map(|c| (c, span)));
            pos = b;
        }
        out.extend_from_slice(&self.chars[pos..]);
        self.chars = out;
    }
}

fn is_cjk(c: char) -> bool {
    matches!(c as u32, 0x4E00..=0x9FFF | 0x3400..=0x4DBF | 0x20000..=0x2A6DF | 0x2A700..=0x2B73F | 0x2B740..=0x2B81F | 0x2B920..=0x2CEAF | 0xF900..=0xFAFF | 0x2F800..=0x2FA1F)
}

#[derive(Debug, Clone)]
pub enum Normalizer {
    Bert { clean_text: bool, handle_chinese_chars: bool, strip_accents: Option<bool>, lowercase: bool },
    Sequence(Vec<Normalizer>),
    Lowercase,
    Uppercase,
    Nfc,
    Nfd,
    Nfkc,
    Nfkd,
    StripAccents,
    Strip { left: bool, right: bool },
    Replace { pattern: Pattern, content: String },
    Prepend(String),
    Append(String),
    /// SentencePiece's precompiled character map (`spm_precompiled`), applied with a double-array trie.
    Precompiled(super::charsmap::CharsMap),
}

impl Normalizer {
    pub fn from_json(j: &Json) -> Result<Normalizer> {
        let ty = j.get("type").and_then(Json::as_str).ok_or_else(|| Error::Msg("tokenizer: normalizer without type".into()))?;
        let flag = |k: &str, d: bool| j.get(k).and_then(Json::as_bool).unwrap_or(d);
        Ok(match ty {
            "BertNormalizer" => Normalizer::Bert {
                clean_text: flag("clean_text", true),
                handle_chinese_chars: flag("handle_chinese_chars", true),
                strip_accents: j.get("strip_accents").and_then(Json::as_bool),
                lowercase: flag("lowercase", true),
            },
            "Sequence" => Normalizer::Sequence(
                j.get("normalizers").and_then(Json::as_array).ok_or_else(|| Error::Msg("tokenizer: Sequence needs normalizers".into()))?.iter().map(Normalizer::from_json).collect::<Result<_>>()?,
            ),
            "Lowercase" => Normalizer::Lowercase,
            "Uppercase" => Normalizer::Uppercase,
            "NFC" => Normalizer::Nfc,
            "NFD" => Normalizer::Nfd,
            "NFKC" => Normalizer::Nfkc,
            "NFKD" => Normalizer::Nfkd,
            "StripAccents" => Normalizer::StripAccents,
            "Strip" => Normalizer::Strip { left: flag("strip_left", true), right: flag("strip_right", true) },
            "Replace" => Normalizer::Replace {
                pattern: Pattern::from_json(j.get("pattern").ok_or_else(|| Error::Msg("tokenizer: Replace needs pattern".into()))?)?,
                content: j.get("content").and_then(Json::as_str).unwrap_or("").to_string(),
            },
            "Prepend" => Normalizer::Prepend(j.get("prepend").and_then(Json::as_str).unwrap_or("").to_string()),
            "Append" => Normalizer::Append(j.get("append").and_then(Json::as_str).unwrap_or("").to_string()),
            "Precompiled" => {
                let b64 = j.get("precompiled_charsmap").and_then(Json::as_str).ok_or_else(|| Error::Msg("tokenizer: Precompiled needs precompiled_charsmap".into()))?;
                Normalizer::Precompiled(super::charsmap::CharsMap::from_bytes(&super::base64_decode(b64)?)?)
            }
            other => return Err(Error::Msg(format!("tokenizer: unsupported normalizer {other}"))),
        })
    }

    pub fn normalize(&self, s: &mut NormalizedString) {
        match self {
            Normalizer::Bert { clean_text, handle_chinese_chars, strip_accents, lowercase } => {
                if *clean_text {
                    s.retain(|c| !(c == '\0' || c == '\u{fffd}' || (c != '\t' && c != '\n' && c != '\r' && matches!(unicode::category(c), "Cc" | "Cf" | "Co"))));
                    s.flat_map(|c| vec![if c.is_whitespace() { ' ' } else { c }]);
                }
                if *handle_chinese_chars {
                    s.flat_map(|c| if is_cjk(c) { vec![' ', c, ' '] } else { vec![c] });
                }
                if strip_accents.unwrap_or(*lowercase) {
                    s.nfd();
                    s.retain(|c| unicode::category(c) != "Mn");
                }
                if *lowercase {
                    s.lowercase();
                }
            }
            Normalizer::Sequence(v) => v.iter().for_each(|n| n.normalize(s)),
            Normalizer::Lowercase => s.lowercase(),
            Normalizer::Uppercase => s.uppercase(),
            Normalizer::Nfc => s.nfc(),
            Normalizer::Nfd => s.nfd(),
            Normalizer::Nfkc => s.nfkc(),
            Normalizer::Nfkd => s.nfkd(),
            Normalizer::StripAccents => s.retain(|c| unicode::category(c) != "Mn"),
            Normalizer::Strip { left, right } => s.strip(*left, *right),
            Normalizer::Replace { pattern, content } => s.replace(pattern, content),
            Normalizer::Prepend(p) => {
                if !s.is_empty() {
                    s.prepend(p)
                }
            }
            Normalizer::Append(p) => {
                if !s.is_empty() {
                    s.append(p)
                }
            }
            Normalizer::Precompiled(m) => m.normalize(s),
        }
    }
}
