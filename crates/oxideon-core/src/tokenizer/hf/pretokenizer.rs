//! Pre-tokenizers: split normalized text into "words" before the model runs (Hugging Face semantics).

use super::normalizer::{NormalizedString, Pattern};
use crate::error::{Error, Result};
use crate::tokenizer::json::Json;
use crate::tokenizer::regex::Regex;
use crate::tokenizer::unicode;

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SplitBehavior {
    Removed,
    Isolated,
    MergedWithPrevious,
    MergedWithNext,
    Contiguous,
}

impl SplitBehavior {
    fn from_json(j: Option<&Json>, default: SplitBehavior) -> Result<Self> {
        Ok(match j.and_then(Json::as_str) {
            None => default,
            Some("Removed") => Self::Removed,
            Some("Isolated") => Self::Isolated,
            Some("MergedWithPrevious") => Self::MergedWithPrevious,
            Some("MergedWithNext") => Self::MergedWithNext,
            Some("Contiguous") => Self::Contiguous,
            Some(o) => return Err(Error::Msg(format!("tokenizer: unknown split behavior {o}"))),
        })
    }
}

#[derive(Debug, Clone)]
pub enum PrependScheme {
    Always,
    First,
    Never,
}

#[derive(Debug, Clone)]
pub enum PreTokenizer {
    Bert,
    ByteLevel { add_prefix_space: bool, use_regex: bool },
    Whitespace,
    WhitespaceSplit,
    Metaspace { replacement: char, prepend: PrependScheme, split: bool },
    Punctuation(SplitBehavior),
    Digits { individual: bool },
    Split { pattern: Pattern, behavior: SplitBehavior, invert: bool },
    CharDelimiter(char),
    Sequence(Vec<PreTokenizer>),
}

/// GPT-2's reversible byte -> printable-unicode map.
pub fn byte_to_char(b: u8) -> char {
    thread_local! {
        static TABLE: [char; 256] = {
            let mut t = ['\0'; 256];
            let mut extra = 0u32;
            for b in 0..=255u32 {
                let keep = (33..=126).contains(&b) || (161..=172).contains(&b) || (174..=255).contains(&b);
                t[b as usize] = if keep { char::from_u32(b).unwrap() } else { extra += 1; char::from_u32(255 + extra).unwrap() };
            }
            t
        };
    }
    TABLE.with(|t| t[b as usize])
}

/// Inverse of [`byte_to_char`].
pub fn char_to_byte(c: char) -> Option<u8> {
    thread_local! {
        static INV: std::collections::HashMap<char, u8> = (0..=255u8).map(|b| (byte_to_char(b), b)).collect();
    }
    INV.with(|m| m.get(&c).copied())
}

const GPT2_PATTERN: &str = r"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+";

thread_local! {
    static GPT2_RE: Regex = Regex::new(GPT2_PATTERN).unwrap();
    static WS_RE: Regex = Regex::new(r"\w+|[^\w\s]+").unwrap();
}

/// Splits a normalized string into segments `(start, end, is_match)` covering all of it, then groups them by `behavior`.
fn split_by(s: &NormalizedString, matches: Vec<(usize, usize)>, invert: bool, behavior: SplitBehavior) -> Vec<NormalizedString> {
    let n = s.chars.len();
    if n == 0 {
        return vec![];
    }
    let mut segs: Vec<((usize, usize), bool)> = Vec::new();
    let mut prev = 0;
    if matches.is_empty() {
        segs.push(((0, n), false));
    }
    for &(a, b) in &matches {
        if prev != a {
            segs.push(((prev, a), false));
        }
        segs.push(((a, b), true));
        prev = b;
    }
    if !matches.is_empty() && prev != n {
        segs.push(((prev, n), false));
    }
    if invert {
        for sg in &mut segs {
            sg.1 = !sg.1;
        }
    }
    let ranges: Vec<((usize, usize), bool)> = match behavior {
        SplitBehavior::Isolated => segs.into_iter().map(|(o, _)| (o, false)).collect(),
        SplitBehavior::Removed => segs,
        SplitBehavior::Contiguous => {
            let mut prev_match = false;
            let mut acc: Vec<((usize, usize), bool)> = vec![];
            for (o, m) in segs {
                if m == prev_match {
                    if let Some(l) = acc.last_mut() {
                        l.0.1 = o.1;
                    } else {
                        acc.push((o, false));
                    }
                } else {
                    acc.push((o, false));
                }
                prev_match = m;
            }
            acc
        }
        SplitBehavior::MergedWithPrevious => {
            let mut prev_match = false;
            let mut acc: Vec<((usize, usize), bool)> = vec![];
            for (o, m) in segs {
                if m && !prev_match {
                    if let Some(l) = acc.last_mut() {
                        l.0.1 = o.1;
                    } else {
                        acc.push((o, false));
                    }
                } else {
                    acc.push((o, false));
                }
                prev_match = m;
            }
            acc
        }
        SplitBehavior::MergedWithNext => {
            let mut prev_match = false;
            let mut acc: Vec<((usize, usize), bool)> = vec![];
            for (o, m) in segs.into_iter().rev() {
                if m && !prev_match {
                    if let Some(l) = acc.last_mut() {
                        l.0.0 = o.0;
                    } else {
                        acc.push((o, false));
                    }
                } else {
                    acc.push((o, false));
                }
                prev_match = m;
            }
            acc.reverse();
            acc
        }
    };
    ranges.into_iter().filter(|&(_, removed)| !removed).map(|((a, b), _)| NormalizedString { chars: s.chars[a..b].to_vec() }).collect()
}

fn split_pred(s: &NormalizedString, pred: impl Fn(char) -> bool, behavior: SplitBehavior) -> Vec<NormalizedString> {
    // runs of matching characters are separate single-char matches, as in HF's `char` pattern
    let matches: Vec<(usize, usize)> = s.chars.iter().enumerate().filter(|&(_, &(c, _))| pred(c)).map(|(i, _)| (i, i + 1)).collect();
    split_by(s, matches, false, behavior)
}

pub fn is_punc(c: char) -> bool {
    c.is_ascii_punctuation() || unicode::is_punctuation(c)
}

impl PreTokenizer {
    pub fn from_json(j: &Json) -> Result<PreTokenizer> {
        let ty = j.get("type").and_then(Json::as_str).ok_or_else(|| Error::Msg("tokenizer: pre_tokenizer without type".into()))?;
        let flag = |k: &str, d: bool| j.get(k).and_then(Json::as_bool).unwrap_or(d);
        Ok(match ty {
            "BertPreTokenizer" => PreTokenizer::Bert,
            "ByteLevel" => PreTokenizer::ByteLevel { add_prefix_space: flag("add_prefix_space", true), use_regex: flag("use_regex", true) },
            "Whitespace" => PreTokenizer::Whitespace,
            "WhitespaceSplit" => PreTokenizer::WhitespaceSplit,
            "Metaspace" => {
                let replacement = j.get("replacement").and_then(Json::as_str).and_then(|s| s.chars().next()).unwrap_or('\u{2581}');
                let prepend = match j.get("prepend_scheme").and_then(Json::as_str) {
                    Some("first") => PrependScheme::First,
                    Some("never") => PrependScheme::Never,
                    Some(_) => PrependScheme::Always,
                    // older files: add_prefix_space true/false
                    None => if flag("add_prefix_space", true) { PrependScheme::Always } else { PrependScheme::Never },
                };
                PreTokenizer::Metaspace { replacement, prepend, split: flag("split", true) }
            }
            "Punctuation" => PreTokenizer::Punctuation(SplitBehavior::from_json(j.get("behavior"), SplitBehavior::Isolated)?),
            "Digits" => PreTokenizer::Digits { individual: flag("individual_digits", false) },
            "Split" => PreTokenizer::Split {
                pattern: Pattern::from_json(j.get("pattern").ok_or_else(|| Error::Msg("tokenizer: Split needs pattern".into()))?)?,
                behavior: SplitBehavior::from_json(j.get("behavior"), SplitBehavior::Isolated)?,
                invert: flag("invert", false),
            },
            "CharDelimiterSplit" => PreTokenizer::CharDelimiter(j.get("delimiter").and_then(Json::as_str).and_then(|s| s.chars().next()).unwrap_or(' ')),
            "Sequence" => PreTokenizer::Sequence(
                j.get("pretokenizers").and_then(Json::as_array).ok_or_else(|| Error::Msg("tokenizer: Sequence needs pretokenizers".into()))?.iter().map(PreTokenizer::from_json).collect::<Result<_>>()?,
            ),
            other => return Err(Error::Msg(format!("tokenizer: unsupported pre-tokenizer {other}"))),
        })
    }

    /// Splits every incoming piece further.
    pub fn pre_tokenize(&self, pieces: Vec<NormalizedString>) -> Vec<NormalizedString> {
        match self {
            PreTokenizer::Sequence(v) => v.iter().fold(pieces, |p, t| t.pre_tokenize(p)),
            _ => pieces.into_iter().enumerate().flat_map(|(idx, p)| self.split_one(p, idx == 0)).collect(),
        }
    }

    fn split_one(&self, mut s: NormalizedString, first: bool) -> Vec<NormalizedString> {
        match self {
            PreTokenizer::Bert => {
                let ws = split_pred(&s, char::is_whitespace, SplitBehavior::Removed);
                ws.into_iter().flat_map(|w| split_pred(&w, is_punc, SplitBehavior::Isolated)).collect()
            }
            PreTokenizer::Whitespace => {
                let plain = s.plain();
                let m = WS_RE.with(|r| r.find_all(&plain));
                split_by(&s, m, true, SplitBehavior::Removed)
            }
            PreTokenizer::WhitespaceSplit => split_pred(&s, char::is_whitespace, SplitBehavior::Removed),
            PreTokenizer::Punctuation(b) => split_pred(&s, is_punc, *b),
            PreTokenizer::Digits { individual } => split_pred(&s, char::is_numeric, if *individual { SplitBehavior::Isolated } else { SplitBehavior::Contiguous }),
            PreTokenizer::CharDelimiter(d) => split_pred(&s, |c| c == *d, SplitBehavior::Removed),
            PreTokenizer::Split { pattern, behavior, invert } => {
                let m = pattern.find_all(&s.plain());
                split_by(&s, m, *invert, *behavior)
            }
            PreTokenizer::Metaspace { replacement, prepend, split } => {
                for (c, _) in s.chars.iter_mut() {
                    if *c == ' ' {
                        *c = *replacement;
                    }
                }
                let wants = match prepend {
                    PrependScheme::Always => true,
                    PrependScheme::First => first,
                    PrependScheme::Never => false,
                };
                if wants && s.chars.first().is_none_or(|&(c, _)| c != *replacement) && !s.is_empty() {
                    s.prepend(&replacement.to_string());
                }
                if *split {
                    let m: Vec<(usize, usize)> = s.chars.iter().enumerate().filter(|&(_, &(c, _))| c == *replacement).map(|(i, _)| (i, i + 1)).collect();
                    split_by(&s, m, false, SplitBehavior::MergedWithNext)
                } else if s.is_empty() {
                    vec![]
                } else {
                    vec![s]
                }
            }
            PreTokenizer::ByteLevel { add_prefix_space, use_regex } => {
                if *add_prefix_space && s.chars.first().is_none_or(|&(c, _)| c != ' ') && !s.is_empty() {
                    s.prepend(" ");
                }
                let pieces = if *use_regex {
                    let plain = s.plain();
                    let m = GPT2_RE.with(|r| r.find_all(&plain));
                    split_by(&s, m, false, SplitBehavior::Isolated)
                } else if s.is_empty() {
                    vec![]
                } else {
                    vec![s]
                };
                pieces
                    .into_iter()
                    .map(|p| {
                        let mut out = Vec::with_capacity(p.chars.len());
                        for (c, sp) in p.chars {
                            let mut buf = [0u8; 4];
                            for &b in c.encode_utf8(&mut buf).as_bytes() {
                                out.push((byte_to_char(b), sp));
                            }
                        }
                        NormalizedString { chars: out }
                    })
                    .collect()
            }
            PreTokenizer::Sequence(_) => unreachable!(),
        }
    }
}
