//! Decoders: turn a list of token strings back into text (Hugging Face `tokenizers` semantics).

use super::normalizer::Pattern;
use super::pretokenizer::char_to_byte;
use crate::error::{Error, Result};
use crate::tokenizer::json::Json;

#[derive(Debug, Clone)]
pub enum Decoder {
    ByteLevel,
    WordPiece { prefix: String, cleanup: bool },
    Metaspace { replacement: char, add_prefix_space: bool },
    Bpe { suffix: String },
    ByteFallback,
    Fuse,
    Strip { content: char, start: usize, stop: usize },
    Replace { pattern: Pattern, content: String },
    Ctc { pad_token: String, word_delimiter_token: String, cleanup: bool },
    Sequence(Vec<Decoder>),
}

fn wordpiece_cleanup(s: &str) -> String {
    s.replace(" .", ".").replace(" ?", "?").replace(" !", "!").replace(" ,", ",").replace(" ' ", "'").replace(" n't", "n't").replace(" 'm", "'m").replace(" do not", " don't").replace(" 's", "'s").replace(" 've", "'ve").replace(" 're", "'re")
}

impl Decoder {
    pub fn from_json(j: &Json) -> Result<Decoder> {
        let ty = j.get("type").and_then(Json::as_str).ok_or_else(|| Error::Msg("tokenizer: decoder without type".into()))?;
        let st = |k: &str, d: &str| j.get(k).and_then(Json::as_str).unwrap_or(d).to_string();
        let flag = |k: &str, d: bool| j.get(k).and_then(Json::as_bool).unwrap_or(d);
        Ok(match ty {
            "ByteLevel" => Decoder::ByteLevel,
            "WordPiece" => Decoder::WordPiece { prefix: st("prefix", "##"), cleanup: flag("cleanup", true) },
            "Metaspace" => Decoder::Metaspace {
                replacement: st("replacement", "\u{2581}").chars().next().unwrap_or('\u{2581}'),
                add_prefix_space: match j.get("prepend_scheme").and_then(Json::as_str) {
                    Some("never") => false,
                    Some(_) => true,
                    None => flag("add_prefix_space", true),
                },
            },
            "BPEDecoder" => Decoder::Bpe { suffix: st("suffix", "</w>") },
            "ByteFallback" => Decoder::ByteFallback,
            "Fuse" => Decoder::Fuse,
            "Strip" => Decoder::Strip {
                content: st("content", " ").chars().next().unwrap_or(' '),
                start: j.get("start").and_then(Json::as_u64).unwrap_or(0) as usize,
                stop: j.get("stop").and_then(Json::as_u64).unwrap_or(0) as usize,
            },
            "Replace" => Decoder::Replace {
                pattern: Pattern::from_json(j.get("pattern").ok_or_else(|| Error::Msg("tokenizer: Replace decoder needs pattern".into()))?)?,
                content: st("content", ""),
            },
            "CTC" => Decoder::Ctc { pad_token: st("pad_token", "<pad>"), word_delimiter_token: st("word_delimiter_token", "|"), cleanup: flag("cleanup", true) },
            "Sequence" => Decoder::Sequence(
                j.get("decoders").and_then(Json::as_array).ok_or_else(|| Error::Msg("tokenizer: Sequence needs decoders".into()))?.iter().map(Decoder::from_json).collect::<Result<_>>()?,
            ),
            other => return Err(Error::Msg(format!("tokenizer: unsupported decoder {other}"))),
        })
    }

    pub fn decode_chain(&self, tokens: Vec<String>) -> Vec<String> {
        match self {
            Decoder::ByteLevel => {
                // all tokens are joined before the UTF-8 decoding, so multi-byte characters may span tokens
                let mut bytes = Vec::new();
                for c in tokens.iter().flat_map(|t| t.chars()) {
                    match char_to_byte(c) {
                        Some(b) => bytes.push(b),
                        None => {
                            let mut buf = [0u8; 4];
                            bytes.extend_from_slice(c.encode_utf8(&mut buf).as_bytes());
                        }
                    }
                }
                vec![String::from_utf8_lossy(&bytes).into_owned()]
            }
            Decoder::WordPiece { prefix, cleanup } => tokens
                .into_iter()
                .enumerate()
                .map(|(i, t)| {
                    let t = if i != 0 {
                        match t.strip_prefix(prefix.as_str()) {
                            Some(r) => r.to_string(),
                            None => format!(" {t}"),
                        }
                    } else {
                        t
                    };
                    if *cleanup { wordpiece_cleanup(&t) } else { t }
                })
                .collect(),
            Decoder::Metaspace { replacement, add_prefix_space } => tokens
                .into_iter()
                .enumerate()
                .map(|(i, t)| {
                    let mut s: String = t.chars().map(|c| if c == *replacement { ' ' } else { c }).collect();
                    if i == 0 && *add_prefix_space && s.starts_with(' ') {
                        s.remove(0);
                    }
                    s
                })
                .collect(),
            Decoder::Bpe { suffix } => {
                let last = tokens.len().saturating_sub(1);
                tokens.into_iter().enumerate().map(|(i, t)| t.replace(suffix.as_str(), if i == last { "" } else { " " })).collect()
            }
            Decoder::ByteFallback => {
                let mut out: Vec<String> = Vec::new();
                let mut run: Vec<u8> = Vec::new();
                let flush = |run: &mut Vec<u8>, out: &mut Vec<String>| {
                    if run.is_empty() {
                        return;
                    }
                    match String::from_utf8(run.clone()) {
                        Ok(s) => out.push(s),
                        Err(_) => out.extend(run.iter().map(|_| "\u{fffd}".to_string())),
                    }
                    run.clear();
                };
                for t in tokens {
                    let byte = t.strip_prefix("<0x").and_then(|r| r.strip_suffix('>')).filter(|h| h.len() == 2).and_then(|h| u8::from_str_radix(h, 16).ok());
                    match byte {
                        Some(b) => run.push(b),
                        None => {
                            flush(&mut run, &mut out);
                            out.push(t);
                        }
                    }
                }
                flush(&mut run, &mut out);
                out
            }
            Decoder::Fuse => vec![tokens.concat()],
            Decoder::Strip { content, start, stop } => tokens
                .into_iter()
                .map(|t| {
                    let chars: Vec<char> = t.chars().collect();
                    let n = chars.len();
                    let mut a = 0;
                    for i in 0..*start {
                        if i < n && chars[i] == *content { a = i + 1 } else { break }
                    }
                    let mut b = n;
                    for i in 0..*stop {
                        if i < n && chars[n - i - 1] == *content { b = n - i - 1 } else { break }
                    }
                    if a >= b { String::new() } else { chars[a..b].iter().collect() }
                })
                .collect(),
            Decoder::Replace { pattern, content } => tokens
                .into_iter()
                .map(|t| {
                    let mut ns = super::normalizer::NormalizedString::new(&t);
                    ns.replace(pattern, content);
                    ns.get()
                })
                .collect(),
            Decoder::Ctc { pad_token, word_delimiter_token, cleanup } => {
                let mut v: Vec<String> = Vec::new();
                for t in tokens {
                    if v.last() != Some(&t) {
                        v.push(t);
                    }
                }
                v.into_iter()
                    .filter(|t| t != pad_token)
                    .map(|t| {
                        let t = t.replace(word_delimiter_token.as_str(), " ");
                        if *cleanup { wordpiece_cleanup(&t).trim().to_string() } else { t }
                    })
                    .collect()
            }
            Decoder::Sequence(v) => v.iter().fold(tokens, |t, d| d.decode_chain(t)),
        }
    }

    pub fn decode(&self, tokens: Vec<String>) -> String {
        self.decode_chain(tokens).concat()
    }
}
