//! SentencePiece's `precompiled_charsmap`: a Darts double-array trie mapping source byte sequences to replacement
//! strings (the NFKC-based `nmt_nfkc` rules of T5, ALBERT, XLNet, XLM-R, mBART, ...).
//!
//! Layout: `u32` trie size in bytes, the trie as little-endian `u32` units, then the pool of NUL-terminated
//! replacement strings (trie values are byte offsets into the pool).

use super::normalizer::NormalizedString;
use crate::error::{Error, Result};

#[derive(Debug, Clone)]
pub struct CharsMap {
    units: Vec<u32>,
    pool: Vec<u8>,
}

impl CharsMap {
    pub fn from_bytes(b: &[u8]) -> Result<Self> {
        if b.len() < 4 {
            return Err(Error::Msg("charsmap: too short".into()));
        }
        let trie_bytes = u32::from_le_bytes([b[0], b[1], b[2], b[3]]) as usize;
        if trie_bytes % 4 != 0 || 4 + trie_bytes > b.len() {
            return Err(Error::Msg("charsmap: bad trie size".into()));
        }
        let units = b[4..4 + trie_bytes].chunks_exact(4).map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
        Ok(Self { units, pool: b[4 + trie_bytes..].to_vec() })
    }

    /// Values of all trie keys that are prefixes of `key`, shortest first.
    fn prefix_values(&self, key: &[u8]) -> Vec<u32> {
        let mut out = Vec::new();
        let offset = |u: u32| (u >> 10) << ((u & 0x200) >> 6);
        let Some(&root) = self.units.first() else { return out };
        let mut pos = offset(root) as usize;
        for &byte in key {
            if byte == 0 {
                break;
            }
            pos ^= byte as usize;
            let Some(&unit) = self.units.get(pos) else { break };
            if unit & 0x8000_00ff != byte as u32 {
                break;
            }
            pos ^= offset(unit) as usize;
            if (unit >> 8) & 1 == 1 {
                if let Some(&leaf) = self.units.get(pos) {
                    out.push(leaf & 0x7fff_ffff);
                }
            }
        }
        out
    }

    fn replacement(&self, index: u32) -> Option<&str> {
        let start = index as usize;
        let end = start + self.pool.get(start..)?.iter().position(|&b| b == 0)?;
        std::str::from_utf8(&self.pool[start..end]).ok()
    }

    /// Longest-match rewriting, as SentencePiece's normalizer does.
    pub fn normalize(&self, s: &mut NormalizedString) {
        let src = std::mem::take(&mut s.chars);
        let text: String = src.iter().map(|&(c, _)| c).collect();
        let bytes = text.as_bytes();
        // byte offset of each char start
        let mut starts = Vec::with_capacity(src.len() + 1);
        let mut acc = 0;
        for &(c, _) in &src {
            starts.push(acc);
            acc += c.len_utf8();
        }
        starts.push(acc);
        let mut i = 0;
        while i < src.len() {
            let vals = self.prefix_values(&bytes[starts[i]..]);
            // the longest trie match wins (values are listed shortest first)
            let chosen = vals.last().copied();
            match chosen {
                Some(v) => {
                    let matched_len = self.match_len(&bytes[starts[i]..]);
                    let mut j = i;
                    while j < src.len() && starts[j] - starts[i] < matched_len {
                        j += 1;
                    }
                    // replacement characters keep the span of the source character at the same position (the last
                    // source character when the replacement is longer), as the reference does
                    match self.replacement(v) {
                        Some(r) => s.chars.extend(r.chars().enumerate().map(|(k, c)| (c, src[(i + k).min(j - 1)].1))),
                        None => s.chars.extend_from_slice(&src[i..j]),
                    }
                    i = j.max(i + 1);
                }
                None => {
                    s.chars.push(src[i]);
                    i += 1;
                }
            }
        }
    }

    /// Byte length of the longest key that is a prefix of `key`.
    fn match_len(&self, key: &[u8]) -> usize {
        let offset = |u: u32| (u >> 10) << ((u & 0x200) >> 6);
        let Some(&root) = self.units.first() else { return 0 };
        let mut pos = offset(root) as usize;
        let mut best = 0;
        for (k, &byte) in key.iter().enumerate() {
            if byte == 0 {
                break;
            }
            pos ^= byte as usize;
            let Some(&unit) = self.units.get(pos) else { break };
            if unit & 0x8000_00ff != byte as u32 {
                break;
            }
            pos ^= offset(unit) as usize;
            if (unit >> 8) & 1 == 1 {
                best = k + 1;
            }
        }
        best
    }
}
