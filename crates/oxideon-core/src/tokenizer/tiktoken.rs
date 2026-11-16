//! tiktoken-compatible byte-level BPE (`r50k_base`, `p50k_base`, `cl100k_base`, `o200k_base`, or any `.tiktoken` rank
//! file): text is split with the encoding's regular expression, every piece is looked up in the rank table or merged
//! pair by pair (lowest rank first), and special tokens are recognised on request.

use super::regex::Regex;
use crate::error::{Error, Result};
use std::collections::{BinaryHeap, HashMap};

/// Which special tokens may appear in the text given to [`Tiktoken::encode`].
#[derive(Debug, Clone)]
pub enum Special<'a> {
    /// No special token is allowed: the text is encoded as plain text, but any special token string in it is an error
    /// (tiktoken's default).
    Disallow,
    /// Special token strings found in the text become their token ids.
    All,
    /// Only these special token strings become ids; any other special token string is an error.
    Only(&'a [&'a str]),
    /// Special token strings are not recognised and encoded like any other text (`disallowed_special=()`).
    Ignore,
}

pub const R50K_PATTERN: &str = r"'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+";
pub const CL100K_PATTERN: &str = r"'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}++|\p{N}{1,3}+| ?[^\s\p{L}\p{N}]++[\r\n]*+|\s++$|\s*[\r\n]|\s+(?!\S)|\s";
pub const O200K_PATTERN: &str = concat!(
    r"[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?",
    r"|[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?",
    r"|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n/]*|\s*[\r\n]+|\s+(?!\S)|\s+"
);

pub struct Tiktoken {
    ranks: HashMap<Vec<u8>, u32>,
    decoder: HashMap<u32, Vec<u8>>,
    special: HashMap<String, u32>,
    special_decoder: HashMap<u32, String>,
    pat: Regex,
}

fn b64(s: &str) -> Result<Vec<u8>> {
    super::hf::base64_decode(s)
}

impl Tiktoken {
    /// Builds an encoding from the text of a `.tiktoken` file (`<base64 token> <rank>` per line).
    pub fn from_tiktoken_text(text: &str, pattern: &str, special_tokens: &[(&str, u32)]) -> Result<Self> {
        let mut ranks = HashMap::new();
        for line in text.lines() {
            if line.trim().is_empty() {
                continue;
            }
            let (tok, rank) = line.split_once(' ').ok_or_else(|| Error::Msg("tiktoken: bad rank line".into()))?;
            let rank: u32 = rank.trim().parse().map_err(|_| Error::Msg("tiktoken: bad rank".into()))?;
            ranks.insert(b64(tok)?, rank);
        }
        Self::new(ranks, pattern, special_tokens)
    }

    pub fn from_tiktoken_file(path: impl AsRef<std::path::Path>, pattern: &str, special_tokens: &[(&str, u32)]) -> Result<Self> {
        let text = std::fs::read_to_string(path.as_ref()).map_err(|e| Error::Msg(format!("tiktoken: cannot read {}: {e}", path.as_ref().display())))?;
        Self::from_tiktoken_text(&text, pattern, special_tokens)
    }

    pub fn new(ranks: HashMap<Vec<u8>, u32>, pattern: &str, special_tokens: &[(&str, u32)]) -> Result<Self> {
        let decoder = ranks.iter().map(|(k, &v)| (v, k.clone())).collect();
        let special: HashMap<String, u32> = special_tokens.iter().map(|&(s, i)| (s.to_string(), i)).collect();
        let special_decoder = special.iter().map(|(k, &v)| (v, k.clone())).collect();
        Ok(Self { ranks, decoder, special, special_decoder, pat: Regex::new(pattern)? })
    }

    /// The named OpenAI encodings with their patterns and special tokens; `ranks_text` is the content of the matching
    /// `.tiktoken` file (`gpt2` / `r50k_base`, `p50k_base`, `cl100k_base`, `o200k_base`).
    pub fn preset(name: &str, ranks_text: &str) -> Result<Self> {
        match name {
            "gpt2" | "r50k_base" | "p50k_base" => Self::from_tiktoken_text(ranks_text, R50K_PATTERN, &[("<|endoftext|>", 50256)]),
            "cl100k_base" => Self::from_tiktoken_text(
                ranks_text,
                CL100K_PATTERN,
                &[("<|endoftext|>", 100257), ("<|fim_prefix|>", 100258), ("<|fim_middle|>", 100259), ("<|fim_suffix|>", 100260), ("<|endofprompt|>", 100276)],
            ),
            "o200k_base" => Self::from_tiktoken_text(ranks_text, O200K_PATTERN, &[("<|endoftext|>", 199999), ("<|endofprompt|>", 200018)]),
            other => Err(Error::Msg(format!("tiktoken: unknown encoding {other}"))),
        }
    }

    /// Number of distinct token ids (the largest id + 1, including special tokens).
    pub fn n_vocab(&self) -> usize {
        let m = self.decoder.keys().chain(self.special_decoder.keys()).max().copied().unwrap_or(0);
        m as usize + 1
    }

    pub fn special_tokens(&self) -> &HashMap<String, u32> {
        &self.special
    }

    /// Splits `piece` into tokens by repeatedly merging the lowest-rank adjacent pair (leftmost on ties).
    fn bpe(&self, piece: &[u8], out: &mut Vec<u32>) -> Result<()> {
        if piece.is_empty() {
            return Ok(());
        }
        if let Some(&r) = self.ranks.get(piece) {
            out.push(r);
            return Ok(());
        }
        let n = piece.len();
        if n == 1 {
            return Err(Error::Msg("tiktoken: byte missing from the rank table".into()));
        }
        // doubly linked list of parts; heap of candidate merges (rank, start) with lazy invalidation
        let start: Vec<usize> = (0..=n).collect();
        let mut next: Vec<usize> = (1..=n + 1).collect();
        let mut prev: Vec<usize> = (0..=n).map(|i| i.wrapping_sub(1)).collect();
        let mut alive = vec![true; n + 1];
        let pair_rank = |s: &[usize], nx: &[usize], i: usize| -> Option<u32> {
            let j = nx[i];
            if j >= n {
                return None;
            }
            let k = nx[j];
            let end = if k <= n { s[k] } else { n };
            self.ranks.get(&piece[s[i]..end]).copied()
        };
        let mut heap: BinaryHeap<std::cmp::Reverse<(u32, usize)>> = BinaryHeap::new();
        for i in 0..n.saturating_sub(1) {
            if let Some(r) = pair_rank(&start, &next, i) {
                heap.push(std::cmp::Reverse((r, i)));
            }
        }
        while let Some(std::cmp::Reverse((r, i))) = heap.pop() {
            if !alive[i] || pair_rank(&start, &next, i) != Some(r) {
                continue;
            }
            // merge part i with its successor j
            let j = next[i];
            alive[j] = false;
            next[i] = next[j];
            if next[j] <= n {
                prev[next[j]] = i;
            }
            if let Some(r2) = pair_rank(&start, &next, i) {
                heap.push(std::cmp::Reverse((r2, i)));
            }
            let p = prev[i];
            if i > 0 && p < n && alive[p] {
                if let Some(r3) = pair_rank(&start, &next, p) {
                    heap.push(std::cmp::Reverse((r3, p)));
                }
            }
        }
        let mut i = 0;
        while i < n {
            let j = next[i].min(n);
            let end = if j >= n { n } else { start[j] };
            let tok = &piece[start[i]..end];
            out.push(*self.ranks.get(tok).ok_or_else(|| Error::Msg("tiktoken: unmergeable byte sequence".into()))?);
            i = j;
        }
        Ok(())
    }

    /// Encodes `text` as plain text (special-token strings are not recognised).
    pub fn encode_ordinary(&self, text: &str) -> Result<Vec<u32>> {
        let chars: Vec<char> = text.chars().collect();
        let mut out = Vec::new();
        for (a, b) in self.pat.find_all(&chars) {
            let piece: String = chars[a..b].iter().collect();
            self.bpe(piece.as_bytes(), &mut out)?;
        }
        Ok(out)
    }

    /// Encodes `text`, handling special tokens as `special` says.
    pub fn encode(&self, text: &str, special: Special<'_>) -> Result<Vec<u32>> {
        let allowed: Vec<&str> = match &special {
            Special::All => self.special.keys().map(String::as_str).collect(),
            Special::Only(v) => v.to_vec(),
            Special::Disallow | Special::Ignore => vec![],
        };
        if let Special::Disallow | Special::Only(_) = &special {
            if let Some(bad) = self.special.keys().find(|k| !allowed.contains(&k.as_str()) && text.contains(k.as_str())) {
                return Err(Error::Msg(format!("tiktoken: the text contains the special token {bad:?}, which is not allowed here")));
            }
        }
        let mut out = Vec::new();
        let mut rest = text;
        loop {
            // leftmost allowed special token (longest first on a shared start)
            let hit = allowed.iter().filter_map(|s| rest.find(s).map(|i| (i, s.len(), *s))).min_by_key(|&(i, len, _)| (i, std::cmp::Reverse(len)));
            match hit {
                Some((i, len, s)) => {
                    out.extend(self.encode_ordinary(&rest[..i])?);
                    out.push(*self.special.get(s).ok_or_else(|| Error::Msg("tiktoken: unknown special token".into()))?);
                    rest = &rest[i + len..];
                }
                None => {
                    out.extend(self.encode_ordinary(rest)?);
                    return Ok(out);
                }
            }
        }
    }

    pub fn encode_batch(&self, texts: &[&str], special: Special<'_>) -> Result<Vec<Vec<u32>>> {
        texts.iter().map(|t| self.encode(t, special.clone())).collect()
    }

    /// The raw bytes of one token (ordinary or special).
    pub fn token_bytes(&self, id: u32) -> Option<Vec<u8>> {
        self.decoder.get(&id).cloned().or_else(|| self.special_decoder.get(&id).map(|s| s.as_bytes().to_vec()))
    }

    pub fn decode_bytes(&self, ids: &[u32]) -> Result<Vec<u8>> {
        let mut out = Vec::new();
        for &id in ids {
            out.extend(self.token_bytes(id).ok_or_else(|| Error::Msg(format!("tiktoken: invalid token id {id}")))?);
        }
        Ok(out)
    }

    /// Decodes ids to text, replacing invalid UTF-8 with U+FFFD.
    pub fn decode(&self, ids: &[u32]) -> Result<String> {
        Ok(String::from_utf8_lossy(&self.decode_bytes(ids)?).into_owned())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tiny() -> Tiktoken {
        // bytes of "abc " plus a few merges
        let mut ranks: HashMap<Vec<u8>, u32> = (0u32..256).map(|b| (vec![b as u8], b)).collect();
        for (i, m) in ["ab", "bc", "abc", " a", " ab"].iter().enumerate() {
            ranks.insert(m.as_bytes().to_vec(), 256 + i as u32);
        }
        Tiktoken::new(ranks, R50K_PATTERN, &[("<|endoftext|>", 1000)]).unwrap()
    }

    #[test]
    fn merges_lowest_rank_first() {
        let t = tiny();
        // "abc": ab(256) and bc(257) both apply; ab has the lower rank, giving [ab, c]... unless "abc"(258) is whole
        assert_eq!(t.encode_ordinary("abc").unwrap(), [258]);
        assert_eq!(t.encode_ordinary("abcab").unwrap(), [258, 256]);
        assert_eq!(t.encode_ordinary(" ab").unwrap(), [260]);
        assert_eq!(t.decode(&t.encode_ordinary("héllo wörld 😀").unwrap()).unwrap(), "héllo wörld 😀");
    }

    #[test]
    fn special_tokens() {
        let t = tiny();
        assert!(t.encode("a<|endoftext|>b", Special::Disallow).is_err());
        assert_eq!(t.encode("a<|endoftext|>b", Special::All).unwrap(), [97, 1000, 98]);
        assert_eq!(t.encode("<|endoftext|>", Special::Ignore).unwrap().len(), 13);
        assert_eq!(t.decode(&[97, 1000]).unwrap(), "a<|endoftext|>");
    }
}
