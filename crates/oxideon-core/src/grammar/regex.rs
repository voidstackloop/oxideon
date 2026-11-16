//! Regex-constrained decoding: only tokens that keep the generated text a prefix of some string matching the pattern are
//! allowed, and the end-of-sequence token is allowed only once the text matches in full.
//!
//! The matcher is a Thompson NFA simulated state-set by state-set (no backtracking, linear time in the text). Supported
//! syntax: literals, `.`, `[...]` classes with ranges and negation, `\d \D \w \W \s \S` (ASCII, like Python's `re.ASCII`),
//! escapes `\n \t \r` and escaped punctuation, groups `( )` / `(?: )`, alternation `|`, and the quantifiers
//! `* + ? {m} {m,} {m,n}`. Anchors and lazy quantifiers are rejected (matching is always over the whole text).
//!
//! ```
//! use oxideon_core::grammar::regex::Regex;
//! let r = Regex::new(r"-?\d+(\.\d+)?").unwrap();
//! assert!(r.is_match("-12.5") && !r.is_match("12."));
//! ```

use crate::error::{Error, Result};
use crate::tensor::Tensor;
use std::collections::{BTreeSet, HashMap};

#[derive(Clone, Debug)]
struct CharSet {
    ranges: Vec<(u32, u32)>,
    neg: bool,
}

impl CharSet {
    fn contains(&self, c: char) -> bool {
        let c = c as u32;
        self.ranges.iter().any(|&(a, b)| a <= c && c <= b) != self.neg
    }
}

#[derive(Clone, Debug)]
enum Ast {
    Empty,
    Set(CharSet),
    Concat(Box<Ast>, Box<Ast>),
    Alt(Box<Ast>, Box<Ast>),
    Star(Box<Ast>),
}

#[derive(Clone, Debug)]
enum State {
    Set(CharSet, usize),
    Split(usize, usize),
    Match,
}

fn err(m: impl Into<String>) -> Error {
    Error::Msg(format!("regex: {}", m.into()))
}

struct Parser<'a> {
    s: std::iter::Peekable<std::str::Chars<'a>>,
}

fn class_escape(c: char) -> Option<CharSet> {
    let (ranges, neg) = match c {
        'd' => (vec![(48, 57)], false),
        'D' => (vec![(48, 57)], true),
        'w' => (vec![(48, 57), (65, 90), (95, 95), (97, 122)], false),
        'W' => (vec![(48, 57), (65, 90), (95, 95), (97, 122)], true),
        's' => (vec![(9, 13), (32, 32)], false),
        'S' => (vec![(9, 13), (32, 32)], true),
        _ => return None,
    };
    Some(CharSet { ranges, neg })
}

fn literal_escape(c: char) -> char {
    match c {
        'n' => '\n',
        't' => '\t',
        'r' => '\r',
        other => other,
    }
}

impl Parser<'_> {
    fn alt(&mut self) -> Result<Ast> {
        let mut left = self.concat()?;
        while self.s.peek() == Some(&'|') {
            self.s.next();
            left = Ast::Alt(Box::new(left), Box::new(self.concat()?));
        }
        Ok(left)
    }

    fn concat(&mut self) -> Result<Ast> {
        let mut acc = Ast::Empty;
        while let Some(&c) = self.s.peek() {
            if c == '|' || c == ')' {
                break;
            }
            let piece = self.postfix()?;
            acc = Ast::Concat(Box::new(acc), Box::new(piece));
        }
        Ok(acc)
    }

    fn number(&mut self) -> Option<usize> {
        let mut n = None;
        while let Some(d) = self.s.peek().and_then(|c| c.to_digit(10)) {
            n = Some(n.unwrap_or(0) * 10 + d as usize);
            self.s.next();
        }
        n
    }

    fn postfix(&mut self) -> Result<Ast> {
        let mut atom = self.atom()?;
        loop {
            let (min, max) = match self.s.peek() {
                Some('*') => (0, None),
                Some('+') => (1, None),
                Some('?') => (0, Some(1)),
                Some('{') => {
                    self.s.next();
                    let min = self.number().ok_or_else(|| err("bad {m,n} quantifier"))?;
                    let max = if self.s.peek() == Some(&',') {
                        self.s.next();
                        self.number()
                    } else {
                        Some(min)
                    };
                    if self.s.peek() != Some(&'}') {
                        return Err(err("unterminated {m,n} quantifier"));
                    }
                    if min > 1000 || max.is_some_and(|m| m < min || m > 1000) {
                        return Err(err("quantifier bounds must satisfy m <= n <= 1000"));
                    }
                    (min, max)
                }
                _ => return Ok(atom),
            };
            self.s.next();
            if matches!(self.s.peek(), Some('?') | Some('+')) {
                return Err(err("lazy / possessive quantifiers are not supported"));
            }
            atom = repeat(atom, min, max);
        }
    }

    fn atom(&mut self) -> Result<Ast> {
        let c = self.s.next().ok_or_else(|| err("unexpected end of pattern"))?;
        Ok(match c {
            '(' => {
                if self.s.peek() == Some(&'?') {
                    self.s.next();
                    if self.s.next() != Some(':') {
                        return Err(err("only (?:...) groups are supported"));
                    }
                }
                let inner = self.alt()?;
                if self.s.next() != Some(')') {
                    return Err(err("missing )"));
                }
                inner
            }
            '.' => Ast::Set(CharSet { ranges: vec![(10, 10)], neg: true }),
            '[' => self.class()?,
            '\\' => {
                let e = self.s.next().ok_or_else(|| err("dangling backslash"))?;
                Ast::Set(class_escape(e).unwrap_or_else(|| {
                    let l = literal_escape(e) as u32;
                    CharSet { ranges: vec![(l, l)], neg: false }
                }))
            }
            '*' | '+' | '?' | '{' => return Err(err(format!("nothing to repeat before '{c}'"))),
            '^' | '$' => return Err(err("anchors are not supported (the whole text is always matched)")),
            other => Ast::Set(CharSet { ranges: vec![(other as u32, other as u32)], neg: false }),
        })
    }

    fn class(&mut self) -> Result<Ast> {
        let neg = self.s.peek() == Some(&'^');
        if neg {
            self.s.next();
        }
        let mut ranges = Vec::new();
        let mut first = true;
        loop {
            let c = self.s.next().ok_or_else(|| err("unterminated character class"))?;
            if c == ']' && !first {
                break;
            }
            first = false;
            let lo = if c == '\\' {
                let e = self.s.next().ok_or_else(|| err("dangling backslash"))?;
                if let Some(cs) = class_escape(e) {
                    if cs.neg {
                        return Err(err("negated escapes inside [...] are not supported"));
                    }
                    ranges.extend(cs.ranges);
                    continue;
                }
                literal_escape(e)
            } else {
                c
            };
            let mut look = self.s.clone();
            if look.next() == Some('-') && look.peek().is_some_and(|&n| n != ']') {
                self.s.next();
                let mut hi = self.s.next().unwrap();
                if hi == '\\' {
                    hi = literal_escape(self.s.next().ok_or_else(|| err("dangling backslash"))?);
                }
                if (hi as u32) < (lo as u32) {
                    return Err(err("reversed range in character class"));
                }
                ranges.push((lo as u32, hi as u32));
            } else {
                ranges.push((lo as u32, lo as u32));
            }
        }
        Ok(Ast::Set(CharSet { ranges, neg }))
    }
}

fn repeat(a: Ast, min: usize, max: Option<usize>) -> Ast {
    let mut out = Ast::Empty;
    for _ in 0..min {
        out = Ast::Concat(Box::new(out), Box::new(a.clone()));
    }
    match max {
        None => Ast::Concat(Box::new(out), Box::new(Ast::Star(Box::new(a)))),
        Some(max) => {
            // (a(a(a)?)?)? for the optional tail, built inside out
            let mut tail = Ast::Empty;
            for _ in 0..max - min {
                tail = Ast::Alt(Box::new(Ast::Concat(Box::new(a.clone()), Box::new(tail))), Box::new(Ast::Empty));
            }
            Ast::Concat(Box::new(out), Box::new(tail))
        }
    }
}

fn compile(ast: &Ast, next: usize, states: &mut Vec<State>) -> usize {
    match ast {
        Ast::Empty => next,
        Ast::Set(cs) => {
            states.push(State::Set(cs.clone(), next));
            states.len() - 1
        }
        Ast::Concat(a, b) => {
            let sb = compile(b, next, states);
            compile(a, sb, states)
        }
        Ast::Alt(a, b) => {
            let sa = compile(a, next, states);
            let sb = compile(b, next, states);
            states.push(State::Split(sa, sb));
            states.len() - 1
        }
        Ast::Star(x) => {
            states.push(State::Split(0, next));
            let split = states.len() - 1;
            let body = compile(x, split, states);
            states[split] = State::Split(body, next);
            split
        }
    }
}

pub struct Regex {
    states: Vec<State>,
    start: usize,
}

/// The set of NFA states the text so far can be in.
pub type MatchState = BTreeSet<usize>;

impl Regex {
    pub fn new(pattern: &str) -> Result<Self> {
        let mut p = Parser { s: pattern.chars().peekable() };
        let ast = p.alt()?;
        if p.s.peek().is_some() {
            return Err(err("unmatched )"));
        }
        let mut states = vec![State::Match];
        let start = compile(&ast, 0, &mut states);
        Ok(Self { states, start })
    }

    fn close(&self, from: usize, into: &mut MatchState) {
        let mut seen = BTreeSet::new();
        let mut stack = vec![from];
        while let Some(s) = stack.pop() {
            if !seen.insert(s) {
                continue; // nullable loops such as (a*)* would otherwise cycle forever
            }
            if let State::Split(a, b) = self.states[s] {
                stack.extend([a, b]);
            } else {
                into.insert(s);
            }
        }
    }

    pub fn initial(&self) -> MatchState {
        let mut s = MatchState::new();
        self.close(self.start, &mut s);
        s
    }

    pub fn step(&self, from: &MatchState, c: char) -> MatchState {
        let mut out = MatchState::new();
        for &s in from {
            if let State::Set(cs, next) = &self.states[s]
                && cs.contains(c)
            {
                self.close(*next, &mut out);
            }
        }
        out
    }

    pub fn accepting(&self, st: &MatchState) -> bool {
        st.iter().any(|&s| matches!(self.states[s], State::Match))
    }

    /// Feeds `text`; an empty result means no continuation can ever match.
    pub fn feed(&self, from: &MatchState, text: &str) -> MatchState {
        let mut cur = from.clone();
        for c in text.chars() {
            cur = self.step(&cur, c);
            if cur.is_empty() {
                break;
            }
        }
        cur
    }

    /// Whether the whole of `text` matches (Python's `re.fullmatch`).
    pub fn is_match(&self, text: &str) -> bool {
        self.accepting(&self.feed(&self.initial(), text))
    }
}

/// Tracks the text generated so far and masks the logits of a vocabulary accordingly.
#[derive(Clone)]
pub struct RegexConstraint {
    re: std::sync::Arc<Regex>,
    state: MatchState,
    masks: HashMap<MatchState, Vec<bool>>,
}

impl RegexConstraint {
    pub fn new(re: std::sync::Arc<Regex>) -> Self {
        let state = re.initial();
        Self { re, state, masks: HashMap::new() }
    }

    pub fn reset(&mut self) {
        self.state = self.re.initial();
    }

    pub fn is_complete(&self) -> bool {
        self.re.accepting(&self.state)
    }

    /// `allowed[i]` is true when appending `vocab[i]` keeps a match possible (empty tokens are never allowed). Cached per
    /// matcher state, so a long generation pays for each distinct state once.
    pub fn allowed(&mut self, vocab: &[String]) -> &[bool] {
        let (re, state) = (&self.re, &self.state);
        self.masks.entry(state.clone()).or_insert_with(|| vocab.iter().map(|t| !t.is_empty() && !re.feed(state, t).is_empty()).collect())
    }

    /// Adds `-inf` to the logits (1-D, one per vocab entry) of every token that is not allowed; `eos` stays allowed only
    /// when the text already matches in full (and is the only way out then if nothing else continues the match).
    pub fn mask_logits(&mut self, logits: &Tensor, vocab: &[String], eos: Option<usize>) -> Result<Tensor> {
        let complete = self.is_complete();
        let mut m: Vec<f32> = self.allowed(vocab).iter().map(|&ok| if ok { 0.0 } else { f32::NEG_INFINITY }).collect();
        if let Some(e) = eos.filter(|&e| e < m.len()) {
            m[e] = if complete { 0.0 } else { f32::NEG_INFINITY };
        }
        if m.iter().all(|v| *v == f32::NEG_INFINITY) {
            return Err(err("no token can continue the match (the vocabulary cannot spell any continuation)"));
        }
        logits.add(&Tensor::from_slice(&m, &[m.len()])?)
    }

    /// Records that `text` was generated.
    pub fn advance(&mut self, text: &str) {
        self.state = self.re.feed(&self.state, text);
    }
}

impl std::fmt::Debug for RegexConstraint {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "RegexConstraint({} live states)", self.state.len())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    // (pattern, text, Python `re.fullmatch(pattern, text, re.ASCII) is not None`)
    const CASES: &[(&str, &str, bool)] = &[
        (r"abc", "abc", true),
        (r"abc", "ab", false),
        (r"a|bc", "bc", true),
        (r"a|bc", "abc", false),
        (r"(ab)*c", "ababc", true),
        (r"(ab)*c", "abac", false),
        (r"a{2,3}", "aa", true),
        (r"a{2,3}", "aaaa", false),
        (r"a{2,}", "aaaaa", true),
        (r"a{0}b", "b", true),
        (r"-?\d+(\.\d+)?", "-12.5", true),
        (r"-?\d+(\.\d+)?", "12.", false),
        (r"[a-c]+x?", "abcax", true),
        (r"[^0-9]+", "ab1", false),
        (r"[\d_]+", "1_2", true),
        (r"\w+@\w+\.com", "me@site.com", true),
        (r"\w+@\w+\.com", "me@site.org", false),
        (r".+", "a\nb", false),
        (r"(?:yes|no)!", "no!", true),
        (r"[a-]+", "a-a", true),
        (r"\s*x", " \t x", true),
        (r"(a|ab)(c|bcd)", "abcd", true),
        (r"", "", true),
        (r"", "a", false),
        (r"(a*)*", "aaa", true),
        (r"(a|b)*abb", "babaabb", true),
    ];

    #[test]
    fn full_match_agrees_with_pythons_re_on_a_table_of_cases() {
        for &(p, t, want) in CASES {
            assert_eq!(Regex::new(p).unwrap().is_match(t), want, "{p:?} vs {t:?}");
        }
        for bad in ["(", "a)", "*a", "[a", "a{2", "a{3,2}", "^a", "a*?", "(?=a)", "\\"] {
            assert!(Regex::new(bad).is_err(), "{bad:?} should be rejected");
        }
    }

    #[test]
    fn constrained_generation_always_produces_a_full_match() {
        // a toy vocabulary of multi-character tokens, an "EOS" at the end
        let vocab: Vec<String> = ["a", "b", "ab", "12", "3", "-", ".", " ", "x", "42", "{", "}", "\"", "name", ":", "<eos>"].iter().map(|s| s.to_string()).collect();
        let eos = vocab.len() - 1;
        for pattern in [r"-?\d+(\.\d+)?", r"(ab|a)+x", r#"\{"name": ?\d{1,3}\}"#] {
            let re = Arc::new(Regex::new(pattern).unwrap());
            for seed in 0..40u64 {
                let mut c = RegexConstraint::new(re.clone());
                let mut text = String::new();
                for step in 0..30 {
                    // pseudo-random "model" preferences
                    let logits: Vec<f32> = (0..vocab.len()).map(|i| (((seed * 31 + step * 17 + i as u64 * 13) * 2654435761) % 1000) as f32 / 100.0).collect();
                    let masked = c.mask_logits(&Tensor::from_slice(&logits, &[vocab.len()]).unwrap(), &vocab, Some(eos)).unwrap().to_vec::<f32>().unwrap();
                    let best = (0..masked.len()).max_by(|&a, &b| masked[a].partial_cmp(&masked[b]).unwrap()).unwrap();
                    assert!(masked[best].is_finite());
                    if best == eos {
                        break;
                    }
                    c.advance(&vocab[best]);
                    text.push_str(&vocab[best]);
                }
                if c.is_complete() {
                    assert!(re.is_match(&text), "{pattern:?} produced {text:?}");
                }
                // whatever was generated is at least a viable prefix
                assert!(!re.feed(&re.initial(), &text).is_empty(), "{pattern:?} dead-ended on {text:?}");
            }
        }
    }

    #[test]
    fn a_vocabulary_that_cannot_continue_the_match_is_reported() {
        let re = Arc::new(Regex::new("abc").unwrap());
        let mut c = RegexConstraint::new(re);
        let vocab = vec!["z".to_string(), "<eos>".to_string()];
        let r = c.mask_logits(&Tensor::from_slice(&[0.0f32, 0.0], &[2]).unwrap(), &vocab, Some(1));
        assert!(r.is_err());
    }
}

