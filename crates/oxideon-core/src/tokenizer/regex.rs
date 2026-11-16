//! A small backtracking regular-expression engine with the syntax subset tokenizer pre-splitting patterns use
//! (GPT-2, Llama-3, Qwen, cl100k / o200k, DeepSeek, BERT-style): literals and escapes, `.` `^` `$` `\b`, classes
//! with ranges / negation / nesting, `\d \w \s`, Unicode properties (`\p{L}`, `\p{Lu}`, `\pN`, a few scripts), groups
//! `(...)` `(?:...)` `(?i:...)` `(?i)`, alternation, greedy / lazy / possessive quantifiers, counted repetition, and
//! look-ahead / look-behind (`(?=` `(?!` `(?<=` `(?<!`). Matching is leftmost-first (Perl / Oniguruma order) over
//! `char`s; positions are `char` indices.

use super::unicode;
use crate::error::{Error, Result};

#[derive(Debug, Clone)]
enum Item {
    Range(char, char),
    Digit(bool),
    Word(bool),
    Space(bool),
    /// Property name (a general-category prefix like `L` / `Lu`, or a supported script) and whether it is negated.
    Prop(String, bool),
    Nested(Box<ClassSet>),
}

#[derive(Debug, Clone)]
struct ClassSet {
    items: Vec<Item>,
    neg: bool,
    ci: bool,
}

fn script_ranges(name: &str) -> Option<&'static [(u32, u32)]> {
    Some(match name {
        "Han" => &[(0x2E80, 0x2FDF), (0x3005, 0x3005), (0x3007, 0x3007), (0x3021, 0x3029), (0x3038, 0x303B), (0x3400, 0x4DBF), (0x4E00, 0x9FFF), (0xF900, 0xFAFF), (0x20000, 0x323AF)],
        "Hiragana" => &[(0x3041, 0x3096), (0x309D, 0x309F), (0x1B001, 0x1B11F), (0x1F200, 0x1F200)],
        "Katakana" => &[(0x30A1, 0x30FA), (0x30FD, 0x30FF), (0x31F0, 0x31FF), (0x32D0, 0x32FE), (0x3300, 0x3357), (0xFF66, 0xFF6F), (0xFF71, 0xFF9D), (0x1B000, 0x1B000)],
        "Hangul" => &[(0x1100, 0x11FF), (0x302E, 0x302F), (0x3131, 0x318E), (0x3200, 0x321E), (0x3260, 0x327E), (0xA960, 0xA97C), (0xAC00, 0xD7A3), (0xD7B0, 0xD7FB), (0xFFA0, 0xFFDC)],
        "Latin" => &[(0x41, 0x5A), (0x61, 0x7A), (0xAA, 0xAA), (0xBA, 0xBA), (0xC0, 0xD6), (0xD8, 0xF6), (0xF8, 0x2B8), (0x1E00, 0x1EFF), (0x2C60, 0x2C7F), (0xA720, 0xA7FF), (0xFB00, 0xFB06), (0xFF21, 0xFF3A), (0xFF41, 0xFF5A)],
        "Greek" => &[(0x370, 0x373), (0x375, 0x377), (0x37A, 0x37D), (0x37F, 0x37F), (0x384, 0x384), (0x386, 0x386), (0x388, 0x3E1), (0x3F0, 0x3FF), (0x1F00, 0x1FFE)],
        "Cyrillic" => &[(0x400, 0x484), (0x487, 0x52F), (0x1C80, 0x1C88), (0x1D2B, 0x1D2B), (0x2DE0, 0x2DFF), (0xA640, 0xA69F)],
        "Arabic" => &[(0x600, 0x604), (0x606, 0x60B), (0x60D, 0x61A), (0x61C, 0x61E), (0x620, 0x63F), (0x641, 0x64A), (0x656, 0x66F), (0x671, 0x6DC), (0x6DE, 0x6FF), (0x750, 0x77F), (0xFB50, 0xFDFF), (0xFE70, 0xFEFC)],
        "Hebrew" => &[(0x591, 0x5C7), (0x5D0, 0x5EA), (0x5EF, 0x5F4), (0xFB1D, 0xFB4F)],
        "Thai" => &[(0xE01, 0xE3A), (0xE40, 0xE5B)],
        "Devanagari" => &[(0x900, 0x950), (0x955, 0x963), (0x966, 0x97F), (0xA8E0, 0xA8FF)],
        _ => return None,
    })
}

fn prop_known(name: &str) -> bool {
    const CATS: [&str; 30] = [
        "Cc", "Cf", "Cn", "Co", "Cs", "Ll", "Lm", "Lo", "Lt", "Lu", "Mc", "Me", "Mn", "Nd", "Nl", "No", "Pc", "Pd", "Pe", "Pf", "Pi", "Po", "Ps", "Sc", "Sk", "Sm", "So", "Zl", "Zp", "Zs",
    ];
    matches!(name, "L" | "M" | "N" | "P" | "S" | "Z" | "C" | "Any" | "Alphabetic" | "White_Space" | "Lowercase" | "Uppercase" | "LC" | "L&")
        || CATS.contains(&name)
        || script_ranges(name).is_some()
}

fn prop_match(name: &str, c: char) -> bool {
    match name {
        "Any" => true,
        "Alphabetic" => c.is_alphabetic(),
        "White_Space" => c.is_whitespace(),
        "Lowercase" => c.is_lowercase(),
        "Uppercase" => c.is_uppercase(),
        "LC" | "L&" => matches!(unicode::category(c), "Lu" | "Ll" | "Lt"),
        _ => match script_ranges(name) {
            Some(r) => r.iter().any(|&(a, b)| (a..=b).contains(&(c as u32))),
            None => unicode::category(c).starts_with(name),
        },
    }
}

/// `\w`: alphabetic characters, marks, decimal digits, connector punctuation and the join controls (U+200C, U+200D),
/// as Oniguruma defines it for Unicode text.
fn is_word(c: char) -> bool {
    if c.is_ascii() {
        return c.is_ascii_alphanumeric() || c == '_';
    }
    let cat = unicode::category(c);
    c.is_alphabetic() || cat.starts_with('L') || cat.starts_with('M') || cat == "Nd" || cat == "Pc" || c == '\u{200c}' || c == '\u{200d}'
}

impl ClassSet {
    fn raw(&self, c: char) -> bool {
        self.items.iter().any(|it| match it {
            Item::Range(a, b) => *a <= c && c <= *b,
            Item::Digit(n) => (unicode::category(c) == "Nd") != *n,
            Item::Word(n) => is_word(c) != *n,
            Item::Space(n) => c.is_whitespace() != *n,
            Item::Prop(p, n) => prop_match(p, c) != *n,
            Item::Nested(s) => s.matches(c),
        })
    }
    fn matches(&self, c: char) -> bool {
        let mut hit = self.raw(c);
        if !hit && self.ci {
            hit = c.to_lowercase().any(|l| l != c && self.raw(l)) || c.to_uppercase().any(|u| u != c && self.raw(u));
        }
        hit != self.neg
    }
}

#[derive(Debug, Clone)]
enum Node {
    Empty,
    Char(char, bool),
    Any(bool),
    Class(ClassSet),
    Seq(Vec<Node>),
    Alt(Vec<Node>),
    Repeat { node: Box<Node>, min: u32, max: u32, greedy: bool, possessive: bool },
    Look { node: Box<Node>, neg: bool, behind: bool },
    Bol,
    Eol,
    WordB(bool),
    TextStart,
    TextEnd,
}

#[derive(Debug, Clone)]
enum Inst {
    Char(char),
    CharCi(char),
    Any(bool),
    Class(usize),
    Split(usize, usize),
    Jmp(usize),
    Bol,
    Eol,
    WordB(bool),
    TextStart,
    TextEnd,
    /// Sub-program starts at the next instruction and ends with `SubEnd`; execution continues at `end`.
    Look { neg: bool, behind: bool, end: usize },
    /// Possessive / atomic group: sub-program starts at the next instruction and ends with `SubEnd`.
    Atomic { end: usize },
    SubEnd,
    Match,
}

struct Parser<'a> {
    s: &'a [char],
    i: usize,
    ci: bool,
    dotall: bool,
}

fn err<T>(msg: impl Into<String>) -> Result<T> {
    Err(Error::Msg(format!("regex: {}", msg.into())))
}

impl Parser<'_> {
    fn peek(&self) -> Option<char> {
        self.s.get(self.i).copied()
    }
    fn eat(&mut self, c: char) -> bool {
        if self.peek() == Some(c) {
            self.i += 1;
            true
        } else {
            false
        }
    }
    fn starts_with(&self, t: &str) -> bool {
        t.chars().enumerate().all(|(k, c)| self.s.get(self.i + k) == Some(&c))
    }

    fn alt(&mut self) -> Result<Node> {
        let mut alts = vec![self.seq()?];
        while self.eat('|') {
            alts.push(self.seq()?);
        }
        Ok(if alts.len() == 1 { alts.pop().unwrap() } else { Node::Alt(alts) })
    }

    fn seq(&mut self) -> Result<Node> {
        let mut items = Vec::new();
        while let Some(c) = self.peek() {
            if c == '|' || c == ')' {
                break;
            }
            let atom = self.atom()?;
            let atom = self.quantify(atom)?;
            items.push(atom);
        }
        Ok(match items.len() {
            0 => Node::Empty,
            1 => items.pop().unwrap(),
            _ => Node::Seq(items),
        })
    }

    fn quantify(&mut self, mut atom: Node) -> Result<Node> {
        loop {
            let (min, max) = match self.peek() {
                Some('*') => (0, u32::MAX),
                Some('+') => (1, u32::MAX),
                Some('?') => (0, 1),
                Some('{') => match self.counted()? {
                    Some(mm) => {
                        // `counted` consumed the braces
                        let (greedy, possessive) = self.suffix();
                        atom = Node::Repeat { node: Box::new(atom), min: mm.0, max: mm.1, greedy, possessive };
                        continue;
                    }
                    None => return Ok(atom),
                },
                _ => return Ok(atom),
            };
            self.i += 1;
            let (greedy, possessive) = self.suffix();
            atom = Node::Repeat { node: Box::new(atom), min, max, greedy, possessive };
        }
    }

    fn suffix(&mut self) -> (bool, bool) {
        if self.eat('?') {
            (false, false)
        } else if self.eat('+') {
            (true, true)
        } else {
            (true, false)
        }
    }

    /// Parses `{n}`, `{n,}` or `{n,m}`; leaves the position untouched and returns `None` when it is a literal `{`.
    fn counted(&mut self) -> Result<Option<(u32, u32)>> {
        let save = self.i;
        self.i += 1;
        let num = |p: &mut Self| -> Option<u32> {
            let st = p.i;
            while p.peek().is_some_and(|c| c.is_ascii_digit()) {
                p.i += 1;
            }
            if p.i == st { None } else { p.s[st..p.i].iter().collect::<String>().parse().ok() }
        };
        let lo = num(self);
        let res = if self.eat(',') {
            let hi = num(self);
            lo.map(|l| (l, hi.unwrap_or(u32::MAX))).or(hi.map(|h| (0, h)))
        } else {
            lo.map(|l| (l, l))
        };
        if let (Some((a, b)), true) = (res, self.eat('}')) {
            if a > b || b != u32::MAX && b > 1000 || a > 1000 {
                return err("repetition count too large or inverted");
            }
            Ok(res)
        } else {
            // not a quantifier: the brace is a literal handled by `atom`
            self.i = save;
            Ok(None)
        }
    }

    fn atom(&mut self) -> Result<Node> {
        let c = self.peek().unwrap();
        self.i += 1;
        match c {
            '.' => Ok(Node::Any(self.dotall)),
            '^' => Ok(Node::Bol),
            '$' => Ok(Node::Eol),
            '(' => self.group(),
            '[' => Ok(Node::Class(self.class()?)),
            '\\' => self.escape_atom(),
            '*' | '+' | '?' => err("nothing to repeat"),
            '{' => Ok(Node::Char('{', false)),
            c => Ok(Node::Char(c, self.ci && c.to_lowercase().chain(c.to_uppercase()).any(|x| x != c))),
        }
    }

    fn group(&mut self) -> Result<Node> {
        let (saved_ci, saved_dot) = (self.ci, self.dotall);
        let mut kind = 'c'; // capture / plain group
        let mut neg = false;
        let mut behind = false;
        if self.eat('?') {
            if self.eat(':') {
                kind = 'n';
            } else if self.eat('>') {
                kind = 'a';
            } else if self.eat('=') {
                kind = 'l';
            } else if self.eat('!') {
                kind = 'l';
                neg = true;
            } else if self.starts_with("<=") || self.starts_with("<!") {
                self.i += 1;
                neg = self.peek() == Some('!');
                self.i += 1;
                kind = 'l';
                behind = true;
            } else if self.eat('<') || self.starts_with("P<") {
                // named capture group (?<name>...) / (?P<name>...)
                if self.peek() == Some('P') {
                    self.i += 2;
                }
                while self.peek().is_some_and(|c| c != '>') {
                    self.i += 1;
                }
                if !self.eat('>') {
                    return err("unterminated group name");
                }
            } else {
                // inline flags: (?i) (?i:...) (?-i:...) (?s) ...
                let mut on = true;
                let (mut ci, mut dot) = (self.ci, self.dotall);
                loop {
                    match self.peek() {
                        Some('-') => on = false,
                        Some('i') => ci = on,
                        Some('s') => dot = on,
                        Some('m') | Some('x') | Some('u') => {}
                        Some(':') => {
                            self.i += 1;
                            self.ci = ci;
                            self.dotall = dot;
                            let inner = self.alt()?;
                            if !self.eat(')') {
                                return err("missing )");
                            }
                            self.ci = saved_ci;
                            self.dotall = saved_dot;
                            return Ok(inner);
                        }
                        Some(')') => {
                            // flags apply to the rest of the enclosing group
                            self.i += 1;
                            self.ci = ci;
                            self.dotall = dot;
                            return Ok(Node::Empty);
                        }
                        _ => return err("unsupported group flag"),
                    }
                    self.i += 1;
                }
            }
        }
        let inner = self.alt()?;
        if !self.eat(')') {
            return err("missing )");
        }
        self.ci = saved_ci;
        self.dotall = saved_dot;
        Ok(match kind {
            'l' => Node::Look { node: Box::new(inner), neg, behind },
            'a' => Node::Repeat { node: Box::new(inner), min: 1, max: 1, greedy: true, possessive: true },
            _ => inner,
        })
    }

    fn hex(&mut self, n: usize) -> Result<char> {
        let mut v = 0u32;
        for _ in 0..n {
            let d = self.peek().and_then(|c| c.to_digit(16)).ok_or_else(|| Error::Msg("regex: bad hex escape".into()))?;
            v = v * 16 + d;
            self.i += 1;
        }
        char::from_u32(v).ok_or_else(|| Error::Msg("regex: bad code point".into()))
    }

    /// Parses an escape that denotes a single class item (after the backslash). Returns `Ok(Err(c))` for a literal.
    fn escape_item(&mut self) -> Result<std::result::Result<Item, char>> {
        let c = self.peek().ok_or_else(|| Error::Msg("regex: trailing backslash".into()))?;
        self.i += 1;
        Ok(match c {
            'd' => Ok(Item::Digit(false)),
            'D' => Ok(Item::Digit(true)),
            'w' => Ok(Item::Word(false)),
            'W' => Ok(Item::Word(true)),
            's' => Ok(Item::Space(false)),
            'S' => Ok(Item::Space(true)),
            'p' | 'P' => {
                let name: String = if self.eat('{') {
                    let st = self.i;
                    while self.peek().is_some_and(|c| c != '}') {
                        self.i += 1;
                    }
                    let n = self.s[st..self.i].iter().collect();
                    if !self.eat('}') {
                        return err("unterminated \\p{");
                    }
                    n
                } else {
                    let n = self.peek().ok_or_else(|| Error::Msg("regex: bad \\p".into()))?;
                    self.i += 1;
                    n.to_string()
                };
                let (name, neg) = match name.strip_prefix('^') {
                    Some(r) => (r.to_string(), c == 'p'),
                    None => (name, c == 'P'),
                };
                let name = name.strip_prefix("Script=").or_else(|| name.strip_prefix("sc=")).unwrap_or(&name).to_string();
                if !prop_known(&name) {
                    return err(format!("unsupported Unicode property {name}"));
                }
                Ok(Item::Prop(name, neg))
            }
            'n' => Err('\n'),
            'r' => Err('\r'),
            't' => Err('\t'),
            'f' => Err('\u{c}'),
            'v' => Err('\u{b}'),
            'a' => Err('\u{7}'),
            'e' => Err('\u{1b}'),
            '0' => Err('\0'),
            'x' => {
                if self.eat('{') {
                    let st = self.i;
                    while self.peek().is_some_and(|c| c != '}') {
                        self.i += 1;
                    }
                    let h: String = self.s[st..self.i].iter().collect();
                    self.i += 1;
                    Err(u32::from_str_radix(&h, 16).ok().and_then(char::from_u32).ok_or_else(|| Error::Msg("regex: bad \\x{}".into()))?)
                } else {
                    Err(self.hex(2)?)
                }
            }
            'u' => {
                if self.eat('{') {
                    let st = self.i;
                    while self.peek().is_some_and(|c| c != '}') {
                        self.i += 1;
                    }
                    let h: String = self.s[st..self.i].iter().collect();
                    self.i += 1;
                    Err(u32::from_str_radix(&h, 16).ok().and_then(char::from_u32).ok_or_else(|| Error::Msg("regex: bad \\u{}".into()))?)
                } else {
                    Err(self.hex(4)?)
                }
            }
            c => Err(c),
        })
    }

    fn escape_atom(&mut self) -> Result<Node> {
        match self.peek() {
            Some('b') => {
                self.i += 1;
                return Ok(Node::WordB(true));
            }
            Some('B') => {
                self.i += 1;
                return Ok(Node::WordB(false));
            }
            Some('A') => {
                self.i += 1;
                return Ok(Node::TextStart);
            }
            Some('z') | Some('Z') => {
                self.i += 1;
                return Ok(Node::TextEnd);
            }
            _ => {}
        }
        Ok(match self.escape_item()? {
            Ok(item) => Node::Class(ClassSet { items: vec![item], neg: false, ci: false }),
            Err(c) => Node::Char(c, self.ci && c.to_lowercase().chain(c.to_uppercase()).any(|x| x != c)),
        })
    }

    fn class(&mut self) -> Result<ClassSet> {
        let neg = self.eat('^');
        let mut items = Vec::new();
        let mut first = true;
        loop {
            let c = self.peek().ok_or_else(|| Error::Msg("regex: unterminated class".into()))?;
            if c == ']' && !first {
                self.i += 1;
                break;
            }
            first = false;
            self.i += 1;
            let lo: Option<char> = match c {
                '[' if self.peek() == Some(':') => return err("POSIX classes are not supported"),
                '[' => {
                    items.push(Item::Nested(Box::new(self.class()?)));
                    None
                }
                '&' if self.peek() == Some('&') => return err("class intersection is not supported"),
                '\\' => match self.escape_item()? {
                    Ok(it) => {
                        items.push(it);
                        None
                    }
                    Err(ch) => Some(ch),
                },
                c => Some(c),
            };
            if let Some(lo) = lo {
                // range `a-z` (a trailing or leading '-' is literal)
                if self.peek() == Some('-') && self.s.get(self.i + 1).is_some_and(|&n| n != ']') {
                    self.i += 1;
                    let hc = self.peek().unwrap();
                    self.i += 1;
                    let hi = if hc == '\\' {
                        match self.escape_item()? {
                            Err(ch) => ch,
                            Ok(_) => return err("class escape used as range end"),
                        }
                    } else {
                        hc
                    };
                    if hi < lo {
                        return err("inverted class range");
                    }
                    items.push(Item::Range(lo, hi));
                } else {
                    items.push(Item::Range(lo, lo));
                }
            }
        }
        Ok(ClassSet { items, neg, ci: self.ci })
    }
}

#[derive(Default)]
struct Compiler {
    prog: Vec<Inst>,
    classes: Vec<ClassSet>,
}

impl Compiler {
    fn emit(&mut self, i: Inst) -> usize {
        self.prog.push(i);
        self.prog.len() - 1
    }

    fn compile(&mut self, n: &Node) {
        match n {
            Node::Empty => {}
            Node::Char(c, ci) => {
                self.emit(if *ci { Inst::CharCi(*c) } else { Inst::Char(*c) });
            }
            Node::Any(d) => {
                self.emit(Inst::Any(*d));
            }
            Node::Class(cs) => {
                self.classes.push(cs.clone());
                let k = self.classes.len() - 1;
                self.emit(Inst::Class(k));
            }
            Node::Seq(v) => v.iter().for_each(|x| self.compile(x)),
            Node::Alt(v) => {
                let mut jumps = Vec::new();
                for (k, alt) in v.iter().enumerate() {
                    if k + 1 < v.len() {
                        let split = self.emit(Inst::Split(0, 0));
                        self.compile(alt);
                        jumps.push(self.emit(Inst::Jmp(0)));
                        let next = self.prog.len();
                        self.prog[split] = Inst::Split(split + 1, next);
                    } else {
                        self.compile(alt);
                    }
                }
                let end = self.prog.len();
                for j in jumps {
                    self.prog[j] = Inst::Jmp(end);
                }
            }
            Node::Repeat { node, min, max, greedy, possessive } => {
                if *possessive {
                    let a = self.emit(Inst::Atomic { end: 0 });
                    self.repeat(node, *min, *max, true);
                    self.emit(Inst::SubEnd);
                    let end = self.prog.len();
                    self.prog[a] = Inst::Atomic { end };
                } else {
                    self.repeat(node, *min, *max, *greedy);
                }
            }
            Node::Look { node, neg, behind } => {
                let l = self.emit(Inst::Look { neg: *neg, behind: *behind, end: 0 });
                self.compile(node);
                self.emit(Inst::SubEnd);
                let end = self.prog.len();
                self.prog[l] = Inst::Look { neg: *neg, behind: *behind, end };
            }
            Node::Bol => {
                self.emit(Inst::Bol);
            }
            Node::Eol => {
                self.emit(Inst::Eol);
            }
            Node::WordB(b) => {
                self.emit(Inst::WordB(*b));
            }
            Node::TextStart => {
                self.emit(Inst::TextStart);
            }
            Node::TextEnd => {
                self.emit(Inst::TextEnd);
            }
        }
    }

    fn repeat(&mut self, node: &Node, min: u32, max: u32, greedy: bool) {
        for _ in 0..min {
            self.compile(node);
        }
        if max == u32::MAX {
            // L: split(body, exit); body; jmp L
            let split = self.emit(Inst::Split(0, 0));
            self.compile(node);
            self.emit(Inst::Jmp(split));
            let exit = self.prog.len();
            self.prog[split] = if greedy { Inst::Split(split + 1, exit) } else { Inst::Split(exit, split + 1) };
        } else {
            let mut splits = Vec::new();
            for _ in min..max {
                splits.push(self.emit(Inst::Split(0, 0)));
                self.compile(node);
            }
            let exit = self.prog.len();
            for s in splits {
                self.prog[s] = if greedy { Inst::Split(s + 1, exit) } else { Inst::Split(exit, s + 1) };
            }
        }
    }
}

/// A compiled pattern.
#[derive(Debug, Clone)]
pub struct Regex {
    prog: Vec<Inst>,
    classes: Vec<ClassSet>,
}

const STEP_LIMIT: usize = 200_000_000;

impl Regex {
    pub fn new(pattern: &str) -> Result<Self> {
        let chars: Vec<char> = pattern.chars().collect();
        let mut p = Parser { s: &chars, i: 0, ci: false, dotall: false };
        let node = p.alt()?;
        if p.i != chars.len() {
            return err("unmatched )");
        }
        let mut c = Compiler::default();
        c.compile(&node);
        c.emit(Inst::Match);
        Ok(Self { prog: c.prog, classes: c.classes })
    }

    /// Runs the program from `pc` at `pos`; `must_end` is set inside look-behind (the sub-match has to end there).
    /// Returns the end position of the match.
    fn run(&self, text: &[char], pc0: usize, pos0: usize, must_end: Option<usize>, steps: &mut usize) -> Option<usize> {
        let mut stack: Vec<(usize, usize)> = Vec::new();
        let (mut pc, mut pos) = (pc0, pos0);
        loop {
            *steps += 1;
            if *steps > STEP_LIMIT {
                return None;
            }
            let ok = match &self.prog[pc] {
                Inst::Match => return Some(pos),
                Inst::SubEnd => {
                    if must_end.is_none_or(|e| e == pos) {
                        return Some(pos);
                    }
                    false
                }
                Inst::Char(c) => {
                    if text.get(pos) == Some(c) {
                        pos += 1;
                        pc += 1;
                        true
                    } else {
                        false
                    }
                }
                Inst::CharCi(c) => match text.get(pos) {
                    Some(&t) if t == *c || t.to_lowercase().eq(c.to_lowercase()) => {
                        pos += 1;
                        pc += 1;
                        true
                    }
                    _ => false,
                },
                Inst::Any(dotall) => match text.get(pos) {
                    Some(&t) if *dotall || t != '\n' => {
                        pos += 1;
                        pc += 1;
                        true
                    }
                    _ => false,
                },
                Inst::Class(k) => match text.get(pos) {
                    Some(&t) if self.classes[*k].matches(t) => {
                        pos += 1;
                        pc += 1;
                        true
                    }
                    _ => false,
                },
                Inst::Split(a, b) => {
                    stack.push((*b, pos));
                    pc = *a;
                    true
                }
                Inst::Jmp(t) => {
                    pc = *t;
                    true
                }
                Inst::Bol => {
                    if pos == 0 || text[pos - 1] == '\n' {
                        pc += 1;
                        true
                    } else {
                        false
                    }
                }
                Inst::Eol => {
                    if pos == text.len() || text[pos] == '\n' {
                        pc += 1;
                        true
                    } else {
                        false
                    }
                }
                Inst::TextStart => {
                    if pos == 0 {
                        pc += 1;
                        true
                    } else {
                        false
                    }
                }
                Inst::TextEnd => {
                    if pos == text.len() {
                        pc += 1;
                        true
                    } else {
                        false
                    }
                }
                Inst::WordB(want) => {
                    let before = pos > 0 && is_word(text[pos - 1]);
                    let after = pos < text.len() && is_word(text[pos]);
                    if (before != after) == *want {
                        pc += 1;
                        true
                    } else {
                        false
                    }
                }
                Inst::Look { neg, behind, end } => {
                    let found = if *behind {
                        (0..=pos).rev().any(|j| self.run(text, pc + 1, j, Some(pos), steps).is_some())
                    } else {
                        self.run(text, pc + 1, pos, None, steps).is_some()
                    };
                    if found != *neg {
                        pc = *end;
                        true
                    } else {
                        false
                    }
                }
                Inst::Atomic { end } => match self.run(text, pc + 1, pos, None, steps) {
                    Some(e) => {
                        pos = e;
                        pc = *end;
                        true
                    }
                    None => false,
                },
            };
            if !ok {
                match stack.pop() {
                    Some((p, q)) => {
                        pc = p;
                        pos = q;
                    }
                    None => return None,
                }
            }
        }
    }

    /// First match starting at or after `start`: `(match_start, match_end)` in `char` indices.
    pub fn find_at(&self, text: &[char], start: usize) -> Option<(usize, usize)> {
        let mut steps = 0usize;
        (start..=text.len()).find_map(|s| self.run(text, 0, s, None, &mut steps).map(|e| (s, e)))
    }

    /// Whether the pattern matches anywhere.
    pub fn is_match(&self, text: &str) -> bool {
        let t: Vec<char> = text.chars().collect();
        self.find_at(&t, 0).is_some()
    }

    /// All non-overlapping matches as `char` index ranges (an empty match advances by one character).
    pub fn find_all(&self, text: &[char]) -> Vec<(usize, usize)> {
        let mut out = Vec::new();
        let mut pos = 0;
        while pos <= text.len() {
            match self.find_at(text, pos) {
                Some((s, e)) => {
                    out.push((s, e));
                    pos = if e > s { e } else { e + 1 };
                }
                None => break,
            }
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn all(p: &str, t: &str) -> Vec<String> {
        let r = Regex::new(p).unwrap();
        let cs: Vec<char> = t.chars().collect();
        r.find_all(&cs).into_iter().map(|(a, b)| cs[a..b].iter().collect()).collect()
    }

    #[test]
    fn gpt2_pattern_splits_words() {
        let p = r"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+";
        assert_eq!(all(p, "Hello world's  42!!  x"), vec!["Hello", " world", "'s", " ", " 42", "!!", " ", " x"]);
    }

    #[test]
    fn lookaround_flags_and_counts() {
        assert_eq!(all(r"(?i:'S|'T)", "it's 'T"), vec!["'s", "'T"]);
        assert_eq!(all(r"\p{N}{1,3}", "1234567"), vec!["123", "456", "7"]);
        assert_eq!(all(r"a(?=b)", "ab ac ab"), vec!["a", "a"]);
        assert_eq!(all(r"(?<=a)b", "ab cb"), vec!["b"]);
        assert_eq!(all(r"a.*?b", "axxbxb"), vec!["axxb"]);
        assert_eq!(all(r"\bfoo\b", "foo food foo"), vec!["foo", "foo"]);
        assert_eq!(all(r"[^
\p{L}\p{N}]?\p{L}+", "x, yz"), vec!["x", " yz"]);
        assert_eq!(all(r"\p{Han}+", "ab中文cd日本"), vec!["中文", "日本"]);
        assert!(Regex::new(r"\p{Bogus}").is_err());
        assert!(Regex::new("(a").is_err());
        assert_eq!(all(r"a{2}", "aaaaa"), vec!["aa", "aa"]);
        assert_eq!(all(r"x{", "x{"), vec!["x{"]);
    }
}
