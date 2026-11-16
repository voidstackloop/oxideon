//! Chat templates: a Jinja2 subset interpreter with the settings Hugging Face uses (`trim_blocks`, `lstrip_blocks`,
//! `loopcontrols`, `tojson`, `raise_exception`, `strftime_now`, `namespace`), enough for the templates shipped with
//! Llama, Mistral, Qwen, Gemma, Phi, ChatML models and the like.
//!
//! Supported: `{{ }}`, `{% if/elif/else %}`, `{% for ... in ... if ... %}` with `loop.*` and `else`, `{% set %}` (also
//! `ns.attr`, tuple targets and block form), `{% macro %}`, `{% break %}` / `{% continue %}`, comments, whitespace
//! control, inline `if`/`else`, slicing, string / list / dict methods, tests (`defined`, `none`, `string`, `mapping`, ...) and
//! the common filters.

use super::json::Json;
use crate::error::{Error, Result};
use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::Rc;

type Shared<T> = Rc<RefCell<T>>;

/// A template value (Python / Jinja semantics; lists and maps are shared references).
#[derive(Clone)]
pub enum Val {
    Undefined,
    None,
    Bool(bool),
    Int(i64),
    Float(f64),
    Str(String),
    List(Shared<Vec<Val>>),
    Map(Shared<Vec<(String, Val)>>),
    Ns(Shared<Vec<(String, Val)>>),
    Macro(Rc<MacroDef>),
}

pub struct MacroDef {
    name: String,
    params: Vec<(String, Option<Expr>)>,
    body: Vec<Node>,
}

fn list(v: Vec<Val>) -> Val {
    Val::List(Rc::new(RefCell::new(v)))
}
fn map(v: Vec<(String, Val)>) -> Val {
    Val::Map(Rc::new(RefCell::new(v)))
}

impl Val {
    pub fn from_json(j: &Json) -> Val {
        match j {
            Json::Null => Val::None,
            Json::Bool(b) => Val::Bool(*b),
            Json::Num(n) => {
                if n.fract() == 0.0 && n.abs() < 9e15 { Val::Int(*n as i64) } else { Val::Float(*n) }
            }
            Json::Str(s) => Val::Str(s.clone()),
            Json::Arr(a) => list(a.iter().map(Val::from_json).collect()),
            Json::Obj(o) => map(o.iter().map(|(k, v)| (k.clone(), Val::from_json(v))).collect()),
        }
    }

    fn truthy(&self) -> bool {
        match self {
            Val::Undefined | Val::None => false,
            Val::Bool(b) => *b,
            Val::Int(i) => *i != 0,
            Val::Float(f) => *f != 0.0,
            Val::Str(s) => !s.is_empty(),
            Val::List(l) => !l.borrow().is_empty(),
            Val::Map(m) | Val::Ns(m) => !m.borrow().is_empty(),
            Val::Macro(_) => true,
        }
    }

    fn type_name(&self) -> &'static str {
        match self {
            Val::Undefined => "undefined",
            Val::None => "none",
            Val::Bool(_) => "bool",
            Val::Int(_) => "int",
            Val::Float(_) => "float",
            Val::Str(_) => "str",
            Val::List(_) => "list",
            Val::Map(_) => "dict",
            Val::Ns(_) => "namespace",
            Val::Macro(_) => "macro",
        }
    }
}

fn float_repr(f: f64) -> String {
    if f.is_nan() {
        return "nan".into();
    }
    if f.is_infinite() {
        return if f > 0.0 { "inf".into() } else { "-inf".into() };
    }
    let s = format!("{f}");
    if s.contains('.') || s.contains('e') { s } else { format!("{s}.0") }
}

fn str_repr(s: &str) -> String {
    let q = if s.contains('\'') && !s.contains('"') { '"' } else { '\'' };
    let mut o = String::new();
    o.push(q);
    for c in s.chars() {
        match c {
            '\\' => o.push_str("\\\\"),
            '\n' => o.push_str("\\n"),
            '\r' => o.push_str("\\r"),
            '\t' => o.push_str("\\t"),
            c if c == q => {
                o.push('\\');
                o.push(c)
            }
            c => o.push(c),
        }
    }
    o.push(q);
    o
}

impl Val {
    /// `repr()` as Python prints it inside containers.
    fn repr(&self) -> String {
        match self {
            Val::Str(s) => str_repr(s),
            Val::List(l) => format!("[{}]", l.borrow().iter().map(Val::repr).collect::<Vec<_>>().join(", ")),
            Val::Map(m) => format!("{{{}}}", m.borrow().iter().map(|(k, v)| format!("{}: {}", str_repr(k), v.repr())).collect::<Vec<_>>().join(", ")),
            other => other.to_str(),
        }
    }

    /// `str()` (what `{{ x }}` prints).
    fn to_str(&self) -> String {
        match self {
            Val::Undefined => String::new(),
            Val::None => "None".into(),
            Val::Bool(true) => "True".into(),
            Val::Bool(false) => "False".into(),
            Val::Int(i) => i.to_string(),
            Val::Float(f) => float_repr(*f),
            Val::Str(s) => s.clone(),
            Val::List(_) | Val::Map(_) => self.repr(),
            Val::Ns(_) => "<Namespace>".into(),
            Val::Macro(m) => format!("<macro {}>", m.name),
        }
    }

    fn eq(&self, o: &Val) -> bool {
        match (self, o) {
            (Val::None, Val::None) | (Val::Undefined, Val::Undefined) => true,
            (Val::Bool(a), Val::Bool(b)) => a == b,
            (Val::Int(a), Val::Int(b)) => a == b,
            (Val::Float(a), Val::Float(b)) => a == b,
            (Val::Int(a), Val::Float(b)) | (Val::Float(b), Val::Int(a)) => *a as f64 == *b,
            (Val::Bool(a), Val::Int(b)) | (Val::Int(b), Val::Bool(a)) => (*a as i64) == *b,
            (Val::Str(a), Val::Str(b)) => a == b,
            (Val::List(a), Val::List(b)) => {
                let (a, b) = (a.borrow(), b.borrow());
                a.len() == b.len() && a.iter().zip(b.iter()).all(|(x, y)| x.eq(y))
            }
            (Val::Map(a), Val::Map(b)) => {
                let (a, b) = (a.borrow(), b.borrow());
                a.len() == b.len() && a.iter().all(|(k, v)| b.iter().find(|(k2, _)| k2 == k).is_some_and(|(_, v2)| v.eq(v2)))
            }
            _ => false,
        }
    }

    fn num(&self) -> Option<f64> {
        match self {
            Val::Int(i) => Some(*i as f64),
            Val::Float(f) => Some(*f),
            Val::Bool(b) => Some(*b as i64 as f64),
            _ => None,
        }
    }

    fn items(&self) -> Result<Vec<Val>> {
        Ok(match self {
            Val::List(l) => l.borrow().clone(),
            Val::Str(s) => s.chars().map(|c| Val::Str(c.to_string())).collect(),
            Val::Map(m) => m.borrow().iter().map(|(k, _)| Val::Str(k.clone())).collect(),
            Val::Undefined | Val::None => vec![],
            other => return Err(Error::Msg(format!("chat template: a {} is not iterable", other.type_name()))),
        })
    }
}

// ---------------------------------------------------------------------------------------------------------------- lexer

#[derive(Debug, Clone, PartialEq)]
enum Tok {
    Name(String),
    Int(i64),
    Float(f64),
    Str(String),
    Op(&'static str),
    End,
}

const OPS: [&str; 26] = ["**", "//", "==", "!=", "<=", ">=", "+", "-", "*", "/", "%", "~", "<", ">", "=", "|", ".", ",", ":", "(", ")", "[", "]", "{", "}", "!"];

fn lex_expr(src: &str) -> Result<Vec<Tok>> {
    let cs: Vec<char> = src.chars().collect();
    let mut i = 0;
    let mut out = Vec::new();
    while i < cs.len() {
        let c = cs[i];
        if c.is_whitespace() {
            i += 1;
        } else if c.is_alphabetic() || c == '_' {
            let st = i;
            while i < cs.len() && (cs[i].is_alphanumeric() || cs[i] == '_') {
                i += 1;
            }
            out.push(Tok::Name(cs[st..i].iter().collect()));
        } else if c.is_ascii_digit() {
            let st = i;
            while i < cs.len() && cs[i].is_ascii_digit() {
                i += 1;
            }
            if i + 1 < cs.len() && cs[i] == '.' && cs[i + 1].is_ascii_digit() {
                i += 1;
                while i < cs.len() && cs[i].is_ascii_digit() {
                    i += 1;
                }
                out.push(Tok::Float(cs[st..i].iter().collect::<String>().parse().map_err(|_| Error::Msg("chat template: bad number".into()))?));
            } else {
                out.push(Tok::Int(cs[st..i].iter().collect::<String>().parse().map_err(|_| Error::Msg("chat template: bad number".into()))?));
            }
        } else if c == '\'' || c == '"' {
            i += 1;
            let mut s = String::new();
            loop {
                let ch = *cs.get(i).ok_or_else(|| Error::Msg("chat template: unterminated string".into()))?;
                i += 1;
                if ch == c {
                    break;
                }
                if ch == '\\' {
                    let e = *cs.get(i).ok_or_else(|| Error::Msg("chat template: bad escape".into()))?;
                    i += 1;
                    match e {
                        'n' => s.push('\n'),
                        't' => s.push('\t'),
                        'r' => s.push('\r'),
                        '\\' => s.push('\\'),
                        '\'' => s.push('\''),
                        '"' => s.push('"'),
                        'u' => {
                            let h: String = cs[i..(i + 4).min(cs.len())].iter().collect();
                            i += 4;
                            s.push(u32::from_str_radix(&h, 16).ok().and_then(char::from_u32).ok_or_else(|| Error::Msg("chat template: bad \\u escape".into()))?);
                        }
                        'x' => {
                            let h: String = cs[i..(i + 2).min(cs.len())].iter().collect();
                            i += 2;
                            s.push(u32::from_str_radix(&h, 16).ok().and_then(char::from_u32).ok_or_else(|| Error::Msg("chat template: bad \\x escape".into()))?);
                        }
                        other => {
                            s.push('\\');
                            s.push(other)
                        }
                    }
                } else {
                    s.push(ch);
                }
            }
            out.push(Tok::Str(s));
        } else {
            let rest: String = cs[i..(i + 2).min(cs.len())].iter().collect();
            let op = OPS.iter().find(|o| rest.starts_with(**o)).ok_or_else(|| Error::Msg(format!("chat template: unexpected character {c:?}")))?;
            i += op.len();
            out.push(Tok::Op(op));
        }
    }
    out.push(Tok::End);
    Ok(out)
}

// ---------------------------------------------------------------------------------------------------------------- AST

#[derive(Debug, Clone)]
pub enum Expr {
    Lit(Lit),
    Name(String),
    Attr(Box<Expr>, String),
    Index(Box<Expr>, Box<Expr>),
    Slice(Box<Expr>, Option<Box<Expr>>, Option<Box<Expr>>, Option<Box<Expr>>),
    Call(Box<Expr>, Vec<Expr>, Vec<(String, Expr)>),
    Filter(Box<Expr>, String, Vec<Expr>, Vec<(String, Expr)>),
    Test(Box<Expr>, String, Vec<Expr>, bool),
    Bin(&'static str, Box<Expr>, Box<Expr>),
    Neg(Box<Expr>),
    Not(Box<Expr>),
    And(Box<Expr>, Box<Expr>),
    Or(Box<Expr>, Box<Expr>),
    Cond(Box<Expr>, Box<Expr>, Option<Box<Expr>>),
    List(Vec<Expr>),
    Dict(Vec<(Expr, Expr)>),
}

#[derive(Debug, Clone)]
pub enum Lit {
    None,
    Bool(bool),
    Int(i64),
    Float(f64),
    Str(String),
}

pub enum Node {
    Text(String),
    Print(Expr),
    If(Vec<(Expr, Vec<Node>)>, Vec<Node>),
    For { targets: Vec<String>, iter: Expr, cond: Option<Expr>, body: Vec<Node>, else_body: Vec<Node> },
    Set(Vec<Expr>, Expr),
    SetBlock(String, Vec<Node>),
    Macro(Rc<MacroDef>),
    Break,
    Continue,
}

type CallArgs = (Vec<Expr>, Vec<(String, Expr)>);

struct P {
    t: Vec<Tok>,
    i: usize,
}

impl P {
    fn peek(&self) -> &Tok {
        &self.t[self.i]
    }
    fn next(&mut self) -> Tok {
        let t = self.t[self.i].clone();
        if self.i + 1 < self.t.len() {
            self.i += 1;
        }
        t
    }
    fn op(&mut self, o: &str) -> bool {
        if matches!(self.peek(), Tok::Op(x) if *x == o) {
            self.i += 1;
            true
        } else {
            false
        }
    }
    fn name_is(&self, n: &str) -> bool {
        matches!(self.peek(), Tok::Name(x) if x == n)
    }
    fn kw(&mut self, n: &str) -> bool {
        if self.name_is(n) {
            self.i += 1;
            true
        } else {
            false
        }
    }
    fn expect_op(&mut self, o: &str) -> Result<()> {
        if self.op(o) { Ok(()) } else { Err(Error::Msg(format!("chat template: expected {o:?}, found {:?}", self.peek()))) }
    }

    fn expr(&mut self) -> Result<Expr> {
        self.ternary()
    }
    fn ternary(&mut self) -> Result<Expr> {
        let body = self.or()?;
        if self.kw("if") {
            let cond = self.or()?;
            let other = if self.kw("else") { Some(Box::new(self.ternary()?)) } else { None };
            return Ok(Expr::Cond(Box::new(body), Box::new(cond), other));
        }
        Ok(body)
    }
    fn or(&mut self) -> Result<Expr> {
        let mut l = self.and()?;
        while self.kw("or") {
            l = Expr::Or(Box::new(l), Box::new(self.and()?));
        }
        Ok(l)
    }
    fn and(&mut self) -> Result<Expr> {
        let mut l = self.not()?;
        while self.kw("and") {
            l = Expr::And(Box::new(l), Box::new(self.not()?));
        }
        Ok(l)
    }
    fn not(&mut self) -> Result<Expr> {
        if self.kw("not") {
            return Ok(Expr::Not(Box::new(self.not()?)));
        }
        self.compare()
    }
    fn compare(&mut self) -> Result<Expr> {
        let mut l = self.concat()?;
        loop {
            let op: Option<&'static str> = match self.peek() {
                Tok::Op(o @ ("==" | "!=" | "<" | ">" | "<=" | ">=")) => Some(o),
                Tok::Name(n) if n == "in" => Some("in"),
                Tok::Name(n) if n == "not" && matches!(self.t.get(self.i + 1), Some(Tok::Name(m)) if m == "in") => Some("not in"),
                Tok::Name(n) if n == "is" => Some("is"),
                _ => None,
            };
            let Some(op) = op else { return Ok(l) };
            self.i += if op == "not in" { 2 } else { 1 };
            if op == "is" {
                let neg = self.kw("not");
                let name = match self.next() {
                    Tok::Name(n) => n,
                    other => return Err(Error::Msg(format!("chat template: expected a test name, found {other:?}"))),
                };
                let mut args = vec![];
                if self.op("(") {
                    args = self.args()?.0;
                } else if !matches!(self.peek(), Tok::End | Tok::Op(")" | "]" | "," | ":" | "}")) && !self.name_is("and") && !self.name_is("or") && !self.name_is("if") && !self.name_is("else") && !self.name_is("is") && !self.name_is("in") {
                    // single bare argument, e.g. `x is divisibleby 3`
                    if matches!(self.peek(), Tok::Int(_) | Tok::Str(_) | Tok::Float(_)) {
                        args.push(self.unary()?);
                    }
                }
                l = Expr::Test(Box::new(l), name, args, neg);
            } else {
                let r = self.concat()?;
                l = Expr::Bin(op, Box::new(l), Box::new(r));
            }
        }
    }
    fn concat(&mut self) -> Result<Expr> {
        let mut l = self.add()?;
        while self.op("~") {
            l = Expr::Bin("~", Box::new(l), Box::new(self.add()?));
        }
        Ok(l)
    }
    fn add(&mut self) -> Result<Expr> {
        let mut l = self.mul()?;
        loop {
            let op = if self.op("+") { "+" } else if self.op("-") { "-" } else { return Ok(l) };
            l = Expr::Bin(op, Box::new(l), Box::new(self.mul()?));
        }
    }
    fn mul(&mut self) -> Result<Expr> {
        let mut l = self.unary()?;
        loop {
            let op = if self.op("*") { "*" } else if self.op("//") { "//" } else if self.op("/") { "/" } else if self.op("%") { "%" } else { return Ok(l) };
            l = Expr::Bin(op, Box::new(l), Box::new(self.unary()?));
        }
    }
    fn unary(&mut self) -> Result<Expr> {
        if self.op("-") {
            return Ok(Expr::Neg(Box::new(self.unary()?)));
        }
        if self.op("+") {
            return self.unary();
        }
        self.pow()
    }
    fn pow(&mut self) -> Result<Expr> {
        let l = self.postfix()?;
        if self.op("**") {
            return Ok(Expr::Bin("**", Box::new(l), Box::new(self.unary()?)));
        }
        Ok(l)
    }
    fn args(&mut self) -> Result<CallArgs> {
        let (mut pos, mut kw) = (vec![], vec![]);
        while !self.op(")") {
            if let (Tok::Name(n), Some(Tok::Op("="))) = (self.peek().clone(), self.t.get(self.i + 1)) {
                self.i += 2;
                kw.push((n, self.expr()?));
            } else {
                pos.push(self.expr()?);
            }
            if !self.op(",") {
                self.expect_op(")")?;
                break;
            }
        }
        Ok((pos, kw))
    }
    fn postfix(&mut self) -> Result<Expr> {
        let mut e = self.primary()?;
        loop {
            if self.op(".") {
                match self.next() {
                    Tok::Name(n) => e = Expr::Attr(Box::new(e), n),
                    Tok::Int(i) => e = Expr::Index(Box::new(e), Box::new(Expr::Lit(Lit::Int(i)))),
                    other => return Err(Error::Msg(format!("chat template: bad attribute {other:?}"))),
                }
            } else if self.op("[") {
                // index or slice
                let first = if matches!(self.peek(), Tok::Op(":")) { None } else { Some(self.expr()?) };
                if self.op(":") {
                    let stop = if matches!(self.peek(), Tok::Op("]" | ":")) { None } else { Some(Box::new(self.expr()?)) };
                    let step = if self.op(":") && !matches!(self.peek(), Tok::Op("]")) { Some(Box::new(self.expr()?)) } else { None };
                    self.expect_op("]")?;
                    e = Expr::Slice(Box::new(e), first.map(Box::new), stop, step);
                } else {
                    self.expect_op("]")?;
                    e = Expr::Index(Box::new(e), Box::new(first.ok_or_else(|| Error::Msg("chat template: empty index".into()))?));
                }
            } else if self.op("(") {
                let (pos, kw) = self.args()?;
                e = Expr::Call(Box::new(e), pos, kw);
            } else if self.op("|") {
                let name = match self.next() {
                    Tok::Name(n) => n,
                    other => return Err(Error::Msg(format!("chat template: bad filter {other:?}"))),
                };
                let (pos, kw) = if self.op("(") { self.args()? } else { (vec![], vec![]) };
                e = Expr::Filter(Box::new(e), name, pos, kw);
            } else {
                return Ok(e);
            }
        }
    }
    fn primary(&mut self) -> Result<Expr> {
        match self.next() {
            Tok::Int(i) => Ok(Expr::Lit(Lit::Int(i))),
            Tok::Float(f) => Ok(Expr::Lit(Lit::Float(f))),
            Tok::Str(s) => {
                // adjacent literals concatenate
                let mut s = s;
                while let Tok::Str(n) = self.peek().clone() {
                    self.i += 1;
                    s.push_str(&n);
                }
                Ok(Expr::Lit(Lit::Str(s)))
            }
            Tok::Name(n) => Ok(match n.as_str() {
                "true" | "True" => Expr::Lit(Lit::Bool(true)),
                "false" | "False" => Expr::Lit(Lit::Bool(false)),
                "none" | "None" => Expr::Lit(Lit::None),
                _ => Expr::Name(n),
            }),
            Tok::Op("(") => {
                if self.op(")") {
                    return Ok(Expr::List(vec![]));
                }
                let first = self.expr()?;
                if self.op(",") {
                    let mut v = vec![first];
                    while !self.op(")") {
                        v.push(self.expr()?);
                        if !self.op(",") {
                            self.expect_op(")")?;
                            break;
                        }
                    }
                    return Ok(Expr::List(v));
                }
                self.expect_op(")")?;
                Ok(first)
            }
            Tok::Op("[") => {
                let mut v = vec![];
                while !self.op("]") {
                    v.push(self.expr()?);
                    if !self.op(",") {
                        self.expect_op("]")?;
                        break;
                    }
                }
                Ok(Expr::List(v))
            }
            Tok::Op("{") => {
                let mut v = vec![];
                while !self.op("}") {
                    let k = self.expr()?;
                    self.expect_op(":")?;
                    v.push((k, self.expr()?));
                    if !self.op(",") {
                        self.expect_op("}")?;
                        break;
                    }
                }
                Ok(Expr::Dict(v))
            }
            other => Err(Error::Msg(format!("chat template: unexpected token {other:?}"))),
        }
    }
}

fn parse_expr(src: &str) -> Result<Expr> {
    let mut p = P { t: lex_expr(src)?, i: 0 };
    let e = p.expr()?;
    if !matches!(p.peek(), Tok::End) {
        return Err(Error::Msg(format!("chat template: unexpected {:?} after expression `{src}`", p.peek())));
    }
    Ok(e)
}

// ---------------------------------------------------------------------------------------------------------------- template parsing

enum Raw {
    Data(String),
    Var(String),
    Block(String),
}

/// Splits into data / `{{ }}` / `{% %}` pieces applying whitespace control, `trim_blocks` and `lstrip_blocks`.
fn split_template(src: &str) -> Result<Vec<Raw>> {
    let cs: Vec<char> = src.chars().collect();
    let mut out: Vec<Raw> = Vec::new();
    let mut data = String::new();
    let mut i = 0;
    let mut lstrip_next = false; // `-%}` / `-}}` strips leading whitespace of the next data
    let mut trim_nl = false; // trim_blocks: drop the first newline after a block tag
    while i < cs.len() {
        let two = if i + 1 < cs.len() { (cs[i], cs[i + 1]) } else { (' ', ' ') };
        let is_open = two == ('{', '{') || two == ('{', '%') || two == ('{', '#');
        if !is_open {
            let c = cs[i];
            if trim_nl {
                trim_nl = false;
                if c == '\n' {
                    i += 1;
                    continue;
                }
                if c == '\r' && cs.get(i + 1) == Some(&'\n') {
                    i += 2;
                    continue;
                }
            }
            if lstrip_next {
                if c.is_whitespace() {
                    i += 1;
                    continue;
                }
                lstrip_next = false;
            }
            data.push(c);
            i += 1;
            continue;
        }
        trim_nl = false;
        lstrip_next = false;
        let kind = two.1;
        let mut j = i + 2;
        let minus_left = cs.get(j) == Some(&'-');
        if minus_left || (kind == '%' && cs.get(j) == Some(&'+')) {
            j += 1;
        }
        // find the closing delimiter, skipping string literals
        let close: [char; 2] = match kind {
            '{' => ['}', '}'],
            '%' => ['%', '}'],
            _ => ['#', '}'],
        };
        let start = j;
        let mut k = j;
        let mut quote: Option<char> = None;
        loop {
            let c = *cs.get(k).ok_or_else(|| Error::Msg("chat template: unterminated tag".into()))?;
            match quote {
                Some(q) => {
                    if c == '\\' {
                        k += 1;
                    } else if c == q {
                        quote = None;
                    }
                }
                None => {
                    if (c == '\'' || c == '"') && kind != '#' {
                        quote = Some(c);
                    } else if c == close[0] && cs.get(k + 1) == Some(&close[1]) {
                        break;
                    }
                }
            }
            k += 1;
        }
        let minus_right = k > start && cs[k - 1] == '-';
        let inner: String = cs[start..if minus_right { k - 1 } else { k }].iter().collect();
        // whitespace before the tag
        if minus_left {
            while data.ends_with(|c: char| c.is_whitespace()) {
                data.pop();
            }
        } else if kind != '{' && cs.get(i + 2) != Some(&'+') {
            // lstrip_blocks: whitespace between the start of the line and the tag goes away
            let line_start = data.rfind('\n').map_or(0, |p| p + 1);
            let tag_starts_line = line_start > 0 || out.is_empty() || matches!(out.last(), Some(Raw::Block(_)));
            if tag_starts_line && data[line_start..].chars().all(|c| c == ' ' || c == '\t') {
                data.truncate(line_start);
            }
        }
        if !data.is_empty() {
            out.push(Raw::Data(std::mem::take(&mut data)));
        }
        match kind {
            '{' => out.push(Raw::Var(inner)),
            '%' => out.push(Raw::Block(inner)),
            _ => {}
        }
        i = k + 2;
        if minus_right {
            lstrip_next = true;
        } else if kind == '%' || kind == '#' {
            trim_nl = true;
        }
    }
    if !data.is_empty() {
        out.push(Raw::Data(data));
    }
    Ok(out)
}

struct TP {
    raw: Vec<Raw>,
    i: usize,
}

fn first_word(s: &str) -> &str {
    s.trim_start().split(|c: char| c.is_whitespace() || c == '(').next().unwrap_or("")
}

impl TP {
    /// Parses nodes until a block whose first word is one of `stops` (left unconsumed).
    fn nodes(&mut self, stops: &[&str]) -> Result<Vec<Node>> {
        let mut out = Vec::new();
        while self.i < self.raw.len() {
            match &self.raw[self.i] {
                Raw::Data(d) => {
                    out.push(Node::Text(d.clone()));
                    self.i += 1;
                }
                Raw::Var(e) => {
                    out.push(Node::Print(parse_expr(e)?));
                    self.i += 1;
                }
                Raw::Block(b) => {
                    let w = first_word(b).to_string();
                    if stops.contains(&w.as_str()) {
                        return Ok(out);
                    }
                    let b = b.clone();
                    self.i += 1;
                    out.push(self.statement(&w, b.trim_start()[w.len()..].trim())?);
                }
            }
        }
        if stops.is_empty() {
            Ok(out)
        } else {
            Err(Error::Msg(format!("chat template: missing {{% {} %}}", stops[0])))
        }
    }

    fn end(&mut self, name: &str) -> Result<()> {
        match self.raw.get(self.i) {
            Some(Raw::Block(b)) if first_word(b) == name => {
                self.i += 1;
                Ok(())
            }
            _ => Err(Error::Msg(format!("chat template: expected {{% {name} %}}"))),
        }
    }

    fn statement(&mut self, w: &str, rest: &str) -> Result<Node> {
        match w {
            "if" => {
                let mut arms = vec![(parse_expr(rest)?, self.nodes(&["elif", "else", "endif"])?)];
                let mut other = vec![];
                loop {
                    let Some(Raw::Block(b)) = self.raw.get(self.i) else { return Err(Error::Msg("chat template: missing {% endif %}".into())) };
                    let word = first_word(b).to_string();
                    let body = b.trim_start()[word.len()..].trim().to_string();
                    self.i += 1;
                    match word.as_str() {
                        "elif" => arms.push((parse_expr(&body)?, self.nodes(&["elif", "else", "endif"])?)),
                        "else" => {
                            other = self.nodes(&["endif"])?;
                        }
                        _ => break,
                    }
                }
                Ok(Node::If(arms, other))
            }
            "for" => {
                let (targets, iter) = rest.split_once(" in ").ok_or_else(|| Error::Msg("chat template: bad for statement".into()))?;
                let targets: Vec<String> = targets.split(',').map(|s| s.trim().trim_matches(|c| c == '(' || c == ')').trim().to_string()).collect();
                // an inline filter: `for x in y if cond`
                let (iter_src, cond) = split_top_level_if(iter);
                let body = self.nodes(&["else", "endfor"])?;
                let mut else_body = vec![];
                if matches!(self.raw.get(self.i), Some(Raw::Block(b)) if first_word(b) == "else") {
                    self.i += 1;
                    else_body = self.nodes(&["endfor"])?;
                }
                self.end("endfor")?;
                Ok(Node::For { targets, iter: parse_expr(&iter_src)?, cond: cond.map(|c| parse_expr(&c)).transpose()?, body, else_body })
            }
            "set" => {
                if let Some((lhs, rhs)) = split_assign(rest) {
                    let targets: Vec<Expr> = lhs.split(',').map(|s| parse_expr(s.trim())).collect::<Result<_>>()?;
                    Ok(Node::Set(targets, parse_expr(rhs)?))
                } else {
                    // block form: {% set x %} ... {% endset %}
                    let body = self.nodes(&["endset"])?;
                    self.end("endset")?;
                    Ok(Node::SetBlock(rest.trim().to_string(), body))
                }
            }
            "macro" => {
                let (name, params) = rest.split_once('(').ok_or_else(|| Error::Msg("chat template: bad macro".into()))?;
                let params = params.trim_end().trim_end_matches(')');
                let mut defs = Vec::new();
                for p in split_commas(params) {
                    if p.trim().is_empty() {
                        continue;
                    }
                    match p.split_once('=') {
                        Some((n, d)) => defs.push((n.trim().to_string(), Some(parse_expr(d.trim())?))),
                        None => defs.push((p.trim().to_string(), None)),
                    }
                }
                let body = self.nodes(&["endmacro"])?;
                self.end("endmacro")?;
                Ok(Node::Macro(Rc::new(MacroDef { name: name.trim().to_string(), params: defs, body })))
            }
            "break" => Ok(Node::Break),
            "continue" => Ok(Node::Continue),
            "generation" | "endgeneration" => Ok(Node::Text(String::new())),
            other => Err(Error::Msg(format!("chat template: unsupported statement `{other}`"))),
        }
    }
}

/// Splits `a, b(c, d), e` at top-level commas.
fn split_commas(s: &str) -> Vec<String> {
    let (mut out, mut cur, mut depth, mut quote) = (Vec::new(), String::new(), 0i32, None::<char>);
    for c in s.chars() {
        match quote {
            Some(q) => {
                cur.push(c);
                if c == q {
                    quote = None;
                }
            }
            None => match c {
                '\'' | '"' => {
                    quote = Some(c);
                    cur.push(c);
                }
                '(' | '[' | '{' => {
                    depth += 1;
                    cur.push(c);
                }
                ')' | ']' | '}' => {
                    depth -= 1;
                    cur.push(c);
                }
                ',' if depth == 0 => out.push(std::mem::take(&mut cur)),
                _ => cur.push(c),
            },
        }
    }
    out.push(cur);
    out
}

/// `x = expr` split at the first top-level `=` that is not part of `==`, `!=`, `<=`, `>=`.
fn split_assign(s: &str) -> Option<(&str, &str)> {
    let b = s.as_bytes();
    let (mut depth, mut quote) = (0i32, None::<u8>);
    let mut i = 0;
    while i < b.len() {
        let c = b[i];
        match quote {
            Some(q) => {
                if c == b'\\' {
                    i += 1;
                } else if c == q {
                    quote = None;
                }
            }
            None => match c {
                b'\'' | b'"' => quote = Some(c),
                b'(' | b'[' | b'{' => depth += 1,
                b')' | b']' | b'}' => depth -= 1,
                b'=' if depth == 0 => {
                    let prev = if i > 0 { b[i - 1] } else { b' ' };
                    let next = b.get(i + 1).copied().unwrap_or(b' ');
                    if next != b'=' && !matches!(prev, b'=' | b'!' | b'<' | b'>') {
                        return Some((&s[..i], &s[i + 1..]));
                    }
                    if next == b'=' {
                        i += 1;
                    }
                }
                _ => {}
            },
        }
        i += 1;
    }
    None
}

/// Splits `y if cond` at a top-level ` if ` (inline for-filter), but leaves a ternary inside brackets alone.
fn split_top_level_if(s: &str) -> (String, Option<String>) {
    let cs: Vec<char> = s.chars().collect();
    let (mut depth, mut quote) = (0i32, None::<char>);
    let mut i = 0;
    while i < cs.len() {
        let c = cs[i];
        match quote {
            Some(q) => {
                if c == q {
                    quote = None;
                }
            }
            None => match c {
                '\'' | '"' => quote = Some(c),
                '(' | '[' | '{' => depth += 1,
                ')' | ']' | '}' => depth -= 1,
                ' ' if depth == 0 && cs[i..].iter().take(4).collect::<String>() == " if " => {
                    return (cs[..i].iter().collect(), Some(cs[i + 4..].iter().collect()));
                }
                _ => {}
            },
        }
        i += 1;
    }
    (s.to_string(), None)
}

// ---------------------------------------------------------------------------------------------------------------- interpreter

enum Flow {
    Normal,
    Break,
    Continue,
}

struct Env {
    scopes: Vec<HashMap<String, Val>>,
    out: Vec<String>,
}

fn int_cmp(a: &Val, b: &Val) -> Result<std::cmp::Ordering> {
    match (a, b) {
        (Val::Str(x), Val::Str(y)) => Ok(x.cmp(y)),
        _ => match (a.num(), b.num()) {
            (Some(x), Some(y)) => x.partial_cmp(&y).ok_or_else(|| Error::Msg("chat template: cannot compare NaN".into())),
            _ => Err(Error::Msg(format!("chat template: cannot compare {} with {}", a.type_name(), b.type_name()))),
        },
    }
}

fn slice_bounds(len: usize, start: Option<i64>, stop: Option<i64>, step: i64) -> Vec<usize> {
    let n = len as i64;
    let norm = |v: i64, lo: i64, hi: i64| (if v < 0 { v + n } else { v }).clamp(lo, hi);
    let mut idx = Vec::new();
    if step > 0 {
        let (s, e) = (start.map_or(0, |v| norm(v, 0, n)), stop.map_or(n, |v| norm(v, 0, n)));
        let mut i = s;
        while i < e {
            idx.push(i as usize);
            i += step;
        }
    } else {
        let (s, e) = (start.map_or(n - 1, |v| norm(v, -1, n - 1)), stop.map_or(-1, |v| norm(v, -1, n - 1)));
        let mut i = s;
        while i > e {
            idx.push(i as usize);
            i += step;
        }
    }
    idx
}

pub fn tojson(v: &Val, indent: Option<usize>, sort_keys: bool, ensure_ascii: bool) -> String {
    fn esc(s: &str, ascii: bool, out: &mut String) {
        out.push('"');
        for c in s.chars() {
            match c {
                '"' => out.push_str("\\\""),
                '\\' => out.push_str("\\\\"),
                '\n' => out.push_str("\\n"),
                '\r' => out.push_str("\\r"),
                '\t' => out.push_str("\\t"),
                '\u{8}' => out.push_str("\\b"),
                '\u{c}' => out.push_str("\\f"),
                c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
                c if ascii && !c.is_ascii() => {
                    let mut buf = [0u16; 2];
                    for u in c.encode_utf16(&mut buf) {
                        out.push_str(&format!("\\u{u:04x}"));
                    }
                }
                c => out.push(c),
            }
        }
        out.push('"');
    }
    fn go(v: &Val, indent: Option<usize>, level: usize, sort: bool, ascii: bool, out: &mut String) {
        let (nl, ind, item_sep) = match indent {
            Some(n) => (format!("\n{}", " ".repeat(n * (level + 1))), format!("\n{}", " ".repeat(n * level)), ",".to_string()),
            None => (String::new(), String::new(), ", ".to_string()),
        };
        match v {
            Val::Undefined | Val::None => out.push_str("null"),
            Val::Bool(b) => out.push_str(if *b { "true" } else { "false" }),
            Val::Int(i) => out.push_str(&i.to_string()),
            Val::Float(f) => out.push_str(&float_repr(*f)),
            Val::Str(s) => esc(s, ascii, out),
            Val::List(l) => {
                let l = l.borrow();
                if l.is_empty() {
                    out.push_str("[]");
                    return;
                }
                out.push('[');
                for (i, x) in l.iter().enumerate() {
                    if i > 0 {
                        out.push_str(&item_sep);
                    }
                    out.push_str(&nl);
                    go(x, indent, level + 1, sort, ascii, out);
                }
                out.push_str(&ind);
                out.push(']');
            }
            Val::Map(m) | Val::Ns(m) => {
                let mut items: Vec<(String, Val)> = m.borrow().clone();
                if sort {
                    items.sort_by(|a, b| a.0.cmp(&b.0));
                }
                if items.is_empty() {
                    out.push_str("{}");
                    return;
                }
                out.push('{');
                for (i, (k, x)) in items.iter().enumerate() {
                    if i > 0 {
                        out.push_str(&item_sep);
                    }
                    out.push_str(&nl);
                    esc(k, ascii, out);
                    out.push_str(": ");
                    go(x, indent, level + 1, sort, ascii, out);
                }
                out.push_str(&ind);
                out.push('}');
            }
            Val::Macro(_) => out.push_str("null"),
        }
    }
    let mut out = String::new();
    go(v, indent, 0, sort_keys, ensure_ascii, &mut out);
    out
}

impl Env {
    fn lookup(&self, n: &str) -> Val {
        self.scopes.iter().rev().find_map(|s| s.get(n)).cloned().unwrap_or(Val::Undefined)
    }
    fn assign(&mut self, n: &str, v: Val) {
        // assignments inside `for` bodies stay local to the loop scope (Jinja scoping); top scope otherwise
        self.scopes.last_mut().unwrap().insert(n.to_string(), v);
    }

    fn emit(&mut self, s: String) {
        if let Some(top) = self.out.last_mut() {
            top.push_str(&s);
        }
    }

    fn run(&mut self, nodes: &[Node]) -> Result<Flow> {
        for n in nodes {
            match n {
                Node::Text(t) => self.emit(t.clone()),
                Node::Print(e) => {
                    let v = self.eval(e)?;
                    self.emit(v.to_str());
                }
                Node::If(arms, other) => {
                    let mut done = false;
                    for (c, body) in arms {
                        if self.eval(c)?.truthy() {
                            match self.run(body)? {
                                Flow::Normal => {}
                                f => return Ok(f),
                            }
                            done = true;
                            break;
                        }
                    }
                    if !done {
                        match self.run(other)? {
                            Flow::Normal => {}
                            f => return Ok(f),
                        }
                    }
                }
                Node::For { targets, iter, cond, body, else_body } => {
                    let items = self.eval(iter)?.items()?;
                    let items: Vec<Val> = match cond {
                        None => items,
                        Some(c) => {
                            let mut kept = Vec::new();
                            self.scopes.push(HashMap::new());
                            for it in items {
                                self.bind(targets, it.clone())?;
                                if self.eval(c)?.truthy() {
                                    kept.push(it);
                                }
                            }
                            self.scopes.pop();
                            kept
                        }
                    };
                    if items.is_empty() {
                        match self.run(else_body)? {
                            Flow::Normal => {}
                            f => return Ok(f),
                        }
                        continue;
                    }
                    let len = items.len();
                    for (i, it) in items.into_iter().enumerate() {
                        self.scopes.push(HashMap::new());
                        self.bind(targets, it)?;
                        let lp = map(vec![
                            ("index".into(), Val::Int(i as i64 + 1)),
                            ("index0".into(), Val::Int(i as i64)),
                            ("revindex".into(), Val::Int((len - i) as i64)),
                            ("revindex0".into(), Val::Int((len - i - 1) as i64)),
                            ("first".into(), Val::Bool(i == 0)),
                            ("last".into(), Val::Bool(i + 1 == len)),
                            ("length".into(), Val::Int(len as i64)),
                        ]);
                        self.scopes.last_mut().unwrap().insert("loop".into(), lp);
                        let f = self.run(body);
                        self.scopes.pop();
                        match f? {
                            Flow::Break => break,
                            Flow::Continue | Flow::Normal => {}
                        }
                    }
                }
                Node::Set(targets, e) => {
                    let v = self.eval(e)?;
                    if targets.len() == 1 {
                        self.assign_target(&targets[0], v)?;
                    } else {
                        let items = v.items()?;
                        if items.len() != targets.len() {
                            return Err(Error::Msg("chat template: cannot unpack the value into the targets of {% set %}".into()));
                        }
                        for (t, x) in targets.iter().zip(items) {
                            self.assign_target(t, x)?;
                        }
                    }
                }
                Node::SetBlock(name, body) => {
                    self.out.push(String::new());
                    let f = self.run(body);
                    let s = self.out.pop().unwrap();
                    f?;
                    self.assign(name, Val::Str(s));
                }
                Node::Macro(m) => self.assign(&m.name, Val::Macro(m.clone())),
                Node::Break => return Ok(Flow::Break),
                Node::Continue => return Ok(Flow::Continue),
            }
        }
        Ok(Flow::Normal)
    }

    fn bind(&mut self, targets: &[String], v: Val) -> Result<()> {
        if targets.len() == 1 {
            self.scopes.last_mut().unwrap().insert(targets[0].clone(), v);
        } else {
            let items = v.items()?;
            if items.len() != targets.len() {
                return Err(Error::Msg("chat template: cannot unpack the loop item".into()));
            }
            for (t, x) in targets.iter().zip(items) {
                self.scopes.last_mut().unwrap().insert(t.clone(), x);
            }
        }
        Ok(())
    }

    fn assign_target(&mut self, t: &Expr, v: Val) -> Result<()> {
        match t {
            Expr::Name(n) => {
                // a `set` inside a loop does not leak, but a name already defined in an outer scope of the same
                // template body is updated in the scope that owns it only for top-level sets
                self.assign(n, v);
                Ok(())
            }
            Expr::Attr(base, attr) => match self.eval(base)? {
                Val::Ns(m) => {
                    let mut m = m.borrow_mut();
                    match m.iter_mut().find(|(k, _)| k == attr) {
                        Some(slot) => slot.1 = v,
                        None => m.push((attr.clone(), v)),
                    }
                    Ok(())
                }
                _ => Err(Error::Msg("chat template: attributes can only be assigned on a namespace".into())),
            },
            _ => Err(Error::Msg("chat template: unsupported {% set %} target".into())),
        }
    }

    fn eval(&mut self, e: &Expr) -> Result<Val> {
        Ok(match e {
            Expr::Lit(l) => match l {
                Lit::None => Val::None,
                Lit::Bool(b) => Val::Bool(*b),
                Lit::Int(i) => Val::Int(*i),
                Lit::Float(f) => Val::Float(*f),
                Lit::Str(s) => Val::Str(s.clone()),
            },
            Expr::Name(n) => self.lookup(n),
            Expr::List(v) => list(v.iter().map(|x| self.eval(x)).collect::<Result<_>>()?),
            Expr::Dict(v) => map(v.iter().map(|(k, x)| Ok((self.eval(k)?.to_str(), self.eval(x)?))).collect::<Result<_>>()?),
            Expr::Not(x) => Val::Bool(!self.eval(x)?.truthy()),
            Expr::And(a, b) => {
                let l = self.eval(a)?;
                if l.truthy() { self.eval(b)? } else { l }
            }
            Expr::Or(a, b) => {
                let l = self.eval(a)?;
                if l.truthy() { l } else { self.eval(b)? }
            }
            Expr::Neg(x) => match self.eval(x)? {
                Val::Int(i) => Val::Int(-i),
                Val::Float(f) => Val::Float(-f),
                other => return Err(Error::Msg(format!("chat template: cannot negate a {}", other.type_name()))),
            },
            Expr::Cond(body, cond, other) => {
                if self.eval(cond)?.truthy() {
                    self.eval(body)?
                } else {
                    match other {
                        Some(o) => self.eval(o)?,
                        None => Val::Undefined,
                    }
                }
            }
            Expr::Attr(b, a) => {
                let base = self.eval(b)?;
                self.getattr(&base, a)
            }
            Expr::Index(b, i) => {
                let base = self.eval(b)?;
                let idx = self.eval(i)?;
                self.getitem(&base, &idx)?
            }
            Expr::Slice(b, s, t, st) => {
                let base = self.eval(b)?;
                let mut get = |x: &Option<Box<Expr>>| -> Result<Option<i64>> {
                    Ok(match x {
                        None => None,
                        Some(e) => match self.eval(e)? {
                            Val::Int(i) => Some(i),
                            Val::None | Val::Undefined => None,
                            other => return Err(Error::Msg(format!("chat template: slice bounds must be integers, not {}", other.type_name()))),
                        },
                    })
                };
                let (s, t, st) = (get(s)?, get(t)?, get(st)?);
                let step = st.unwrap_or(1);
                if step == 0 {
                    return Err(Error::Msg("chat template: slice step cannot be zero".into()));
                }
                match base {
                    Val::List(l) => {
                        let l = l.borrow();
                        list(slice_bounds(l.len(), s, t, step).into_iter().map(|i| l[i].clone()).collect())
                    }
                    Val::Str(sv) => {
                        let cs: Vec<char> = sv.chars().collect();
                        Val::Str(slice_bounds(cs.len(), s, t, step).into_iter().map(|i| cs[i]).collect())
                    }
                    other => return Err(Error::Msg(format!("chat template: cannot slice a {}", other.type_name()))),
                }
            }
            Expr::Bin(op, a, b) => {
                let (x, y) = (self.eval(a)?, self.eval(b)?);
                self.binary(op, x, y)?
            }
            Expr::Test(x, name, args, neg) => {
                let v = self.eval(x)?;
                let args: Vec<Val> = args.iter().map(|a| self.eval(a)).collect::<Result<_>>()?;
                Val::Bool(self.test(name, &v, &args)? != *neg)
            }
            Expr::Filter(x, name, pos, kw) => {
                let v = self.eval(x)?;
                let pos: Vec<Val> = pos.iter().map(|a| self.eval(a)).collect::<Result<_>>()?;
                let kw: Vec<(String, Val)> = kw.iter().map(|(k, a)| Ok((k.clone(), self.eval(a)?))).collect::<Result<_>>()?;
                self.filter(name, v, &pos, &kw)?
            }
            Expr::Call(f, pos, kw) => {
                let pos_v: Vec<Val> = pos.iter().map(|a| self.eval(a)).collect::<Result<_>>()?;
                let kw_v: Vec<(String, Val)> = kw.iter().map(|(k, a)| Ok((k.clone(), self.eval(a)?))).collect::<Result<_>>()?;
                match &**f {
                    Expr::Attr(base, method) => {
                        let b = self.eval(base)?;
                        self.method(&b, method, &pos_v, &kw_v)?
                    }
                    Expr::Name(n) => {
                        let v = self.lookup(n);
                        match v {
                            Val::Macro(m) => self.call_macro(&m, pos_v, kw_v)?,
                            _ => self.global_fn(n, &pos_v, &kw_v)?,
                        }
                    }
                    _ => return Err(Error::Msg("chat template: only named functions and methods can be called".into())),
                }
            }
        })
    }

    fn call_macro(&mut self, m: &Rc<MacroDef>, pos: Vec<Val>, kw: Vec<(String, Val)>) -> Result<Val> {
        let mut scope = HashMap::new();
        for (i, (name, default)) in m.params.iter().enumerate() {
            let v = if i < pos.len() {
                pos[i].clone()
            } else if let Some((_, v)) = kw.iter().find(|(k, _)| k == name) {
                v.clone()
            } else if let Some(d) = default {
                self.eval(d)?
            } else {
                Val::Undefined
            };
            scope.insert(name.clone(), v);
        }
        self.scopes.push(scope);
        self.out.push(String::new());
        let f = self.run(&m.body);
        let s = self.out.pop().unwrap();
        self.scopes.pop();
        f?;
        Ok(Val::Str(s))
    }

    fn global_fn(&mut self, name: &str, pos: &[Val], kw: &[(String, Val)]) -> Result<Val> {
        Ok(match name {
            "raise_exception" => return Err(Error::Msg(format!("chat template error: {}", pos.first().map(Val::to_str).unwrap_or_default()))),
            "namespace" => Val::Ns(Rc::new(RefCell::new(kw.to_vec()))),
            "range" => {
                let n: Vec<i64> = pos.iter().filter_map(|v| if let Val::Int(i) = v { Some(*i) } else { None }).collect();
                let (s, e, st) = match n.as_slice() {
                    [e] => (0, *e, 1),
                    [s, e] => (*s, *e, 1),
                    [s, e, st] => (*s, *e, *st),
                    _ => return Err(Error::Msg("chat template: range expects 1 to 3 integers".into())),
                };
                let mut v = Vec::new();
                let mut i = s;
                while (st > 0 && i < e) || (st < 0 && i > e) {
                    v.push(Val::Int(i));
                    i += st;
                }
                list(v)
            }
            "dict" => map(kw.to_vec()),
            "strftime_now" => Val::Str(strftime(pos.first().map(Val::to_str).unwrap_or_default().as_str())),
            "len" => Val::Int(pos.first().map_or(0, |v| v.items().map_or(0, |i| i.len())) as i64),
            "str" => Val::Str(pos.first().map(Val::to_str).unwrap_or_default()),
            "int" => Val::Int(pos.first().and_then(|v| v.num().map(|f| f as i64).or_else(|| v.to_str().trim().parse().ok())).unwrap_or(0)),
            other => return Err(Error::Msg(format!("chat template: unknown function `{other}`"))),
        })
    }

    fn getattr(&self, base: &Val, a: &str) -> Val {
        match base {
            Val::Map(m) | Val::Ns(m) => m.borrow().iter().find(|(k, _)| k == a).map(|(_, v)| v.clone()).unwrap_or(Val::Undefined),
            _ => Val::Undefined,
        }
    }

    fn getitem(&self, base: &Val, idx: &Val) -> Result<Val> {
        Ok(match (base, idx) {
            (Val::List(l), Val::Int(i)) => {
                let l = l.borrow();
                let k = if *i < 0 { *i + l.len() as i64 } else { *i };
                if k < 0 || k as usize >= l.len() { Val::Undefined } else { l[k as usize].clone() }
            }
            (Val::Str(s), Val::Int(i)) => {
                let cs: Vec<char> = s.chars().collect();
                let k = if *i < 0 { *i + cs.len() as i64 } else { *i };
                if k < 0 || k as usize >= cs.len() { Val::Undefined } else { Val::Str(cs[k as usize].to_string()) }
            }
            (Val::Map(m) | Val::Ns(m), Val::Str(k)) => m.borrow().iter().find(|(kk, _)| kk == k).map(|(_, v)| v.clone()).unwrap_or(Val::Undefined),
            (Val::Undefined, _) => Val::Undefined,
            _ => Val::Undefined,
        })
    }

    fn binary(&self, op: &str, x: Val, y: Val) -> Result<Val> {
        use std::cmp::Ordering::*;
        Ok(match op {
            "==" => Val::Bool(x.eq(&y)),
            "!=" => Val::Bool(!x.eq(&y)),
            "<" => Val::Bool(int_cmp(&x, &y)? == Less),
            ">" => Val::Bool(int_cmp(&x, &y)? == Greater),
            "<=" => Val::Bool(int_cmp(&x, &y)? != Greater),
            ">=" => Val::Bool(int_cmp(&x, &y)? != Less),
            "in" | "not in" => {
                let found = match &y {
                    Val::Str(s) => s.contains(&x.to_str()),
                    Val::List(l) => l.borrow().iter().any(|v| v.eq(&x)),
                    Val::Map(m) | Val::Ns(m) => m.borrow().iter().any(|(k, _)| *k == x.to_str()),
                    Val::Undefined | Val::None => false,
                    other => return Err(Error::Msg(format!("chat template: `in` needs a container, not a {}", other.type_name()))),
                };
                Val::Bool(found == (op == "in"))
            }
            "~" => Val::Str(format!("{}{}", x.to_str(), y.to_str())),
            "+" => match (&x, &y) {
                (Val::Str(a), Val::Str(b)) => Val::Str(format!("{a}{b}")),
                (Val::List(a), Val::List(b)) => list(a.borrow().iter().chain(b.borrow().iter()).cloned().collect()),
                (Val::Int(a), Val::Int(b)) => Val::Int(a + b),
                _ => match (x.num(), y.num()) {
                    (Some(a), Some(b)) => Val::Float(a + b),
                    _ => return Err(Error::Msg(format!("chat template: cannot add a {} and a {}", x.type_name(), y.type_name()))),
                },
            },
            "-" | "*" | "/" | "//" | "%" | "**" => match (&x, &y) {
                (Val::Int(a), Val::Int(b)) if op != "/" => match op {
                    "-" => Val::Int(a - b),
                    "*" => Val::Int(a * b),
                    "//" if *b != 0 => Val::Int(a.div_euclid(*b) - if (a.rem_euclid(*b) != 0) && (*b < 0) { 1 } else { 0 }),
                    "%" if *b != 0 => Val::Int(a.rem_euclid(*b)),
                    "**" if *b >= 0 => Val::Int(a.pow(*b as u32)),
                    _ => return Err(Error::Msg("chat template: invalid integer arithmetic".into())),
                },
                (Val::Str(s), Val::Int(n)) if op == "*" => Val::Str(s.repeat((*n).max(0) as usize)),
                _ => match (x.num(), y.num()) {
                    (Some(a), Some(b)) => Val::Float(match op {
                        "-" => a - b,
                        "*" => a * b,
                        "/" => a / b,
                        "//" => (a / b).floor(),
                        "%" => a - b * (a / b).floor(),
                        _ => a.powf(b),
                    }),
                    _ => return Err(Error::Msg(format!("chat template: cannot apply `{op}` to a {} and a {}", x.type_name(), y.type_name()))),
                },
            },
            other => return Err(Error::Msg(format!("chat template: unknown operator {other}"))),
        })
    }

    fn test(&self, name: &str, v: &Val, args: &[Val]) -> Result<bool> {
        Ok(match name {
            "defined" => !matches!(v, Val::Undefined),
            "undefined" => matches!(v, Val::Undefined),
            "none" => matches!(v, Val::None),
            "string" => matches!(v, Val::Str(_)),
            "number" => matches!(v, Val::Int(_) | Val::Float(_)),
            "integer" => matches!(v, Val::Int(_)),
            "float" => matches!(v, Val::Float(_)),
            "boolean" => matches!(v, Val::Bool(_)),
            "mapping" => matches!(v, Val::Map(_)),
            "sequence" | "iterable" => matches!(v, Val::List(_) | Val::Str(_) | Val::Map(_)),
            "true" => matches!(v, Val::Bool(true)),
            "false" => matches!(v, Val::Bool(false)),
            "lower" => matches!(v, Val::Str(s) if *s == s.to_lowercase()),
            "upper" => matches!(v, Val::Str(s) if *s == s.to_uppercase()),
            "even" => matches!(v, Val::Int(i) if i % 2 == 0),
            "odd" => matches!(v, Val::Int(i) if i % 2 != 0),
            "divisibleby" => match (v, args.first()) {
                (Val::Int(a), Some(Val::Int(b))) if *b != 0 => a % b == 0,
                _ => false,
            },
            "eq" | "equalto" | "==" => args.first().is_some_and(|a| v.eq(a)),
            "ne" | "!=" => args.first().is_some_and(|a| !v.eq(a)),
            "in" => args.first().is_some_and(|a| self.binary("in", v.clone(), a.clone()).map(|r| r.truthy()).unwrap_or(false)),
            other => return Err(Error::Msg(format!("chat template: unknown test `{other}`"))),
        })
    }

    fn kwarg<'a>(kw: &'a [(String, Val)], name: &str) -> Option<&'a Val> {
        kw.iter().find(|(k, _)| k == name).map(|(_, v)| v)
    }

    fn filter(&mut self, name: &str, v: Val, pos: &[Val], kw: &[(String, Val)]) -> Result<Val> {
        Ok(match name {
            "trim" | "strip" => match pos.first() {
                Some(Val::Str(chars)) => Val::Str(v.to_str().trim_matches(|c| chars.contains(c)).to_string()),
                _ => Val::Str(v.to_str().trim().to_string()),
            },
            "length" | "count" => Val::Int(match &v {
                Val::Str(s) => s.chars().count(),
                other => other.items()?.len(),
            } as i64),
            "lower" => Val::Str(v.to_str().to_lowercase()),
            "upper" => Val::Str(v.to_str().to_uppercase()),
            "capitalize" => {
                let s = v.to_str();
                let mut c = s.chars();
                Val::Str(match c.next() {
                    Some(f) => f.to_uppercase().collect::<String>() + &c.as_str().to_lowercase(),
                    None => String::new(),
                })
            }
            "title" => Val::Str(
                v.to_str()
                    .split(' ')
                    .map(|w| {
                        let mut c = w.chars();
                        match c.next() {
                            Some(f) => f.to_uppercase().collect::<String>() + &c.as_str().to_lowercase(),
                            None => String::new(),
                        }
                    })
                    .collect::<Vec<_>>()
                    .join(" "),
            ),
            "string" => Val::Str(v.to_str()),
            "int" => match &v {
                Val::Int(_) => v.clone(),
                Val::Float(f) => Val::Int(*f as i64),
                Val::Bool(b) => Val::Int(*b as i64),
                Val::Str(s) => Val::Int(s.trim().parse::<i64>().ok().or_else(|| s.trim().parse::<f64>().ok().map(|f| f as i64)).unwrap_or(pos.first().and_then(|d| if let Val::Int(i) = d { Some(*i) } else { None }).unwrap_or(0))),
                _ => Val::Int(0),
            },
            "float" => Val::Float(v.num().or_else(|| v.to_str().trim().parse().ok()).unwrap_or(0.0)),
            "abs" => match v {
                Val::Int(i) => Val::Int(i.abs()),
                Val::Float(f) => Val::Float(f.abs()),
                other => other,
            },
            "round" => Val::Float((v.num().unwrap_or(0.0) * 10f64.powi(pos.first().and_then(Val::num).unwrap_or(0.0) as i32)).round() / 10f64.powi(pos.first().and_then(Val::num).unwrap_or(0.0) as i32)),
            "list" => list(v.items()?),
            "first" => v.items()?.first().cloned().unwrap_or(Val::Undefined),
            "last" => v.items()?.last().cloned().unwrap_or(Val::Undefined),
            "reverse" => match &v {
                Val::Str(s) => Val::Str(s.chars().rev().collect()),
                other => list(other.items()?.into_iter().rev().collect()),
            },
            "join" => {
                let sep = pos.first().map(Val::to_str).unwrap_or_default();
                let items = v.items()?;
                let attr = Self::kwarg(kw, "attribute").map(Val::to_str);
                Val::Str(items.iter().map(|i| match &attr { Some(a) => self.getattr(i, a).to_str(), None => i.to_str() }).collect::<Vec<_>>().join(&sep))
            }
            "default" | "d" => {
                let boolean = pos.get(1).is_some_and(Val::truthy) || Self::kwarg(kw, "boolean").is_some_and(Val::truthy);
                let dflt = pos.first().cloned().or_else(|| Self::kwarg(kw, "default_value").cloned()).unwrap_or(Val::Str(String::new()));
                if matches!(v, Val::Undefined) || (boolean && !v.truthy()) { dflt } else { v }
            }
            "tojson" => {
                let indent = pos.first().or_else(|| Self::kwarg(kw, "indent")).and_then(|i| if let Val::Int(n) = i { Some(*n as usize) } else { None });
                let sort = Self::kwarg(kw, "sort_keys").is_some_and(Val::truthy);
                let ascii = Self::kwarg(kw, "ensure_ascii").is_some_and(Val::truthy);
                Val::Str(tojson(&v, indent, sort, ascii))
            }
            "replace" => {
                let (a, b) = (pos.first().map(Val::to_str).unwrap_or_default(), pos.get(1).map(Val::to_str).unwrap_or_default());
                Val::Str(v.to_str().replace(&a, &b))
            }
            "safe" | "forceescape" | "e" | "escape" => v,
            "items" => match &v {
                Val::Map(m) | Val::Ns(m) => list(m.borrow().iter().map(|(k, x)| list(vec![Val::Str(k.clone()), x.clone()])).collect()),
                _ => list(vec![]),
            },
            "map" => {
                let items = v.items()?;
                let attr = Self::kwarg(kw, "attribute").map(Val::to_str);
                match (attr, pos.first()) {
                    (Some(a), _) => list(items.iter().map(|i| self.getattr(i, &a)).collect()),
                    (None, Some(Val::Str(f))) => {
                        let mut out = Vec::new();
                        for i in items {
                            out.push(self.filter(f, i, &pos[1..], kw)?);
                        }
                        list(out)
                    }
                    _ => list(items),
                }
            }
            "select" | "reject" | "selectattr" | "rejectattr" => {
                let items = v.items()?;
                let want = name == "select" || name == "selectattr";
                let by_attr = name.ends_with("attr");
                let (attr, test_idx) = if by_attr { (pos.first().map(Val::to_str), 1) } else { (None, 0) };
                let mut out = Vec::new();
                for i in items {
                    let subject = match &attr { Some(a) => self.getattr(&i, a), None => i.clone() };
                    let ok = match pos.get(test_idx) {
                        Some(Val::Str(t)) => self.test(t, &subject, &pos[test_idx + 1..])?,
                        _ => subject.truthy(),
                    };
                    if ok == want {
                        out.push(i);
                    }
                }
                list(out)
            }
            "sort" => {
                let mut items = v.items()?;
                let attr = Self::kwarg(kw, "attribute").map(Val::to_str);
                items.sort_by(|a, b| {
                    let (x, y) = match &attr { Some(at) => (self.getattr(a, at), self.getattr(b, at)), None => (a.clone(), b.clone()) };
                    int_cmp(&x, &y).unwrap_or(std::cmp::Ordering::Equal)
                });
                if Self::kwarg(kw, "reverse").is_some_and(Val::truthy) {
                    items.reverse();
                }
                list(items)
            }
            "unique" => {
                let mut out: Vec<Val> = Vec::new();
                for i in v.items()? {
                    if !out.iter().any(|o| o.eq(&i)) {
                        out.push(i);
                    }
                }
                list(out)
            }
            "sum" => {
                let items = v.items()?;
                if items.iter().all(|i| matches!(i, Val::Int(_))) {
                    Val::Int(items.iter().map(|i| if let Val::Int(n) = i { *n } else { 0 }).sum())
                } else {
                    Val::Float(items.iter().filter_map(Val::num).sum())
                }
            }
            "min" | "max" => {
                let items = v.items()?;
                let mut best: Option<Val> = None;
                for i in items {
                    best = Some(match best {
                        None => i,
                        Some(b) => {
                            let o = int_cmp(&i, &b)?;
                            if (name == "min" && o == std::cmp::Ordering::Less) || (name == "max" && o == std::cmp::Ordering::Greater) { i } else { b }
                        }
                    });
                }
                best.unwrap_or(Val::Undefined)
            }
            "dictsort" => match &v {
                Val::Map(m) => {
                    let mut it: Vec<(String, Val)> = m.borrow().clone();
                    it.sort_by(|a, b| a.0.cmp(&b.0));
                    list(it.into_iter().map(|(k, x)| list(vec![Val::Str(k), x])).collect())
                }
                _ => list(vec![]),
            },
            "indent" => {
                let n = pos.first().and_then(Val::num).unwrap_or(4.0) as usize;
                let pad = " ".repeat(n);
                Val::Str(v.to_str().split('\n').enumerate().map(|(i, l)| if i == 0 || l.is_empty() { l.to_string() } else { format!("{pad}{l}") }).collect::<Vec<_>>().join("\n"))
            }
            other => return Err(Error::Msg(format!("chat template: unknown filter `{other}`"))),
        })
    }

    fn method(&mut self, base: &Val, name: &str, pos: &[Val], kw: &[(String, Val)]) -> Result<Val> {
        let arg_str = |i: usize| pos.get(i).map(Val::to_str);
        Ok(match (base, name) {
            (Val::Str(s), "strip") => Val::Str(match arg_str(0) { Some(c) => s.trim_matches(|x| c.contains(x)).to_string(), None => s.trim().to_string() }),
            (Val::Str(s), "lstrip") => Val::Str(match arg_str(0) { Some(c) => s.trim_start_matches(|x| c.contains(x)).to_string(), None => s.trim_start().to_string() }),
            (Val::Str(s), "rstrip") => Val::Str(match arg_str(0) { Some(c) => s.trim_end_matches(|x| c.contains(x)).to_string(), None => s.trim_end().to_string() }),
            (Val::Str(s), "startswith") => Val::Bool(match pos.first() {
                Some(Val::List(l)) => l.borrow().iter().any(|p| s.starts_with(&p.to_str())),
                _ => s.starts_with(&arg_str(0).unwrap_or_default()),
            }),
            (Val::Str(s), "endswith") => Val::Bool(match pos.first() {
                Some(Val::List(l)) => l.borrow().iter().any(|p| s.ends_with(&p.to_str())),
                _ => s.ends_with(&arg_str(0).unwrap_or_default()),
            }),
            (Val::Str(s), "split") => {
                let parts: Vec<Val> = match arg_str(0) {
                    Some(sep) if !sep.is_empty() => match pos.get(1) {
                        Some(Val::Int(n)) => s.splitn(*n as usize + 1, sep.as_str()).map(|x| Val::Str(x.to_string())).collect(),
                        _ => s.split(sep.as_str()).map(|x| Val::Str(x.to_string())).collect(),
                    },
                    _ => s.split_whitespace().map(|x| Val::Str(x.to_string())).collect(),
                };
                list(parts)
            }
            (Val::Str(s), "rsplit") => {
                let sep = arg_str(0).unwrap_or_else(|| " ".into());
                match pos.get(1) {
                    Some(Val::Int(n)) => {
                        let mut v: Vec<Val> = s.rsplitn(*n as usize + 1, sep.as_str()).map(|x| Val::Str(x.to_string())).collect();
                        v.reverse();
                        list(v)
                    }
                    _ => list(s.split(sep.as_str()).map(|x| Val::Str(x.to_string())).collect()),
                }
            }
            (Val::Str(s), "replace") => Val::Str(s.replace(&arg_str(0).unwrap_or_default(), &arg_str(1).unwrap_or_default())),
            (Val::Str(s), "lower") => Val::Str(s.to_lowercase()),
            (Val::Str(s), "upper") => Val::Str(s.to_uppercase()),
            (Val::Str(s), "title") | (Val::Str(s), "capitalize") => self.filter(name, Val::Str(s.clone()), &[], &[])?,
            (Val::Str(s), "find") => Val::Int(s.find(&arg_str(0).unwrap_or_default()).map_or(-1, |b| s[..b].chars().count() as i64)),
            (Val::Str(s), "count") => Val::Int(s.matches(&arg_str(0).unwrap_or_default()).count() as i64),
            (Val::Str(s), "join") => Val::Str(pos.first().map(|l| l.items()).transpose()?.unwrap_or_default().iter().map(Val::to_str).collect::<Vec<_>>().join(s)),
            (Val::Str(s), "format") => {
                let mut out = String::new();
                let mut k = 0;
                let mut it = s.chars().peekable();
                while let Some(c) = it.next() {
                    if c == '{' && it.peek() == Some(&'}') {
                        it.next();
                        out.push_str(&pos.get(k).map(Val::to_str).unwrap_or_default());
                        k += 1;
                    } else {
                        out.push(c);
                    }
                }
                Val::Str(out)
            }
            (Val::Map(m) | Val::Ns(m), "get") => {
                let k = arg_str(0).unwrap_or_default();
                m.borrow().iter().find(|(kk, _)| *kk == k).map(|(_, v)| v.clone()).unwrap_or_else(|| pos.get(1).cloned().unwrap_or(Val::None))
            }
            (Val::Map(m), "items") => list(m.borrow().iter().map(|(k, v)| list(vec![Val::Str(k.clone()), v.clone()])).collect()),
            (Val::Map(m), "keys") => list(m.borrow().iter().map(|(k, _)| Val::Str(k.clone())).collect()),
            (Val::Map(m), "values") => list(m.borrow().iter().map(|(_, v)| v.clone()).collect()),
            (Val::Map(m), "update") => {
                if let Some(Val::Map(o)) = pos.first() {
                    for (k, v) in o.borrow().iter() {
                        let mut mm = m.borrow_mut();
                        match mm.iter_mut().find(|(kk, _)| kk == k) {
                            Some(slot) => slot.1 = v.clone(),
                            None => mm.push((k.clone(), v.clone())),
                        }
                    }
                }
                Val::None
            }
            (Val::Map(m), "pop") => {
                let k = arg_str(0).unwrap_or_default();
                let mut mm = m.borrow_mut();
                match mm.iter().position(|(kk, _)| *kk == k) {
                    Some(i) => mm.remove(i).1,
                    None => pos.get(1).cloned().unwrap_or(Val::Undefined),
                }
            }
            (Val::List(l), "append") => {
                l.borrow_mut().push(pos.first().cloned().unwrap_or(Val::None));
                Val::None
            }
            (Val::List(l), "extend") => {
                let extra = pos.first().map(|v| v.items()).transpose()?.unwrap_or_default();
                l.borrow_mut().extend(extra);
                Val::None
            }
            (Val::List(l), "pop") => {
                let mut ll = l.borrow_mut();
                let i = match pos.first() { Some(Val::Int(i)) if *i < 0 => (ll.len() as i64 + i).max(0) as usize, Some(Val::Int(i)) => *i as usize, _ => ll.len().saturating_sub(1) };
                if i < ll.len() { ll.remove(i) } else { Val::Undefined }
            }
            (Val::List(l), "index") => Val::Int(l.borrow().iter().position(|x| pos.first().is_some_and(|p| x.eq(p))).map_or(-1, |i| i as i64)),
            (Val::Undefined, _) => Val::Undefined,
            (b, m) => {
                let _ = kw;
                return Err(Error::Msg(format!("chat template: a {} has no method `{m}`", b.type_name())));
            }
        })
    }
}

/// `strftime` for the handful of directives templates use (UTC).
fn strftime(fmt: &str) -> String {
    let secs = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map_or(0, |d| d.as_secs() as i64);
    let (days, rem) = (secs.div_euclid(86400), secs.rem_euclid(86400));
    // civil from days (Howard Hinnant)
    let z = days + 719468;
    let era = z.div_euclid(146097);
    let doe = z.rem_euclid(146097);
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = if m <= 2 { y + 1 } else { y };
    const MONTHS: [&str; 12] = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"];
    const DAYS: [&str; 7] = ["Thursday", "Friday", "Saturday", "Sunday", "Monday", "Tuesday", "Wednesday"];
    let mut out = String::new();
    let mut it = fmt.chars();
    while let Some(c) = it.next() {
        if c != '%' {
            out.push(c);
            continue;
        }
        match it.next() {
            Some('Y') => out.push_str(&y.to_string()),
            Some('y') => out.push_str(&format!("{:02}", y % 100)),
            Some('m') => out.push_str(&format!("{m:02}")),
            Some('d') => out.push_str(&format!("{d:02}")),
            Some('e') => out.push_str(&format!("{d:>2}")),
            Some('B') => out.push_str(MONTHS[(m - 1) as usize]),
            Some('b') => out.push_str(&MONTHS[(m - 1) as usize][..3]),
            Some('A') => out.push_str(DAYS[days.rem_euclid(7) as usize]),
            Some('a') => out.push_str(&DAYS[days.rem_euclid(7) as usize][..3]),
            Some('H') => out.push_str(&format!("{:02}", rem / 3600)),
            Some('M') => out.push_str(&format!("{:02}", rem % 3600 / 60)),
            Some('S') => out.push_str(&format!("{:02}", rem % 60)),
            Some('%') => out.push('%'),
            Some(o) => {
                out.push('%');
                out.push(o)
            }
            None => out.push('%'),
        }
    }
    out
}

/// A parsed chat template.
pub struct ChatTemplate {
    nodes: Vec<Node>,
}

impl ChatTemplate {
    pub fn new(source: &str) -> Result<Self> {
        let raw = split_template(source)?;
        let mut tp = TP { raw, i: 0 };
        let nodes = tp.nodes(&[])?;
        Ok(Self { nodes })
    }

    /// Renders with the given variables (`messages`, `bos_token`, `add_generation_prompt`, ...).
    pub fn render(&self, vars: &[(&str, Val)]) -> Result<String> {
        let mut env = Env { scopes: vec![HashMap::new()], out: vec![String::new()] };
        for (k, v) in vars {
            env.scopes[0].insert((*k).to_string(), v.clone());
        }
        env.run(&self.nodes)?;
        Ok(env.out.pop().unwrap_or_default())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn render(t: &str, vars: &[(&str, Val)]) -> String {
        ChatTemplate::new(t).unwrap().render(vars).unwrap()
    }

    #[test]
    fn chatml_template() {
        let t = "{% for m in messages %}{{ '<|im_start|>' + m['role'] + '\\n' + m['content'] + '<|im_end|>' + '\\n' }}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\\n' }}{% endif %}";
        let msgs = Val::from_json(&Json::parse(r#"[{"role":"user","content":"hi"},{"role":"assistant","content":"yo"}]"#).unwrap());
        let out = render(t, &[("messages", msgs), ("add_generation_prompt", Val::Bool(true))]);
        assert_eq!(out, "<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\nyo<|im_end|>\n<|im_start|>assistant\n");
    }

    #[test]
    fn whitespace_control_loops_and_namespaces() {
        let t = "{%- set ns = namespace(n=0) -%}\n{%- for x in [1, 2, 3] %}\n  {%- set ns.n = ns.n + x %}\n{{ loop.index }}:{{ x }}{{ ',' if not loop.last }}\n{%- endfor %}\n= {{ ns.n }}";
        assert_eq!(render(t, &[]), "1:1,2:2,3:3= 6");
        assert_eq!(render("{{ [1, 2, 3][::-1] | join('-') }} {{ 'abc'[1:] }} {{ 7 // 2 }} {{ -7 // 2 }} {{ 7 % 3 }}", &[]), "3-2-1 bc 3 -4 1");
        assert_eq!(render("{{ {'a': 1, 'b': [1, 'x']} | tojson }} {{ none is none }} {{ x is defined }}", &[]), "{\"a\": 1, \"b\": [1, \"x\"]} True False");
        assert!(ChatTemplate::new("{{ raise_exception('bad') }}").unwrap().render(&[]).is_err());
    }
}
