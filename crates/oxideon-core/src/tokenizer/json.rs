//! A minimal JSON reader (enough for `tokenizer.json`, `tokenizer_config.json`, `vocab.json`, `special_tokens_map.json`).
//! Objects keep their key order; numbers are kept as `f64`.

use crate::error::{Error, Result};

#[derive(Debug, Clone, PartialEq)]
pub enum Json {
    Null,
    Bool(bool),
    Num(f64),
    Str(String),
    Arr(Vec<Json>),
    Obj(Vec<(String, Json)>),
}

impl Json {
    pub fn parse(text: &str) -> Result<Json> {
        let mut p = P { b: text.as_bytes(), i: 0, depth: 0 };
        p.ws();
        let v = p.value()?;
        p.ws();
        if p.i != p.b.len() {
            return Err(p.err("trailing characters"));
        }
        Ok(v)
    }

    /// Looks a key up in an object (`None` for other kinds or a missing key; the last duplicate wins).
    pub fn get(&self, key: &str) -> Option<&Json> {
        match self {
            Json::Obj(kv) => kv.iter().rev().find(|(k, _)| k == key).map(|(_, v)| v),
            _ => None,
        }
    }
    pub fn as_str(&self) -> Option<&str> {
        if let Json::Str(s) = self { Some(s) } else { None }
    }
    pub fn as_f64(&self) -> Option<f64> {
        if let Json::Num(n) = self { Some(*n) } else { None }
    }
    pub fn as_u64(&self) -> Option<u64> {
        self.as_f64().filter(|n| *n >= 0.0 && n.fract() == 0.0).map(|n| n as u64)
    }
    pub fn as_bool(&self) -> Option<bool> {
        if let Json::Bool(b) = self { Some(*b) } else { None }
    }
    pub fn as_array(&self) -> Option<&[Json]> {
        if let Json::Arr(a) = self { Some(a) } else { None }
    }
    pub fn as_object(&self) -> Option<&[(String, Json)]> {
        if let Json::Obj(o) = self { Some(o) } else { None }
    }
    pub fn is_null(&self) -> bool {
        matches!(self, Json::Null)
    }

    /// Compact JSON text.
    pub fn to_text(&self) -> String {
        let mut out = String::new();
        self.write(&mut out);
        out
    }

    fn write(&self, out: &mut String) {
        match self {
            Json::Null => out.push_str("null"),
            Json::Bool(b) => out.push_str(if *b { "true" } else { "false" }),
            Json::Num(n) => {
                if n.fract() == 0.0 && n.abs() < 1e15 {
                    out.push_str(&format!("{}", *n as i64));
                } else {
                    out.push_str(&format!("{n}"));
                }
            }
            Json::Str(s) => write_str(s, out),
            Json::Arr(a) => {
                out.push('[');
                for (i, v) in a.iter().enumerate() {
                    if i > 0 {
                        out.push(',');
                    }
                    v.write(out);
                }
                out.push(']');
            }
            Json::Obj(o) => {
                out.push('{');
                for (i, (k, v)) in o.iter().enumerate() {
                    if i > 0 {
                        out.push(',');
                    }
                    write_str(k, out);
                    out.push(':');
                    v.write(out);
                }
                out.push('}');
            }
        }
    }

    /// Replaces (or appends) a key of an object; no-op for other kinds.
    pub fn set(&mut self, key: &str, value: Json) {
        if let Json::Obj(o) = self {
            match o.iter_mut().find(|(k, _)| k == key) {
                Some(slot) => slot.1 = value,
                None => o.push((key.to_string(), value)),
            }
        }
    }
}

fn write_str(s: &str, out: &mut String) {
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out.push('"');
}

struct P<'a> {
    b: &'a [u8],
    i: usize,
    depth: usize,
}

impl P<'_> {
    fn err(&self, msg: &str) -> Error {
        Error::Msg(format!("json: {msg} at byte {}", self.i))
    }
    fn ws(&mut self) {
        while self.i < self.b.len() && matches!(self.b[self.i], b' ' | b'\t' | b'\n' | b'\r') {
            self.i += 1;
        }
    }
    fn lit(&mut self, word: &str, v: Json) -> Result<Json> {
        if self.b[self.i..].starts_with(word.as_bytes()) {
            self.i += word.len();
            Ok(v)
        } else {
            Err(self.err("bad literal"))
        }
    }
    fn value(&mut self) -> Result<Json> {
        match self.b.get(self.i).copied() {
            None => Err(self.err("unexpected end")),
            Some(b'n') => self.lit("null", Json::Null),
            Some(b't') => self.lit("true", Json::Bool(true)),
            Some(b'f') => self.lit("false", Json::Bool(false)),
            Some(b'"') => Ok(Json::Str(self.string()?)),
            Some(b'[') => {
                self.enter()?;
                self.i += 1;
                let mut v = Vec::new();
                self.ws();
                if self.b.get(self.i) == Some(&b']') {
                    self.i += 1;
                } else {
                    loop {
                        self.ws();
                        v.push(self.value()?);
                        self.ws();
                        match self.b.get(self.i) {
                            Some(b',') => self.i += 1,
                            Some(b']') => {
                                self.i += 1;
                                break;
                            }
                            _ => return Err(self.err("expected , or ]")),
                        }
                    }
                }
                self.depth -= 1;
                Ok(Json::Arr(v))
            }
            Some(b'{') => {
                self.enter()?;
                self.i += 1;
                let mut kv = Vec::new();
                self.ws();
                if self.b.get(self.i) == Some(&b'}') {
                    self.i += 1;
                } else {
                    loop {
                        self.ws();
                        if self.b.get(self.i) != Some(&b'"') {
                            return Err(self.err("expected object key"));
                        }
                        let k = self.string()?;
                        self.ws();
                        if self.b.get(self.i) != Some(&b':') {
                            return Err(self.err("expected :"));
                        }
                        self.i += 1;
                        self.ws();
                        kv.push((k, self.value()?));
                        self.ws();
                        match self.b.get(self.i) {
                            Some(b',') => self.i += 1,
                            Some(b'}') => {
                                self.i += 1;
                                break;
                            }
                            _ => return Err(self.err("expected , or }")),
                        }
                    }
                }
                self.depth -= 1;
                Ok(Json::Obj(kv))
            }
            Some(c) if c == b'-' || c.is_ascii_digit() => {
                let st = self.i;
                while self.i < self.b.len() && matches!(self.b[self.i], b'-' | b'+' | b'.' | b'e' | b'E' | b'0'..=b'9') {
                    self.i += 1;
                }
                std::str::from_utf8(&self.b[st..self.i]).ok().and_then(|s| s.parse::<f64>().ok()).map(Json::Num).ok_or_else(|| self.err("bad number"))
            }
            Some(_) => Err(self.err("unexpected character")),
        }
    }
    fn enter(&mut self) -> Result<()> {
        self.depth += 1;
        if self.depth > 256 { Err(self.err("nesting too deep")) } else { Ok(()) }
    }
    fn hex4(&mut self) -> Result<u32> {
        let h = self.b.get(self.i..self.i + 4).and_then(|s| std::str::from_utf8(s).ok()).and_then(|s| u32::from_str_radix(s, 16).ok());
        let v = h.ok_or_else(|| self.err("bad \\u escape"))?;
        self.i += 4;
        Ok(v)
    }
    fn string(&mut self) -> Result<String> {
        self.i += 1; // opening quote
        let mut out: Vec<u8> = Vec::new();
        loop {
            let c = *self.b.get(self.i).ok_or_else(|| self.err("unterminated string"))?;
            self.i += 1;
            match c {
                b'"' => break,
                b'\\' => {
                    let e = *self.b.get(self.i).ok_or_else(|| self.err("bad escape"))?;
                    self.i += 1;
                    let ch = match e {
                        b'"' => '"',
                        b'\\' => '\\',
                        b'/' => '/',
                        b'b' => '\u{8}',
                        b'f' => '\u{c}',
                        b'n' => '\n',
                        b'r' => '\r',
                        b't' => '\t',
                        b'u' => {
                            let hi = self.hex4()?;
                            let cp = if (0xD800..0xDC00).contains(&hi) {
                                if self.b.get(self.i..self.i + 2) == Some(b"\\u") {
                                    self.i += 2;
                                    let lo = self.hex4()?;
                                    if !(0xDC00..0xE000).contains(&lo) {
                                        return Err(self.err("bad surrogate pair"));
                                    }
                                    0x10000 + ((hi - 0xD800) << 10) + (lo - 0xDC00)
                                } else {
                                    return Err(self.err("lone surrogate"));
                                }
                            } else {
                                hi
                            };
                            char::from_u32(cp).ok_or_else(|| self.err("bad code point"))?
                        }
                        _ => return Err(self.err("bad escape")),
                    };
                    let mut buf = [0u8; 4];
                    out.extend_from_slice(ch.encode_utf8(&mut buf).as_bytes());
                }
                c => out.push(c),
            }
        }
        String::from_utf8(out).map_err(|_| self.err("invalid UTF-8"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_nested_values() {
        let j = Json::parse(r#" {"a": [1, 2.5e1, -3, true, null], "s": "x\"é😀\n", "o": {}} "#).unwrap();
        assert_eq!(j.get("a").unwrap().as_array().unwrap().len(), 5);
        assert_eq!(j.get("a").unwrap().as_array().unwrap()[1].as_f64(), Some(25.0));
        assert_eq!(j.get("s").unwrap().as_str(), Some("x\"é😀\n"));
        assert_eq!(Json::parse(&j.to_text()).unwrap(), j);
        assert!(Json::parse("{\"a\":").is_err() && Json::parse("[1,]").is_err() && Json::parse("1 2").is_err());
    }
}
