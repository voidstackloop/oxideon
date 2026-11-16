//! Dependency-free data readers/writers: RFC-4180 [`parse_csv`] / [`write_csv`] with a small [`Table`] type,
//! [`parse_libsvm`] / [`write_libsvm`] (sparse `label idx:val` lines), and the IDX binary format used by MNIST
//! ([`parse_idx`] / [`write_idx`]).

use crate::error::{Error, Result};

/// Rows of string cells with optional header names.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct Table {
    pub header: Option<Vec<String>>,
    pub rows: Vec<Vec<String>>,
}

impl Table {
    /// Index of a named column.
    pub fn column_index(&self, name: &str) -> Option<usize> {
        self.header.as_ref()?.iter().position(|h| h == name)
    }

    /// All values of one column as `f64`; empty cells become `NaN`, other non-numeric cells are an error.
    pub fn numeric_column(&self, idx: usize) -> Result<Vec<f64>> {
        self.rows
            .iter()
            .enumerate()
            .map(|(r, row)| {
                let cell = row.get(idx).ok_or_else(|| Error::Msg(format!("row {r} has no column {idx}")))?.trim();
                if cell.is_empty() { Ok(f64::NAN) } else { cell.parse::<f64>().map_err(|_| Error::Msg(format!("row {r}, column {idx}: {cell:?} is not a number"))) }
            })
            .collect()
    }

    /// All columns as a numeric matrix (rows = samples).
    pub fn to_matrix(&self) -> Result<Vec<Vec<f64>>> {
        let width = self.rows.first().map_or(0, Vec::len);
        let cols: Vec<Vec<f64>> = (0..width).map(|c| self.numeric_column(c)).collect::<Result<_>>()?;
        Ok((0..self.rows.len()).map(|r| cols.iter().map(|c| c[r]).collect()).collect())
    }
}

/// Parses CSV text: quoted fields may contain the delimiter, newlines and doubled quotes. Every row must have the
/// same number of fields. A trailing blank line is ignored.
pub fn parse_csv(text: &str, delimiter: char, has_header: bool) -> Result<Table> {
    let mut rows: Vec<Vec<String>> = Vec::new();
    let (mut row, mut field) = (Vec::new(), String::new());
    let (mut in_quotes, mut at_field_start, mut seen_any) = (false, true, false);
    let mut chars = text.chars().peekable();
    while let Some(c) = chars.next() {
        seen_any = true;
        if in_quotes {
            if c == '"' {
                if chars.peek() == Some(&'"') {
                    field.push('"');
                    chars.next();
                } else {
                    in_quotes = false;
                }
            } else {
                field.push(c);
            }
            continue;
        }
        match c {
            '"' if at_field_start => {
                in_quotes = true;
                at_field_start = false;
            }
            c if c == delimiter => {
                row.push(std::mem::take(&mut field));
                at_field_start = true;
            }
            '\r' => {}
            '\n' => {
                row.push(std::mem::take(&mut field));
                rows.push(std::mem::take(&mut row));
                at_field_start = true;
            }
            c => {
                field.push(c);
                at_field_start = false;
            }
        }
    }
    if in_quotes {
        return Err(Error::Msg("parse_csv: unterminated quoted field".into()));
    }
    if seen_any && (!field.is_empty() || !row.is_empty()) {
        row.push(field);
        rows.push(row);
    }
    let rows: Vec<Vec<String>> = rows.into_iter().filter(|r| !(r.len() == 1 && r[0].is_empty())).collect();
    if let Some(w) = rows.first().map(Vec::len) {
        if let Some(bad) = rows.iter().position(|r| r.len() != w) {
            return Err(Error::Msg(format!("parse_csv: row {} has {} fields, expected {w}", bad + 1, rows[bad].len())));
        }
    }
    let mut table = Table { header: None, rows };
    if has_header && !table.rows.is_empty() {
        table.header = Some(table.rows.remove(0));
    }
    Ok(table)
}

/// Serializes a table, quoting fields that need it.
pub fn write_csv(table: &Table, delimiter: char) -> String {
    let quote = |f: &str| -> String {
        if f.contains(delimiter) || f.contains('"') || f.contains('\n') || f.contains('\r') { format!("\"{}\"", f.replace('"', "\"\"")) } else { f.to_string() }
    };
    let line = |r: &[String]| r.iter().map(|f| quote(f)).collect::<Vec<_>>().join(&delimiter.to_string());
    let mut out = String::new();
    if let Some(h) = &table.header {
        out.push_str(&line(h));
        out.push('\n');
    }
    for r in &table.rows {
        out.push_str(&line(r));
        out.push('\n');
    }
    out
}

/// One sparse example of the libsvm format.
#[derive(Debug, Clone, PartialEq)]
pub struct SparseExample {
    pub label: f64,
    /// `(feature index, value)` with 1-based indices as in the file, ascending.
    pub features: Vec<(usize, f64)>,
}

pub fn parse_libsvm(text: &str) -> Result<Vec<SparseExample>> {
    text.lines()
        .enumerate()
        .filter(|(_, l)| !l.trim().is_empty() && !l.trim_start().starts_with('#'))
        .map(|(n, line)| {
            let mut parts = line.split('#').next().unwrap().split_whitespace();
            let label = parts.next().and_then(|t| t.parse::<f64>().ok()).ok_or_else(|| Error::Msg(format!("parse_libsvm: line {}: bad label", n + 1)))?;
            let mut features = Vec::new();
            for p in parts {
                let (i, v) = p.split_once(':').ok_or_else(|| Error::Msg(format!("parse_libsvm: line {}: expected idx:value, got {p:?}", n + 1)))?;
                let (i, v): (usize, f64) = (i.parse().map_err(|_| Error::Msg(format!("parse_libsvm: line {}: bad index {i:?}", n + 1)))?, v.parse().map_err(|_| Error::Msg(format!("parse_libsvm: line {}: bad value {v:?}", n + 1)))?);
                if i == 0 || features.last().is_some_and(|l: &(usize, f64)| l.0 >= i) {
                    return Err(Error::Msg(format!("parse_libsvm: line {}: indices must be >= 1 and strictly ascending", n + 1)));
                }
                features.push((i, v));
            }
            Ok(SparseExample { label, features })
        })
        .collect()
}

pub fn write_libsvm(examples: &[SparseExample]) -> String {
    examples.iter().map(|e| format!("{} {}\n", e.label, e.features.iter().map(|(i, v)| format!("{i}:{v}")).collect::<Vec<_>>().join(" "))).collect()
}

/// Densifies libsvm examples into a matrix with `n_features` columns (0-based after the 1-based file indices).
pub fn libsvm_to_dense(examples: &[SparseExample], n_features: usize) -> Result<(Vec<Vec<f64>>, Vec<f64>)> {
    let mut x = vec![vec![0.0; n_features]; examples.len()];
    for (r, e) in examples.iter().enumerate() {
        for &(i, v) in &e.features {
            if i > n_features {
                return Err(Error::Msg(format!("libsvm_to_dense: feature index {i} exceeds n_features = {n_features}")));
            }
            x[r][i - 1] = v;
        }
    }
    Ok((x, examples.iter().map(|e| e.label).collect()))
}

/// An IDX array (dimensions plus row-major values converted to `f64`).
#[derive(Debug, Clone, PartialEq)]
pub struct IdxArray {
    pub dims: Vec<usize>,
    pub data: Vec<f64>,
}

/// Parses the IDX format: magic `00 00 <type> <ndims>`, big-endian `u32` dimension sizes, then the payload.
pub fn parse_idx(bytes: &[u8]) -> Result<IdxArray> {
    if bytes.len() < 4 || bytes[0] != 0 || bytes[1] != 0 {
        return Err(Error::Msg("parse_idx: bad magic number".into()));
    }
    let (ty, nd) = (bytes[2], bytes[3] as usize);
    let width = match ty {
        0x08 | 0x09 => 1,
        0x0B => 2,
        0x0C | 0x0D => 4,
        0x0E => 8,
        _ => return Err(Error::Msg(format!("parse_idx: unsupported element type 0x{ty:02x}"))),
    };
    let header = 4 + 4 * nd;
    if bytes.len() < header {
        return Err(Error::Msg("parse_idx: truncated header".into()));
    }
    let dims: Vec<usize> = (0..nd).map(|i| u32::from_be_bytes(bytes[4 + 4 * i..8 + 4 * i].try_into().unwrap()) as usize).collect();
    let count = dims.iter().try_fold(1usize, |a, &d| a.checked_mul(d)).ok_or_else(|| Error::Msg("parse_idx: dimension product overflows".into()))?;
    if bytes.len() - header != count.checked_mul(width).ok_or_else(|| Error::Msg("parse_idx: payload size overflows".into()))? {
        return Err(Error::Msg(format!("parse_idx: payload is {} bytes, expected {}", bytes.len() - header, count * width)));
    }
    let data = bytes[header..]
        .chunks(width)
        .map(|c| match ty {
            0x08 => c[0] as f64,
            0x09 => c[0] as i8 as f64,
            0x0B => i16::from_be_bytes([c[0], c[1]]) as f64,
            0x0C => i32::from_be_bytes(c.try_into().unwrap()) as f64,
            0x0D => f32::from_be_bytes(c.try_into().unwrap()) as f64,
            _ => f64::from_be_bytes(c.try_into().unwrap()),
        })
        .collect();
    Ok(IdxArray { dims, data })
}

/// Encodes values as an unsigned-byte (`0x08`) IDX array; every value must be an integer in `0..=255`.
pub fn write_idx(dims: &[usize], data: &[f64]) -> Result<Vec<u8>> {
    if dims.iter().product::<usize>() != data.len() || dims.len() > 255 {
        return Err(Error::Msg("write_idx: dims do not match the data length".into()));
    }
    if data.iter().any(|&v| v.fract() != 0.0 || !(0.0..=255.0).contains(&v)) {
        return Err(Error::Msg("write_idx: values must be integers in 0..=255".into()));
    }
    let mut out = vec![0, 0, 0x08, dims.len() as u8];
    dims.iter().for_each(|&d| out.extend((d as u32).to_be_bytes()));
    out.extend(data.iter().map(|&v| v as u8));
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn csv_quoting_and_round_trip() {
        let text = "name,note,score\r\nAda,\"likes, commas\",9.5\nBob,\"multi\nline \"\"quoted\"\"\",\nCy,plain,7\n";
        let t = parse_csv(text, ',', true).unwrap();
        assert_eq!(t.header.as_ref().unwrap(), &["name", "note", "score"]);
        assert_eq!(t.rows.len(), 3);
        assert_eq!(t.rows[0][1], "likes, commas");
        assert_eq!(t.rows[1][1], "multi\nline \"quoted\"");
        let scores = t.numeric_column(t.column_index("score").unwrap()).unwrap();
        assert_eq!(scores[0], 9.5);
        assert!(scores[1].is_nan());
        assert_eq!(scores[2], 7.0);
        assert!(t.numeric_column(0).is_err());
        // round trip through write_csv
        let back = parse_csv(&write_csv(&t, ','), ',', true).unwrap();
        assert_eq!(back, t);
        // other delimiter, no header, ragged rows rejected, unterminated quote rejected
        let tsv = parse_csv("1\t2\n3\t4", '\t', false).unwrap();
        assert_eq!(tsv.to_matrix().unwrap(), vec![vec![1.0, 2.0], vec![3.0, 4.0]]);
        assert!(parse_csv("a,b\n1\n", ',', true).is_err());
        assert!(parse_csv("a,\"b", ',', false).is_err());
        assert_eq!(parse_csv("", ',', true).unwrap(), Table::default());
    }

    #[test]
    fn libsvm_round_trip() {
        let text = "1 1:0.5 3:2\n-1 2:1.5 # comment\n\n0\n";
        let ex = parse_libsvm(text).unwrap();
        assert_eq!(ex.len(), 3);
        assert_eq!(ex[0], SparseExample { label: 1.0, features: vec![(1, 0.5), (3, 2.0)] });
        assert!(ex[2].features.is_empty());
        assert_eq!(parse_libsvm(&write_libsvm(&ex)).unwrap(), ex);
        let (x, y) = libsvm_to_dense(&ex, 3).unwrap();
        assert_eq!(x, vec![vec![0.5, 0.0, 2.0], vec![0.0, 1.5, 0.0], vec![0.0; 3]]);
        assert_eq!(y, vec![1.0, -1.0, 0.0]);
        assert!(libsvm_to_dense(&ex, 2).is_err());
        assert!(parse_libsvm("1 0:1").is_err() && parse_libsvm("1 2:1 1:1").is_err() && parse_libsvm("x 1:1").is_err());
    }

    #[test]
    fn idx_format() {
        // a 2x2x2 ubyte tensor
        let bytes = write_idx(&[2, 2, 2], &[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 255.0]).unwrap();
        assert_eq!(&bytes[..4], &[0, 0, 8, 3]);
        let a = parse_idx(&bytes).unwrap();
        assert_eq!(a.dims, vec![2, 2, 2]);
        assert_eq!(a.data[7], 255.0);
        // MNIST label-file style: magic 0x00000801, count, labels
        let labels = [0u8, 0, 8, 1, 0, 0, 0, 3, 7, 2, 1];
        assert_eq!(parse_idx(&labels).unwrap(), IdxArray { dims: vec![3], data: vec![7.0, 2.0, 1.0] });
        // big-endian float payload
        let mut f = vec![0u8, 0, 0x0D, 1, 0, 0, 0, 1];
        f.extend(1.5f32.to_be_bytes());
        assert_eq!(parse_idx(&f).unwrap().data, vec![1.5]);
        assert!(parse_idx(&[1, 0, 8, 1]).is_err() && parse_idx(&labels[..9]).is_err() && parse_idx(&[0, 0, 0x77, 1, 0, 0, 0, 1, 0]).is_err());
        assert!(write_idx(&[2], &[1.0, 300.0]).is_err() && write_idx(&[3], &[1.0]).is_err());
    }
}
