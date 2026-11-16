//! Pure-Rust HuggingFace-compatible `.safetensors` serialization and deserialization.
//!
//! Format specification:
//! - 8 bytes: little-endian `u64` length `N` of the UTF-8 JSON header.
//! - `N` bytes: JSON object mapping tensor names to
//!   `{"dtype": "F32", "shape": [d0, d1, ...], "data_offsets": [begin, end]}`.
//! - Remaining bytes: raw little-endian tensor buffers concatenated in order.

use std::collections::HashMap;
use std::fs;
use std::path::Path;

use crate::backend::{CpuStorage, Device, Storage};
use crate::dtype::{DType, bf16, f16, f8e4m3, f8e5m2};
use crate::error::{Error, Result};
use crate::layout::Layout;
use crate::tensor::Tensor;

/// Serializes a map of named tensors into the `.safetensors` binary format in memory.
pub fn serialize_safetensors(tensors: &HashMap<String, Tensor>) -> Result<Vec<u8>> {
    serialize_safetensors_with_metadata(tensors, None)
}

/// Serializes a map of named tensors and optional string metadata into the `.safetensors` binary format.
pub fn serialize_safetensors_with_metadata(
    tensors: &HashMap<String, Tensor>,
    metadata: Option<&HashMap<String, String>>,
) -> Result<Vec<u8>> {
    let mut keys: Vec<&String> = tensors.keys().collect();
    keys.sort();

    let mut raw_body = Vec::<u8>::new();
    let mut entries = Vec::<String>::with_capacity(keys.len() + 1);

    if let Some(meta) = metadata {
        if !meta.is_empty() {
            let mut meta_keys: Vec<&String> = meta.keys().collect();
            meta_keys.sort();
            let mut meta_entries = Vec::with_capacity(meta_keys.len());
            for mk in meta_keys {
                meta_entries.push(format!(
                    "\"{}\":\"{}\"",
                    escape_json(mk),
                    escape_json(&meta[mk])
                ));
            }
            entries.push(format!("\"__metadata__\":{{{}}}", meta_entries.join(",")));
        }
    }

    for k in keys {
        let t = &tensors[k];
        let bytes = tensor_to_le_bytes(t)?;
        let start = raw_body.len();
        raw_body.extend_from_slice(&bytes);
        let end = raw_body.len();

        let shape_str = t
            .shape()
            .iter()
            .map(usize::to_string)
            .collect::<Vec<_>>()
            .join(",");
        let entry = format!(
            "\"{}\":{{\"dtype\":\"{}\",\"shape\":[{}],\"data_offsets\":[{},{}]}}",
            escape_json(k),
            t.dtype().safetensors_name(),
            shape_str,
            start,
            end
        );
        entries.push(entry);
    }

    let mut header = format!("{{{}}}", entries.join(",")).into_bytes();
    // Pad header with spaces to 8-byte alignment per safetensors spec.
    while header.len() % 8 != 0 {
        header.push(b' ');
    }

    let header_len = header.len() as u64;
    let mut out = Vec::with_capacity(8 + header.len() + raw_body.len());
    out.extend_from_slice(&header_len.to_le_bytes());
    out.extend_from_slice(&header);
    out.extend_from_slice(&raw_body);
    Ok(out)
}

/// Deserializes tensors from `.safetensors` bytes onto `Device::Cpu`.
pub fn deserialize_safetensors(bytes: &[u8]) -> Result<HashMap<String, Tensor>> {
    let (tensors, _) = deserialize_safetensors_with_metadata(bytes)?;
    Ok(tensors)
}

/// Deserializes tensors and string metadata from `.safetensors` bytes onto `Device::Cpu`.
pub fn deserialize_safetensors_with_metadata(
    bytes: &[u8],
) -> Result<(HashMap<String, Tensor>, HashMap<String, String>)> {
    if bytes.len() < 8 {
        return Err(Error::Serialization("safetensors file too short".into()));
    }
    let header_len = u64::from_le_bytes(bytes[0..8].try_into().unwrap());
    // `8 + header_len` can overflow for a hostile length; compare against what is left instead.
    let header_len = usize::try_from(header_len)
        .ok()
        .filter(|&n| n <= bytes.len() - 8)
        .ok_or_else(|| Error::Serialization("safetensors header length exceeds buffer".into()))?;
    let header_str = std::str::from_utf8(&bytes[8..8 + header_len])
        .map_err(|e| Error::Serialization(format!("invalid UTF-8 header: {e}")))?;
    let body = &bytes[8 + header_len..];

    parse_safetensors_header(header_str, body)
}

/// Saves a map of named tensors to a `.safetensors` file.
pub fn save_safetensors(tensors: &HashMap<String, Tensor>, path: impl AsRef<Path>) -> Result<()> {
    save_safetensors_with_metadata(tensors, None, path)
}

/// Saves a map of named tensors and optional metadata to a `.safetensors` file.
pub fn save_safetensors_with_metadata(
    tensors: &HashMap<String, Tensor>,
    metadata: Option<&HashMap<String, String>>,
    path: impl AsRef<Path>,
) -> Result<()> {
    let bytes = serialize_safetensors_with_metadata(tensors, metadata)?;
    fs::write(path, bytes).map_err(|e| Error::Serialization(e.to_string()))
}

/// Loads a map of named tensors from a `.safetensors` file.
pub fn load_safetensors(path: impl AsRef<Path>) -> Result<HashMap<String, Tensor>> {
    let (tensors, _) = load_safetensors_with_metadata(path)?;
    Ok(tensors)
}

/// Loads a map of named tensors and metadata from a `.safetensors` file.
pub fn load_safetensors_with_metadata(
    path: impl AsRef<Path>,
) -> Result<(HashMap<String, Tensor>, HashMap<String, String>)> {
    let bytes = fs::read(path).map_err(|e| Error::Serialization(e.to_string()))?;
    deserialize_safetensors_with_metadata(&bytes)
}

/// Reinterprets a typed slice as bytes. Little-endian hosts store integers and
/// floats in LE order already, so this replaces per-element `to_le_bytes` loops.
fn as_le_bytes<T>(v: &[T]) -> &[u8] {
    // SAFETY: `v` is a valid shared slice, so its full footprint is readable for the
    // borrow's lifetime; `u8` accepts any alignment, so the cast is always sound.
    unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, std::mem::size_of_val(v)) }
}

/// Copies LE bytes into an aligned typed vector (the `from_le_bytes` counterpart).
fn vec_from_le_bytes<T: Copy>(raw: &[u8]) -> Vec<T> {
    let size = std::mem::size_of::<T>();
    assert!(size > 0 && raw.len() % size == 0);
    let mut out = Vec::with_capacity(raw.len() / size);
    // SAFETY: `out` has capacity for `raw.len()` bytes; the copy fills exactly
    // `raw.len() / size` initialized elements, which `set_len` exposes. `T: Copy`
    // makes the bitwise copy sound (no `Drop` to run twice).
    unsafe {
        std::ptr::copy_nonoverlapping(raw.as_ptr(), out.as_mut_ptr() as *mut u8, raw.len());
        out.set_len(raw.len() / size);
    }
    out
}

fn tensor_to_le_bytes(t: &Tensor) -> Result<Vec<u8>> {
    let cpu = t.storage().cpu_ref()?.to_contiguous(t.layout())?;
    let mut out = Vec::with_capacity(t.numel() * t.dtype().size_in_bytes());
    match cpu {
        CpuStorage::Bool(v) => out.extend(v.into_iter().map(u8::from)),
        CpuStorage::U8(v) => out.extend_from_slice(&v),
        CpuStorage::I8(v) => out.extend(v.into_iter().map(|x| x as u8)),
        CpuStorage::I16(v) => v.into_iter().for_each(|x| out.extend_from_slice(&x.to_le_bytes())),
        CpuStorage::U16(v) => v.into_iter().for_each(|x| out.extend_from_slice(&x.to_le_bytes())),
        CpuStorage::U32(v) => v.into_iter().for_each(|x| out.extend_from_slice(&x.to_le_bytes())),
        CpuStorage::U64(v) => v.into_iter().for_each(|x| out.extend_from_slice(&x.to_le_bytes())),
        CpuStorage::I32(v) => {
            if cfg!(target_endian = "little") {
                out.extend_from_slice(as_le_bytes(&v));
            } else {
                for x in v {
                    out.extend_from_slice(&x.to_le_bytes());
                }
            }
        }
        CpuStorage::I64(v) => {
            if cfg!(target_endian = "little") {
                out.extend_from_slice(as_le_bytes(&v));
            } else {
                for x in v {
                    out.extend_from_slice(&x.to_le_bytes());
                }
            }
        }
        CpuStorage::F16(v) => {
            if cfg!(target_endian = "little") {
                out.extend_from_slice(as_le_bytes(&v));
            } else {
                for x in v {
                    out.extend_from_slice(&x.to_bits().to_le_bytes());
                }
            }
        }
        CpuStorage::F8E4M3(v) => out.extend(v.into_iter().map(f8e4m3::to_bits)),
        CpuStorage::F8E5M2(v) => out.extend(v.into_iter().map(f8e5m2::to_bits)),
        CpuStorage::BF16(v) => {
            if cfg!(target_endian = "little") {
                out.extend_from_slice(as_le_bytes(&v));
            } else {
                for x in v {
                    out.extend_from_slice(&x.to_bits().to_le_bytes());
                }
            }
        }
        CpuStorage::F32(v) => {
            if cfg!(target_endian = "little") {
                out.extend_from_slice(as_le_bytes(&v));
            } else {
                for x in v {
                    out.extend_from_slice(&x.to_le_bytes());
                }
            }
        }
        CpuStorage::F64(v) => {
            if cfg!(target_endian = "little") {
                out.extend_from_slice(as_le_bytes(&v));
            } else {
                for x in v {
                    out.extend_from_slice(&x.to_le_bytes());
                }
            }
        }
        CpuStorage::C64(v) => {
            for [r, i] in v {
                out.extend_from_slice(&r.to_le_bytes());
                out.extend_from_slice(&i.to_le_bytes());
            }
        }
        CpuStorage::C128(v) => {
            for [r, i] in v {
                out.extend_from_slice(&r.to_le_bytes());
                out.extend_from_slice(&i.to_le_bytes());
            }
        }
    }
    Ok(out)
}

fn tensor_from_le_bytes(dtype: DType, shape: Vec<usize>, raw: &[u8]) -> Result<Tensor> {
    let expected_bytes = shape
        .iter()
        .try_fold(dtype.size_in_bytes(), |acc, &d| acc.checked_mul(d))
        .ok_or_else(|| Error::Serialization(format!("safetensors shape {shape:?} overflows")))?;
    if raw.len() != expected_bytes {
        return Err(Error::Serialization(format!(
            "safetensors buffer size mismatch: expected {expected_bytes}, got {}",
            raw.len()
        )));
    }
    if cfg!(target_endian = "little") {
        let cpu = match dtype {
            DType::Bool => CpuStorage::Bool(raw.iter().map(|&b| b != 0).collect()),
            DType::U8 => CpuStorage::U8(raw.to_vec()),
            DType::I8 => CpuStorage::I8(raw.iter().map(|&b| b as i8).collect()),
            DType::I16 => CpuStorage::I16(raw.chunks_exact(2).map(|c| i16::from_le_bytes(c.try_into().unwrap())).collect()),
            DType::U16 => CpuStorage::U16(raw.chunks_exact(2).map(|c| u16::from_le_bytes(c.try_into().unwrap())).collect()),
            DType::U32 => CpuStorage::U32(raw.chunks_exact(4).map(|c| u32::from_le_bytes(c.try_into().unwrap())).collect()),
            DType::U64 => CpuStorage::U64(raw.chunks_exact(8).map(|c| u64::from_le_bytes(c.try_into().unwrap())).collect()),
            DType::I32 => CpuStorage::I32(vec_from_le_bytes(raw)),
            DType::I64 => CpuStorage::I64(vec_from_le_bytes(raw)),
            DType::F16 => CpuStorage::F16(vec_from_le_bytes(raw)),
            DType::BF16 => CpuStorage::BF16(vec_from_le_bytes(raw)),
            DType::F8E4M3 => CpuStorage::F8E4M3(raw.iter().map(|&b| f8e4m3::from_bits(b)).collect()),
            DType::F8E5M2 => CpuStorage::F8E5M2(raw.iter().map(|&b| f8e5m2::from_bits(b)).collect()),
            DType::F32 => CpuStorage::F32(vec_from_le_bytes(raw)),
            DType::F64 => CpuStorage::F64(vec_from_le_bytes(raw)),
            DType::C64 => CpuStorage::C64(
                raw.chunks_exact(8)
                    .map(|c| [
                        f32::from_le_bytes(c[0..4].try_into().unwrap()),
                        f32::from_le_bytes(c[4..8].try_into().unwrap()),
                    ])
                    .collect(),
            ),
            DType::C128 => CpuStorage::C128(
                raw.chunks_exact(16)
                    .map(|c| [
                        f64::from_le_bytes(c[0..8].try_into().unwrap()),
                        f64::from_le_bytes(c[8..16].try_into().unwrap()),
                    ])
                    .collect(),
            ),
        };
        let _ = Device::Cpu;
        return Ok(Tensor::from_storage(
            Storage::new_cpu(cpu),
            Layout::contiguous(shape),
        ));
    }
    let cpu = match dtype {
        DType::Bool => CpuStorage::Bool(raw.iter().map(|&b| b != 0).collect()),
        DType::U8 => CpuStorage::U8(raw.to_vec()),
        DType::I8 => CpuStorage::I8(raw.iter().map(|&b| b as i8).collect()),
        DType::I16 => CpuStorage::I16(raw.chunks_exact(2).map(|c| i16::from_le_bytes(c.try_into().unwrap())).collect()),
        DType::U16 => CpuStorage::U16(raw.chunks_exact(2).map(|c| u16::from_le_bytes(c.try_into().unwrap())).collect()),
        DType::U32 => CpuStorage::U32(raw.chunks_exact(4).map(|c| u32::from_le_bytes(c.try_into().unwrap())).collect()),
        DType::U64 => CpuStorage::U64(raw.chunks_exact(8).map(|c| u64::from_le_bytes(c.try_into().unwrap())).collect()),
        DType::I32 => CpuStorage::I32(
            raw.chunks_exact(4)
                .map(|c| i32::from_le_bytes(c.try_into().unwrap()))
                .collect(),
        ),
        DType::I64 => CpuStorage::I64(
            raw.chunks_exact(8)
                .map(|c| i64::from_le_bytes(c.try_into().unwrap()))
                .collect(),
        ),
        DType::F16 => CpuStorage::F16(
            raw.chunks_exact(2)
                .map(|c| f16::from_bits(u16::from_le_bytes(c.try_into().unwrap())))
                .collect(),
        ),
        DType::F8E4M3 => CpuStorage::F8E4M3(raw.iter().map(|&b| f8e4m3::from_bits(b)).collect()),
        DType::F8E5M2 => CpuStorage::F8E5M2(raw.iter().map(|&b| f8e5m2::from_bits(b)).collect()),
        DType::BF16 => CpuStorage::BF16(
            raw.chunks_exact(2)
                .map(|c| bf16::from_bits(u16::from_le_bytes(c.try_into().unwrap())))
                .collect(),
        ),
        DType::F32 => CpuStorage::F32(
            raw.chunks_exact(4)
                .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
                .collect(),
        ),
        DType::F64 => CpuStorage::F64(
            raw.chunks_exact(8)
                .map(|c| f64::from_le_bytes(c.try_into().unwrap()))
                .collect(),
        ),
        DType::C64 => CpuStorage::C64(
            raw.chunks_exact(8)
                .map(|c| [
                    f32::from_le_bytes(c[0..4].try_into().unwrap()),
                    f32::from_le_bytes(c[4..8].try_into().unwrap()),
                ])
                .collect(),
        ),
        DType::C128 => CpuStorage::C128(
            raw.chunks_exact(16)
                .map(|c| [
                    f64::from_le_bytes(c[0..8].try_into().unwrap()),
                    f64::from_le_bytes(c[8..16].try_into().unwrap()),
                ])
                .collect(),
        ),
    };
    let _ = Device::Cpu;
    Ok(Tensor::from_storage(
        Storage::new_cpu(cpu),
        Layout::contiguous(shape),
    ))
}

fn escape_json(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 2);
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            '\u{0008}' => out.push_str("\\b"),
            '\u{000C}' => out.push_str("\\f"),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out
}

fn parse_safetensors_header(
    header: &str,
    body: &[u8],
) -> Result<(HashMap<String, Tensor>, HashMap<String, String>)> {
    let s = header.trim();
    if !s.starts_with('{') || !s.ends_with('}') {
        return Err(Error::Serialization(
            "invalid safetensors JSON object".into(),
        ));
    }
    let inner = &s[1..s.len() - 1];
    let mut out = HashMap::new();
    let mut meta = HashMap::new();
    let mut seen = std::collections::HashSet::new();
    let mut pos = 0usize;
    let bytes = inner.as_bytes();

    while pos < bytes.len() {
        while pos < bytes.len() && (bytes[pos].is_ascii_whitespace() || bytes[pos] == b',') {
            pos += 1;
        }
        if pos >= bytes.len() {
            break;
        }
        let (key, next_pos) = parse_json_string(inner, pos)?;
        pos = next_pos;
        while pos < bytes.len() && (bytes[pos].is_ascii_whitespace() || bytes[pos] == b':') {
            pos += 1;
        }
        let (obj_str, next_pos) = extract_balanced(inner, pos, b'{', b'}')?;
        pos = next_pos;

        if !seen.insert(key.clone()) {
            return Err(Error::Serialization(format!(
                "duplicate header key '{key}' (spec disallows duplicate keys)"
            )));
        }
        if key == "__metadata__" {
            meta = parse_metadata_object(obj_str)?;
            continue;
        }

        let dtype_str = extract_field_string(obj_str, "dtype")?;
        let dtype = DType::from_safetensors_name(&dtype_str)?;
        let shape = extract_field_usize_list(obj_str, "shape")?;
        let offsets = extract_field_usize_list(obj_str, "data_offsets")?;
        if offsets.len() != 2 || offsets[1] > body.len() || offsets[0] > offsets[1] {
            return Err(Error::Serialization(format!(
                "invalid data_offsets for '{key}': {offsets:?}"
            )));
        }
        let tensor = tensor_from_le_bytes(dtype, shape, &body[offsets[0]..offsets[1]])?;
        out.insert(key, tensor);
    }

    Ok((out, meta))
}

/// Parses the `__metadata__` string-to-string map. The spec requires all values to be
/// strings, so anything else (numbers, nested objects, malformed pairs) is an error: the
/// previous silent `break` truncated the map and dropped entries without telling the caller.
fn parse_metadata_object(obj_str: &str) -> Result<HashMap<String, String>> {
    let mut meta = HashMap::new();
    let s = obj_str.trim();
    if !s.starts_with('{') || !s.ends_with('}') {
        return Err(Error::Serialization("malformed __metadata__ object".into()));
    }
    let inner = &s[1..s.len() - 1];
    let bytes = inner.as_bytes();
    let mut pos = 0usize;
    while pos < bytes.len() {
        while pos < bytes.len() && (bytes[pos].is_ascii_whitespace() || bytes[pos] == b',') {
            pos += 1;
        }
        if pos >= bytes.len() {
            break;
        }
        let (k, next_pos) = parse_json_string(inner, pos)
            .map_err(|_| Error::Serialization("malformed key in __metadata__".into()))?;
        pos = next_pos;
        while pos < bytes.len() && (bytes[pos].is_ascii_whitespace() || bytes[pos] == b':') {
            pos += 1;
        }
        let (v, next_pos2) = parse_json_string(inner, pos)
            .map_err(|_| Error::Serialization("non-string value in __metadata__".into()))?;
        pos = next_pos2;
        meta.insert(k, v);
    }
    Ok(meta)
}

/// Parses a JSON string literal starting at `s.as_bytes()[start] == '"'`.
///
/// Collects raw bytes (not `char`s) and decodes the whole run as UTF-8 once at the end: a
/// multi-byte UTF-8 sequence (any non-ASCII text in a tensor name or `__metadata__` value, which
/// is valid unescaped in JSON) has continuation bytes that are not valid standalone `char`s, so
/// casting each byte to `char` one at a time — the previous approach — produced mojibake
/// (`"café"` decoded as `"cafÃ©"`).
fn parse_json_string(s: &str, start: usize) -> Result<(String, usize)> {
    let b = s.as_bytes();
    if b.get(start) != Some(&b'"') {
        return Err(Error::Serialization("expected '\"' in JSON".into()));
    }
    let mut i = start + 1;
    let mut out = Vec::new();
    while i < b.len() {
        if b[i] == b'\\' && i + 1 < b.len() {
            match b[i + 1] {
                b'"' => {
                    out.push(b'"');
                    i += 2;
                }
                b'\\' => {
                    out.push(b'\\');
                    i += 2;
                }
                b'/' => {
                    out.push(b'/');
                    i += 2;
                }
                b'n' => {
                    out.push(b'\n');
                    i += 2;
                }
                b'r' => {
                    out.push(b'\r');
                    i += 2;
                }
                b't' => {
                    out.push(b'\t');
                    i += 2;
                }
                b'b' => {
                    out.push(0x08);
                    i += 2;
                }
                b'f' => {
                    out.push(0x0C);
                    i += 2;
                }
                b'u' => {
                    let cp = parse_json_unicode_escape(b, i)?;
                    i = cp.1;
                    let mut buf = [0u8; 4];
                    out.extend_from_slice(cp.0.encode_utf8(&mut buf).as_bytes());
                }
                e => {
                    return Err(Error::Serialization(format!(
                        "invalid JSON escape '\\{}'",
                        e as char
                    )));
                }
            }
        } else if b[i] == b'"' {
            let decoded = String::from_utf8(out)
                .map_err(|e| Error::Serialization(format!("invalid UTF-8 in JSON string: {e}")))?;
            return Ok((decoded, i + 1));
        } else {
            out.push(b[i]);
            i += 1;
        }
    }
    Err(Error::Serialization("unterminated JSON string".into()))
}

/// Parses a `\uXXXX` escape (plus a low-surrogate half for astral code
/// points) at `b[i] == b'\\'`, returning the `char` and the next index.
fn parse_json_unicode_escape(b: &[u8], i: usize) -> Result<(char, usize)> {
    let hex4 = |j: usize| -> Result<u32> {
        if j + 4 > b.len() {
            return Err(Error::Serialization("truncated \\u escape".into()));
        }
        let mut v = 0u32;
        for k in 0..4 {
            let h = (b[j + k] as char)
                .to_digit(16)
                .ok_or_else(|| Error::Serialization("invalid hex in \\u escape".into()))?;
            v = v * 16 + h;
        }
        Ok(v)
    };
    let hi = hex4(i + 2)?;
    let mut next = i + 6;
    let cp = if (0xD800..0xDC00).contains(&hi) {
        if b.get(next) != Some(&b'\\') || b.get(next + 1) != Some(&b'u') {
            return Err(Error::Serialization("lone high surrogate".into()));
        }
        let lo = hex4(next + 2)?;
        if !(0xDC00..0xE000).contains(&lo) {
            return Err(Error::Serialization("lone high surrogate".into()));
        }
        next += 6;
        0x10000 + ((hi - 0xD800) << 10) + (lo - 0xDC00)
    } else if (0xDC00..0xE000).contains(&hi) {
        return Err(Error::Serialization("lone low surrogate".into()));
    } else {
        hi
    };
    let ch = char::from_u32(cp).ok_or_else(|| Error::Serialization("invalid code point".into()))?;
    Ok((ch, next))
}

fn extract_balanced(s: &str, start: usize, open: u8, close: u8) -> Result<(&str, usize)> {
    let b = s.as_bytes();
    if b.get(start) != Some(&open) {
        return Err(Error::Serialization("expected opening delimiter".into()));
    }
    let mut depth = 0usize;
    let mut in_str = false;
    let mut i = start;
    while i < b.len() {
        if b[i] == b'\\' && in_str {
            i += 2;
            continue;
        }
        if b[i] == b'"' {
            in_str = !in_str;
        } else if !in_str {
            if b[i] == open {
                depth += 1;
            } else if b[i] == close {
                depth = depth.saturating_sub(1);
                if depth == 0 {
                    return Ok((&s[start..=i], i + 1));
                }
            }
        }
        i += 1;
    }
    Err(Error::Serialization("unbalanced JSON braces".into()))
}

/// Returns the raw value substring for a top-level key of a flat JSON object. Single pass,
/// depth- and string-aware, so a decoy key inside a nested value cannot shadow the real
/// field. First match wins (same as the old find-first lookup for duplicate keys).
fn top_level_field_value<'a>(obj: &'a str, field: &str) -> Option<&'a str> {
    let b = obj.as_bytes();
    let mut i = 0;
    while i < b.len() && b[i].is_ascii_whitespace() {
        i += 1;
    }
    if b.get(i) != Some(&b'{') {
        return None;
    }
    i += 1;
    loop {
        while i < b.len() && (b[i].is_ascii_whitespace() || b[i] == b',') {
            i += 1;
        }
        if i >= b.len() || b[i] == b'}' {
            return None;
        }
        let (key, next) = parse_json_string(obj, i).ok()?;
        i = next;
        while i < b.len() && (b[i].is_ascii_whitespace() || b[i] == b':') {
            i += 1;
        }
        if i >= b.len() {
            return None;
        }
        let val_start = i;
        let val_end = if b[i] == b'{' || b[i] == b'[' {
            let open = b[i];
            let close = if open == b'{' { b'}' } else { b']' };
            extract_balanced(obj, i, open, close).ok()?.1
        } else if b[i] == b'"' {
            parse_json_string(obj, i).ok()?.1
        } else {
            let mut j = i;
            while j < b.len() && b[j] != b',' && b[j] != b'}' {
                j += 1;
            }
            j
        };
        if key == field {
            return obj.get(val_start..val_end);
        }
        i = val_end;
    }
}

fn extract_field_string(obj: &str, field: &str) -> Result<String> {
    let v = top_level_field_value(obj, field)
        .ok_or_else(|| Error::Serialization(format!("missing field '{field}'")))?;
    if !v.starts_with('"') {
        return Err(Error::Serialization(format!(
            "malformed string field '{field}'"
        )));
    }
    Ok(parse_json_string(v, 0)?.0)
}

fn extract_field_usize_list(obj: &str, field: &str) -> Result<Vec<usize>> {
    let v = top_level_field_value(obj, field)
        .ok_or_else(|| Error::Serialization(format!("missing field '{field}'")))?;
    if !v.starts_with('[') {
        return Err(Error::Serialization(format!("expected '[' for '{field}'")));
    }
    let (arr_str, _) = extract_balanced(v, 0, b'[', b']')?;
    let inner = arr_str[1..arr_str.len() - 1].trim();
    if inner.is_empty() {
        return Ok(Vec::new());
    }
    inner
        .split(',')
        .map(|tok| {
            tok.trim()
                .parse::<usize>()
                .map_err(|e| Error::Serialization(e.to_string()))
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Control characters and quotes in metadata must survive a save/load
    /// round trip (the writer used to emit them raw: invalid JSON).
    #[test]
    fn json_metadata_with_controls_round_trips() {
        let mut tensors = HashMap::new();
        tensors.insert(
            "w".to_string(),
            Tensor::from_slice(&[1.0f32], &[1]).unwrap(),
        );
        let mut meta = HashMap::new();
        meta.insert(
            "note".to_string(),
            "line1\nline2\ttab\"quote\"\\slash\r\ncaf\u{e9}".to_string(),
        );
        let bytes = serialize_safetensors_with_metadata(&tensors, Some(&meta)).unwrap();
        let (_, back) = deserialize_safetensors_with_metadata(&bytes).unwrap();
        assert_eq!(back.get("note"), meta.get("note"));
    }

    /// Standard JSON escapes (including \uXXXX and surrogate pairs) must
    /// decode; invalid escapes must error instead of mis-decoding.
    #[test]
    fn json_string_parses_standard_escapes() {
        let (s, _) = parse_json_string("\"a\\nb\\tc\\\"d\\\\e\\/f\\u00e9\"", 0).unwrap();
        assert_eq!(s, "a\nb\tc\"d\\e/f\u{e9}");
        let (pair, _) = parse_json_string("\"A\\ud834\\udd1e\"", 0).unwrap();
        assert_eq!(pair, "A\u{1d11e}");
        assert!(parse_json_string("\"\\x\"", 0).is_err());
        assert!(parse_json_string("\"\\ud834\"", 0).is_err());
        assert!(parse_json_string("\"\\udc00\"", 0).is_err());
    }

    /// Hostile or truncated input must be an `Err`, never a panic or an arithmetic overflow.
    #[test]
    fn malformed_files_are_errors_not_panics() {
        let mut tensors = HashMap::new();
        tensors.insert(
            "w".to_string(),
            Tensor::from_slice(&[1.0f32, 2.0, 3.0], &[3]).unwrap(),
        );
        let mut meta = HashMap::new();
        meta.insert("k".to_string(), "v".to_string());
        let good = serialize_safetensors_with_metadata(&tensors, Some(&meta)).unwrap();
        for cut in 0..good.len() {
            let _ = deserialize_safetensors_with_metadata(&good[..cut]); // must not panic
        }
        // Every single-byte corruption must also be survivable.
        for i in 0..good.len() {
            let mut bad = good.clone();
            bad[i] ^= 0xFF;
            let _ = deserialize_safetensors_with_metadata(&bad);
        }
        // header_len = u64::MAX: `8 + header_len` used to overflow.
        let mut huge = good.clone();
        huge[0..8].copy_from_slice(&u64::MAX.to_le_bytes());
        assert!(deserialize_safetensors(&huge).is_err());
        // Shape whose byte size overflows usize.
        let header = br#"{"w":{"dtype":"F32","shape":[4294967296,4294967296,4294967296],"data_offsets":[0,0]}}"#;
        let mut buf = (header.len() as u64).to_le_bytes().to_vec();
        buf.extend_from_slice(header);
        assert!(deserialize_safetensors(&buf).is_err());
    }

    /// Tensor entries may carry extra nested objects (forward-compat with foreign writers); a
    /// decoy `"shape"` inside one must not shadow the real field.
    #[test]
    fn extra_nested_fields_do_not_shadow_real_ones() {
        let header =
            br#"{"w":{"meta":{"shape":[999]},"dtype":"F32","shape":[2],"data_offsets":[0,8]}}"#;
        let mut buf = (header.len() as u64).to_le_bytes().to_vec();
        buf.extend_from_slice(header);
        buf.extend_from_slice(&1.0f32.to_le_bytes());
        buf.extend_from_slice(&2.0f32.to_le_bytes());
        let out = deserialize_safetensors(&buf).unwrap();
        let t = &out["w"];
        assert_eq!(t.shape(), &[2]);
        assert_eq!(t.to_vec::<f32>().unwrap(), vec![1.0f32, 2.0]);
    }

    /// Non-ASCII text in `__metadata__` (common in real HF model cards: author names, licenses,
    /// descriptions) is valid unescaped in a JSON string. Each byte of a multi-byte UTF-8
    /// sequence is not a valid standalone `char`, so decoding byte-by-byte instead of decoding
    /// the whole run at once previously turned `"café"` into `"cafÃ©"`.
    #[test]
    fn metadata_round_trips_non_ascii_text() {
        let mut tensors = HashMap::new();
        tensors.insert(
            "café".to_string(),
            Tensor::from_slice(&[1.0f32, 2.0], &[2]).unwrap(),
        );
        let mut meta = HashMap::new();
        meta.insert("license".to_string(), "café \u{1F980}".to_string()); // crab emoji too
        let bytes = serialize_safetensors_with_metadata(&tensors, Some(&meta)).unwrap();
        let (out_tensors, out_meta) = deserialize_safetensors_with_metadata(&bytes).unwrap();
        assert!(out_tensors.contains_key("café"));
        assert_eq!(out_meta.get("license"), Some(&"café \u{1F980}".to_string()));
    }

    /// Every dtype must survive a save/load round trip bit-exact: signed zeros, infinities,
    /// NaN payloads, integer extremes, and i64 values beyond 2^53. Compared against what
    /// `from_slice` produced (not the literals), so this pins the serialization path only.
    #[test]
    fn all_dtypes_round_trip_bit_exact() {
        let mut tensors = HashMap::new();
        tensors.insert(
            "b".to_string(),
            Tensor::from_slice(&[true, false, true], &[3]).unwrap(),
        );
        tensors.insert(
            "u8".to_string(),
            Tensor::from_slice(&[0u8, 1, 128, 255], &[4]).unwrap(),
        );
        tensors.insert(
            "i32".to_string(),
            Tensor::from_slice(&[0i32, -1, 1, i32::MIN, i32::MAX], &[5]).unwrap(),
        );
        tensors.insert(
            "i64".to_string(),
            Tensor::from_slice(&[0i64, -1, i64::MIN, i64::MAX, (1i64 << 53) + 1], &[5]).unwrap(),
        );
        let f16_bits = [0x0000u16, 0x3C00, 0xBC00, 0x7C00, 0xFC00, 0x7E00, 0x7FFF];
        let f16_vals: Vec<f16> = f16_bits.iter().map(|&b| f16::from_bits(b)).collect();
        tensors.insert(
            "f16".to_string(),
            Tensor::from_slice(&f16_vals, &[7]).unwrap(),
        );
        let bf16_bits = [0x0000u16, 0x3F80, 0xBF80, 0x7F80, 0xFF80, 0x7FC0, 0x7FFF];
        let bf16_vals: Vec<bf16> = bf16_bits.iter().map(|&b| bf16::from_bits(b)).collect();
        tensors.insert(
            "bf16".to_string(),
            Tensor::from_slice(&bf16_vals, &[7]).unwrap(),
        );
        tensors.insert(
            "f32".to_string(),
            Tensor::from_slice(
                &[
                    0.0f32,
                    -0.0,
                    1.5,
                    f32::INFINITY,
                    f32::NEG_INFINITY,
                    f32::NAN,
                ],
                &[6],
            )
            .unwrap(),
        );
        tensors.insert(
            "f64".to_string(),
            Tensor::from_slice(
                &[
                    0.0f64,
                    -0.0,
                    1.5,
                    f64::INFINITY,
                    f64::NEG_INFINITY,
                    f64::NAN,
                ],
                &[6],
            )
            .unwrap(),
        );
        let bytes = serialize_safetensors(&tensors).unwrap();
        let back = deserialize_safetensors(&bytes).unwrap();
        assert_eq!(back.len(), tensors.len());
        for (k, orig) in &tensors {
            let rt = &back[k];
            assert_eq!(rt.dtype(), orig.dtype(), "dtype of {k}");
            assert_eq!(rt.shape(), orig.shape(), "shape of {k}");
        }
        assert_eq!(
            back["b"].to_vec::<bool>().unwrap(),
            tensors["b"].to_vec::<bool>().unwrap()
        );
        assert_eq!(
            back["u8"].to_vec::<u8>().unwrap(),
            tensors["u8"].to_vec::<u8>().unwrap()
        );
        assert_eq!(
            back["i32"].to_vec::<i32>().unwrap(),
            tensors["i32"].to_vec::<i32>().unwrap()
        );
        assert_eq!(
            back["i64"].to_vec::<i64>().unwrap(),
            tensors["i64"].to_vec::<i64>().unwrap()
        );
        let bits = |t: &Tensor| {
            t.to_vec::<f16>()
                .unwrap()
                .iter()
                .map(|x| x.to_bits())
                .collect::<Vec<_>>()
        };
        assert_eq!(bits(&back["f16"]), bits(&tensors["f16"]));
        let bits16 = |t: &Tensor| {
            t.to_vec::<bf16>()
                .unwrap()
                .iter()
                .map(|x| x.to_bits())
                .collect::<Vec<_>>()
        };
        assert_eq!(bits16(&back["bf16"]), bits16(&tensors["bf16"]));
        let bits32 = |t: &Tensor| {
            t.to_vec::<f32>()
                .unwrap()
                .iter()
                .map(|x| x.to_bits())
                .collect::<Vec<_>>()
        };
        assert_eq!(bits32(&back["f32"]), bits32(&tensors["f32"]));
        let bits64 = |t: &Tensor| {
            t.to_vec::<f64>()
                .unwrap()
                .iter()
                .map(|x| x.to_bits())
                .collect::<Vec<_>>()
        };
        assert_eq!(bits64(&back["f64"]), bits64(&tensors["f64"]));
    }

    /// Scalars (`shape: []`), zero-element tensors, and non-contiguous views must round-trip
    /// with values intact (the writer densifies strided storage).
    #[test]
    fn scalar_empty_and_strided_round_trip() {
        let mut tensors = HashMap::new();
        tensors.insert(
            "scalar".to_string(),
            Tensor::from_slice(&[3.25f32], &[]).unwrap(),
        );
        tensors.insert(
            "empty".to_string(),
            Tensor::zeros(&[0], DType::F32, &Device::Cpu).unwrap(),
        );
        let base = Tensor::from_slice(&[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]).unwrap();
        let view = base.transpose(0, 1).unwrap();
        assert!(!view.is_contiguous());
        tensors.insert("strided".to_string(), view);
        let bytes = serialize_safetensors(&tensors).unwrap();
        let back = deserialize_safetensors(&bytes).unwrap();
        assert_eq!(back["scalar"].shape(), &[] as &[usize]);
        assert_eq!(back["scalar"].to_vec::<f32>().unwrap(), vec![3.25f32]);
        assert_eq!(back["empty"].shape(), &[0]);
        assert!(back["empty"].to_vec::<f32>().unwrap().is_empty());
        assert_eq!(back["strided"].shape(), &[3, 2]);
        assert_eq!(
            back["strided"].to_vec::<f32>().unwrap(),
            vec![1.0f32, 4.0, 2.0, 5.0, 3.0, 6.0]
        );
    }

    /// The emitted file must match the safetensors spec byte-for-byte: u64-LE header length,
    /// compact JSON header (space-padded to 8-byte alignment, which the spec explicitly
    /// permits: "The header data MAY be trailing padded with whitespace"), then the raw
    /// little-endian body. A single-entry map serializes deterministically, so the whole file
    /// is pinned (strongest cross-compat oracle available without the reference library: any
    /// spec reader must accept these exact bytes).
    #[test]
    fn spec_byte_layout_matches_hand_computed_file() {
        let mut tensors = HashMap::new();
        tensors.insert(
            "w".to_string(),
            Tensor::from_slice(&[1.0f32, -2.0], &[2]).unwrap(),
        );
        let bytes = serialize_safetensors(&tensors).unwrap();
        let mut header = br#"{"w":{"dtype":"F32","shape":[2],"data_offsets":[0,8]}}"#.to_vec();
        while header.len() % 8 != 0 {
            header.push(b' ');
        }
        let mut expected = (header.len() as u64).to_le_bytes().to_vec();
        expected.extend_from_slice(&header);
        expected.extend_from_slice(&1.0f32.to_le_bytes());
        expected.extend_from_slice(&(-2.0f32).to_le_bytes());
        assert_eq!(bytes, expected);
    }

    /// Semantically invalid headers must be `Err` (not panics, not silent mis-decoding):
    /// unknown dtypes (e.g. I16, which oxideon has no storage for), shape/byte-size
    /// mismatch, reversed or out-of-range offsets, missing fields, and non-JSON headers.
    #[test]
    fn semantic_errors_reject_bad_headers() {
        fn file(header: &[u8], body: &[u8]) -> Vec<u8> {
            let mut buf = (header.len() as u64).to_le_bytes().to_vec();
            buf.extend_from_slice(header);
            buf.extend_from_slice(body);
            buf
        }
        let body8 = 1.0f32.to_le_bytes().to_vec();
        // Unknown dtype.
        let buf = file(
            br#"{"w":{"dtype":"C64","shape":[1],"data_offsets":[0,8]}}"#,
            &[0, 0],
        );
        assert!(deserialize_safetensors(&buf).is_err());
        // Shape claims 3 f32s but the slice holds 1.
        let buf = file(
            br#"{"w":{"dtype":"F32","shape":[3],"data_offsets":[0,4]}}"#,
            &body8,
        );
        assert!(deserialize_safetensors(&buf).is_err());
        // Reversed offsets.
        let buf = file(
            br#"{"w":{"dtype":"F32","shape":[1],"data_offsets":[4,0]}}"#,
            &body8,
        );
        assert!(deserialize_safetensors(&buf).is_err());
        // Offsets past the end of the body.
        let buf = file(
            br#"{"w":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}}"#,
            &[],
        );
        assert!(deserialize_safetensors(&buf).is_err());
        // Missing "shape" field.
        let buf = file(br#"{"w":{"dtype":"F32","data_offsets":[0,4]}}"#, &body8);
        assert!(deserialize_safetensors(&buf).is_err());
        // Non-JSON header.
        let buf = file(b"not json", &body8);
        assert!(deserialize_safetensors(&buf).is_err());
        // Empty buffer.
        assert!(deserialize_safetensors(&[]).is_err());
        // Duplicate keys are disallowed by the spec (used to last-win silently).
        let buf = file(
            br#"{"w":{"dtype":"F32","shape":[1],"data_offsets":[0,4]},"w":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}}"#,
            &body8,
        );
        assert!(deserialize_safetensors(&buf).is_err());
        // Non-string __metadata__ values violate the spec (used to truncate the map silently).
        let buf = file(
            br#"{"__metadata__":{"k":1},"w":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}}"#,
            &body8,
        );
        assert!(deserialize_safetensors_with_metadata(&buf).is_err());
        // Malformed __metadata__ pair (used to stop parsing and return a partial map).
        let buf = file(
            br#"{"__metadata__":{"k":"v",oops},"w":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}}"#,
            &body8,
        );
        assert!(deserialize_safetensors_with_metadata(&buf).is_err());
    }

    /// Tensor *names* need the same escaping as metadata values: quotes, backslashes, and
    /// control characters in a key must round-trip (the reference format allows arbitrary
    /// unicode keys, and HF repos do use odd names).
    #[test]
    fn tensor_names_with_escapes_round_trip() {
        let mut tensors = HashMap::new();
        let tricky = "a\"b\\c\nd☃".to_string();
        tensors.insert(tricky.clone(), Tensor::from_slice(&[7.0f32], &[1]).unwrap());
        let bytes = serialize_safetensors(&tensors).unwrap();
        let back = deserialize_safetensors(&bytes).unwrap();
        assert_eq!(back.len(), 1);
        assert_eq!(back[&tricky].to_vec::<f32>().unwrap(), vec![7.0f32]);
    }
}
