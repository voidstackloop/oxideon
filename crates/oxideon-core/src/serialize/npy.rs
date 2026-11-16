//! NumPy interchange: `.npy` arrays and `.npz` archives, read and write, without external dependencies.
//!
//! * [`save_npy`] / [`load_npy`] / [`npy_bytes`] / [`from_npy_bytes`]: format versions 1.0 - 3.0, little- and
//!   big-endian, C and Fortran order. Supported dtypes are `float16/32/64`, `int8/16/32/64`, `uint8/16/32/64` and
//!   `bool`; narrower integers widen to `I32`, `uint32` / `uint64` and `int64` become `I64`.
//! * [`save_npz`] / [`load_npz`]: ZIP archives of `.npy` entries. Writing stores entries uncompressed (what
//!   `numpy.savez` does); reading also handles the DEFLATE-compressed entries of `numpy.savez_compressed`.

use crate::dtype::{DType, f16};
use crate::error::{Error, Result};
use crate::tensor::Tensor;
use std::collections::HashMap;
use std::path::Path;

fn err(msg: impl Into<String>) -> Error {
    Error::Serialization(msg.into())
}

const MAGIC: &[u8; 6] = b"\x93NUMPY";

// ---------------------------------------------------------------------------------------------------------------
// .npy
// ---------------------------------------------------------------------------------------------------------------

fn descr_of(dtype: DType) -> Result<&'static str> {
    Ok(match dtype {
        DType::F16 => "<f2",
        DType::F32 => "<f4",
        DType::F64 => "<f8",
        DType::C64 => "<c8",
        DType::C128 => "<c16",
        DType::I32 => "<i4",
        DType::I64 => "<i8",
        DType::U8 => "|u1",
        DType::I8 => "|i1",
        DType::I16 => "<i2",
        DType::U16 => "<u2",
        DType::U32 => "<u4",
        DType::U64 => "<u8",
        DType::Bool => "|b1",
        DType::BF16 => return Err(err("bfloat16 has no NumPy dtype; cast to f32 before saving")),
        DType::F8E4M3 | DType::F8E5M2 => return Err(err("8-bit floats have no NumPy dtype; cast to f32 before saving")),
    })
}

/// Serializes a tensor as `.npy` (version 1.0, C order).
pub fn npy_bytes(t: &Tensor) -> Result<Vec<u8>> {
    let descr = descr_of(t.dtype())?;
    let shape = match t.shape() {
        [] => "()".to_string(),
        [n] => format!("({n},)"),
        s => format!("({})", s.iter().map(usize::to_string).collect::<Vec<_>>().join(", ")),
    };
    let mut header = format!("{{'descr': '{descr}', 'fortran_order': False, 'shape': {shape}, }}");
    // magic (6) + version (2) + length (2) + header + '\n' must be a multiple of 64
    let total = 10 + header.len() + 1;
    header.push_str(&" ".repeat((64 - total % 64) % 64));
    header.push('\n');
    let mut out = Vec::with_capacity(10 + header.len() + t.numel() * t.dtype().size_in_bytes());
    out.extend_from_slice(MAGIC);
    out.extend_from_slice(&[1, 0]);
    out.extend_from_slice(&(header.len() as u16).to_le_bytes());
    out.extend_from_slice(header.as_bytes());
    match t.dtype() {
        DType::F16 => t.to_vec::<f16>()?.iter().for_each(|v| out.extend_from_slice(&v.to_bits().to_le_bytes())),
        DType::F32 => t.to_vec::<f32>()?.iter().for_each(|v| out.extend_from_slice(&v.to_le_bytes())),
        DType::F64 => t.to_vec::<f64>()?.iter().for_each(|v| out.extend_from_slice(&v.to_le_bytes())),
        DType::C64 => t.to_c64_vec()?.iter().for_each(|[r, i]| {
            out.extend_from_slice(&r.to_le_bytes());
            out.extend_from_slice(&i.to_le_bytes());
        }),
        DType::C128 => t.to_c128_vec()?.iter().for_each(|[r, i]| {
            out.extend_from_slice(&r.to_le_bytes());
            out.extend_from_slice(&i.to_le_bytes());
        }),
        DType::I32 => t.to_vec::<i32>()?.iter().for_each(|v| out.extend_from_slice(&v.to_le_bytes())),
        DType::I64 => t.to_vec::<i64>()?.iter().for_each(|v| out.extend_from_slice(&v.to_le_bytes())),
        DType::U8 => out.extend(t.to_vec::<u8>()?),
        DType::I8 => out.extend(t.to_vec::<i8>()?.into_iter().map(|v| v as u8)),
        DType::I16 => t.to_vec::<i16>()?.iter().for_each(|v| out.extend_from_slice(&v.to_le_bytes())),
        DType::U16 => t.to_vec::<u16>()?.iter().for_each(|v| out.extend_from_slice(&v.to_le_bytes())),
        DType::U32 => t.to_vec::<u32>()?.iter().for_each(|v| out.extend_from_slice(&v.to_le_bytes())),
        DType::U64 => t.to_vec::<u64>()?.iter().for_each(|v| out.extend_from_slice(&v.to_le_bytes())),
        DType::Bool => out.extend(t.to_vec::<bool>()?.into_iter().map(u8::from)),
        DType::BF16 | DType::F8E4M3 | DType::F8E5M2 => unreachable!("rejected above"),
    }
    Ok(out)
}

pub fn save_npy(t: &Tensor, path: impl AsRef<Path>) -> Result<()> {
    std::fs::write(path, npy_bytes(t)?).map_err(|e| err(format!("save_npy: {e}")))
}

struct Header {
    descr: String,
    fortran: bool,
    shape: Vec<usize>,
}

fn parse_header(text: &str) -> Result<Header> {
    let field = |key: &str| -> Result<&str> {
        let at = text.find(&format!("'{key}'")).ok_or_else(|| err(format!("npy header lacks '{key}'")))?;
        let rest = &text[at + key.len() + 2..];
        let colon = rest.find(':').ok_or_else(|| err("malformed npy header"))?;
        Ok(rest[colon + 1..].trim_start())
    };
    let d = field("descr")?;
    let descr = d
        .strip_prefix('\'')
        .and_then(|r| r.split('\'').next())
        .or_else(|| d.strip_prefix('"').and_then(|r| r.split('"').next()))
        .ok_or_else(|| err("unsupported npy descr (structured arrays are not supported)"))?
        .to_string();
    let fortran = field("fortran_order")?.starts_with("True");
    let s = field("shape")?;
    let inner = s
        .strip_prefix('(')
        .and_then(|r| r.split(')').next())
        .ok_or_else(|| err("malformed npy shape"))?;
    let shape = inner
        .split(',')
        .map(str::trim)
        .filter(|p| !p.is_empty())
        .map(|p| p.trim_end_matches('L').parse::<usize>().map_err(|_| err("malformed npy shape")))
        .collect::<Result<Vec<_>>>()?;
    Ok(Header { descr, fortran, shape })
}

/// What the header of an `.npy` file says (`data_offset` is where the raw array starts).
pub struct NpyInfo {
    pub descr: String,
    pub fortran: bool,
    pub shape: Vec<usize>,
    pub data_offset: usize,
}

/// Reads just the header from the first bytes of an `.npy` file.
pub fn npy_header(bytes: &[u8]) -> Result<NpyInfo> {
    if bytes.len() < 10 || &bytes[..6] != MAGIC {
        return Err(err("not an .npy file (bad magic)"));
    }
    let (hl, start) = match bytes[6] {
        1 => (u16::from_le_bytes([bytes[8], bytes[9]]) as usize, 10),
        2 | 3 if bytes.len() >= 12 => (u32::from_le_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]) as usize, 12),
        _ => return Err(err("unsupported or truncated .npy header")),
    };
    let end = start + hl;
    if bytes.len() < end {
        return Err(err("truncated .npy header"));
    }
    let h = parse_header(std::str::from_utf8(&bytes[start..end]).map_err(|_| err("npy header is not valid UTF-8"))?)?;
    if h.descr.len() < 3 {
        return Err(err(format!("unsupported npy dtype '{}'", h.descr)));
    }
    Ok(NpyInfo { descr: h.descr, fortran: h.fortran, shape: h.shape, data_offset: end })
}

/// Parses `.npy` bytes into a tensor.
pub fn from_npy_bytes(bytes: &[u8]) -> Result<Tensor> {
    if bytes.len() < 10 || &bytes[..6] != MAGIC {
        return Err(err("not an .npy file (bad magic)"));
    }
    let (major, header_len, start) = match bytes[6] {
        1 => (1, u16::from_le_bytes([bytes[8], bytes[9]]) as usize, 10),
        2 | 3 => {
            if bytes.len() < 12 {
                return Err(err("truncated .npy header"));
            }
            (bytes[6], u32::from_le_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]) as usize, 12)
        }
        v => return Err(err(format!("unsupported .npy version {v}"))),
    };
    let _ = major;
    let body_at = start + header_len;
    if bytes.len() < body_at {
        return Err(err("truncated .npy header"));
    }
    let text = std::str::from_utf8(&bytes[start..body_at]).map_err(|_| err("npy header is not valid UTF-8"))?;
    let h = parse_header(text)?;
    let data = &bytes[body_at..];
    let count: usize = h.shape.iter().product();
    let d = h.descr.as_bytes();
    if d.len() < 3 {
        return Err(err(format!("unsupported npy dtype '{}'", h.descr)));
    }
    let big = d[0] == b'>';
    let kind = d[1];
    let size: usize = h.descr[2..].parse().map_err(|_| err(format!("unsupported npy dtype '{}'", h.descr)))?;
    if data.len() < count * size {
        return Err(err(format!("npy data holds {} bytes, expected {}", data.len(), count * size)));
    }
    let chunk = |i: usize| -> Vec<u8> {
        let mut c = data[i * size..(i + 1) * size].to_vec();
        if big {
            c.reverse();
        }
        c
    };
    let t = match (kind, size) {
        (b'f', 2) => {
            let v: Vec<f16> = (0..count).map(|i| { let c = chunk(i); f16::from_bits(u16::from_le_bytes([c[0], c[1]])) }).collect();
            Tensor::from_slice(&v, &h.shape)?
        }
        (b'f', 4) => {
            let v: Vec<f32> = (0..count).map(|i| f32::from_le_bytes(chunk(i).try_into().unwrap())).collect();
            Tensor::from_slice(&v, &h.shape)?
        }
        (b'f', 8) => {
            let v: Vec<f64> = (0..count).map(|i| f64::from_le_bytes(chunk(i).try_into().unwrap())).collect();
            Tensor::from_slice(&v, &h.shape)?
        }
        (b'i', 1) => Tensor::from_slice(&(0..count).map(|i| data[i] as i8).collect::<Vec<_>>(), &h.shape)?,
        (b'i', 2) => Tensor::from_slice(&(0..count).map(|i| { let c = chunk(i); i16::from_le_bytes([c[0], c[1]]) }).collect::<Vec<_>>(), &h.shape)?,
        (b'i', 4) => Tensor::from_slice(&(0..count).map(|i| i32::from_le_bytes(chunk(i).try_into().unwrap())).collect::<Vec<_>>(), &h.shape)?,
        (b'i', 8) => Tensor::from_slice(&(0..count).map(|i| i64::from_le_bytes(chunk(i).try_into().unwrap())).collect::<Vec<_>>(), &h.shape)?,
        (b'u', 1) => Tensor::from_slice(&data[..count], &h.shape)?,
        (b'u', 2) => Tensor::from_slice(&(0..count).map(|i| { let c = chunk(i); u16::from_le_bytes([c[0], c[1]]) }).collect::<Vec<_>>(), &h.shape)?,
        (b'u', 4) => Tensor::from_slice(&(0..count).map(|i| u32::from_le_bytes(chunk(i).try_into().unwrap())).collect::<Vec<_>>(), &h.shape)?,
        (b'u', 8) => Tensor::from_slice(&(0..count).map(|i| u64::from_le_bytes(chunk(i).try_into().unwrap())).collect::<Vec<_>>(), &h.shape)?,
        (b'b', 1) => Tensor::from_slice(&data[..count].iter().map(|&b| b != 0).collect::<Vec<_>>(), &h.shape)?,
        _ => return Err(err(format!("unsupported npy dtype '{}'", h.descr))),
    };
    if h.fortran && h.shape.len() > 1 {
        // the buffer is column-major: reinterpret with reversed dims, then reverse the axes
        let rev: Vec<usize> = h.shape.iter().rev().copied().collect();
        let axes: Vec<usize> = (0..h.shape.len()).rev().collect();
        return t.reshape(&rev)?.permute(&axes)?.contiguous();
    }
    Ok(t)
}

pub fn load_npy(path: impl AsRef<Path>) -> Result<Tensor> {
    from_npy_bytes(&std::fs::read(path).map_err(|e| err(format!("load_npy: {e}")))?)
}

// ---------------------------------------------------------------------------------------------------------------
// ZIP container (.npz)
// ---------------------------------------------------------------------------------------------------------------

fn crc32(data: &[u8]) -> u32 {
    let mut table = [0u32; 256];
    for (i, slot) in table.iter_mut().enumerate() {
        let mut c = i as u32;
        for _ in 0..8 {
            c = if c & 1 != 0 { 0xEDB8_8320 ^ (c >> 1) } else { c >> 1 };
        }
        *slot = c;
    }
    !data.iter().fold(!0u32, |c, &b| table[((c ^ u32::from(b)) & 0xFF) as usize] ^ (c >> 8))
}

/// Serializes named tensors as an uncompressed `.npz` archive (entry `name` becomes `name.npy`).
pub fn npz_bytes(tensors: &HashMap<String, Tensor>) -> Result<Vec<u8>> {
    let mut names: Vec<&String> = tensors.keys().collect();
    names.sort();
    let mut out: Vec<u8> = Vec::new();
    let mut central: Vec<u8> = Vec::new();
    for name in names {
        let data = npy_bytes(&tensors[name])?;
        let fname = format!("{name}.npy");
        let crc = crc32(&data);
        let offset = out.len() as u32;
        let size = data.len() as u32;
        let (n, e) = (fname.len() as u16, 0u16);
        // local file header
        out.extend_from_slice(&0x0403_4b50u32.to_le_bytes());
        out.extend_from_slice(&[20, 0, 0, 0, 0, 0, 0, 0, 0, 0]); // version, flags, method (stored), time, date
        out.extend_from_slice(&crc.to_le_bytes());
        out.extend_from_slice(&size.to_le_bytes());
        out.extend_from_slice(&size.to_le_bytes());
        out.extend_from_slice(&n.to_le_bytes());
        out.extend_from_slice(&e.to_le_bytes());
        out.extend_from_slice(fname.as_bytes());
        out.extend_from_slice(&data);
        // central directory entry
        central.extend_from_slice(&0x0201_4b50u32.to_le_bytes());
        central.extend_from_slice(&[20, 0, 20, 0, 0, 0, 0, 0, 0, 0, 0, 0]);
        central.extend_from_slice(&crc.to_le_bytes());
        central.extend_from_slice(&size.to_le_bytes());
        central.extend_from_slice(&size.to_le_bytes());
        central.extend_from_slice(&n.to_le_bytes());
        central.extend_from_slice(&[0; 8]); // extra len, comment len, disk, internal attrs
        central.extend_from_slice(&0u32.to_le_bytes()); // external attrs
        central.extend_from_slice(&offset.to_le_bytes());
        central.extend_from_slice(fname.as_bytes());
    }
    let cd_offset = out.len() as u32;
    let cd_size = central.len() as u32;
    let count = tensors.len() as u16;
    out.extend(central);
    out.extend_from_slice(&0x0605_4b50u32.to_le_bytes());
    out.extend_from_slice(&[0, 0, 0, 0]);
    out.extend_from_slice(&count.to_le_bytes());
    out.extend_from_slice(&count.to_le_bytes());
    out.extend_from_slice(&cd_size.to_le_bytes());
    out.extend_from_slice(&cd_offset.to_le_bytes());
    out.extend_from_slice(&0u16.to_le_bytes());
    Ok(out)
}

pub fn save_npz(tensors: &HashMap<String, Tensor>, path: impl AsRef<Path>) -> Result<()> {
    std::fs::write(path, npz_bytes(tensors)?).map_err(|e| err(format!("save_npz: {e}")))
}

fn u16_at(b: &[u8], i: usize) -> Result<usize> {
    b.get(i..i + 2).map(|s| u16::from_le_bytes([s[0], s[1]]) as usize).ok_or_else(|| err("truncated zip archive"))
}

fn u32_at(b: &[u8], i: usize) -> Result<usize> {
    b.get(i..i + 4)
        .map(|s| u32::from_le_bytes([s[0], s[1], s[2], s[3]]) as usize)
        .ok_or_else(|| err("truncated zip archive"))
}

/// Every entry of a zip archive (stored or deflated) as `(name, bytes)`, in directory order.
pub(crate) fn zip_entries(bytes: &[u8]) -> Result<Vec<(String, Vec<u8>)>> {
    // end-of-central-directory record: search backwards for its signature
    let eocd = (0..bytes.len().saturating_sub(21))
        .rev()
        .find(|&i| bytes[i..i + 4] == [0x50, 0x4b, 0x05, 0x06])
        .ok_or_else(|| err("not a zip archive (no end-of-central-directory record)"))?;
    let count = u16_at(bytes, eocd + 10)?;
    let mut at = u32_at(bytes, eocd + 16)?;
    let mut out = Vec::with_capacity(count);
    for _ in 0..count {
        if u32_at(bytes, at)? != 0x0201_4b50 {
            return Err(err("corrupt zip central directory"));
        }
        let method = u16_at(bytes, at + 10)?;
        let csize = u32_at(bytes, at + 20)?;
        let usize_ = u32_at(bytes, at + 24)?;
        let (n, m, k) = (u16_at(bytes, at + 28)?, u16_at(bytes, at + 30)?, u16_at(bytes, at + 32)?);
        let local = u32_at(bytes, at + 42)?;
        let name = std::str::from_utf8(bytes.get(at + 46..at + 46 + n).ok_or_else(|| err("truncated zip archive"))?)
            .map_err(|_| err("zip entry name is not UTF-8"))?
            .to_string();
        at += 46 + n + m + k;
        if csize == 0xFFFF_FFFF || usize_ == 0xFFFF_FFFF {
            return Err(err("zip64 archives are not supported"));
        }
        if u32_at(bytes, local)? != 0x0403_4b50 {
            return Err(err("corrupt zip local header"));
        }
        let data_at = local + 30 + u16_at(bytes, local + 26)? + u16_at(bytes, local + 28)?;
        let raw = bytes.get(data_at..data_at + csize).ok_or_else(|| err("truncated zip entry"))?;
        let data = match method {
            0 => raw.to_vec(),
            8 => inflate(raw, usize_)?,
            other => return Err(err(format!("unsupported zip compression method {other}"))),
        };
        if data.len() != usize_ {
            return Err(err(format!("zip entry '{name}' has {} bytes, expected {usize_}", data.len())));
        }
        out.push((name, data));
    }
    Ok(out)
}

/// Reads every `.npy` entry of an `.npz` archive.
pub fn from_npz_bytes(bytes: &[u8]) -> Result<HashMap<String, Tensor>> {
    let mut out = HashMap::new();
    for (name, data) in zip_entries(bytes)? {
        if let Some(stem) = name.strip_suffix(".npy") {
            out.insert(stem.to_string(), from_npy_bytes(&data)?);
        }
    }
    Ok(out)
}

pub fn load_npz(path: impl AsRef<Path>) -> Result<HashMap<String, Tensor>> {
    from_npz_bytes(&std::fs::read(path).map_err(|e| err(format!("load_npz: {e}")))?)
}

// ---------------------------------------------------------------------------------------------------------------
// DEFLATE (RFC 1951) decoder
// ---------------------------------------------------------------------------------------------------------------

struct Bits<'a> {
    data: &'a [u8],
    pos: usize,
    bit: u32,
    nbits: u32,
}

impl Bits<'_> {
    fn need(&mut self, n: u32) -> Result<()> {
        while self.nbits < n {
            let b = *self.data.get(self.pos).ok_or_else(|| err("deflate stream ended early"))?;
            self.pos += 1;
            self.bit |= u32::from(b) << self.nbits;
            self.nbits += 8;
        }
        Ok(())
    }
    fn bits(&mut self, n: u32) -> Result<u32> {
        if n == 0 {
            return Ok(0);
        }
        self.need(n)?;
        let v = self.bit & ((1u32 << n) - 1);
        self.bit >>= n;
        self.nbits -= n;
        Ok(v)
    }
}

struct Huffman {
    count: [u16; 16],
    symbol: Vec<u16>,
}

impl Huffman {
    fn new(lengths: &[u8]) -> Huffman {
        let mut count = [0u16; 16];
        for &l in lengths {
            count[l as usize] += 1;
        }
        let mut offs = [0u16; 16];
        for i in 1..15 {
            offs[i + 1] = offs[i] + count[i];
        }
        let mut symbol = vec![0u16; lengths.len()];
        for (s, &l) in lengths.iter().enumerate() {
            if l != 0 {
                symbol[offs[l as usize] as usize] = s as u16;
                offs[l as usize] += 1;
            }
        }
        Huffman { count, symbol }
    }

    fn decode(&self, b: &mut Bits) -> Result<usize> {
        let (mut code, mut first, mut index) = (0i32, 0i32, 0i32);
        for len in 1..16 {
            code |= b.bits(1)? as i32;
            let count = i32::from(self.count[len]);
            if code - count < first {
                return Ok(self.symbol[(index + (code - first)) as usize] as usize);
            }
            index += count;
            first += count;
            first <<= 1;
            code <<= 1;
        }
        Err(err("invalid Huffman code in deflate stream"))
    }
}

const LEN_BASE: [u16; 29] = [3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 15, 17, 19, 23, 27, 31, 35, 43, 51, 59, 67, 83, 99, 115, 131, 163, 195, 227, 258];
const LEN_EXTRA: [u8; 29] = [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 0];
const DIST_BASE: [u16; 30] = [1, 2, 3, 4, 5, 7, 9, 13, 17, 25, 33, 49, 65, 97, 129, 193, 257, 385, 513, 769, 1025, 1537, 2049, 3073, 4097, 6145, 8193, 12289, 16385, 24577];
const DIST_EXTRA: [u8; 30] = [0, 0, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 10, 10, 11, 11, 12, 12, 13, 13];

fn inflate_codes(b: &mut Bits, out: &mut Vec<u8>, lit: &Huffman, dist: &Huffman) -> Result<()> {
    loop {
        let sym = lit.decode(b)?;
        match sym {
            0..=255 => out.push(sym as u8),
            256 => return Ok(()),
            257..=285 => {
                let i = sym - 257;
                let len = LEN_BASE[i] as usize + b.bits(u32::from(LEN_EXTRA[i]))? as usize;
                let d = dist.decode(b)?;
                if d >= 30 {
                    return Err(err("invalid deflate distance code"));
                }
                let distance = DIST_BASE[d] as usize + b.bits(u32::from(DIST_EXTRA[d]))? as usize;
                if distance > out.len() {
                    return Err(err("deflate distance reaches before the start of the output"));
                }
                for _ in 0..len {
                    let v = out[out.len() - distance];
                    out.push(v);
                }
            }
            _ => return Err(err("invalid deflate length code")),
        }
    }
}

/// Decompresses a raw DEFLATE stream (as stored in ZIP entries).
pub fn inflate(data: &[u8], size_hint: usize) -> Result<Vec<u8>> {
    let mut b = Bits { data, pos: 0, bit: 0, nbits: 0 };
    let mut out = Vec::with_capacity(size_hint);
    loop {
        let last = b.bits(1)?;
        match b.bits(2)? {
            0 => {
                // stored block: skip to a byte boundary
                b.bit = 0;
                b.nbits = 0;
                let len = u16_at(data, b.pos)?;
                let nlen = u16_at(data, b.pos + 2)?;
                if len != (!nlen) & 0xFFFF {
                    return Err(err("corrupt stored deflate block"));
                }
                let start = b.pos + 4;
                out.extend_from_slice(data.get(start..start + len).ok_or_else(|| err("truncated stored block"))?);
                b.pos = start + len;
            }
            1 => {
                let mut l = [0u8; 288];
                l[..144].fill(8);
                l[144..256].fill(9);
                l[256..280].fill(7);
                l[280..].fill(8);
                inflate_codes(&mut b, &mut out, &Huffman::new(&l), &Huffman::new(&[5u8; 30]))?;
            }
            2 => {
                let (nlen, ndist, ncode) = (b.bits(5)? as usize + 257, b.bits(5)? as usize + 1, b.bits(4)? as usize + 4);
                const ORDER: [usize; 19] = [16, 17, 18, 0, 8, 7, 9, 6, 10, 5, 11, 4, 12, 3, 13, 2, 14, 1, 15];
                let mut cl = [0u8; 19];
                for &o in ORDER.iter().take(ncode) {
                    cl[o] = b.bits(3)? as u8;
                }
                let code_huff = Huffman::new(&cl);
                let mut lengths = vec![0u8; nlen + ndist];
                let mut i = 0;
                while i < nlen + ndist {
                    let sym = code_huff.decode(&mut b)?;
                    if sym < 16 {
                        lengths[i] = sym as u8;
                        i += 1;
                    } else {
                        let (prev, rep) = match sym {
                            16 => {
                                if i == 0 {
                                    return Err(err("deflate repeats a length before any was set"));
                                }
                                (lengths[i - 1], 3 + b.bits(2)? as usize)
                            }
                            17 => (0, 3 + b.bits(3)? as usize),
                            _ => (0, 11 + b.bits(7)? as usize),
                        };
                        if i + rep > nlen + ndist {
                            return Err(err("too many code lengths in a deflate block"));
                        }
                        lengths[i..i + rep].fill(prev);
                        i += rep;
                    }
                }
                inflate_codes(&mut b, &mut out, &Huffman::new(&lengths[..nlen]), &Huffman::new(&lengths[nlen..]))?;
            }
            _ => return Err(err("invalid deflate block type")),
        }
        if last == 1 {
            return Ok(out);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn v32(t: &Tensor) -> Vec<f32> {
        t.to_vec::<f32>().unwrap()
    }
    const DIR: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/src/serialize/testdata");

    #[test]
    fn round_trips_every_dtype() {
        let cases: Vec<Tensor> = vec![
            Tensor::from_slice(&[1.5f32, -2.0, 3.25, 0.0, 7.0, 8.0], &[2, 3]).unwrap(),
            Tensor::from_slice(&[1.0f64, 2.0], &[2]).unwrap(),
            Tensor::from_slice(&[-1i32, 5, 7], &[3]).unwrap(),
            Tensor::from_slice(&[i64::MIN, 0, i64::MAX], &[3]).unwrap(),
            Tensor::from_slice(&[0u8, 255, 3], &[3]).unwrap(),
            Tensor::from_slice(&[true, false], &[2]).unwrap(),
            Tensor::from_slice(&[f16::from_f32(0.5), f16::from_f32(-2.0)], &[2]).unwrap(),
            Tensor::from_slice(&[4.0f32], &[]).unwrap(),
            Tensor::zeros(&[0, 3], DType::F32, &crate::backend::Device::Cpu).unwrap(),
        ];
        for t in cases {
            let back = from_npy_bytes(&npy_bytes(&t).unwrap()).unwrap();
            assert_eq!(back.shape(), t.shape());
            assert_eq!(back.dtype(), t.dtype());
            assert_eq!(back.to_vec::<f64>().unwrap(), t.to_vec::<f64>().unwrap());
        }
        // header length is a multiple of 64 like NumPy's
        assert_eq!(npy_bytes(&Tensor::from_slice(&[1.0f32], &[1]).unwrap()).unwrap().len() % 64, 4);
        assert!(npy_bytes(&Tensor::from_slice(&[1.0f32], &[1]).unwrap().to_dtype(DType::BF16).unwrap()).is_err());
    }

    #[test]
    fn reads_files_written_by_numpy() {
        let f = load_npy(format!("{DIR}/fortran_i32.npy")).unwrap();
        assert_eq!(f.shape(), [2, 3]);
        assert_eq!(f.to_vec::<i32>().unwrap(), vec![0, 1, 2, 3, 4, 5]); // arange(6).reshape(2, 3), stored column-major
        let be = load_npy(format!("{DIR}/big_endian_f4.npy")).unwrap();
        assert_eq!(v32(&be), vec![0.0, 1.0, 2.0, 3.0]);
        let i16s = load_npy(format!("{DIR}/int16.npy")).unwrap();
        assert_eq!((i16s.dtype(), i16s.to_vec::<i32>().unwrap()), (DType::I16, vec![-3, 7, 300]));
    }

    #[test]
    fn reads_stored_and_deflated_archives() {
        for file in ["sample_stored.npz", "sample_deflated.npz"] {
            let z = load_npz(format!("{DIR}/{file}")).unwrap();
            assert_eq!(v32(&z["a"]), (0..12).map(|i| i as f32).collect::<Vec<_>>(), "{file}");
            assert_eq!(z["a"].shape(), [3, 4]);
            assert_eq!(z["b"].to_vec::<i64>().unwrap(), vec![0, 3, 6, 9, 12]);
            assert_eq!(z["c"].to_vec::<bool>().unwrap(), vec![true, false, true]);
        }
        let stored = load_npz(format!("{DIR}/sample_stored.npz")).unwrap();
        assert_eq!(stored["d"].shape(), Vec::<usize>::new());
        assert_eq!(stored["d"].to_vec::<f64>().unwrap(), vec![2.5]);
        assert_eq!(stored["u8"].to_vec::<u8>().unwrap(), (0..7).collect::<Vec<u8>>());
        assert_eq!(stored["h"].to_vec::<f64>().unwrap(), vec![0.0, 0.25, 0.5, 0.75]);
        let big = &load_npz(format!("{DIR}/sample_deflated.npz")).unwrap()["big"];
        assert_eq!(big.shape(), [5000]);
        let v = v32(big);
        assert!(v.iter().enumerate().all(|(i, &x)| x == (i % 100) as f32));
    }

    #[test]
    fn npz_round_trip_and_crc() {
        let mut m = HashMap::new();
        m.insert("w".to_string(), Tensor::from_slice(&[1.0f32, 2.0, 3.0, 4.0], &[2, 2]).unwrap());
        m.insert("idx".to_string(), Tensor::from_slice(&[7i64, 8], &[2]).unwrap());
        let back = from_npz_bytes(&npz_bytes(&m).unwrap()).unwrap();
        assert_eq!(v32(&back["w"]), vec![1.0, 2.0, 3.0, 4.0]);
        assert_eq!(back["idx"].to_vec::<i64>().unwrap(), vec![7, 8]);
        assert_eq!(crc32(b"123456789"), 0xCBF4_3926);
        // the archive we write is a valid zip for NumPy-style readers: re-reading our own bytes through the stored path
        assert!(from_npz_bytes(b"not a zip").is_err());
    }

    #[test]
    fn inflate_known_streams() {
        // raw deflate of "hello" (fixed Huffman)
        assert_eq!(inflate(&[0xcb, 0x48, 0xcd, 0xc9, 0xc9, 0x07, 0x00], 5).unwrap(), b"hello");
        // a stored block
        assert_eq!(inflate(&[0x01, 0x03, 0x00, 0xfc, 0xff, b'a', b'b', b'c'], 3).unwrap(), b"abc");
        assert!(inflate(&[0x07], 0).is_err()); // invalid block type 3
    }
}

#[cfg(test)]
mod interop_files {
    use super::*;

    /// Leaves `oxideon_interop.npz` / `.npy` in the temp dir so an external NumPy can check them.
    #[test]
    fn writes_files_for_numpy() {
        let mut m = HashMap::new();
        m.insert("w".to_string(), Tensor::from_slice(&[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]).unwrap());
        m.insert("idx".to_string(), Tensor::from_slice(&[7i64, 8], &[2]).unwrap());
        save_npz(&m, std::env::temp_dir().join("oxideon_interop.npz")).unwrap();
        save_npy(&m["w"], std::env::temp_dir().join("oxideon_interop.npy")).unwrap();
    }
}
