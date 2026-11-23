//! Reader for PyTorch checkpoints (`torch.save`, `.pt` / `.pth` / `.bin`): a zip archive holding a pickle (`data.pkl`) and
//! one raw file per tensor storage. The pickle is interpreted by a small virtual machine that understands only the handful
//! of globals tensors and state dicts use (`OrderedDict`, `torch._utils._rebuild_tensor_v2`, storage classes, ...); nothing
//! else is ever called, so loading an untrusted checkpoint cannot execute code (like `torch.load(weights_only=True)`).
//!
//! Strided and shared-storage tensors are materialized contiguously. Supported dtypes: float64 / float32 / float16 /
//! bfloat16, int64 / int32, uint8 and bool (int16 / int8 have no counterpart in this library and report an error).

use super::npy::zip_entries;
use crate::dtype::DType;
use crate::error::{Error, Result};
use crate::tensor::Tensor;
use std::collections::HashMap;

fn err(msg: impl Into<String>) -> Error {
    Error::Msg(format!("torch checkpoint: {}", msg.into()))
}

/// Pickle values; scalars other than sizes are parsed but not needed.
#[allow(dead_code)]
#[derive(Debug, Clone)]
enum Val {
    None,
    Bool(bool),
    Int(i64),
    Float(f64),
    Str(String),
    Bytes(Vec<u8>),
    List(Vec<Val>),
    Tuple(Vec<Val>),
    Dict(Vec<(Val, Val)>),
    Global(String, String),
    Storage { dtype: Option<DType>, type_name: String, key: String },
    Tensor { key: String, dtype: Option<DType>, type_name: String, offset: usize, size: Vec<usize>, stride: Vec<usize> },
    Obj,
    Mark,
}

fn storage_dtype(class: &str) -> Option<DType> {
    Some(match class {
        "FloatStorage" | "float32" => DType::F32,
        "DoubleStorage" | "float64" => DType::F64,
        "HalfStorage" | "float16" => DType::F16,
        "BFloat16Storage" | "bfloat16" => DType::BF16,
        "float8_e4m3fn" => DType::F8E4M3,
        "float8_e5m2" => DType::F8E5M2,
        "LongStorage" | "int64" => DType::I64,
        "IntStorage" | "int32" => DType::I32,
        "ByteStorage" | "uint8" => DType::U8,
        "CharStorage" | "int8" => DType::I8,
        "ShortStorage" | "int16" => DType::I16,
        "BoolStorage" | "bool" => DType::Bool,
        _ => return None,
    })
}

struct Vm<'a> {
    b: &'a [u8],
    i: usize,
    stack: Vec<Val>,
    memo: HashMap<usize, Val>,
}

impl Vm<'_> {
    fn take(&mut self, n: usize) -> Result<&[u8]> {
        let s = self.b.get(self.i..self.i + n).ok_or_else(|| err("truncated pickle"))?;
        self.i += n;
        Ok(s)
    }
    fn u8(&mut self) -> Result<u8> {
        Ok(self.take(1)?[0])
    }
    fn le(&mut self, n: usize) -> Result<u64> {
        Ok(self.take(n)?.iter().rev().fold(0u64, |a, &b| (a << 8) | b as u64))
    }
    fn line(&mut self) -> Result<String> {
        let st = self.i;
        while *self.b.get(self.i).ok_or_else(|| err("truncated pickle"))? != b'\n' {
            self.i += 1;
        }
        let s = String::from_utf8_lossy(&self.b[st..self.i]).to_string();
        self.i += 1;
        Ok(s)
    }
    fn pop(&mut self) -> Result<Val> {
        self.stack.pop().ok_or_else(|| err("pickle stack underflow"))
    }
    fn pop_mark(&mut self) -> Result<Vec<Val>> {
        let at = self.stack.iter().rposition(|v| matches!(v, Val::Mark)).ok_or_else(|| err("pickle MARK missing"))?;
        let items = self.stack.split_off(at + 1);
        self.stack.pop();
        Ok(items)
    }

    fn ints(v: &Val) -> Result<Vec<usize>> {
        match v {
            Val::Tuple(t) | Val::List(t) => t.iter().map(|x| if let Val::Int(i) = x { usize::try_from(*i).map_err(|_| err("negative size")) } else { Err(err("size entries must be integers")) }).collect(),
            _ => Err(err("expected a tuple of integers")),
        }
    }

    fn reduce(&mut self, f: Val, args: Val) -> Result<Val> {
        let Val::Global(module, name) = f else { return Ok(Val::Obj) };
        let args = match args {
            Val::Tuple(a) => a,
            _ => return Ok(Val::Obj),
        };
        Ok(match (module.as_str(), name.as_str()) {
            ("collections", "OrderedDict") => Val::Dict(vec![]),
            ("torch._utils", "_rebuild_tensor_v2") | ("torch._utils", "_rebuild_tensor") => {
                let (Some(Val::Storage { dtype, type_name, key }), Some(Val::Int(off)), Some(size), Some(stride)) = (args.first().cloned(), args.get(1).cloned(), args.get(2), args.get(3)) else {
                    return Err(err("unexpected arguments to _rebuild_tensor"));
                };
                Val::Tensor { key, dtype, type_name, offset: off as usize, size: Self::ints(size)?, stride: Self::ints(stride)? }
            }
            ("torch._utils", "_rebuild_parameter") | ("torch._utils", "_rebuild_parameter_with_state") => args.into_iter().next().ok_or_else(|| err("empty _rebuild_parameter"))?,
            ("torch._utils", "_rebuild_qtensor") | ("torch._utils", "_rebuild_sparse_tensor") => return Err(err(format!("{name} (quantized / sparse tensors) is not supported"))),
            _ => Val::Obj,
        })
    }

    fn run(&mut self) -> Result<Val> {
        loop {
            let op = self.u8()?;
            match op {
                b'\x80' => {
                    self.u8()?;
                }
                b'\x95' => {
                    self.take(8)?;
                } // FRAME
                b'.' => return self.pop(),
                b'N' => self.stack.push(Val::None),
                b'\x88' => self.stack.push(Val::Bool(true)),
                b'\x89' => self.stack.push(Val::Bool(false)),
                b'K' => {
                    let v = self.u8()?;
                    self.stack.push(Val::Int(v as i64))
                }
                b'M' => {
                    let v = self.le(2)?;
                    self.stack.push(Val::Int(v as i64))
                }
                b'J' => {
                    let v = self.le(4)? as u32 as i32;
                    self.stack.push(Val::Int(v as i64))
                }
                b'\x8a' => {
                    let n = self.u8()? as usize;
                    let raw = self.take(n)?.to_vec();
                    let mut v: i64 = 0;
                    for (k, b) in raw.iter().enumerate().take(8) {
                        v |= (*b as i64) << (8 * k);
                    }
                    if n < 8 && n > 0 && raw[n - 1] & 0x80 != 0 {
                        v |= -1i64 << (8 * n);
                    }
                    self.stack.push(Val::Int(v))
                }
                b'G' => {
                    let b: [u8; 8] = self.take(8)?.try_into().unwrap();
                    self.stack.push(Val::Float(f64::from_be_bytes(b)))
                }
                b'X' => {
                    let n = self.le(4)? as usize;
                    let s = String::from_utf8_lossy(self.take(n)?).to_string();
                    self.stack.push(Val::Str(s))
                }
                b'\x8c' => {
                    let n = self.u8()? as usize;
                    let s = String::from_utf8_lossy(self.take(n)?).to_string();
                    self.stack.push(Val::Str(s))
                }
                b'\x8d' => {
                    let n = self.le(8)? as usize;
                    let s = String::from_utf8_lossy(self.take(n)?).to_string();
                    self.stack.push(Val::Str(s))
                }
                b'U' => {
                    let n = self.u8()? as usize;
                    let s = String::from_utf8_lossy(self.take(n)?).to_string();
                    self.stack.push(Val::Str(s))
                }
                b'T' => {
                    let n = self.le(4)? as usize;
                    let s = String::from_utf8_lossy(self.take(n)?).to_string();
                    self.stack.push(Val::Str(s))
                }
                b'C' => {
                    let n = self.u8()? as usize;
                    let s = self.take(n)?.to_vec();
                    self.stack.push(Val::Bytes(s))
                }
                b'B' => {
                    let n = self.le(4)? as usize;
                    let s = self.take(n)?.to_vec();
                    self.stack.push(Val::Bytes(s))
                }
                b'}' => self.stack.push(Val::Dict(vec![])),
                b']' => self.stack.push(Val::List(vec![])),
                b')' => self.stack.push(Val::Tuple(vec![])),
                b'(' => self.stack.push(Val::Mark),
                b't' => {
                    let items = self.pop_mark()?;
                    self.stack.push(Val::Tuple(items))
                }
                b'\x85' => {
                    let a = self.pop()?;
                    self.stack.push(Val::Tuple(vec![a]))
                }
                b'\x86' => {
                    let b = self.pop()?;
                    let a = self.pop()?;
                    self.stack.push(Val::Tuple(vec![a, b]))
                }
                b'\x87' => {
                    let c = self.pop()?;
                    let b = self.pop()?;
                    let a = self.pop()?;
                    self.stack.push(Val::Tuple(vec![a, b, c]))
                }
                b'l' => {
                    let items = self.pop_mark()?;
                    self.stack.push(Val::List(items))
                }
                b'd' => {
                    let items = self.pop_mark()?;
                    self.stack.push(Val::Dict(items.chunks(2).filter(|c| c.len() == 2).map(|c| (c[0].clone(), c[1].clone())).collect()))
                }
                b'a' => {
                    let v = self.pop()?;
                    if let Some(Val::List(l)) = self.stack.last_mut() {
                        l.push(v)
                    }
                }
                b'e' => {
                    let items = self.pop_mark()?;
                    if let Some(Val::List(l)) = self.stack.last_mut() {
                        l.extend(items)
                    }
                }
                b's' => {
                    let v = self.pop()?;
                    let k = self.pop()?;
                    if let Some(Val::Dict(d)) = self.stack.last_mut() {
                        d.push((k, v))
                    }
                }
                b'u' => {
                    let items = self.pop_mark()?;
                    if let Some(Val::Dict(d)) = self.stack.last_mut() {
                        d.extend(items.chunks(2).filter(|c| c.len() == 2).map(|c| (c[0].clone(), c[1].clone())))
                    }
                }
                b'c' => {
                    let m = self.line()?;
                    let n = self.line()?;
                    self.stack.push(Val::Global(m, n))
                }
                b'\x93' => {
                    let n = self.pop()?;
                    let m = self.pop()?;
                    match (m, n) {
                        (Val::Str(m), Val::Str(n)) => self.stack.push(Val::Global(m, n)),
                        _ => return Err(err("STACK_GLOBAL needs two strings")),
                    }
                }
                b'R' => {
                    let args = self.pop()?;
                    let f = self.pop()?;
                    let v = self.reduce(f, args)?;
                    self.stack.push(v)
                }
                b'\x81' => {
                    // NEWOBJ cls args: treat as an opaque object
                    self.pop()?;
                    self.pop()?;
                    self.stack.push(Val::Obj)
                }
                b'b' => {
                    self.pop()?; // BUILD: object state (e.g. a state dict's _metadata) is ignored
                }
                b'Q' => {
                    // BINPERSID: ('storage', storage_type, key, location, numel)
                    let pid = self.pop()?;
                    let Val::Tuple(t) = pid else { return Err(err("unexpected persistent id")) };
                    match (t.first(), t.get(1), t.get(2)) {
                        (Some(Val::Str(tag)), Some(ty), Some(Val::Str(key))) if tag == "storage" => {
                            let type_name = match ty {
                                Val::Global(_, n) => n.clone(),
                                _ => String::new(),
                            };
                            self.stack.push(Val::Storage { dtype: storage_dtype(&type_name), type_name, key: key.clone() })
                        }
                        _ => return Err(err("unsupported persistent id")),
                    }
                }
                b'q' => {
                    let k = self.u8()? as usize;
                    let v = self.stack.last().cloned().ok_or_else(|| err("BINPUT on an empty stack"))?;
                    self.memo.insert(k, v);
                }
                b'r' => {
                    let k = self.le(4)? as usize;
                    let v = self.stack.last().cloned().ok_or_else(|| err("LONG_BINPUT on an empty stack"))?;
                    self.memo.insert(k, v);
                }
                b'\x94' => {
                    let v = self.stack.last().cloned().ok_or_else(|| err("MEMOIZE on an empty stack"))?;
                    let k = self.memo.len();
                    self.memo.insert(k, v);
                }
                b'h' => {
                    let k = self.u8()? as usize;
                    let v = self.memo.get(&k).cloned().ok_or_else(|| err("BINGET of an unknown memo slot"))?;
                    self.stack.push(v)
                }
                b'j' => {
                    let k = self.le(4)? as usize;
                    let v = self.memo.get(&k).cloned().ok_or_else(|| err("LONG_BINGET of an unknown memo slot"))?;
                    self.stack.push(v)
                }
                b'0' => {
                    self.pop()?;
                }
                b'2' => {
                    let v = self.stack.last().cloned().ok_or_else(|| err("DUP on an empty stack"))?;
                    self.stack.push(v)
                }
                other => return Err(err(format!("unsupported pickle opcode 0x{other:02x}"))),
            }
        }
    }
}

fn to_tensor(raw: &[u8], dtype: DType, offset: usize, size: &[usize], stride: &[usize]) -> Result<Tensor> {
    let esz = match dtype {
        DType::C128 => 16,
        DType::F64 | DType::I64 | DType::U64 | DType::C64 => 8,
        DType::F32 | DType::I32 | DType::U32 => 4,
        DType::F16 | DType::BF16 | DType::I16 | DType::U16 => 2,
        DType::U8 | DType::I8 | DType::Bool | DType::F8E4M3 | DType::F8E5M2 => 1,
    };
    let numel: usize = size.iter().product();
    // element index (in storage elements) of every output element, row-major over `size`
    let mut data: Vec<u8> = Vec::with_capacity(numel * esz);
    let mut idx = vec![0usize; size.len()];
    for _ in 0..numel {
        let at = offset + idx.iter().zip(stride).map(|(i, s)| i * s).sum::<usize>();
        data.extend_from_slice(raw.get(at * esz..(at + 1) * esz).ok_or_else(|| err("tensor reads past the end of its storage"))?);
        for k in (0..size.len()).rev() {
            idx[k] += 1;
            if idx[k] < size[k] {
                break;
            }
            idx[k] = 0;
        }
    }
    let c = |w: usize| data.chunks_exact(w);
    Ok(match dtype {
        DType::F32 => Tensor::from_vec(c(4).map(|b| f32::from_le_bytes(b.try_into().unwrap())).collect::<Vec<_>>(), size)?,
        DType::F64 => Tensor::from_vec(c(8).map(|b| f64::from_le_bytes(b.try_into().unwrap())).collect::<Vec<_>>(), size)?,
        DType::I64 => Tensor::from_vec(c(8).map(|b| i64::from_le_bytes(b.try_into().unwrap())).collect::<Vec<_>>(), size)?,
        DType::I32 => Tensor::from_vec(c(4).map(|b| i32::from_le_bytes(b.try_into().unwrap())).collect::<Vec<_>>(), size)?,
        DType::U8 => Tensor::from_vec(data.clone(), size)?,
        DType::I8 => Tensor::from_vec(data.iter().map(|&b| b as i8).collect::<Vec<_>>(), size)?,
        DType::I16 => Tensor::from_vec(c(2).map(|b| i16::from_le_bytes([b[0], b[1]])).collect::<Vec<_>>(), size)?,
        DType::U16 => Tensor::from_vec(c(2).map(|b| u16::from_le_bytes([b[0], b[1]])).collect::<Vec<_>>(), size)?,
        DType::U32 => Tensor::from_vec(c(4).map(|b| u32::from_le_bytes(b.try_into().unwrap())).collect::<Vec<_>>(), size)?,
        DType::U64 => Tensor::from_vec(c(8).map(|b| u64::from_le_bytes(b.try_into().unwrap())).collect::<Vec<_>>(), size)?,
        DType::Bool => Tensor::from_vec(data.iter().map(|&b| b != 0).collect::<Vec<bool>>(), size)?,
        // half precision values widen exactly to f32 and narrow back without loss
        DType::F16 => {
            let v: Vec<f32> = c(2).map(|b| half_to_f32(u16::from_le_bytes([b[0], b[1]]))).collect();
            Tensor::from_vec(v, size)?.to_dtype(DType::F16)?
        }
        DType::F8E4M3 => Tensor::from_vec(data.iter().map(|&b| crate::dtype::f8e4m3::from_bits(b)).collect::<Vec<_>>(), size)?,
        DType::F8E5M2 => Tensor::from_vec(data.iter().map(|&b| crate::dtype::f8e5m2::from_bits(b)).collect::<Vec<_>>(), size)?,
        DType::BF16 => {
            let v: Vec<f32> = c(2).map(|b| f32::from_bits((u16::from_le_bytes([b[0], b[1]]) as u32) << 16)).collect();
            Tensor::from_vec(v, size)?.to_dtype(DType::BF16)?
        }
        DType::C64 => Tensor::from_c64_vec(
            c(8)
                .map(|b| [
                    f32::from_le_bytes(b[0..4].try_into().unwrap()),
                    f32::from_le_bytes(b[4..8].try_into().unwrap()),
                ])
                .collect(),
            size,
        )?,
        DType::C128 => Tensor::from_c128_vec(
            c(16)
                .map(|b| [
                    f64::from_le_bytes(b[0..8].try_into().unwrap()),
                    f64::from_le_bytes(b[8..16].try_into().unwrap()),
                ])
                .collect(),
            size,
        )?,
    })
}

fn half_to_f32(h: u16) -> f32 {
    let (sign, exp, frac) = (((h >> 15) & 1) as u32, ((h >> 10) & 0x1f) as i32, (h & 0x3ff) as u32);
    let v = match exp {
        0 => (frac as f32) * 2f32.powi(-24),
        31 => if frac == 0 { f32::INFINITY } else { f32::NAN },
        e => (1.0 + frac as f32 / 1024.0) * 2f32.powi(e - 15),
    };
    if sign == 1 { -v } else { v }
}

fn flatten(prefix: &str, v: &Val, files: &HashMap<String, Vec<u8>>, root: &str, out: &mut Vec<(String, Tensor)>) -> Result<()> {
    let join = |k: &str| if prefix.is_empty() { k.to_string() } else { format!("{prefix}.{k}") };
    match v {
        Val::Tensor { key, dtype, type_name, offset, size, stride } => {
            let dtype = dtype.ok_or_else(|| err(format!("tensor '{prefix}' has the unsupported storage type {type_name}")))?;
            let raw = files.get(&format!("{root}data/{key}")).ok_or_else(|| err(format!("storage file data/{key} is missing from the archive")))?;
            out.push((prefix.to_string(), to_tensor(raw, dtype, *offset, size, stride)?));
        }
        Val::Dict(items) => {
            for (k, x) in items {
                let name = match k {
                    Val::Str(s) => s.clone(),
                    Val::Int(i) => i.to_string(),
                    _ => continue,
                };
                flatten(&join(&name), x, files, root, out)?;
            }
        }
        Val::List(items) | Val::Tuple(items) => {
            for (i, x) in items.iter().enumerate() {
                flatten(&join(&i.to_string()), x, files, root, out)?;
            }
        }
        _ => {}
    }
    Ok(())
}

/// All tensors of a checkpoint with dotted names (`"layer1.weight"`, `"model_state_dict.fc.bias"`, `"0"` for list items),
/// in file order. Non-tensor entries (epoch counters, optimizer hyper-parameters) are skipped.
pub fn load_torch_tensors_from_bytes(bytes: &[u8]) -> Result<Vec<(String, Tensor)>> {
    if bytes.len() < 4 || bytes[..2] != *b"PK" {
        return Err(err("not a zip-based checkpoint (the legacy pre-1.6 format is not supported)"));
    }
    let entries = zip_entries(bytes)?;
    let pkl_name = entries.iter().find(|(n, _)| n.ends_with("data.pkl")).map(|(n, _)| n.clone()).ok_or_else(|| err("the archive has no data.pkl"))?;
    let root = pkl_name.strip_suffix("data.pkl").unwrap_or("").to_string();
    let files: HashMap<String, Vec<u8>> = entries.into_iter().collect();
    let mut vm = Vm { b: &files[&pkl_name], i: 0, stack: Vec::new(), memo: HashMap::new() };
    let top = vm.run()?;
    let mut out = Vec::new();
    flatten("", &top, &files, &root, &mut out)?;
    Ok(out)
}

/// Like [`load_torch_tensors_from_bytes`], reading a file.
pub fn load_torch_tensors(path: impl AsRef<std::path::Path>) -> Result<Vec<(String, Tensor)>> {
    let b = std::fs::read(path.as_ref()).map_err(|e| err(format!("cannot read {}: {e}", path.as_ref().display())))?;
    load_torch_tensors_from_bytes(&b)
}

/// The tensors as a map (a later duplicate name replaces an earlier one).
pub fn load_torch_state_dict(path: impl AsRef<std::path::Path>) -> Result<HashMap<String, Tensor>> {
    Ok(load_torch_tensors(path)?.into_iter().collect())
}

fn torch_storage_name(dtype: DType) -> Result<&'static str> {
    match dtype {
        DType::F32 => Ok("FloatStorage"),
        DType::F64 => Ok("DoubleStorage"),
        DType::F16 => Ok("HalfStorage"),
        DType::BF16 => Ok("BFloat16Storage"),
        DType::I64 => Ok("LongStorage"),
        DType::I32 => Ok("IntStorage"),
        DType::I16 => Ok("ShortStorage"),
        DType::I8 => Ok("CharStorage"),
        DType::U8 => Ok("ByteStorage"),
        DType::Bool => Ok("BoolStorage"),
        _ => Err(err(format!("unsupported dtype {dtype:?} for .pt export"))),
    }
}

fn pkl_unicode(s: &str, out: &mut Vec<u8>) {
    out.push(b'X');
    out.extend_from_slice(&(s.len() as u32).to_le_bytes());
    out.extend_from_slice(s.as_bytes());
}

fn pkl_int(v: usize, out: &mut Vec<u8>) {
    if v <= 255 {
        out.push(b'K');
        out.push(v as u8);
    } else if v <= i32::MAX as usize {
        out.push(b'J');
        out.extend_from_slice(&(v as i32).to_le_bytes());
    } else {
        // LONG1: length byte, then the little-endian two's-complement value (a zero byte keeps it positive)
        let mut b = (v as u64).to_le_bytes().to_vec();
        while b.len() > 1 && b[b.len() - 1] == 0 {
            b.pop();
        }
        if b[b.len() - 1] & 0x80 != 0 {
            b.push(0);
        }
        out.push(0x8a);
        out.push(b.len() as u8);
        out.extend_from_slice(&b);
    }
}

const CRC_TABLE: [u32; 256] = {
    let mut t = [0u32; 256];
    let mut i = 0;
    while i < 256 {
        let mut c = i as u32;
        let mut k = 0;
        while k < 8 {
            c = if c & 1 != 0 { 0xEDB8_8320 ^ (c >> 1) } else { c >> 1 };
            k += 1;
        }
        t[i] = c;
        i += 1;
    }
    t
};

fn zip_crc32(data: &[u8]) -> u32 {
    let mut c = 0xFFFF_FFFFu32;
    for &b in data {
        c = CRC_TABLE[((c ^ b as u32) & 0xFF) as usize] ^ (c >> 8);
    }
    !c
}

fn build_store_zip(entries: &[(&str, &[u8])]) -> Result<Vec<u8>> {
    // 32-bit ZIP fields: larger archives would need ZIP64, so refuse them instead of writing a corrupt file
    let total: usize = entries.iter().map(|(n, d)| d.len() + n.len() * 2 + 100).sum();
    if total >= u32::MAX as usize || entries.len() >= u16::MAX as usize {
        return Err(err("checkpoint is too large for a classic ZIP (4 GiB / 65535 entries); ZIP64 is not supported"));
    }
    let mut out = Vec::new();
    let mut central = Vec::new();
    for &(name, data) in entries {
        let local_off = out.len() as u32;
        let crc = zip_crc32(data);
        let sz = data.len() as u32;
        let nlen = name.len() as u16;
        // Local file header
        out.extend_from_slice(b"PK\x03\x04");
        out.extend_from_slice(&20u16.to_le_bytes());
        out.extend_from_slice(&0u16.to_le_bytes());
        out.extend_from_slice(&0u16.to_le_bytes()); // STORE
        out.extend_from_slice(&0u16.to_le_bytes());
        out.extend_from_slice(&0u16.to_le_bytes());
        out.extend_from_slice(&crc.to_le_bytes());
        out.extend_from_slice(&sz.to_le_bytes());
        out.extend_from_slice(&sz.to_le_bytes());
        out.extend_from_slice(&nlen.to_le_bytes());
        out.extend_from_slice(&0u16.to_le_bytes());
        out.extend_from_slice(name.as_bytes());
        out.extend_from_slice(data);

        // Central directory header
        central.extend_from_slice(b"PK\x01\x02");
        central.extend_from_slice(&20u16.to_le_bytes());
        central.extend_from_slice(&20u16.to_le_bytes());
        central.extend_from_slice(&0u16.to_le_bytes());
        central.extend_from_slice(&0u16.to_le_bytes());
        central.extend_from_slice(&0u16.to_le_bytes());
        central.extend_from_slice(&0u16.to_le_bytes());
        central.extend_from_slice(&crc.to_le_bytes());
        central.extend_from_slice(&sz.to_le_bytes());
        central.extend_from_slice(&sz.to_le_bytes());
        central.extend_from_slice(&nlen.to_le_bytes());
        central.extend_from_slice(&0u16.to_le_bytes());
        central.extend_from_slice(&0u16.to_le_bytes());
        central.extend_from_slice(&0u16.to_le_bytes());
        central.extend_from_slice(&0u16.to_le_bytes());
        central.extend_from_slice(&0u32.to_le_bytes());
        central.extend_from_slice(&local_off.to_le_bytes());
        central.extend_from_slice(name.as_bytes());
    }
    let cd_off = out.len() as u32;
    let cd_sz = central.len() as u32;
    let cnt = entries.len() as u16;
    out.extend_from_slice(&central);
    out.extend_from_slice(b"PK\x05\x06");
    out.extend_from_slice(&0u16.to_le_bytes());
    out.extend_from_slice(&0u16.to_le_bytes());
    out.extend_from_slice(&cnt.to_le_bytes());
    out.extend_from_slice(&cnt.to_le_bytes());
    out.extend_from_slice(&cd_sz.to_le_bytes());
    out.extend_from_slice(&cd_off.to_le_bytes());
    out.extend_from_slice(&0u16.to_le_bytes());
    Ok(out)
}

fn tensor_raw_le_bytes(t: &Tensor) -> Result<Vec<u8>> {
    use crate::backend::{CpuStorage, StorageInner};
    let guard = t.storage().read();
    let cpu = match &*guard {
        StorageInner::Cpu(c) => c,
        _ => return Err(err("expected CPU tensor for .pt export")),
    };
    let mut out = Vec::new();
    match cpu {
        CpuStorage::F32(v) => v.iter().for_each(|x| out.extend_from_slice(&x.to_le_bytes())),
        CpuStorage::F64(v) => v.iter().for_each(|x| out.extend_from_slice(&x.to_le_bytes())),
        CpuStorage::F16(v) => v.iter().for_each(|x| out.extend_from_slice(&x.to_bits().to_le_bytes())),
        CpuStorage::BF16(v) => v.iter().for_each(|x| out.extend_from_slice(&x.to_bits().to_le_bytes())),
        CpuStorage::I64(v) => v.iter().for_each(|x| out.extend_from_slice(&x.to_le_bytes())),
        CpuStorage::I32(v) => v.iter().for_each(|x| out.extend_from_slice(&x.to_le_bytes())),
        CpuStorage::I16(v) => v.iter().for_each(|x| out.extend_from_slice(&x.to_le_bytes())),
        CpuStorage::I8(v) => v.iter().for_each(|&x| out.push(x as u8)),
        CpuStorage::U8(v) => out.extend_from_slice(v),
        CpuStorage::Bool(v) => v.iter().for_each(|&x| out.push(u8::from(x))),
        _ => return Err(err(format!("unsupported dtype {:?} for .pt export", t.dtype()))),
    }
    Ok(out)
}

/// Pickles one tensor as `torch._utils._rebuild_tensor_v2(storage, offset, shape, strides, requires_grad, hooks)`; the data
/// itself goes into the archive entry `data/{idx}`.
fn pkl_tensor(pkl: &mut Vec<u8>, t: &Tensor, idx: usize, requires_grad: bool) -> Result<()> {
    let scls = torch_storage_name(t.dtype())?;
    pkl.extend_from_slice(b"ctorch._utils\n_rebuild_tensor_v2\n(");
    // BINPERSID tuple: ('storage', StorageClass, key, 'cpu', numel)
    pkl.push(b'(');
    pkl_unicode("storage", pkl);
    pkl.extend_from_slice(format!("ctorch\n{scls}\n").as_bytes());
    pkl_unicode(&idx.to_string(), pkl);
    pkl_unicode("cpu", pkl);
    pkl_int(t.numel(), pkl);
    pkl.extend_from_slice(b"tQ");
    pkl_int(0, pkl); // storage_offset
    pkl.push(b'(');
    for &d in t.shape() {
        pkl_int(d, pkl);
    }
    pkl.push(b't');
    pkl.push(b'(');
    for &st in t.layout().strides() {
        pkl_int(st as usize, pkl);
    }
    pkl.push(b't');
    pkl.push(if requires_grad { 0x88 } else { 0x89 }); // NEWTRUE / NEWFALSE
    pkl.extend_from_slice(b"ccollections\nOrderedDict\n)RtR");
    Ok(())
}

/// Serializes a named tensor map into a PyTorch-compatible `.pt` ZIP+Pickle archive.
pub fn serialize_torch_state_dict(tensors: &HashMap<String, Tensor>) -> Result<Vec<u8>> {
    let mut keys: Vec<&String> = tensors.keys().collect();
    keys.sort();

    let mut pkl = Vec::new();
    pkl.extend_from_slice(&[0x80, 0x02]); // PROTO 2
    pkl.extend_from_slice(b"ccollections\nOrderedDict\n)R(");

    let mut blobs: Vec<(String, Vec<u8>)> = Vec::with_capacity(keys.len());
    for (idx, name) in keys.iter().enumerate() {
        let t = tensors[*name].to_device(&crate::backend::Device::Cpu)?.contiguous()?;
        pkl_unicode(name, &mut pkl);
        pkl_tensor(&mut pkl, &t, idx, false)?;

        let raw = tensor_raw_le_bytes(&t)?;
        blobs.push((format!("archive/data/{idx}"), raw));
    }
    pkl.extend_from_slice(b"u.");

    let mut zip_entries: Vec<(&str, &[u8])> = Vec::with_capacity(blobs.len() + 3);
    zip_entries.push(("archive/data.pkl", &pkl));
    zip_entries.push(("archive/version", b"3\n"));
    zip_entries.push(("archive/byteorder", b"little"));
    for (n, b) in &blobs {
        zip_entries.push((n.as_str(), b.as_slice()));
    }
    build_store_zip(&zip_entries)
}

/// Serializes `tensors` and a TorchScript `forward` into an archive that `torch.jit.load` (and the C++ / mobile runtimes)
/// accept: a scripted module `Model` whose floating-point tensors are parameters and every other tensor a buffer.
///
/// `forward_body` is the body of `forward(self, x: Tensor) -> Tensor` in TorchScript syntax, one statement per line
/// (without indentation); tensors are available as locals named after their (sanitized) keys, e.g. a key `fc1.weight`
/// is the local `fc1_weight`. The last statement must `return` the result.
pub fn serialize_torchscript_module(tensors: &HashMap<String, Tensor>, forward_body: &str) -> Result<Vec<u8>> {
    let mut entries: Vec<(String, &String)> = Vec::with_capacity(tensors.len());
    let mut keys: Vec<&String> = tensors.keys().collect();
    keys.sort();
    for k in keys {
        let mut attr: String = k.chars().map(|c| if c.is_ascii_alphanumeric() || c == '_' { c } else { '_' }).collect();
        if attr.is_empty() || attr.starts_with(|c: char| c.is_ascii_digit()) {
            attr.insert(0, '_');
        }
        if ["x", "self", "training", "forward"].contains(&attr.as_str()) || entries.iter().any(|(a, _)| *a == attr) {
            return Err(err(format!("tensor name '{k}' maps to the reserved or duplicate TorchScript attribute '{attr}'")));
        }
        entries.push((attr, k));
    }

    let mut pkl = vec![0x80, 0x02];
    pkl.extend_from_slice(b"c__torch__\nModel\n)\x81}(");
    let mut blobs: Vec<Vec<u8>> = Vec::with_capacity(entries.len());
    let (mut params, mut buffers) = (Vec::new(), Vec::new());
    for (idx, (attr, key)) in entries.iter().enumerate() {
        let t = tensors[*key].to_device(&crate::backend::Device::Cpu)?.contiguous()?;
        let is_param = t.dtype().is_floating_point();
        pkl_unicode(attr, &mut pkl);
        pkl_tensor(&mut pkl, &t, idx, is_param)?;
        blobs.push(tensor_raw_le_bytes(&t)?);
        if is_param { params.push(attr.clone()) } else { buffers.push(attr.clone()) }
    }
    pkl_unicode("training", &mut pkl);
    pkl.push(0x89); // training = False
    pkl_unicode("_is_full_backward_hook", &mut pkl);
    pkl.push(b'N');
    pkl.extend_from_slice(b"ub.");

    let list = |v: &[String]| v.iter().map(|n| format!("\"{n}\", ")).collect::<String>();
    let mut code = format!("class Model(Module):\n  __parameters__ = [{}]\n  __buffers__ = [{}]\n", list(&params), list(&buffers));
    for (attr, _) in &entries {
        code.push_str(&format!("  {attr} : Tensor\n"));
    }
    code.push_str("  training : bool\n  _is_full_backward_hook : Optional[bool]\n  def forward(self: __torch__.Model,\n    x: Tensor) -> Tensor:\n");
    for (attr, _) in &entries {
        code.push_str(&format!("    {attr} = self.{attr}\n"));
    }
    for line in forward_body.lines().filter(|l| !l.trim().is_empty()) {
        code.push_str(&format!("    {}\n", line.trim()));
    }

    let blob_names: Vec<String> = (0..blobs.len()).map(|i| format!("archive/data/{i}")).collect();
    let mut zip_entries: Vec<(&str, &[u8])> = vec![
        ("archive/data.pkl", &pkl),
        ("archive/code/__torch__.py", code.as_bytes()),
        ("archive/constants.pkl", &[0x80, 0x02, b')', b'.']),
        ("archive/version", b"3\n"),
        ("archive/byteorder", b"little"),
    ];
    for (n, b) in blob_names.iter().zip(&blobs) {
        zip_entries.push((n.as_str(), b.as_slice()));
    }
    build_store_zip(&zip_entries)
}

/// Writes a named tensor map as a PyTorch `.pt` checkpoint file.
pub fn save_torch_state_dict(
    tensors: &HashMap<String, Tensor>,
    path: impl AsRef<std::path::Path>,
) -> Result<()> {
    let bytes = serialize_torch_state_dict(tensors)?;
    std::fs::write(path.as_ref(), bytes)
        .map_err(|e| err(format!("cannot write {}: {e}", path.as_ref().display())))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reads_a_real_linear_state_dict() {
        // written by torch.save(nn.Linear(8, 4).state_dict())
        let ts = load_torch_tensors_from_bytes(include_bytes!("testdata/torch_linear.pt")).unwrap();
        let names: Vec<&str> = ts.iter().map(|(n, _)| n.as_str()).collect();
        assert_eq!(names, ["weight", "bias"]);
        assert_eq!(ts[0].1.shape(), [4, 8]);
        assert_eq!(ts[1].1.shape(), [4]);
        let w = ts[0].1.to_vec::<f32>().unwrap();
        assert!(w.iter().all(|x| x.abs() <= 0.36)); // Linear(8, 4) initializes within +-1/sqrt(8)
        assert!(load_torch_tensors_from_bytes(b"not a checkpoint").is_err());
    }

    #[test]
    fn zip_crc32_matches_the_standard_check_value() {
        assert_eq!(zip_crc32(b"123456789"), 0xCBF4_3926);
        assert_eq!(zip_crc32(b""), 0);
    }

    #[test]
    fn pickle_ints_cover_every_range() {
        let enc = |v: usize| {
            let mut o = Vec::new();
            pkl_int(v, &mut o);
            o
        };
        assert_eq!(enc(7), vec![b'K', 7]);
        assert_eq!(enc(70_000), [vec![b'J'], 70_000i32.to_le_bytes().to_vec()].concat());
        // 3_000_000_000 = 0xB2D05E00: top bit set, so a zero byte keeps it positive
        assert_eq!(enc(3_000_000_000), vec![0x8a, 5, 0x00, 0x5E, 0xD0, 0xB2, 0x00]);
        assert_eq!(enc(1 << 40), vec![0x8a, 6, 0, 0, 0, 0, 0, 1]);
    }

    #[test]
    fn writes_and_reads_torch_state_dict() {
        let mut sd = HashMap::new();
        sd.insert("w".to_string(), Tensor::from_slice(&[1.0f32, -2.0, 3.0, 4.0], &[2, 2]).unwrap());
        sd.insert("b".to_string(), Tensor::from_slice(&[0.5f32, -0.5], &[2]).unwrap());
        let bytes = serialize_torch_state_dict(&sd).unwrap();
        let loaded: HashMap<String, Tensor> = load_torch_tensors_from_bytes(&bytes).unwrap().into_iter().collect();
        assert_eq!(loaded["w"].to_vec::<f32>().unwrap(), vec![1.0, -2.0, 3.0, 4.0]);
        assert_eq!(loaded["b"].to_vec::<f32>().unwrap(), vec![0.5, -0.5]);
    }
}
