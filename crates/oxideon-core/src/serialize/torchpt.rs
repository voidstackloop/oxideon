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
}
