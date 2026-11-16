//! ONNX model import and execution: parses a `.onnx` protobuf and interprets its graph on this library's tensors. Covers the
//! operators of typical vision, MLP, transformer and recurrent-free inference graphs (see [`OnnxModel`] for the list); an
//! unsupported operator is reported by name when the model is loaded, never silently skipped.

use crate::dtype::DType;
use crate::error::{Error, Result};
use crate::nn::functional_ext::layer_norm;
use crate::tensor::Tensor;
use std::collections::HashMap;

fn err(msg: impl Into<String>) -> Error {
    Error::Msg(format!("onnx: {}", msg.into()))
}

// ------------------------------------------------------------------------------------------------ protobuf

struct Rd<'a> {
    b: &'a [u8],
    i: usize,
}

enum Wire<'a> {
    Varint(u64),
    Bytes(&'a [u8]),
    Fixed32(u32),
    Fixed64,
}

impl<'a> Rd<'a> {
    fn varint(&mut self) -> Result<u64> {
        let (mut v, mut sh) = (0u64, 0);
        loop {
            let c = *self.b.get(self.i).ok_or_else(|| err("truncated protobuf"))?;
            self.i += 1;
            v |= ((c & 0x7f) as u64) << sh;
            if c & 0x80 == 0 {
                return Ok(v);
            }
            sh += 7;
            if sh > 63 {
                return Err(err("bad varint"));
            }
        }
    }
    fn field(&mut self) -> Result<Option<(u32, Wire<'a>)>> {
        if self.i >= self.b.len() {
            return Ok(None);
        }
        let tag = self.varint()?;
        let w = match tag & 7 {
            0 => Wire::Varint(self.varint()?),
            1 => {
                let _skipped = self.b.get(self.i..self.i + 8).ok_or_else(|| err("truncated protobuf"))?;
                self.i += 8;
                Wire::Fixed64
            }
            2 => {
                let n = self.varint()? as usize;
                let s = self.b.get(self.i..self.i + n).ok_or_else(|| err("truncated protobuf"))?;
                self.i += n;
                Wire::Bytes(s)
            }
            5 => {
                let s = self.b.get(self.i..self.i + 4).ok_or_else(|| err("truncated protobuf"))?;
                self.i += 4;
                Wire::Fixed32(u32::from_le_bytes(s.try_into().unwrap()))
            }
            o => return Err(err(format!("unsupported protobuf wire type {o}"))),
        };
        Ok(Some(((tag >> 3) as u32, w)))
    }
}

fn packed_varints(b: &[u8]) -> Result<Vec<i64>> {
    let mut r = Rd { b, i: 0 };
    let mut out = Vec::new();
    while r.i < b.len() {
        out.push(r.varint()? as i64);
    }
    Ok(out)
}

fn text(b: &[u8]) -> String {
    String::from_utf8_lossy(b).to_string()
}

// ------------------------------------------------------------------------------------------------ model

#[derive(Debug, Clone)]
enum Attr {
    F(f32),
    I(i64),
    S(String),
    T(Tensor),
    Fs(Vec<f32>),
    Is(Vec<i64>),
}

#[derive(Debug, Clone)]
struct Node {
    op: String,
    inputs: Vec<String>,
    outputs: Vec<String>,
    attrs: HashMap<String, Attr>,
}

/// A loaded ONNX graph. Supported operators: Identity, Constant, ConstantOfShape, Add, Sub, Mul, Div, Pow, Neg, Abs, Sqrt, Exp,
/// Log, Erf, Tanh, Sigmoid, Relu, LeakyRelu, Gelu, Reciprocal, Floor, Ceil, Softmax, LogSoftmax, MatMul, Gemm, Transpose, Reshape,
/// Flatten, Concat, Squeeze, Unsqueeze, Shape, Gather, Slice, Cast, ReduceMean, ReduceSum, ReduceMax, Clip, Where, Expand,
/// Conv (2-D, grouped), MaxPool, AveragePool, GlobalAveragePool, BatchNormalization, LayerNormalization, Dropout.
pub struct OnnxModel {
    nodes: Vec<Node>,
    initializers: HashMap<String, Tensor>,
    inputs: Vec<String>,
    outputs: Vec<String>,
    opset: i64,
}

const SUPPORTED: &[&str] = &[
    "Identity", "Constant", "ConstantOfShape", "Add", "Sub", "Mul", "Div", "Pow", "Neg", "Abs", "Sqrt", "Exp", "Log", "Erf", "Tanh", "Sigmoid", "Relu", "LeakyRelu", "Gelu", "Reciprocal", "Floor", "Ceil",
    "Softmax", "LogSoftmax", "MatMul", "Gemm", "Transpose", "Reshape", "Flatten", "Concat", "Squeeze", "Unsqueeze", "Shape", "Gather", "Slice", "Cast", "ReduceMean", "ReduceSum", "ReduceMax", "Clip", "Where",
    "Expand", "Conv", "MaxPool", "AveragePool", "GlobalAveragePool", "BatchNormalization", "LayerNormalization", "Dropout",
];

fn parse_tensor(b: &[u8]) -> Result<(String, Tensor)> {
    let mut r = Rd { b, i: 0 };
    let (mut dims, mut dt, mut name, mut raw) = (Vec::<i64>::new(), 1i64, String::new(), None::<&[u8]>);
    let (mut floats, mut ints32, mut ints64, mut doubles) = (Vec::<f32>::new(), Vec::<i64>::new(), Vec::<i64>::new(), Vec::<f64>::new());
    while let Some((f, w)) = r.field()? {
        match (f, w) {
            (1, Wire::Varint(v)) => dims.push(v as i64),
            (1, Wire::Bytes(p)) => dims.extend(packed_varints(p)?),
            (2, Wire::Varint(v)) => dt = v as i64,
            (4, Wire::Bytes(p)) => floats.extend(p.chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap()))),
            (4, Wire::Fixed32(v)) => floats.push(f32::from_bits(v)),
            (5, Wire::Bytes(p)) => ints32.extend(packed_varints(p)?),
            (5, Wire::Varint(v)) => ints32.push(v as i64),
            (7, Wire::Bytes(p)) => ints64.extend(packed_varints(p)?),
            (7, Wire::Varint(v)) => ints64.push(v as i64),
            (8, Wire::Bytes(p)) => name = text(p),
            (9, Wire::Bytes(p)) => raw = Some(p),
            (10, Wire::Bytes(p)) => doubles.extend(p.chunks_exact(8).map(|c| f64::from_le_bytes(c.try_into().unwrap()))),
            _ => {}
        }
    }
    let shape: Vec<usize> = dims.iter().map(|&d| d as usize).collect();
    let n: usize = shape.iter().product();
    let t = match dt {
        1 => {
            let v: Vec<f32> = match raw {
                Some(p) => p.chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect(),
                None => floats,
            };
            Tensor::from_vec(v, &shape)?
        }
        11 => {
            let v: Vec<f64> = match raw {
                Some(p) => p.chunks_exact(8).map(|c| f64::from_le_bytes(c.try_into().unwrap())).collect(),
                None => doubles,
            };
            Tensor::from_vec(v, &shape)?
        }
        7 => {
            let v: Vec<i64> = match raw {
                Some(p) => p.chunks_exact(8).map(|c| i64::from_le_bytes(c.try_into().unwrap())).collect(),
                None => ints64,
            };
            Tensor::from_vec(v, &shape)?
        }
        6 => {
            let v: Vec<i32> = match raw {
                Some(p) => p.chunks_exact(4).map(|c| i32::from_le_bytes(c.try_into().unwrap())).collect(),
                None => ints32.iter().map(|&x| x as i32).collect(),
            };
            Tensor::from_vec(v, &shape)?
        }
        9 => {
            let v: Vec<bool> = match raw {
                Some(p) => p.iter().map(|&b| b != 0).collect(),
                None => ints32.iter().map(|&x| x != 0).collect(),
            };
            Tensor::from_vec(v, &shape)?
        }
        2 => {
            let v: Vec<u8> = match raw {
                Some(p) => p.to_vec(),
                None => ints32.iter().map(|&x| x as u8).collect(),
            };
            Tensor::from_vec(v, &shape)?
        }
        other => return Err(err(format!("unsupported tensor element type {other}"))),
    };
    if t.numel() != n {
        return Err(err(format!("tensor '{name}' holds {} values for shape {shape:?}", t.numel())));
    }
    Ok((name, t))
}

fn parse_attr(b: &[u8]) -> Result<(String, Attr)> {
    let mut r = Rd { b, i: 0 };
    let (mut name, mut f, mut i, mut s, mut t) = (String::new(), None, None, None, None);
    let (mut fs, mut is) = (Vec::new(), Vec::new());
    while let Some((fld, w)) = r.field()? {
        match (fld, w) {
            (1, Wire::Bytes(p)) => name = text(p),
            (2, Wire::Fixed32(v)) => f = Some(f32::from_bits(v)),
            (3, Wire::Varint(v)) => i = Some(v as i64),
            (4, Wire::Bytes(p)) => s = Some(text(p)),
            (5, Wire::Bytes(p)) => t = Some(parse_tensor(p)?.1),
            (7, Wire::Fixed32(v)) => fs.push(f32::from_bits(v)),
            (7, Wire::Bytes(p)) => fs.extend(p.chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap()))),
            (8, Wire::Varint(v)) => is.push(v as i64),
            (8, Wire::Bytes(p)) => is.extend(packed_varints(p)?),
            _ => {}
        }
    }
    let a = if let Some(t) = t {
        Attr::T(t)
    } else if let Some(i) = i {
        Attr::I(i)
    } else if let Some(f) = f {
        Attr::F(f)
    } else if let Some(s) = s {
        Attr::S(s)
    } else if !is.is_empty() {
        Attr::Is(is)
    } else {
        Attr::Fs(fs)
    };
    Ok((name, a))
}

fn parse_node(b: &[u8]) -> Result<Node> {
    let mut r = Rd { b, i: 0 };
    let mut n = Node { op: String::new(), inputs: vec![], outputs: vec![], attrs: HashMap::new() };
    while let Some((f, w)) = r.field()? {
        match (f, w) {
            (1, Wire::Bytes(p)) => n.inputs.push(text(p)),
            (2, Wire::Bytes(p)) => n.outputs.push(text(p)),
            (4, Wire::Bytes(p)) => n.op = text(p),
            (5, Wire::Bytes(p)) => {
                let (k, v) = parse_attr(p)?;
                n.attrs.insert(k, v);
            }
            _ => {}
        }
    }
    Ok(n)
}

fn value_info_name(b: &[u8]) -> String {
    let mut r = Rd { b, i: 0 };
    while let Ok(Some((f, w))) = r.field() {
        if let (1, Wire::Bytes(p)) = (f, w) {
            return text(p);
        }
    }
    String::new()
}

impl OnnxModel {
    pub fn from_file(path: impl AsRef<std::path::Path>) -> Result<Self> {
        Self::from_bytes(&std::fs::read(path.as_ref()).map_err(|e| err(format!("cannot read {}: {e}", path.as_ref().display())))?)
    }

    pub fn from_bytes(bytes: &[u8]) -> Result<Self> {
        let mut r = Rd { b: bytes, i: 0 };
        let (mut graph, mut opset) = (None, 13i64);
        while let Some((f, w)) = r.field()? {
            match (f, w) {
                (7, Wire::Bytes(g)) => graph = Some(g),
                (8, Wire::Bytes(o)) => {
                    let mut or = Rd { b: o, i: 0 };
                    let (mut domain, mut ver) = (String::new(), 0);
                    while let Some((f2, w2)) = or.field()? {
                        match (f2, w2) {
                            (1, Wire::Bytes(d)) => domain = text(d),
                            (2, Wire::Varint(v)) => ver = v as i64,
                            _ => {}
                        }
                    }
                    if domain.is_empty() || domain == "ai.onnx" {
                        opset = ver;
                    }
                }
                _ => {}
            }
        }
        let graph = graph.ok_or_else(|| err("the file has no graph"))?;
        let mut m = OnnxModel { nodes: vec![], initializers: HashMap::new(), inputs: vec![], outputs: vec![], opset };
        let mut declared_inputs = Vec::new();
        let mut gr = Rd { b: graph, i: 0 };
        while let Some((f, w)) = gr.field()? {
            match (f, w) {
                (1, Wire::Bytes(n)) => m.nodes.push(parse_node(n)?),
                (5, Wire::Bytes(t)) => {
                    let (name, t) = parse_tensor(t)?;
                    m.initializers.insert(name, t);
                }
                (11, Wire::Bytes(v)) => declared_inputs.push(value_info_name(v)),
                (12, Wire::Bytes(v)) => m.outputs.push(value_info_name(v)),
                _ => {}
            }
        }
        m.inputs = declared_inputs.into_iter().filter(|n| !m.initializers.contains_key(n)).collect();
        for n in &m.nodes {
            if !SUPPORTED.contains(&n.op.as_str()) {
                return Err(err(format!("operator {} is not supported", n.op)));
            }
        }
        Ok(m)
    }

    /// The default-domain opset the model declares.
    pub fn opset(&self) -> i64 {
        self.opset
    }
    pub fn input_names(&self) -> &[String] {
        &self.inputs
    }
    pub fn output_names(&self) -> &[String] {
        &self.outputs
    }

    /// Runs the graph on named inputs and returns the graph outputs in order.
    pub fn run(&self, feeds: &[(&str, Tensor)]) -> Result<Vec<Tensor>> {
        let mut env: HashMap<String, Tensor> = self.initializers.clone();
        for (k, v) in feeds {
            env.insert((*k).to_string(), v.clone());
        }
        for name in &self.inputs {
            if !env.contains_key(name) {
                return Err(err(format!("missing input '{name}'")));
            }
        }
        for node in &self.nodes {
            let outs = self.exec(node, &env).map_err(|e| err(format!("{} node producing '{}': {e}", node.op, node.outputs.first().map_or("", |s| s))))?;
            for (name, t) in node.outputs.iter().zip(outs) {
                if !name.is_empty() {
                    env.insert(name.clone(), t);
                }
            }
        }
        self.outputs.iter().map(|o| env.get(o).cloned().ok_or_else(|| err(format!("output '{o}' was not produced")))).collect()
    }

    fn exec(&self, n: &Node, env: &HashMap<String, Tensor>) -> Result<Vec<Tensor>> {
        let inp = |i: usize| -> Result<Tensor> {
            let name = n.inputs.get(i).ok_or_else(|| err(format!("missing input {i}")))?;
            env.get(name).cloned().ok_or_else(|| err(format!("tensor '{name}' is not available")))
        };
        let opt = |i: usize| -> Option<Tensor> { n.inputs.get(i).filter(|s| !s.is_empty()).and_then(|s| env.get(s).cloned()) };
        let int = |k: &str, d: i64| if let Some(Attr::I(v)) = n.attrs.get(k) { *v } else { d };
        let float = |k: &str, d: f32| if let Some(Attr::F(v)) = n.attrs.get(k) { *v } else { d };
        let ints = |k: &str| if let Some(Attr::Is(v)) = n.attrs.get(k) { Some(v.clone()) } else { None };
        let ivec = |t: &Tensor| -> Result<Vec<i64>> { t.to_dtype(DType::I64)?.to_vec::<i64>() };
        let one = |t: Tensor| Ok(vec![t]);
        match n.op.as_str() {
            "Identity" | "Dropout" => one(inp(0)?),
            "Constant" => match n.attrs.get("value") {
                Some(Attr::T(t)) => one(t.clone()),
                _ => match (n.attrs.get("value_float"), n.attrs.get("value_int"), n.attrs.get("value_ints"), n.attrs.get("value_floats")) {
                    (Some(Attr::F(f)), ..) => one(Tensor::from_slice(&[*f], &[1])?.reshape(&[])?),
                    (_, Some(Attr::I(i)), ..) => one(Tensor::from_slice(&[*i], &[1])?.reshape(&[])?),
                    (_, _, Some(Attr::Is(v)), _) => one(Tensor::from_slice(v, &[v.len()])?),
                    (_, _, _, Some(Attr::Fs(v))) => one(Tensor::from_slice(v, &[v.len()])?),
                    _ => Err(err("Constant without a value")),
                },
            },
            "ConstantOfShape" => {
                let shape: Vec<usize> = ivec(&inp(0)?)?.iter().map(|&d| d as usize).collect();
                match n.attrs.get("value") {
                    Some(Attr::T(t)) => {
                        let v = t.to_dtype(DType::F64)?.to_vec::<f64>()?[0];
                        one(Tensor::full(&shape, v, t.dtype(), &crate::backend::Device::Cpu)?)
                    }
                    _ => one(Tensor::zeros(&shape, DType::F32, &crate::backend::Device::Cpu)?),
                }
            }
            "Add" => one(inp(0)?.add(&inp(1)?)?),
            "Sub" => one(inp(0)?.sub(&inp(1)?)?),
            "Mul" => one(inp(0)?.mul(&inp(1)?)?),
            "Div" => one(inp(0)?.div(&inp(1)?)?),
            "Pow" => {
                let (x, y) = (inp(0)?, inp(1)?);
                one(x.pow(&y.to_dtype(x.dtype())?)?)
            }
            "Neg" => one(inp(0)?.neg()?),
            "Abs" => one(inp(0)?.abs()?),
            "Sqrt" => one(inp(0)?.sqrt()?),
            "Exp" => one(inp(0)?.exp()?),
            "Log" => one(inp(0)?.ln()?),
            "Erf" => one(inp(0)?.erf()?),
            "Tanh" => one(inp(0)?.tanh()?),
            "Sigmoid" => one(inp(0)?.sigmoid()?),
            "Relu" => one(inp(0)?.relu()?),
            "Gelu" => one(inp(0)?.gelu()?),
            "Reciprocal" => one(inp(0)?.recip()?),
            "Floor" => one(inp(0)?.floor()?),
            "Ceil" => one(inp(0)?.ceil()?),
            "LeakyRelu" => {
                let x = inp(0)?;
                let a = float("alpha", 0.01) as f64;
                one(x.relu()?.add(&x.neg()?.relu()?.mul_scalar(-a)?)?)
            }
            "Softmax" => one(inp(0)?.softmax(int("axis", -1) as isize)?),
            "LogSoftmax" => one(inp(0)?.log_softmax(int("axis", -1) as isize)?),
            "MatMul" => one(inp(0)?.matmul(&inp(1)?)?),
            "Gemm" => {
                let (mut a, mut b) = (inp(0)?, inp(1)?);
                if int("transA", 0) == 1 {
                    a = a.t()?;
                }
                if int("transB", 0) == 1 {
                    b = b.t()?;
                }
                let mut y = a.matmul(&b)?;
                let alpha = float("alpha", 1.0) as f64;
                if alpha != 1.0 {
                    y = y.mul_scalar(alpha)?;
                }
                if let Some(c) = opt(2) {
                    let beta = float("beta", 1.0) as f64;
                    y = y.add(&if beta != 1.0 { c.mul_scalar(beta)? } else { c })?;
                }
                one(y)
            }
            "Transpose" => {
                let x = inp(0)?;
                let perm: Vec<usize> = match ints("perm") {
                    Some(p) => p.iter().map(|&v| v as usize).collect(),
                    None => (0..x.ndim()).rev().collect(),
                };
                one(x.permute(&perm)?)
            }
            "Reshape" => {
                let x = inp(0)?;
                let spec = ivec(&inp(1)?)?;
                let allow_zero = int("allowzero", 0) == 1;
                let mut shape: Vec<i64> = spec.iter().enumerate().map(|(i, &d)| if d == 0 && !allow_zero { x.shape().get(i).copied().unwrap_or(1) as i64 } else { d }).collect();
                if let Some(pos) = shape.iter().position(|&d| d == -1) {
                    let known: i64 = shape.iter().filter(|&&d| d != -1).product();
                    shape[pos] = if known == 0 { 0 } else { x.numel() as i64 / known };
                }
                let shape: Vec<usize> = shape.iter().map(|&d| d as usize).collect();
                one(x.contiguous()?.reshape(&shape)?)
            }
            "Flatten" => {
                let x = inp(0)?;
                let axis = int("axis", 1);
                let axis = if axis < 0 { axis + x.ndim() as i64 } else { axis } as usize;
                let (a, b): (usize, usize) = (x.shape()[..axis].iter().product(), x.shape()[axis..].iter().product());
                one(x.contiguous()?.reshape(&[a, b])?)
            }
            "Concat" => {
                let ts: Vec<Tensor> = (0..n.inputs.len()).map(inp).collect::<Result<_>>()?;
                one(Tensor::cat(&ts, int("axis", 0) as isize)?)
            }
            "Squeeze" | "Unsqueeze" => {
                let x = inp(0)?;
                let axes: Vec<i64> = match opt(1) {
                    Some(a) => ivec(&a)?,
                    None => ints("axes").unwrap_or_default(),
                };
                if n.op == "Unsqueeze" {
                    let out_rank = x.ndim() + axes.len();
                    let mut norm: Vec<usize> = axes.iter().map(|&a| if a < 0 { (a + out_rank as i64) as usize } else { a as usize }).collect();
                    norm.sort_unstable();
                    let mut y = x;
                    for a in norm {
                        y = y.unsqueeze(a as isize)?;
                    }
                    one(y)
                } else if axes.is_empty() {
                    one(x.squeeze()?)
                } else {
                    let r = x.ndim() as i64;
                    let mut norm: Vec<i64> = axes.iter().map(|&a| if a < 0 { a + r } else { a }).collect();
                    norm.sort_unstable_by(|a, b| b.cmp(a));
                    let mut y = x;
                    for a in norm {
                        y = y.squeeze_axis(a as isize)?;
                    }
                    one(y)
                }
            }
            "Shape" => {
                let x = inp(0)?;
                let v: Vec<i64> = x.shape().iter().map(|&d| d as i64).collect();
                one(Tensor::from_slice(&v, &[v.len()])?)
            }
            "Gather" => {
                let (x, idx) = (inp(0)?, inp(1)?);
                let axis = int("axis", 0) as isize;
                let flat = idx.contiguous()?.reshape(&[idx.numel()])?.to_dtype(DType::I64)?;
                let sel = x.index_select(axis, &flat)?;
                let ax = if axis < 0 { axis + x.ndim() as isize } else { axis } as usize;
                let mut shape: Vec<usize> = x.shape()[..ax].to_vec();
                shape.extend_from_slice(idx.shape());
                shape.extend_from_slice(&x.shape()[ax + 1..]);
                one(sel.contiguous()?.reshape(&shape)?)
            }
            "Slice" => {
                let x = inp(0)?;
                let starts = ivec(&inp(1)?)?;
                let ends = ivec(&inp(2)?)?;
                let axes: Vec<i64> = match opt(3) {
                    Some(a) => ivec(&a)?,
                    None => (0..starts.len() as i64).collect(),
                };
                let steps: Vec<i64> = match opt(4) {
                    Some(s) => ivec(&s)?,
                    None => vec![1; starts.len()],
                };
                let mut y = x;
                for (k, &ax) in axes.iter().enumerate() {
                    let clamp = |v: i64| if v > i32::MAX as i64 { None } else if v < i32::MIN as i64 { Some(isize::MIN / 2) } else { Some(v as isize) };
                    y = y.slice(ax as isize, clamp(starts[k]), clamp(ends[k]), steps[k] as isize)?;
                }
                one(y)
            }
            "Cast" => {
                let to = match int("to", 1) {
                    1 => DType::F32,
                    11 => DType::F64,
                    7 => DType::I64,
                    6 => DType::I32,
                    9 => DType::Bool,
                    2 => DType::U8,
                    other => return Err(err(format!("Cast to type {other} is not supported"))),
                };
                one(inp(0)?.to_dtype(to)?)
            }
            "ReduceMean" | "ReduceSum" | "ReduceMax" => {
                let x = inp(0)?;
                let axes: Vec<isize> = match opt(1) {
                    Some(a) => ivec(&a)?.iter().map(|&v| v as isize).collect(),
                    None => ints("axes").map(|v| v.iter().map(|&a| a as isize).collect()).unwrap_or_else(|| (0..x.ndim() as isize).collect()),
                };
                let keep = int("keepdims", 1) == 1;
                one(match n.op.as_str() {
                    "ReduceMean" => x.mean_axes(&axes, keep)?,
                    "ReduceSum" => x.sum_axes(&axes, keep)?,
                    _ => {
                        let mut y = x;
                        let mut ax: Vec<isize> = axes.iter().map(|&a| if a < 0 { a + y.ndim() as isize } else { a }).collect();
                        ax.sort_unstable_by(|a, b| b.cmp(a));
                        for a in ax {
                            y = y.max_axis(a, keep)?;
                        }
                        y
                    }
                })
            }
            "Clip" => {
                let x = inp(0)?;
                let lo = opt(1).map(|t| t.to_dtype(DType::F64).and_then(|t| t.to_vec::<f64>()).map(|v| v[0])).transpose()?.or(match n.attrs.get("min") {
                    Some(Attr::F(f)) => Some(*f as f64),
                    _ => None,
                });
                let hi = opt(2).map(|t| t.to_dtype(DType::F64).and_then(|t| t.to_vec::<f64>()).map(|v| v[0])).transpose()?.or(match n.attrs.get("max") {
                    Some(Attr::F(f)) => Some(*f as f64),
                    _ => None,
                });
                one(x.clamp(lo.unwrap_or(f64::NEG_INFINITY), hi.unwrap_or(f64::INFINITY))?)
            }
            "Where" => one(Tensor::where_cond(&inp(0)?, &inp(1)?, &inp(2)?)?),
            "Expand" => {
                let x = inp(0)?;
                let shape: Vec<usize> = ivec(&inp(1)?)?.iter().map(|&d| d as usize).collect();
                let target = crate::layout::broadcast_shapes("Expand", x.shape(), &shape)?;
                one(x.broadcast_to(&target)?)
            }
            "Conv" => {
                let (x, w, b) = (inp(0)?, inp(1)?, opt(2));
                if x.ndim() != 4 {
                    return Err(err("only 2-D convolutions are supported"));
                }
                let strides = ints("strides").unwrap_or_else(|| vec![1, 1]);
                let dil = ints("dilations").unwrap_or_else(|| vec![1, 1]);
                let pads = ints("pads").unwrap_or_else(|| vec![0, 0, 0, 0]);
                if pads[0] != pads[2] || pads[1] != pads[3] {
                    return Err(err("asymmetric convolution padding is not supported"));
                }
                if matches!(n.attrs.get("auto_pad"), Some(Attr::S(s)) if s != "NOTSET") {
                    return Err(err("auto_pad is not supported"));
                }
                one(x.conv2d_grouped(&w, b.as_ref(), (strides[0] as usize, strides[1] as usize), (pads[0] as usize, pads[1] as usize), (dil[0] as usize, dil[1] as usize), int("group", 1) as usize)?)
            }
            "MaxPool" | "AveragePool" => {
                let x = inp(0)?;
                let k = ints("kernel_shape").ok_or_else(|| err("pooling needs kernel_shape"))?;
                let s = ints("strides").unwrap_or_else(|| vec![1; k.len()]);
                let p = ints("pads").unwrap_or_else(|| vec![0, 0, 0, 0]);
                if k.len() != 2 || p[0] != p[2] || p[1] != p[3] || int("ceil_mode", 0) == 1 {
                    return Err(err("only symmetric 2-D pooling without ceil_mode is supported"));
                }
                let (kk, ss, pp) = ((k[0] as usize, k[1] as usize), (s[0] as usize, s[1] as usize), (p[0] as usize, p[1] as usize));
                one(if n.op == "MaxPool" { x.max_pool2d(kk, ss, pp)? } else { x.avg_pool2d(kk, ss, pp)? })
            }
            "GlobalAveragePool" => {
                let x = inp(0)?;
                let axes: Vec<isize> = (2..x.ndim() as isize).collect();
                one(x.mean_axes(&axes, true)?)
            }
            "BatchNormalization" => {
                let (x, scale, bias, mean, var) = (inp(0)?, inp(1)?, inp(2)?, inp(3)?, inp(4)?);
                let eps = float("epsilon", 1e-5) as f64;
                let mut shape = vec![1usize; x.ndim()];
                shape[1] = x.shape()[1];
                let r = |t: &Tensor| t.reshape(&shape);
                let inv = r(&var)?.add_scalar(eps)?.rsqrt()?.mul(&r(&scale)?)?;
                one(x.sub(&r(&mean)?)?.mul(&inv)?.add(&r(&bias)?)?)
            }
            "LayerNormalization" => {
                let (x, scale) = (inp(0)?, inp(1)?);
                let axis = int("axis", -1);
                let axis = if axis < 0 { axis + x.ndim() as i64 } else { axis } as usize;
                let shape: Vec<usize> = x.shape()[axis..].to_vec();
                let (b, s) = (opt(2), scale.reshape(&shape)?);
                one(layer_norm(&x, &shape, Some(&s), b.as_ref().map(|b| b.reshape(&shape)).transpose()?.as_ref(), float("epsilon", 1e-5) as f64)?)
            }
            other => Err(err(format!("operator {other} is not supported"))),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::nn::{Conv2d, Linear, Module};
    use crate::serialize::onnx::{Dim, OnnxBuilder};

    #[test]
    fn exported_models_import_and_match_the_eager_forward() {
        crate::rng::manual_seed(4);
        // MLP
        let (l1, l2) = (Linear::new(6, 10).unwrap(), Linear::new(10, 3).unwrap());
        let mut b = OnnxBuilder::new("x", vec![Dim::Named("batch".into()), Dim::Fixed(6)]);
        b.linear(&l1).unwrap().relu().linear(&l2).unwrap().softmax(-1);
        let bytes = b.finish("y", false, vec![Dim::Named("batch".into()), Dim::Fixed(3)]).unwrap();
        let m = OnnxModel::from_bytes(&bytes).unwrap();
        assert_eq!(m.input_names(), ["x"]);
        let x = Tensor::randn(&[5, 6]).unwrap();
        let got = m.run(&[("x", x.clone())]).unwrap().remove(0);
        let want = l2.forward(&l1.forward(&x).unwrap().relu().unwrap()).unwrap().softmax(-1).unwrap();
        assert!(got.allclose(&want, 1e-5, 1e-5).unwrap());
        // convolution + pooling + flatten + linear
        let conv = Conv2d::new(2, 4, (3, 3), (1, 1), (1, 1)).unwrap();
        let fc = Linear::new(4 * 3 * 3, 2).unwrap();
        let mut b = OnnxBuilder::new("img", vec![Dim::Fixed(2), Dim::Fixed(2), Dim::Fixed(6), Dim::Fixed(6)]);
        b.conv2d(&conv).unwrap().relu().max_pool2d(&crate::nn::MaxPool2d::new((2, 2), (2, 2))).flatten().linear(&fc).unwrap();
        let m = OnnxModel::from_bytes(&b.finish("out", false, vec![Dim::Fixed(2), Dim::Fixed(2)]).unwrap()).unwrap();
        let x = Tensor::randn(&[2, 2, 6, 6]).unwrap();
        let got = m.run(&[("img", x.clone())]).unwrap().remove(0);
        let want = fc.forward(&conv.forward(&x).unwrap().relu().unwrap().max_pool2d((2, 2), (2, 2), (0, 0)).unwrap().contiguous().unwrap().reshape(&[2, 36]).unwrap()).unwrap();
        assert!(got.allclose(&want, 1e-4, 1e-4).unwrap());
        // an operator that is not implemented is named at load time
        assert!(OnnxModel::from_bytes(b"not a model").is_err());
    }
}
