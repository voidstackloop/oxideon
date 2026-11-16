//! ONNX export of feed-forward networks, with no dependency on a protobuf library.
//!
//! [`OnnxBuilder`] records a chain of layers (the way a `Sequential` would run them) and
//! [`OnnxBuilder::finish`] encodes an ONNX `ModelProto` (IR version 8, opset 17) with the weights as initializers.
//! Supported layers: `Linear`, `Conv2d`, `MaxPool2d`, `AvgPool2d`, `BatchNorm2d` (inference statistics), `LayerNorm`,
//! `Embedding`, `Flatten`, and the activations ReLU, LeakyReLU, Sigmoid, Tanh, GELU (exact), SiLU, Softplus, Softmax and
//! LogSoftmax. Dropout is the identity at inference and is simply left out. `Linear` is exported as `MatMul + Add`
//! so that inputs of any rank work. All tensors are exported as `float32` (indices as `int64`).

use crate::error::{Error, Result};
use crate::nn::attention_ext::MultiheadAttention;
use crate::nn::{AvgPool2d, BatchNorm2d, BiLSTM, Conv2d, ConvTranspose2d, Embedding, GRU, GroupNorm, LSTM, LayerNorm, Linear, MaxPool2d, PReLU};
use crate::tensor::Tensor;

// ---------------------------------------------------------------------------------------------------------------
// protobuf wire format
// ---------------------------------------------------------------------------------------------------------------

fn varint(out: &mut Vec<u8>, mut v: u64) {
    while v >= 0x80 {
        out.push((v as u8 & 0x7f) | 0x80);
        v >>= 7;
    }
    out.push(v as u8);
}

fn key(out: &mut Vec<u8>, field: u32, wire: u8) {
    varint(out, u64::from(field) << 3 | u64::from(wire));
}

fn put_varint(out: &mut Vec<u8>, field: u32, v: i64) {
    key(out, field, 0);
    varint(out, v as u64);
}

fn put_bytes(out: &mut Vec<u8>, field: u32, b: &[u8]) {
    key(out, field, 2);
    varint(out, b.len() as u64);
    out.extend_from_slice(b);
}

fn put_str(out: &mut Vec<u8>, field: u32, s: &str) {
    put_bytes(out, field, s.as_bytes());
}

fn put_f32(out: &mut Vec<u8>, field: u32, v: f32) {
    key(out, field, 5);
    out.extend_from_slice(&v.to_le_bytes());
}

// ONNX TensorProto.DataType
const FLOAT: i64 = 1;
const INT64: i64 = 7;

fn tensor_proto(name: &str, dims: &[usize], dtype: i64, raw: &[u8]) -> Vec<u8> {
    let mut t = Vec::new();
    for &d in dims {
        put_varint(&mut t, 1, d as i64);
    }
    put_varint(&mut t, 2, dtype);
    put_str(&mut t, 8, name);
    put_bytes(&mut t, 9, raw);
    t
}

enum Attr {
    Str(&'static str, &'static str),
    Int(&'static str, i64),
    Float(&'static str, f32),
    Ints(&'static str, Vec<i64>),
}

struct Node {
    op: &'static str,
    inputs: Vec<String>,
    output: String,
    attrs: Vec<Attr>,
}

impl Node {
    fn encode(&self, name: &str) -> Vec<u8> {
        let mut n = Vec::new();
        self.inputs.iter().for_each(|i| put_str(&mut n, 1, i));
        put_str(&mut n, 2, &self.output);
        put_str(&mut n, 3, name);
        put_str(&mut n, 4, self.op);
        for a in &self.attrs {
            let mut at = Vec::new();
            match a {
                Attr::Str(nm, v) => {
                    put_str(&mut at, 1, nm);
                    put_bytes(&mut at, 4, v.as_bytes());
                    put_varint(&mut at, 20, 3);
                }
                Attr::Int(nm, v) => {
                    put_str(&mut at, 1, nm);
                    put_varint(&mut at, 3, *v);
                    put_varint(&mut at, 20, 2);
                }
                Attr::Float(nm, v) => {
                    put_str(&mut at, 1, nm);
                    put_f32(&mut at, 2, *v);
                    put_varint(&mut at, 20, 1);
                }
                Attr::Ints(nm, vs) => {
                    put_str(&mut at, 1, nm);
                    vs.iter().for_each(|v| put_varint(&mut at, 8, *v));
                    put_varint(&mut at, 20, 7);
                }
            }
            put_bytes(&mut n, 5, &at);
        }
        n
    }
}

/// A dimension of the exported input: fixed or symbolic (for example the batch size).
#[derive(Clone, Debug)]
pub enum Dim {
    Fixed(usize),
    Named(String),
}

impl From<usize> for Dim {
    fn from(v: usize) -> Self {
        Dim::Fixed(v)
    }
}

impl From<&str> for Dim {
    fn from(v: &str) -> Self {
        Dim::Named(v.to_string())
    }
}

fn value_info(name: &str, dtype: i64, shape: &[Dim]) -> Vec<u8> {
    let mut dims = Vec::new();
    for d in shape {
        let mut dim = Vec::new();
        match d {
            Dim::Fixed(n) => put_varint(&mut dim, 1, *n as i64),
            Dim::Named(s) => put_str(&mut dim, 2, s),
        }
        put_bytes(&mut dims, 1, &dim);
    }
    let mut tensor_type = Vec::new();
    put_varint(&mut tensor_type, 1, dtype);
    put_bytes(&mut tensor_type, 2, &dims);
    let mut type_proto = Vec::new();
    put_bytes(&mut type_proto, 1, &tensor_type);
    let mut vi = Vec::new();
    put_str(&mut vi, 1, name);
    put_bytes(&mut vi, 2, &type_proto);
    vi
}

// ---------------------------------------------------------------------------------------------------------------
// builder
// ---------------------------------------------------------------------------------------------------------------

/// Records layers and encodes them as an ONNX model.
pub struct OnnxBuilder {
    input_name: String,
    input_shape: Vec<Dim>,
    nodes: Vec<Node>,
    inits: Vec<Vec<u8>>,
    cur: String,
    n: usize,
    reverse_time: bool,
    extra_inputs: Vec<(String, i64, Vec<Dim>)>,
}

fn f32_bytes(t: &Tensor) -> Result<Vec<u8>> {
    Ok(t.to_vec::<f32>()?.iter().flat_map(|v| v.to_le_bytes()).collect())
}

impl OnnxBuilder {
    /// Starts a graph whose input is called `input` and has `shape`.
    pub fn new(input: &str, shape: Vec<Dim>) -> Self {
        Self { input_name: input.to_string(), input_shape: shape, nodes: Vec::new(), inits: Vec::new(), cur: input.to_string(), n: 0, reverse_time: false, extra_inputs: Vec::new() }
    }

    fn fresh(&mut self, base: &str) -> String {
        self.n += 1;
        format!("{base}_{}", self.n)
    }

    fn init(&mut self, base: &str, t: &Tensor) -> Result<String> {
        let name = self.fresh(base);
        self.inits.push(tensor_proto(&name, t.shape(), FLOAT, &f32_bytes(t)?));
        Ok(name)
    }

    fn init_scalar(&mut self, base: &str, v: f32) -> String {
        let name = self.fresh(base);
        self.inits.push(tensor_proto(&name, &[], FLOAT, &v.to_le_bytes()));
        name
    }

    fn push(&mut self, op: &'static str, extra_inputs: Vec<String>, attrs: Vec<Attr>) -> &mut Self {
        let output = self.fresh(&op.to_lowercase());
        let mut inputs = vec![self.cur.clone()];
        inputs.extend(extra_inputs);
        self.nodes.push(Node { op, inputs, output: output.clone(), attrs });
        self.cur = output;
        self
    }

    /// `y = x W^T + b` (`MatMul` with the transposed weight, then `Add`).
    pub fn linear(&mut self, l: &Linear) -> Result<&mut Self> {
        let wt = self.init("linear_weight_t", &l.weight.t()?.contiguous()?)?;
        self.push("MatMul", vec![wt], vec![]);
        if let Some(b) = &l.bias {
            let bn = self.init("linear_bias", b)?;
            let prev = self.cur.clone();
            let out = self.fresh("add");
            self.nodes.push(Node { op: "Add", inputs: vec![prev, bn], output: out.clone(), attrs: vec![] });
            self.cur = out;
        }
        Ok(self)
    }

    pub fn conv2d(&mut self, c: &Conv2d) -> Result<&mut Self> {
        let w = self.init("conv_weight", &c.weight)?;
        let mut inputs = vec![w];
        if let Some(b) = &c.bias {
            inputs.push(self.init("conv_bias", b)?);
        }
        let k = &c.weight.shape()[2..];
        Ok(self.push(
            "Conv",
            inputs,
            vec![
                Attr::Ints("kernel_shape", vec![k[0] as i64, k[1] as i64]),
                Attr::Ints("strides", vec![c.stride.0 as i64, c.stride.1 as i64]),
                Attr::Ints("pads", vec![c.padding.0 as i64, c.padding.1 as i64, c.padding.0 as i64, c.padding.1 as i64]),
                Attr::Ints("dilations", vec![c.dilation.0 as i64, c.dilation.1 as i64]),
                Attr::Int("group", c.groups as i64),
            ],
        ))
    }

    pub fn max_pool2d(&mut self, p: &MaxPool2d) -> &mut Self {
        self.push(
            "MaxPool",
            vec![],
            vec![
                Attr::Ints("kernel_shape", vec![p.kernel_size.0 as i64, p.kernel_size.1 as i64]),
                Attr::Ints("strides", vec![p.stride.0 as i64, p.stride.1 as i64]),
                Attr::Ints("pads", vec![p.padding.0 as i64, p.padding.1 as i64, p.padding.0 as i64, p.padding.1 as i64]),
            ],
        )
    }

    pub fn avg_pool2d(&mut self, p: &AvgPool2d) -> &mut Self {
        self.push(
            "AveragePool",
            vec![],
            vec![
                Attr::Ints("kernel_shape", vec![p.kernel_size.0 as i64, p.kernel_size.1 as i64]),
                Attr::Ints("strides", vec![p.stride.0 as i64, p.stride.1 as i64]),
                Attr::Ints("pads", vec![p.padding.0 as i64, p.padding.1 as i64, p.padding.0 as i64, p.padding.1 as i64]),
                Attr::Int("count_include_pad", 1),
            ],
        )
    }

    /// Batch normalization with the running statistics (inference mode).
    pub fn batch_norm2d(&mut self, b: &BatchNorm2d) -> Result<&mut Self> {
        let flat = |t: &Tensor| t.reshape(&[t.numel()]);
        let scale = self.init("bn_scale", &flat(&b.weight)?)?;
        let bias = self.init("bn_bias", &flat(&b.bias)?)?;
        let mean = self.init("bn_mean", &flat(&b.running_mean)?)?;
        let var = self.init("bn_var", &flat(&b.running_var)?)?;
        Ok(self.push("BatchNormalization", vec![scale, bias, mean, var], vec![Attr::Float("epsilon", b.eps as f32)]))
    }

    /// Layer normalization over the trailing `normalized_shape` dimensions.
    pub fn layer_norm(&mut self, l: &LayerNorm) -> Result<&mut Self> {
        let w = self.init("ln_weight", &l.weight)?;
        let b = self.init("ln_bias", &l.bias)?;
        Ok(self.push("LayerNormalization", vec![w, b], vec![Attr::Int("axis", -(l.normalized_shape.len() as i64)), Attr::Float("epsilon", l.eps as f32)]))
    }

    /// Embedding lookup: the graph input must then hold `int64` token ids, so build with [`OnnxBuilder::new_int_input`].
    pub fn embedding(&mut self, e: &Embedding) -> Result<&mut Self> {
        let w = self.init("embedding_weight", &e.weight)?;
        let ids = std::mem::replace(&mut self.cur, w);
        let out = self.fresh("gather");
        self.nodes.push(Node { op: "Gather", inputs: vec![self.cur.clone(), ids], output: out.clone(), attrs: vec![Attr::Int("axis", 0)] });
        self.cur = out;
        Ok(self)
    }

    pub fn flatten(&mut self) -> &mut Self {
        self.push("Flatten", vec![], vec![Attr::Int("axis", 1)])
    }
    pub fn relu(&mut self) -> &mut Self {
        self.push("Relu", vec![], vec![])
    }
    pub fn leaky_relu(&mut self, slope: f32) -> &mut Self {
        self.push("LeakyRelu", vec![], vec![Attr::Float("alpha", slope)])
    }
    pub fn sigmoid(&mut self) -> &mut Self {
        self.push("Sigmoid", vec![], vec![])
    }
    pub fn tanh(&mut self) -> &mut Self {
        self.push("Tanh", vec![], vec![])
    }
    pub fn softplus(&mut self) -> &mut Self {
        self.push("Softplus", vec![], vec![])
    }
    pub fn softmax(&mut self, axis: i64) -> &mut Self {
        self.push("Softmax", vec![], vec![Attr::Int("axis", axis)])
    }
    pub fn log_softmax(&mut self, axis: i64) -> &mut Self {
        self.push("LogSoftmax", vec![], vec![Attr::Int("axis", axis)])
    }

    /// `x * sigmoid(x)`.
    pub fn silu(&mut self) -> &mut Self {
        let x = self.cur.clone();
        self.sigmoid();
        let s = self.cur.clone();
        let out = self.fresh("mul");
        self.nodes.push(Node { op: "Mul", inputs: vec![x, s], output: out.clone(), attrs: vec![] });
        self.cur = out;
        self
    }

    /// Exact GELU: `0.5 x (1 + erf(x / sqrt 2))`.
    pub fn gelu(&mut self) -> &mut Self {
        let x = self.cur.clone();
        let inv_sqrt2 = self.init_scalar("inv_sqrt2", std::f32::consts::FRAC_1_SQRT_2);
        let half = self.init_scalar("half", 0.5);
        let one = self.init_scalar("one", 1.0);
        let steps: [(&'static str, Vec<String>); 5] = [
            ("Mul", vec![x.clone(), inv_sqrt2]),
            ("Erf", vec![]),
            ("Add", vec![String::new(), one]),
            ("Mul", vec![x, String::new()]),
            ("Mul", vec![String::new(), half]),
        ];
        let mut prev = String::new();
        for (op, mut inputs) in steps {
            for i in &mut inputs {
                if i.is_empty() {
                    *i = prev.clone();
                }
            }
            if inputs.is_empty() {
                inputs.push(prev.clone());
            }
            let out = self.fresh(&op.to_lowercase());
            self.nodes.push(Node { op, inputs, output: out.clone(), attrs: vec![] });
            prev = out;
        }
        self.cur = prev;
        self
    }

    // -----------------------------------------------------------------------------------------------------------
    // Composite blocks: residual connections, attention, recurrent layers
    // -----------------------------------------------------------------------------------------------------------

    fn op(&mut self, op: &'static str, inputs: Vec<String>, attrs: Vec<Attr>) -> String {
        let output = self.fresh(&op.to_lowercase());
        self.nodes.push(Node { op, inputs, output: output.clone(), attrs });
        output
    }

    fn init_i64_vec(&mut self, base: &str, v: &[i64]) -> String {
        let name = self.fresh(base);
        let raw: Vec<u8> = v.iter().flat_map(|x| x.to_le_bytes()).collect();
        self.inits.push(tensor_proto(&name, &[v.len()], INT64, &raw));
        name
    }

    fn init_i64_scalar(&mut self, base: &str, v: i64) -> String {
        let name = self.fresh(base);
        self.inits.push(tensor_proto(&name, &[], INT64, &v.to_le_bytes()));
        name
    }

    fn linear_on(&mut self, x: &str, l: &Linear) -> Result<String> {
        let wt = self.init("linear_weight_t", &l.weight.t()?.contiguous()?)?;
        let y = self.op("MatMul", vec![x.to_string(), wt], vec![]);
        match &l.bias {
            Some(b) => {
                let bn = self.init("linear_bias", b)?;
                Ok(self.op("Add", vec![y, bn], vec![]))
            }
            None => Ok(y),
        }
    }

    fn slice_last(&mut self, x: &str, start: usize, end: usize) -> String {
        let (s, e, ax) = (self.init_i64_vec("starts", &[start as i64]), self.init_i64_vec("ends", &[end as i64]), self.init_i64_vec("axes", &[-1]));
        self.op("Slice", vec![x.to_string(), s, e, ax], vec![])
    }

    /// The name of the tensor the next layer will read.
    pub fn mark(&self) -> String {
        self.cur.clone()
    }

    /// Makes `name` (a value returned by [`mark`](Self::mark)) the input of the next layer, to build branches.
    pub fn resume(&mut self, name: &str) -> &mut Self {
        self.cur = name.to_string();
        self
    }

    /// `cur = skip + cur` for a tensor remembered with [`mark`](Self::mark).
    pub fn add_marked(&mut self, skip: &str) -> &mut Self {
        let out = self.op("Add", vec![skip.to_string(), self.cur.clone()], vec![]);
        self.cur = out;
        self
    }

    /// Runs the layers added by `body` and adds the block's input to its output (`x + body(x)`).
    pub fn residual(&mut self, body: impl FnOnce(&mut Self) -> Result<()>) -> Result<&mut Self> {
        let skip = self.mark();
        body(self)?;
        Ok(self.add_marked(&skip))
    }

    /// Multi-head self-attention over a `[N, L, E]` input (`batch_first`), without padding masks. With
    /// `causal_len = Some(L)` a causal mask for sequence length `L` is baked in.
    pub fn self_attention(&mut self, m: &MultiheadAttention, causal_len: Option<usize>) -> Result<&mut Self> {
        let (h, hd, e) = (m.num_heads as i64, m.head_dim as i64, m.embed_dim as i64);
        let x = self.cur.clone();
        let split = self.init_i64_vec("split_heads", &[0, 0, h, hd]);
        let mut heads = Vec::new();
        for lin in [&m.q_proj, &m.k_proj, &m.v_proj] {
            let y = self.linear_on(&x, lin)?;
            let r = self.op("Reshape", vec![y, split.clone()], vec![]);
            heads.push(r);
        }
        let q = self.op("Transpose", vec![heads[0].clone()], vec![Attr::Ints("perm", vec![0, 2, 1, 3])]);
        let kt = self.op("Transpose", vec![heads[1].clone()], vec![Attr::Ints("perm", vec![0, 2, 3, 1])]);
        let v = self.op("Transpose", vec![heads[2].clone()], vec![Attr::Ints("perm", vec![0, 2, 1, 3])]);
        let scores = self.op("MatMul", vec![q, kt], vec![]);
        let scale = self.init_scalar("attn_scale", 1.0 / (m.head_dim as f32).sqrt());
        let mut scores = self.op("Mul", vec![scores, scale], vec![]);
        if let Some(l) = causal_len {
            let mask: Vec<f32> = (0..l * l).map(|ix| if ix % l > ix / l { f32::NEG_INFINITY } else { 0.0 }).collect();
            let mt = Tensor::from_slice(&mask, &[l, l])?;
            let mn = self.init("causal_mask", &mt)?;
            scores = self.op("Add", vec![scores, mn], vec![]);
        }
        let w = self.op("Softmax", vec![scores], vec![Attr::Int("axis", -1)]);
        let ctx = self.op("MatMul", vec![w, v], vec![]);
        let ctx = self.op("Transpose", vec![ctx], vec![Attr::Ints("perm", vec![0, 2, 1, 3])]);
        let merge = self.init_i64_vec("merge_heads", &[0, 0, e]);
        let merged = self.op("Reshape", vec![ctx, merge], vec![]);
        self.cur = self.linear_on(&merged, &m.out_proj)?;
        Ok(self)
    }

    /// Unrolls an LSTM over `seq_len` steps on a `[N, T, input]` input, producing the `[N, T, hidden]` output sequence
    /// (zero initial state). Gate order and arithmetic are exactly those of `LSTMCell`.
    pub fn lstm(&mut self, lstm: &LSTM, seq_len: usize) -> Result<&mut Self> {
        let mut seq = self.cur.clone();
        for cell in &lstm.cells {
            let hs = cell.hidden_size;
            let zeros = self.init("zero_state_proj", &Tensor::zeros_f32(&[cell.input_size, hs])?)?;
            let mut outs = Vec::with_capacity(seq_len);
            let mut state: Option<(String, String)> = None;
            for step in 0..seq_len {
                let t = if self.reverse_time { seq_len - 1 - step } else { step };
                let idx = self.init_i64_scalar("t", t as i64);
                let xt = self.op("Gather", vec![seq.clone(), idx], vec![Attr::Int("axis", 1)]);
                let (h_prev, c_prev) = match state.take() {
                    Some(s) => s,
                    None => {
                        let z = self.op("MatMul", vec![xt.clone(), zeros.clone()], vec![]);
                        (z.clone(), z)
                    }
                };
                let gi = self.linear_on(&xt, &cell.ih)?;
                let gh = self.linear_on(&h_prev, &cell.hh)?;
                let gates = self.op("Add", vec![gi, gh], vec![]);
                let part = |s: &mut Self, k: usize, act: &'static str| {
                    let sl = s.slice_last(&gates, k * hs, (k + 1) * hs);
                    s.op(act, vec![sl], vec![])
                };
                let (i, f, g, o) = (part(self, 0, "Sigmoid"), part(self, 1, "Sigmoid"), part(self, 2, "Tanh"), part(self, 3, "Sigmoid"));
                let fc = self.op("Mul", vec![f, c_prev], vec![]);
                let ig = self.op("Mul", vec![i, g], vec![]);
                let c = self.op("Add", vec![fc, ig], vec![]);
                let tc = self.op("Tanh", vec![c.clone()], vec![]);
                let h = self.op("Mul", vec![o, tc], vec![]);
                let axes = self.init_i64_vec("unsq", &[1]);
                outs.push(self.op("Unsqueeze", vec![h.clone(), axes], vec![]));
                state = Some((h, c));
            }
            if self.reverse_time {
                outs.reverse();
            }
            seq = self.op("Concat", outs, vec![Attr::Int("axis", 1)]);
        }
        self.cur = seq;
        Ok(self)
    }

    /// Unrolls a GRU over `seq_len` steps (see [`lstm`](Self::lstm)); arithmetic follows `GRUCell`.
    pub fn gru(&mut self, gru: &GRU, seq_len: usize) -> Result<&mut Self> {
        let mut seq = self.cur.clone();
        for cell in &gru.cells {
            let hs = cell.hidden_size;
            let zeros = self.init("zero_state_proj", &Tensor::zeros_f32(&[cell.input_size, hs])?)?;
            let one = self.init_scalar("one", 1.0);
            let mut outs = Vec::with_capacity(seq_len);
            let mut h_prev: Option<String> = None;
            for step in 0..seq_len {
                let t = if self.reverse_time { seq_len - 1 - step } else { step };
                let idx = self.init_i64_scalar("t", t as i64);
                let xt = self.op("Gather", vec![seq.clone(), idx], vec![Attr::Int("axis", 1)]);
                let hp = match h_prev.take() {
                    Some(h) => h,
                    None => self.op("MatMul", vec![xt.clone(), zeros.clone()], vec![]),
                };
                let gi = self.linear_on(&xt, &cell.ih)?;
                let gh = self.linear_on(&hp, &cell.hh)?;
                let (gi_r, gi_z, gi_n) = (self.slice_last(&gi, 0, hs), self.slice_last(&gi, hs, 2 * hs), self.slice_last(&gi, 2 * hs, 3 * hs));
                let (gh_r, gh_z, gh_n) = (self.slice_last(&gh, 0, hs), self.slice_last(&gh, hs, 2 * hs), self.slice_last(&gh, 2 * hs, 3 * hs));
                let r_in = self.op("Add", vec![gi_r, gh_r], vec![]);
                let r = self.op("Sigmoid", vec![r_in], vec![]);
                let z_in = self.op("Add", vec![gi_z, gh_z], vec![]);
                let z = self.op("Sigmoid", vec![z_in], vec![]);
                let rg = self.op("Mul", vec![r, gh_n], vec![]);
                let n_in = self.op("Add", vec![gi_n, rg], vec![]);
                let n = self.op("Tanh", vec![n_in], vec![]);
                let omz = self.op("Sub", vec![one.clone(), z.clone()], vec![]);
                let a = self.op("Mul", vec![omz, n], vec![]);
                let b = self.op("Mul", vec![z, hp], vec![]);
                let h = self.op("Add", vec![a, b], vec![]);
                let axes = self.init_i64_vec("unsq", &[1]);
                outs.push(self.op("Unsqueeze", vec![h.clone(), axes], vec![]));
                h_prev = Some(h);
            }
            if self.reverse_time {
                outs.reverse();
            }
            seq = self.op("Concat", outs, vec![Attr::Int("axis", 1)]);
        }
        self.cur = seq;
        Ok(self)
    }

    // -----------------------------------------------------------------------------------------------------------
    // More layers
    // -----------------------------------------------------------------------------------------------------------

    /// Declares an additional graph input (for example an attention mask) and returns its name.
    pub fn add_input(&mut self, name: &str, int64: bool, shape: Vec<Dim>) -> String {
        self.extra_inputs.push((name.to_string(), if int64 { INT64 } else { FLOAT }, shape));
        name.to_string()
    }

    /// Transposed convolution (`ConvTranspose`; weight layout `[in, out, kh, kw]`).
    pub fn conv_transpose2d(&mut self, c: &ConvTranspose2d) -> Result<&mut Self> {
        let w = self.init("convt_weight", &c.weight)?;
        let mut inputs = vec![w];
        if let Some(b) = &c.bias {
            inputs.push(self.init("convt_bias", b)?);
        }
        let k = &c.weight.shape()[2..];
        Ok(self.push(
            "ConvTranspose",
            inputs,
            vec![
                Attr::Ints("kernel_shape", vec![k[0] as i64, k[1] as i64]),
                Attr::Ints("strides", vec![c.stride.0 as i64, c.stride.1 as i64]),
                Attr::Ints("pads", vec![c.padding.0 as i64, c.padding.1 as i64, c.padding.0 as i64, c.padding.1 as i64]),
                Attr::Ints("output_padding", vec![c.output_padding.0 as i64, c.output_padding.1 as i64]),
                Attr::Ints("dilations", vec![c.dilation.0 as i64, c.dilation.1 as i64]),
            ],
        ))
    }

    /// Group normalization of `[N, C, ...]`, decomposed into reshape + `InstanceNormalization` + affine.
    pub fn group_norm(&mut self, g: &GroupNorm) -> Result<&mut Self> {
        let x = self.cur.clone();
        let groups = g.num_groups as i64;
        let shape_in = self.op("Shape", vec![x.clone()], vec![]);
        let split = self.init_i64_vec("gn_split", &[0, groups, -1]);
        let grouped = self.op("Reshape", vec![x, split], vec![]);
        let ones = self.init("gn_ones", &Tensor::ones_f32(&[g.num_groups])?)?;
        let zeros = self.init("gn_zeros", &Tensor::zeros_f32(&[g.num_groups])?)?;
        let normed = self.op("InstanceNormalization", vec![grouped, ones, zeros], vec![Attr::Float("epsilon", g.eps as f32)]);
        let back = self.op("Reshape", vec![normed, shape_in], vec![]);
        // per-channel affine, broadcast over the spatial dims through a [1, C, 1, ...] view
        let c = g.num_channels as i64;
        let rank = self.init_i64_vec("gn_affine_shape", &[1, c, 1, 1]);
        let w = self.init("gn_weight", &g.weight.reshape(&[g.num_channels])?)?;
        let b = self.init("gn_bias", &g.bias.reshape(&[g.num_channels])?)?;
        let wv = self.op("Reshape", vec![w, rank.clone()], vec![]);
        let bv = self.op("Reshape", vec![b, rank], vec![]);
        let scaled = self.op("Mul", vec![back, wv], vec![]);
        self.cur = self.op("Add", vec![scaled, bv], vec![]);
        Ok(self)
    }

    /// Resizes the spatial dimensions of `[N, C, H, W]` by `scales` (`Resize`), nearest or bilinear.
    pub fn upsample2d(&mut self, scale: (f32, f32), bilinear: bool) -> &mut Self {
        let scales = {
            let name = self.fresh("scales");
            let raw: Vec<u8> = [1.0f32, 1.0, scale.0, scale.1].iter().flat_map(|v| v.to_le_bytes()).collect();
            self.inits.push(tensor_proto(&name, &[4], FLOAT, &raw));
            name
        };
        let attrs = if bilinear {
            vec![Attr::Str("mode", "linear"), Attr::Str("coordinate_transformation_mode", "half_pixel")]
        } else {
            vec![Attr::Str("mode", "nearest"), Attr::Str("coordinate_transformation_mode", "asymmetric"), Attr::Str("nearest_mode", "floor")]
        };
        self.push("Resize", vec![String::new(), scales], attrs)
    }

    pub fn prelu(&mut self, p: &PReLU) -> Result<&mut Self> {
        let n = p.weight.numel();
        let slope = self.init("prelu_slope", &p.weight.reshape(&if n > 1 { vec![n, 1, 1] } else { vec![1] })?)?;
        Ok(self.push("PRelu", vec![slope], vec![]))
    }

    pub fn elu(&mut self, alpha: f32) -> &mut Self {
        self.push("Elu", vec![], vec![Attr::Float("alpha", alpha)])
    }

    pub fn hardswish(&mut self) -> &mut Self {
        self.push("HardSwish", vec![], vec![])
    }

    /// Global average pooling `[N, C, H, W] -> [N, C, 1, 1]`.
    pub fn global_avg_pool(&mut self) -> &mut Self {
        self.push("GlobalAveragePool", vec![], vec![])
    }

    /// Attention of the current tensor (queries) over `memory` (keys and values; a name from [`mark`](Self::mark) or
    /// an input), optionally with an additive float mask input of shape broadcastable to `[N, H, L, S]` (declare it
    /// with [`add_input`](Self::add_input)) and a baked-in causal mask for square `L = S = causal_len`.
    pub fn attention(&mut self, m: &MultiheadAttention, memory: &str, mask_input: Option<&str>, causal_len: Option<usize>) -> Result<&mut Self> {
        let (h, hd, e) = (m.num_heads as i64, m.head_dim as i64, m.embed_dim as i64);
        let query = self.cur.clone();
        let split = self.init_i64_vec("split_heads", &[0, 0, h, hd]);
        let head = |s: &mut Self, src: &str, lin: &Linear, perm: Vec<i64>| -> Result<String> {
            let y = s.linear_on(src, lin)?;
            let r = s.op("Reshape", vec![y, split.clone()], vec![]);
            Ok(s.op("Transpose", vec![r], vec![Attr::Ints("perm", perm)]))
        };
        let q = head(self, &query, &m.q_proj, vec![0, 2, 1, 3])?;
        let kt = head(self, memory, &m.k_proj, vec![0, 2, 3, 1])?;
        let v = head(self, memory, &m.v_proj, vec![0, 2, 1, 3])?;
        let scores = self.op("MatMul", vec![q, kt], vec![]);
        let scale = self.init_scalar("attn_scale", 1.0 / (m.head_dim as f32).sqrt());
        let mut scores = self.op("Mul", vec![scores, scale], vec![]);
        if let Some(l) = causal_len {
            let mask: Vec<f32> = (0..l * l).map(|ix| if ix % l > ix / l { f32::NEG_INFINITY } else { 0.0 }).collect();
            let mn = self.init("causal_mask", &Tensor::from_slice(&mask, &[l, l])?)?;
            scores = self.op("Add", vec![scores, mn], vec![]);
        }
        if let Some(mask) = mask_input {
            scores = self.op("Add", vec![scores, mask.to_string()], vec![]);
        }
        let w = self.op("Softmax", vec![scores], vec![Attr::Int("axis", -1)]);
        let ctx = self.op("MatMul", vec![w, v], vec![]);
        let ctx = self.op("Transpose", vec![ctx], vec![Attr::Ints("perm", vec![0, 2, 1, 3])]);
        let merge = self.init_i64_vec("merge_heads", &[0, 0, e]);
        let merged = self.op("Reshape", vec![ctx, merge], vec![]);
        self.cur = self.linear_on(&merged, &m.out_proj)?;
        Ok(self)
    }

    /// Bidirectional LSTM over `[N, T, input]` giving `[N, T, 2 * hidden]` (forward and backward outputs
    /// concatenated), matching `BiLSTM`.
    pub fn bi_lstm(&mut self, b: &BiLSTM, seq_len: usize) -> Result<&mut Self> {
        let input = self.cur.clone();
        self.lstm(&b.fwd_lstm, seq_len)?;
        let fwd = self.cur.clone();
        self.cur = input;
        self.reverse_time = true;
        let r = self.lstm(&b.bwd_lstm, seq_len).map(|_| ());
        self.reverse_time = false;
        r?;
        let bwd = self.cur.clone();
        self.cur = self.op("Concat", vec![fwd, bwd], vec![Attr::Int("axis", 2)]);
        Ok(self)
    }

    /// An input of `int64` token ids (for [`embedding`](Self::embedding)); call before adding layers.
    pub fn new_int_input(input: &str, shape: Vec<Dim>) -> Self {
        let mut b = Self::new(input, shape);
        b.n = 0;
        b.input_name = input.to_string();
        b
    }

    /// Encodes the model; the last layer's output is named `output`.
    pub fn finish(mut self, output: &str, int_input: bool, output_shape: Vec<Dim>) -> Result<Vec<u8>> {
        if self.nodes.is_empty() {
            return Err(Error::Msg("OnnxBuilder: the graph has no layers".into()));
        }
        // rename the last output
        let last = self.cur.clone();
        let tail = self.nodes.last_mut().expect("non-empty");
        tail.output = output.to_string();
        for n in &mut self.nodes {
            for i in &mut n.inputs {
                if *i == last {
                    *i = output.to_string();
                }
            }
        }
        let mut graph = Vec::new();
        for (i, n) in self.nodes.iter().enumerate() {
            put_bytes(&mut graph, 1, &n.encode(&format!("node_{i}")));
        }
        put_str(&mut graph, 2, "oxideon_graph");
        for t in &self.inits {
            put_bytes(&mut graph, 5, t);
        }
        put_bytes(&mut graph, 11, &value_info(&self.input_name, if int_input { INT64 } else { FLOAT }, &self.input_shape));
        for (name, dt, shape) in &self.extra_inputs {
            put_bytes(&mut graph, 11, &value_info(name, *dt, shape));
        }
        put_bytes(&mut graph, 12, &value_info(output, FLOAT, &output_shape));
        let mut opset = Vec::new();
        put_varint(&mut opset, 2, 17);
        let mut model = Vec::new();
        put_varint(&mut model, 1, 8); // ir_version
        put_str(&mut model, 2, "oxideon");
        put_bytes(&mut model, 7, &graph);
        put_bytes(&mut model, 8, &opset);
        Ok(model)
    }

    pub fn save(self, path: impl AsRef<std::path::Path>, output: &str, int_input: bool, output_shape: Vec<Dim>) -> Result<()> {
        let bytes = self.finish(output, int_input, output_shape)?;
        std::fs::write(path, bytes).map_err(|e| Error::Serialization(format!("onnx save: {e}")))
    }
}

/// Exports a general SSA [`crate::compiler::IrGraph`] as an ONNX `ModelProto` (IR version 8, opset 17).
pub fn export_ir_graph(
    graph_ir: &crate::compiler::IrGraph,
    input_shapes: &[Vec<usize>],
    model_name: &str,
) -> Result<Vec<u8>> {
    use crate::compiler::IrOp;
    if graph_ir.outputs.is_empty() {
        return Err(Error::Msg("export_ir_graph: graph has no outputs".into()));
    }
    if input_shapes.len() != graph_ir.num_inputs {
        return Err(Error::Msg(format!(
            "export_ir_graph: expected {} input shapes, got {}",
            graph_ir.num_inputs,
            input_shapes.len()
        )));
    }

    let mut nodes: Vec<Node> = Vec::new();
    let mut names: std::collections::HashMap<usize, String> = std::collections::HashMap::new();
    let mut seq = 0usize;
    let mut fresh = |prefix: &str| -> String {
        let s = format!("{prefix}_{seq}");
        seq += 1;
        s
    };

    for (id, ir_node) in graph_ir.nodes.iter().enumerate() {
        let out_name = match &ir_node.op {
            IrOp::Input { index } => format!("input_{index}"),
            IrOp::Add(a, b) => {
                let out = fresh("add");
                nodes.push(Node { op: "Add", inputs: vec![names[a].clone(), names[b].clone()], output: out.clone(), attrs: vec![] });
                out
            }
            IrOp::Sub(a, b) => {
                let out = fresh("sub");
                nodes.push(Node { op: "Sub", inputs: vec![names[a].clone(), names[b].clone()], output: out.clone(), attrs: vec![] });
                out
            }
            IrOp::Mul(a, b) => {
                let out = fresh("mul");
                nodes.push(Node { op: "Mul", inputs: vec![names[a].clone(), names[b].clone()], output: out.clone(), attrs: vec![] });
                out
            }
            IrOp::Div(a, b) => {
                let out = fresh("div");
                nodes.push(Node { op: "Div", inputs: vec![names[a].clone(), names[b].clone()], output: out.clone(), attrs: vec![] });
                out
            }
            IrOp::Matmul(a, b) => {
                let out = fresh("matmul");
                nodes.push(Node { op: "MatMul", inputs: vec![names[a].clone(), names[b].clone()], output: out.clone(), attrs: vec![] });
                out
            }
            IrOp::Transpose2D(a) => {
                let out = fresh("transpose");
                nodes.push(Node { op: "Transpose", inputs: vec![names[a].clone()], output: out.clone(), attrs: vec![Attr::Ints("perm", vec![1, 0])] });
                out
            }
            IrOp::Relu(a) => {
                let out = fresh("relu");
                nodes.push(Node { op: "Relu", inputs: vec![names[a].clone()], output: out.clone(), attrs: vec![] });
                out
            }
            IrOp::Tanh(a) => {
                let out = fresh("tanh");
                nodes.push(Node { op: "Tanh", inputs: vec![names[a].clone()], output: out.clone(), attrs: vec![] });
                out
            }
            IrOp::Exp(a) => {
                let out = fresh("exp");
                nodes.push(Node { op: "Exp", inputs: vec![names[a].clone()], output: out.clone(), attrs: vec![] });
                out
            }
            IrOp::Ln(a) => {
                let out = fresh("log");
                nodes.push(Node { op: "Log", inputs: vec![names[a].clone()], output: out.clone(), attrs: vec![] });
                out
            }
            IrOp::Neg(a) => {
                let out = fresh("neg");
                nodes.push(Node { op: "Neg", inputs: vec![names[a].clone()], output: out.clone(), attrs: vec![] });
                out
            }
            IrOp::Square(a) => {
                let out = fresh("square");
                nodes.push(Node { op: "Mul", inputs: vec![names[a].clone(), names[a].clone()], output: out.clone(), attrs: vec![] });
                out
            }
            IrOp::Silu(a) => {
                let sig = fresh("sig");
                let out = fresh("silu");
                nodes.push(Node { op: "Sigmoid", inputs: vec![names[a].clone()], output: sig.clone(), attrs: vec![] });
                nodes.push(Node { op: "Mul", inputs: vec![names[a].clone(), sig], output: out.clone(), attrs: vec![] });
                out
            }
            IrOp::Gelu(a) => {
                let out = fresh("gelu");
                nodes.push(Node { op: "Gelu", inputs: vec![names[a].clone()], output: out.clone(), attrs: vec![] });
                out
            }
            IrOp::SumAll(a) => {
                let out = fresh("sum");
                nodes.push(Node { op: "ReduceSum", inputs: vec![names[a].clone()], output: out.clone(), attrs: vec![Attr::Int("keepdims", 0)] });
                out
            }
            IrOp::MeanAll(a) => {
                let out = fresh("mean");
                nodes.push(Node { op: "ReduceMean", inputs: vec![names[a].clone()], output: out.clone(), attrs: vec![Attr::Int("keepdims", 0)] });
                out
            }
            IrOp::FusedFma(a, b, c) => {
                let ab = fresh("fma_mul");
                let out = fresh("fma_add");
                nodes.push(Node { op: "Mul", inputs: vec![names[a].clone(), names[b].clone()], output: ab.clone(), attrs: vec![] });
                nodes.push(Node { op: "Add", inputs: vec![ab, names[c].clone()], output: out.clone(), attrs: vec![] });
                out
            }
            IrOp::FusedLinear { x, w, b } => {
                let mm = fresh("lin_mm");
                let out = fresh("lin_add");
                nodes.push(Node { op: "MatMul", inputs: vec![names[x].clone(), names[w].clone()], output: mm.clone(), attrs: vec![] });
                nodes.push(Node { op: "Add", inputs: vec![mm, names[b].clone()], output: out.clone(), attrs: vec![] });
                out
            }
            IrOp::FusedLinearRelu { x, w, b } => {
                let mm = fresh("lin_mm");
                let add = fresh("lin_add");
                let out = fresh("lin_relu");
                nodes.push(Node { op: "MatMul", inputs: vec![names[x].clone(), names[w].clone()], output: mm.clone(), attrs: vec![] });
                nodes.push(Node { op: "Add", inputs: vec![mm, names[b].clone()], output: add.clone(), attrs: vec![] });
                nodes.push(Node { op: "Relu", inputs: vec![add], output: out.clone(), attrs: vec![] });
                out
            }
            IrOp::FusedLinearSilu { x, w, b } => {
                let mm = fresh("lin_mm");
                let add = fresh("lin_add");
                let sig = fresh("lin_sig");
                let out = fresh("lin_silu");
                nodes.push(Node { op: "MatMul", inputs: vec![names[x].clone(), names[w].clone()], output: mm.clone(), attrs: vec![] });
                nodes.push(Node { op: "Add", inputs: vec![mm, names[b].clone()], output: add.clone(), attrs: vec![] });
                nodes.push(Node { op: "Sigmoid", inputs: vec![add.clone()], output: sig.clone(), attrs: vec![] });
                nodes.push(Node { op: "Mul", inputs: vec![add, sig], output: out.clone(), attrs: vec![] });
                out
            }
            IrOp::FusedLinearGelu { x, w, b } => {
                let mm = fresh("lin_mm");
                let add = fresh("lin_add");
                let out = fresh("lin_gelu");
                nodes.push(Node { op: "MatMul", inputs: vec![names[x].clone(), names[w].clone()], output: mm.clone(), attrs: vec![] });
                nodes.push(Node { op: "Add", inputs: vec![mm, names[b].clone()], output: add.clone(), attrs: vec![] });
                nodes.push(Node { op: "Gelu", inputs: vec![add], output: out.clone(), attrs: vec![] });
                out
            }
            IrOp::ZeroLike { .. } | IrOp::OneLike { .. } => {
                return Err(Error::Msg("export_ir_graph: ZeroLike/OneLike require constant materialization".into()));
            }
        };
        names.insert(id, out_name);
    }

    let mut graph = Vec::new();
    for (i, n) in nodes.iter().enumerate() {
        put_bytes(&mut graph, 1, &n.encode(&format!("node_{i}")));
    }
    put_str(&mut graph, 2, model_name);
    for (i, shape) in input_shapes.iter().enumerate() {
        let dims: Vec<Dim> = shape.iter().map(|&d| Dim::Fixed(d)).collect();
        put_bytes(&mut graph, 11, &value_info(&format!("input_{i}"), FLOAT, &dims));
    }
    for &out_id in &graph_ir.outputs {
        let out_name = names.get(&out_id).ok_or_else(|| Error::Msg("export_ir_graph: invalid output ValueId".into()))?;
        put_bytes(&mut graph, 12, &value_info(out_name, FLOAT, &[]));
    }

    let mut opset = Vec::new();
    put_varint(&mut opset, 2, 17);
    let mut model = Vec::new();
    put_varint(&mut model, 1, 8);
    put_str(&mut model, 2, "oxideon");
    put_bytes(&mut model, 7, &graph);
    put_bytes(&mut model, 8, &opset);
    Ok(model)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::nn::Module;

    #[test]
    fn varint_encoding() {
        let mut v = Vec::new();
        varint(&mut v, 300);
        assert_eq!(v, vec![0xAC, 0x02]);
        let mut v = Vec::new();
        varint(&mut v, 1);
        assert_eq!(v, vec![1]);
    }

    #[test]
    fn empty_graph_is_rejected() {
        assert!(OnnxBuilder::new("x", vec![Dim::Fixed(1)]).finish("y", false, vec![Dim::Fixed(1)]).is_err());
    }

    /// Exports small networks, runs them in onnxruntime (when a Python with `onnx` + `onnxruntime` is available at
    /// `ONNX_PYTHON` or `C:/Projects/llvenv`) and compares with this library's own forward pass.
    #[test]
    fn onnxruntime_agrees_with_the_library() {
        let py = std::env::var("ONNX_PYTHON").unwrap_or_else(|_| "C:/Projects/llvenv/Scripts/python.exe".into());
        if !std::path::Path::new(&py).exists() {
            eprintln!("skipping: no Python with onnxruntime at {py}");
            return;
        }
        let dir = std::env::temp_dir().join("oxideon_onnx_test");
        std::fs::create_dir_all(&dir).unwrap();

        // MLP: Linear -> GELU -> Linear -> SiLU -> Linear -> Softmax
        let (l1, l2, l3) = (Linear::new(6, 8).unwrap(), Linear::new(8, 8).unwrap(), Linear::new(8, 3).unwrap());
        let x = Tensor::randn(&[4, 6]).unwrap();
        let h = l1.forward(&x).unwrap().gelu().unwrap();
        let h = l2.forward(&h).unwrap().silu().unwrap();
        let want = l3.forward(&h).unwrap().softmax(-1).unwrap();
        let mut b = OnnxBuilder::new("x", vec!["batch".into(), 6usize.into()]);
        b.linear(&l1).unwrap().gelu().linear(&l2).unwrap().silu().linear(&l3).unwrap().softmax(-1);
        b.save(dir.join("mlp.onnx"), "y", false, vec!["batch".into(), 3usize.into()]).unwrap();
        crate::serialize::save_npy(&x, dir.join("mlp_x.npy")).unwrap();
        crate::serialize::save_npy(&want, dir.join("mlp_y.npy")).unwrap();

        // CNN: Conv -> BatchNorm -> ReLU -> MaxPool -> Conv -> LeakyReLU -> AvgPool -> Flatten -> Linear -> Tanh
        let conv1 = Conv2d::new(2, 4, (3, 3), (1, 1), (1, 1)).unwrap();
        let bn = BatchNorm2d::new(4).unwrap();
        bn.running_mean.assign(&Tensor::randn(&[1, 4, 1, 1]).unwrap()).unwrap();
        bn.running_var.assign(&Tensor::rand_uniform(&[1, 4, 1, 1], 0.5, 2.0).unwrap()).unwrap();
        bn.eval();
        let conv2 = Conv2d::new(4, 3, (3, 3), (2, 2), (0, 0)).unwrap();
        let mp = MaxPool2d::new((2, 2), (2, 2));
        let fc = Linear::new(3, 2).unwrap();
        let xi = Tensor::randn(&[2, 2, 8, 8]).unwrap();
        let mut b = OnnxBuilder::new("x", vec!["batch".into(), 2usize.into(), 8usize.into(), 8usize.into()]);
        b.conv2d(&conv1).unwrap().batch_norm2d(&bn).unwrap().relu().max_pool2d(&mp).conv2d(&conv2).unwrap().leaky_relu(0.1).flatten().linear(&fc).unwrap().tanh();
        let cnn_want = {
            let c = conv2.forward(&mp.forward(&bn.forward(&conv1.forward(&xi).unwrap()).unwrap().relu().unwrap()).unwrap()).unwrap();
            let c = Tensor::where_cond(&c.ge(&c.zeros_like().unwrap()).unwrap(), &c, &c.mul_scalar(0.1).unwrap()).unwrap();
            fc.forward(&c.flatten().unwrap()).unwrap().tanh().unwrap()
        };
        b.save(dir.join("cnn.onnx"), "y", false, vec!["batch".into(), 2usize.into()]).unwrap();
        crate::serialize::save_npy(&xi, dir.join("cnn_x.npy")).unwrap();
        crate::serialize::save_npy(&cnn_want, dir.join("cnn_y.npy")).unwrap();

        // Embedding + LayerNorm
        let emb = Embedding::new(10, 4).unwrap();
        let ln = LayerNorm::new(4, 1e-5).unwrap();
        let ids = Tensor::from_slice(&[1i64, 5, 9, 2, 2, 0], &[2, 3]).unwrap();
        let emb_want = ln.forward(&emb.forward(&ids).unwrap()).unwrap();
        let mut b = OnnxBuilder::new_int_input("ids", vec![2usize.into(), 3usize.into()]);
        b.embedding(&emb).unwrap().layer_norm(&ln).unwrap();
        b.save(dir.join("emb.onnx"), "y", true, vec![2usize.into(), 3usize.into(), 4usize.into()]).unwrap();
        crate::serialize::save_npy(&ids, dir.join("emb_x.npy")).unwrap();
        crate::serialize::save_npy(&emb_want, dir.join("emb_y.npy")).unwrap();

        let script = r#"
import sys, numpy as np, onnx, onnxruntime as ort
d = sys.argv[1]
worst = 0.0
for name in ("mlp", "cnn", "emb"):
    m = onnx.load(f"{d}/{name}.onnx"); onnx.checker.check_model(m)
    s = ort.InferenceSession(f"{d}/{name}.onnx")
    x = np.load(f"{d}/{name}_x.npy"); want = np.load(f"{d}/{name}_y.npy")
    got = s.run(None, {s.get_inputs()[0].name: x})[0]
    assert got.shape == want.shape, (name, got.shape, want.shape)
    worst = max(worst, float(np.abs(got - want).max()))
print("MAXDIFF", worst)
"#;
        let out = std::process::Command::new(&py).arg("-c").arg(script).arg(&dir).output().unwrap();
        let text = String::from_utf8_lossy(&out.stdout).to_string() + &String::from_utf8_lossy(&out.stderr);
        assert!(out.status.success(), "onnxruntime check failed:\n{text}");
        let diff: f64 = text.lines().find_map(|l| l.strip_prefix("MAXDIFF ")).unwrap().trim().parse().unwrap();
        assert!(diff < 1e-4, "max difference {diff}");
        let _ = std::fs::remove_dir_all(&dir);
    }
}

#[cfg(test)]
mod composite_tests {
    use super::*;
    use crate::nn::Module;

    /// Residual attention block, LSTM and GRU exported and run in onnxruntime.
    #[test]
    fn onnxruntime_agrees_on_residual_attention_and_recurrent_layers() {
        let py = std::env::var("ONNX_PYTHON").unwrap_or_else(|_| "C:/Projects/llvenv/Scripts/python.exe".into());
        if !std::path::Path::new(&py).exists() {
            eprintln!("skipping: no Python with onnxruntime at {py}");
            return;
        }
        let dir = std::env::temp_dir().join("oxideon_onnx_composite");
        std::fs::create_dir_all(&dir).unwrap();
        let save = |name: &str, b: OnnxBuilder, x: &Tensor, y: &Tensor, out_shape: Vec<Dim>| {
            b.save(dir.join(format!("{name}.onnx")), "y", false, out_shape).unwrap();
            crate::serialize::save_npy(x, dir.join(format!("{name}_x.npy"))).unwrap();
            crate::serialize::save_npy(y, dir.join(format!("{name}_y.npy"))).unwrap();
        };

        // pre-norm Transformer block with causal attention and a GELU MLP
        let (e, l) = (8usize, 5usize);
        let (ln1, ln2) = (LayerNorm::new(e, 1e-5).unwrap(), LayerNorm::new(e, 1e-5).unwrap());
        let attn = MultiheadAttention::new(e, 2, 0.0, true, None, None, true).unwrap();
        let (up, down) = (Linear::new(e, 16).unwrap(), Linear::new(16, e).unwrap());
        let x = Tensor::randn(&[3, l, e]).unwrap();
        let a = attn.forward_self(&ln1.forward(&x).unwrap(), None, None, true).unwrap();
        let x1 = x.add(&a).unwrap();
        let want = x1.add(&down.forward(&up.forward(&ln2.forward(&x1).unwrap()).unwrap().gelu().unwrap()).unwrap()).unwrap();
        let mut b = OnnxBuilder::new("x", vec!["batch".into(), l.into(), e.into()]);
        b.residual(|b| {
            b.layer_norm(&ln1)?.self_attention(&attn, Some(l))?;
            Ok(())
        })
        .unwrap();
        b.residual(|b| {
            b.layer_norm(&ln2)?.linear(&up)?.gelu().linear(&down)?;
            Ok(())
        })
        .unwrap();
        save("block", b, &x, &want, vec!["batch".into(), l.into(), e.into()]);

        // two-layer LSTM and a GRU
        let lstm = crate::nn::LSTM::new(4, 5, 2).unwrap();
        let xs = Tensor::randn(&[3, 6, 4]).unwrap();
        let want = lstm.forward_seq(&xs, None).unwrap().0;
        let mut b = OnnxBuilder::new("x", vec!["batch".into(), 6usize.into(), 4usize.into()]);
        b.lstm(&lstm, 6).unwrap();
        save("lstm", b, &xs, &want, vec!["batch".into(), 6usize.into(), 5usize.into()]);
        let gru = crate::nn::GRU::new(4, 3, 1).unwrap();
        let want = gru.forward_seq(&xs, None).unwrap().0;
        let mut b = OnnxBuilder::new("x", vec!["batch".into(), 6usize.into(), 4usize.into()]);
        b.gru(&gru, 6).unwrap();
        save("gru", b, &xs, &want, vec!["batch".into(), 6usize.into(), 3usize.into()]);

        let script = r#"
import sys, numpy as np, onnx, onnxruntime as ort
d = sys.argv[1]; worst = 0.0
for name in ("block", "lstm", "gru"):
    m = onnx.load(f"{d}/{name}.onnx"); onnx.checker.check_model(m)
    s = ort.InferenceSession(f"{d}/{name}.onnx")
    x = np.load(f"{d}/{name}_x.npy"); want = np.load(f"{d}/{name}_y.npy")
    got = s.run(None, {s.get_inputs()[0].name: x})[0]
    assert got.shape == want.shape, (name, got.shape, want.shape)
    worst = max(worst, float(np.abs(got - want).max()))
    # a different batch size must work too (dynamic batch dimension)
    s.run(None, {s.get_inputs()[0].name: np.concatenate([x, x], axis=0)})
print("MAXDIFF", worst)
"#;
        let out = std::process::Command::new(&py).arg("-c").arg(script).arg(&dir).output().unwrap();
        let text = String::from_utf8_lossy(&out.stdout).to_string() + &String::from_utf8_lossy(&out.stderr);
        assert!(out.status.success(), "onnxruntime check failed:\n{text}");
        let diff: f64 = text.lines().find_map(|l| l.strip_prefix("MAXDIFF ")).unwrap().trim().parse().unwrap();
        assert!(diff < 1e-4, "max difference {diff}");
        let _ = std::fs::remove_dir_all(&dir);
    }
}

#[cfg(test)]
mod more_layer_tests {
    use super::*;
    use crate::nn::{ELU, Hardswish, Module, Upsample2d};
    use std::collections::HashMap;

    #[test]
    fn onnxruntime_agrees_on_more_layers() {
        let py = std::env::var("ONNX_PYTHON").unwrap_or_else(|_| "C:/Projects/llvenv/Scripts/python.exe".into());
        if !std::path::Path::new(&py).exists() {
            eprintln!("skipping: no Python with onnxruntime at {py}");
            return;
        }
        let dir = std::env::temp_dir().join("oxideon_onnx_more");
        std::fs::create_dir_all(&dir).unwrap();
        let save = |name: &str, b: OnnxBuilder, inputs: HashMap<String, Tensor>, y: &Tensor, out_shape: Vec<Dim>| {
            b.save(dir.join(format!("{name}.onnx")), "y", false, out_shape).unwrap();
            crate::serialize::save_npz(&inputs, dir.join(format!("{name}_in.npz"))).unwrap();
            crate::serialize::save_npy(y, dir.join(format!("{name}_y.npy"))).unwrap();
        };

        // conv -> group norm -> PReLU -> nearest x2 -> transposed conv -> ELU -> hardswish -> global pool
        let conv = Conv2d::new(2, 4, (3, 3), (1, 1), (1, 1)).unwrap();
        let gn = GroupNorm::new(2, 4, 1e-5).unwrap();
        gn.weight.assign(&Tensor::rand_uniform(gn.weight.shape(), 0.5, 1.5).unwrap()).unwrap();
        gn.bias.assign(&Tensor::randn(gn.bias.shape()).unwrap()).unwrap();
        let prelu = PReLU::new(4, 0.25).unwrap();
        prelu.weight.assign(&Tensor::rand_uniform(prelu.weight.shape(), 0.1, 0.4).unwrap()).unwrap();
        let up = Upsample2d::nearest((12, 12));
        let mut convt = ConvTranspose2d::new(4, 3, (3, 3), (2, 2), (1, 1)).unwrap();
        convt.output_padding = (1, 1);
        let x = Tensor::randn(&[2, 2, 6, 6]).unwrap();
        let mut h = prelu.forward(&gn.forward(&conv.forward(&x).unwrap()).unwrap()).unwrap();
        h = up.forward(&h).unwrap();
        h = Hardswish.forward(&ELU(1.0).forward(&convt.forward(&h).unwrap()).unwrap()).unwrap();
        let want = h.mean_axes(&[2, 3], true).unwrap();
        let mut b = OnnxBuilder::new("x", vec!["batch".into(), 2usize.into(), 6usize.into(), 6usize.into()]);
        b.conv2d(&conv).unwrap().group_norm(&gn).unwrap().prelu(&prelu).unwrap().upsample2d((2.0, 2.0), false);
        b.conv_transpose2d(&convt).unwrap().elu(1.0).hardswish().global_avg_pool();
        save("vision", b, HashMap::from([("x".to_string(), x)]), &want, vec!["batch".into(), 3usize.into(), 1usize.into(), 1usize.into()]);

        // cross-attention with a key padding mask (additive, -inf on padded memory slots)
        let attn = MultiheadAttention::new(8, 2, 0.0, true, None, None, true).unwrap();
        let q = Tensor::randn(&[2, 3, 8]).unwrap();
        let mem = Tensor::randn(&[2, 5, 8]).unwrap();
        let pad = Tensor::from_slice(&[false, false, false, true, true, false, false, false, false, true], &[2, 5]).unwrap();
        let want = attn.forward_full(&q, &mem, &mem, Some(&pad), false, None, true, false).unwrap().0;
        let mask: Vec<f32> = pad.to_vec::<bool>().unwrap().iter().map(|&p| if p { f32::NEG_INFINITY } else { 0.0 }).collect();
        let mask = Tensor::from_slice(&mask, &[2, 1, 1, 5]).unwrap();
        let mut b = OnnxBuilder::new("x", vec!["batch".into(), 3usize.into(), 8usize.into()]);
        let m = b.add_input("memory", false, vec!["batch".into(), 5usize.into(), 8usize.into()]);
        let k = b.add_input("mask", false, vec!["batch".into(), 1usize.into(), 1usize.into(), 5usize.into()]);
        b.attention(&attn, &m, Some(&k), None).unwrap();
        save("cross", b, HashMap::from([("x".to_string(), q), ("memory".to_string(), mem), ("mask".to_string(), mask)]), &want, vec!["batch".into(), 3usize.into(), 8usize.into()]);

        // bidirectional LSTM
        let bi = BiLSTM::new(3, 4, 2).unwrap();
        let xs = Tensor::randn(&[2, 5, 3]).unwrap();
        let want = bi.forward(&xs).unwrap();
        let mut b = OnnxBuilder::new("x", vec!["batch".into(), 5usize.into(), 3usize.into()]);
        b.bi_lstm(&bi, 5).unwrap();
        save("bilstm", b, HashMap::from([("x".to_string(), xs)]), &want, vec!["batch".into(), 5usize.into(), 8usize.into()]);

        let script = r#"
import sys, numpy as np, onnx, onnxruntime as ort
d = sys.argv[1]; worst = 0.0
for name in ("vision", "cross", "bilstm"):
    m = onnx.load(f"{d}/{name}.onnx"); onnx.checker.check_model(m)
    s = ort.InferenceSession(f"{d}/{name}.onnx")
    feeds = dict(np.load(f"{d}/{name}_in.npz")); want = np.load(f"{d}/{name}_y.npy")
    got = s.run(None, feeds)[0]
    assert got.shape == want.shape, (name, got.shape, want.shape)
    diff = float(np.abs(got - want).max()); print(name, diff); worst = max(worst, diff)
print("MAXDIFF", worst)
"#;
        let out = std::process::Command::new(&py).arg("-c").arg(script).arg(&dir).output().unwrap();
        let text = String::from_utf8_lossy(&out.stdout).to_string() + &String::from_utf8_lossy(&out.stderr);
        assert!(out.status.success(), "onnxruntime check failed:\n{text}");
        let diff: f64 = text.lines().find_map(|l| l.strip_prefix("MAXDIFF ")).unwrap().trim().parse().unwrap();
        assert!(diff < 1e-4, "max difference {diff}\n{text}");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn compiled_ir_graph_exports_valid_onnx() {
        use crate::compiler::{IrGraph, IrOp};
        let mut g = IrGraph::new();
        let x = g.add_input();
        let w = g.add_input();
        let b = g.add_input();
        let mm = g.push_op(IrOp::Matmul(x, w));
        let add = g.push_op(IrOp::Add(mm, b));
        let act = g.push_op(IrOp::Relu(add));
        g.set_outputs(vec![act]);
        let compiled = g.compile().unwrap();
        let bytes = compiled.to_onnx(&[vec![2, 4], vec![4, 3], vec![3]], "mlp").unwrap();
        assert!(!bytes.is_empty());
        // Verify ONNX ModelProto header (ir_version=8, producer="oxideon") and node ops
        let text = String::from_utf8_lossy(&bytes);
        assert!(text.contains("oxideon"));
        assert!(text.contains("MatMul"));
        assert!(text.contains("Relu"));
    }
}
