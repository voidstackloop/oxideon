//! A shape-aware operator graph with whole-graph optimization and tiled fused execution (the `torch.compile` role).
//!
//! Build a graph with [`OpGraph`] (every node carries its shape and dtype, broadcasting is inferred), call
//! [`OpGraph::compile`] and run the result any number of times with [`CompiledGraph::run`]. Passes: constant folding,
//! algebraic simplification, common-subexpression elimination, dead-code elimination and **elementwise fusion**: a chain of
//! unary / binary operations becomes one kernel that streams the data through cache-sized tiles instead of writing every
//! intermediate to memory. Fused kernels call the very functions the eager CPU backend uses, so results are bit-identical
//! to eager execution. Matrix products, reductions, softmax and view operations run as the eager operations they are.
//!
//! Supported dtypes: `F32` and `F64`.

use crate::backend::cpu::{CpuStorage, eval_bin_f32, eval_bin_f64, eval_unary_f32, eval_unary_f64};
use crate::backend::{BinaryOp, StorageInner, UnaryOp};
use crate::dtype::DType;
use crate::error::{Error, Result};
use crate::layout::broadcast_shapes;
use crate::tensor::Tensor;
use std::collections::HashMap;

pub type Id = usize;

/// One instruction of a fused kernel. Registers `0..n_inputs` are the kernel inputs, later ones the results in order.
#[derive(Debug, Clone)]
enum Instr {
    Un(UnaryOp, usize),
    Bin(BinaryOp, usize, usize),
    /// `register op immediate`
    BinImm(BinaryOp, usize, f64),
    /// `immediate op register`
    ImmBin(BinaryOp, f64, usize),
}

#[derive(Debug, Clone)]
struct Fused {
    inputs: Vec<Id>,
    prog: Vec<Instr>,
}

#[derive(Clone)]
enum Op {
    Input(usize),
    Const(Tensor),
    Unary(UnaryOp, Id),
    Binary(BinaryOp, Id, Id),
    Matmul(Id, Id),
    Reshape(Id),
    Permute(Id, Vec<usize>),
    Broadcast(Id),
    Sum { src: Id, axes: Vec<isize>, keepdim: bool },
    Mean { src: Id, axes: Vec<isize>, keepdim: bool },
    Max { src: Id, axis: isize, keepdim: bool },
    Softmax(Id, isize),
    Fused(Box<Fused>),
}

#[derive(Clone)]
struct Node {
    op: Op,
    shape: Vec<usize>,
    dtype: DType,
}

impl Op {
    fn operands(&self) -> Vec<Id> {
        match self {
            Op::Input(_) | Op::Const(_) => vec![],
            Op::Unary(_, a) | Op::Reshape(a) | Op::Permute(a, _) | Op::Broadcast(a) | Op::Softmax(a, _) => vec![*a],
            Op::Sum { src, .. } | Op::Mean { src, .. } | Op::Max { src, .. } => vec![*src],
            Op::Binary(_, a, b) | Op::Matmul(a, b) => vec![*a, *b],
            Op::Fused(f) => f.inputs.clone(),
        }
    }

    fn remap(&mut self, f: &dyn Fn(Id) -> Id) {
        match self {
            Op::Input(_) | Op::Const(_) => {}
            Op::Unary(_, a) | Op::Reshape(a) | Op::Permute(a, _) | Op::Broadcast(a) | Op::Softmax(a, _) => *a = f(*a),
            Op::Sum { src, .. } | Op::Mean { src, .. } | Op::Max { src, .. } => *src = f(*src),
            Op::Binary(_, a, b) | Op::Matmul(a, b) => {
                *a = f(*a);
                *b = f(*b);
            }
            Op::Fused(fu) => fu.inputs.iter_mut().for_each(|i| *i = f(*i)),
        }
    }

    fn elementwise(&self) -> bool {
        matches!(self, Op::Unary(..) | Op::Binary(..))
    }
}

/// What the passes did.
#[derive(Debug, Clone, Default)]
pub struct CompileStats {
    pub nodes_before: usize,
    pub nodes_after: usize,
    pub folded_constants: usize,
    pub simplified: usize,
    pub eliminated_common: usize,
    pub fused_kernels: usize,
    pub fused_ops: usize,
}

/// The graph under construction.
#[derive(Clone, Default)]
pub struct OpGraph {
    nodes: Vec<Node>,
    outputs: Vec<Id>,
    n_inputs: usize,
}

fn dtype_ok(d: DType) -> Result<()> {
    if matches!(d, DType::F32 | DType::F64) { Ok(()) } else { Err(Error::Msg(format!("opgraph: only F32 and F64 graphs are supported, not {d:?}"))) }
}

impl OpGraph {
    pub fn new() -> Self {
        Self::default()
    }

    fn push(&mut self, op: Op, shape: Vec<usize>, dtype: DType) -> Id {
        self.nodes.push(Node { op, shape, dtype });
        self.nodes.len() - 1
    }

    pub fn shape(&self, id: Id) -> &[usize] {
        &self.nodes[id].shape
    }

    /// A runtime input of the given shape (the n-th call, in order, is the n-th argument of `run`).
    pub fn input(&mut self, shape: &[usize], dtype: DType) -> Result<Id> {
        dtype_ok(dtype)?;
        self.n_inputs += 1;
        Ok(self.push(Op::Input(self.n_inputs - 1), shape.to_vec(), dtype))
    }

    /// A tensor baked into the graph (weights, biases, scalars).
    pub fn constant(&mut self, t: &Tensor) -> Result<Id> {
        dtype_ok(t.dtype())?;
        Ok(self.push(Op::Const(t.detach()), t.shape().to_vec(), t.dtype()))
    }

    /// A one-element constant of the dtype of `like`.
    pub fn scalar(&mut self, value: f64, like: Id) -> Result<Id> {
        let dt = self.nodes[like].dtype;
        let t = if dt == DType::F32 { Tensor::from_slice(&[value as f32], &[1])? } else { Tensor::from_slice(&[value], &[1])? };
        self.constant(&t)
    }

    pub fn set_outputs(&mut self, outputs: &[Id]) {
        self.outputs = outputs.to_vec();
    }

    fn unary(&mut self, op: UnaryOp, a: Id) -> Result<Id> {
        let n = &self.nodes[a];
        let (shape, dt) = (n.shape.clone(), n.dtype);
        Ok(self.push(Op::Unary(op, a), shape, dt))
    }
    pub fn neg(&mut self, a: Id) -> Result<Id> { self.unary(UnaryOp::Neg, a) }
    pub fn abs(&mut self, a: Id) -> Result<Id> { self.unary(UnaryOp::Abs, a) }
    pub fn exp(&mut self, a: Id) -> Result<Id> { self.unary(UnaryOp::Exp, a) }
    pub fn ln(&mut self, a: Id) -> Result<Id> { self.unary(UnaryOp::Ln, a) }
    pub fn sqrt(&mut self, a: Id) -> Result<Id> { self.unary(UnaryOp::Sqrt, a) }
    pub fn rsqrt(&mut self, a: Id) -> Result<Id> { self.unary(UnaryOp::Rsqrt, a) }
    pub fn tanh(&mut self, a: Id) -> Result<Id> { self.unary(UnaryOp::Tanh, a) }
    pub fn sigmoid(&mut self, a: Id) -> Result<Id> { self.unary(UnaryOp::Sigmoid, a) }
    pub fn relu(&mut self, a: Id) -> Result<Id> { self.unary(UnaryOp::Relu, a) }
    pub fn gelu(&mut self, a: Id) -> Result<Id> { self.unary(UnaryOp::Gelu, a) }
    pub fn silu(&mut self, a: Id) -> Result<Id> { self.unary(UnaryOp::Silu, a) }
    pub fn square(&mut self, a: Id) -> Result<Id> { self.unary(UnaryOp::Square, a) }
    pub fn recip(&mut self, a: Id) -> Result<Id> { self.unary(UnaryOp::Recip, a) }

    fn binary(&mut self, op: BinaryOp, a: Id, b: Id) -> Result<Id> {
        let (na, nb) = (&self.nodes[a], &self.nodes[b]);
        if na.dtype != nb.dtype {
            return Err(Error::DTypeMismatch { op: "opgraph binary", expected: na.dtype, got: nb.dtype });
        }
        let shape = broadcast_shapes("opgraph binary", &na.shape, &nb.shape)?;
        let dt = na.dtype;
        Ok(self.push(Op::Binary(op, a, b), shape, dt))
    }
    pub fn add(&mut self, a: Id, b: Id) -> Result<Id> { self.binary(BinaryOp::Add, a, b) }
    pub fn sub(&mut self, a: Id, b: Id) -> Result<Id> { self.binary(BinaryOp::Sub, a, b) }
    pub fn mul(&mut self, a: Id, b: Id) -> Result<Id> { self.binary(BinaryOp::Mul, a, b) }
    pub fn div(&mut self, a: Id, b: Id) -> Result<Id> { self.binary(BinaryOp::Div, a, b) }
    pub fn maximum(&mut self, a: Id, b: Id) -> Result<Id> { self.binary(BinaryOp::Maximum, a, b) }
    pub fn minimum(&mut self, a: Id, b: Id) -> Result<Id> { self.binary(BinaryOp::Minimum, a, b) }
    pub fn pow(&mut self, a: Id, b: Id) -> Result<Id> { self.binary(BinaryOp::Pow, a, b) }
    pub fn add_scalar(&mut self, a: Id, c: f64) -> Result<Id> {
        let s = self.scalar(c, a)?;
        self.add(a, s)
    }
    pub fn mul_scalar(&mut self, a: Id, c: f64) -> Result<Id> {
        let s = self.scalar(c, a)?;
        self.mul(a, s)
    }

    pub fn matmul(&mut self, a: Id, b: Id) -> Result<Id> {
        let (na, nb) = (&self.nodes[a], &self.nodes[b]);
        if na.shape.len() < 2 || nb.shape.len() < 2 {
            return Err(Error::Msg("opgraph matmul: both operands need at least two dimensions".into()));
        }
        let (m, k) = (na.shape[na.shape.len() - 2], na.shape[na.shape.len() - 1]);
        let (k2, n) = (nb.shape[nb.shape.len() - 2], nb.shape[nb.shape.len() - 1]);
        if k != k2 || na.dtype != nb.dtype {
            return Err(Error::Msg(format!("opgraph matmul: cannot multiply {:?} by {:?}", na.shape, nb.shape)));
        }
        let mut shape = broadcast_shapes("opgraph matmul", &na.shape[..na.shape.len() - 2], &nb.shape[..nb.shape.len() - 2])?;
        shape.extend([m, n]);
        let dt = na.dtype;
        Ok(self.push(Op::Matmul(a, b), shape, dt))
    }

    pub fn reshape(&mut self, a: Id, shape: &[usize]) -> Result<Id> {
        let n = &self.nodes[a];
        if n.shape.iter().product::<usize>() != shape.iter().product::<usize>() {
            return Err(Error::Msg(format!("opgraph reshape: cannot reshape {:?} to {shape:?}", n.shape)));
        }
        let dt = n.dtype;
        Ok(self.push(Op::Reshape(a), shape.to_vec(), dt))
    }

    pub fn permute(&mut self, a: Id, axes: &[usize]) -> Result<Id> {
        let n = &self.nodes[a];
        let mut seen = vec![false; n.shape.len()];
        if axes.len() != n.shape.len() || axes.iter().any(|&x| x >= seen.len() || std::mem::replace(&mut seen[x], true)) {
            return Err(Error::Msg(format!("opgraph permute: {axes:?} is not a permutation of {} axes", n.shape.len())));
        }
        let (shape, dt) = (axes.iter().map(|&x| n.shape[x]).collect(), n.dtype);
        Ok(self.push(Op::Permute(a, axes.to_vec()), shape, dt))
    }

    pub fn broadcast_to(&mut self, a: Id, shape: &[usize]) -> Result<Id> {
        let n = &self.nodes[a];
        if broadcast_shapes("opgraph broadcast_to", &n.shape, shape)? != shape {
            return Err(Error::Msg(format!("opgraph broadcast_to: {:?} does not broadcast to {shape:?}", n.shape)));
        }
        let dt = n.dtype;
        Ok(self.push(Op::Broadcast(a), shape.to_vec(), dt))
    }

    fn reduced_shape(shape: &[usize], axes: &[isize], keepdim: bool) -> Result<Vec<usize>> {
        let r = shape.len() as isize;
        let mut norm = Vec::new();
        for &a in axes {
            let a = if a < 0 { a + r } else { a };
            if a < 0 || a >= r {
                return Err(Error::Msg(format!("opgraph reduction: axis {a} out of range for {r} dimensions")));
            }
            norm.push(a as usize);
        }
        Ok(shape.iter().enumerate().filter_map(|(i, &d)| if norm.contains(&i) { keepdim.then_some(1) } else { Some(d) }).collect())
    }

    pub fn sum(&mut self, a: Id, axes: &[isize], keepdim: bool) -> Result<Id> {
        let (shape, dt) = (Self::reduced_shape(&self.nodes[a].shape, axes, keepdim)?, self.nodes[a].dtype);
        Ok(self.push(Op::Sum { src: a, axes: axes.to_vec(), keepdim }, shape, dt))
    }
    pub fn mean(&mut self, a: Id, axes: &[isize], keepdim: bool) -> Result<Id> {
        let (shape, dt) = (Self::reduced_shape(&self.nodes[a].shape, axes, keepdim)?, self.nodes[a].dtype);
        Ok(self.push(Op::Mean { src: a, axes: axes.to_vec(), keepdim }, shape, dt))
    }
    pub fn max(&mut self, a: Id, axis: isize, keepdim: bool) -> Result<Id> {
        let (shape, dt) = (Self::reduced_shape(&self.nodes[a].shape, &[axis], keepdim)?, self.nodes[a].dtype);
        Ok(self.push(Op::Max { src: a, axis, keepdim }, shape, dt))
    }
    pub fn softmax(&mut self, a: Id, axis: isize) -> Result<Id> {
        Self::reduced_shape(&self.nodes[a].shape, &[axis], true)?;
        let (shape, dt) = (self.nodes[a].shape.clone(), self.nodes[a].dtype);
        Ok(self.push(Op::Softmax(a, axis), shape, dt))
    }

    /// `(x - mean) / sqrt(var + eps) * gamma + beta` over the last axis, built from primitive operations.
    pub fn layer_norm(&mut self, x: Id, gamma: Id, beta: Id, eps: f64) -> Result<Id> {
        let mu = self.mean(x, &[-1], true)?;
        let c = self.sub(x, mu)?;
        let sq = self.square(c)?;
        let var = self.mean(sq, &[-1], true)?;
        let ve = self.add_scalar(var, eps)?;
        let inv = self.rsqrt(ve)?;
        let n = self.mul(c, inv)?;
        let g = self.mul(n, gamma)?;
        self.add(g, beta)
    }

    // ------------------------------------------------------------------------------------------------ passes

    fn consumers(&self) -> Vec<usize> {
        let mut c = vec![0usize; self.nodes.len()];
        for n in &self.nodes {
            for o in n.op.operands() {
                c[o] += 1;
            }
        }
        for &o in &self.outputs {
            c[o] += 1;
        }
        c
    }

    /// Redirects every use of a node to another and drops the dead ones, keeping the order.
    fn rewrite(&mut self, alias: &HashMap<Id, Id>) {
        let resolve = |mut i: Id| {
            while let Some(&j) = alias.get(&i) {
                i = j;
            }
            i
        };
        for n in &mut self.nodes {
            n.op.remap(&resolve);
        }
        for o in &mut self.outputs {
            *o = resolve(*o);
        }
    }

    fn dce(&mut self) {
        let mut live = vec![false; self.nodes.len()];
        let mut stack: Vec<Id> = self.outputs.clone();
        while let Some(i) = stack.pop() {
            if !std::mem::replace(&mut live[i], true) {
                stack.extend(self.nodes[i].op.operands());
            }
        }
        // inputs stay (they define the calling convention)
        for (i, n) in self.nodes.iter().enumerate() {
            if matches!(n.op, Op::Input(_)) {
                live[i] = true;
            }
        }
        let mut new_id = vec![usize::MAX; self.nodes.len()];
        let mut kept = Vec::new();
        for (i, n) in self.nodes.iter().enumerate() {
            if live[i] {
                new_id[i] = kept.len();
                kept.push(n.clone());
            }
        }
        for n in &mut kept {
            n.op.remap(&|i| new_id[i]);
        }
        self.outputs = self.outputs.iter().map(|&o| new_id[o]).collect();
        self.nodes = kept;
    }

    fn fold_constants(&mut self, stats: &mut CompileStats) -> Result<()> {
        for i in 0..self.nodes.len() {
            let ops = self.nodes[i].op.operands();
            let foldable = !ops.is_empty() && !matches!(self.nodes[i].op, Op::Fused(_)) && ops.iter().all(|&o| matches!(self.nodes[o].op, Op::Const(_)));
            if foldable {
                let vals: HashMap<Id, Tensor> = ops.iter().filter_map(|&o| if let Op::Const(t) = &self.nodes[o].op { Some((o, t.clone())) } else { None }).collect();
                let t = eval_node(&self.nodes[i], &|id| vals.get(&id).cloned().ok_or_else(|| Error::Msg("opgraph: missing operand".into())), &[])?;
                self.nodes[i].op = Op::Const(t);
                stats.folded_constants += 1;
            }
        }
        Ok(())
    }

    fn scalar_value(&self, id: Id) -> Option<f64> {
        match &self.nodes[id].op {
            Op::Const(t) if t.numel() == 1 => t.to_dtype(DType::F64).ok()?.to_vec::<f64>().ok()?.first().copied(),
            _ => None,
        }
    }

    fn simplify(&mut self, stats: &mut CompileStats) {
        let mut alias: HashMap<Id, Id> = HashMap::new();
        for i in 0..self.nodes.len() {
            let n = self.nodes[i].clone();
            let same = |a: Id| self.nodes[a].shape == n.shape;
            let target = match &n.op {
                Op::Binary(BinaryOp::Add, a, b) if self.scalar_value(*b) == Some(0.0) && same(*a) => Some(*a),
                Op::Binary(BinaryOp::Add, a, b) if self.scalar_value(*a) == Some(0.0) && same(*b) => Some(*b),
                Op::Binary(BinaryOp::Sub, a, b) if self.scalar_value(*b) == Some(0.0) && same(*a) => Some(*a),
                Op::Binary(BinaryOp::Mul, a, b) if self.scalar_value(*b) == Some(1.0) && same(*a) => Some(*a),
                Op::Binary(BinaryOp::Mul, a, b) if self.scalar_value(*a) == Some(1.0) && same(*b) => Some(*b),
                Op::Binary(BinaryOp::Div, a, b) if self.scalar_value(*b) == Some(1.0) && same(*a) => Some(*a),
                Op::Unary(UnaryOp::Neg, a) => match &self.nodes[*a].op {
                    Op::Unary(UnaryOp::Neg, inner) => Some(*inner),
                    _ => None,
                },
                Op::Reshape(a) | Op::Broadcast(a) if same(*a) => Some(*a),
                Op::Reshape(a) => match &self.nodes[*a].op {
                    // reshape(reshape(x)) = reshape(x)
                    Op::Reshape(inner) => {
                        self.nodes[i].op = Op::Reshape(*inner);
                        stats.simplified += 1;
                        None
                    }
                    _ => None,
                },
                Op::Permute(a, axes) if axes.iter().enumerate().all(|(k, &x)| k == x) => Some(*a),
                _ => None,
            };
            if let Some(t) = target {
                alias.insert(i, t);
                stats.simplified += 1;
            }
        }
        self.rewrite(&alias);
    }

    fn cse(&mut self, stats: &mut CompileStats) {
        let mut seen: HashMap<String, Id> = HashMap::new();
        let mut alias: HashMap<Id, Id> = HashMap::new();
        for i in 0..self.nodes.len() {
            let mut op = self.nodes[i].op.clone();
            op.remap(&|x| {
                let mut x = x;
                while let Some(&y) = alias.get(&x) {
                    x = y;
                }
                x
            });
            let key = match &op {
                Op::Input(_) | Op::Fused(_) => None,
                Op::Const(t) => Some(format!("const#{}", t.id())),
                Op::Unary(u, a) => Some(format!("u{u:?}({a})")),
                Op::Binary(b, x, y) => Some(format!("b{b:?}({x},{y})")),
                Op::Matmul(x, y) => Some(format!("mm({x},{y})")),
                Op::Reshape(a) => Some(format!("rs{:?}({a})", self.nodes[i].shape)),
                Op::Permute(a, ax) => Some(format!("pm{ax:?}({a})")),
                Op::Broadcast(a) => Some(format!("bc{:?}({a})", self.nodes[i].shape)),
                Op::Sum { src, axes, keepdim } => Some(format!("sum{axes:?}{keepdim}({src})")),
                Op::Mean { src, axes, keepdim } => Some(format!("mean{axes:?}{keepdim}({src})")),
                Op::Max { src, axis, keepdim } => Some(format!("max{axis}{keepdim}({src})")),
                Op::Softmax(a, ax) => Some(format!("sm{ax}({a})")),
            };
            if let Some(k) = key {
                match seen.get(&k) {
                    Some(&first) => {
                        alias.insert(i, first);
                        stats.eliminated_common += 1;
                    }
                    None => {
                        seen.insert(k, i);
                    }
                }
            }
        }
        self.rewrite(&alias);
    }

    /// Groups elementwise operations into fused kernels (regions of at least two operations).
    fn fuse(&mut self, stats: &mut CompileStats) {
        let cons = self.consumers();
        let n = self.nodes.len();
        let mut absorbed = vec![false; n];
        let is_output: Vec<bool> = (0..n).map(|i| self.outputs.contains(&i)).collect();
        for root in (0..n).rev() {
            if absorbed[root] || !self.nodes[root].op.elementwise() {
                continue;
            }
            // collect the members: operands with the root's shape that feed only this region
            let mut members = vec![root];
            let mut frontier = vec![root];
            while let Some(m) = frontier.pop() {
                for o in self.nodes[m].op.operands() {
                    if self.nodes[o].op.elementwise() && !absorbed[o] && !members.contains(&o) && cons[o] == 1 && !is_output[o] && self.nodes[o].shape == self.nodes[root].shape {
                        members.push(o);
                        frontier.push(o);
                    }
                }
            }
            if members.len() < 2 {
                continue;
            }
            members.sort_unstable();
            let (inputs, prog) = self.build_program(&members);
            for &m in &members[..members.len() - 1] {
                absorbed[m] = true;
            }
            self.nodes[root].op = Op::Fused(Box::new(Fused { inputs, prog }));
            stats.fused_kernels += 1;
            stats.fused_ops += members.len();
        }
        let alias: HashMap<Id, Id> = HashMap::new();
        self.rewrite(&alias);
        self.dce();
    }

    fn build_program(&self, members: &[Id]) -> (Vec<Id>, Vec<Instr>) {
        let member_set: std::collections::HashSet<Id> = members.iter().copied().collect();
        let mut inputs: Vec<Id> = Vec::new();
        for &m in members {
            for o in self.nodes[m].op.operands() {
                if !member_set.contains(&o) && self.scalar_value(o).is_none() && !inputs.contains(&o) {
                    inputs.push(o);
                }
            }
        }
        let mut reg: HashMap<Id, usize> = inputs.iter().enumerate().map(|(r, &i)| (i, r)).collect();
        let mut prog = Vec::new();
        for &m in members {
            let operand = |id: Id| -> std::result::Result<usize, f64> {
                match reg.get(&id) {
                    Some(&r) => Ok(r),
                    None => Err(self.scalar_value(id).unwrap_or(0.0)),
                }
            };
            let instr = match &self.nodes[m].op {
                Op::Unary(u, a) => Instr::Un(*u, operand(*a).unwrap_or(0)),
                Op::Binary(b, x, y) => match (operand(*x), operand(*y)) {
                    (Ok(rx), Ok(ry)) => Instr::Bin(*b, rx, ry),
                    (Ok(rx), Err(c)) => Instr::BinImm(*b, rx, c),
                    (Err(c), Ok(ry)) => Instr::ImmBin(*b, c, ry),
                    (Err(_), Err(_)) => unreachable!("constant folding removes all-constant operations"),
                },
                _ => unreachable!("only elementwise nodes are fused"),
            };
            reg.insert(m, inputs.len() + prog.len());
            prog.push(instr);
        }
        (inputs, prog)
    }

    /// Runs the optimization passes and returns the executable graph.
    pub fn compile(mut self) -> Result<CompiledGraph> {
        if self.outputs.is_empty() {
            return Err(Error::Msg("opgraph: set_outputs was not called".into()));
        }
        let mut stats = CompileStats { nodes_before: self.nodes.len(), ..Default::default() };
        self.fold_constants(&mut stats)?;
        self.simplify(&mut stats);
        self.cse(&mut stats);
        self.dce();
        self.fuse(&mut stats);
        stats.nodes_after = self.nodes.len();
        // last use of every value, for freeing intermediates early
        let mut last_use = vec![usize::MAX; self.nodes.len()];
        for (i, n) in self.nodes.iter().enumerate() {
            for o in n.op.operands() {
                last_use[o] = i;
            }
        }
        let mut input_shapes = vec![(Vec::new(), DType::F32); self.n_inputs];
        for n in &self.nodes {
            if let Op::Input(k) = n.op {
                input_shapes[k] = (n.shape.clone(), n.dtype);
            }
        }
        Ok(CompiledGraph { graph: self, last_use, stats, input_shapes })
    }
}

// ---------------------------------------------------------------------------------------------------------- execution

/// Read-only view of a CPU tensor's elements (contiguous copy only when the layout needs one).
enum View<'a, T> {
    Borrowed(std::sync::RwLockReadGuard<'a, StorageInner>, usize, usize, std::marker::PhantomData<T>),
    Owned(Vec<T>),
}

trait Float: Copy + Send + Sync + 'static {
    fn un(x: Self, op: UnaryOp) -> Self;
    fn bin(x: Self, y: Self, op: BinaryOp) -> Self;
    fn from_f64(v: f64) -> Self;
    fn slice(c: &CpuStorage) -> Option<&[Self]>;
    fn tensor(v: Vec<Self>, shape: &[usize]) -> Result<Tensor>;
    fn to_vec(t: &Tensor) -> Result<Vec<Self>>;
}

macro_rules! float_impl {
    ($t:ty, $variant:ident, $un:ident, $bin:ident) => {
        impl Float for $t {
            fn un(x: Self, op: UnaryOp) -> Self {
                $un(x, op)
            }
            fn bin(x: Self, y: Self, op: BinaryOp) -> Self {
                $bin(x, y, op)
            }
            fn from_f64(v: f64) -> Self {
                v as $t
            }
            fn slice(c: &CpuStorage) -> Option<&[Self]> {
                if let CpuStorage::$variant(v) = c { Some(v) } else { None }
            }
            fn tensor(v: Vec<Self>, shape: &[usize]) -> Result<Tensor> {
                Tensor::from_vec(v, shape)
            }
            fn to_vec(t: &Tensor) -> Result<Vec<Self>> {
                t.to_vec::<$t>()
            }
        }
    };
}
float_impl!(f32, F32, eval_unary_f32, eval_bin_f32);
float_impl!(f64, F64, eval_unary_f64, eval_bin_f64);

impl<'a, T: Float> View<'a, T> {
    fn new(t: &'a Tensor) -> Result<Self> {
        if t.is_contiguous() {
            let guard = t.storage().read();
            let cpu = match &*guard {
                StorageInner::Cpu(c) => Some(c),
                StorageInner::Accelerator { storage, .. } => Some(storage),
                #[allow(unreachable_patterns)]
                _ => None,
            };
            if let Some(s) = cpu.and_then(T::slice) {
                let (off, n) = (t.storage_offset(), t.numel());
                if off + n <= s.len() {
                    return Ok(View::Borrowed(guard, off, n, std::marker::PhantomData));
                }
            }
        }
        Ok(View::Owned(T::to_vec(t)?))
    }

    fn data(&self) -> &[T] {
        match self {
            View::Owned(v) => v,
            View::Borrowed(g, off, n, _) => {
                let cpu = match &**g {
                    StorageInner::Cpu(c) => c,
                    StorageInner::Accelerator { storage, .. } => storage,
                    #[allow(unreachable_patterns)]
                    _ => unreachable!(),
                };
                &T::slice(cpu).expect("dtype checked at construction")[*off..*off + *n]
            }
        }
    }
}

const TILE: usize = 4096;

/// Fills `dst` with elements `start..start + dst.len()` of `src` (of shape `in_shape`) broadcast to `out_shape`.
fn gather<T: Copy>(src: &[T], in_shape: &[usize], out_shape: &[usize], start: usize, dst: &mut [T]) {
    let r = out_shape.len();
    let pad = r - in_shape.len();
    // stride of the source along every output axis (0 where broadcast)
    let mut stride = vec![0usize; r];
    let mut acc = 1;
    for k in (0..in_shape.len()).rev() {
        stride[pad + k] = if in_shape[k] == 1 { 0 } else { acc };
        acc *= in_shape[k];
    }
    let last = out_shape[r - 1].max(1);
    let mut idx = vec![0usize; r];
    let mut rem = start;
    for k in (0..r).rev() {
        idx[k] = rem % out_shape[k].max(1);
        rem /= out_shape[k].max(1);
    }
    let mut done = 0;
    while done < dst.len() {
        let base: usize = idx.iter().zip(&stride).map(|(i, s)| i * s).sum();
        let run = (last - idx[r - 1]).min(dst.len() - done);
        if stride[r - 1] == 1 {
            dst[done..done + run].copy_from_slice(&src[base..base + run]);
        } else {
            dst[done..done + run].fill(src[base]);
        }
        done += run;
        // advance the odometer by `run` along the last axis, carrying outward
        idx[r - 1] += run;
        let mut k = r - 1;
        while idx[k] >= out_shape[k].max(1) && k > 0 {
            idx[k] = 0;
            idx[k - 1] += 1;
            k -= 1;
        }
    }
}

fn run_fused<T: Float>(f: &Fused, inputs: &[&Tensor], out_shape: &[usize]) -> Result<Tensor> {
    let views: Vec<View<T>> = inputs.iter().map(|t| View::<T>::new(t)).collect::<Result<_>>()?;
    let n: usize = out_shape.iter().product();
    let n_in = f.inputs.len();
    let in_shapes: Vec<&[usize]> = inputs.iter().map(|t| t.shape()).collect();
    let mut out: Vec<T> = Vec::with_capacity(n);
    // SAFETY-free initialization: elements are overwritten tile by tile before being read
    out.resize(n, T::from_f64(0.0));
    let work = |tile_no: usize, chunk: &mut [T], scratch: &mut Vec<Vec<T>>| {
        let start = tile_no * TILE;
        let len = chunk.len();
        // inputs that do not already have the output shape are gathered into a tile buffer first
        let mut gathered: Vec<Option<Vec<T>>> = Vec::with_capacity(n_in);
        for (i, v) in views.iter().enumerate() {
            if in_shapes[i] == out_shape {
                gathered.push(None);
            } else {
                let mut buf = vec![T::from_f64(0.0); len];
                gather(v.data(), in_shapes[i], out_shape, start, &mut buf);
                gathered.push(Some(buf));
            }
        }
        let n_regs = f.prog.len();
        scratch.resize_with(n_regs, Vec::new);
        for r in 0..n_regs {
            scratch[r].resize(len, T::from_f64(0.0));
        }
        for (k, ins) in f.prog.iter().enumerate() {
            let mut dst = std::mem::take(&mut scratch[k]);
            {
                let src = |reg: usize| -> &[T] {
                    if reg < n_in {
                        match &gathered[reg] {
                            Some(b) => &b[..len],
                            None => &views[reg].data()[start..start + len],
                        }
                    } else {
                        &scratch[reg - n_in][..len]
                    }
                };
                match *ins {
                    Instr::Un(op, a) => {
                        let a = src(a);
                        for i in 0..len {
                            dst[i] = T::un(a[i], op);
                        }
                    }
                    Instr::Bin(op, a, b) => {
                        let (a, b) = (src(a), src(b));
                        for i in 0..len {
                            dst[i] = T::bin(a[i], b[i], op);
                        }
                    }
                    Instr::BinImm(op, a, c) => {
                        let (a, c) = (src(a), T::from_f64(c));
                        for i in 0..len {
                            dst[i] = T::bin(a[i], c, op);
                        }
                    }
                    Instr::ImmBin(op, c, b) => {
                        let (b, c) = (src(b), T::from_f64(c));
                        for i in 0..len {
                            dst[i] = T::bin(c, b[i], op);
                        }
                    }
                }
            }
            scratch[k] = dst;
        }
        chunk.copy_from_slice(&scratch[n_regs - 1][..len]);
    };
    #[cfg(feature = "rayon")]
    {
        use rayon::prelude::*;
        out.par_chunks_mut(TILE).enumerate().for_each_init(Vec::new, |scratch, (i, chunk)| work(i, chunk, scratch));
    }
    #[cfg(not(feature = "rayon"))]
    {
        let mut scratch = Vec::new();
        for (i, chunk) in out.chunks_mut(TILE).enumerate() {
            work(i, chunk, &mut scratch);
        }
    }
    T::tensor(out, out_shape)
}

/// Evaluates one node given its operands (used for execution and for constant folding).
fn eval_node(node: &Node, get: &dyn Fn(Id) -> Result<Tensor>, _inputs: &[Tensor]) -> Result<Tensor> {
    let single = |instr: Instr, ins: Vec<Id>| -> Result<Tensor> {
        let f = Fused { inputs: ins.clone(), prog: vec![instr] };
        let ts: Vec<Tensor> = ins.iter().map(|&i| get(i)).collect::<Result<_>>()?;
        let refs: Vec<&Tensor> = ts.iter().collect();
        match node.dtype {
            DType::F32 => run_fused::<f32>(&f, &refs, &node.shape),
            _ => run_fused::<f64>(&f, &refs, &node.shape),
        }
    };
    match &node.op {
        Op::Input(_) => Err(Error::Msg("opgraph: inputs are bound by the executor".into())),
        Op::Const(t) => Ok(t.clone()),
        Op::Unary(u, a) => single(Instr::Un(*u, 0), vec![*a]),
        Op::Binary(b, x, y) => {
            if x == y {
                // both operands are the same value: feed it once
                single(Instr::Bin(*b, 0, 0), vec![*x])
            } else {
                single(Instr::Bin(*b, 0, 1), vec![*x, *y])
            }
        }
        Op::Matmul(a, b) => get(*a)?.matmul(&get(*b)?),
        Op::Reshape(a) => get(*a)?.contiguous()?.reshape(&node.shape),
        Op::Permute(a, axes) => get(*a)?.permute(axes),
        Op::Broadcast(a) => get(*a)?.broadcast_to(&node.shape),
        Op::Sum { src, axes, keepdim } => get(*src)?.sum_axes(axes, *keepdim),
        Op::Mean { src, axes, keepdim } => get(*src)?.mean_axes(axes, *keepdim),
        Op::Max { src, axis, keepdim } => get(*src)?.max_axis(*axis, *keepdim),
        Op::Softmax(a, axis) => get(*a)?.softmax(*axis),
        Op::Fused(f) => {
            let ts: Vec<Tensor> = f.inputs.iter().map(|&i| get(i)).collect::<Result<_>>()?;
            let refs: Vec<&Tensor> = ts.iter().collect();
            match node.dtype {
                DType::F32 => run_fused::<f32>(f, &refs, &node.shape),
                _ => run_fused::<f64>(f, &refs, &node.shape),
            }
        }
    }
}

/// An optimized graph ready to run.
pub struct CompiledGraph {
    graph: OpGraph,
    last_use: Vec<usize>,
    stats: CompileStats,
    input_shapes: Vec<(Vec<usize>, DType)>,
}

impl CompiledGraph {
    pub fn stats(&self) -> &CompileStats {
        &self.stats
    }

    /// Runs the graph. Inputs must match the declared shapes and dtypes; intermediates are freed after their last use.
    pub fn run(&self, inputs: &[Tensor]) -> Result<Vec<Tensor>> {
        if inputs.len() != self.input_shapes.len() {
            return Err(Error::Msg(format!("opgraph: expected {} inputs, got {}", self.input_shapes.len(), inputs.len())));
        }
        for (i, (t, (shape, dt))) in inputs.iter().zip(&self.input_shapes).enumerate() {
            if t.shape() != shape.as_slice() || t.dtype() != *dt {
                return Err(Error::Msg(format!("opgraph: input {i} is {:?} {:?}, the graph was built for {shape:?} {dt:?}", t.shape(), t.dtype())));
            }
        }
        let g = &self.graph;
        let mut values: Vec<Option<Tensor>> = vec![None; g.nodes.len()];
        for (i, node) in g.nodes.iter().enumerate() {
            let v = match &node.op {
                Op::Input(k) => inputs[*k].clone(),
                _ => eval_node(node, &|id| values[id].clone().ok_or_else(|| Error::Msg("opgraph: value was freed too early".into())), inputs)?,
            };
            values[i] = Some(v);
            for o in node.op.operands() {
                if self.last_use[o] == i && !g.outputs.contains(&o) {
                    values[o] = None;
                }
            }
        }
        g.outputs.iter().map(|&o| values[o].clone().ok_or_else(|| Error::Msg("opgraph: output missing".into()))).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close(a: &Tensor, b: &Tensor) {
        assert_eq!(a.shape(), b.shape());
        let (x, y) = (a.to_vec::<f32>().unwrap(), b.to_vec::<f32>().unwrap());
        assert!(x.iter().zip(&y).all(|(p, q)| (p - q).abs() <= 1e-5 * (1.0 + q.abs())), "{x:?} vs {y:?}");
    }

    #[test]
    fn mlp_block_with_layer_norm_matches_eager_and_fuses() {
        crate::rng::manual_seed(3);
        let (b, d, h) = (6, 16, 32);
        let (w1, b1) = (Tensor::randn(&[d, h]).unwrap(), Tensor::randn(&[h]).unwrap());
        let (w2, gamma, beta) = (Tensor::randn(&[h, d]).unwrap(), Tensor::randn(&[d]).unwrap(), Tensor::randn(&[d]).unwrap());
        let x = Tensor::randn(&[b, d]).unwrap();
        let mut g = OpGraph::new();
        let xi = g.input(&[b, d], DType::F32).unwrap();
        let (cw1, cb1, cw2, cg, cb) = (g.constant(&w1).unwrap(), g.constant(&b1).unwrap(), g.constant(&w2).unwrap(), g.constant(&gamma).unwrap(), g.constant(&beta).unwrap());
        let y = g.matmul(xi, cw1).unwrap();
        let y = g.add(y, cb1).unwrap();
        let y = g.gelu(y).unwrap();
        let y = g.matmul(y, cw2).unwrap();
        let y = g.add(y, xi).unwrap();
        let y = g.layer_norm(y, cg, cb, 1e-5).unwrap();
        let y = g.softmax(y, -1).unwrap();
        g.set_outputs(&[y]);
        let c = g.compile().unwrap();
        assert!(c.stats().fused_kernels >= 2, "{:?}", c.stats());
        let eager = {
            let y = x.matmul(&w1).unwrap().add(&b1).unwrap().gelu().unwrap().matmul(&w2).unwrap().add(&x).unwrap();
            let mu = y.mean_axes(&[-1], true).unwrap();
            let cen = y.sub(&mu).unwrap();
            let var = cen.square().unwrap().mean_axes(&[-1], true).unwrap();
            let n = cen.mul(&var.add_scalar(1e-5).unwrap().rsqrt().unwrap()).unwrap();
            n.mul(&gamma).unwrap().add(&beta).unwrap().softmax(-1).unwrap()
        };
        let out = c.run(std::slice::from_ref(&x)).unwrap();
        close(&out[0], &eager);
        // a second run with other data reuses the compiled graph
        let x2 = Tensor::randn(&[b, d]).unwrap();
        assert_eq!(c.run(&[x2]).unwrap()[0].shape(), &[b, d]);
        assert!(c.run(&[Tensor::randn(&[b + 1, d]).unwrap()]).is_err());
    }

    #[test]
    fn broadcasting_folding_cse_and_simplification() {
        let x = Tensor::randn(&[4, 1, 5]).unwrap();
        let col = Tensor::randn(&[3, 1]).unwrap();
        let mut g = OpGraph::new();
        let xi = g.input(&[4, 1, 5], DType::F32).unwrap();
        let ci = g.constant(&col).unwrap();
        let two = g.scalar(2.0, xi).unwrap();
        let three = g.scalar(3.0, xi).unwrap();
        let k = g.mul(two, three).unwrap(); // folds to 6
        let a = g.add(xi, ci).unwrap(); // [4, 3, 5]
        let b = g.mul(a, k).unwrap();
        let dup = g.mul(a, k).unwrap(); // same as b: common subexpression
        let sum = g.add(b, dup).unwrap();
        let zero = g.scalar(0.0, xi).unwrap();
        let plus0 = g.add(sum, zero).unwrap(); // x + 0 simplifies away
        let out = g.exp(plus0).unwrap();
        g.set_outputs(&[out]);
        let c = g.compile().unwrap();
        let s = c.stats().clone();
        assert!(s.folded_constants >= 1 && s.eliminated_common >= 1 && s.simplified >= 1, "{s:?}");
        let want = x.add(&col).unwrap().mul_scalar(6.0).unwrap();
        let want = want.add(&want).unwrap().exp().unwrap();
        close(&c.run(&[x]).unwrap()[0], &want);
    }

    #[test]
    fn multiple_outputs_reductions_views_and_f64() {
        let x = Tensor::randn(&[2, 3, 4]).unwrap().to_dtype(DType::F64).unwrap();
        let mut g = OpGraph::new();
        let xi = g.input(&[2, 3, 4], DType::F64).unwrap();
        let p = g.permute(xi, &[0, 2, 1]).unwrap();
        let r = g.reshape(p, &[8, 3]).unwrap();
        let sq = g.square(r).unwrap();
        let sig = g.sigmoid(sq).unwrap();
        let s = g.sum(sig, &[0], false).unwrap();
        let m = g.max(r, 1, true).unwrap();
        g.set_outputs(&[s, m]);
        let c = g.compile().unwrap();
        let o = c.run(std::slice::from_ref(&x)).unwrap();
        let r = x.permute(&[0, 2, 1]).unwrap().contiguous().unwrap().reshape(&[8, 3]).unwrap();
        let want_s = r.square().unwrap().sigmoid().unwrap().sum_axes(&[0], false).unwrap();
        assert!(o[0].allclose(&want_s, 1e-12, 1e-12).unwrap());
        assert!(o[1].allclose(&r.max_axis(1, true).unwrap(), 0.0, 0.0).unwrap());
        assert!(OpGraph::new().input(&[2], DType::I32).is_err());
    }

    #[test]
    fn fused_kernels_are_bit_identical_to_eager_beyond_one_tile() {
        crate::rng::manual_seed(8);
        let n = TILE * 3 + 17;
        let x = Tensor::randn(&[n]).unwrap();
        let mut g = OpGraph::new();
        let xi = g.input(&[n], DType::F32).unwrap();
        let a = g.tanh(xi).unwrap();
        let b = g.mul_scalar(a, 1.7).unwrap();
        let c = g.silu(b).unwrap();
        let d = g.add(c, xi).unwrap();
        let e = g.exp(d).unwrap();
        g.set_outputs(&[e]);
        let comp = g.compile().unwrap();
        assert_eq!(comp.stats().fused_kernels, 1);
        let want = x.tanh().unwrap().mul_scalar(1.7).unwrap().silu().unwrap().add(&x).unwrap().exp().unwrap();
        assert_eq!(comp.run(&[x]).unwrap()[0].to_vec::<f32>().unwrap(), want.to_vec::<f32>().unwrap());
    }
}
