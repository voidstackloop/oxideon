//! Native WGSL & SPIR-V JIT Kernel Generator and Fused Operator Engine.
//!
//! Synthesizes WebGPU Shading Language (`WGSL`) compute shaders and CPU micro-loops
//! for arbitrarily fused elementwise and reduction computational subgraphs.

use crate::error::{Error, Result};
use crate::tensor::Tensor;

/// Primitive arithmetic and activation operations supported by the JIT builder.
#[derive(Debug, Clone, PartialEq)]
pub enum JitOp {
    Add,
    Sub,
    Mul,
    Div,
    Fma, // fma(a, b, c) = a * b + c
    Silu,
    Gelu,
    Relu,
    Exp,
    Neg,
    Clamp(f32, f32),
}

/// A node in the JIT compute graph.
#[derive(Debug, Clone)]
pub enum JitNode {
    Input { name: String, binding: usize },
    Constant(f32),
    Op { op: JitOp, args: Vec<usize> },
}

/// WGSL identifier check (also rejects WGSL reserved words that would parse
/// as keywords instead of binding names).
fn is_wgsl_ident(name: &str) -> bool {
    const KEYWORDS: &[&str] = &[
        "fn", "let", "var", "if", "else", "for", "while", "loop", "return", "break", "continue",
        "struct", "true", "false", "in", "out", "inout", "const", "override", "alias", "discard",
        "switch", "case", "default",
    ];
    let mut chars = name.chars();
    match chars.next() {
        Some(c) if c.is_ascii_alphabetic() || c == '_' => {}
        _ => return false,
    }
    if !chars.all(|c| c.is_ascii_alphanumeric() || c == '_') {
        return false;
    }
    !KEYWORDS.contains(&name)
}

/// Programmable JIT compute kernel builder.
#[derive(Debug, Clone, Default)]
pub struct JitKernelBuilder {
    pub name: String,
    pub nodes: Vec<JitNode>,
    pub outputs: Vec<usize>,
    pub input_count: usize,
}

impl JitKernelBuilder {
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            nodes: Vec::new(),
            outputs: Vec::new(),
            input_count: 0,
        }
    }

    /// Registers a tensor input to the kernel.
    pub fn add_input(&mut self, name: impl Into<String>) -> usize {
        let binding = self.input_count;
        self.input_count += 1;
        let id = self.nodes.len();
        self.nodes.push(JitNode::Input {
            name: name.into(),
            binding,
        });
        id
    }

    /// Registers a scalar constant.
    pub fn add_constant(&mut self, val: f32) -> usize {
        let id = self.nodes.len();
        self.nodes.push(JitNode::Constant(val));
        id
    }

    /// Appends an arithmetic or activation operation.
    pub fn add_op(&mut self, op: JitOp, args: &[usize]) -> usize {
        let id = self.nodes.len();
        self.nodes.push(JitNode::Op {
            op,
            args: args.to_vec(),
        });
        id
    }

    /// Marks a node ID as the primary output of the fused kernel.
    pub fn set_output(&mut self, id: usize) {
        self.outputs.push(id);
    }

    /// Validates the kernel graph: non-empty in-range outputs, per-op arity,
    /// in-range topologically-ordered args, valid unique input names, in-range
    /// bindings, and finite constants (NaN/inf have no WGSL literal).
    pub fn validate(&self) -> Result<()> {
        if self.outputs.is_empty() {
            return Err(Error::Msg("JitKernel has no outputs".into()));
        }
        for &o in &self.outputs {
            if o >= self.nodes.len() {
                return Err(Error::Msg(format!(
                    "JitKernel output id {o} out of range ({} nodes)",
                    self.nodes.len()
                )));
            }
        }
        let mut names = std::collections::HashSet::new();
        for (i, node) in self.nodes.iter().enumerate() {
            match node {
                JitNode::Input { name, binding } => {
                    if !is_wgsl_ident(name) {
                        return Err(Error::Msg(format!(
                            "JitKernel input name {name:?} is not a valid WGSL identifier"
                        )));
                    }
                    if !names.insert(name.clone()) {
                        return Err(Error::Msg(format!(
                            "JitKernel duplicate input name {name:?}"
                        )));
                    }
                    if *binding >= self.input_count {
                        return Err(Error::Msg(format!(
                            "JitKernel input {name:?} binding {binding} out of range ({} inputs)",
                            self.input_count
                        )));
                    }
                }
                JitNode::Constant(v) => {
                    if !v.is_finite() {
                        return Err(Error::Msg(
                            "JitKernel constant must be finite for WGSL emission".into(),
                        ));
                    }
                }
                JitNode::Op { op, args } => {
                    let arity = match op {
                        JitOp::Add | JitOp::Sub | JitOp::Mul | JitOp::Div => 2,
                        JitOp::Fma => 3,
                        JitOp::Silu
                        | JitOp::Gelu
                        | JitOp::Relu
                        | JitOp::Exp
                        | JitOp::Neg
                        | JitOp::Clamp(..) => 1,
                    };
                    if args.len() != arity {
                        return Err(Error::Msg(format!(
                            "JitKernel node {i} ({op:?}) needs {arity} args, got {}",
                            args.len()
                        )));
                    }
                    for &a in args {
                        if a >= i {
                            return Err(Error::Msg(format!(
                                "JitKernel node {i} references out-of-order arg {a}"
                            )));
                        }
                    }
                    if let JitOp::Clamp(lo, hi) = op
                        && (!lo.is_finite() || !hi.is_finite() || lo > hi)
                    {
                        return Err(Error::Msg(format!(
                            "JitKernel node {i} has an invalid clamp range [{lo}, {hi}]"
                        )));
                    }
                }
            }
        }
        Ok(())
    }

    /// Synthesizes complete, valid WebGPU Shading Language (`WGSL`) compute shader code.
    pub fn build_wgsl(&self, workgroup_size: u32) -> Result<String> {
        if workgroup_size == 0 {
            return Err(Error::Msg(
                "build_wgsl: workgroup_size must be at least 1".into(),
            ));
        }
        self.validate()?;
        let mut code = String::new();
        code.push_str("// Auto-generated by Oxideon JIT Kernel Compiler\n\n");

        // Helper functions
        code.push_str("fn silu(x: f32) -> f32 {\n    return x / (1.0 + exp(-x));\n}\n\n");
        code.push_str("fn gelu(x: f32) -> f32 {\n    return 0.5 * x * (1.0 + tanh(0.79788456 * (x + 0.044715 * x * x * x)));\n}\n\n");

        // Buffer bindings for inputs
        let mut binding_idx = 0;
        for node in &self.nodes {
            if let JitNode::Input { name, .. } = node {
                code.push_str(&format!(
                    "@group(0) @binding({binding_idx}) var<storage, read> {name}: array<f32>;\n"
                ));
                binding_idx += 1;
            }
        }

        // Buffer bindings for outputs
        for (out_idx, _) in self.outputs.iter().enumerate() {
            code.push_str(&format!(
                "@group(0) @binding({binding_idx}) var<storage, read_write> out_{out_idx}: array<f32>;\n"
            ));
            binding_idx += 1;
        }

        code.push_str(&format!("\n@compute @workgroup_size({workgroup_size})\n"));
        code.push_str("fn main(@builtin(global_invocation_id) global_id: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {\n");
        code.push_str(&format!("    let idx = global_id.y * (nwg.x * {workgroup_size}u) + global_id.x;\n"));
        code.push_str("    if (idx >= arrayLength(&out_0)) { return; }\n\n");

        // Instruction evaluation
        for (i, node) in self.nodes.iter().enumerate() {
            match node {
                JitNode::Input { name, .. } => {
                    code.push_str(&format!("    let r_{i} = {name}[idx];\n"));
                }
                JitNode::Constant(v) => {
                    code.push_str(&format!("    let r_{i} = {v:.6}f;\n"));
                }
                JitNode::Op { op, args } => match op {
                    JitOp::Add => {
                        code.push_str(&format!("    let r_{i} = r_{} + r_{};\n", args[0], args[1]));
                    }
                    JitOp::Sub => {
                        code.push_str(&format!("    let r_{i} = r_{} - r_{};\n", args[0], args[1]));
                    }
                    JitOp::Mul => {
                        code.push_str(&format!("    let r_{i} = r_{} * r_{};\n", args[0], args[1]));
                    }
                    JitOp::Div => {
                        code.push_str(&format!("    let r_{i} = r_{} / r_{};\n", args[0], args[1]));
                    }
                    JitOp::Fma => {
                        code.push_str(&format!(
                            "    let r_{i} = fma(r_{}, r_{}, r_{});\n",
                            args[0], args[1], args[2]
                        ));
                    }
                    JitOp::Silu => {
                        code.push_str(&format!("    let r_{i} = silu(r_{});\n", args[0]));
                    }
                    JitOp::Gelu => {
                        code.push_str(&format!("    let r_{i} = gelu(r_{});\n", args[0]));
                    }
                    JitOp::Relu => {
                        code.push_str(&format!("    let r_{i} = max(r_{}, 0.0);\n", args[0]));
                    }
                    JitOp::Exp => {
                        code.push_str(&format!("    let r_{i} = exp(r_{});\n", args[0]));
                    }
                    JitOp::Neg => {
                        code.push_str(&format!("    let r_{i} = -r_{};\n", args[0]));
                    }
                    JitOp::Clamp(min, max) => {
                        code.push_str(&format!(
                            "    let r_{i} = clamp(r_{}, {min:.6}f, {max:.6}f);\n",
                            args[0]
                        ));
                    }
                },
            }
        }

        // Write outputs
        for (out_idx, &node_id) in self.outputs.iter().enumerate() {
            code.push_str(&format!("    out_{out_idx}[idx] = r_{node_id};\n"));
        }

        code.push_str("}\n");
        Ok(code)
    }

    /// Evaluates the fused kernel on GPU (`Wgpu`) when inputs are `F32` on a WGPU device,
    /// or falls back to vectorized CPU execution.
    pub fn execute(&self, inputs: &[&Tensor]) -> Result<Tensor> {
        if self.outputs.len() != 1 {
            return Err(Error::Msg(format!(
                "JitKernel::execute supports exactly one output, kernel has {}; use execute_all",
                self.outputs.len()
            )));
        }
        let mut all = self.execute_all(inputs)?;
        all.pop()
            .ok_or_else(|| Error::Msg("JitKernel::execute: kernel has no outputs".into()))
    }

    /// Evaluates the fused kernel on GPU (`Wgpu`) when inputs are `F32` on a WGPU device,
    /// returning one tensor per registered output.
    pub fn execute_all(&self, inputs: &[&Tensor]) -> Result<Vec<Tensor>> {
        self.validate()?;
        if inputs.len() != self.input_count {
            return Err(Error::Msg(format!(
                "JitKernel expected {} inputs, got {}",
                self.input_count,
                inputs.len()
            )));
        }
        #[cfg(feature = "wgpu")]
        if let Some(first) = inputs.first()
            && let crate::device::Device::Wgpu(wgpu_dev) = first.device()
        {
            let shape = first.shape();
            let numel = first.numel();
            let all_compat = inputs.iter().all(|t| {
                t.device() == first.device()
                    && t.dtype() == crate::dtype::DType::F32
                    && t.shape() == shape
            });
            if all_compat && numel > 0 && wgpu_dev.fits_grid(numel, 256) {
                use crate::backend::{Storage, StorageInner, WgpuStorage};
                use crate::layout::Layout;
                let wgsl = self.build_wgsl(256)?;
                let mut contig_storages: Vec<WgpuStorage> = Vec::with_capacity(inputs.len());
                for t in inputs {
                    let guard = t.storage().read();
                    let StorageInner::Wgpu(ws) = &*guard else {
                        unreachable!();
                    };
                    contig_storages.push(ws.to_contiguous(t.layout())?);
                }
                wgpu_dev.check_bytes("JIT GPU tensor", (numel as u64) * 4)?;
                let out_bufs: Vec<wgpu::Buffer> = (0..self.outputs.len())
                    .map(|_| wgpu_dev.out_buffer(numel))
                    .collect();
                let mut buf_refs: Vec<&wgpu::Buffer> =
                    Vec::with_capacity(contig_storages.len() + out_bufs.len());
                for cs in &contig_storages {
                    buf_refs.push(cs.buffer());
                }
                for ob in &out_bufs {
                    buf_refs.push(ob);
                }
                wgpu_dev.dispatch_wgsl(&wgsl, &buf_refs, None, wgpu_dev.grid_1d(numel, 256))?;
                return Ok(out_bufs
                    .into_iter()
                    .map(|ob| {
                        let ws = WgpuStorage::from_buffer(ob, numel, &wgpu_dev);
                        Tensor::from_storage(Storage::new_wgpu(ws), Layout::contiguous(shape))
                    })
                    .collect());
            }
        }
        let cpu_outs = self.execute_cpu_all(inputs)?;
        if let Some(first) = inputs.first() {
            let dev = first.device();
            if !dev.is_cpu() {
                return cpu_outs.into_iter().map(|t| t.to_device(&dev)).collect();
            }
        }
        Ok(cpu_outs)
    }

    /// Evaluates the fused kernel graph in pure vectorized CPU host execution.
    ///
    /// Single-output kernels only; use [`JitKernelBuilder::execute_cpu_all`]
    /// for multi-output kernels (silently dropping outputs would diverge
    /// from [`JitKernelBuilder::build_wgsl`], which emits every output).
    pub fn execute_cpu(&self, inputs: &[&Tensor]) -> Result<Tensor> {
        if self.outputs.len() != 1 {
            return Err(Error::Msg(format!(
                "JitKernel::execute_cpu supports exactly one output, kernel has {}; use execute_cpu_all",
                self.outputs.len()
            )));
        }
        let mut all = self.execute_cpu_all(inputs)?;
        all.pop()
            .ok_or_else(|| Error::Msg("JitKernel::execute_cpu: kernel has no outputs".into()))
    }

    /// Evaluates the fused kernel graph on CPU, returning one tensor per
    /// registered output (same order as [`JitKernelBuilder::build_wgsl`]).
    pub fn execute_cpu_all(&self, inputs: &[&Tensor]) -> Result<Vec<Tensor>> {
        self.validate()?;
        if inputs.len() != self.input_count {
            return Err(Error::Msg(format!(
                "JitKernel expected {} inputs, got {}",
                self.input_count,
                inputs.len()
            )));
        }
        let Some(first) = inputs.first() else {
            return Err(Error::Msg("JitKernel has no inputs to execute".into()));
        };

        let total_len = first.numel();
        let shape = first.shape().to_vec();

        // Convert inputs to raw flat f32 vectors
        let in_data: Vec<Vec<f32>> = inputs
            .iter()
            .map(|t| t.to_vec::<f32>())
            .collect::<Result<Vec<_>>>()?;
        for (k, data) in in_data.iter().enumerate() {
            if data.len() != total_len {
                return Err(Error::Msg(format!(
                    "JitKernel input {k} has {} elements, expected {total_len}",
                    data.len()
                )));
            }
        }

        let mut out_data = vec![vec![0.0f32; total_len]; self.outputs.len()];
        let mut registers = vec![0.0f32; self.nodes.len()];

        for i in 0..total_len {
            for (node_idx, node) in self.nodes.iter().enumerate() {
                registers[node_idx] = match node {
                    JitNode::Input { binding, .. } => in_data[*binding][i],
                    JitNode::Constant(val) => *val,
                    JitNode::Op { op, args } => match op {
                        JitOp::Add => registers[args[0]] + registers[args[1]],
                        JitOp::Sub => registers[args[0]] - registers[args[1]],
                        JitOp::Mul => registers[args[0]] * registers[args[1]],
                        JitOp::Div => registers[args[0]] / registers[args[1]],
                        JitOp::Fma => registers[args[0]] * registers[args[1]] + registers[args[2]],
                        JitOp::Silu => {
                            let x = registers[args[0]];
                            x / (1.0 + (-x).exp())
                        }
                        JitOp::Gelu => {
                            let x = registers[args[0]];
                            0.5 * x * (1.0 + (0.797_884_6 * (x + 0.044715 * x * x * x)).tanh())
                        }
                        JitOp::Relu => { let x = registers[args[0]]; if x < 0.0 { 0.0 } else { x } } // NaN passes through
                        JitOp::Exp => registers[args[0]].exp(),
                        JitOp::Neg => -registers[args[0]],
                        JitOp::Clamp(min, max) => registers[args[0]].clamp(*min, *max),
                    },
                };
            }

            for (k, &out_id) in self.outputs.iter().enumerate() {
                out_data[k][i] = registers[out_id];
            }
        }

        out_data
            .into_iter()
            .map(|d| {
                Tensor::from_vec(d, &shape)?
                    .to_device(&inputs[0].device())?
                    .to_dtype(inputs[0].dtype())
            })
            .collect()
    }

    /// Pre-assembled fused kernel for residual addition followed by SiLU activation:
    /// $$y = \operatorname{silu}(x + \text{residual})$$
    pub fn fused_add_silu() -> Self {
        let mut b = Self::new("fused_add_silu");
        let in_x = b.add_input("x");
        let in_res = b.add_input("res");
        let added = b.add_op(JitOp::Add, &[in_x, in_res]);
        let act = b.add_op(JitOp::Silu, &[added]);
        b.set_output(act);
        b
    }

    /// Pre-assembled fused kernel for SwiGLU gated activation:
    /// $$y = \operatorname{silu}(\text{gate}) \cdot \text{val}$$
    pub fn fused_swiglu_gate() -> Self {
        let mut b = Self::new("fused_swiglu_gate");
        let in_gate = b.add_input("gate");
        let in_val = b.add_input("val");
        let act = b.add_op(JitOp::Silu, &[in_gate]);
        let mult = b.add_op(JitOp::Mul, &[act, in_val]);
        b.set_output(mult);
        b
    }

    /// Synthesizes an NVIDIA PTX kernel for this fused elementwise subgraph.
    pub fn build_ptx(&self, kernel_name: &str) -> Result<String> {
        if self.outputs.is_empty() {
            return Err(Error::Msg("JitKernelBuilder::build_ptx: kernel has no outputs".into()));
        }
        if !is_wgsl_ident(kernel_name) {
            return Err(Error::Msg(format!("JitKernelBuilder::build_ptx: invalid kernel name '{kernel_name}'")));
        }

        let mut ptx = String::new();
        ptx.push_str(".version 7.0\n.target sm_50\n.address_size 64\n\n");
        ptx.push_str(&format!(".visible .entry {kernel_name}(\n"));
        for i in 0..self.input_count {
            ptx.push_str(&format!("    .param .u64 in_{i},\n"));
        }
        for k in 0..self.outputs.len() {
            ptx.push_str(&format!("    .param .u64 out_{k},\n"));
        }
        ptx.push_str("    .param .u32 n\n)\n{\n");

        let n_nodes = self.nodes.len().max(1);
        let n_regs = n_nodes + 12;
        ptx.push_str("    .reg .pred %p<2>;\n");
        ptx.push_str("    .reg .b32 %r<6>;\n");
        ptx.push_str(&format!("    .reg .f32 %f<{n_regs}>;\n"));
        ptx.push_str("    .reg .b64 %rd<4>;\n\n");

        ptx.push_str("    ld.param.u32 %r1, [n];\n");
        ptx.push_str("    mov.u32 %r2, %ctaid.x;\n");
        ptx.push_str("    mov.u32 %r3, %ntid.x;\n");
        ptx.push_str("    mov.u32 %r4, %tid.x;\n");
        ptx.push_str("    mad.lo.s32 %r5, %r2, %r3, %r4;\n");
        ptx.push_str("    setp.ge.u32 %p1, %r5, %r1;\n");
        ptx.push_str("    @%p1 bra DONE;\n\n");
        ptx.push_str("    mul.wide.u32 %rd1, %r5, 4;\n");

        let t0 = n_nodes;
        let t1 = n_nodes + 1;
        let t2 = n_nodes + 2;
        let t3 = n_nodes + 3;
        let t4 = n_nodes + 4;
        let t5 = n_nodes + 5;
        let hex = |v: f32| format!("0f{:08X}", v.to_bits());

        for (id, node) in self.nodes.iter().enumerate() {
            match node {
                JitNode::Input { binding, .. } => {
                    ptx.push_str(&format!("    ld.param.u64 %rd2, [in_{binding}];\n"));
                    ptx.push_str("    add.s64 %rd3, %rd2, %rd1;\n");
                    ptx.push_str(&format!("    ld.global.f32 %f{id}, [%rd3];\n"));
                }
                JitNode::Constant(v) => {
                    ptx.push_str(&format!("    mov.f32 %f{id}, {};\n", hex(*v)));
                }
                JitNode::Op { op, args } => match op {
                    JitOp::Add => ptx.push_str(&format!("    add.f32 %f{id}, %f{}, %f{};\n", args[0], args[1])),
                    JitOp::Sub => ptx.push_str(&format!("    sub.f32 %f{id}, %f{}, %f{};\n", args[0], args[1])),
                    JitOp::Mul => ptx.push_str(&format!("    mul.f32 %f{id}, %f{}, %f{};\n", args[0], args[1])),
                    JitOp::Div => ptx.push_str(&format!("    div.rn.f32 %f{id}, %f{}, %f{};\n", args[0], args[1])),
                    JitOp::Fma => ptx.push_str(&format!("    fma.rn.f32 %f{id}, %f{}, %f{}, %f{};\n", args[0], args[1], args[2])),
                    JitOp::Neg => ptx.push_str(&format!("    neg.f32 %f{id}, %f{};\n", args[0])),
                    JitOp::Relu => {
                        ptx.push_str(&format!("    mov.f32 %f{t0}, 0f00000000;\n"));
                        ptx.push_str(&format!("    max.f32 %f{id}, %f{}, %f{t0};\n", args[0]));
                    }
                    JitOp::Clamp(lo, hi) => {
                        ptx.push_str(&format!("    mov.f32 %f{t0}, {};\n", hex(*lo)));
                        ptx.push_str(&format!("    mov.f32 %f{t1}, {};\n", hex(*hi)));
                        ptx.push_str(&format!("    max.f32 %f{t2}, %f{}, %f{t0};\n", args[0]));
                        ptx.push_str(&format!("    min.f32 %f{id}, %f{t2}, %f{t1};\n"));
                    }
                    JitOp::Exp => {
                        ptx.push_str(&format!("    mov.f32 %f{t0}, {};\n", hex(std::f32::consts::LOG2_E)));
                        ptx.push_str(&format!("    mul.f32 %f{t1}, %f{}, %f{t0};\n", args[0]));
                        ptx.push_str(&format!("    ex2.approx.f32 %f{id}, %f{t1};\n"));
                    }
                    JitOp::Silu => {
                        let a = args[0];
                        ptx.push_str(&format!("    neg.f32 %f{t0}, %f{a};\n"));
                        ptx.push_str(&format!("    mov.f32 %f{t1}, {};\n", hex(std::f32::consts::LOG2_E)));
                        ptx.push_str(&format!("    mul.f32 %f{t2}, %f{t0}, %f{t1};\n"));
                        ptx.push_str(&format!("    ex2.approx.f32 %f{t3}, %f{t2};\n"));
                        ptx.push_str(&format!("    mov.f32 %f{t4}, {};\n", hex(1.0)));
                        ptx.push_str(&format!("    add.f32 %f{t5}, %f{t3}, %f{t4};\n"));
                        ptx.push_str(&format!("    div.rn.f32 %f{id}, %f{a}, %f{t5};\n"));
                    }
                    JitOp::Gelu => {
                        // 0.5 * x * (1 + tanh(u)), tanh(u) = (exp(2u) - 1) / (exp(2u) + 1)
                        let a = args[0];
                        ptx.push_str(&format!("    mul.f32 %f{t0}, %f{a}, %f{a};\n"));
                        ptx.push_str(&format!("    mul.f32 %f{t0}, %f{t0}, %f{a};\n"));
                        ptx.push_str(&format!("    mov.f32 %f{t1}, {};\n", hex(0.044715)));
                        ptx.push_str(&format!("    fma.rn.f32 %f{t0}, %f{t1}, %f{t0}, %f{a};\n"));
                        ptx.push_str(&format!("    mov.f32 %f{t1}, {};\n", hex(2.0 * 0.797_884_6 * std::f32::consts::LOG2_E)));
                        ptx.push_str(&format!("    mul.f32 %f{t2}, %f{t0}, %f{t1};\n"));
                        ptx.push_str(&format!("    ex2.approx.f32 %f{t3}, %f{t2};\n"));
                        ptx.push_str(&format!("    mov.f32 %f{t4}, {};\n", hex(1.0)));
                        ptx.push_str(&format!("    add.f32 %f{t5}, %f{t3}, %f{t4};\n"));
                        ptx.push_str(&format!("    div.rn.f32 %f{t3}, %f{t3}, %f{t5};\n"));
                        ptx.push_str(&format!("    mul.f32 %f{id}, %f{a}, %f{t3};\n"));
                    }
                },
            }
        }

        for (k, &out_id) in self.outputs.iter().enumerate() {
            ptx.push_str(&format!("    ld.param.u64 %rd2, [out_{k}];\n"));
            ptx.push_str("    add.s64 %rd3, %rd2, %rd1;\n");
            ptx.push_str(&format!("    st.global.f32 [%rd3], %f{out_id};\n"));
        }
        ptx.push_str("DONE:\n    ret;\n}\n");
        Ok(ptx)
    }

    /// Executes this fused JIT kernel directly on an NVIDIA CUDA GPU via `nvcuda` PTX loading.
    pub fn execute_cuda(&self, device_ordinal: usize, inputs: &[&[f32]]) -> Result<Vec<Vec<f32>>> {
        use crate::backend::cuda_driver::{CudaBuffer, CudaPtxModule, cuda_synchronize};
        if inputs.len() != self.input_count {
            return Err(Error::Msg(format!(
                "JitKernel::execute_cuda expects {} inputs, got {}",
                self.input_count,
                inputs.len()
            )));
        }
        let n = inputs.first().map(|s| s.len()).unwrap_or(0);
        for (i, inp) in inputs.iter().enumerate() {
            if inp.len() != n {
                return Err(Error::Msg(format!(
                    "JitKernel::execute_cuda input {i} length {} != {n}",
                    inp.len()
                )));
            }
        }
        if n == 0 {
            return Ok(vec![Vec::new(); self.outputs.len()]);
        }
        let kname = if is_wgsl_ident(&self.name) {
            self.name.clone()
        } else {
            "oxideon_jit_kernel".to_string()
        };
        let ptx = self.build_ptx(&kname)?;
        let module = CudaPtxModule::load(device_ordinal, &ptx)?;

        let in_bufs: Vec<CudaBuffer> = inputs
            .iter()
            .map(|s| CudaBuffer::from_host_f32(device_ordinal, s))
            .collect::<Result<Vec<_>>>()?;
        let out_bytes = n * std::mem::size_of::<f32>();
        let out_bufs: Vec<CudaBuffer> = (0..self.outputs.len())
            .map(|_| CudaBuffer::alloc(device_ordinal, out_bytes))
            .collect::<Result<Vec<_>>>()?;

        let mut ptrs: Vec<u64> = in_bufs
            .iter()
            .chain(out_bufs.iter())
            .map(CudaBuffer::device_ptr)
            .collect();
        let mut n_u32 = n as u32;
        let mut params: Vec<*mut std::ffi::c_void> = ptrs
            .iter_mut()
            .map(|p| p as *mut u64 as *mut std::ffi::c_void)
            .collect();
        params.push(&mut n_u32 as *mut u32 as *mut std::ffi::c_void);

        module.launch_1d(&kname, n, &mut params)?;
        cuda_synchronize(device_ordinal)?;

        out_bufs
            .iter()
            .map(|b| b.to_host_f32(n))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn jit_fused_kernels_cpu_and_wgpu() {
        let x = Tensor::from_slice(&[1.0f32, -2.0, 0.5, 3.0], &[2, 2]).unwrap();
        let res = Tensor::from_slice(&[0.5f32, 1.0, -0.5, -1.0], &[2, 2]).unwrap();
        let k = JitKernelBuilder::fused_add_silu();
        let cpu_out = k.execute(&[&x, &res]).unwrap().to_vec::<f32>().unwrap();

        #[cfg(feature = "wgpu")]
        if let Ok(wgpu_dev) = crate::backend::WgpuDevice::new() {
            let dev = crate::device::Device::Wgpu(wgpu_dev.clone());
            let gx = x.to_device(&dev).unwrap();
            let gr = res.to_device(&dev).unwrap();
            let r0 = wgpu_dev.readbacks();
            let gy = k.execute(&[&gx, &gr]).unwrap();
            assert_eq!(wgpu_dev.readbacks(), r0, "JIT WGSL dispatch must not read back");
            let gpu_out = gy.to_vec::<f32>().unwrap();
            for (a, b) in cpu_out.iter().zip(gpu_out.iter()) {
                assert!((a - b).abs() < 1e-5, "{a} vs {b}");
            }
        }
        let _ = cpu_out;
    }

    #[test]
    fn jit_fused_ptx_executes_on_cuda() {
        let k = JitKernelBuilder::fused_add_silu();
        let ptx = k.build_ptx("fused_add_silu").unwrap();
        assert!(ptx.contains(".visible .entry fused_add_silu("));
        assert!(ptx.contains("ex2.approx.f32"));

        if crate::backend::cuda_driver::cuda_available() {
            let x = [1.0f32, -2.0, 0.5, 3.0];
            let r = [0.5f32, 1.0, -0.5, -1.0];
            let gpu_out = k.execute_cuda(0, &[&x, &r]).unwrap();
            let cpu_out = k
                .execute(&[
                    &Tensor::from_slice(&x, &[4]).unwrap(),
                    &Tensor::from_slice(&r, &[4]).unwrap(),
                ])
                .unwrap()
                .to_vec::<f32>()
                .unwrap();
            for (a, b) in cpu_out.iter().zip(gpu_out[0].iter()) {
                assert!((a - b).abs() < 1e-4, "cpu={a} vs cuda_ptx={b}");
            }
        }
    }
}
