//! Embedded Tile-IR and Dynamic Kernel Fusion JIT Emitter.
//!
//! Provides hardware-independent tiled intermediate representation (Tile-IR) and an
//! automated kernel fusion pass that merges elementwise operations, normalization,
//! activations, attention, RoPE, quantized GEMV, and matrix multiplications into a
//! single high-throughput GPU kernel across PTX, Metal MSL, and WebGPU WGSL.

use crate::driver::{MslKernelEmitter, PtxKernelEmitter, WgslKernelBuilder};

fn is_ident(s: &str) -> bool {
    let mut chars = s.chars();
    matches!(chars.next(), Some(c) if c.is_ascii_alphabetic() || c == '_')
        && chars.all(|c| c.is_ascii_alphanumeric() || c == '_')
}

fn is_sm_arch(target_sm: &str) -> bool {
    target_sm.len() > 3
        && target_sm.starts_with("sm_")
        && target_sm[3..].chars().all(|c| c.is_ascii_digit())
}

/// Compute ops with memory-movement markers stripped (template matching).
fn compute_ops(ops: &[TileOp]) -> Vec<&TileOp> {
    ops.iter()
        .filter(|op| !matches!(op, TileOp::Load { .. } | TileOp::Store { .. }))
        .collect()
}

fn op_name(op: &TileOp) -> String {
    match op {
        TileOp::Load { name, .. } => format!("Load({})", clean(name)),
        TileOp::Store { name, .. } => format!("Store({})", clean(name)),
        TileOp::Add => "Add".into(),
        TileOp::Sub => "Sub".into(),
        TileOp::Mul => "Mul".into(),
        TileOp::Div => "Div".into(),
        TileOp::SiLU => "SiLU".into(),
        TileOp::GELU => "GELU".into(),
        TileOp::ReLU => "ReLU".into(),
        TileOp::RMSNorm { eps } => format!("RMSNorm({eps})"),
        TileOp::Gemm { m, n, k } => format!("Gemm({m},{n},{k})"),
        TileOp::FlashAttention {
            head_dim,
            is_causal,
        } => format!("FlashAttention({head_dim},{is_causal})"),
        TileOp::RoPE { head_dim } => format!("RoPE({head_dim})"),
        TileOp::QuantizedGemvQ4 { group_size } => format!("QuantizedGemvQ4({group_size})"),
        TileOp::Accumulate => "Accumulate".into(),
    }
}

/// Strips everything but comment/string-safe characters so plan-controlled text
/// can never break out of an emitted comment or `#error` string.
fn clean(s: &str) -> String {
    s.chars()
        .filter(|c| {
            c.is_ascii_alphanumeric()
                || matches!(
                    c,
                    '_' | '.' | '-' | '[' | ']' | '(' | ')' | ',' | ' ' | ':' | '=' | '/'
                )
        })
        .collect()
}

/// Simulates the elementwise value stack: `false` when the op sequence would
/// underflow it (checked before emission so emitters never panic).
fn stack_ok(num_inputs: usize, ops: &[&TileOp]) -> bool {
    let mut depth = num_inputs;
    for op in ops {
        match op {
            TileOp::Add | TileOp::Sub | TileOp::Mul | TileOp::Div => {
                if depth < 2 {
                    return false;
                }
                depth -= 1;
            }
            TileOp::ReLU | TileOp::SiLU | TileOp::GELU => {
                if depth < 1 {
                    return false;
                }
            }
            _ => return false,
        }
    }
    !ops.is_empty() && depth >= 1
}

/// Tiled memory scope in hardware hierarchy.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MemoryScope {
    Global,
    SharedThreadgroup,
    RegisterTile,
}

/// Elementwise or reduction mathematical operation inside a tile.
#[derive(Debug, Clone, PartialEq)]
pub enum TileOp {
    Load { scope: MemoryScope, name: String },
    Store { scope: MemoryScope, name: String },
    Add,
    Sub,
    Mul,
    Div,
    SiLU,
    GELU,
    ReLU,
    RMSNorm { eps: f64 },
    Gemm { m: usize, n: usize, k: usize },
    FlashAttention { head_dim: usize, is_causal: bool },
    RoPE { head_dim: usize },
    QuantizedGemvQ4 { group_size: usize },
    Accumulate,
}

/// Sequence of operations within a fused kernel stage.
#[derive(Debug, Clone)]
pub struct TileKernelPlan {
    pub name: String,
    pub tile_m: usize,
    pub tile_n: usize,
    pub tile_k: usize,
    pub ops: Vec<TileOp>,
    pub inputs: Vec<String>,
    pub output: String,
}

impl TileKernelPlan {
    pub fn new(name: impl Into<String>, tile_m: usize, tile_n: usize, tile_k: usize) -> Self {
        Self {
            name: name.into(),
            tile_m,
            tile_n,
            tile_k,
            ops: Vec::new(),
            inputs: Vec::new(),
            output: "out".into(),
        }
    }

    pub fn add_input(&mut self, input_name: impl Into<String>) -> &mut Self {
        self.inputs.push(input_name.into());
        self
    }

    pub fn set_output(&mut self, output_name: impl Into<String>) -> &mut Self {
        self.output = output_name.into();
        self
    }

    pub fn push_op(&mut self, op: TileOp) -> &mut Self {
        self.ops.push(op);
        self
    }

    fn plan_header(&self, lang: &str) -> String {
        let ops = self.ops.iter().map(op_name).collect::<Vec<_>>().join(", ");
        let inputs = self
            .inputs
            .iter()
            .map(|s| clean(s))
            .collect::<Vec<_>>()
            .join(", ");
        format!(
            "// Oxideon Tile-IR Fused Kernel Plan ({lang}): {}\n// tiles=({},{},{}) inputs=[{}] output={} ops=[{}]\n",
            clean(&self.name),
            self.tile_m,
            self.tile_n,
            self.tile_k,
            inputs,
            clean(&self.output),
            if ops.is_empty() { "(none)".into() } else { ops },
        )
    }

    fn plan_error(&self) -> Option<String> {
        if !is_ident(&self.name) {
            return Some(format!("invalid kernel name {:?}", self.name));
        }
        if !is_ident(&self.output) {
            return Some(format!("invalid output name {:?}", self.output));
        }
        if self.inputs.is_empty() {
            return Some("plan declares no inputs".into());
        }
        for input in &self.inputs {
            if !is_ident(input) {
                return Some(format!("invalid input name {input:?}"));
            }
        }
        if self.tile_m == 0 || self.tile_n == 0 || self.tile_k == 0 {
            return Some("tile dims must be positive".into());
        }
        if compute_ops(&self.ops).is_empty() {
            return Some("plan declares no compute ops".into());
        }
        None
    }

    /// Emits complete NVIDIA PTX assembly for this fused kernel plan.
    pub fn emit_ptx(&self, target_sm: &str) -> String {
        let mut ptx = self.plan_header("PTX");
        if !is_sm_arch(target_sm) {
            return ptx
                + &Self::ptx_trap(
                    "oxideon_invalid_plan",
                    &format!("bad SM arch {target_sm:?}"),
                );
        }
        if let Some(err) = self.plan_error() {
            return ptx + &Self::ptx_trap("oxideon_invalid_plan", &err);
        }
        let emitter = PtxKernelEmitter::new(&self.name, target_sm);
        let compute = compute_ops(&self.ops);
        let is_ptx_elementwise = compute.iter().all(|op| {
            matches!(
                op,
                TileOp::Add | TileOp::Sub | TileOp::Mul | TileOp::Div | TileOp::ReLU | TileOp::SiLU
            )
        });
        let body = match compute.as_slice() {
            [TileOp::Add, TileOp::SiLU] => emitter.emit_fused_add_silu_ptx(),
            [TileOp::Gemm { .. }] => emitter.emit_gemm_fma_ptx(),
            [TileOp::RMSNorm { eps }] | [TileOp::Add, TileOp::RMSNorm { eps }, TileOp::SiLU] => {
                emitter.emit_warp_rmsnorm_ptx(*eps as f32)
            }
            [TileOp::FlashAttention { head_dim, .. }] => {
                emitter.emit_flash_attention_mma_ptx(*head_dim)
            }
            [TileOp::RoPE { .. }] => emitter.emit_rope_ptx(),
            [TileOp::QuantizedGemvQ4 { group_size }] => {
                emitter.emit_int4_dequant_gemv_ptx(*group_size)
            }
            _ if is_ptx_elementwise && stack_ok(self.inputs.len(), &compute) => {
                Self::ptx_elementwise(&self.name, target_sm, &self.inputs, &compute)
            }
            _ => {
                let seq = compute
                    .iter()
                    .map(|op| op_name(op))
                    .collect::<Vec<_>>()
                    .join(", ");
                Self::ptx_trap(
                    &self.name,
                    &format!("no PTX lowering for op sequence [{seq}]"),
                )
            }
        };
        ptx.push_str(&body);
        ptx
    }

    /// Emits Apple Metal Shading Language (`MSL`) for this fused kernel plan.
    pub fn emit_msl(&self) -> String {
        let mut msl = self.plan_header("MSL");
        if let Some(err) = self.plan_error() {
            return msl + &Self::msl_error(&err);
        }
        let emitter = MslKernelEmitter::new(&self.name);
        let compute = compute_ops(&self.ops);
        let is_msl_elementwise = compute.iter().all(|op| {
            matches!(
                op,
                TileOp::Add
                    | TileOp::Sub
                    | TileOp::Mul
                    | TileOp::Div
                    | TileOp::ReLU
                    | TileOp::SiLU
                    | TileOp::GELU
            )
        });
        let body = match compute.as_slice() {
            [TileOp::Add, TileOp::SiLU] => emitter.emit_fused_add_silu_msl(),
            [TileOp::Gemm { .. }] => emitter.emit_matmul_tiled_msl(),
            [TileOp::RMSNorm { eps }] | [TileOp::Add, TileOp::RMSNorm { eps }, TileOp::SiLU] => {
                emitter.emit_fused_rmsnorm_msl(self.tile_n.max(32), *eps as f32)
            }
            [TileOp::FlashAttention {
                head_dim,
                is_causal,
            }] => emitter.emit_flash_attention_msl(
                *head_dim,
                self.tile_m.max(16),
                self.tile_n.max(16),
                *is_causal,
            ),
            [TileOp::RoPE { head_dim }] => emitter.emit_rope_msl(*head_dim, 10000.0),
            [TileOp::QuantizedGemvQ4 { group_size }] => {
                emitter.emit_quantized_affine_q4_matvec_msl(*group_size)
            }
            _ if is_msl_elementwise && stack_ok(self.inputs.len(), &compute) => {
                Self::msl_elementwise(&self.name, &self.inputs, &self.output, &compute)
            }
            _ => {
                let seq = compute
                    .iter()
                    .map(|op| op_name(op))
                    .collect::<Vec<_>>()
                    .join(", ");
                Self::msl_error(&format!("no MSL lowering for op sequence [{seq}]"))
            }
        };
        msl.push_str(&body);
        msl
    }

    /// Emits WebGPU Shading Language (`WGSL`) for this fused kernel plan.
    pub fn emit_wgsl(&self) -> String {
        let mut wgsl = self.plan_header("WGSL");
        if let Some(err) = self.plan_error() {
            return wgsl + &format!("// ERROR: {}\n", clean(&err));
        }
        let builder = WgslKernelBuilder::new(&self.name, 64);
        let compute = compute_ops(&self.ops);
        let body = match compute.as_slice() {
            [TileOp::Add, TileOp::SiLU] | [TileOp::SiLU, TileOp::Mul] => {
                builder.emit_fused_swiglu_wgsl()
            }
            [TileOp::Gemm { .. }] => builder.emit_tiled_matmul_wgsl(self.tile_m.max(16)),
            [TileOp::RMSNorm { eps }] | [TileOp::Add, TileOp::RMSNorm { eps }, TileOp::SiLU] => {
                builder.emit_fused_rmsnorm_wgsl(self.tile_n.max(32), *eps as f32)
            }
            [TileOp::FlashAttention {
                head_dim,
                is_causal,
            }] => builder.emit_flash_attention_wgsl(*head_dim, self.tile_m.max(16), *is_causal),
            [TileOp::RoPE { head_dim }] => builder.emit_rope_wgsl(*head_dim),
            [TileOp::QuantizedGemvQ4 { group_size }] => builder.emit_q4_gemv_wgsl(*group_size),
            _ => {
                let seq = compute
                    .iter()
                    .map(|op| op_name(op))
                    .collect::<Vec<_>>()
                    .join(", ");
                format!("// ERROR: no WGSL lowering for op sequence [{seq}]\n")
            }
        };
        wgsl.push_str(&body);
        wgsl
    }

    /// PTX entry that traps at runtime; used when a plan has no lowering so a
    /// wrong kernel is never silently emitted.
    fn ptx_trap(safe_name: &str, reason: &str) -> String {
        format!(
            "// ERROR: {}; emitting trapping stub.\n.visible .entry {safe_name}(\n    .param .u64 param_out\n) {{\n    trap;\n    ret;\n}}\n",
            clean(reason),
        )
    }

    /// MSL translation that fails compilation loudly when a plan has no lowering.
    fn msl_error(reason: &str) -> String {
        format!("#error \"Oxideon Tile-IR: {}\"\n", clean(reason))
    }

    /// Generates a per-thread elementwise PTX kernel over the plan inputs for
    /// pure Add/Sub/Mul/Div/ReLU/SiLU sequences (seeded stack machine).
    fn ptx_elementwise(name: &str, target_sm: &str, inputs: &[String], ops: &[&TileOp]) -> String {
        let mut params = String::new();
        for (i, _) in inputs.iter().enumerate() {
            params.push_str(&format!("    .param .u64 p_in{i},\n"));
        }
        params.push_str("    .param .u64 p_out,\n    .param .u32 param_n\n");
        let fregs = inputs.len() + 6 * ops.len() + 8;
        let rdregs = inputs.len() + 4;
        let mut body = String::new();
        body.push_str("    mov.u32     %r0, %ctaid.x;\n    mov.u32     %r1, %ntid.x;\n    mov.u32     %r2, %tid.x;\n    mad.lo.s32  %r3, %r0, %r1, %r2;\n");
        body.push_str("    ld.param.u32 %r4, [param_n];\n    setp.ge.s32  %p0, %r3, %r4;\n    @%p0 bra     $DONE;\n");
        for (i, _) in inputs.iter().enumerate() {
            body.push_str(&format!("    ld.param.u64 %rd{i}, [p_in{i}];\n"));
        }
        let rd_out = inputs.len();
        let rd_off = inputs.len() + 1;
        body.push_str(&format!("    ld.param.u64 %rd{rd_out}, [p_out];\n"));
        body.push_str(&format!("    mul.wide.s32 %rd{rd_off}, %r3, 4;\n"));
        let mut next_f = inputs.len();
        for (i, _) in inputs.iter().enumerate() {
            body.push_str(&format!(
                "    add.u64      %rd{}, %rd{i}, %rd{rd_off};\n    ld.global.f32 %f{i}, [%rd{}];\n",
                inputs.len() + 2,
                inputs.len() + 2,
            ));
        }
        let mut stack: Vec<usize> = (0..inputs.len()).collect();
        for op in ops {
            match op {
                TileOp::Add | TileOp::Sub | TileOp::Mul | TileOp::Div => {
                    let rhs = stack.pop().expect("elementwise stack underflow");
                    let lhs = stack.pop().expect("elementwise stack underflow");
                    let dst = next_f;
                    next_f += 1;
                    let inst = match op {
                        TileOp::Add => "add.f32",
                        TileOp::Sub => "sub.f32",
                        TileOp::Mul => "mul.f32",
                        _ => "div.rn.f32",
                    };
                    body.push_str(&format!("    {inst}      %f{dst}, %f{lhs}, %f{rhs};\n"));
                    stack.push(dst);
                }
                TileOp::ReLU => {
                    let src = stack.pop().expect("elementwise stack underflow");
                    let dst = next_f;
                    next_f += 1;
                    body.push_str(&format!(
                        "    setp.gt.f32  %p1, %f{src}, 0.0;\n    selp.f32      %f{dst}, %f{src}, 0.0, %p1;\n"
                    ));
                    stack.push(dst);
                }
                TileOp::SiLU => {
                    let src = stack.pop().expect("elementwise stack underflow");
                    let dst = next_f;
                    next_f += 1;
                    let t0 = next_f;
                    let t1 = next_f + 1;
                    let t2 = next_f + 2;
                    let t3 = next_f + 3;
                    let t4 = next_f + 4;
                    next_f += 5;
                    body.push_str(&format!(
                        "    neg.f32      %f{t0}, %f{src};\n    mul.f32      %f{t1}, %f{t0}, 1.4426950408889634;\n    ex2.approx.f32 %f{t2}, %f{t1};\n    add.f32      %f{t3}, %f{t2}, 1.0;\n    rcp.approx.f32 %f{t4}, %f{t3};\n    mul.f32      %f{dst}, %f{src}, %f{t4};\n"
                    ));
                    stack.push(dst);
                }
                _ => unreachable!("non-elementwise op reached PTX elementwise emitter"),
            }
        }
        let top = *stack.last().expect("empty elementwise stack");
        body.push_str(&format!(
            "    add.u64      %rd{}, %rd{rd_out}, %rd{rd_off};\n    st.global.f32 [%rd{}], %f{top};\n",
            inputs.len() + 3,
            inputs.len() + 3,
        ));
        format!(
            ".version 8.0\n.target {target_sm}\n.address_size 64\n\n.visible .entry {name}(\n{params}) {{\n    .reg .pred  %p<2>;\n    .reg .b32   %r<8>;\n    .reg .b64   %rd<{rdregs}>;\n    .reg .f32   %f<{fregs}>;\n\n{body}\n$DONE:\n    ret;\n}}\n"
        )
    }

    /// Generates a per-thread elementwise MSL kernel over the plan inputs.
    fn msl_elementwise(name: &str, inputs: &[String], output: &str, ops: &[&TileOp]) -> String {
        let mut params = String::new();
        for (i, input) in inputs.iter().enumerate() {
            params.push_str(&format!(
                "    device const float* {input} [[buffer({i})]],\n"
            ));
        }
        params.push_str(&format!(
            "    device float* {output} [[buffer({})]],\n    constant uint& n [[buffer({})]],\n    uint tid [[thread_position_in_grid]]\n",
            inputs.len(),
            inputs.len() + 1,
        ));
        let mut body = String::from("    if (tid >= n) return;\n");
        for (i, input) in inputs.iter().enumerate() {
            body.push_str(&format!("    float s{i} = {input}[tid];\n"));
        }
        let mut stack: Vec<String> = (0..inputs.len()).map(|i| format!("s{i}")).collect();
        let mut tmp = 0usize;
        for op in ops {
            match op {
                TileOp::Add | TileOp::Sub | TileOp::Mul | TileOp::Div => {
                    let rhs = stack.pop().expect("elementwise stack underflow");
                    let lhs = stack.pop().expect("elementwise stack underflow");
                    let sym = match op {
                        TileOp::Add => "+",
                        TileOp::Sub => "-",
                        TileOp::Mul => "*",
                        _ => "/",
                    };
                    body.push_str(&format!("    float t{tmp} = {lhs} {sym} {rhs};\n"));
                    stack.push(format!("t{tmp}"));
                    tmp += 1;
                }
                TileOp::ReLU => {
                    let src = stack.pop().expect("elementwise stack underflow");
                    body.push_str(&format!("    float t{tmp} = fmax({src}, 0.0f);\n"));
                    stack.push(format!("t{tmp}"));
                    tmp += 1;
                }
                TileOp::SiLU => {
                    let src = stack.pop().expect("elementwise stack underflow");
                    body.push_str(&format!(
                        "    float t{tmp} = {src} / (1.0f + exp(-{src}));\n"
                    ));
                    stack.push(format!("t{tmp}"));
                    tmp += 1;
                }
                TileOp::GELU => {
                    let src = stack.pop().expect("elementwise stack underflow");
                    body.push_str(&format!(
                        "    float t{tmp} = 0.5f * {src} * (1.0f + tanh(0.7978845608028654f * ({src} + 0.044715f * {src} * {src} * {src})));\n"
                    ));
                    stack.push(format!("t{tmp}"));
                    tmp += 1;
                }
                _ => unreachable!("non-elementwise op reached MSL elementwise emitter"),
            }
        }
        let top = stack.last().expect("empty elementwise stack");
        body.push_str(&format!("    {output}[tid] = {top};\n"));
        format!(
            "#include <metal_stdlib>\nusing namespace metal;\n\nkernel void {name}(\n{params}) {{\n{body}}}\n"
        )
    }
}

/// Automated Kernel Fusion Engine.
/// Detects fusible subgraphs such as `Linear + Activation + ResidualAdd + RMSNorm`,
/// `FlashAttention`, `RoPE`, and `QuantizedGemvQ4`.
#[derive(Debug, Clone, Default)]
pub struct KernelFusionEngine;

impl KernelFusionEngine {
    pub fn new() -> Self {
        Self
    }

    /// Generates a fused GEMM + SwiGLU kernel plan:
    /// $\operatorname{SiLU}(x W_{gate}) \odot (x W_{up})$.
    pub fn plan_fused_linear_swiglu(name: &str, m: usize, n: usize, k: usize) -> TileKernelPlan {
        let mut plan = TileKernelPlan::new(name, 16, 16, 16);
        plan.add_input("in_a")
            .add_input("weight_gate")
            .add_input("weight_up")
            .set_output("fused_out")
            .push_op(TileOp::Gemm { m, n, k })
            .push_op(TileOp::Gemm { m, n, k })
            .push_op(TileOp::SiLU)
            .push_op(TileOp::Mul);
        plan
    }

    /// Generates a fused Add + RMSNorm + SiLU kernel plan.
    pub fn plan_fused_add_rmsnorm(name: &str, dim: usize, eps: f64) -> TileKernelPlan {
        let mut plan = TileKernelPlan::new(name, 1, dim, 1);
        plan.add_input("x")
            .add_input("residual")
            .add_input("weight")
            .set_output("norm_out")
            .push_op(TileOp::Add)
            .push_op(TileOp::RMSNorm { eps })
            .push_op(TileOp::SiLU);
        plan
    }

    /// Generates a fused FlashAttention kernel plan lowered across PTX, Metal MSL, and WGSL.
    pub fn plan_flash_attention(name: &str, head_dim: usize, is_causal: bool) -> TileKernelPlan {
        let mut plan = TileKernelPlan::new(name, 16, 16, head_dim.max(2));
        plan.add_input("Q")
            .add_input("K")
            .add_input("V")
            .set_output("O")
            .push_op(TileOp::FlashAttention {
                head_dim: head_dim.max(2),
                is_causal,
            });
        plan
    }

    /// Generates a fused 4-bit quantized affine GEMV (`Q4 GEMV`) kernel plan.
    pub fn plan_quantized_q4_gemv(name: &str, group_size: usize) -> TileKernelPlan {
        let mut plan = TileKernelPlan::new(name, 1, group_size.max(8), group_size.max(8));
        plan.add_input("x")
            .add_input("w_q4")
            .add_input("scales")
            .add_input("biases")
            .set_output("out")
            .push_op(TileOp::QuantizedGemvQ4 {
                group_size: group_size.max(8),
            });
        plan
    }

    /// Generates a fused Rotary Position Embedding (RoPE) kernel plan.
    pub fn plan_fused_rope(name: &str, head_dim: usize) -> TileKernelPlan {
        let mut plan = TileKernelPlan::new(name, 1, head_dim.max(2), 1);
        plan.add_input("x")
            .set_output("out")
            .push_op(TileOp::RoPE {
                head_dim: head_dim.max(2),
            });
        plan
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// SwiGLU needs two projections (gate + up); the plan used to emit one
    /// GEMM while listing both weights as inputs.
    #[test]
    fn swiglu_plan_has_two_gemms() {
        let plan = KernelFusionEngine::plan_fused_linear_swiglu("s", 4, 4, 4);
        let gemms = plan
            .ops
            .iter()
            .filter(|op| matches!(op, TileOp::Gemm { .. }))
            .count();
        assert_eq!(gemms, 2);
        assert!(plan.inputs.iter().any(|i| i == "weight_gate"));
        assert!(plan.inputs.iter().any(|i| i == "weight_up"));
    }
}
