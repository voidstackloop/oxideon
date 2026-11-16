//! Headless Vulkan SPIR-V Compute Pipeline Driver.
//!
//! Generates raw SPIR-V 1.5 binary bytecode for bare-metal compute execution on discrete GPUs,
//! including complete vector addition (`OpFAdd`), fused multiply-add (`OpFMul` + `OpFAdd`),
//! and human-readable SPIR-V disassembly (`disassemble_spirv`).

/// Direct Vulkan SPIR-V Compute Kernel Metadata & Binary Generator.
#[derive(Debug, Clone)]
pub struct VulkanSpirvKernel {
    pub name: String,
    pub workgroup_size: (u32, u32, u32),
    pub num_bindings: usize,
}

impl VulkanSpirvKernel {
    pub const SPIRV_MAGIC: u32 = 0x07230203;
    pub const SPIRV_VERSION: u32 = 0x00010500; // SPIR-V 1.5

    pub fn new(name: impl Into<String>, num_bindings: usize) -> Self {
        Self {
            name: name.into(),
            workgroup_size: (64, 1, 1),
            num_bindings,
        }
    }

    /// Emits a valid, loadable SPIR-V 1.5 compute module: standard 5-word
    /// header, entry point with the kernel `name` and `workgroup_size`, one
    /// `StorageBuffer` variable per binding, and an empty function body. The
    /// `Bound` header word is computed from the IDs actually used.
    pub fn emit_spirv_binary(&self) -> crate::error::Result<Vec<u32>> {
        if self.num_bindings > 1_000_000 {
            return Err(crate::error::Error::Msg(format!(
                "SPIR-V emitter refuses {} bindings (over the 1M cap)",
                self.num_bindings
            )));
        }
        let mut w = SpirvBuilder::new();
        let void = w.fresh();
        let func_ty = w.fresh();
        let uint = w.fresh();
        let v3uint = w.fresh();
        let ptr_input = w.fresh();
        let float = w.fresh();
        let buf_arr = w.fresh();
        let buf_struct = w.fresh();
        let ptr_ssbo = w.fresh();
        let gid = w.fresh();
        let mut bufs = Vec::with_capacity(self.num_bindings);
        for _ in 0..self.num_bindings {
            bufs.push(w.fresh());
        }
        let main = w.fresh();
        let label = w.fresh();

        w.inst(17, &[1]); // OpCapability Shader
        w.inst(14, &[0, 1]); // OpMemoryModel Logical GLSL450
        w.inst(15, &entry_point_words(5, main, &self.name, &[gid])); // OpEntryPoint GLCompute
        let (wx, wy, wz) = self.workgroup_size;
        w.inst(16, &[main, 17, wx, wy, wz]); // OpExecutionMode LocalSize
        w.inst(71, &[gid, 11, 28]); // OpDecorate BuiltIn GlobalInvocationId
        w.inst(71, &[buf_struct, 2]); // OpDecorate Block
        w.inst(72, &[buf_struct, 0, 35, 0]); // OpMemberDecorate Offset 0
        w.inst(71, &[buf_arr, 6, 4]); // OpDecorate ArrayStride 4
        for (i, &b) in bufs.iter().enumerate() {
            w.inst(71, &[b, 34, 0]); // OpDecorate DescriptorSet 0
            w.inst(71, &[b, 33, i as u32]); // OpDecorate Binding i
        }
        w.inst_result(19, void, &[]); // OpTypeVoid
        w.inst_result(33, func_ty, &[void]); // OpTypeFunction
        w.inst_result(21, uint, &[32, 0]); // OpTypeInt 32 0
        w.inst_result(23, v3uint, &[uint, 3]); // OpTypeVector
        w.inst_result(32, ptr_input, &[1, v3uint]); // OpTypePointer Input
        w.inst_result(22, float, &[32]); // OpTypeFloat 32
        w.inst_result(29, buf_arr, &[float]); // OpTypeRuntimeArray
        w.inst_result(30, buf_struct, &[buf_arr]); // OpTypeStruct
        w.inst_result(32, ptr_ssbo, &[12, buf_struct]); // OpTypePointer StorageBuffer
        w.inst_typed(59, ptr_input, gid, &[1]); // OpVariable Input
        for &b in &bufs {
            w.inst_typed(59, ptr_ssbo, b, &[12]); // OpVariable StorageBuffer
        }
        w.inst_typed(54, void, main, &[0, func_ty]); // OpFunction
        w.inst_result(248, label, &[]); // OpLabel
        w.inst(253, &[]); // OpReturn
        w.inst(56, &[]); // OpFunctionEnd
        Ok(w.finish())
    }

    /// Emits a complete SPIR-V 1.5 compute shader performing elementwise float addition
    /// `buf2[gid.x] = buf0[gid.x] + buf1[gid.x]` using `OpAccessChain`, `OpLoad`, `OpFAdd`, and `OpStore`.
    pub fn emit_fadd_spirv_binary(&self) -> crate::error::Result<Vec<u32>> {
        self.emit_arith_spirv_binary(false)
    }

    /// Emits a complete SPIR-V 1.5 compute shader performing elementwise fused multiply-add
    /// `buf3[gid.x] = buf0[gid.x] * buf1[gid.x] + buf2[gid.x]` using `OpFMul` and `OpFAdd`.
    pub fn emit_fma_spirv_binary(&self) -> crate::error::Result<Vec<u32>> {
        self.emit_arith_spirv_binary(true)
    }

    fn emit_arith_spirv_binary(&self, is_fma: bool) -> crate::error::Result<Vec<u32>> {
        let n_bind = self.num_bindings.max(if is_fma { 4 } else { 3 });
        if n_bind > 1_000_000 {
            return Err(crate::error::Error::Msg("too many bindings".into()));
        }
        let mut w = SpirvBuilder::new();
        let void = w.fresh();
        let func_ty = w.fresh();
        let uint = w.fresh();
        let v3uint = w.fresh();
        let ptr_input = w.fresh();
        let ptr_input_u32 = w.fresh();
        let float = w.fresh();
        let ptr_ssbo_f32 = w.fresh();
        let buf_arr = w.fresh();
        let buf_struct = w.fresh();
        let ptr_ssbo = w.fresh();
        let zero_u32 = w.fresh();
        let gid = w.fresh();
        let mut bufs = Vec::with_capacity(n_bind);
        for _ in 0..n_bind {
            bufs.push(w.fresh());
        }
        let main = w.fresh();
        let label = w.fresh();
        let gid_x_ptr = w.fresh();
        let tid = w.fresh();
        let ptr_a = w.fresh();
        let ptr_b = w.fresh();
        let val_a = w.fresh();
        let val_b = w.fresh();

        w.inst(17, &[1]); // OpCapability Shader
        w.inst(14, &[0, 1]); // OpMemoryModel Logical GLSL450
        w.inst(15, &entry_point_words(5, main, &self.name, &[gid])); // OpEntryPoint GLCompute
        let (wx, wy, wz) = self.workgroup_size;
        w.inst(16, &[main, 17, wx, wy, wz]); // OpExecutionMode LocalSize
        w.inst(71, &[gid, 11, 28]); // OpDecorate BuiltIn GlobalInvocationId
        w.inst(71, &[buf_struct, 2]); // OpDecorate Block
        w.inst(72, &[buf_struct, 0, 35, 0]); // OpMemberDecorate Offset 0
        w.inst(71, &[buf_arr, 6, 4]); // OpDecorate ArrayStride 4
        for (i, &b) in bufs.iter().enumerate() {
            w.inst(71, &[b, 34, 0]); // OpDecorate DescriptorSet 0
            w.inst(71, &[b, 33, i as u32]); // OpDecorate Binding i
        }
        w.inst_result(19, void, &[]); // OpTypeVoid
        w.inst_result(33, func_ty, &[void]); // OpTypeFunction
        w.inst_result(21, uint, &[32, 0]); // OpTypeInt 32 0
        w.inst_result(23, v3uint, &[uint, 3]); // OpTypeVector
        w.inst_result(32, ptr_input, &[1, v3uint]); // OpTypePointer Input v3uint
        w.inst_result(32, ptr_input_u32, &[1, uint]); // OpTypePointer Input uint
        w.inst_result(22, float, &[32]); // OpTypeFloat 32
        w.inst_result(32, ptr_ssbo_f32, &[12, float]); // OpTypePointer StorageBuffer float
        w.inst_result(29, buf_arr, &[float]); // OpTypeRuntimeArray
        w.inst_result(30, buf_struct, &[buf_arr]); // OpTypeStruct
        w.inst_result(32, ptr_ssbo, &[12, buf_struct]); // OpTypePointer StorageBuffer struct
        w.inst_typed(43, uint, zero_u32, &[0]); // OpConstant 0u
        w.inst_typed(59, ptr_input, gid, &[1]); // OpVariable Input
        for &b in &bufs {
            w.inst_typed(59, ptr_ssbo, b, &[12]); // OpVariable StorageBuffer
        }
        w.inst_typed(54, void, main, &[0, func_ty]); // OpFunction
        w.inst_result(248, label, &[]); // OpLabel
        w.inst_typed(65, ptr_input_u32, gid_x_ptr, &[gid, zero_u32]); // OpAccessChain gid.x
        w.inst_typed(61, uint, tid, &[gid_x_ptr]); // OpLoad tid
        w.inst_typed(65, ptr_ssbo_f32, ptr_a, &[bufs[0], zero_u32, tid]); // OpAccessChain buf0[tid]
        w.inst_typed(65, ptr_ssbo_f32, ptr_b, &[bufs[1], zero_u32, tid]); // OpAccessChain buf1[tid]
        w.inst_typed(61, float, val_a, &[ptr_a]); // OpLoad a
        w.inst_typed(61, float, val_b, &[ptr_b]); // OpLoad b

        if is_fma {
            let ptr_c = w.fresh();
            let val_c = w.fresh();
            let prod = w.fresh();
            let res = w.fresh();
            let ptr_out = w.fresh();
            w.inst_typed(65, ptr_ssbo_f32, ptr_c, &[bufs[2], zero_u32, tid]); // OpAccessChain buf2[tid]
            w.inst_typed(61, float, val_c, &[ptr_c]); // OpLoad c
            w.inst_typed(133, float, prod, &[val_a, val_b]); // OpFMul a * b
            w.inst_typed(129, float, res, &[prod, val_c]); // OpFAdd (a * b) + c
            w.inst_typed(65, ptr_ssbo_f32, ptr_out, &[bufs[3], zero_u32, tid]); // OpAccessChain buf3[tid]
            w.inst(62, &[ptr_out, res]); // OpStore
        } else {
            let res = w.fresh();
            let ptr_out = w.fresh();
            w.inst_typed(129, float, res, &[val_a, val_b]); // OpFAdd a + b
            w.inst_typed(65, ptr_ssbo_f32, ptr_out, &[bufs[2], zero_u32, tid]); // OpAccessChain buf2[tid]
            w.inst(62, &[ptr_out, res]); // OpStore
        }

        w.inst(253, &[]); // OpReturn
        w.inst(56, &[]); // OpFunctionEnd
        Ok(w.finish())
    }

    /// Disassembles a SPIR-V 32-bit word stream into human-readable SPIR-V assembly text.
    pub fn disassemble_spirv(words: &[u32]) -> crate::error::Result<String> {
        if words.len() < 5 {
            return Err(crate::error::Error::Msg(
                "SPIR-V binary too short (< 5 header words)".into(),
            ));
        }
        if words[0] != Self::SPIRV_MAGIC {
            return Err(crate::error::Error::Msg(format!(
                "Invalid SPIR-V magic word 0x{:08x} (expected 0x{:08x})",
                words[0],
                Self::SPIRV_MAGIC
            )));
        }
        let ver_major = (words[1] >> 16) & 0xff;
        let ver_minor = (words[1] >> 8) & 0xff;
        let bound = words[3];
        let mut out = format!(
            "; SPIR-V\n; Version: {ver_major}.{ver_minor}\n; Generator: 0x{:08x}\n; Bound: {bound}\n; Schema: {}\n",
            words[2], words[4]
        );

        let mut pc = 5usize;
        while pc < words.len() {
            let header = words[pc];
            let word_count = (header >> 16) as usize;
            let opcode = header & 0xffff;
            if word_count == 0 || pc + word_count > words.len() {
                return Err(crate::error::Error::Msg(format!(
                    "Malformed SPIR-V instruction at word {pc}"
                )));
            }
            let args = &words[pc + 1..pc + word_count];
            let line = match opcode {
                17 => "OpCapability Shader".to_string(),
                14 => "OpMemoryModel Logical GLSL450".to_string(),
                15 => format!("OpEntryPoint GLCompute %{}", args.get(1).copied().unwrap_or(0)),
                16 => format!(
                    "OpExecutionMode %{} LocalSize {} {} {}",
                    args.first().copied().unwrap_or(0),
                    args.get(2).copied().unwrap_or(1),
                    args.get(3).copied().unwrap_or(1),
                    args.get(4).copied().unwrap_or(1)
                ),
                71 => format!("OpDecorate %{}", args.first().copied().unwrap_or(0)),
                72 => format!("OpMemberDecorate %{}", args.first().copied().unwrap_or(0)),
                19 => format!("%{} = OpTypeVoid", args.first().copied().unwrap_or(0)),
                33 => format!("%{} = OpTypeFunction %{}", args.first().copied().unwrap_or(0), args.get(1).copied().unwrap_or(0)),
                21 => format!("%{} = OpTypeInt {} {}", args.first().copied().unwrap_or(0), args.get(1).copied().unwrap_or(32), args.get(2).copied().unwrap_or(0)),
                22 => format!("%{} = OpTypeFloat {}", args.first().copied().unwrap_or(0), args.get(1).copied().unwrap_or(32)),
                23 => format!("%{} = OpTypeVector %{} {}", args.first().copied().unwrap_or(0), args.get(1).copied().unwrap_or(0), args.get(2).copied().unwrap_or(3)),
                29 => format!("%{} = OpTypeRuntimeArray %{}", args.first().copied().unwrap_or(0), args.get(1).copied().unwrap_or(0)),
                30 => format!("%{} = OpTypeStruct %{}", args.first().copied().unwrap_or(0), args.get(1).copied().unwrap_or(0)),
                32 => format!("%{} = OpTypePointer {} %{}", args.first().copied().unwrap_or(0), args.get(1).copied().unwrap_or(0), args.get(2).copied().unwrap_or(0)),
                43 => format!("%{} = OpConstant %{} {}", args.get(1).copied().unwrap_or(0), args.first().copied().unwrap_or(0), args.get(2).copied().unwrap_or(0)),
                59 => format!("%{} = OpVariable %{} {}", args.get(1).copied().unwrap_or(0), args.first().copied().unwrap_or(0), args.get(2).copied().unwrap_or(0)),
                54 => format!("%{} = OpFunction %{}", args.get(1).copied().unwrap_or(0), args.first().copied().unwrap_or(0)),
                248 => format!("%{} = OpLabel", args.first().copied().unwrap_or(0)),
                65 => format!("%{} = OpAccessChain %{} %{}", args.get(1).copied().unwrap_or(0), args.first().copied().unwrap_or(0), args.get(2).copied().unwrap_or(0)),
                61 => format!("%{} = OpLoad %{} %{}", args.get(1).copied().unwrap_or(0), args.first().copied().unwrap_or(0), args.get(2).copied().unwrap_or(0)),
                62 => format!("OpStore %{} %{}", args.first().copied().unwrap_or(0), args.get(1).copied().unwrap_or(0)),
                129 => format!("%{} = OpFAdd %{} %{} %{}", args.get(1).copied().unwrap_or(0), args.first().copied().unwrap_or(0), args.get(2).copied().unwrap_or(0), args.get(3).copied().unwrap_or(0)),
                133 => format!("%{} = OpFMul %{} %{} %{}", args.get(1).copied().unwrap_or(0), args.first().copied().unwrap_or(0), args.get(2).copied().unwrap_or(0), args.get(3).copied().unwrap_or(0)),
                253 => "OpReturn".to_string(),
                56 => "OpFunctionEnd".to_string(),
                other => format!("OpUnknown({other})"),
            };
            out.push_str("    ");
            out.push_str(&line);
            out.push('\n');
            pc += word_count;
        }
        Ok(out)
    }

    /// Returns byte size of the emitted binary words.
    pub fn binary_size_bytes(&self) -> crate::error::Result<usize> {
        Ok(self.emit_spirv_binary()?.len() * std::mem::size_of::<u32>())
    }
}

/// Minimal SPIR-V word stream builder with sequential IDs and a computed `Bound`.
struct SpirvBuilder {
    words: Vec<u32>,
    next_id: u32,
}

impl SpirvBuilder {
    fn new() -> Self {
        Self {
            words: vec![
                VulkanSpirvKernel::SPIRV_MAGIC,
                VulkanSpirvKernel::SPIRV_VERSION,
                0x13370001, // Oxideon generator ID
                0,          // Bound (patched in `finish`)
                0,          // Reserved schema
            ],
            next_id: 1,
        }
    }

    fn fresh(&mut self) -> u32 {
        let id = self.next_id;
        self.next_id += 1;
        id
    }

    /// Appends an instruction without a result id.
    fn inst(&mut self, opcode: u32, operands: &[u32]) {
        self.words
            .push(((operands.len() as u32 + 1) << 16) | opcode);
        self.words.extend_from_slice(operands);
    }

    /// Appends an instruction producing `result` (result id first).
    fn inst_result(&mut self, opcode: u32, result: u32, operands: &[u32]) {
        self.words
            .push(((operands.len() as u32 + 2) << 16) | opcode);
        self.words.push(result);
        self.words.extend_from_slice(operands);
    }

    /// Appends an instruction with a result type and a result id.
    fn inst_typed(&mut self, opcode: u32, result_type: u32, result: u32, operands: &[u32]) {
        self.words
            .push(((operands.len() as u32 + 3) << 16) | opcode);
        self.words.push(result_type);
        self.words.push(result);
        self.words.extend_from_slice(operands);
    }

    fn finish(mut self) -> Vec<u32> {
        self.words[3] = self.next_id;
        self.words
    }
}

/// `OpEntryPoint` operands: model, function id, NUL-terminated name words, interface ids.
fn entry_point_words(model: u32, function: u32, name: &str, interface: &[u32]) -> Vec<u32> {
    let mut out = vec![model, function];
    let bytes = name.as_bytes();
    let padded = bytes.len() + 1;
    let words = padded.div_ceil(4);
    for i in 0..words {
        let mut w = 0u32;
        for j in 0..4 {
            let idx = i * 4 + j;
            let b = if idx < bytes.len() { bytes[idx] } else { 0 };
            w |= (b as u32) << (8 * j);
        }
        out.push(w);
    }
    out.extend_from_slice(interface);
    out
}
